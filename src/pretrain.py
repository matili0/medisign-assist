"""SPOTER pretraining: fixed signer split, macro metrics, AMP, full resume."""

import argparse
import json
import math
import random
import time
from pathlib import Path

if __name__ == "__main__":
    print("Starting pretraining: loading libraries...", flush=True)

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from artifacts import atomic_save, restore_rng, rng_state, safe_output, write_json
from data import (
    PoseDataset,
    available_rows,
    coverage,
    disk_cache,
    load_split,
    pretrain_candidates,
)
from evaluate import metrics_from_confusion
from model import SPOTER

ROOT = Path(__file__).resolve().parents[1]


def precision_settings(mode, device):
    if mode == "auto":
        mode = "fp16" if device.type == "cuda" else "fp32"
    if mode != "fp32" and device.type != "cuda":
        raise ValueError("Mixed precision requires CUDA in this script.")
    if mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is not available on this GPU.")
    return mode, torch.bfloat16 if mode == "bf16" else torch.float16


def class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = 1 / np.sqrt(counts + 1e-6)
    return weights / weights.sum() * n_classes


def create_model(cfg, n_classes, device):
    return SPOTER(
        num_classes=n_classes,
        feature_size=225,
        dropout=cfg["pretrain"]["dropout"],
        **cfg["model"],
    ).to(device)


def make_loader(dataset, batch, workers, device, generator, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def train_epoch(model, loader, criterion, optimizer, scaler, device, mode, dtype):
    model.train()
    loss_sum = correct = total = 0
    skipped_steps = batches = 0
    start = time.perf_counter()
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=dtype, enabled=mode != "fp32"
        ):
            out = model(x)
            loss = criterion(out, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss. Try --precision bf16 or fp32.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if mode != "fp16" and not torch.isfinite(grad_norm):
            raise FloatingPointError(
                "Non-finite gradients. Stopping before replacing the previous checkpoint."
            )
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped_steps += int(scaler.get_scale() < previous_scale)
        batches += 1
        loss_sum += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    if skipped_steps == batches:
        raise FloatingPointError("AMP skipped every update. Use bf16 or fp32.")
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "skipped_amp_steps": skipped_steps,
        "seconds": time.perf_counter() - start,
        "samples": total,
    }


@torch.no_grad()
def evaluate(model, loader, device, mode, dtype, train_counts):
    model.eval()
    n = len(train_counts)
    cm = np.zeros((n, n), dtype=np.int64)
    loss_sum = total = 0
    for x, y in tqdm(loader, desc="validation", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=dtype, enabled=mode != "fp32"
        ):
            out = model(x)
        # Same unweighted validation loss for both training-loss variants.
        loss = nn.functional.cross_entropy(out.float(), y, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite validation loss. Check the precision setting."
            )
        loss_sum += loss.item()
        total += len(y)
        codes = (y * n + out.argmax(1)).cpu().numpy()
        cm += np.bincount(codes, minlength=n * n).reshape(n, n)
    result = metrics_from_confusion(cm, train_counts)
    result["loss"] = loss_sum / total
    return result, cm


def export_best(state, cfg, df, classes, outdir, score, epoch, settings):
    """Save the best weights with their classes and pretraining signer lists."""
    path = outdir / "pretrained.pt"
    atomic_save(state, path)
    write_json(
        {
            "n_classes_pretrain": len(classes),
            "feature_size": 225,
            "hidden_dim": cfg["model"]["hidden_dim"],
            "max_len": cfg["model"]["max_len"],
            "training_signers": sorted(
                df[df.split == "train"].signer.unique().tolist()
            ),
            "validation_signers": sorted(
                df[df.split == "val"].signer.unique().tolist()
            ),
            "classes": classes,
            "config": cfg,
            "settings": settings,
            "best_macro_f1": score,
            "best_epoch": epoch,
        },
        outdir / "pretrained_meta.json",
    )


def main():
    # 1. Read command options and recover settings when resuming.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", type=Path, help="last.pt checkpoint created by this script"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--loss-weighting", choices=["none", "sqrt"])
    parser.add_argument("--cache", choices=["files", "mmap"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--epochs",
        type=int,
        help="Total number of epochs for the learning-rate schedule",
    )
    parser.add_argument(
        "--run-epochs", type=int, help="Maximum epochs to run in this session"
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    settings_config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = None
    if args.resume:
        # Only load checkpoints produced locally by this script.
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("format") != "medisign-pretrain-v3":
            raise ValueError("This file is not a compatible full resume checkpoint.")
        settings_config = saved["config"]
    defaults = {
        "precision": settings_config["pretrain"].get("precision", "auto"),
        "loss_weighting": settings_config["pretrain"].get("loss_weighting", "none"),
        "cache": settings_config["pretrain"].get("cache", "files"),
        "batch_size": settings_config["pretrain"]["batch_size"],
        "workers": settings_config["pretrain"].get("workers", 2),
        "epochs": settings_config["pretrain"]["epochs"],
        "seed": 42,
    }
    settings = dict(saved["settings"] if saved else defaults)
    for key in defaults:
        value = getattr(args, key)
        if value is not None:
            if saved and value != settings[key]:
                raise ValueError(
                    f"Cannot change {key} on resume. Start a separate experiment."
                )
            settings[key] = value
    if (
        settings["batch_size"] < 1
        or settings["epochs"] < 1
        or settings["workers"] < 0
        or (args.run_epochs is not None and args.run_epochs < 1)
    ):
        parser.error("Sizes and durations must be positive; workers must be >= 0.")

    # 2. Set numeric precision and seed the random generators.
    mode, dtype = precision_settings(settings["precision"], device)
    if saved and saved.get("resolved_precision") != mode:
        raise ValueError("Resume precision does not match the checkpoint.")
    torch.backends.mha.set_fastpath_enabled(False)
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])

    # 3. Select non-medical signs from training and validation signers only.
    data = (ROOT / settings_config["dataset_path"]).resolve()
    print("Loading the signer split...", flush=True)
    corpus, info = load_split()
    examples = available_rows(
        pretrain_candidates(corpus, info), data / "pretrain_landmarks"
    )
    train_classes = sorted(examples[examples.split == "train"].sign.unique().tolist())
    excluded = sorted(set(examples.sign) - set(train_classes))
    if excluded:
        print("Excluding classes without training examples:", excluded)
    examples = examples[examples.sign.isin(train_classes)].copy().reset_index(drop=True)
    if any(examples[examples.split == part].empty for part in ("train", "val")):
        raise ValueError("Training or validation set is empty.")
    labels = {name: i for i, name in enumerate(train_classes)}
    examples["label"] = examples.sign.map(labels)
    examples["path"] = examples.id.map(
        lambda x: str(data / "pretrain_landmarks" / f"{x}.npy")
    )
    train_rows, val_rows = (
        examples[examples.split == "train"],
        examples[examples.split == "val"],
    )
    # Save the actual rows: resume checks stay easy to inspect.
    data_rows = examples[["id", "sign", "signer", "split"]].values.tolist()
    print(
        f"{device} | train={len(train_rows)}, val={len(val_rows)}, classes={len(labels)}, precision={mode}"
    )
    print(
        f"Validation has no examples for {(coverage(examples).val == 0).sum()} classes."
    )
    if saved and saved["data_rows"] != data_rows:
        raise ValueError("Examples or data split differ from the checkpoint.")

    # 4. Save experiment settings and prepare datasets and mini-batches.
    default_out = ROOT / "runs" / "pretrain"
    output_dir = safe_output(
        args.output_dir or (args.resume.parent if args.resume else default_out)
    )
    if not saved and any(
        (output_dir / name).exists() for name in ["last.pt", "pretrained.pt"]
    ):
        raise FileExistsError(
            "This folder already contains weights. Use --resume or another --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    examples[["id", "sign", "signer", "split"]].to_csv(
        output_dir / "pretrain_instances_split.csv", index=False
    )
    coverage(examples).to_csv(output_dir / "pretrain_class_counts.csv")
    write_json(
        {
            "settings": settings,
            "config": settings_config,
            "split": info,
            "macro_scope": "classes present in validation",
            "rare_scope": "rare: train count <= 25th percentile; frequent: >= 75th percentile",
        },
        output_dir / "protocol.json",
    )
    generators = {
        key: torch.Generator().manual_seed(settings["seed"] + offset)
        for offset, key in enumerate(["sampler", "train_loader", "val_loader"])
    }
    datasets = []
    for name, rows in [("train", train_rows), ("val", val_rows)]:
        cache = None
        if settings["cache"] == "mmap":
            cache = disk_cache(
                rows.path.tolist(),
                settings_config["model"]["max_len"],
                ROOT / "cache" / f"{name}.npy",
            )
        datasets.append(
            PoseDataset(
                rows.path, rows.label, settings_config["model"]["max_len"], cache
            )
        )
    counts = np.bincount(train_rows.label, minlength=len(labels))
    sampling_weights = class_weights(train_rows.label.to_numpy(), len(labels))
    sampler = WeightedRandomSampler(
        torch.tensor(sampling_weights[train_rows.label], dtype=torch.double),
        len(train_rows),
        replacement=True,
        generator=generators["sampler"],
    )
    train_loader = make_loader(
        datasets[0],
        settings["batch_size"],
        settings["workers"],
        device,
        generators["train_loader"],
        sampler,
    )
    val_loader = make_loader(
        datasets[1],
        settings["batch_size"],
        settings["workers"],
        device,
        generators["val_loader"],
    )

    # 5. Build the model, optimizer and warmup/cosine learning-rate schedule.
    model = create_model(settings_config, len(labels), device)
    weights = (
        torch.tensor(sampling_weights, device=device)
        if settings["loss_weighting"] == "sqrt"
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings_config["pretrain"]["learning_rate"],
        weight_decay=1e-4,
    )

    def lr_schedule(epoch):
        return (
            (epoch + 1) / 5
            if epoch < 5
            else 0.5
            * (1 + math.cos(math.pi * (epoch - 5) / max(1, settings["epochs"] - 5)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=mode == "fp16")
    start, best, best_epoch, patience = 1, -1.0, 0, 0
    history = []
    best_state = None

    # Restore all training state before the next epoch.
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start, best, best_epoch = (
            saved["epoch"] + 1,
            saved["best_score"],
            saved["best_epoch"],
        )
        patience, history, best_state = (
            saved["patience"],
            saved["history"],
            saved["best_model"],
        )
        restore_rng(saved["rng"], generators)
        export_best(
            best_state,
            settings_config,
            examples,
            train_classes,
            output_dir,
            best,
            best_epoch,
            settings,
        )
    if patience >= settings_config["pretrain"]["patience"]:
        print("This checkpoint already reached early stopping. No automatic restart.")
        return
    end = (
        min(settings["epochs"], start + args.run_epochs - 1)
        if args.run_epochs
        else settings["epochs"]
    )

    # 6. Train, evaluate and save enough state to resume after each epoch.
    for epoch in range(start, end + 1):
        training = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            mode,
            dtype,
        )
        validation, cm = evaluate(
            model, val_loader, device, mode, dtype, counts
        )
        class_dir = output_dir / "class_metrics"
        class_dir.mkdir(exist_ok=True)
        pd.DataFrame(validation["per_class"], index=train_classes).to_csv(
            class_dir / f"epoch_{epoch:03d}.csv"
        )
        scheduler.step()
        score = validation["macro_f1"]
        history.append(
            {
                "epoch": epoch,
                "train": training,
                "val": {k: v for k, v in validation.items() if k != "per_class"},
            }
        )
        print(
            f"Epoch {epoch}: train_loss={training['loss']:.4f} "
            f"val_acc={validation['accuracy']:.3f} F1_macro={score:.3f} "
            f"macro_recall={validation['macro_recall']:.3f} "
            f"train={training['seconds']:.1f}s"
        )
        if score > best + 1e-6:
            best, best_epoch, patience = score, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            export_best(
                best_state,
                settings_config,
                examples,
                train_classes,
                output_dir,
                best,
                best_epoch,
                settings,
            )
            pd.DataFrame(validation["per_class"], index=train_classes).to_csv(
                output_dir / "best_class_metrics.csv"
            )
            pd.DataFrame(cm, index=train_classes, columns=train_classes).to_csv(
                output_dir / "best_confusion.csv"
            )
        else:
            patience += 1

        # Save the latest state even when this epoch did not improve the score.
        atomic_save(
            {
                "format": "medisign-pretrain-v3",
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng_state(generators),
                "best_model": best_state,
                "best_score": best,
                "best_epoch": best_epoch,
                "patience": patience,
                "history": history,
                "settings": settings,
                "config": settings_config,
                "data_rows": data_rows,
                "resolved_precision": mode,
            },
            output_dir / "last.pt",
        )
        write_json(history, output_dir / "history.json")
        if patience >= settings_config["pretrain"]["patience"]:
            print("Early stopping based on validation macro F1.")
            break
    print(f"Best weights: {output_dir / 'pretrained.pt'}")
    print(f'Resume: python src/pretrain.py --resume "{output_dir / "last.pt"}"')


if __name__ == "__main__":
    main()
