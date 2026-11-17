"""Train classical models or fine-tune SPOTER; select on validation only."""

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from artifacts import (
    ROOT,
    atomic_save,
    config,
    start_run,
    verify_pretrained,
    write_json,
)
from data import load_part, medical_rows
from evaluate import save_predictions, scores
from model import SPOTER
from preprocessing import PREPROCESSING, augment_class, padded


def weights(y, n):
    """Give rare classes more weight using inverse square-root frequencies."""
    w = 1 / np.sqrt(np.bincount(y, minlength=n) + 1e-6)
    return (w / w.mean()).astype(np.float32)


def predict_batches(model, loader, device):
    """Predict labels without changing weights or tracking gradients."""
    model.eval()
    predictions = []
    with torch.inference_mode():
        for x, _ in loader:
            predictions.extend(model(x.to(device)).argmax(1).cpu().tolist())
    return np.asarray(predictions)


def train_classical(args):
    """Fit each candidate on training data and select its best validation score."""
    rows, classes, split, data = medical_rows()

    # 1. Build the same feature representation for training and validation.
    train_features, train_labels, _ = load_part(
        rows, "train", data, classes, "features"
    )
    val_features, val_labels, val_rows = load_part(
        rows, "val", data, classes, "features"
    )
    # 2. Compare a small, explicit grid of classical models.
    families = {
        "majority": [("default", DummyClassifier(strategy="most_frequent"))],
        "logistic": [
            (
                str(regularization),
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=regularization,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            )
            for regularization in (0.1, 1.0, 10.0)
        ],
        "svm": [
            (
                str(regularization),
                make_pipeline(
                    StandardScaler(),
                    SVC(
                        C=regularization,
                        gamma="scale",
                        class_weight="balanced",
                        cache_size=512,
                    ),
                ),
            )
            for regularization in (0.1, 1.0, 10.0)
        ],
    }
    for name, candidates in families.items():
        if args.model != "all" and name != args.model:
            continue
        output_dir = start_run(
            Path(args.output_dir) / name,
            rows,
            classes,
            split,
            {
                "kind": "classical",
                "family": name,
                "augmentation": False,
                "selection": "val_f1_macro",
                "grid_C": [0.1, 1.0, 10.0],
            },
        )
        best_score = -1.0
        best_model = None
        trials = []
        for regularization, model in candidates:
            start = time.perf_counter()
            model.fit(train_features, train_labels)
            predictions = model.predict(val_features)
            result = scores(val_labels, predictions, classes)
            trials.append(
                {
                    "C": regularization,
                    **result,
                    "fit_and_val_seconds": time.perf_counter() - start,
                }
            )
            print(name, regularization, result, flush=True)
            if result["f1_macro"] > best_score:
                best_model, best_score, best_regularization = (
                    model,
                    result["f1_macro"],
                    regularization,
                )
        # 3. Save the selected model and its validation results together.
        joblib.dump(best_model, output_dir / "model.joblib")
        pd.DataFrame(trials).to_csv(output_dir / "validation_search.csv", index=False)
        write_json(
            {"kind": "classical", "family": name, "C": best_regularization},
            output_dir / "meta.json",
        )
        save_predictions(
            output_dir,
            "val",
            val_rows,
            val_labels,
            best_model.predict(val_features),
            classes,
        )
        write_json({"status": "complete"}, output_dir / "status.json")


def train_medical(args):
    """Fine-tune the classifier first, then unfreeze the full pretrained model."""
    # 1. Check data and pretraining provenance before starting.
    settings_config = config()
    finetune = settings_config["finetune"]
    if not 0 <= args.freeze_epochs < finetune["epochs"]:
        raise ValueError("freeze-epochs must be between 0 and epochs-1")
    rows, classes, split, data = medical_rows()
    provenance = verify_pretrained(args.pretrained, split)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    settings = {
        **vars(args),
        "pretrained": str(args.pretrained.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "kind": "spoter",
        "selection": "val_f1_macro",
        "pretrained_provenance": provenance,
    }
    output_dir = start_run(args.output_dir, rows, classes, split, settings)
    # 2. Prepare training sequences and the untouched validation set.
    train_sequences, original_labels, _ = load_part(
        rows, "train", data, classes, "sequences"
    )
    val_inputs, val_labels, val_rows = load_part(rows, "val", data, classes)

    # Use the original class counts for both augmentation settings.
    class_weights = weights(original_labels, len(classes))
    augmented_sequences, augmented_labels = list(train_sequences), list(original_labels)
    if args.augmentation == "on":
        groups = defaultdict(list)
        for x, y in zip(train_sequences, original_labels):
            groups[int(y)].append(x)
        for y, sequences in groups.items():
            added = augment_class(sequences, finetune["min_samples_after_augment"])
            augmented_sequences.extend(added)
            augmented_labels.extend([y] * len(added))

    # 3. Pad sequences and build mini-batches.
    max_len = settings_config["model"]["max_len"]
    train_inputs = np.stack([padded(x, max_len) for x in augmented_sequences])
    train_labels = np.asarray(augmented_labels, dtype=np.int64)
    sampler = WeightedRandomSampler(
        torch.tensor(class_weights[train_labels]), len(train_labels), replacement=True
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_inputs), torch.from_numpy(train_labels)),
        batch_size=finetune["batch_size"],
        sampler=sampler,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_inputs), torch.from_numpy(val_labels)),
        batch_size=finetune["batch_size"],
    )

    # 4. Load the pretrained backbone and initialize the medical classifier.
    model = SPOTER(
        num_classes=len(classes),
        feature_size=225,
        dropout=finetune["dropout"],
        **settings_config["model"],
    ).to(device)
    state = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    result = model.load_state_dict(
        {k: v for k, v in state.items() if not k.startswith("linear_class")},
        strict=False,
    )
    if result.unexpected_keys or any(
        not k.startswith("linear_class") for k in result.missing_keys
    ):
        raise ValueError("Architecture does not match the pretrained weights")
    for name, param in model.named_parameters():
        param.requires_grad = args.freeze_epochs == 0 or name.startswith("linear_class")
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=device), label_smoothing=0.1
    )

    def optimizer_for(lr):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
        )

    # 5. Train in two phases, selecting checkpoints by validation macro F1.
    optimizer = optimizer_for(finetune["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune["epochs"]
    )
    best_score, epochs_without_improvement, history = -1.0, 0, []
    write_json(
        {
            "kind": "spoter",
            "n_classes": len(classes),
            "feature_size": 225,
            "dropout": finetune["dropout"],
            "preprocessing": PREPROCESSING,
            **settings_config["model"],
        },
        output_dir / "meta.json",
    )
    write_json(
        {
            "status": "running",
            "train_original": len(original_labels),
            "train_augmented": len(train_labels),
        },
        output_dir / "status.json",
    )

    for epoch in range(1, finetune["epochs"] + 1):
        if args.freeze_epochs and epoch == args.freeze_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optimizer_for(finetune["learning_rate"] * 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, finetune["epochs"] - args.freeze_epochs)
            )
            epochs_without_improvement = 0  # Global best is NOT reset at unfreezing.

        # Update weights using training batches only.
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)

        # Evaluate without updates, then save the best checkpoint.
        predictions = predict_batches(model, val_loader, device)
        result = scores(val_labels, predictions, classes)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_labels),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **result,
            }
        )
        scheduler.step()
        write_json(history, output_dir / "history.json")
        print(history[-1], flush=True)
        if result["f1_macro"] > best_score + 1e-6:
            best_score, epochs_without_improvement = result["f1_macro"], 0
            atomic_save(model.state_dict(), output_dir / "best_model.pt")
            write_json(
                {"epoch": epoch, "metric": "val_f1_macro", "value": best_score},
                output_dir / "selection.json",
            )
        else:
            epochs_without_improvement += 1
        # Always allow the full fine-tuning phase to begin.
        if (
            epoch > args.freeze_epochs
            and epochs_without_improvement >= finetune["patience"]
        ):
            break

    # 6. Export validation predictions from the best checkpoint.
    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    save_predictions(
        output_dir,
        "val",
        val_rows,
        val_labels,
        predict_batches(model, val_loader, device),
        classes,
    )
    write_json(
        {"status": "complete", "epochs": len(history), "best_val_f1_macro": best_score},
        output_dir / "status.json",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser(
        "classical", help="Majority baseline, logistic regression and SVM"
    )
    p.add_argument(
        "--model", choices=["all", "majority", "logistic", "svm"], default="all"
    )
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs/classical")
    p = commands.add_parser(
        "medical", help="Fine-tune pretrained SPOTER on medical signs"
    )
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--augmentation", choices=["on", "off"], default="on")
    p.add_argument("--freeze-epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.command == "classical":
        train_classical(args)
    else:
        train_medical(args)


if __name__ == "__main__":
    main()
