"""Evaluate models, compare results and study abstention without training."""

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from artifacts import safe_output, verify_run, write_json


def scores(y, pred, classes):
    """Score all medical classes, including those with no correct predictions."""
    labels = list(range(len(classes)))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(
            f1_score(y, pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y, pred, labels=labels, average="weighted", zero_division=0)
        ),
    }


def save_predictions(out, part, rows, y, pred, classes):
    """Save predictions, aggregate scores, per-class metrics and confusion matrix."""
    out = Path(out)
    frame = rows[["id", "signer", "sign"]].copy()
    frame["predicted"] = [classes[int(i)] for i in pred]
    frame.to_csv(out / f"{part}_predictions.csv", index=False)
    result = scores(y, pred, classes)
    write_json(result, out / f"{part}_metrics.json")
    report = classification_report(
        y,
        pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    write_json(report, out / f"{part}_report.json")
    pd.DataFrame(
        confusion_matrix(y, pred, labels=list(range(len(classes)))),
        index=classes,
        columns=classes,
    ).to_csv(out / f"{part}_confusion.csv")
    return result


def metrics_from_confusion(cm, train_counts):
    """Summarize auxiliary validation, excluding classes absent from that split."""
    cm = np.asarray(cm, dtype=np.int64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = cm.diagonal()
    precision = np.divide(
        tp, predicted, out=np.zeros(len(tp), dtype=float), where=predicted > 0
    )
    recall = np.divide(
        tp, support, out=np.zeros(len(tp), dtype=float), where=support > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(tp), dtype=float),
        where=(precision + recall) > 0,
    )
    present = support > 0
    rare = present & (train_counts <= np.quantile(train_counts, 0.25))
    frequent = present & (train_counts >= np.quantile(train_counts, 0.75))

    def average(values, mask):
        return float(values[mask].mean()) if mask.any() else None

    return {
        "accuracy": float(tp.sum() / max(1, support.sum())),
        "macro_f1": average(f1, present),
        "macro_recall": average(recall, present),
        "rare_macro_f1": average(f1, rare),
        "frequent_macro_f1": average(f1, frequent),
        "evaluated_classes": int(present.sum()),
        "absent_classes": int((~present).sum()),
        "per_class": [
            {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(n),
            }
            for p, r, f, n in zip(precision, recall, f1, support)
        ],
    }


def evaluate_run(args):
    """Measure scores and CPU prediction time for a saved model."""
    import psutil
    import torch

    from data import load_part, medical_rows
    from model import RecognitionEngine

    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    from threadpoolctl import threadpool_limits

    # Load the same examples and preparation used during training.
    rows, classes, _, data = medical_rows()
    protocol = verify_run(args.run, rows)
    engine = RecognitionEngine(args.run)
    if classes != engine.classes:
        raise ValueError("Class order does not match")
    args.output_dir = safe_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    kind = engine.meta["kind"]
    x, y, selected = load_part(
        rows, args.split, data, classes, "tensor" if kind == "spoter" else "features"
    )
    artifact = args.run / ("best_model.pt" if kind == "spoter" else "model.joblib")

    def predict_one(i):
        return int(engine.predict_prepared(x[i : i + 1])[0])

    # Warm up, then time one prepared example per prediction.
    with threadpool_limits(limits=args.threads):
        for i in range(min(10, len(y))):
            predict_one(i)
        predicted, durations = [], []
        for i in range(len(y)):
            start = time.perf_counter()
            predicted.append(predict_one(i))
            durations.append(1000 * (time.perf_counter() - start))
    save_predictions(args.output_dir, args.split, selected, y, predicted, classes)

    write_json(
        {
            "source_run": str(args.run),
            "split": args.split,
            "source_protocol": protocol,
            "model_bytes": artifact.stat().st_size,
            "cpu_median_ms": float(np.median(durations)),
            "cpu_p95_ms": float(np.percentile(durations, 95)),
            "threads": args.threads,
            "batch_size": 1,
            "warmup": min(10, len(y)),
            "machine": platform.platform(),
            "machine_id": args.machine_id or socket.gethostname(),
            "processor": platform.processor(),
            "process_rss_bytes_after_evaluation": psutil.Process().memory_info().rss,
            "timing_scope": "prediction only; excludes pose extraction and preprocessing",
            "memory_scope": "whole Python process including loaded data; not model-only or peak",
        },
        args.output_dir / "evaluation.json",
    )
    print(f"Evaluation complete: {args.output_dir}")


def compare_results(args):
    """Compare saved scores on matching examples and, optionally, matching hardware."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries, reference = [], None
    for folder in args.evaluations:
        info = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
        part = info["split"]
        frame = pd.read_csv(folder / f"{part}_predictions.csv", dtype=str)
        key = frame[["id", "signer", "sign"]].sort_values("id").reset_index(drop=True)
        signature = (
            part,
            info["source_protocol"]["preprocessing"],
            info["source_protocol"]["config"]["model"]["max_len"],
            info["source_protocol"]["signers"],
        )
        hardware = (
            info.get("machine_id"),
            info.get("processor"),
            info["machine"],
            info["threads"],
            info["batch_size"],
            info["timing_scope"],
        )
        if reference is None:
            reference, ref_sig, ref_hardware = key, signature, hardware
        elif not key.equals(reference) or signature != ref_sig:
            raise ValueError("Cannot compare scores: examples or preprocessing differ")
        if not args.scores_only and (not hardware[0] or hardware != ref_hardware):
            raise ValueError(
                "CPU timings are not comparable. Use --scores-only to compare scores alone."
            )
        metrics = json.loads((folder / f"{part}_metrics.json").read_text())
        summary = {
            "model": folder.name,
            "split": part,
            **metrics,
            "model_bytes": info["model_bytes"],
        }
        if not args.scores_only:
            summary["cpu_median_ms"] = info["cpu_median_ms"]
            summary["cpu_p95_ms"] = info["cpu_p95_ms"]
        summaries.append(summary)
    args.output_dir = safe_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(summaries)
    table.to_csv(args.output_dir / "comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(table.model, table.f1_macro)
    ax.set(ylabel="F1 macro", ylim=(0, 1), title=f"Comparison - {ref_sig[0]}")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(args.output_dir / "comparison.png", dpi=160)
    plt.close(fig)
    (args.output_dir / "interpretation.md").write_text(
        "# Analysis to complete\n\nResults are available in comparison.csv.\n\n"
        "- How much does macro F1 improve, and for which signs?\n"
        "- What are the CPU cost, model size and limitations?\n"
        "- Classical statistics summarize time; SPOTER processes the sequence.\n"
        "- Pretrained SPOTER uses auxiliary data: this compares whole pipelines, "
        "rather than isolating the effect of architecture.\n"
        "- Dataset scores do not measure webcam performance.\n",
        encoding="utf-8",
    )


def sweep_threshold(args):
    """Report coverage and accepted errors for candidate thresholds on validation."""
    from data import load_part, medical_rows
    from model import RecognitionEngine

    rows, classes, _, data = medical_rows()
    verify_run(args.run, rows)
    engine = RecognitionEngine(args.run)
    if engine.meta["kind"] != "spoter":
        raise ValueError("Threshold analysis is available for SPOTER")
    if classes != engine.classes:
        raise ValueError("Class order does not match")
    x, y, selected = load_part(rows, "val", data, classes)
    probs = np.concatenate(
        [
            engine.probabilities_prepared(x[start : start + 32])
            for start in range(0, len(y), 32)
        ]
    )
    confidence, pred = probs.max(1), probs.argmax(1)
    results = []
    for threshold in np.linspace(0, 1, 21):
        keep = confidence >= threshold
        count = int(keep.sum())
        results.append(
            {
                "threshold": float(threshold),
                "accepted": count,
                "coverage": float(keep.mean()),
                "accuracy_accepted": float((pred[keep] == y[keep]).mean())
                if count
                else None,
                "accepted_errors": int((pred[keep] != y[keep]).sum()),
            }
        )
    args.output_dir = safe_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(results).to_csv(args.output_dir / "thresholds.csv", index=False)
    frame = selected[["id", "sign", "signer"]].copy()
    frame["predicted"] = [classes[i] for i in pred]
    frame["score"] = confidence
    frame.to_csv(args.output_dir / "validation_scores.csv", index=False)
    write_json(
        {
            "split": "val",
            "run": str(args.run),
            "note": "Choose the coverage/error tradeoff on validation; scores are not calibrated and unknown gestures may be accepted.",
        },
        args.output_dir / "protocol.json",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run", help="Evaluate a model without training")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--machine-id", help="Measurement machine ID (default: local hostname)"
    )
    p = commands.add_parser("compare", help="Compare saved evaluations")
    p.add_argument("--evaluations", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scores-only", action="store_true")
    p = commands.add_parser(
        "threshold", help="Study SPOTER coverage and errors on validation"
    )
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        evaluate_run(args)
    elif args.command == "compare":
        compare_results(args)
    else:
        sweep_threshold(args)


if __name__ == "__main__":
    main()
