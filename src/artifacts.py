"""Read settings, save experiments and check model files before use.

Each medical experiment keeps its settings, classes and data split beside its
weights.
"""

import json
import os
import random
from pathlib import Path

import numpy as np

from preprocessing import PREPROCESSING

ROOT = Path(__file__).resolve().parents[1]


def config():
    """Load the shared settings from config.json."""
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def write_json(value, path):
    """Write readable JSON, replacing the old file only after success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_save(value, path):
    """Save Torch weights or training state without truncating the previous file."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def rng_state(generators):
    """Remember random draws so pretraining can resume where it stopped."""
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "generators": {
            name: generator.get_state() for name, generator in generators.items()
        },
    }


def restore_rng(state, generators):
    """Restore the random states stored in a pretraining checkpoint."""
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    for name, value in state["generators"].items():
        generators[name].set_state(value.cpu())


def safe_output(path):
    """Keep generated files outside the shared dataset."""
    target = Path(path).resolve()
    dataset = (ROOT / config()["dataset_path"]).resolve()
    if target == dataset or dataset in target.parents:
        raise ValueError(
            "The shared dataset is read-only. Choose another output folder."
        )
    return target


def verify_pretrained(checkpoint, split):
    """Check pretraining signers and exclude the medical vocabulary."""
    checkpoint = Path(checkpoint)
    metadata_file = checkpoint.with_name("pretrained_meta.json")
    if not checkpoint.is_file() or not metadata_file.is_file():
        raise ValueError("Pretrained weights or pretrained_meta.json are missing.")

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    for field, part in [("training_signers", "train"), ("validation_signers", "val")]:
        signers = set(metadata.get(field, []))
        if not signers or not signers <= set(split["signers"][part]):
            raise ValueError(
                f"Pretraining {part} signers do not match the current split."
            )

    classes = metadata.get("classes", [])
    if not classes or set(classes) & set(split["medical_classes"]):
        raise ValueError(
            "Pretraining classes must be documented and exclude medical signs."
        )
    return metadata


def start_run(output_dir, rows, classes, split, settings):
    """Create a medical experiment folder without overwriting an existing run."""
    output_dir = safe_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    rows[["id", "sign", "signer", "split"]].to_csv(
        output_dir / "instances_split.csv", index=False
    )
    write_json(classes, output_dir / "classes.json")
    write_json(
        {
            "config": config(),
            "signers": split["signers"],
            "settings": settings,
            "preprocessing": PREPROCESSING,
        },
        output_dir / "protocol.json",
    )
    return output_dir


def validate_bundle(run):
    """Read a completed model folder and check its classes, settings and weights."""
    run = Path(run)
    metadata = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    classes = json.loads((run / "classes.json").read_text(encoding="utf-8"))
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))

    if status.get("status") != "complete":
        raise ValueError("This training run is incomplete.")
    if protocol.get("preprocessing") != PREPROCESSING:
        raise ValueError("The model uses a different preprocessing method.")
    if not isinstance(classes, list) or not classes:
        raise ValueError("The model needs a nonempty class list.")
    if any(not isinstance(label, str) or not label for label in classes):
        raise ValueError("Each class must have a name.")
    if len(set(classes)) != len(classes):
        raise ValueError("The class list contains duplicates.")

    model_config = protocol["config"]["model"]
    if not isinstance(model_config["max_len"], int) or model_config["max_len"] < 1:
        raise ValueError("The maximum sequence length must be a positive integer.")

    kind = metadata.get("kind")
    if kind == "spoter":
        if (
            metadata.get("n_classes") != len(classes)
            or metadata.get("feature_size") != 225
        ):
            raise ValueError("Model dimensions do not match the input or class list.")
        for name, value in model_config.items():
            if metadata.get(name) != value:
                raise ValueError(
                    f"Model metadata and training settings disagree: {name}."
                )
        weights = run / "best_model.pt"
    elif kind == "classical":
        weights = run / "model.joblib"
    else:
        raise ValueError("Unknown model type.")

    if not weights.is_file() or weights.stat().st_size == 0:
        raise ValueError("Model weights are missing or empty.")
    return metadata, classes, protocol


def verify_run(run, rows):
    """Before evaluation, check the saved labels, data split and model settings."""
    import pandas as pd

    _, _, protocol = validate_bundle(run)
    saved = pd.read_csv(Path(run) / "instances_split.csv", dtype=str)
    columns = ["id", "sign", "signer", "split"]
    saved = saved[columns].sort_values("id").reset_index(drop=True)
    current = rows[columns].astype(str).sort_values("id").reset_index(drop=True)

    if not saved.equals(current):
        raise ValueError("The examples, labels or data split differ from training.")
    if protocol["config"]["model"] != config()["model"]:
        raise ValueError("Model settings have changed since training.")
    return protocol
