"""Shared corpus reading and audits; no preparation command writes to the corpus."""

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from artifacts import ROOT, config, safe_output, write_json
from preprocessing import features, normalized, padded, padded_sequence

PARTS = ("train", "val", "test")


def check_groups(groups, expected):
    """Each signer must belong to exactly one nonempty partition."""
    train, validation, test = [set(groups[part]) for part in PARTS]
    if not train or not validation or not test:
        raise ValueError("Each partition must contain at least one signer.")
    if train & validation or train & test or validation & test:
        raise ValueError("A signer appears in multiple partitions.")
    if train | validation | test != set(expected):
        raise ValueError("Signer groups do not match the corpus.")


def load_split():
    """Read the existing signer groups without creating or changing a split."""
    cfg = config()
    data = (ROOT / cfg["dataset_path"]).resolve()
    corpus = pd.read_csv(data / "instances.csv", dtype={"id": str, "signer": str})
    if (
        corpus[["id", "sign", "signer"]].isna().any().any()
        or corpus.id.duplicated().any()
    ):
        raise ValueError("Missing metadata or duplicate example IDs.")
    path = data / "splits" / "signers.json"
    if not path.exists():
        raise FileNotFoundError(
            "The saved signer split is missing. Check dataset_path."
        )
    info = json.loads(path.read_text(encoding="utf-8"))
    check_groups(info["signers"], corpus.signer.unique())
    # Keep only the readable fields needed by this project.
    info = {"signers": info["signers"], "medical_classes": info["medical_classes"]}
    mapping = {s: p for p, signers in info["signers"].items() for s in signers}
    corpus["split"] = corpus.signer.map(mapping)
    return corpus, info


def available_rows(frame, directory):
    """Missing nonempty files are errors; known empty source poses are excluded."""
    directory = Path(directory)
    keep, empty = [], []
    for row in tqdm(
        frame.itertuples(),
        total=len(frame),
        desc=f"Checking {directory.name}",
        unit="file",
    ):
        path = directory / f"{row.id}.npy"
        if path.exists():
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if arr.ndim != 2 or arr.shape[1] != 225:
                raise ValueError(f"Invalid coordinate shape: {path}")
            if len(arr):
                keep.append(row.id)
            else:
                empty.append(row.id)
        else:
            raw = directory.parent / "poses" / "pose" / f"{row.id}.npy"
            if (
                raw.exists()
                and len(np.load(raw, mmap_mode="r", allow_pickle=False)) == 0
            ):
                empty.append(row.id)
            else:
                raise FileNotFoundError(
                    f"Missing prepared file: {path}. Finish data preparation before training."
                )
    result = frame[frame.id.isin(keep)].copy().reset_index(drop=True)
    if result.empty:
        raise ValueError(f"No usable sequence in {directory}")
    if empty:
        print(f"{len(empty)} empty sequences excluded from {directory.name}")
    return result


def coverage(frame):
    return pd.crosstab(frame.sign, frame.split).reindex(columns=PARTS, fill_value=0)


def pretrain_candidates(corpus, info, minimum=50):
    """Exclude medical signs and test signers from the auxiliary task."""
    frame = corpus[~corpus.sign.isin(info["medical_classes"])].copy()
    counts = frame.sign.value_counts()
    frame = frame[frame.sign.isin(counts[counts >= minimum].index)]
    return frame[frame.split.isin(["train", "val"])].copy()


def medical_rows():
    """Select the medical vocabulary and check class coverage in every partition."""
    cfg = config()
    data = ROOT / cfg["dataset_path"]
    corpus, split = load_split()
    index = pd.read_csv(data / "index.csv", dtype={"id": str})
    if index.id.duplicated().any() or not set(index.id) <= set(corpus.id):
        raise ValueError("Invalid medical index")
    rows = available_rows(corpus[corpus.id.isin(index.id)].copy(), data / "landmarks")
    classes = sorted(split["medical_classes"])
    for part in ("train", "val", "test"):
        if set(rows.loc[rows.split == part, "sign"]) != set(classes):
            raise ValueError(f"Missing medical classes in {part}")
    return rows, classes, split, data


def load_part(rows, part, data, classes, representation="tensor"):
    """Build features, normalized sequences or padded tensors for one partition."""
    selected = rows[rows.split == part].copy().reset_index(drop=True)
    max_len = config()["model"]["max_len"]
    inputs = []
    for row in selected.itertuples():
        sequence = np.load(data / "landmarks" / f"{row.id}.npy", allow_pickle=False)
        if representation == "features":
            inputs.append(features(sequence, max_len))
        else:
            inputs.append(normalized(sequence))
    labels = np.array([classes.index(s) for s in selected.sign], dtype=np.int64)
    if representation == "tensor":
        inputs = np.stack([padded(sequence, max_len) for sequence in inputs])
    elif representation == "features":
        inputs = np.stack(inputs)
    return inputs, labels, selected


class PoseDataset(Dataset):
    """Load already-normalized auxiliary poses, optionally from a disk cache."""

    def __init__(self, paths, labels, max_len, cache_path=None):
        self.paths = [str(p) for p in paths]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.max_len = max_len
        self.cache_path = str(cache_path) if cache_path else None
        self._cache = None

    def __len__(self):
        return len(self.labels)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None  # Windows workers open their own disk mapping.
        return state

    def __getitem__(self, index):
        if self.cache_path:
            if self._cache is None:
                self._cache = np.load(
                    self.cache_path, mmap_mode="r", allow_pickle=False
                )
            x = np.array(self._cache[index], copy=True)
        else:
            x = padded_sequence(self.paths[index], self.max_len)
        return torch.from_numpy(x), torch.tensor(self.labels[index], dtype=torch.long)


def disk_cache(paths, max_len, path):
    """Reuse padded arrays while source paths, sizes and modification times match."""
    path = safe_output(path)
    records = []
    for name in paths:
        p = Path(name)
        st = p.stat()
        records.append([str(p.resolve()), st.st_size, st.st_mtime_ns])
    cache_info = {"max_len": max_len, "sources": records}
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        saved = json.loads(meta.read_text(encoding="utf-8"))
        if saved == cache_info:
            try:
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
                valid = (
                    arr.shape == (len(paths), max_len, 225) and arr.dtype == np.float32
                )
                del arr
            except (ValueError, EOFError, OSError):
                valid = False
            if valid:
                return path
    path.parent.mkdir(parents=True, exist_ok=True)
    required = len(paths) * max_len * 225 * 4 + 1024 * 1024
    if shutil.disk_usage(path.parent).free < required:
        raise OSError(
            f"Not enough disk space for the cache ({required / 1e9:.2f} GB). Use --cache files."
        )
    tmp = path.with_name(path.name + ".tmp")
    arr = np.lib.format.open_memmap(
        tmp, mode="w+", dtype="float32", shape=(len(paths), max_len, 225)
    )
    for i, source in enumerate(tqdm(paths, desc="Building disk cache")):
        arr[i] = padded_sequence(source, max_len)
    arr.flush()
    del arr
    os.replace(tmp, path)
    write_json(cache_info, meta)
    return path


def audit(args):
    """Check array shapes and values, then write counts and a readable file list."""
    rows, classes, split, data = medical_rows()
    args.output_dir = safe_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows.to_csv(args.output_dir / "instances_split.csv", index=False)
    pd.crosstab(rows.sign, rows.split).to_csv(args.output_dir / "class_counts.csv")
    all_rows = [(rows, "landmarks", False)]
    if args.include_pretrain:
        corpus, _ = load_split()
        auxiliary = available_rows(
            pretrain_candidates(corpus, split), data / "pretrain_landmarks"
        )
        all_rows.append((auxiliary, "pretrain_landmarks", True))
    records, errors = [], []
    for frame, folder, already_normalized in all_rows:
        for row in frame.itertuples():
            path = data / folder / f"{row.id}.npy"
            try:
                x = np.load(path, allow_pickle=False)
                if (
                    x.ndim != 2
                    or x.shape[1] != 225
                    or not len(x)
                    or not np.isfinite(x).all()
                ):
                    raise ValueError("Invalid shape, length or values")
                if not already_normalized:
                    normalized(x)
                records.append({"id": row.id, "folder": folder, "frames": len(x)})
            except (ValueError, OSError) as exc:
                errors.append({"id": row.id, "folder": folder, "error": str(exc)})
    pd.DataFrame(records).to_csv(args.output_dir / "files.csv", index=False)
    write_json(
        {
            "valid_files": len(records),
            "errors": errors,
            "classes": classes,
            "signers": split["signers"],
            "includes_pretrain": args.include_pretrain,
        },
        args.output_dir / "audit.json",
    )
    if errors:
        raise SystemExit(f"Audit failed: {len(errors)} invalid files")
    print(f"Audit passed: {len(records)} files checked")


def check_environment():
    """Check imports, dataset paths and local camera assets without opening a camera."""
    checks = {}
    for name in (
        "numpy",
        "pandas",
        "torch",
        "sklearn",
        "cv2",
        "mediapipe",
        "PIL",
        "tkinter",
        "psutil",
    ):
        try:
            importlib.import_module(name)
            checks[name] = "OK"
        except Exception as exc:
            checks[name] = str(exc)
    source = (ROOT / config()["dataset_path"]).resolve()
    for relative in ("instances.csv", "splits/signers.json"):
        checks[str(source / relative)] = (
            "OK" if (source / relative).is_file() else "Dataset missing"
        )
    for relative in ("pose_landmarker_lite.task", "hand_landmarker.task"):
        checks[relative] = (
            "OK"
            if (ROOT / "models/mediapipe" / relative).is_file()
            else "Detector missing"
        )
    checks["accelerator"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    checks["python"] = sys.executable
    print(json.dumps(checks, indent=2))
    return (
        0
        if all(
            v == "OK" for k, v in checks.items() if k not in ("accelerator", "python")
        )
        else 1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "check-env", help="Check installation without camera access or training"
    )
    p = commands.add_parser("audit", help="Read and check the existing corpus")
    p.add_argument("--include-pretrain", action="store_true")
    p.add_argument("--output-dir", type=Path, default=ROOT / "evidence/data_audit_full")
    args = parser.parse_args()
    if args.command == "check-env":
        raise SystemExit(check_environment())
    audit(args)


if __name__ == "__main__":
    main()
