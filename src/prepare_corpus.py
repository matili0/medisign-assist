"""Download LSFB-ISOL poses and prepare the inputs used by MediSign.

Only metadata and body/hand poses are fetched. No videos or face landmarks.
The source paths follow the official lsfb-dataset downloader's LSFB v2 layout.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from tqdm import tqdm

from artifacts import ROOT, config, write_json
from data import check_groups, pretrain_candidates
from preprocessing import normalized

SOURCE = "https://lsfb.info.unamur.be/static/datasets/lsfb_v2/isol/"
COMPONENTS = {"pose": 33, "left_hand": 21, "right_hand": 21}


def validate(content, relative):
    """Reject HTML, corrupt files and unexpected coordinate layouts."""
    if relative == "instances.csv":
        frame = pd.read_csv(io.BytesIO(content), dtype=str)
        if not {"id", "sign", "signer"} <= set(frame.columns):
            raise ValueError("Unexpected LSFB metadata columns")
        if frame[["id", "sign", "signer"]].isna().any().any() or frame.id.duplicated().any():
            raise ValueError("Missing or duplicate corpus identifiers")
        # IDs become filenames: reject path components explicitly.
        if not frame.id.str.fullmatch(r"[A-Za-z0-9_-]+").all():
            raise ValueError("Invalid corpus identifier")
    else:
        array = np.load(io.BytesIO(content), allow_pickle=False)
        component = Path(relative).parent.name
        if array.ndim != 3 or array.shape[1:] != (COMPONENTS[component], 3):
            raise ValueError(f"Unexpected {component} shape: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite coordinates: {relative}")


def fetch(destination, relative):
    """Reuse valid local files; download new files atomically with bounded retries."""
    path = destination / relative
    if path.is_file():
        validate(path.read_bytes(), relative)
        return
    for attempt in range(3):
        try:
            with urlopen(SOURCE + relative, timeout=90) as response:
                content = response.read()
            validate(content, relative)
            break
        except HTTPError as exc:
            if exc.code < 500 or attempt == 2:
                raise
            time.sleep(2 ** attempt)
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def merge_pose(destination, identifier):
    """Concatenate body, left hand and right hand without normalizing medical poses."""
    parts = []
    for name in COMPONENTS:
        relative = f"poses/{name}/{identifier}.npy"
        content = (destination / relative).read_bytes()
        validate(content, relative)
        parts.append(np.load(io.BytesIO(content), allow_pickle=False))
    frames = parts[0].shape[0]
    if any(part.shape[0] != frames for part in parts):
        raise ValueError(f"Body and hand frame counts differ: {identifier}")
    return np.concatenate([part.reshape(frames, count * 3)
                           for part, count in zip(parts, COMPONENTS.values())], axis=1).astype(np.float32)


def prepare(destination, rows, directory, normalize=False):
    """Write raw medical poses or once-normalized auxiliary sequences."""
    target = destination / directory
    target.mkdir(parents=True, exist_ok=True)
    empty = 0
    for identifier in tqdm(rows.id, desc=f"Preparing {directory}"):
        sequence = merge_pose(destination, identifier)
        if len(sequence) == 0:
            empty += 1
            continue  # Keep the empty source body file for available_rows().
        if normalize:
            sequence = normalized(sequence)
        path = target / f"{identifier}.npy"
        if path.exists():
            existing = np.load(path, allow_pickle=False)
            if not np.array_equal(existing, sequence):
                raise ValueError(f"Prepared file differs; use a fresh dataset directory: {path}")
        else:
            temporary = path.with_suffix(".npy.part")
            with temporary.open("wb") as handle:
                np.save(handle, sequence)
            os.replace(temporary, path)
    print(f"{directory}: {len(rows) - empty} usable sequences, {empty} empty source sequences")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-pretrain", action="store_true",
                        help="Also download auxiliary nonmedical signs for train/validation signers")
    parser.add_argument("--prepare-only", action="store_true", help="Use already downloaded files without network")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    destination = (ROOT / config()["dataset_path"]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not args.prepare_only:
        fetch(destination, "instances.csv")
    validate((destination / "instances.csv").read_bytes(), "instances.csv")
    corpus = pd.read_csv(destination / "instances.csv", dtype=str)
    bundle = ROOT / "models/spoter_legacy"
    protocol = json.loads((bundle / "protocol.json").read_text(encoding="utf-8"))
    split = protocol["split"]
    classes = json.loads((bundle / "classes.json").read_text(encoding="utf-8"))
    check_groups(split["signers"], corpus.signer.unique())
    mapping = {s: part for part, signers in split["signers"].items() for s in signers}
    corpus["split"] = corpus.signer.map(mapping)
    medical = corpus[corpus.sign.isin(classes)].copy()
    if set(medical.sign) != set(classes):
        raise ValueError("The downloaded corpus does not contain the full medical vocabulary")
    auxiliary = pretrain_candidates(corpus, split) if args.include_pretrain else corpus.iloc[:0]
    selected = pd.concat([medical, auxiliary])
    if not args.prepare_only:
        origins = [f"poses/{name}/{identifier}.npy" for identifier in selected.id for name in COMPONENTS]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(tqdm(pool.map(lambda relative: fetch(destination, relative), origins),
                      total=len(origins), desc="Downloading body and hand poses"))
    # Preserve the recorded signer partition; never infer a new split from test scores.
    split_path = destination / "splits/signers.json"
    if split_path.exists() and json.loads(split_path.read_text(encoding="utf-8")) != split:
        raise ValueError("An existing signer split differs; use a fresh dataset directory")
    prepare(destination, medical, "landmarks")
    medical["label"] = medical.sign.map({name: index for index, name in enumerate(sorted(classes))})
    medical[["id", "sign", "label"]].to_csv(destination / "index.csv", index=False)
    write_json(split, split_path)
    if args.include_pretrain:
        prepare(destination, auxiliary, "pretrain_landmarks", normalize=True)
    print("Prepared corpus:", destination)


if __name__ == "__main__":
    main()
