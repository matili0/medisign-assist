"""Corpus preparation checks using small local fixtures; no downloads or training."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare_corpus as preparation
from preprocessing import normalized


class PreparationTests(unittest.TestCase):
    def make_poses(self, root, identifier="sample", frames=3):
        for component, points in preparation.COMPONENTS.items():
            path = root / "poses" / component / f"{identifier}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            array = np.arange(frames * points * 3, dtype=np.float32).reshape(frames, points, 3) / 100
            np.save(path, array)

    def test_medical_raw_and_auxiliary_normalized_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_poses(root)
            rows = pd.DataFrame({"id": ["sample"]})
            merged = preparation.merge_pose(root, "sample")
            preparation.prepare(root, rows, "landmarks")
            preparation.prepare(root, rows, "pretrain_landmarks", normalize=True)
            np.testing.assert_array_equal(np.load(root / "landmarks/sample.npy"), merged)
            np.testing.assert_array_equal(np.load(root / "pretrain_landmarks/sample.npy"), normalized(merged))
            # Repeating preparation must preserve exactly the same coordinates.
            preparation.prepare(root, rows, "pretrain_landmarks", normalize=True)

    def test_empty_source_retained_but_not_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_poses(root, frames=0)
            preparation.prepare(root, pd.DataFrame({"id": ["sample"]}), "landmarks")
            self.assertFalse((root / "landmarks/sample.npy").exists())
            self.assertTrue((root / "poses/pose/sample.npy").exists())

    def test_frame_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_poses(root)
            np.save(root / "poses/left_hand/sample.npy", np.zeros((2, 21, 3)))
            with self.assertRaisesRegex(ValueError, "frame counts"):
                preparation.merge_pose(root, "sample")

    def test_existing_valid_file_needs_no_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_poses(root)
            with patch.object(preparation, "urlopen", side_effect=AssertionError("network")):
                preparation.fetch(root, "poses/pose/sample.npy")

    def test_prepare_only_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            self.make_poses(data, "medical")
            self.make_poses(data, "auxiliary")
            # Include metadata for all three signers so group validation is exercised.
            rows = [{"id": "medical", "sign": "BOIRE", "signer": "S1"},
                    {"id": "other", "sign": "OTHER", "signer": "S2"},
                    {"id": "heldout", "sign": "OTHER", "signer": "S3"}]
            pd.DataFrame(rows).to_csv(data / "instances.csv", index=False)
            bundle = root / "models/spoter_legacy"
            bundle.mkdir(parents=True)
            split = {"signers": {"train": ["S1"], "val": ["S2"], "test": ["S3"]},
                     "medical_classes": ["BOIRE"]}
            (bundle / "protocol.json").write_text(json.dumps({"split": split}))
            (bundle / "classes.json").write_text(json.dumps(["BOIRE"]))
            with patch.object(preparation, "ROOT", root), \
                    patch.object(preparation, "config", return_value={"dataset_path": "data"}), \
                    patch.object(sys, "argv", ["prepare_corpus.py", "--prepare-only"]), \
                    patch.object(preparation, "urlopen", side_effect=AssertionError("network")):
                preparation.main()
            self.assertEqual(json.loads((data / "splits/signers.json").read_text()), split)
            self.assertEqual(pd.read_csv(data / "index.csv").id.tolist(), ["medical"])


if __name__ == "__main__":
    unittest.main()
