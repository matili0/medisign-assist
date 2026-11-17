"""Integration checks for the supplied model, failure cases and local detectors."""
import json
from pathlib import Path
from types import SimpleNamespace
import shutil
import socket
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from artifacts import validate_bundle
from app import CaptureState
from camera import Camera, get_landmarks
from model import RecognitionEngine


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.engine = RecognitionEngine(ROOT / 'models/spoter_legacy')
        cls.example = np.load(ROOT / 'examples/boire.npy', allow_pickle=False)

    def test_supplied_example_without_network(self):
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')):
            engine = RecognitionEngine(ROOT / 'models/spoter_legacy')
            result = engine.predict(self.example)
        self.assertEqual(len(result['top']), 5)
        self.assertTrue(all(name in engine.classes and 0 <= score <= 1 for name, score in result['top']))

    def test_empty_rejected(self):
        self.assertIsNone(self.engine.predict(np.empty((0, 225)))['label'])

    def test_nonfinite_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            x = self.example.copy()
            x[0, 0] = value
            self.assertIsNone(self.engine.predict(x)['label'])

    def test_wrong_dimensions_rejected(self):
        for x in (np.zeros((10, 224)), np.zeros(225), np.zeros((2, 3, 225))):
            self.assertIsNone(self.engine.predict(x)['label'])

    def test_no_landmarks_rejected(self):
        self.assertIsNone(self.engine.predict(np.zeros((20, 225)))['label'])

    def test_body_and_hand_must_coincide(self):
        x = np.zeros((2, 225), dtype=np.float32)
        x[0, :99] = 0.4
        x[1, 99:] = 0.5
        self.assertIsNone(self.engine.predict(x)['label'])

    def test_truncation_ignores_frames_after_80(self):
        x = np.tile(self.example, (80 // len(self.example) + 1, 1))[:80]
        y = np.concatenate((x, np.full((7, 225), 0.42, dtype=np.float32)))
        first, second = self.engine.predict(x), self.engine.predict(y)
        self.assertEqual(first['top'], second['top'])

    def test_single_frame_does_not_crash(self):
        result = self.engine.predict(self.example[:1])
        self.assertIn('label', result)

    def test_threshold_one_abstains(self):
        engine = RecognitionEngine(ROOT / 'models/spoter_legacy', threshold=1)
        result = engine.predict(self.example)
        self.assertLess(result['score'], 1)
        self.assertIsNone(result['label'])

    def test_missing_bundle_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                RecognitionEngine(folder)

    def test_tampered_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'bundle'
            shutil.copytree(ROOT / 'models/spoter_legacy', target)
            meta = json.loads((target / 'meta.json').read_text())
            meta['max_len'] = 79
            (target / 'meta.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                validate_bundle(target)

    def test_capture_start_clears_previous_frames(self):
        state = CaptureState(frames=[np.ones(225)])
        state.start()
        self.assertTrue(state.recording)
        self.assertEqual(state.frames, [])
        state.reset()
        self.assertFalse(state.recording)
        self.assertIsNone(state.started)

    def test_camera_close_releases_resources_once(self):
        camera = Camera()
        cap, pose, hands = Mock(), Mock(), Mock()
        camera.cap, camera.pose, camera.hands = cap, pose, hands
        camera.close()
        camera.close()
        cap.release.assert_called_once()
        pose.close.assert_called_once()
        hands.close.assert_called_once()

    def test_detectors_load_locally_and_blank_frame_has_no_landmarks(self):
        video = Mock()
        video.read.return_value = True, np.zeros((256, 256, 3), dtype=np.uint8)
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')), \
                patch('cv2.VideoCapture', return_value=video), patch('camera.monotonic_ns', return_value=42):
            camera = Camera().open()
            try:
                for _ in range(3):
                    _, coordinates = camera.read(extract=True, show_body=True)
                    self.assertEqual(coordinates.shape, (225,))
                    self.assertFalse(coordinates.any())
            finally:
                camera.close()

    def test_two_hands_with_same_label_follow_the_body_wrists(self):
        def point(x):
            return SimpleNamespace(x=x, y=0.5, z=0.0, visibility=1.0)

        body = [point(0.5) for _ in range(33)]
        body[15], body[16] = point(0.8), point(0.2)
        pose, hands = Mock(), Mock()
        pose.detect_for_video.return_value.pose_landmarks = [body]
        hand_result = hands.detect_for_video.return_value
        hand_result.hand_landmarks = [[point(x)] * 21 for x in (0.22, 0.79)]
        hand_result.handedness = [[SimpleNamespace(category_name='Left', score=0.9)]] * 2
        rgb = np.zeros((100, 120, 3), dtype=np.uint8)
        coords = get_landmarks(rgb, pose, hands, timestamp_ms=10)
        np.testing.assert_allclose(coords[[99, 162]], [0.79, 0.22])


if __name__ == '__main__':
    unittest.main(verbosity=2)
