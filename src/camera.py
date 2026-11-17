"""Camera acquisition and raw body/hand points; no model or GUI dependencies."""

from pathlib import Path
from time import monotonic_ns

import numpy as np

from artifacts import ROOT

N_POSE, N_HAND = 33, 21


def _hand_slots(result, body, aspect):
    """Match detected hands to anatomical wrists, in the order used by LSFB."""
    slots = np.zeros((2, N_HAND, 3), dtype=np.float32)
    hands = np.array([[(p.x, p.y, p.z) for p in hand]
                      for hand in result.hand_landmarks], dtype=np.float32)
    if not len(hands):
        return slots

    if body and all(body[i].visibility >= 0.5 for i in (15, 16)):
        wrists = np.array([(body[i].x, body[i].y) for i in (15, 16)])
        delta = (hands[:, None, 0, :2] - wrists) * [aspect, 1]
        distance = np.linalg.norm(delta, axis=2)
        if len(hands) == 1:
            slots[distance[0].argmin()] = hands[0]
        elif distance[0, 0] + distance[1, 1] <= distance[0, 1] + distance[1, 0]:
            slots[:] = hands
        else:
            slots[:] = hands[::-1]
    else:
        # With unreliable body wrists, keep the best hand per detector label.
        scores = [-1., -1.]
        for hand, categories in zip(hands, result.handedness):
            side = categories[0]
            index = 0 if side.category_name == "Left" else 1
            if side.score > scores[index]:
                slots[index] = hand
                scores[index] = side.score
    return slots


def get_landmarks(rgb, pose_det, hand_det, show_body=False, timestamp_ms=None):
    """Extract 225 coordinates for recognition and optionally mark the preview."""
    import mediapipe as mp

    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def detect(detector):
        if timestamp_ms is None:
            return detector.detect(img)
        return detector.detect_for_video(img, timestamp_ms)

    pr = detect(pose_det)
    body = pr.pose_landmarks[0] if pr.pose_landmarks else []
    hr = detect(hand_det) if hand_det is not None else None

    if show_body:
        import cv2

        height, width = rgb.shape[:2]
        for lm in body:
            if lm.visibility >= 0.5 and 0 <= lm.x < 1 and 0 <= lm.y < 1:
                center = (int(lm.x * width), int(lm.y * height))
                cv2.circle(rgb, center, 4, (64, 255, 100), -1, cv2.LINE_AA)
        if hr is not None:
            for hand in hr.hand_landmarks:
                for lm in hand:
                    if 0 <= lm.x < 1 and 0 <= lm.y < 1:
                        cv2.circle(rgb, (int(lm.x * width), int(lm.y * height)),
                                   2, (255, 200, 70), -1, cv2.LINE_AA)

    if hr is None:
        return None

    pose = (
        np.array([[lm.x, lm.y, lm.z] for lm in body], dtype=np.float32).flatten()
        if body
        else np.zeros(N_POSE * 3, dtype=np.float32)
    )

    hands = _hand_slots(hr, body, rgb.shape[1] / rgb.shape[0])
    return np.concatenate([pose, hands.ravel()])


class Camera:
    """Manage a webcam and the two local MediaPipe landmark detectors."""

    def __init__(self, index=0, directory=None):
        self.index = index
        self.directory = Path(directory) if directory else ROOT / "models/mediapipe"
        self.cap = self.pose = self.hands = None
        self.last_timestamp_ms = -1

    def open(self):
        import cv2
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        try:
            self.pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(
                            self.directory / "pose_landmarker_lite.task"
                        )
                    ),
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                )
            )
            self.hands = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(self.directory / "hand_landmarker.task")
                    ),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=2,
                )
            )
            self.cap = cv2.VideoCapture(self.index)
            if not self.cap.isOpened():
                raise RuntimeError("Camera is unavailable")
        except Exception:
            self.close()
            raise
        return self

    def read(self, extract=False, show_body=False):
        import cv2

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Could not read a camera frame")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        points = None
        if extract or show_body:
            self.last_timestamp_ms = max(self.last_timestamp_ms + 1, monotonic_ns() // 1_000_000)
            points = get_landmarks(
                rgb, self.pose, self.hands, show_body=show_body, timestamp_ms=self.last_timestamp_ms
            )
        return rgb, points if extract else None

    def close(self):
        for key in ("cap", "pose", "hands"):
            obj = getattr(self, key)
            if obj is not None:
                if key == "cap":
                    obj.release()
                else:
                    obj.close()
                setattr(self, key, None)
