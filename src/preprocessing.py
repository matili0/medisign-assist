"""Geometric normalization and training augmentations for body and hand poses."""

import numpy as np

# 225 features, excluding face landmarks:
#   indices   0 -  98: 33 body joints x 3 coordinates (x, y, z)
#   indices  99 - 161: 21 left-hand joints x 3 coordinates
#   indices 162 - 224: 21 right-hand joints x 3 coordinates

N_BODY = 33
N_HAND = 21
LEFT_HAND_START = N_BODY * 3  # 99
RIGHT_HAND_START = LEFT_HAND_START + N_HAND * 3  # 162
RIGHT_HAND_END = RIGHT_HAND_START + N_HAND * 3  # 225

# Body reference coordinates used for normalization.
_LS_X, _LS_Y = 33, 34  # leftShoulder  (joint 11 × 3)
_RS_X, _RS_Y = 36, 37  # rightShoulder (joint 12 × 3)
_LE_Y = 7  # leftEye y     (joint  2 × 3 + 1)


# Geometric normalization (Bohacek).


def normalize_body(seq: np.ndarray) -> np.ndarray:
    """Normalize body x/y using a box six times the shoulder distance.

    The shoulder midpoint anchors x; the left eye anchors y. Keep z unchanged."""
    out = seq.copy()

    ls_x, ls_y = seq[:, _LS_X], seq[:, _LS_Y]
    rs_x, rs_y = seq[:, _RS_X], seq[:, _RS_Y]
    le_y = seq[:, _LE_Y]

    neck_x = (ls_x + rs_x) * 0.5
    head_metric = np.sqrt((ls_x - rs_x) ** 2 + (ls_y - rs_y) ** 2)

    sp_x = neck_x - 3.0 * head_metric
    sp_y = le_y + head_metric
    ep_x = neck_x + 3.0 * head_metric
    ep_y = sp_y - 6.0 * head_metric

    bbox_w = ep_x - sp_x
    bbox_h = sp_y - ep_y
    valid = (bbox_w > 1e-6) & (bbox_h > 1e-6) & (ls_x != 0) & (rs_x != 0)

    for j in range(N_BODY):
        xi, yi = j * 3, j * 3 + 1
        np.divide(seq[:, xi] - sp_x, bbox_w, out=out[:, xi], where=valid)
        np.divide(seq[:, yi] - ep_y, bbox_h, out=out[:, yi], where=valid)

    return out


def normalize_hands(seq: np.ndarray) -> np.ndarray:
    """Normalize each hand in its own bounding box, frame by frame.

    Hand x/y coordinates become local; z coordinates remain unchanged."""
    out = seq.copy()

    for hand_start in (LEFT_HAND_START, RIGHT_HAND_START):
        x_idx = np.arange(hand_start, hand_start + N_HAND * 3, 3)
        y_idx = np.arange(hand_start + 1, hand_start + N_HAND * 3, 3)

        for t in range(len(seq)):
            x_vals, y_vals = seq[t, x_idx], seq[t, y_idx]
            mask = x_vals != 0
            if mask.sum() < 3:
                continue

            vx, vy = x_vals[mask], y_vals[mask]
            w, h = vx.max() - vx.min(), vy.max() - vy.min()

            if w > h:
                dx, dy = 0.1 * w, 0.1 * w + (w - h) * 0.5
            else:
                dy, dx = 0.1 * h, 0.1 * h + (h - w) * 0.5

            sp_x, ep_x = vx.min() - dx, vx.max() + dx
            sp_y, ep_y = vy.min() - dy, vy.max() + dy
            bw, bh = ep_x - sp_x, ep_y - sp_y

            if bw < 1e-6 or bh < 1e-6:
                continue

            for j in range(N_HAND):
                if seq[t, x_idx[j]] == 0:
                    continue
                out[t, x_idx[j]] = (seq[t, x_idx[j]] - sp_x) / bw
                out[t, y_idx[j]] = (seq[t, y_idx[j]] - sp_y) / bh

    return out


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Normalize the body, then the hands. Input and output: (frames, 225)."""
    return normalize_hands(normalize_body(seq))


# AUGMENTATIONS


def flip_horizontal(seq):
    """Mirror x coordinates and swap the left and right hands."""
    s = seq.copy()
    for start, n in [
        (0, N_BODY),
        (LEFT_HAND_START, N_HAND),
        (RIGHT_HAND_START, N_HAND),
    ]:
        for j in range(n):
            s[:, start + j * 3] = 1.0 - s[:, start + j * 3]
    left = s[:, LEFT_HAND_START:RIGHT_HAND_START].copy()
    right = s[:, RIGHT_HAND_START:RIGHT_HAND_END].copy()
    s[:, LEFT_HAND_START:RIGHT_HAND_START] = right
    s[:, RIGHT_HAND_START:RIGHT_HAND_END] = left
    return s


def add_noise(seq, std=0.005):
    """Add Gaussian noise to all landmarks."""
    return (seq + np.random.normal(0, std, seq.shape)).astype(np.float32)


def scale(seq, low=0.9, high=1.1):
    """Scale around 0.5 to simulate changes in signer-camera distance."""
    return ((seq - 0.5) * np.random.uniform(low, high) + 0.5).astype(np.float32)


def time_warp(seq, low=0.8, high=1.2):
    """Resample time to simulate different signing speeds."""
    T, F = seq.shape
    new_T = max(2, int(T * np.random.uniform(low, high)))
    out = np.empty((new_T, F), dtype=np.float32)
    for i in range(new_T):
        idx = i * (T - 1) / (new_T - 1)
        lo = int(idx)
        hi = min(lo + 1, T - 1)
        a = idx - lo
        out[i] = seq[lo] * (1 - a) + seq[hi] * a
    return out


def temporal_permutation(seq):
    """Shuffle the first third and last quarter of frames; keep the centre.

    Source: SL-SLR (arXiv 2509.05188)."""
    seq = seq.copy()
    T = len(seq)
    n1, n2 = T // 3, T // 4
    if n1 > 1:
        seq[:n1] = seq[np.random.permutation(n1)]
    if n2 > 1:
        seq[T - n2 :] = seq[T - n2 :][np.random.permutation(n2)]
    return seq


def hand_local_noise(seq, std=0.02):
    """Add stronger noise to the hands; adapted from Fink et al., IDA 2025."""
    out = seq.copy()
    out[:, LEFT_HAND_START:RIGHT_HAND_END] += np.random.normal(
        0, std, (len(seq), RIGHT_HAND_END - LEFT_HAND_START)
    )
    return out.astype(np.float32)


def hand_scale(seq, low=0.85, high=1.15):
    """Scale each hand around its wrist; adapted from Fink et al., IDA 2025."""
    factor = np.random.uniform(low, high)
    out = seq.copy()
    T = len(seq)
    for hand_start, hand_end in [
        (LEFT_HAND_START, RIGHT_HAND_START),
        (RIGHT_HAND_START, RIGHT_HAND_END),
    ]:
        wrist = out[:, hand_start : hand_start + 3]
        hand = out[:, hand_start:hand_end].reshape(T, N_HAND, 3)
        hand = wrist[:, np.newaxis, :] + factor * (hand - wrist[:, np.newaxis, :])
        out[:, hand_start:hand_end] = hand.reshape(T, N_HAND * 3)
    return out.astype(np.float32)


def temporal_shift(seq, max_shift=0.1):
    """Trim a short random segment from the start or end."""
    T = len(seq)
    shift = np.random.randint(0, max(1, int(T * max_shift)) + 1)
    if shift == 0:
        return seq
    return seq[shift:] if np.random.rand() < 0.5 else seq[:-shift]


# Transformations sampled by the medical training script.
AUGMENTATIONS = [
    flip_horizontal,
    add_noise,
    scale,
    time_warp,
    temporal_permutation,
    hand_local_noise,
    hand_scale,
    temporal_shift,
]


def augment_class(samples, target):
    """Generate extra sequences until the class reaches the target count."""
    new = []
    while len(samples) + len(new) < target:
        src = samples[np.random.randint(len(samples))]
        aug_fn = AUGMENTATIONS[np.random.randint(len(AUGMENTATIONS))]
        new.append(aug_fn(src))
    return new


PREPROCESSING = "xy-normalization-first80-leftpad-v1"


def normalized(sequence):
    """Validate raw coordinates, then normalize body and hands exactly once."""
    x = np.asarray(sequence, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 225 or not len(x):
        raise ValueError("Expected shape: (frames, 225), with at least one frame")
    if not np.isfinite(x).all():
        raise ValueError("Coordinates contain NaN or infinity")
    y = normalize_sequence(x.copy())
    if not np.isfinite(y).all():
        raise ValueError("Normalization produced NaN or infinity")
    return y


def padded(sequence, max_len=80):
    """Keep the first frames of a normalized sequence and pad on the left."""
    if max_len < 1:
        raise ValueError("max_len must be positive")
    x = np.asarray(sequence, dtype=np.float32)[:max_len]
    if x.ndim != 2 or x.shape[1] != 225 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid normalized sequence")
    out = np.zeros((max_len, 225), dtype=np.float32)
    out[-len(x) :] = x
    return out


def features(sequence, max_len=80):
    """Summarize a sequence as 225 means, deviations and mean absolute movements."""
    x = normalized(sequence)[:max_len].astype(np.float64)
    movement = np.abs(np.diff(x, axis=0)).mean(0) if len(x) > 1 else np.zeros(225)
    return np.concatenate([x.mean(0), x.std(0), movement])


def padded_sequence(path, max_len):
    """Auxiliary arrays are ALREADY normalized; never normalize them twice."""
    return padded(np.load(path, allow_pickle=False), max_len)
