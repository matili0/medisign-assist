"""SPOTER architecture and prediction for SPOTER or classical models."""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from artifacts import validate_bundle
from preprocessing import features, normalized, padded


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class SPOTERDecoderLayer(nn.TransformerDecoderLayer):
    """Decoder without self-attention: a single class query reads the sequence.

    Keep the self_attn attribute because PyTorch uses its batch_first setting."""

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=None,
        memory_is_causal=None,
    ):
        # The class query attends to the encoded sequence.
        # Query and memory use batch-first tensors.
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # Feed-forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SPOTER(nn.Module):
    """Classify pose sequences with a Transformer encoder and a class query.

    Inputs have shape (batch, frames, 225). Outputs are logits for each sign.
    The input projection and learned positions feed the encoder; the decoder
    uses one learned query to summarize the sequence for classification."""

    # Training and inference pass settings from config.json or saved metadata.
    def __init__(
        self,
        num_classes,
        feature_size=225,
        hidden_dim=64,
        nhead=4,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=256,
        dropout=0.1,
        max_len=80,
    ):
        super().__init__()

        # Project each frame into the Transformer feature space.
        self.input_proj = (
            nn.Linear(feature_size, hidden_dim)
            if feature_size != hidden_dim
            else nn.Identity()
        )

        # Add one learned position vector per frame.
        self.pos = nn.Parameter(torch.rand(1, max_len, hidden_dim))

        # A learned query summarizes the sequence.
        self.class_query = nn.Parameter(torch.rand(1, 1, hidden_dim))

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        # Use the decoder layer defined above.
        custom_dec = SPOTERDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, "relu", batch_first=True
        )
        self.transformer.decoder.layers = _get_clones(custom_dec, num_decoder_layers)

        # Map the sequence representation to class logits.
        self.linear_class = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x : (B, T, feature_size)
        h = self.input_proj(x) + self.pos  # (B, T, hidden_dim)
        query = self.class_query.expand(x.size(0), 1, -1)  # (B, 1, hidden_dim)
        out = self.transformer(h, query)  # (B, 1, hidden_dim)
        return self.linear_class(out.squeeze(1))  # (B, num_classes)


FEATURE_SIZE = 225
DEFAULT_THRESHOLD = 0.70


def load_model(model_dir, device):
    """Rebuild SPOTER from its saved settings and load the selected weights."""
    model_dir = Path(model_dir)
    meta, classes, _ = validate_bundle(model_dir)
    if meta["kind"] != "spoter":
        raise ValueError("This model folder does not contain SPOTER")
    ckpt_path = model_dir / "best_model.pt"
    model = SPOTER(
        num_classes=len(classes),
        feature_size=FEATURE_SIZE,
        hidden_dim=meta["hidden_dim"],
        nhead=meta["nhead"],
        num_encoder_layers=meta["num_encoder_layers"],
        num_decoder_layers=meta["num_decoder_layers"],
        dim_feedforward=meta["dim_feedforward"],
        dropout=meta["dropout"],
        max_len=meta["max_len"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded: {len(classes)} classes — {device}", file=sys.stderr)
    return model, classes, meta["max_len"]


def predict(buffer, model, classes, max_len, device, threshold=DEFAULT_THRESHOLD):
    """Prepare raw poses and return the label, softmax score and top five signs.

    Use the same first-frame selection and left padding as training. A score
    below the threshold returns an abstention; it is not a calibrated probability."""
    if not isinstance(max_len, int) or max_len <= 0:
        raise ValueError("max_len must be a positive integer")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if len(buffer) == 0:
        return "NO DETECTION", 0.0, []
    seq = np.asarray(buffer, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != FEATURE_SIZE:
        raise ValueError("Expected raw coordinates with shape (frames, 225)")
    if not np.isfinite(seq).all():
        return "INVALID INPUT", 0.0, []
    # One hand may be absent. Require the body and at least one hand
    # in the same frame, within the part of the sequence used by the model.
    window = seq[:max_len]
    detected = np.any(window[:, :99] != 0, axis=1) & np.any(window[:, 99:] != 0, axis=1)
    if not detected.any():
        return "NO DETECTION", 0.0, []
    seq = normalized(seq)
    if not np.isfinite(seq).all():
        return "INVALID INPUT", 0.0, []

    inp = padded(seq, max_len)[None]

    with torch.no_grad():
        logits = model(torch.tensor(inp).to(device))
        if logits.shape != (1, len(classes)) or not len(classes):
            raise ValueError("Model output does not match the class list")
        if not torch.isfinite(logits).all():
            return "INVALID OUTPUT", 0.0, []
        probs = torch.softmax(logits[0], dim=0).cpu().numpy()

    top5 = [(classes[i], float(probs[i])) for i in probs.argsort()[::-1][:5]]
    conf = top5[0][1]
    label = top5[0][0] if conf >= threshold else "..."
    return label, conf, top5


class RecognitionEngine:
    """Load one model and expose the same prediction entry point to the app."""

    def __init__(self, run, threshold=0.7):
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.run = Path(run)
        self.threshold = threshold
        self.meta, self.classes, protocol = validate_bundle(self.run)
        self.max_len = protocol["config"]["model"]["max_len"]
        if self.meta["kind"] == "spoter":
            import torch

            self.device = torch.device("cpu")
            self.model, classes, _ = load_model(self.run, self.device)
            if classes != self.classes:
                raise ValueError("Class lists do not match")
        else:
            import joblib

            self.model = joblib.load(self.run / "model.joblib")
            if not np.array_equal(
                getattr(self.model, "classes_", None), np.arange(len(self.classes))
            ):
                raise ValueError(
                    "Classifier class order does not match the saved labels"
                )

    def predict(self, sequence):
        x = np.asarray(sequence, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 225 or not len(x) or not np.isfinite(x).all():
            return {"label": None, "message": "Sequence is missing or invalid"}
        window = x[: self.max_len]
        valid = np.any(window[:, :99] != 0, axis=1) & np.any(
            window[:, 99:] != 0, axis=1
        )
        if not valid.any():
            return {
                "label": None,
                "message": "Body and hand landmarks were not detected together",
            }
        start = time.perf_counter()
        if self.meta["kind"] == "spoter":
            label, score, top = predict(
                x, self.model, self.classes, self.max_len, self.device, self.threshold
            )
            accepted = label in self.classes
            result = {
                "label": label if accepted else None,
                "score": score,
                "top": top,
                "message": "" if accepted else "Uncertain result: please try again",
            }
        else:
            idx = int(self.model.predict(features(x, self.max_len)[None])[0])
            result = {
                "label": self.classes[idx],
                "score": None,
                "message": "",
            }
        result["processing_ms"] = (time.perf_counter() - start) * 1000
        return result

    def predict_prepared(self, batch):
        if self.meta["kind"] == "spoter":
            with torch.inference_mode():
                logits = self.model(torch.from_numpy(batch).to(self.device))
                if not torch.isfinite(logits).all():
                    raise ValueError("Model output contains NaN or infinity")
                return logits.argmax(1).cpu().numpy()
        return self.model.predict(batch)

    def probabilities_prepared(self, batch):
        if self.meta["kind"] != "spoter":
            raise ValueError("Probabilities are not available for this classifier")
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(batch).to(self.device))
            if not torch.isfinite(logits).all():
                raise ValueError("Model output contains NaN or infinity")
            return torch.softmax(logits, dim=1).cpu().numpy()
