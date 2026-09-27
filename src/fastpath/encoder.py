"""Small deterministic encoder used by the alpha reference kernel."""

from __future__ import annotations

import hashlib
import re
from itertools import pairwise

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9_.:/-]+")


class HashingEncoder:
    """Signed feature hashing with L2 normalization.

    This is intentionally simple and auditable. Production backbones can implement
    the same ``encode`` interface using ONNX transformer embeddings.
    """

    def __init__(self, dimensions: int = 1024, seed: int = 17):
        if dimensions < 64:
            raise ValueError("dimensions must be at least 64")
        self.dimensions = dimensions
        self.seed = seed

    def encode(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("state must be a non-empty string")
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = [token.lower() for token in TOKEN_RE.findall(text)]
        features = tokens + [f"{a}::{b}" for a, b in pairwise(tokens)]
        for feature in features:
            digest = hashlib.blake2b(
                feature.encode(), digest_size=8, person=self.seed.to_bytes(8, "little")
            ).digest()
            raw = int.from_bytes(digest, "little")
            index = raw % self.dimensions
            vector[index] += 1.0 if (raw >> 63) == 0 else -1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector
