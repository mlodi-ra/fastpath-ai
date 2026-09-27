"""Portable, inspectable FastPath model artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ArtifactError


@dataclass(frozen=True)
class ModelArtifact:
    format_version: int
    model_id: str
    contract_fingerprint: str
    dimensions: int
    seed: int
    heads: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, path: str | Path) -> ModelArtifact:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            artifact = cls(**data)
        except (OSError, ValueError, TypeError) as exc:
            raise ArtifactError(f"cannot load artifact: {exc}") from exc
        artifact.validate()
        return artifact

    def validate(self) -> None:
        if self.format_version != 1:
            raise ArtifactError(f"unsupported artifact format {self.format_version}")
        if self.dimensions < 64 or not self.heads:
            raise ArtifactError("invalid artifact dimensions or empty heads")
        for name, head in self.heads.items():
            weights = np.asarray(head.get("weights"), dtype=np.float32)
            if weights.ndim != 2 or weights.shape[1] != self.dimensions:
                raise ArtifactError(f"head '{name}' has invalid weight shape")
