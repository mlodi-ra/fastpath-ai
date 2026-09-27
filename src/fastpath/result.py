from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class FastPathResult(Mapping[str, Any]):
    """Read-only mapping with attribute access for decision heads."""

    def __init__(self, values: dict[str, Any], model_id: str, latency_ms: float):
        self._values = values
        self.model_id = model_id
        self.latency_ms = latency_ms

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def model_dump(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "decision": {key: value.model_dump() for key, value in self._values.items()},
        }
