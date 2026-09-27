"""Compile Pydantic models into immutable FastPath decision contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import BaseModel

from .contracts import ChoiceSpec, ChoiceValue, NoulValue, ScoreSpec, ScoreValue
from .errors import ContractError


@dataclass(frozen=True)
class HeadContract:
    name: str
    kind: Literal["choice", "score", "noul"]
    options: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class DecisionContract:
    name: str
    heads: tuple[HeadContract, ...]
    fingerprint: str


def compile_schema(schema: type[BaseModel]) -> DecisionContract:
    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        raise ContractError("schema must be a Pydantic BaseModel class")
    heads: list[HeadContract] = []
    for name, field in schema.model_fields.items():
        choice = next((m for m in field.metadata if isinstance(m, ChoiceSpec)), None)
        score = next((m for m in field.metadata if isinstance(m, ScoreSpec)), None)
        if choice and field.annotation is ChoiceValue:
            heads.append(HeadContract(name=name, kind="choice", options=choice.options))
        elif score and field.annotation is ScoreValue:
            heads.append(
                HeadContract(
                    name=name,
                    kind="score",
                    minimum=score.minimum,
                    maximum=score.maximum,
                )
            )
        elif field.annotation is NoulValue and "fastpath:noul" in field.metadata:
            heads.append(HeadContract(name=name, kind="noul"))
        else:
            raise ContractError(f"field '{name}' is not Choice[...], Score[min, max], or Noul")
    if not heads:
        raise ContractError("schema must define at least one decision head")
    canonical = json.dumps([asdict(h) for h in heads], sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return DecisionContract(name=schema.__name__, heads=tuple(heads), fingerprint=fingerprint)
