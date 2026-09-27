"""Typed decision contracts backed by Pydantic Annotated metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class ChoiceSpec:
    options: tuple[str, ...]


@dataclass(frozen=True)
class ScoreSpec:
    minimum: float
    maximum: float


class ChoiceValue(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool = False


class ScoreValue(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: float
    confidence: float = Field(ge=0.0, le=1.0)
    variance: float = Field(ge=0.0)
    abstained: bool = False


class NoulValue(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: bool
    prob: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool = False


class Choice:
    """Annotation factory: ``Choice["A", "B"]``."""

    def __class_getitem__(cls, options: Any) -> Any:
        if not isinstance(options, tuple):
            options = (options,)
        if len(options) < 2 or len(options) > 255:
            raise TypeError("Choice requires between 2 and 255 options")
        if any(not isinstance(option, str) or not option for option in options):
            raise TypeError("Choice options must be non-empty strings")
        if len(set(options)) != len(options):
            raise TypeError("Choice options must be unique")
        return Annotated[ChoiceValue, ChoiceSpec(tuple(options))]


class Score:
    """Annotation factory: ``Score[1, 5]``."""

    def __class_getitem__(cls, bounds: Any) -> Any:
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise TypeError("Score requires minimum and maximum bounds")
        minimum, maximum = float(bounds[0]), float(bounds[1])
        if minimum >= maximum:
            raise TypeError("Score minimum must be less than maximum")
        return Annotated[ScoreValue, ScoreSpec(minimum, maximum)]


Noul = Annotated[NoulValue, "fastpath:noul"]
