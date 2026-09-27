"""FastPath single-encode, parallel-head inference runtime."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .artifact import ModelArtifact
from .contracts import ChoiceValue, NoulValue, ScoreValue
from .encoder import HashingEncoder
from .errors import ArtifactError
from .result import FastPathResult
from .schema import HeadContract, compile_schema


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


class FastPathEngine:
    def __init__(self, model: str | Path, abstain_threshold: float = 0.60):
        if not 0.0 <= abstain_threshold <= 1.0:
            raise ValueError("abstain_threshold must be between 0 and 1")
        self.artifact = ModelArtifact.load(model)
        self.encoder = HashingEncoder(self.artifact.dimensions, self.artifact.seed)
        self.abstain_threshold = abstain_threshold

    def evaluate(self, state: str, schema: type[BaseModel]) -> FastPathResult:
        started = time.perf_counter_ns()
        contract = compile_schema(schema)
        if contract.fingerprint != self.artifact.contract_fingerprint:
            raise ArtifactError(
                "schema does not match model artifact: "
                f"expected {self.artifact.contract_fingerprint}, got {contract.fingerprint}"
            )
        features = self.encoder.encode(state)
        values = {head.name: self._evaluate_head(head, features) for head in contract.heads}
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        return FastPathResult(values, self.artifact.model_id, latency_ms)

    def _evaluate_head(self, contract: HeadContract, features: np.ndarray):
        spec = self.artifact.heads.get(contract.name)
        if spec is None or spec.get("kind") != contract.kind:
            raise ArtifactError(f"missing or incompatible head '{contract.name}'")
        weights = np.asarray(spec["weights"], dtype=np.float32)
        bias = np.asarray(spec.get("bias", [0.0] * len(weights)), dtype=np.float32)
        raw = weights @ features + bias
        temperature = max(float(spec.get("temperature", 1.0)), 1e-4)

        if contract.kind == "choice":
            if tuple(spec.get("options", ())) != contract.options:
                raise ArtifactError(f"options mismatch for head '{contract.name}'")
            probabilities = _softmax(raw / temperature)
            index = int(np.argmax(probabilities))
            confidence = float(probabilities[index])
            return ChoiceValue(
                value=contract.options[index],
                probabilities=dict(zip(contract.options, map(float, probabilities))),
                confidence=confidence,
                abstained=confidence < self.abstain_threshold,
            )
        if contract.kind == "noul":
            probability = _sigmoid(float(raw[0]) / temperature)
            confidence = max(probability, 1.0 - probability)
            return NoulValue(
                value=probability >= 0.5,
                prob=probability,
                confidence=confidence,
                abstained=confidence < self.abstain_threshold,
            )

        minimum, maximum = float(contract.minimum), float(contract.maximum)
        value = minimum + (maximum - minimum) * _sigmoid(float(raw[0]))
        variance = max(float(spec.get("variance", 0.25)), 0.0)
        confidence = float(math.exp(-variance / max((maximum - minimum) ** 2, 1e-9)))
        return ScoreValue(
            value=value,
            confidence=confidence,
            variance=variance,
            abstained=confidence < self.abstain_threshold,
        )
