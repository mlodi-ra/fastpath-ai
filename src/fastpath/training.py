"""Reference trainer for compact multi-head decision artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from .encoder import HashingEncoder
from .schema import compile_schema


def _temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Select temperature by held-out negative log likelihood."""
    choices = np.geomspace(0.25, 4.0, 49)
    losses = []
    for value in choices:
        scaled = logits / value
        if scaled.shape[1] == 1:
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(scaled[:, 0], -40, 40)))
            losses.append(
                -np.mean(
                    labels * np.log(probabilities + 1e-9)
                    + (1 - labels) * np.log(1 - probabilities + 1e-9)
                )
            )
        else:
            scaled -= scaled.max(axis=1, keepdims=True)
            probabilities = np.exp(scaled)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            losses.append(-np.mean(np.log(probabilities[np.arange(len(labels)), labels] + 1e-9)))
    return float(choices[int(np.argmin(losses))])


def train_artifact(
    records: list[dict[str, Any]],
    schema: type[BaseModel],
    output: str | Path,
    *,
    model_id: str,
    dimensions: int = 1024,
    seed: int = 17,
) -> dict[str, Any]:
    """Train the alpha reference kernel.

    Records must contain ``state`` plus one ground-truth value per schema head.
    A fixed split is used to make artifacts reproducible.
    """
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install fastpath-ai[train] to train models") from exc

    if len(records) < 20:
        raise ValueError("at least 20 records are required")
    contract = compile_schema(schema)
    encoder = HashingEncoder(dimensions, seed)
    matrix = np.stack([encoder.encode(str(record["state"])) for record in records])
    split = max(int(len(records) * 0.8), 1)
    train_x, calibration_x = matrix[:split], matrix[split:]
    heads: dict[str, dict[str, Any]] = {}

    for head in contract.heads:
        values = [record[head.name] for record in records]
        if head.kind == "choice":
            option_index = {option: index for index, option in enumerate(head.options)}
            labels = np.asarray([option_index[value] for value in values], dtype=np.int64)
            model = LogisticRegression(max_iter=1000, C=8.0, random_state=seed)
            model.fit(train_x, labels[:split])
            weights = np.zeros((len(head.options), dimensions), dtype=np.float32)
            bias = np.zeros(len(head.options), dtype=np.float32)
            for row, class_id in enumerate(model.classes_):
                weights[int(class_id)] = model.coef_[row]
                bias[int(class_id)] = model.intercept_[row]
            logits = calibration_x @ weights.T + bias
            temperature = _temperature(logits, labels[split:]) if len(calibration_x) else 1.0
            heads[head.name] = {
                "kind": "choice",
                "options": list(head.options),
                "weights": weights.tolist(),
                "bias": bias.tolist(),
                "temperature": temperature,
            }
        elif head.kind == "noul":
            labels = np.asarray(values, dtype=np.int64)
            model = LogisticRegression(max_iter=1000, C=8.0, random_state=seed)
            model.fit(train_x, labels[:split])
            weights = model.coef_.astype(np.float32)
            bias = model.intercept_.astype(np.float32)
            logits = calibration_x @ weights.T + bias
            temperature = _temperature(logits, labels[split:]) if len(calibration_x) else 1.0
            heads[head.name] = {
                "kind": "noul",
                "weights": weights.tolist(),
                "bias": bias.tolist(),
                "temperature": temperature,
            }
        else:
            labels = np.asarray(values, dtype=np.float32)
            minimum, maximum = float(head.minimum), float(head.maximum)
            clipped = np.clip((labels - minimum) / (maximum - minimum), 1e-4, 1 - 1e-4)
            transformed = np.log(clipped / (1 - clipped))
            model = Ridge(alpha=1.0)
            model.fit(train_x, transformed[:split])
            weights = np.asarray(model.coef_, dtype=np.float32).reshape(1, -1)
            bias = np.asarray([model.intercept_], dtype=np.float32)
            predicted = calibration_x @ weights.T + bias
            variance = (
                float(np.var(transformed[split:] - predicted[:, 0])) if len(calibration_x) else 0.25
            )
            heads[head.name] = {
                "kind": "score",
                "weights": weights.tolist(),
                "bias": bias.tolist(),
                "variance": variance,
            }

    artifact = {
        "format_version": 1,
        "model_id": model_id,
        "contract_fingerprint": contract.fingerprint,
        "dimensions": dimensions,
        "seed": seed,
        "heads": heads,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    return artifact


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
