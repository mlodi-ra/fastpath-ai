import json

import numpy as np
import pytest
from pydantic import BaseModel

from fastpath import Choice, FastPathEngine, Noul, Score
from fastpath.errors import ArtifactError
from fastpath.schema import compile_schema


class Contract(BaseModel):
    route: Choice["Infrastructure", "Security"]  # noqa: F821
    severity: Score[1, 5]
    page: Noul


def _artifact(path):
    contract = compile_schema(Contract)
    dimensions = 64
    heads = {
        "route": {
            "kind": "choice",
            "options": ["Infrastructure", "Security"],
            "weights": np.zeros((2, dimensions)).tolist(),
            "bias": [3.0, 0.0],
            "temperature": 1.0,
        },
        "severity": {
            "kind": "score",
            "weights": np.zeros((1, dimensions)).tolist(),
            "bias": [2.0],
            "variance": 0.1,
        },
        "page": {
            "kind": "noul",
            "weights": np.zeros((1, dimensions)).tolist(),
            "bias": [4.0],
            "temperature": 1.0,
        },
    }
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "model_id": "test-v0",
                "contract_fingerprint": contract.fingerprint,
                "dimensions": dimensions,
                "seed": 17,
                "heads": heads,
            }
        )
    )
    return path


def test_output_is_typed_bounded_and_deterministic(tmp_path):
    engine = FastPathEngine(_artifact(tmp_path / "model.json"))
    first = engine.evaluate("database unavailable", Contract)
    second = engine.evaluate("database unavailable", Contract)
    assert first.route.value == "Infrastructure"
    assert set(first.route.probabilities) == {"Infrastructure", "Security"}
    assert 1 <= first.severity.value <= 5
    assert first.page.value is True
    assert first.page.prob == second.page.prob


def test_contract_mismatch_fails_closed(tmp_path):
    class Other(BaseModel):
        route: Choice["A", "B"]  # noqa: F821

    engine = FastPathEngine(_artifact(tmp_path / "model.json"))
    with pytest.raises(ArtifactError, match="schema does not match"):
        engine.evaluate("state", Other)


def test_invalid_state_is_rejected(tmp_path):
    engine = FastPathEngine(_artifact(tmp_path / "model.json"))
    with pytest.raises(ValueError):
        engine.evaluate("", Contract)
