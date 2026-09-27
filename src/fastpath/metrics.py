from __future__ import annotations

from itertools import pairwise

import numpy as np


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared probability error. Lower is better."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    if probabilities.ndim == 1:
        return float(np.mean((probabilities - labels) ** 2))
    one_hot = np.eye(probabilities.shape[1])[labels.astype(int)]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, bins: int = 10
) -> float:
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in pairwise(edges):
        selected = (confidences > lower) & (confidences <= upper)
        if np.any(selected):
            error += selected.mean() * abs(correct[selected].mean() - confidences[selected].mean())
    return float(error)
