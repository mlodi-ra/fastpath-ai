import numpy as np

from fastpath.metrics import brier_score, expected_calibration_error


def test_binary_brier_score():
    assert brier_score(np.array([0.0, 1.0]), np.array([0, 1])) == 0.0


def test_calibration_error_is_zero_for_matched_bin():
    confidences = np.full(10, 0.8)
    correct = np.array([1] * 8 + [0] * 2)
    assert expected_calibration_error(confidences, correct) < 1e-12
