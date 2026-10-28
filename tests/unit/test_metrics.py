import numpy as np
import pytest

from house_ml.evaluation.metrics import (
    calculate_regression_metrics,
    validate_predictions,
)


def test_calculate_regression_metrics():
    y_true = np.array([100.0, 200.0, 300.0])

    y_pred = np.array([100.0, 200.0, 300.0])

    metrics = calculate_regression_metrics(y_true, y_pred)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_validate_predictions_accepts_valid_values():
    predictions = np.array([1.0, 2.0, 3.0])

    validate_predictions(predictions)


def test_validate_predictions_rejects_nan():
    predictions = np.array([1.0, np.nan, 3.0])

    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_predictions(predictions)


def test_validate_predictions_rejects_inf():
    predictions = np.array([1.0, np.inf, 3.0])

    with pytest.raises(
        ValueError,
    ):
        validate_predictions(predictions)
