import numpy as np

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def calculate_regression_metrics(y_true, y_pred) -> dict[str, float]:

    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def validate_predictions(predictions) -> None:
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Model produced NaN or infinite predictions")
