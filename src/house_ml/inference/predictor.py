import logging
from pathlib import Path

import numpy as np
import pandas as pd

from house_ml.artifacts.model_io import load_model
from house_ml.config import PROJECT_ROOT, AppConfig
from house_ml.inference.schemas import (
    HousingPredictionRequest,
    HousingPredictionResponse,
)

logger = logging.getLogger(__name__)


class HousingPredictor:
    def __init__(self, model_path: Path) -> None:

        logger.info("Initializing predictor")

        self.model = load_model(model_path)

        logger.info("Predictor initialized")

    @classmethod
    def from_config(cls, config: AppConfig):

        model_path = (
            PROJECT_ROOT / config.artifacts.directory / config.artifacts.model_filename
        )

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        return cls(model_path=model_path)

    def predict(
        self,
        request: HousingPredictionRequest,
    ) -> HousingPredictionResponse:

        input_df = pd.DataFrame([request.model_dump()])

        predictions = self.model.predict(input_df)

        if len(predictions) != 1:
            raise RuntimeError("Expected exactly one prediction")

        prediction = float(predictions[0])

        if not np.isfinite(prediction):
            raise RuntimeError("Prediction is not finite")

        return HousingPredictionResponse(predicted_median_house_value=prediction)
