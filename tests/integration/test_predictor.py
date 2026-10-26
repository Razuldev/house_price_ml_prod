import joblib
import pandas as pd

from house_ml.config import (
    load_config,
)
from house_ml.inference.predictor import (
    HousingPredictor,
)
from house_ml.inference.schemas import (
    HousingPredictionRequest,
)
from house_ml.models.pipeline import (
    build_pipeline,
)


def test_predictor_end_to_end(
    tmp_path,
):
    config = load_config()

    X = pd.DataFrame(
        {
            "longitude": [
                -122.0,
                -121.0,
                -120.0,
                -119.0,
            ],
            "latitude": [
                37.0,
                36.0,
                35.0,
                34.0,
            ],
            "housing_median_age": [
                20,
                30,
                40,
                25,
            ],
            "total_rooms": [
                1000,
                2000,
                1500,
                1200,
            ],
            "total_bedrooms": [
                200,
                400,
                300,
                250,
            ],
            "population": [
                500,
                1000,
                750,
                600,
            ],
            "households": [
                200,
                400,
                300,
                250,
            ],
            "median_income": [
                3.0,
                4.0,
                5.0,
                6.0,
            ],
            "ocean_proximity": [
                "INLAND",
                "NEAR BAY",
                "<1H OCEAN",
                "NEAR OCEAN",
            ],
        }
    )

    y = pd.Series(
        [
            100000,
            150000,
            200000,
            250000,
        ]
    )

    pipeline = build_pipeline(config)

    pipeline.fit(X, y)

    model_path = tmp_path / "model.joblib"

    joblib.dump(pipeline, model_path)

    predictor = HousingPredictor(model_path=model_path)

    request = HousingPredictionRequest(
        longitude=-122.0,
        latitude=37.0,
        housing_median_age=20,
        total_rooms=800,
        total_bedrooms=200,
        population=300,
        households=200,
        median_income=4.0,
        ocean_proximity="INLAND",
    )

    response = predictor.predict(request)

    assert response.predicted_median_house_value > 0
