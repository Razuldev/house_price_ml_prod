import numpy as np
import pandas as pd

from house_ml.config import load_config
from house_ml.training.train import build_training_pipeline


def make_small_dataset():
    X = pd.DataFrame(
        {
            "longitude": [-122.0, -121.0, -120.0, -119.0, -118.0, -117.0],
            "latitude": [37.0, 36.0, 35.0, 34.0, 33.0, 32.0],
            "housing_median_age": [10, 20, 30, 40, 50, 60],
            "total_rooms": [100, 200, 300, 400, 500, 600],
            "population": [
                50,
                100,
                150,
                200,
                250,
                300,
            ],
            "total_bedrooms": [
                20,
                40,
                60,
                80,
                100,
                120,
            ],
            "households": [
                10,
                20,
                30,
                40,
                50,
                60,
            ],
            "median_income": [
                2.0,
                3.0,
                4.0,
                5.0,
                3.0,
                2.5,
            ],
            "ocean_proximity": [
                "INLAND",
                "NEAR BAY",
                "NEAR OCEAN",
                "INLAND",
                "NEAR BAY",
                "NEAR BAY",
            ],
        }
    )

    y = pd.Series(
        [
            100000,
            200000,
            300000,
            250000,
            400000,
            200000,
        ]
    )

    return X, y


def test_training_pipeline_smoke():
    config = load_config()

    pipeline = build_training_pipeline(config)

    X, y = make_small_dataset()

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)

    assert np.isfinite(predictions).all()
