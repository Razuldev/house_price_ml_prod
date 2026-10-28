from house_ml.config import load_config
from house_ml.inference.predictor import HousingPredictor
from house_ml.inference.schemas import HousingPredictionRequest
from house_ml.logging_config import configure_logging


def main() -> None:
    config = load_config()

    configure_logging(config.logging.level)

    predictor = HousingPredictor.from_config(config)

    request = HousingPredictionRequest(
        longitude=-122.22,
        latitude=40.0,
        housing_median_age=42,
        total_rooms=800,
        total_bedrooms=120,
        population=500,
        households=120,
        median_income=7,
        ocean_proximity="NEAR BAY",
    )

    response = predictor.predict(request)

    print(response.model_dump())


if __name__ == "__main__":
    main()
