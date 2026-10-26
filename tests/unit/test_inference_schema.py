import pytest
from pydantic import ValidationError

from house_ml.inference.schemas import (
    HousingPredictionRequest,
)


def valid_request_data():
    return {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41,
        "total_rooms": 880,
        "total_bedrooms": 129,
        "population": 322,
        "households": 126,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
    }


def test_valid_prediction_request():
    request = HousingPredictionRequest(**valid_request_data())

    assert request.households == 126

    assert request.ocean_proximity == "NEAR BAY"


def test_negative_households_rejected():
    data = valid_request_data()

    data["households"] = -1

    with pytest.raises(ValidationError):
        HousingPredictionRequest(**data)


def test_invalid_ocean_proximity_rejected():
    data = valid_request_data()

    data["ocean_proximity"] = "MIDDLE OF ATLANTIC"

    with pytest.raises(ValidationError):
        HousingPredictionRequest(**data)


def test_invalid_latitude_rejected():
    data = valid_request_data()

    data["latitude"] = 300

    with pytest.raises(ValidationError):
        HousingPredictionRequest(**data)
