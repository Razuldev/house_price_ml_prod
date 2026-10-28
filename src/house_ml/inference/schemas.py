from typing import Literal

from pydantic import BaseModel, Field

OceanProximity = Literal["1<H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


class HousingPredictionRequest(BaseModel):
    longitude: float = Field(ge=-180, le=180)

    latitude: float = Field(ge=-90, le=90)

    housing_median_age: float = Field(
        ge=0,
    )

    total_rooms: float = Field(
        gt=0,
    )

    total_bedrooms: float = Field(ge=0)

    population: float = Field(ge=0)

    households: float = Field(gt=0)

    median_income: float = Field(ge=0)

    ocean_proximity: OceanProximity


class HousingPredictionResponse(BaseModel):
    predicted_median_house_value: float
