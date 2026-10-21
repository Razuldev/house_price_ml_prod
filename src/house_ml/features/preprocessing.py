import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from house_ml.data.dataset import split_features_target
from house_ml.data.load import load_housing_data
from house_ml.data.split import stratified_split


def column_ratio(X):
    """
    Calculate ratio between two columns.
    """

    denominator = X[:, [1]]

    return np.divide(
        X[:, [0]],
        denominator,
        out=np.zeros_like(X[:, [0]], dtype=float),
        where=denominator != 0,
    )


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]


def build_ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


def build_log_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(
            np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one"
        ),
        StandardScaler(),
    )


def build_default_numeric_pipeline():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


def build_categorical_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
    )


def build_preprocessor() -> ColumnTransformer:
    """
    Build complete feature preprocessing pipeline
    """

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "bedrooms_ratio",
                build_ratio_pipeline(),
                ["total_bedrooms", "total_rooms"],
            ),
            (
                "rooms_per_household",
                build_ratio_pipeline(),
                ["total_rooms", "households"],
            ),
            (
                "people_per_household",
                build_ratio_pipeline(),
                ["population", "households"],
            ),
            (
                "log_features",
                build_log_pipeline(),
                [
                    "total_bedrooms",
                    "total_rooms",
                    "population",
                    "households",
                    "median_income",
                ],
            ),
            ("categorical", build_categorical_pipeline(), ["ocean_proximity"]),
        ],
        remainder=build_default_numeric_pipeline(),
    )

    return preprocessing


if __name__ == "__main__":
    housing = load_housing_data()

    train_set, _ = stratified_split(housing)

    X_train, _ = split_features_target(train_set)

    preprocessor = build_preprocessor()

    X_transformed = preprocessor.fit_transform(X_train)

    print("Input shape:")
    print(X_train.shape)

    print("\nTransformed shape:")
    print(X_transformed.shape)

    print("\nFirst Transformed row:")
    print(X_transformed[0])
