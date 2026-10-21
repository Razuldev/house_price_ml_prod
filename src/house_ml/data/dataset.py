import pandas as pd

from house_ml.config import load_config, TargetConfig
from house_ml.data.load import load_housing_data
from house_ml.data.split import stratified_split


def split_features_target(
    dataset: pd.DataFrame, config: TargetConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate model features from target
    """

    target_column = config.column

    if target_column not in dataset.columns:
        raise ValueError(f"Target column {target_column} Does not exist!")

    X = dataset.drop(columns=[target_column])
    y = dataset[target_column].copy()

    return X, y


if __name__ == "__main__":
    config = load_config()

    housing = load_housing_data(config.data)

    train_set, test_set = stratified_split(housing, config.split)

    X_train, y_train = split_features_target(train_set, config.target)

    X_test, y_test = split_features_target(test_set, config.target)

    print("\nFeature columns:")
    print(X_train.columns.tolist())

    print("\nTarget column:")
    print(y_train.name)
