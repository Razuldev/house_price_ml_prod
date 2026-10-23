import pandas as pd
from sklearn.model_selection import train_test_split

from house_ml.config import SplitConfig, load_config
from house_ml.data.load import load_housing_data


def create_income_category(housing: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """
    Add income category used for stratified sampling
    """

    housing = housing.copy()

    bins = config.income_category.bins
    labels = config.income_category.labels

    housing["income_cat"] = pd.cut(housing["median_income"], bins=bins, labels=labels)

    return housing


def stratified_split(
    housing: pd.DataFrame, config: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:

    housing = create_income_category(housing, config)

    train_set, test_set = train_test_split(
        housing,
        test_size=config.test_size,
        stratify=housing["income_cat"],
        random_state=config.random_state,
    )

    train_set = train_set.drop(columns=["income_cat"])
    test_set = test_set.drop(columns=["income_cat"])

    return train_set, test_set


if __name__ == "__main__":
    config = load_config()
    housing = load_housing_data(config.data)
    train_set, test_set = stratified_split(housing, config.split)

    print(f"Full dataset: {housing.shape}")
    print(f"Train dataset: {train_set.shape}")
    print(f"Test dataset: {test_set.shape}")
