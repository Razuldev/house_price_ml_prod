from pathlib import Path
import pandas as pd
import logging

from house_ml.config import PROJECT_ROOT, DataConfig, load_config
from house_ml.data.download import ensure_dataset_exists


logger = logging.getLogger(__name__)


def get_csv_path(config: DataConfig) -> Path:
    return (
        PROJECT_ROOT / config.directory / config.extracted_directory / config.csv_name
    )


def load_housing_data(config: DataConfig) -> pd.DataFrame:
    """
    Load the housing dataset into a pandas DataFrame.

    Returns:
        pd.DataFrame: The housing dataset.
    """
    # Ensure the dataset is downloaded and extracted
    ensure_dataset_exists(config)

    csv_path = get_csv_path(config)

    if not csv_path.exists():
        raise FileNotFoundError(f"""{csv_path} does not exist.
                                Please ensure the dataset is downloaded and extracted.""")

    logger.info("Loading dataset from %s", csv_path)

    housing = pd.read_csv(csv_path)

    logger.info("Loaded %d rows and %d columns", housing.shape[0], housing.shape[1])

    return housing


if __name__ == "__main__":
    app_config = load_config()

    housing = load_housing_data(app_config.data)

    print(housing.head())