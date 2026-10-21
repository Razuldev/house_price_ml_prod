from pathlib import Path
import logging
import tarfile
import urllib.request

from house_ml.config import PROJECT_ROOT, DataConfig


logger = logging.getLogger(__name__)


def get_data_paths(config: DataConfig) -> tuple[Path, Path, Path]:

    data_dir = PROJECT_ROOT / config.directory

    tarball_path = data_dir / config.tarball_name

    extracted_dir = data_dir / config.extracted_directory

    return (data_dir, tarball_path, extracted_dir)


def download_dataset(config: DataConfig) -> Path:
    """
    Download and extract the dataset from the given URL.

    Args:
        config: DataConfig

    Returns:
        Path: Path to downloaded tarball.
    """

    data_dir, tarball_path, _ = get_data_paths(config)

    data_dir.parent.mkdir(parents=True, exist_ok=True)

    if tarball_path.exists():
        logger.info("Dataset archive already exists: %s", tarball_path)
        return tarball_path

    logger.info("Downloading dataset from %s", config.url)

    urllib.request.urlretrieve(config.url, tarball_path)

    return tarball_path


def extract_dataset(config: DataConfig, tarball_path: Path) -> Path:
    """
    Extract the dataset from the downloaded tarball.

    Args:
        tarball_path (Path): The path to the downloaded tarball.
        config: DataConfig

    Returns:
        Path: The path to the extracted dataset directory.
    """

    _, _, extracted_dir = get_data_paths(config)

    if extracted_dir.exists():
        logger.info("Dataset already extracted: %s", extracted_dir)
        return extracted_dir

    logger.info("Extracting dataset to %s", extracted_dir)

    with tarfile.open(tarball_path) as tarball:
        tarball.extractall(path=extracted_dir, filter="data")

    return extracted_dir


def ensure_dataset_exists(config: DataConfig) -> Path:
    """
    Ensure that the dataset is downloaded and extracted.
    If not, download and extract it.

    Returns:
        Path: The path to the extracted dataset directory.
    """
    tarball_path = download_dataset(config)
    dataset_dir = extract_dataset(config, tarball_path)

    return dataset_dir
