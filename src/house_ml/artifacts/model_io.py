import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from house_ml.config import PROJECT_ROOT, ArtifactConfig

logger = logging.getLogger(__name__)


def get_git_commit() -> str | None:

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout

    except Exception:
        return None


def save_model(model, config: ArtifactConfig) -> Path:

    artifact_dir = PROJECT_ROOT / config.directory

    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / config.model_filename

    logger.info("Saving model to %s", model_path)

    joblib.dump(model, model_path)

    return model_path


def save_metadata(metrics: dict[str, float], config: ArtifactConfig) -> Path:

    artifact_dir = PROJECT_ROOT / config.directory

    artifact_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = artifact_dir / config.metadata_filename

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "metrics": metrics,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    logger.info("Saved metadata to %s", metadata_path)

    return metadata_path


def load_model(model_path: Path):
    logger.info("Loading model from %s", model_path)

    return joblib.load(model_path)
