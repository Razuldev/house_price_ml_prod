from sklearn.pipeline import Pipeline

from house_ml.config import AppConfig
from house_ml.features.preprocessing import build_preprocessor
from house_ml.models.model import build_model


def build_pipeline(config: AppConfig) -> Pipeline:
    """
    Build complete preprocessing + model pipeline
    """
    preprocessor = build_preprocessor()

    model = build_model(config.model)

    pipeline = Pipeline(steps=[("preprocessing", preprocessor), ("model", model)])

    return pipeline
