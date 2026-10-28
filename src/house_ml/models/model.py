from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor

from house_ml.config import ModelConfig, load_config


def build_model(config: ModelConfig) -> RegressorMixin:
    """
    Build and return a RandomForestRegressor model based on the configuration.

    Returns:
        RegressorMixin: A configured RandomForestRegressor model.
    """

    if config.type == "random_forest":
        params = config.random_forest

        return RandomForestRegressor(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            random_state=params.random_state,
            n_jobs=params.n_jobs,
        )

    raise ValueError(f"Unsupported model type: {config.type}")


if __name__ == "__main__":
    config = load_config()
    model = build_model(config.model)
    print(model)
