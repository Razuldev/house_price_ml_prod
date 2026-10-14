import logging
import numpy as np
from sklearn.model_selection import cross_val_score
from house_ml.config import EvaluationConfig

logger = logging.getLogger(__name__)

def run_cross_validation(
    pipeline,
    X,
    y,
    config: EvaluationConfig
) -> dict[str, float]:
    logger.info(
        'Running %d-fold cross-validation',
        config.cv_folds
        )

    scores = cross_val_score(pipeline,
                             X,
                             y,
                             cv=config.cv_folds,
                             scoring=config.scoring)

    rmse_scores = -scores

    results = {
        'cv_rmse_mean': float(np.mean(rmse_scores)),
        'cv_rmse_std': float(np.std(rmse_scores))
    }

    logger.info(
        'CV RMSE result: %.2f std: %.2f',
        results['cv_rmse_mean'],
        results['cv_rmse_std']
    )

    return results