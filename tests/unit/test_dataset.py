import pandas as pd
import pytest

from house_ml.config import TargetConfig
from house_ml.data.dataset import split_features_target

def test_split_features_target():
    dataset = pd.DataFrame(
        {
        "feature_a": [1, 2, 3],
        "feature_b": [4, 5, 6],
        "target": [10, 20, 30]
        }
    )

    config = TargetConfig(column="target")

    X,y = split_features_target(
        dataset, config
    )


    assert list(X.columns) == ["feature_a", "feature_b"]

    assert list(y) == [10, 20, 30]

    assert 'target' not in X.columns

def test_split_features_target_missing_target():
    dataset = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
        }
    )

    config = TargetConfig(column="target")

    with pytest.raises(
        ValueError,
        match="Target column"
    ):
        split_features_target(dataset, config)