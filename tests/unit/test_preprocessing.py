import numpy as np

from house_ml.features.preprocessing import column_ratio

def test_column_ratio():
    X = np.array([
        [10.0,2.0],
        [9.0,3.0]
    ])
    result = column_ratio(X)

    expected = np.array([
        [5.0],
        [3.0]
    ])

    np.testing.assert_allclose(result, expected)

def test_column_ratio_handle_zero_denominator():
    X=np.array([
        [10.0, 0],
    ])

    result = column_ratio(X)

    assert np.isfinite(result).all()

    assert result[0,0] == 0.0