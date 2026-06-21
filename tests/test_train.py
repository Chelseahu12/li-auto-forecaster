import numpy as np
import pandas as pd
import pytest
from src.model.train import QuantileForestForecaster, FEATURE_COLS, TARGET_COLS


@pytest.fixture
def small_train_data():
    rng = np.random.default_rng(1)
    n = 30
    X = pd.DataFrame({c: rng.standard_normal(n) for c in FEATURE_COLS})
    y = pd.DataFrame({c: rng.integers(3000, 15000, n) for c in TARGET_COLS})
    return X, y


def test_fit_and_predict_shape(small_train_data):
    X, y = small_train_data
    model = QuantileForestForecaster(quantiles=[0.1, 0.5, 0.9], n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X.iloc[:3])
    assert preds.shape == (3, len(TARGET_COLS), 3)


def test_quantile_ordering(small_train_data):
    """q=0.1 predictions must be <= q=0.5 <= q=0.9."""
    X, y = small_train_data
    model = QuantileForestForecaster(quantiles=[0.1, 0.5, 0.9], n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds[:, :, 0] <= preds[:, :, 1])
    assert np.all(preds[:, :, 1] <= preds[:, :, 2])


def test_feature_cols_match_expected():
    assert "percentile_m1" in FEATURE_COLS
    assert "pc1" in FEATURE_COLS
    assert "sent_range" in FEATURE_COLS
