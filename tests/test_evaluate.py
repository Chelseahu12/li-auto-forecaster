import numpy as np
import pytest
from src.model.evaluate import (
    pinball_loss,
    mape,
    interval_coverage,
    incremental_value_test,
)


def test_pinball_loss_at_median_symmetric():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([100.0, 200.0])
    assert pinball_loss(y_true, y_pred, q=0.5) == 0.0


def test_pinball_loss_above_penalizes_correctly():
    # pred > true, q=0.9: loss = (1-0.9)*(pred-true) = 0.1*10 = 1.0
    assert abs(pinball_loss(np.array([100.0]), np.array([110.0]), q=0.9) - 1.0) < 1e-6


def test_mape_perfect_prediction():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([100.0, 200.0])
    assert mape(y_true, y_pred) == 0.0


def test_mape_known_value():
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    assert abs(mape(y_true, y_pred) - 10.0) < 1e-6


def test_interval_coverage_all_inside():
    y_true = np.array([100.0, 200.0])
    y_lower = np.array([90.0, 180.0])
    y_upper = np.array([110.0, 220.0])
    assert interval_coverage(y_true, y_lower, y_upper) == 1.0


def test_interval_coverage_none_inside():
    y_true = np.array([100.0, 200.0])
    y_lower = np.array([110.0, 210.0])
    y_upper = np.array([120.0, 230.0])
    assert interval_coverage(y_true, y_lower, y_upper) == 0.0


def test_incremental_value_test_returns_dict():
    rng = np.random.default_rng(7)
    n = 20
    groups = {
        "trajectory": rng.standard_normal((n, 6)),
        "specs": rng.standard_normal((n, 3)),
        "sentiment": rng.standard_normal((n, 5)),
    }
    y = rng.integers(3000, 15000, (n, 9)).astype(float)
    result = incremental_value_test(groups, y)
    assert set(result.keys()) == {"trajectory", "specs", "sentiment"}
    for v in result.values():
        assert "delta_pinball" in v
