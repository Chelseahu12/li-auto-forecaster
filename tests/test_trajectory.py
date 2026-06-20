import pandas as pd
import numpy as np
import pytest
from src.features.trajectory import TrajectoryTransformer


@pytest.fixture
def sales_df():
    rows = []
    rng = np.random.default_rng(0)
    for model in ["L8", "L9", "MEGA", "L6", "L7"]:
        for m in range(1, 13):
            rows.append({"model": model, "month_since_launch": m,
                         "sales": int(rng.integers(3000, 15000))})
    return pd.DataFrame(rows)


def test_fit_stores_train_medians(sales_df):
    train_models = ["L8", "L9", "MEGA"]
    tf = TrajectoryTransformer(feature_months=[1, 2, 3])
    tf.fit(sales_df[sales_df["model"].isin(train_models)])
    assert set(tf.train_medians_.keys()) == {1, 2, 3}


def test_transform_returns_correct_columns(sales_df):
    train_models = ["L8", "L9", "MEGA"]
    tf = TrajectoryTransformer(feature_months=[1, 2, 3])
    tf.fit(sales_df[sales_df["model"].isin(train_models)])
    result = tf.transform(sales_df[sales_df["model"].isin(["L6", "L7"])])
    expected_cols = {"model", "percentile_m1", "percentile_m2", "percentile_m3",
                     "ratio_m1", "ratio_m2", "ratio_m3"}
    assert expected_cols.issubset(set(result.columns))


def test_percentile_in_0_1_range(sales_df):
    train_models = ["L8", "L9", "MEGA"]
    tf = TrajectoryTransformer(feature_months=[1, 2, 3])
    tf.fit(sales_df[sales_df["model"].isin(train_models)])
    result = tf.transform(sales_df)
    for m in [1, 2, 3]:
        assert result[f"percentile_m{m}"].between(0, 1).all()


def test_no_leakage_fit_on_train_only(sales_df):
    """Percentile rank must be computed against train set, not full dataset."""
    train_models = ["L8", "L9", "MEGA"]
    tf = TrajectoryTransformer(feature_months=[1, 2, 3])
    train_df = sales_df[sales_df["model"].isin(train_models)]
    tf.fit(train_df)

    for m in [1, 2, 3]:
        expected_median = train_df[train_df["month_since_launch"] == m]["sales"].median()
        assert abs(tf.train_medians_[m] - expected_median) < 1
