import pandas as pd
import pytest
from src.features.assembly import assemble_features


@pytest.fixture
def trajectory_df():
    return pd.DataFrame([
        {"model": m, "percentile_m1": 0.5, "percentile_m2": 0.6, "percentile_m3": 0.7,
         "ratio_m1": 1.0, "ratio_m2": 1.1, "ratio_m3": 1.2}
        for m in ["L6", "L7", "L8", "L9", "MEGA"]
    ])


@pytest.fixture
def spec_df():
    return pd.DataFrame([
        {"model": m, "is_erev": 1, "pc1": 0.1, "pc2": -0.2}
        for m in ["L6", "L7", "L8", "L9", "MEGA"]
    ])


@pytest.fixture
def sentiment_df():
    return pd.DataFrame([
        {"model": m, "sent_range": 0.7, "sent_interior": 0.6,
         "sent_price_value": 0.5, "sent_performance": 0.8, "sent_aftersales": 0.6}
        for m in ["L6", "L7", "L8", "L9", "MEGA"]
    ])


def test_assemble_features_shape(trajectory_df, spec_df, sentiment_df):
    result = assemble_features(trajectory_df, spec_df, sentiment_df)
    assert len(result) == 5
    assert "model" in result.columns


def test_assemble_features_no_nulls(trajectory_df, spec_df, sentiment_df):
    result = assemble_features(trajectory_df, spec_df, sentiment_df)
    assert not result.isnull().any().any()


def test_assemble_features_contains_all_groups(trajectory_df, spec_df, sentiment_df):
    result = assemble_features(trajectory_df, spec_df, sentiment_df)
    assert "percentile_m1" in result.columns
    assert "pc1" in result.columns
    assert "sent_range" in result.columns
