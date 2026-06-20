import pandas as pd
import numpy as np
import pytest
from src.features.specs import SpecTransformer


@pytest.fixture
def specs_df():
    return pd.DataFrame({
        "model": ["L6", "L7", "L8", "L9", "MEGA"],
        "acceleration_0_100": [4.4, 5.3, 5.9, 6.0, 4.8],
        "range_km": [555, 530, 510, 500, 710],
        "power_kw": [449, 330, 330, 240, 400],
        "seats": [6, 5, 6, 8, 5],
        "is_erev": [1, 1, 1, 1, 0],
        "base_price_cny": [249800, 239800, 339800, 399800, 559800],
    })


def test_transform_output_columns(specs_df):
    train = specs_df[specs_df["model"].isin(["L8", "L9", "MEGA"])]
    st = SpecTransformer(variance_threshold=0.95)
    st.fit(train)
    result = st.transform(specs_df)
    assert "model" in result.columns
    assert "is_erev" in result.columns
    pca_cols = [c for c in result.columns if c.startswith("pc")]
    assert len(pca_cols) >= 1


def test_pca_fit_on_train_only(specs_df):
    """PCA scaler mean must be derived only from training rows."""
    train = specs_df[specs_df["model"].isin(["L8", "L9", "MEGA"])]
    st = SpecTransformer(variance_threshold=0.95)
    st.fit(train)
    train_mean = train["range_km"].mean()
    range_idx = ["acceleration_0_100", "range_km", "power_kw", "seats", "base_price_cny"].index("range_km")
    fitted_mean = st.scaler_.mean_[range_idx]
    assert abs(fitted_mean - train_mean) < 1.0


def test_no_nan_in_output(specs_df):
    train = specs_df[specs_df["model"].isin(["L8", "L9", "MEGA"])]
    st = SpecTransformer(variance_threshold=0.95)
    st.fit(train)
    result = st.transform(specs_df)
    assert not result.isnull().any().any()
