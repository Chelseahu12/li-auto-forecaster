import pandas as pd
import pytest
from src.viz.choropleth import build_choropleth_data, regional_ranking_table


@pytest.fixture
def regional_df():
    return pd.DataFrame([
        {"model": "L9", "province": "广东", "month_since_launch": 1, "sales": 2000},
        {"model": "L9", "province": "浙江", "month_since_launch": 1, "sales": 1500},
        {"model": "L9", "province": "北京", "month_since_launch": 1, "sales": 1200},
        {"model": "L9", "province": "广东", "month_since_launch": 2, "sales": 2200},
        {"model": "L9", "province": "浙江", "month_since_launch": 2, "sales": 1600},
    ])


def test_build_choropleth_data_columns(regional_df):
    result = build_choropleth_data(regional_df, model="L9")
    assert set(result.columns) >= {"province", "month_since_launch", "sales"}


def test_regional_ranking_returns_top_n(regional_df):
    result = regional_ranking_table(regional_df, model="L9", top_n=2)
    assert len(result) <= 2
    assert "province" in result.columns
    assert "total_sales" in result.columns
