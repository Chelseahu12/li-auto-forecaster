import pandas as pd
import numpy as np
import pytest

MODELS = ["L6", "L7", "L8", "L9", "MEGA"]

LAUNCH_DATES = {
    "L9": "2022-09",
    "L8": "2023-05",
    "L7": "2023-08",
    "L6": "2024-05",
    "MEGA": "2024-03",
}

TRAIN_MODELS = ["L8", "L9", "MEGA"]
TEST_MODELS = ["L6", "L7"]


@pytest.fixture
def raw_sales_df():
    """Monthly sales per model aligned to launch month."""
    rows = []
    rng = np.random.default_rng(42)
    for model in MODELS:
        for month in range(1, 13):
            rows.append({
                "model": model,
                "month_since_launch": month,
                "sales": int(rng.integers(3000, 15000)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def raw_specs_df():
    """One row per model with raw spec fields."""
    return pd.DataFrame({
        "model": MODELS,
        "acceleration_0_100": [4.4, 5.3, 5.9, 6.0, 4.8],
        "range_km": [555, 530, 510, 500, 710],
        "power_kw": [449, 330, 330, 240, 400],
        "seats": [6, 5, 6, 8, 5],
        "is_erev": [1, 1, 1, 1, 0],
        "base_price_cny": [249800, 239800, 339800, 399800, 559800],
    })


@pytest.fixture
def raw_reviews_df():
    """Reviews with month_since_launch <= 3, text in Chinese."""
    rows = []
    texts = [
        "续航很好，充电也方便，整体非常满意",
        "内饰豪华，驾驶体验出色，价格偏高",
        "加速很快，空间宽敞，性价比高",
        "售后服务一般，但车本身很好",
        "智能驾驶很强，续航略有焦虑",
    ]
    rng = np.random.default_rng(0)
    for model in MODELS:
        for month in range(1, 4):
            for i in range(3):
                rows.append({
                    "model": model,
                    "month_since_launch": month,
                    "review_text": texts[rng.integers(len(texts))],
                })
    return pd.DataFrame(rows)


@pytest.fixture
def feature_df(raw_sales_df, raw_specs_df, raw_reviews_df):
    """Minimal assembled feature DataFrame for model tests."""
    rows = []
    for model in MODELS:
        row = {"model": model}
        for m in range(1, 4):
            row[f"percentile_m{m}"] = 0.5
            row[f"ratio_m{m}"] = 1.0
        row["pc1"] = 0.0
        row["pc2"] = 0.0
        row["is_erev"] = 1
        for aspect in ["range", "interior", "price_value", "performance", "aftersales"]:
            row[f"sent_{aspect}"] = 0.6
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def target_df(raw_sales_df):
    """Wide target: one column per target month (4-12)."""
    wide = (
        raw_sales_df[raw_sales_df["month_since_launch"] >= 4]
        .pivot(index="model", columns="month_since_launch", values="sales")
        .rename(columns=lambda m: f"sales_m{m}")
        .reset_index()
    )
    return wide
