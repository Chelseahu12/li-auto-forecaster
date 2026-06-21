import pandas as pd
import pytest
from unittest.mock import patch
from src.features.sentiment import score_reviews, aggregate_sentiment, ASPECTS


@pytest.fixture
def reviews_df():
    return pd.DataFrame([
        {"model": "L6", "month_since_launch": 1, "review_text": "续航很好，充电方便"},
        {"model": "L6", "month_since_launch": 1, "review_text": "价格偏高但值得"},
        {"model": "L6", "month_since_launch": 2, "review_text": "内饰豪华，驾驶感出色"},
        {"model": "L9", "month_since_launch": 1, "review_text": "售后服务不太好"},
    ])


def test_aggregate_sentiment_returns_model_scores(reviews_df):
    def fake_score(texts):
        return [{a: 0.7 for a in ASPECTS} for _ in texts]

    with patch("src.features.sentiment.score_reviews", side_effect=fake_score):
        result = aggregate_sentiment(reviews_df, feature_months=[1, 2, 3])

    assert isinstance(result, pd.DataFrame)
    assert "model" in result.columns
    for aspect in ASPECTS:
        assert f"sent_{aspect}" in result.columns


def test_aggregate_sentiment_one_row_per_model(reviews_df):
    def fake_score(texts):
        return [{a: 0.6 for a in ASPECTS} for _ in texts]

    with patch("src.features.sentiment.score_reviews", side_effect=fake_score):
        result = aggregate_sentiment(reviews_df, feature_months=[1, 2, 3])

    assert len(result) == reviews_df["model"].nunique()


def test_aggregate_sentiment_scores_in_0_1(reviews_df):
    def fake_score(texts):
        return [{a: 0.8 for a in ASPECTS} for _ in texts]

    with patch("src.features.sentiment.score_reviews", side_effect=fake_score):
        result = aggregate_sentiment(reviews_df, feature_months=[1, 2, 3])

    for aspect in ASPECTS:
        col = f"sent_{aspect}"
        assert result[col].between(0, 1).all()
