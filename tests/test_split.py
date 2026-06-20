import pandas as pd
import pytest
from src.model.split import cohort_split, TRAIN_MODELS, TEST_MODELS


@pytest.fixture
def feature_df():
    return pd.DataFrame({"model": ["L6", "L7", "L8", "L9", "MEGA"],
                         "percentile_m1": [0.5] * 5})


@pytest.fixture
def target_df():
    return pd.DataFrame({"model": ["L6", "L7", "L8", "L9", "MEGA"],
                         "sales_m4": [5000] * 5, "sales_m5": [6000] * 5})


def test_cohort_split_train_models(feature_df, target_df):
    X_train, X_test, y_train, y_test = cohort_split(feature_df, target_df)
    assert set(X_train["model"]) == set(TRAIN_MODELS)


def test_cohort_split_test_models(feature_df, target_df):
    X_train, X_test, y_train, y_test = cohort_split(feature_df, target_df)
    assert set(X_test["model"]) == set(TEST_MODELS)


def test_no_overlap_between_train_and_test(feature_df, target_df):
    X_train, X_test, y_train, y_test = cohort_split(feature_df, target_df)
    assert set(X_train["model"]).isdisjoint(set(X_test["model"]))


def test_y_shape_matches_x(feature_df, target_df):
    X_train, X_test, y_train, y_test = cohort_split(feature_df, target_df)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
