import pandas as pd

TRAIN_MODELS = ["L8", "L9", "MEGA"]
TEST_MODELS = ["L6", "L7"]

TARGET_COLS = [f"sales_m{m}" for m in range(4, 13)]


def cohort_split(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by launch cohort. Never random.

    Returns (X_train, X_test, y_train, y_test).
    """
    X_train = feature_df[feature_df["model"].isin(TRAIN_MODELS)].reset_index(drop=True)
    X_test = feature_df[feature_df["model"].isin(TEST_MODELS)].reset_index(drop=True)

    available_target_cols = [c for c in TARGET_COLS if c in target_df.columns]
    y_train = target_df[target_df["model"].isin(TRAIN_MODELS)][available_target_cols].reset_index(drop=True)
    y_test = target_df[target_df["model"].isin(TEST_MODELS)][available_target_cols].reset_index(drop=True)

    return X_train, X_test, y_train, y_test
