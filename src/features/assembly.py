import pandas as pd


def assemble_features(
    trajectory_df: pd.DataFrame,
    spec_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the three feature groups on model into a single feature matrix."""
    df = trajectory_df.merge(spec_df, on="model", how="inner")
    df = df.merge(sentiment_df, on="model", how="inner")
    return df.reset_index(drop=True)
