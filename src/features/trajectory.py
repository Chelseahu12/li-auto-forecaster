import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


class TrajectoryTransformer:
    """Compute launch-relative percentile and ratio features.

    Must be fit on training models only, then applied to any model.
    """

    def __init__(self, feature_months: list[int] = None):
        self.feature_months = feature_months or [1, 2, 3]
        self.train_sales_by_month_: dict[int, list[float]] = {}
        self.train_medians_: dict[int, float] = {}

    def fit(self, sales_df: pd.DataFrame) -> "TrajectoryTransformer":
        """Compute training-set sales distribution for each feature month."""
        for m in self.feature_months:
            month_sales = (
                sales_df[sales_df["month_since_launch"] == m]["sales"].tolist()
            )
            self.train_sales_by_month_[m] = month_sales
            self.train_medians_[m] = float(np.median(month_sales))
        return self

    def transform(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Return one row per model with percentile and ratio features."""
        rows = []
        for model, group in sales_df.groupby("model"):
            row = {"model": model}
            for m in self.feature_months:
                model_sales = group[group["month_since_launch"] == m]["sales"]
                if model_sales.empty:
                    row[f"percentile_m{m}"] = np.nan
                    row[f"ratio_m{m}"] = np.nan
                    continue
                val = float(model_sales.iloc[0])
                pct = percentileofscore(self.train_sales_by_month_[m], val) / 100.0
                ratio = val / self.train_medians_[m] if self.train_medians_[m] != 0 else np.nan
                row[f"percentile_m{m}"] = pct
                row[f"ratio_m{m}"] = ratio
            rows.append(row)
        return pd.DataFrame(rows)

    def fit_transform(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(sales_df).transform(sales_df)
