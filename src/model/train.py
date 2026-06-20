import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURE_COLS = [
    "percentile_m1", "percentile_m2", "percentile_m3",
    "ratio_m1", "ratio_m2", "ratio_m3",
    "pc1", "pc2",
    "is_erev",
    "sent_range", "sent_interior", "sent_price_value",
    "sent_performance", "sent_aftersales",
]

TARGET_COLS = [f"sales_m{m}" for m in range(4, 13)]


class QuantileForestForecaster:
    """Multi-output quantile regression forest.

    Fits one RandomForest per quantile using per-tree quantile extraction.
    predict() returns array of shape (n_samples, n_months, n_quantiles).
    """

    def __init__(
        self,
        quantiles: list[float] = None,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.forest_: RandomForestRegressor | None = None
        self.target_cols_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "QuantileForestForecaster":
        X_arr = X[FEATURE_COLS].values
        self.target_cols_ = [c for c in TARGET_COLS if c in y.columns]
        y_arr = y[self.target_cols_].values.astype(float)

        self.forest_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self.forest_.fit(X_arr, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return shape (n_samples, n_months, n_quantiles)."""
        X_arr = X[FEATURE_COLS].values
        # per-tree predictions: (n_trees, n_samples, n_months)
        tree_preds = np.stack(
            [tree.predict(X_arr) for tree in self.forest_.estimators_], axis=0
        )
        results = []
        for q in self.quantiles:
            q_pred = np.quantile(tree_preds, q, axis=0)  # (n_samples, n_months)
            results.append(q_pred)
        return np.stack(results, axis=-1)  # (n_samples, n_months, n_quantiles)
