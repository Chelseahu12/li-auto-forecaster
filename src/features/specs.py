import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

NUMERIC_SPEC_COLS = [
    "acceleration_0_100",
    "range_km",
    "power_kw",
    "seats",
    "base_price_cny",
]


class SpecTransformer:
    """PCA on numeric spec fields. is_erev kept as-is (binary, not scaled).

    Fit on training models only. Apply to all models.
    """

    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.scaler_ = StandardScaler()
        self.pca_: PCA | None = None
        self.n_components_: int = 0

    def fit(self, specs_df: pd.DataFrame) -> "SpecTransformer":
        X = specs_df[NUMERIC_SPEC_COLS].values
        X_scaled = self.scaler_.fit_transform(X)

        pca_full = PCA().fit(X_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        self.n_components_ = int(np.searchsorted(cumvar, self.variance_threshold)) + 1
        self.pca_ = PCA(n_components=self.n_components_).fit(X_scaled)
        return self

    def transform(self, specs_df: pd.DataFrame) -> pd.DataFrame:
        X = specs_df[NUMERIC_SPEC_COLS].values
        X_scaled = self.scaler_.transform(X)
        components = self.pca_.transform(X_scaled)

        result = specs_df[["model", "is_erev"]].copy().reset_index(drop=True)
        for i in range(self.n_components_):
            result[f"pc{i + 1}"] = components[:, i]
        return result
