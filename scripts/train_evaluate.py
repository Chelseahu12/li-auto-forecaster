#!/usr/bin/env python3
"""Train model and print evaluation report."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

features = pd.read_parquet("data/processed/features.parquet")
target = pd.read_parquet("data/processed/target.parquet")

from src.model.split import cohort_split, TRAIN_MODELS, TEST_MODELS
from src.model.train import QuantileForestForecaster, FEATURE_COLS
from src.model.bass import BassBaseline
from src.model.evaluate import pinball_loss, mape, interval_coverage, incremental_value_test

X_train, X_test, y_train, y_test = cohort_split(features, target)

print(f"Train models: {TRAIN_MODELS}  ({len(X_train)} rows)")
print(f"Test models:  {TEST_MODELS}  ({len(X_test)} rows)\n")

# --- Quantile forest ---
qf = QuantileForestForecaster(quantiles=[0.1, 0.5, 0.9], n_estimators=200)
qf.fit(X_train[FEATURE_COLS], y_train)
preds = qf.predict(X_test[FEATURE_COLS])  # (n_test, 9, 3)

y_true = y_test.values
y_lower = preds[:, :, 0]
y_median = preds[:, :, 1]
y_upper = preds[:, :, 2]

print("=== Quantile Forest ===")
print(f"Pinball q=0.1:  {pinball_loss(y_true.ravel(), y_lower.ravel(), 0.1):.1f}")
print(f"Pinball q=0.5:  {pinball_loss(y_true.ravel(), y_median.ravel(), 0.5):.1f}")
print(f"Pinball q=0.9:  {pinball_loss(y_true.ravel(), y_upper.ravel(), 0.9):.1f}")
print(f"MAPE (median):  {mape(y_true.ravel(), y_median.ravel()):.1f}%")
print(f"80% coverage:   {interval_coverage(y_true.ravel(), y_lower.ravel(), y_upper.ravel()):.2%}\n")

# --- Incremental-value test ---
print("=== Incremental-Value Test ===")
traj_cols = [c for c in FEATURE_COLS if "percentile" in c or "ratio" in c]
spec_cols = [c for c in FEATURE_COLS if c.startswith("pc") or c == "is_erev"]
sent_cols = [c for c in FEATURE_COLS if c.startswith("sent_")]

groups = {
    "trajectory": X_train[traj_cols].values,
    "specs": X_train[spec_cols].values,
    "sentiment": X_train[sent_cols].values,
}
iv_results = incremental_value_test(groups, y_train.values)
for group, metrics in iv_results.items():
    print(f"{group:12s}  delta_pinball={metrics['delta_pinball']:+.1f}  "
          f"({'adds signal' if metrics['delta_pinball'] < 0 else 'no added signal'})")
