#!/usr/bin/env python3
"""Parse raw data and build feature + target parquet files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.fetch.li_auto_ir import fetch_and_cache_deliveries, load_deliveries
from src.fetch.cpca import load_cpca
from src.fetch.autohome import load_specs, load_reviews
from src.features.trajectory import TrajectoryTransformer
from src.features.specs import SpecTransformer
from src.features.sentiment import aggregate_sentiment
from src.features.assembly import assemble_features
from src.model.split import TRAIN_MODELS, TARGET_COLS

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Parse Li Auto IR HTML pages ---
ir_pages_dir = Path("data/raw/li_auto_ir/pages")
if ir_pages_dir.exists():
    html_pages = [p.read_text() for p in sorted(ir_pages_dir.glob("*.html"))]
    deliveries = fetch_and_cache_deliveries(html_pages)
else:
    print("Loading cached deliveries...")
    deliveries = load_deliveries()

# --- Merge CPCA peers ---
cpca = load_cpca()
all_sales = pd.concat([deliveries, cpca], ignore_index=True)

# --- Target: pivot to wide format ---
target = (
    all_sales[all_sales["month_since_launch"] >= 4]
    .pivot(index="model", columns="month_since_launch", values="sales")
    .rename(columns=lambda m: f"sales_m{m}")
    .reset_index()
)

# --- Feature Group 1: Trajectory ---
train_sales = all_sales[all_sales["model"].isin(TRAIN_MODELS)]
traj = TrajectoryTransformer(feature_months=[1, 2, 3])
traj.fit(train_sales)
traj_features = traj.transform(all_sales)

# --- Feature Group 2: Specs ---
specs_df = load_specs()
train_specs = specs_df[specs_df["model"].isin(TRAIN_MODELS)]
spec_tf = SpecTransformer(variance_threshold=0.95)
spec_tf.fit(train_specs)
spec_features = spec_tf.transform(specs_df)

# --- Feature Group 3: Sentiment ---
reviews_df = load_reviews()
sentiment_features = aggregate_sentiment(reviews_df, feature_months=[1, 2, 3])

# --- Assemble ---
features = assemble_features(traj_features, spec_features, sentiment_features)

features.to_parquet(OUT_DIR / "features.parquet", index=False)
target.to_parquet(OUT_DIR / "target.parquet", index=False)
print(f"Saved features {features.shape} and target {target.shape} to {OUT_DIR}")
