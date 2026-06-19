# Li Auto Sales Forecaster — Design Spec

**Date:** 2026-06-19
**Author:** Chelsea Hu

---

## Problem

Forecast a newly launched Li Auto model's monthly unit sales for months 4–12 post-launch, using only data available in months 1–3. Produce prediction intervals, not just point estimates. Quantify whether each feature group adds signal beyond the others.

---

## Data Sources

| Source | What it provides | Access method |
|---|---|---|
| Li Auto IR press releases | Monthly delivery counts per model (L6/L7/L8/L9/MEGA), exact launch dates | Scrape public IR site |
| CPCA monthly rankings | Model-level monthly sales for comparable EREV/BEV peers (NIO, Aito, BYD, etc.) | Scrape public CPCA site |
| Autohome (汽车之家) | Spec tables per model; user reviews in Chinese | Scrape public pages |
| CarAPIs (`api.carapis.com/v2/listings`) | Used-car listing prices and inventory volume by model, dealer location | REST API, Bearer token from `AUTO_API_KEY` env var |

**Security:** API key read exclusively from `os.environ["AUTO_API_KEY"]`. Never hardcoded or written to disk. `.env` and `data/raw/` are gitignored.

**Caching:** All fetch responses cached to `data/raw/` as JSON/parquet with a timestamp. Re-fetch only if cache is stale (>7 days) or explicitly invalidated.

---

## Feature Windows & Target

- **Feature window:** months 1–3 since launch (inclusive)
- **Target window:** months 4–12 since launch (9 monthly values per model)
- **No leakage:** feature construction uses only data with `month_since_launch ≤ 3`; target uses only `month_since_launch ∈ [4, 12]`

---

## Feature Groups

### Group 1 — Launch-relative trajectory

Align every model to its own launch month. For each month k ∈ {1, 2, 3}:

```
percentile_rank(k) = rank(new_model_cumulative_sales_at_k) among train models at same k
ratio(k)           = new_model_sales_at_k / median(train_sales_at_k)
```

Produces 6 features. Percentile transform fit on training models only, frozen before applying to test. No leakage.

### Group 2 — Spec quality

Raw fields: 0–100 acceleration (s), WLTP range (km), max power (kW), seat count, powertrain type (EREV=1 / BEV=0), base price (CNY).

Collinearity handling: **PCA fit on training models only**, retaining components that explain ≥95% of variance (expected: 2–3 components). Same PCA applied to test. The binary EREV/BEV flag is kept separate (not entered into PCA).

### Group 3 — Review sentiment

Scrape Autohome/Dongchedi reviews per model, filtered to `month_posted ≤ launch_month + 3`. Run **`hfl/chinese-roberta-wwm-ext`** to score five aspects per review:

- Range anxiety
- Interior quality
- Price-value
- Performance
- After-sales service

Aggregate to model-month mean. For sparse early months: Laplace-smoothed mean (shrink toward train-set prior) to prevent noisy single reviews dominating.

---

## Train / Test Split

Split by launch cohort — never random:

- **Train:** L8 (2023-05), L9 (2022-09), MEGA (2024-03) + CPCA peer models (~20–40 total trajectories)
- **Test:** L6 (2024-05), L7 (2023-08)

CPCA peers included in training only. Percentile transforms and PCA fit on this full training set.

---

## Model

**Primary: Multi-output Quantile Regression Forest**

- Quantiles: q = 0.1, 0.5, 0.9 (80% prediction interval + median)
- One forest per quantile, each predicting all 9 target months jointly
- No feature scaling required
- Implemented via `sklearn`'s `RandomForestRegressor` with quantile-regression wrapper or `lightgbm` quantile objective

**Baseline: Bass diffusion curve**

- Fit 3-parameter Bass curve to months 1–3 per model, project months 4–12
- Used as a sanity check: primary model must beat Bass on pinball loss to justify complexity

---

## Evaluation

| Metric | Description |
|---|---|
| Pinball loss (q=0.1/0.5/0.9) | Primary metric for quantile accuracy |
| MAPE | Interpretable point-estimate error on median prediction |
| Interval coverage | Fraction of actuals falling inside 80% interval (target ≈ 80%) |

### Incremental-value test

For each feature group G, orthogonalize G against the other two groups using residual-on-residual regression, then measure pinball loss drop from adding the orthogonal residual. Three comparisons:

```
Full model vs. Full − trajectory features
Full model vs. Full − spec PCA features
Full model vs. Full − sentiment features
```

Reports whether each group adds signal *beyond* the other two, not just in isolation.

---

## Geographic Visualization

1. **Choropleth (primary):** CPCA province-level sales data → Plotly animated choropleth, one frame per month-since-launch, one map per model
2. **Regional ranking table:** Top 10 provinces by sales per quarter, with rank-change indicators
3. **Fallback:** If CPCA province data unavailable for a model, use CarAPIs dealer location density as geographic demand proxy

---

## Project Structure

```
li-auto-forecaster/
├── .env.example              # AUTO_API_KEY=your_key_here
├── .gitignore                # .env, data/raw/, data/processed/
├── data/
│   ├── raw/                  # cached responses (gitignored)
│   └── processed/            # featurized parquet files (gitignored)
├── src/
│   ├── fetch/
│   │   ├── li_auto_ir.py     # Li Auto IR delivery tables
│   │   ├── cpca.py           # CPCA monthly model rankings
│   │   ├── autohome.py       # specs + Chinese reviews
│   │   └── carapis.py        # used-car listing demand signal
│   ├── features/
│   │   ├── trajectory.py     # launch alignment, percentile transform
│   │   ├── specs.py          # PCA on spec fields
│   │   └── sentiment.py      # Chinese RoBERTa aspect sentiment
│   ├── model/
│   │   ├── split.py          # cohort-based train/test split
│   │   ├── train.py          # quantile regression forest + Bass baseline
│   │   └── evaluate.py       # pinball loss, MAPE, coverage, incremental-value test
│   └── viz/
│       ├── choropleth.py     # province choropleth + regional ranking
│       └── plots.py          # trajectory fan charts, feature importance
├── notebooks/
│   └── forecaster.ipynb      # end-to-end walkthrough + all charts
├── scripts/
│   ├── fetch_all.py          # populate data/raw/
│   ├── build_features.py     # raw → processed
│   └── train_evaluate.py     # train + report metrics
└── requirements.txt
```

---

## Output Artifacts

- `data/processed/features.parquet` — final feature matrix
- `data/processed/predictions.parquet` — median + interval predictions per model-month
- `notebooks/forecaster.ipynb` — fully rendered with all charts
- Console report: pinball loss table, incremental-value test results, interval coverage

---

## Non-negotiables (from requirements)

- No leakage: feature window ends strictly before target window
- Cohort split only: never random
- Percentile transform and PCA fit on train only
- Prediction intervals reported (not point estimates alone)
- Incremental-value test run for all three feature groups
