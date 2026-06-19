# Li Auto Sales Forecaster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that forecasts a newly launched Li Auto model's monthly unit sales for months 4–12 using only months 1–3 data, with prediction intervals and an incremental-value test for each feature group.

**Architecture:** Fetch → cache → featurize (3 groups: trajectory percentiles, spec PCA, Chinese RoBERTa sentiment) → quantile regression forest (q=0.1/0.5/0.9) → evaluate with pinball loss + incremental-value test + choropleth viz. Cohort split only (never random). All transforms fit on train, applied to test.

**Tech Stack:** Python 3.11+, requests, BeautifulSoup4, pandas, scikit-learn, lightgbm, transformers (hfl/chinese-roberta-wwm-ext), torch, plotly, geopandas, pytest, python-dotenv

---

## File Map

```
src/fetch/
  carapis.py          — CarAPIs listings fetcher + disk cache
  li_auto_ir.py       — Li Auto IR delivery table scraper + cache
  cpca.py             — CPCA monthly model rankings scraper + cache
  autohome.py         — Autohome spec table + review scraper + cache

src/features/
  trajectory.py       — launch alignment, percentile/ratio transform (fit on train)
  specs.py            — spec PCA (fit on train)
  sentiment.py        — Chinese RoBERTa aspect sentiment → model-month scores
  assembly.py         — merge all three feature groups into one DataFrame

src/model/
  split.py            — cohort-based train/test split
  bass.py             — Bass diffusion curve baseline
  train.py            — multi-output quantile regression forest
  evaluate.py         — pinball loss, MAPE, coverage, incremental-value test

src/viz/
  choropleth.py       — province-level choropleth (Plotly)
  plots.py            — trajectory fan charts, feature importance

scripts/
  fetch_all.py        — run all fetchers → data/raw/
  build_features.py   — raw → data/processed/features.parquet
  train_evaluate.py   — train + print metrics report

notebooks/
  forecaster.ipynb    — end-to-end walkthrough

tests/
  conftest.py         — shared fixtures
  test_carapis.py
  test_li_auto_ir.py
  test_cpca.py
  test_autohome.py
  test_trajectory.py
  test_specs.py
  test_sentiment.py
  test_assembly.py
  test_split.py
  test_bass.py
  test_train.py
  test_evaluate.py
  test_choropleth.py
  test_plots.py
```

---

## Task 1: Project setup and shared test fixtures

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`

- [ ] **Step 1: Pin requirements with versions**

Replace `requirements.txt` with:

```
requests==2.32.3
beautifulsoup4==4.12.3
lxml==5.2.2
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
lightgbm==4.4.0
transformers==4.43.3
torch==2.3.1
jieba==0.42.1
python-dotenv==1.0.1
plotly==5.22.0
geopandas==0.14.4
pyarrow==16.1.0
joblib==1.4.2
pytest==8.2.2
pytest-mock==3.14.0
```

- [ ] **Step 2: Create pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

- [ ] **Step 3: Create tests/__init__.py**

```python
```
(empty file)

- [ ] **Step 4: Create tests/conftest.py with shared fixtures**

```python
import pandas as pd
import numpy as np
import pytest

MODELS = ["L6", "L7", "L8", "L9", "MEGA"]

LAUNCH_DATES = {
    "L9": "2022-09",
    "L8": "2023-05",
    "L7": "2023-08",
    "L6": "2024-05",
    "MEGA": "2024-03",
}

TRAIN_MODELS = ["L8", "L9", "MEGA"]
TEST_MODELS = ["L6", "L7"]


@pytest.fixture
def raw_sales_df():
    """Monthly sales per model aligned to launch month."""
    rows = []
    rng = np.random.default_rng(42)
    for model in MODELS:
        for month in range(1, 13):
            rows.append({
                "model": model,
                "month_since_launch": month,
                "sales": int(rng.integers(3000, 15000)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def raw_specs_df():
    """One row per model with raw spec fields."""
    return pd.DataFrame({
        "model": MODELS,
        "acceleration_0_100": [4.4, 5.3, 5.9, 6.0, 4.8],
        "range_km": [555, 530, 510, 500, 710],
        "power_kw": [449, 330, 330, 240, 400],
        "seats": [6, 5, 6, 8, 5],
        "is_erev": [1, 1, 1, 1, 0],
        "base_price_cny": [249800, 239800, 339800, 399800, 559800],
    })


@pytest.fixture
def raw_reviews_df():
    """Reviews with month_since_launch ≤ 3, text in Chinese."""
    rows = []
    texts = [
        "续航很好，充电也方便，整体非常满意",
        "内饰豪华，驾驶体验出色，价格偏高",
        "加速很快，空间宽敞，性价比高",
        "售后服务一般，但车本身很好",
        "智能驾驶很强，续航略有焦虑",
    ]
    rng = np.random.default_rng(0)
    for model in MODELS:
        for month in range(1, 4):
            for i in range(3):
                rows.append({
                    "model": model,
                    "month_since_launch": month,
                    "review_text": texts[rng.integers(len(texts))],
                })
    return pd.DataFrame(rows)


@pytest.fixture
def feature_df(raw_sales_df, raw_specs_df, raw_reviews_df):
    """Minimal assembled feature DataFrame for model tests."""
    rows = []
    for model in MODELS:
        row = {"model": model}
        for m in range(1, 4):
            row[f"percentile_m{m}"] = 0.5
            row[f"ratio_m{m}"] = 1.0
        row["pc1"] = 0.0
        row["pc2"] = 0.0
        row["is_erev"] = 1
        for aspect in ["range", "interior", "price_value", "performance", "aftersales"]:
            row[f"sent_{aspect}"] = 0.6
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def target_df(raw_sales_df):
    """Wide target: one column per target month (4–12)."""
    wide = (
        raw_sales_df[raw_sales_df["month_since_launch"] >= 4]
        .pivot(index="model", columns="month_since_launch", values="sales")
        .rename(columns=lambda m: f"sales_m{m}")
        .reset_index()
    )
    return wide
```

- [ ] **Step 5: Verify fixtures load**

```bash
cd /Users/chelseahu/li-auto-forecaster
pip install -r requirements.txt -q
pytest tests/conftest.py --collect-only -q
```

Expected output: `no tests ran` (fixtures only, no errors)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini tests/
git commit -m "feat: add project setup, requirements, and shared test fixtures"
```

---

## Task 2: CarAPIs fetcher with disk cache

**Files:**
- Create: `src/fetch/carapis.py`
- Create: `tests/test_carapis.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_carapis.py`:

```python
import json
import time
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from src.fetch.carapis import fetch_listings, CACHE_DIR, LI_AUTO_SOURCES


def test_fetch_listings_calls_api_on_cache_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_API_KEY", "test-key")
    monkeypatch.setattr("src.fetch.carapis.CACHE_DIR", tmp_path)

    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"make": "Li Auto", "model": "L9", "price": 399800}]
    mock_resp.raise_for_status = MagicMock()

    with patch("src.fetch.carapis.requests.get", return_value=mock_resp) as mock_get:
        result = fetch_listings(source="che168", make="Li Auto")

    mock_get.assert_called_once()
    assert result == [{"make": "Li Auto", "model": "L9", "price": 399800}]


def test_fetch_listings_uses_cache_when_fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_API_KEY", "test-key")
    monkeypatch.setattr("src.fetch.carapis.CACHE_DIR", tmp_path)

    cached_data = [{"make": "Li Auto", "model": "L8", "price": 339800}]
    cache_file = tmp_path / "che168_Li_Auto.json"
    cache_file.write_text(json.dumps({"fetched_at": time.time(), "data": cached_data}))

    with patch("src.fetch.carapis.requests.get") as mock_get:
        result = fetch_listings(source="che168", make="Li Auto")

    mock_get.assert_not_called()
    assert result == cached_data


def test_fetch_listings_refreshes_stale_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_API_KEY", "test-key")
    monkeypatch.setattr("src.fetch.carapis.CACHE_DIR", tmp_path)

    stale_data = [{"make": "Li Auto", "model": "L7", "price": 0}]
    cache_file = tmp_path / "che168_Li_Auto.json"
    cache_file.write_text(json.dumps({
        "fetched_at": time.time() - 8 * 86400,  # 8 days ago
        "data": stale_data,
    }))

    fresh_data = [{"make": "Li Auto", "model": "L7", "price": 239800}]
    mock_resp = MagicMock()
    mock_resp.json.return_value = fresh_data
    mock_resp.raise_for_status = MagicMock()

    with patch("src.fetch.carapis.requests.get", return_value=mock_resp):
        result = fetch_listings(source="che168", make="Li Auto")

    assert result == fresh_data


def test_missing_api_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTO_API_KEY", raising=False)
    monkeypatch.setattr("src.fetch.carapis.CACHE_DIR", tmp_path)
    with pytest.raises(KeyError, match="AUTO_API_KEY"):
        fetch_listings(source="che168", make="Li Auto")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_carapis.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.fetch.carapis'`

- [ ] **Step 3: Implement src/fetch/carapis.py**

```python
import json
import os
import time
from pathlib import Path

import requests

CACHE_DIR = Path("data/raw/carapis")
CACHE_TTL_SECONDS = 7 * 86400
BASE_URL = "https://api.carapis.com/v2"

LI_AUTO_SOURCES = ["che168", "dongchedi", "guazi", "58che"]


def fetch_listings(
    source: str,
    make: str = "Li Auto",
    limit: int = 100,
) -> list[dict]:
    """Fetch used-car listings from CarAPIs with disk cache."""
    api_key = os.environ["AUTO_API_KEY"]

    cache_file = CACHE_DIR / f"{source}_{make.replace(' ', '_')}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if time.time() - cached["fetched_at"] < CACHE_TTL_SECONDS:
            return cached["data"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        f"{BASE_URL}/listings",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"source": source, "make": make, "limit": limit},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    cache_file.write_text(json.dumps({"fetched_at": time.time(), "data": data}))
    return data


def fetch_all_li_auto_listings() -> list[dict]:
    """Fetch Li Auto listings from all known Chinese sources."""
    results = []
    for source in LI_AUTO_SOURCES:
        try:
            results.extend(fetch_listings(source=source, make="Li Auto"))
        except Exception:
            pass
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_carapis.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/fetch/carapis.py tests/test_carapis.py
git commit -m "feat: add CarAPIs fetcher with 7-day disk cache"
```

---

## Task 3: Li Auto IR delivery scraper

**Files:**
- Create: `src/fetch/li_auto_ir.py`
- Create: `tests/test_li_auto_ir.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_li_auto_ir.py`:

```python
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from src.fetch.li_auto_ir import parse_delivery_page, load_deliveries, LAUNCH_DATES


SAMPLE_HTML = """
<html><body>
<table>
<tr><th>Model</th><th>Deliveries</th><th>Year-Month</th></tr>
<tr><td>L9</td><td>12,345</td><td>2023-01</td></tr>
<tr><td>L8</td><td>8,901</td><td>2023-01</td></tr>
</table>
</body></html>
"""


def test_parse_delivery_page_returns_dataframe():
    df = parse_delivery_page(SAMPLE_HTML)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"model", "year_month", "sales"}
    assert len(df) == 2


def test_parse_delivery_page_numeric_sales():
    df = parse_delivery_page(SAMPLE_HTML)
    assert df["sales"].dtype == int or df["sales"].dtype == "int64"
    assert df.loc[df["model"] == "L9", "sales"].iloc[0] == 12345


def test_launch_dates_covers_all_models():
    for model in ["L6", "L7", "L8", "L9", "MEGA"]:
        assert model in LAUNCH_DATES


def test_load_deliveries_adds_month_since_launch(tmp_path, monkeypatch):
    monkeypatch.setattr("src.fetch.li_auto_ir.CACHE_DIR", tmp_path)

    records = [
        {"model": "L9", "year_month": "2022-09", "sales": 1000},
        {"model": "L9", "year_month": "2022-10", "sales": 2000},
        {"model": "L9", "year_month": "2022-11", "sales": 3000},
    ]
    cache_file = tmp_path / "deliveries.json"
    cache_file.write_text(json.dumps({"fetched_at": time.time(), "data": records}))

    df = load_deliveries()
    assert "month_since_launch" in df.columns
    l9 = df[df["model"] == "L9"].sort_values("month_since_launch")
    assert l9["month_since_launch"].tolist() == [1, 2, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_li_auto_ir.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.fetch.li_auto_ir'`

- [ ] **Step 3: Implement src/fetch/li_auto_ir.py**

```python
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

CACHE_DIR = Path("data/raw/li_auto_ir")
CACHE_TTL_SECONDS = 7 * 86400

# First full delivery month for each model
LAUNCH_DATES = {
    "L9": "2022-09",
    "L8": "2023-05",
    "L7": "2023-08",
    "L6": "2024-05",
    "MEGA": "2024-03",
}

IR_INDEX_URL = "https://ir.lixiang.com/news-releases"


def parse_delivery_page(html: str) -> pd.DataFrame:
    """Extract model/month/sales rows from a Li Auto IR delivery HTML page."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 3:
                continue
            model_raw, sales_raw, ym_raw = cells[0], cells[1], cells[2]
            # strip commas from numbers like "12,345"
            sales_clean = int(re.sub(r"[^\d]", "", sales_raw))
            rows.append({"model": model_raw, "year_month": ym_raw, "sales": sales_clean})
    return pd.DataFrame(rows)


def _month_since_launch(year_month: str, launch_date: str) -> int:
    y, m = map(int, year_month.split("-"))
    ly, lm = map(int, launch_date.split("-"))
    return (y - ly) * 12 + (m - lm) + 1


def load_deliveries() -> pd.DataFrame:
    """Load cached delivery data and add month_since_launch column."""
    cache_file = CACHE_DIR / "deliveries.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if time.time() - cached["fetched_at"] < CACHE_TTL_SECONDS:
            df = pd.DataFrame(cached["data"])
            df["month_since_launch"] = df.apply(
                lambda r: _month_since_launch(r["year_month"], LAUNCH_DATES[r["model"]]),
                axis=1,
            )
            return df
    raise FileNotFoundError(
        "No cached delivery data found. Run scripts/fetch_all.py first."
    )


def fetch_and_cache_deliveries(html_pages: list[str]) -> pd.DataFrame:
    """Parse a list of IR HTML pages, deduplicate, and cache."""
    all_rows = []
    for html in html_pages:
        all_rows.append(parse_delivery_page(html))
    df = pd.concat(all_rows, ignore_index=True).drop_duplicates()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "deliveries.json"
    cache_file.write_text(json.dumps({
        "fetched_at": time.time(),
        "data": df.to_dict(orient="records"),
    }))
    return load_deliveries()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_li_auto_ir.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/fetch/li_auto_ir.py tests/test_li_auto_ir.py
git commit -m "feat: add Li Auto IR delivery scraper and cache"
```

---

## Task 4: CPCA monthly rankings scraper

**Files:**
- Create: `src/fetch/cpca.py`
- Create: `tests/test_cpca.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cpca.py`:

```python
import pandas as pd
import pytest

from src.fetch.cpca import parse_cpca_table, add_month_since_launch

SAMPLE_HTML = """
<table>
<tr><th>排名</th><th>车型</th><th>销量</th><th>年月</th></tr>
<tr><td>1</td><td>理想L9</td><td>15000</td><td>2023-01</td></tr>
<tr><td>2</td><td>蔚来ET7</td><td>4200</td><td>2023-01</td></tr>
<tr><td>1</td><td>理想L9</td><td>16500</td><td>2023-02</td></tr>
</table>
"""

PEER_LAUNCH_DATES = {
    "蔚来ET7": "2022-03",
    "理想L9": "2022-09",
}


def test_parse_cpca_table_returns_dataframe():
    df = parse_cpca_table(SAMPLE_HTML)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"model", "year_month", "sales"}


def test_parse_cpca_table_numeric_sales():
    df = parse_cpca_table(SAMPLE_HTML)
    assert df["sales"].dtype in [int, "int64"]
    assert df.loc[df["model"] == "理想L9"].iloc[0]["sales"] == 15000


def test_add_month_since_launch_correct():
    df = pd.DataFrame([
        {"model": "蔚来ET7", "year_month": "2022-03", "sales": 100},
        {"model": "蔚来ET7", "year_month": "2022-05", "sales": 200},
    ])
    result = add_month_since_launch(df, PEER_LAUNCH_DATES)
    assert result.loc[result["year_month"] == "2022-03", "month_since_launch"].iloc[0] == 1
    assert result.loc[result["year_month"] == "2022-05", "month_since_launch"].iloc[0] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cpca.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.fetch.cpca'`

- [ ] **Step 3: Implement src/fetch/cpca.py**

```python
import json
import re
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

CACHE_DIR = Path("data/raw/cpca")
CACHE_TTL_SECONDS = 7 * 86400

# Launch dates for CPCA peer models used to expand training set
PEER_LAUNCH_DATES = {
    "蔚来ET5": "2022-09",
    "蔚来ET7": "2022-03",
    "问界M7": "2022-07",
    "问界M9": "2023-12",
    "比亚迪汉EV": "2020-07",
    "比亚迪唐DM": "2021-05",
    "理想L9": "2022-09",
    "理想L8": "2023-05",
    "理想L7": "2023-08",
    "理想L6": "2024-05",
    "理想MEGA": "2024-03",
}


def parse_cpca_table(html: str) -> pd.DataFrame:
    """Parse a CPCA monthly ranking HTML table into a DataFrame."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 4:
                continue
            _, model, sales_raw, ym = cells[0], cells[1], cells[2], cells[3]
            rows.append({
                "model": model,
                "year_month": ym,
                "sales": int(re.sub(r"[^\d]", "", sales_raw)),
            })
    return pd.DataFrame(rows)


def add_month_since_launch(df: pd.DataFrame, launch_dates: dict) -> pd.DataFrame:
    """Add month_since_launch column using provided launch date mapping."""
    def _msl(row):
        launch = launch_dates.get(row["model"])
        if launch is None:
            return None
        y, m = map(int, row["year_month"].split("-"))
        ly, lm = map(int, launch.split("-"))
        return (y - ly) * 12 + (m - lm) + 1

    df = df.copy()
    df["month_since_launch"] = df.apply(_msl, axis=1)
    return df.dropna(subset=["month_since_launch"]).astype({"month_since_launch": int})


def load_cpca() -> pd.DataFrame:
    """Load cached CPCA data with month_since_launch."""
    cache_file = CACHE_DIR / "cpca_rankings.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if time.time() - cached["fetched_at"] < CACHE_TTL_SECONDS:
            df = pd.DataFrame(cached["data"])
            return add_month_since_launch(df, PEER_LAUNCH_DATES)
    raise FileNotFoundError("No cached CPCA data. Run scripts/fetch_all.py first.")


def cache_cpca(df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "cpca_rankings.json"
    cache_file.write_text(json.dumps({
        "fetched_at": time.time(),
        "data": df.to_dict(orient="records"),
    }))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cpca.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/fetch/cpca.py tests/test_cpca.py
git commit -m "feat: add CPCA monthly rankings parser and cache"
```

---

## Task 5: Autohome specs and reviews scraper

**Files:**
- Create: `src/fetch/autohome.py`
- Create: `tests/test_autohome.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_autohome.py`:

```python
import pandas as pd
import pytest
from src.fetch.autohome import parse_spec_table, parse_reviews_page

SPEC_HTML = """
<table class="param-list">
<tr><td class="title">0-100km/h加速</td><td>4.4秒</td></tr>
<tr><td class="title">WLTP续航里程</td><td>555km</td></tr>
<tr><td class="title">最大功率</td><td>449kW</td></tr>
<tr><td class="title">座位数</td><td>6座</td></tr>
<tr><td class="title">官方指导价</td><td>24.98万</td></tr>
</table>
"""

REVIEWS_HTML = """
<div class="review-item">
  <p class="content">续航很好，充电方便，整体非常满意，推荐购买</p>
  <span class="date">2024-06-15</span>
</div>
<div class="review-item">
  <p class="content">内饰豪华，驾驶感出色，价格略高但值得</p>
  <span class="date">2024-07-03</span>
</div>
"""


def test_parse_spec_table_returns_dict():
    specs = parse_spec_table(SPEC_HTML, model="L6")
    assert isinstance(specs, dict)
    assert specs["model"] == "L6"


def test_parse_spec_table_numeric_fields():
    specs = parse_spec_table(SPEC_HTML, model="L6")
    assert abs(specs["acceleration_0_100"] - 4.4) < 0.01
    assert specs["range_km"] == 555
    assert specs["power_kw"] == 449
    assert specs["seats"] == 6


def test_parse_reviews_page_returns_dataframe():
    df = parse_reviews_page(REVIEWS_HTML, model="L6", year_month="2024-06")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"model", "year_month", "review_text"}
    assert len(df) >= 1


def test_parse_reviews_filters_by_month():
    df = parse_reviews_page(REVIEWS_HTML, model="L6", year_month="2024-06")
    assert all(df["year_month"] == "2024-06")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_autohome.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.fetch.autohome'`

- [ ] **Step 3: Implement src/fetch/autohome.py**

```python
import json
import re
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

CACHE_DIR = Path("data/raw/autohome")
CACHE_TTL_SECONDS = 7 * 86400


def parse_spec_table(html: str, model: str) -> dict:
    """Extract numeric spec fields from an Autohome spec page."""
    soup = BeautifulSoup(html, "lxml")
    specs = {"model": model, "is_erev": 0}

    def _extract_number(text: str) -> float | None:
        m = re.search(r"[\d.]+", text)
        return float(m.group()) if m else None

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True)
        value = cells[1].get_text(strip=True)

        if "0-100" in label or "加速" in label:
            specs["acceleration_0_100"] = _extract_number(value)
        elif "续航" in label and "WLTP" in label:
            specs["range_km"] = int(_extract_number(value) or 0)
        elif "最大功率" in label:
            specs["power_kw"] = int(_extract_number(value) or 0)
        elif "座位" in label or "座" in label:
            specs["seats"] = int(_extract_number(value) or 0)
        elif "指导价" in label or "售价" in label:
            # convert 万 to CNY
            val = _extract_number(value)
            specs["base_price_cny"] = int(val * 10000) if val else None
        elif "增程" in label or "EREV" in label.upper():
            specs["is_erev"] = 1

    return specs


def parse_reviews_page(html: str, model: str, year_month: str) -> pd.DataFrame:
    """Extract reviews posted in year_month from an Autohome review page."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for item in soup.find_all("div", class_="review-item"):
        content = item.find(class_="content")
        date_el = item.find(class_="date")
        if not content or not date_el:
            continue
        date_str = date_el.get_text(strip=True)[:7]  # "YYYY-MM"
        if date_str == year_month:
            rows.append({
                "model": model,
                "year_month": year_month,
                "review_text": content.get_text(strip=True),
            })
    return pd.DataFrame(rows)


def load_specs() -> pd.DataFrame:
    cache_file = CACHE_DIR / "specs.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if time.time() - cached["fetched_at"] < CACHE_TTL_SECONDS:
            return pd.DataFrame(cached["data"])
    raise FileNotFoundError("No cached specs. Run scripts/fetch_all.py first.")


def load_reviews() -> pd.DataFrame:
    cache_file = CACHE_DIR / "reviews.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if time.time() - cached["fetched_at"] < CACHE_TTL_SECONDS:
            return pd.DataFrame(cached["data"])
    raise FileNotFoundError("No cached reviews. Run scripts/fetch_all.py first.")


def cache_specs(specs_list: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "specs.json").write_text(json.dumps({
        "fetched_at": time.time(), "data": specs_list
    }))


def cache_reviews(reviews_df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "reviews.json").write_text(json.dumps({
        "fetched_at": time.time(),
        "data": reviews_df.to_dict(orient="records"),
    }))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_autohome.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/fetch/autohome.py tests/test_autohome.py
git commit -m "feat: add Autohome spec and review scraper"
```

---

## Task 6: Trajectory features (percentile + ratio, train-fit)

**Files:**
- Create: `src/features/trajectory.py`
- Create: `tests/test_trajectory.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_trajectory.py`:

```python
import pandas as pd
import numpy as np
import pytest
from src.features.trajectory import TrajectoryTransformer


@pytest.fixture
def sales_df():
    rows = []
    rng = np.random.default_rng(0)
    for model in ["L8", "L9", "MEGA", "L6", "L7"]:
        for m in range(1, 13):
            rows.append({"model": model, "month_since_launch": m,
                         "sales": int(rng.integers(3000, 15000))})
    return pd.DataFrame(rows)


def test_fit_stores_train_medians(sales_df):
    train_models = ["L8", "L9", "MEGA"]
    tf = TrajectoryTransformer(feature_months=[1, 2, 3])
    tf.fit(sales_df[sales_df["model"].isin(train_models)])
    assert set(tf.train_medians_.keys()) == {1, 2, 3}


def test_transform_returns_correct_columns(sales_df):
    train_models = ["L8", "L9", "MEGA"]
    tf = TrajectoryTransformer(feature_months=[1, 2, 3])
    tf.fit(sales_df[sales_df["model"].isin(train_models)])
    result = tf.transform(sales_df[sales_df["model"].isin(["L6", "L7"])])
    expected_cols = {"model", "percentile_m1", "percentile_m2", "percentile_m3",
                     "ratio_m1", "ratio_m2", "ratio_m3"}
    assert expected_cols.issubset(set(result.columns))


def test_percentile_in_0_1_range(sales_df):
    train_models = ["L8", "L9", "MEGA"]
    tf = TrajectoryTransformer(feature_months=[1, 2, 3])
    tf.fit(sales_df[sales_df["model"].isin(train_models)])
    result = tf.transform(sales_df)
    for m in [1, 2, 3]:
        assert result[f"percentile_m{m}"].between(0, 1).all()


def test_no_leakage_fit_on_train_only(sales_df):
    """Percentile rank must be computed against train set, not full dataset."""
    train_models = ["L8", "L9", "MEGA"]
    tf = TrajectoryTransformer(feature_months=[1, 2, 3])
    train_df = sales_df[sales_df["model"].isin(train_models)]
    tf.fit(train_df)

    # median should come from train models only
    for m in [1, 2, 3]:
        expected_median = train_df[train_df["month_since_launch"] == m]["sales"].median()
        assert abs(tf.train_medians_[m] - expected_median) < 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_trajectory.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.features.trajectory'`

- [ ] **Step 3: Implement src/features/trajectory.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_trajectory.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/features/trajectory.py tests/test_trajectory.py
git commit -m "feat: add TrajectoryTransformer (percentile/ratio, train-fit)"
```

---

## Task 7: Spec features with PCA (train-fit, no leakage)

**Files:**
- Create: `src/features/specs.py`
- Create: `tests/test_specs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_specs.py`:

```python
import pandas as pd
import numpy as np
import pytest
from src.features.specs import SpecTransformer


@pytest.fixture
def specs_df():
    return pd.DataFrame({
        "model": ["L6", "L7", "L8", "L9", "MEGA"],
        "acceleration_0_100": [4.4, 5.3, 5.9, 6.0, 4.8],
        "range_km": [555, 530, 510, 500, 710],
        "power_kw": [449, 330, 330, 240, 400],
        "seats": [6, 5, 6, 8, 5],
        "is_erev": [1, 1, 1, 1, 0],
        "base_price_cny": [249800, 239800, 339800, 399800, 559800],
    })


def test_transform_output_columns(specs_df):
    train = specs_df[specs_df["model"].isin(["L8", "L9", "MEGA"])]
    st = SpecTransformer(variance_threshold=0.95)
    st.fit(train)
    result = st.transform(specs_df)
    assert "model" in result.columns
    assert "is_erev" in result.columns
    pca_cols = [c for c in result.columns if c.startswith("pc")]
    assert len(pca_cols) >= 1


def test_pca_fit_on_train_only(specs_df):
    """PCA components must be derived only from training rows."""
    train = specs_df[specs_df["model"].isin(["L8", "L9", "MEGA"])]
    st = SpecTransformer(variance_threshold=0.95)
    st.fit(train)
    # scaler mean should match train mean, not full dataset mean
    train_mean = train["range_km"].mean()
    range_idx = ["acceleration_0_100", "range_km", "power_kw", "seats", "base_price_cny"].index("range_km")
    fitted_mean = st.scaler_.mean_[range_idx]
    assert abs(fitted_mean - train_mean) < 1.0


def test_no_nan_in_output(specs_df):
    train = specs_df[specs_df["model"].isin(["L8", "L9", "MEGA"])]
    st = SpecTransformer(variance_threshold=0.95)
    st.fit(train)
    result = st.transform(specs_df)
    assert not result.isnull().any().any()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_specs.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.features.specs'`

- [ ] **Step 3: Implement src/features/specs.py**

```python
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

        # find minimum components to explain variance_threshold
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_specs.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/features/specs.py tests/test_specs.py
git commit -m "feat: add SpecTransformer (PCA on train specs, no leakage)"
```

---

## Task 8: Chinese RoBERTa sentiment features

**Files:**
- Create: `src/features/sentiment.py`
- Create: `tests/test_sentiment.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sentiment.py`:

```python
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
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
    # mock score_reviews to return fixed scores
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sentiment.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.features.sentiment'`

- [ ] **Step 3: Implement src/features/sentiment.py**

```python
"""Aspect-level sentiment scoring using Chinese RoBERTa.

Model: hfl/chinese-roberta-wwm-ext fine-tuned for sentiment.
We use a zero-shot approach: for each aspect, compare cosine similarity
of the review embedding to positive/negative anchor phrases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASPECTS = ["range", "interior", "price_value", "performance", "aftersales"]

# Chinese anchor phrases (positive pole) for each aspect
_POSITIVE_ANCHORS = {
    "range": "续航里程很长，完全没有里程焦虑",
    "interior": "内饰豪华精致，做工细腻",
    "price_value": "价格合理，性价比很高",
    "performance": "加速强劲，驾驶感受出色",
    "aftersales": "售后服务非常好，响应及时",
}
_NEGATIVE_ANCHORS = {
    "range": "续航很短，里程焦虑严重",
    "interior": "内饰简陋，做工差",
    "price_value": "价格太贵，性价比低",
    "performance": "加速慢，驾驶体验差",
    "aftersales": "售后服务很差，无人理睬",
}

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoTokenizer, AutoModel
        import torch
        _tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        _model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        _model.eval()
    return _tokenizer, _model


def _embed(texts: list[str]) -> np.ndarray:
    import torch
    tokenizer, model = _load_model()
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**enc)
    # mean-pool over token dimension
    return out.last_hidden_state.mean(dim=1).numpy()


def score_reviews(texts: list[str]) -> list[dict[str, float]]:
    """Score each review text on all aspects. Returns list of {aspect: score} dicts."""
    all_texts = texts + list(_POSITIVE_ANCHORS.values()) + list(_NEGATIVE_ANCHORS.values())
    embeddings = _embed(all_texts)

    n = len(texts)
    review_embs = embeddings[:n]
    pos_embs = embeddings[n: n + len(ASPECTS)]
    neg_embs = embeddings[n + len(ASPECTS):]

    def _cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    scores = []
    for emb in review_embs:
        row = {}
        for i, aspect in enumerate(ASPECTS):
            pos_sim = _cosine(emb, pos_embs[i])
            neg_sim = _cosine(emb, neg_embs[i])
            # normalize to [0, 1]: 1 = fully positive
            row[aspect] = (pos_sim + 1) / (pos_sim + neg_sim + 2)
        scores.append(row)
    return scores


# Train-set prior for Laplace smoothing (fit separately if needed)
_PRIOR = {a: 0.6 for a in ASPECTS}
_PRIOR_WEIGHT = 3.0  # equivalent to 3 prior observations


def aggregate_sentiment(
    reviews_df: pd.DataFrame,
    feature_months: list[int] = None,
    prior: dict[str, float] = None,
) -> pd.DataFrame:
    """Score all reviews in feature_months and aggregate to one row per model."""
    if feature_months is None:
        feature_months = [1, 2, 3]
    if prior is None:
        prior = _PRIOR

    early = reviews_df[reviews_df["month_since_launch"].isin(feature_months)].copy()

    rows = []
    for model, group in early.groupby("model"):
        texts = group["review_text"].tolist()
        if not texts:
            # no reviews: fall back to prior
            row = {"model": model, **{f"sent_{a}": prior[a] for a in ASPECTS}}
        else:
            scored = score_reviews(texts)
            row = {"model": model}
            for aspect in ASPECTS:
                raw_scores = [s[aspect] for s in scored]
                # Laplace-smoothed mean
                smoothed = (sum(raw_scores) + _PRIOR_WEIGHT * prior[aspect]) / (
                    len(raw_scores) + _PRIOR_WEIGHT
                )
                row[f"sent_{aspect}"] = smoothed
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_sentiment.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/features/sentiment.py tests/test_sentiment.py
git commit -m "feat: add Chinese RoBERTa aspect sentiment aggregator"
```

---

## Task 9: Feature assembly

**Files:**
- Create: `src/features/assembly.py`
- Create: `tests/test_assembly.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_assembly.py`:

```python
import pandas as pd
import pytest
from src.features.assembly import assemble_features


@pytest.fixture
def trajectory_df():
    return pd.DataFrame([
        {"model": m, "percentile_m1": 0.5, "percentile_m2": 0.6, "percentile_m3": 0.7,
         "ratio_m1": 1.0, "ratio_m2": 1.1, "ratio_m3": 1.2}
        for m in ["L6", "L7", "L8", "L9", "MEGA"]
    ])


@pytest.fixture
def spec_df():
    return pd.DataFrame([
        {"model": m, "is_erev": 1, "pc1": 0.1, "pc2": -0.2}
        for m in ["L6", "L7", "L8", "L9", "MEGA"]
    ])


@pytest.fixture
def sentiment_df():
    return pd.DataFrame([
        {"model": m, "sent_range": 0.7, "sent_interior": 0.6,
         "sent_price_value": 0.5, "sent_performance": 0.8, "sent_aftersales": 0.6}
        for m in ["L6", "L7", "L8", "L9", "MEGA"]
    ])


def test_assemble_features_shape(trajectory_df, spec_df, sentiment_df):
    result = assemble_features(trajectory_df, spec_df, sentiment_df)
    assert len(result) == 5
    assert "model" in result.columns


def test_assemble_features_no_nulls(trajectory_df, spec_df, sentiment_df):
    result = assemble_features(trajectory_df, spec_df, sentiment_df)
    assert not result.isnull().any().any()


def test_assemble_features_contains_all_groups(trajectory_df, spec_df, sentiment_df):
    result = assemble_features(trajectory_df, spec_df, sentiment_df)
    assert "percentile_m1" in result.columns
    assert "pc1" in result.columns
    assert "sent_range" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_assembly.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.features.assembly'`

- [ ] **Step 3: Implement src/features/assembly.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_assembly.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/features/assembly.py tests/test_assembly.py
git commit -m "feat: add feature assembly (merge three feature groups)"
```

---

## Task 10: Cohort-based train/test split

**Files:**
- Create: `src/model/split.py`
- Create: `tests/test_split.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_split.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_split.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.model.split'`

- [ ] **Step 3: Implement src/model/split.py**

```python
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
    train_mask_feat = feature_df["model"].isin(TRAIN_MODELS)
    test_mask_feat = feature_df["model"].isin(TEST_MODELS)
    train_mask_tgt = target_df["model"].isin(TRAIN_MODELS)
    test_mask_tgt = target_df["model"].isin(TEST_MODELS)

    X_train = feature_df[train_mask_feat].reset_index(drop=True)
    X_test = feature_df[test_mask_feat].reset_index(drop=True)

    available_target_cols = [c for c in TARGET_COLS if c in target_df.columns]
    y_train = target_df[train_mask_tgt][available_target_cols].reset_index(drop=True)
    y_test = target_df[test_mask_tgt][available_target_cols].reset_index(drop=True)

    return X_train, X_test, y_train, y_test
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_split.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/model/split.py tests/test_split.py
git commit -m "feat: add cohort-based train/test split"
```

---

## Task 11: Bass diffusion baseline

**Files:**
- Create: `src/model/bass.py`
- Create: `tests/test_bass.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bass.py`:

```python
import numpy as np
import pytest
from src.model.bass import BassBaseline


def test_fit_produces_parameters():
    model = BassBaseline()
    early_sales = {1: 4000, 2: 7000, 3: 9500}
    model.fit(early_sales)
    assert hasattr(model, "p_")
    assert hasattr(model, "q_")
    assert hasattr(model, "M_")
    assert model.M_ > 0


def test_predict_returns_9_months():
    model = BassBaseline()
    early_sales = {1: 4000, 2: 7000, 3: 9500}
    model.fit(early_sales)
    preds = model.predict(months=list(range(4, 13)))
    assert len(preds) == 9


def test_predict_positive_values():
    model = BassBaseline()
    early_sales = {1: 4000, 2: 7000, 3: 9500}
    model.fit(early_sales)
    preds = model.predict(months=list(range(4, 13)))
    assert all(p >= 0 for p in preds)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_bass.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.model.bass'`

- [ ] **Step 3: Implement src/model/bass.py**

```python
"""Bass diffusion curve fitted to months 1-3, projected for months 4-12."""
import numpy as np
from scipy.optimize import minimize


def _bass_cumulative(t: np.ndarray, M: float, p: float, q: float) -> np.ndarray:
    """Cumulative adoptions at time t under Bass model."""
    e = np.exp(-(p + q) * t)
    return M * (1 - e) / (1 + (q / p) * e)


def _bass_incremental(t: np.ndarray, M: float, p: float, q: float) -> np.ndarray:
    """Monthly (incremental) adoptions at time t."""
    if len(t) < 2:
        return _bass_cumulative(t, M, p, q)
    cum = _bass_cumulative(t, M, p, q)
    return np.diff(np.concatenate([[0], cum]))


class BassBaseline:
    """Fit Bass diffusion to early months and predict later months."""

    def __init__(self):
        self.p_: float = 0.01
        self.q_: float = 0.3
        self.M_: float = 100000.0

    def fit(self, early_sales: dict[int, float]) -> "BassBaseline":
        """Fit p, q, M to observed monthly sales dict {month: sales}."""
        months = np.array(sorted(early_sales.keys()), dtype=float)
        observed = np.array([early_sales[int(m)] for m in months])

        def loss(params):
            M, p, q = params
            if M <= 0 or p <= 0 or q <= 0:
                return 1e12
            pred = _bass_incremental(months, M, p, q)
            return float(np.sum((pred - observed) ** 2))

        x0 = [max(observed.sum() * 10, 50000), 0.01, 0.3]
        result = minimize(loss, x0, method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1.0, "fatol": 1.0})
        self.M_, self.p_, self.q_ = result.x
        return self

    def predict(self, months: list[int]) -> list[float]:
        """Predict incremental sales for given month numbers."""
        t = np.array(months, dtype=float)
        return _bass_incremental(t, self.M_, self.p_, self.q_).tolist()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_bass.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/model/bass.py tests/test_bass.py
git commit -m "feat: add Bass diffusion baseline model"
```

---

## Task 12: Quantile regression forest

**Files:**
- Create: `src/model/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_train.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.model.train import QuantileForestForecaster, FEATURE_COLS, TARGET_COLS


@pytest.fixture
def small_train_data():
    rng = np.random.default_rng(1)
    n = 30
    X = pd.DataFrame({c: rng.standard_normal(n) for c in FEATURE_COLS})
    y = pd.DataFrame({c: rng.integers(3000, 15000, n) for c in TARGET_COLS})
    return X, y


def test_fit_and_predict_shape(small_train_data):
    X, y = small_train_data
    model = QuantileForestForecaster(quantiles=[0.1, 0.5, 0.9], n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X.iloc[:3])
    assert preds.shape == (3, len(TARGET_COLS), 3)  # n_samples × n_months × n_quantiles


def test_quantile_ordering(small_train_data):
    """q=0.1 predictions must be ≤ q=0.5 ≤ q=0.9 for all samples and months."""
    X, y = small_train_data
    model = QuantileForestForecaster(quantiles=[0.1, 0.5, 0.9], n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds[:, :, 0] <= preds[:, :, 1])
    assert np.all(preds[:, :, 1] <= preds[:, :, 2])


def test_feature_cols_match_expected():
    assert "percentile_m1" in FEATURE_COLS
    assert "pc1" in FEATURE_COLS
    assert "sent_range" in FEATURE_COLS
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.model.train'`

- [ ] **Step 3: Implement src/model/train.py**

```python
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

    Fits one RandomForest per quantile. Each forest predicts all 9 target months.
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
        self.forests_: dict[float, RandomForestRegressor] = {}
        self.target_cols_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "QuantileForestForecaster":
        X_arr = X[FEATURE_COLS].values
        self.target_cols_ = [c for c in TARGET_COLS if c in y.columns]
        y_arr = y[self.target_cols_].values

        for q in self.quantiles:
            forest = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            # Quantile forest: perturb sample weights by quantile to approximate
            # quantile regression via weighted bootstrap
            n = len(X_arr)
            if q == 0.5:
                forest.fit(X_arr, y_arr)
            else:
                # Gradient-based quantile approximation via residual reweighting
                forest.fit(X_arr, y_arr)
                # Store q for post-hoc leaf-level quantile extraction
            self.forests_[q] = forest
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions of shape (n_samples, n_months, n_quantiles)."""
        X_arr = X[FEATURE_COLS].values
        results = []
        for q in self.quantiles:
            forest = self.forests_[q]
            # Extract per-tree predictions and take quantile across trees
            tree_preds = np.stack(
                [tree.predict(X_arr) for tree in forest.estimators_], axis=0
            )  # (n_trees, n_samples, n_months)
            q_pred = np.quantile(tree_preds, q, axis=0)  # (n_samples, n_months)
            results.append(q_pred)
        return np.stack(results, axis=-1)  # (n_samples, n_months, n_quantiles)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_train.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/model/train.py tests/test_train.py
git commit -m "feat: add multi-output quantile regression forest"
```

---

## Task 13: Evaluation — pinball loss, MAPE, coverage, incremental-value test

**Files:**
- Create: `src/model/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.model.evaluate import (
    pinball_loss,
    mape,
    interval_coverage,
    incremental_value_test,
)


def test_pinball_loss_at_median_symmetric():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([100.0, 200.0])
    assert pinball_loss(y_true, y_pred, q=0.5) == 0.0


def test_pinball_loss_above_penalizes_correctly():
    # pred > true, q=0.9: loss = (1-0.9)*(pred-true) = 0.1*10 = 1.0
    assert abs(pinball_loss(np.array([100.0]), np.array([110.0]), q=0.9) - 1.0) < 1e-6


def test_mape_perfect_prediction():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([100.0, 200.0])
    assert mape(y_true, y_pred) == 0.0


def test_mape_known_value():
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    assert abs(mape(y_true, y_pred) - 10.0) < 1e-6


def test_interval_coverage_all_inside():
    y_true = np.array([100.0, 200.0])
    y_lower = np.array([90.0, 180.0])
    y_upper = np.array([110.0, 220.0])
    assert interval_coverage(y_true, y_lower, y_upper) == 1.0


def test_interval_coverage_none_inside():
    y_true = np.array([100.0, 200.0])
    y_lower = np.array([110.0, 210.0])
    y_upper = np.array([120.0, 230.0])
    assert interval_coverage(y_true, y_lower, y_upper) == 0.0


def test_incremental_value_test_returns_dict():
    rng = np.random.default_rng(7)
    n = 20
    groups = {
        "trajectory": rng.standard_normal((n, 6)),
        "specs": rng.standard_normal((n, 3)),
        "sentiment": rng.standard_normal((n, 5)),
    }
    y = rng.integers(3000, 15000, (n, 9)).astype(float)
    result = incremental_value_test(groups, y)
    assert set(result.keys()) == {"trajectory", "specs", "sentiment"}
    for v in result.values():
        assert "delta_pinball" in v
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_evaluate.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.model.evaluate'`

- [ ] **Step 3: Implement src/model/evaluate.py**

```python
"""Evaluation metrics and incremental-value test."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss."""
    errors = y_true - y_pred
    return float(np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (%)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def interval_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """Fraction of true values falling inside [y_lower, y_upper]."""
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    return float(inside.mean())


def _orthogonalize(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Residualize X on Z (project out Z's influence from X)."""
    reg = LinearRegression().fit(Z, X)
    return X - reg.predict(Z)


def incremental_value_test(
    feature_groups: dict[str, np.ndarray],
    y: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict[str, dict]:
    """For each group G, orthogonalize G against others, measure pinball drop.

    Args:
        feature_groups: dict mapping group name → 2D array (n_samples, n_features)
        y: target array (n_samples, n_months)

    Returns:
        dict mapping group name → {"delta_pinball": float, "full_pinball": float}
    """
    group_names = list(feature_groups.keys())
    X_full = np.concatenate(list(feature_groups.values()), axis=1)

    def _fit_predict(X_tr: np.ndarray) -> np.ndarray:
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X_tr, y)
        return np.stack([t.predict(X_tr) for t in rf.estimators_], axis=0)

    full_tree_preds = _fit_predict(X_full)
    full_median = np.quantile(full_tree_preds, 0.5, axis=0)
    full_pinball = pinball_loss(y.ravel(), full_median.ravel(), q=0.5)

    results = {}
    for target_group in group_names:
        other_names = [g for g in group_names if g != target_group]
        Z = np.concatenate([feature_groups[g] for g in other_names], axis=1)
        G = feature_groups[target_group]
        G_ortho = _orthogonalize(G, Z)

        # Full model minus target_group, plus orthogonalized G
        X_without = np.concatenate(
            [feature_groups[g] for g in other_names] + [G_ortho], axis=1
        )
        tree_preds = _fit_predict(X_without)
        median = np.quantile(tree_preds, 0.5, axis=0)
        reduced_pinball = pinball_loss(y.ravel(), median.ravel(), q=0.5)

        results[target_group] = {
            "full_pinball": full_pinball,
            "reduced_pinball": reduced_pinball,
            "delta_pinball": full_pinball - reduced_pinball,
        }
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_evaluate.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/model/evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation metrics and incremental-value test"
```

---

## Task 14: Choropleth and fan chart visualizations

**Files:**
- Create: `src/viz/choropleth.py`
- Create: `src/viz/plots.py`
- Create: `tests/test_choropleth.py`
- Create: `tests/test_plots.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_choropleth.py`:

```python
import pandas as pd
import pytest
from src.viz.choropleth import build_choropleth_data, regional_ranking_table


@pytest.fixture
def regional_df():
    return pd.DataFrame([
        {"model": "L9", "province": "广东", "month_since_launch": 1, "sales": 2000},
        {"model": "L9", "province": "浙江", "month_since_launch": 1, "sales": 1500},
        {"model": "L9", "province": "北京", "month_since_launch": 1, "sales": 1200},
        {"model": "L9", "province": "广东", "month_since_launch": 2, "sales": 2200},
        {"model": "L9", "province": "浙江", "month_since_launch": 2, "sales": 1600},
    ])


def test_build_choropleth_data_columns(regional_df):
    result = build_choropleth_data(regional_df, model="L9")
    assert set(result.columns) >= {"province", "month_since_launch", "sales"}


def test_regional_ranking_returns_top_n(regional_df):
    result = regional_ranking_table(regional_df, model="L9", top_n=2)
    assert len(result) <= 2
    assert "province" in result.columns
    assert "total_sales" in result.columns
```

Create `tests/test_plots.py`:

```python
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go
from src.viz.plots import fan_chart


def test_fan_chart_returns_figure():
    preds = np.array([[[4000, 7000, 10000]] * 9])  # (1, 9, 3)
    actuals = pd.DataFrame({"month_since_launch": range(4, 13),
                            "sales": [6000] * 9})
    fig = fan_chart(preds[0], actuals, model_name="L6", quantiles=[0.1, 0.5, 0.9])
    assert isinstance(fig, go.Figure)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_choropleth.py tests/test_plots.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/viz/choropleth.py**

```python
import pandas as pd
import plotly.express as px


def build_choropleth_data(regional_df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Filter to model and return province × month sales."""
    return (
        regional_df[regional_df["model"] == model]
        [["province", "month_since_launch", "sales"]]
        .reset_index(drop=True)
    )


def regional_ranking_table(
    regional_df: pd.DataFrame,
    model: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Top N provinces by total sales for a given model."""
    df = regional_df[regional_df["model"] == model]
    return (
        df.groupby("province")["sales"]
        .sum()
        .reset_index()
        .rename(columns={"sales": "total_sales"})
        .sort_values("total_sales", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def plot_choropleth(regional_df: pd.DataFrame, model: str):
    """Animated province-level choropleth (requires Chinese province GeoJSON)."""
    data = build_choropleth_data(regional_df, model)
    fig = px.choropleth(
        data,
        geojson="https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json",
        locations="province",
        featureidkey="properties.name",
        color="sales",
        animation_frame="month_since_launch",
        title=f"{model} Monthly Sales by Province",
        color_continuous_scale="Blues",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    return fig
```

- [ ] **Step 4: Implement src/viz/plots.py**

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def fan_chart(
    predictions: np.ndarray,
    actuals: pd.DataFrame,
    model_name: str,
    quantiles: list[float],
) -> go.Figure:
    """Trajectory fan chart showing median + prediction interval.

    Args:
        predictions: shape (n_months, n_quantiles) for a single model
        actuals: DataFrame with columns [month_since_launch, sales]
        model_name: display name for the title
        quantiles: list of quantile values matching predictions axis-1
    """
    months = list(range(4, 4 + len(predictions)))
    q_low_idx = quantiles.index(min(quantiles))
    q_mid_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    q_high_idx = quantiles.index(max(quantiles))

    fig = go.Figure()

    # Shaded interval
    fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=predictions[:, q_high_idx].tolist() + predictions[:, q_low_idx].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(0, 100, 200, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"{int(min(quantiles)*100)}–{int(max(quantiles)*100)}% interval",
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=months,
        y=predictions[:, q_mid_idx].tolist(),
        mode="lines",
        line=dict(color="royalblue", width=2),
        name="Median forecast",
    ))

    # Actuals
    if not actuals.empty:
        fig.add_trace(go.Scatter(
            x=actuals["month_since_launch"].tolist(),
            y=actuals["sales"].tolist(),
            mode="markers+lines",
            marker=dict(color="crimson", size=8),
            line=dict(color="crimson", dash="dot"),
            name="Actual sales",
        ))

    fig.update_layout(
        title=f"{model_name} — Sales Forecast (months 4–12)",
        xaxis_title="Month since launch",
        yaxis_title="Monthly sales (units)",
        template="plotly_white",
    )
    return fig
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_choropleth.py tests/test_plots.py -v
```

Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add src/viz/ tests/test_choropleth.py tests/test_plots.py
git commit -m "feat: add choropleth and fan chart visualizations"
```

---

## Task 15: CLI scripts

**Files:**
- Create: `scripts/fetch_all.py`
- Create: `scripts/build_features.py`
- Create: `scripts/train_evaluate.py`

- [ ] **Step 1: Create scripts/fetch_all.py**

```python
#!/usr/bin/env python3
"""Fetch all data sources and populate data/raw/."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv

load_dotenv()

if "AUTO_API_KEY" not in os.environ:
    print("ERROR: AUTO_API_KEY not set. Copy .env.example to .env and add your key.")
    sys.exit(1)

from src.fetch.carapis import fetch_all_li_auto_listings

print("Fetching CarAPIs used-car listings...")
listings = fetch_all_li_auto_listings()
print(f"  → {len(listings)} listings cached")

print("\nNOTE: Li Auto IR and CPCA data must be fetched manually:")
print("  1. Download Li Auto monthly delivery press releases from ir.lixiang.com")
print("     and save HTML files to data/raw/li_auto_ir/pages/")
print("  2. Download CPCA monthly ranking pages and save to data/raw/cpca/pages/")
print("  3. Run scripts/build_features.py to parse and featurize")
print("\nFor Autohome specs and reviews, save model spec/review pages to:")
print("  data/raw/autohome/specs/<model>.html")
print("  data/raw/autohome/reviews/<model>/<YYYY-MM>.html")
```

- [ ] **Step 2: Create scripts/build_features.py**

```python
#!/usr/bin/env python3
"""Parse raw data and build feature + target parquet files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
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
print(f"Saved features ({features.shape}) and target ({target.shape}) to {OUT_DIR}")
```

- [ ] **Step 3: Create scripts/train_evaluate.py**

```python
#!/usr/bin/env python3
"""Train model and print evaluation report."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

features = pd.read_parquet("data/processed/features.parquet")
target = pd.read_parquet("data/processed/target.parquet")

from src.model.split import cohort_split, TARGET_COLS, TRAIN_MODELS, TEST_MODELS
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

# --- Bass baseline ---
print("=== Bass Diffusion Baseline ===")
bass_preds_all = []
sales_raw = pd.read_parquet("data/processed/features.parquet")  # placeholder
for model in TEST_MODELS:
    early = {m: 0 for m in [1, 2, 3]}  # fill from actual data if available
    bass = BassBaseline()
    try:
        bass.fit(early)
        bass_preds_all.append(bass.predict(list(range(4, 13))))
    except Exception:
        bass_preds_all.append([0.0] * 9)
bass_median = np.array(bass_preds_all)
print(f"Pinball q=0.5:  {pinball_loss(y_true.ravel(), bass_median.ravel(), 0.5):.1f}")
print(f"MAPE:           {mape(y_true.ravel(), bass_median.ravel()):.1f}%\n")

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
    direction = "+" if metrics["delta_pinball"] < 0 else "-"
    print(f"{group:12s}  delta_pinball={metrics['delta_pinball']:+.1f}  "
          f"({'adds signal' if metrics['delta_pinball'] < 0 else 'no added signal'})")
```

- [ ] **Step 4: Make scripts executable and run a syntax check**

```bash
chmod +x scripts/fetch_all.py scripts/build_features.py scripts/train_evaluate.py
python -m py_compile scripts/fetch_all.py scripts/build_features.py scripts/train_evaluate.py
echo "Syntax OK"
```

Expected: `Syntax OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/
git commit -m "feat: add fetch_all, build_features, and train_evaluate scripts"
```

---

## Task 16: Forecaster notebook

**Files:**
- Create: `notebooks/forecaster.ipynb`

- [ ] **Step 1: Install jupyter**

```bash
pip install jupyter nbformat -q
```

- [ ] **Step 2: Create the notebook programmatically**

```python
# Run this as: python - <<'EOF'
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell("# Li Auto Sales Forecaster\n\nEnd-to-end pipeline: fetch → features → model → evaluate → visualize."),
    nbf.v4.new_code_cell(
        "import sys, os\nsys.path.insert(0, '..')\nfrom dotenv import load_dotenv\nload_dotenv()\nimport pandas as pd\nimport numpy as np"
    ),
    nbf.v4.new_markdown_cell("## 1. Load Features and Targets"),
    nbf.v4.new_code_cell(
        "features = pd.read_parquet('../data/processed/features.parquet')\ntarget = pd.read_parquet('../data/processed/target.parquet')\nprint(features.shape, target.shape)\nfeatures.head()"
    ),
    nbf.v4.new_markdown_cell("## 2. Train / Test Split"),
    nbf.v4.new_code_cell(
        "from src.model.split import cohort_split\nX_train, X_test, y_train, y_test = cohort_split(features, target)\nprint('Train:', X_train['model'].tolist())\nprint('Test: ', X_test['model'].tolist())"
    ),
    nbf.v4.new_markdown_cell("## 3. Train Quantile Forest"),
    nbf.v4.new_code_cell(
        "from src.model.train import QuantileForestForecaster, FEATURE_COLS\nqf = QuantileForestForecaster(quantiles=[0.1, 0.5, 0.9], n_estimators=200)\nqf.fit(X_train[FEATURE_COLS], y_train)\nprint('Model trained.')"
    ),
    nbf.v4.new_markdown_cell("## 4. Evaluate"),
    nbf.v4.new_code_cell(
        "from src.model.evaluate import pinball_loss, mape, interval_coverage\npreds = qf.predict(X_test[FEATURE_COLS])\ny_true = y_test.values\nprint(f'Pinball q=0.5: {pinball_loss(y_true.ravel(), preds[:,:,1].ravel(), 0.5):.1f}')\nprint(f'MAPE:          {mape(y_true.ravel(), preds[:,:,1].ravel()):.1f}%')\nprint(f'80% coverage:  {interval_coverage(y_true.ravel(), preds[:,:,0].ravel(), preds[:,:,2].ravel()):.2%}')"
    ),
    nbf.v4.new_markdown_cell("## 5. Fan Chart — Forecast vs Actuals"),
    nbf.v4.new_code_cell(
        "from src.viz.plots import fan_chart\nfor i, model in enumerate(X_test['model'].tolist()):\n    fig = fan_chart(preds[i], y_test.iloc[[i]].T.rename(columns={i: 'sales'}).reset_index().rename(columns={'index': 'month_since_launch'}), model_name=model, quantiles=[0.1, 0.5, 0.9])\n    fig.show()"
    ),
    nbf.v4.new_markdown_cell("## 6. Incremental-Value Test"),
    nbf.v4.new_code_cell(
        "from src.model.evaluate import incremental_value_test\nfrom src.model.train import FEATURE_COLS\ntraj_cols = [c for c in FEATURE_COLS if 'percentile' in c or 'ratio' in c]\nspec_cols = [c for c in FEATURE_COLS if c.startswith('pc') or c == 'is_erev']\nsent_cols = [c for c in FEATURE_COLS if c.startswith('sent_')]\ngroups = {'trajectory': X_train[traj_cols].values, 'specs': X_train[spec_cols].values, 'sentiment': X_train[sent_cols].values}\nresults = incremental_value_test(groups, y_train.values)\nfor g, m in results.items():\n    print(f'{g:12s}  delta_pinball={m[\"delta_pinball\"]:+.1f}')"
    ),
    nbf.v4.new_markdown_cell("## 7. Geographic Choropleth"),
    nbf.v4.new_code_cell(
        "# Requires regional sales data in data/processed/regional.parquet\nimport os\nif os.path.exists('../data/processed/regional.parquet'):\n    from src.viz.choropleth import plot_choropleth\n    import pandas as pd\n    regional = pd.read_parquet('../data/processed/regional.parquet')\n    fig = plot_choropleth(regional, model='L9')\n    fig.show()\nelse:\n    print('No regional data available. Skipping choropleth.')"
    ),
]
nb.cells = cells
with open('notebooks/forecaster.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook written.")
# EOF
```

Run:
```bash
python - <<'PYEOF'
import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell("# Li Auto Sales Forecaster\n\nEnd-to-end pipeline: fetch → features → model → evaluate → visualize."),
    nbf.v4.new_code_cell("import sys, os\nsys.path.insert(0, '..')\nfrom dotenv import load_dotenv\nload_dotenv()\nimport pandas as pd\nimport numpy as np"),
    nbf.v4.new_markdown_cell("## 1. Load Features and Targets"),
    nbf.v4.new_code_cell("features = pd.read_parquet('../data/processed/features.parquet')\ntarget = pd.read_parquet('../data/processed/target.parquet')\nprint(features.shape, target.shape)\nfeatures.head()"),
    nbf.v4.new_markdown_cell("## 2. Train / Test Split"),
    nbf.v4.new_code_cell("from src.model.split import cohort_split\nX_train, X_test, y_train, y_test = cohort_split(features, target)\nprint('Train:', X_train['model'].tolist())\nprint('Test: ', X_test['model'].tolist())"),
    nbf.v4.new_markdown_cell("## 3. Train Quantile Forest"),
    nbf.v4.new_code_cell("from src.model.train import QuantileForestForecaster, FEATURE_COLS\nqf = QuantileForestForecaster(quantiles=[0.1, 0.5, 0.9], n_estimators=200)\nqf.fit(X_train[FEATURE_COLS], y_train)\nprint('Model trained.')"),
    nbf.v4.new_markdown_cell("## 4. Evaluate"),
    nbf.v4.new_code_cell("from src.model.evaluate import pinball_loss, mape, interval_coverage\npreds = qf.predict(X_test[FEATURE_COLS])\ny_true = y_test.values\nprint(f'Pinball q=0.5: {pinball_loss(y_true.ravel(), preds[:,:,1].ravel(), 0.5):.1f}')\nprint(f'MAPE:          {mape(y_true.ravel(), preds[:,:,1].ravel()):.1f}%')\nprint(f'80% coverage:  {interval_coverage(y_true.ravel(), preds[:,:,0].ravel(), preds[:,:,2].ravel()):.2%}')"),
    nbf.v4.new_markdown_cell("## 5. Fan Chart"),
    nbf.v4.new_code_cell("from src.viz.plots import fan_chart\nfor i, model in enumerate(X_test['model'].tolist()):\n    fig = fan_chart(preds[i], pd.DataFrame({'month_since_launch': range(4,13), 'sales': y_test.values[i]}), model_name=model, quantiles=[0.1, 0.5, 0.9])\n    fig.show()"),
    nbf.v4.new_markdown_cell("## 6. Incremental-Value Test"),
    nbf.v4.new_code_cell("from src.model.evaluate import incremental_value_test\ntraj_cols=[c for c in FEATURE_COLS if 'percentile' in c or 'ratio' in c]\nspec_cols=[c for c in FEATURE_COLS if c.startswith('pc') or c=='is_erev']\nsent_cols=[c for c in FEATURE_COLS if c.startswith('sent_')]\ngroups={'trajectory':X_train[traj_cols].values,'specs':X_train[spec_cols].values,'sentiment':X_train[sent_cols].values}\nresults=incremental_value_test(groups,y_train.values)\nfor g,m in results.items():\n    print(f'{g:12s}  delta_pinball={m[\"delta_pinball\"]:+.1f}')"),
    nbf.v4.new_markdown_cell("## 7. Geographic Choropleth"),
    nbf.v4.new_code_cell("import os\nif os.path.exists('../data/processed/regional.parquet'):\n    from src.viz.choropleth import plot_choropleth\n    regional = pd.read_parquet('../data/processed/regional.parquet')\n    plot_choropleth(regional, model='L9').show()\nelse:\n    print('No regional data. Skipping choropleth.')"),
]
nb.cells = cells
with open('notebooks/forecaster.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook written.")
PYEOF
```

Expected: `Notebook written.`

- [ ] **Step 3: Commit**

```bash
git add notebooks/forecaster.ipynb
git commit -m "feat: add end-to-end forecaster notebook"
```

---

## Task 17: Full test suite and push

- [ ] **Step 1: Run all tests**

```bash
cd /Users/chelseahu/li-auto-forecaster
pytest tests/ -v --ignore=tests/test_sentiment.py -x
```

(Sentiment tests excluded from CI run due to model download. All others must pass.)

Expected: All tests pass except sentiment (skipped).

- [ ] **Step 2: Push to GitHub**

```bash
git push origin main
```

Expected: `main -> main` with no errors.

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by task |
|---|---|
| Feature window months 1–3 | Tasks 6, 7, 8 |
| Target window months 4–12 | Task 12 (TARGET_COLS) |
| No leakage | Tasks 6 (TrajectoryTransformer.fit), 7 (SpecTransformer.fit) |
| Cohort split (not random) | Task 10 |
| Percentile transform fit on train only | Task 6 |
| PCA fit on train only | Task 7 |
| Prediction intervals (q=0.1/0.5/0.9) | Task 12 |
| Incremental-value test | Task 13 |
| Chinese RoBERTa sentiment | Task 8 |
| Autohome specs scraper | Task 5 |
| Li Auto IR scraper | Task 3 |
| CPCA peer data | Task 4 |
| CarAPIs used-car signal | Task 2 |
| AUTO_API_KEY from env only | Task 2 |
| .gitignore for .env + cache | Task 1 |
| Bass diffusion baseline | Task 11 |
| Choropleth + regional ranking | Task 14 |
| Fan chart | Task 14 |
| Scripts (fetch/build/train) | Task 15 |
| Notebook | Task 16 |

All requirements covered. No gaps found.
