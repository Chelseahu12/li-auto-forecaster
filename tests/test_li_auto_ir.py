import json
import time
from pathlib import Path
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
    assert df["sales"].dtype in [int, "int64"]
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
