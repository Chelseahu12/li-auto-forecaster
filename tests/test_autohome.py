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
