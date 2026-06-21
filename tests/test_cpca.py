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
