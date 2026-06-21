import json
import re
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

CACHE_DIR = Path("data/raw/cpca")
CACHE_TTL_SECONDS = 7 * 86400

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
