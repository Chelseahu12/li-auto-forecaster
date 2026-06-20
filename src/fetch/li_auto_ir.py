import json
import re
import time
from pathlib import Path

import pandas as pd
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
