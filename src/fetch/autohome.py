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

    def _extract_number(text: str):
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
        elif "座位" in label or ("座" in label and "数" in label):
            specs["seats"] = int(_extract_number(value) or 0)
        elif "指导价" in label or "售价" in label:
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
        date_str = date_el.get_text(strip=True)[:7]
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
