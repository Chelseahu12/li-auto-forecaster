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
    """Extract model/month/sales from a Li Auto IR press release page.

    Tries table parsing first (older format, 2022-2023).
    Falls back to prose extraction for newer single-total format.
    """
    soup = BeautifulSoup(html, "lxml")
    rows = []

    # --- Attempt 1: structured table with model-level rows ---
    for table in soup.find_all("table"):
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 3:
                continue
            model_raw, sales_raw, ym_raw = cells[0], cells[1], cells[2]
            num = re.sub(r"[^\d]", "", sales_raw)
            if num:
                rows.append({"model": model_raw, "year_month": ym_raw, "sales": int(num)})
    if rows:
        return pd.DataFrame(rows)

    # --- Attempt 2: extract month/year from URL/title + total from prose ---
    # Derive year_month from page title or canonical URL
    year_month = None
    canonical = soup.find("link", rel="canonical")
    if canonical:
        url = canonical.get("href", "")
        # e.g. ".../li-auto-inc-may-2026-delivery-update"
        month_map = {
            "january": "01", "february": "02", "march": "03", "april": "04",
            "may": "05", "june": "06", "july": "07", "august": "08",
            "september": "09", "october": "10", "november": "11", "december": "12",
        }
        for month_name, month_num in month_map.items():
            m = re.search(rf"{month_name}-(\d{{4}})", url)
            if m:
                year_month = f"{m.group(1)}-{month_num}"
                break

    if not year_month:
        return pd.DataFrame()

    # Extract model-level numbers from prose
    text = soup.get_text(" ")

    # Pattern: "delivered X,XXX [model]" or "[model] delivered X,XXX"
    model_patterns = {
        "L6": r"L[i\s]*6[^\d]*?([\d,]+)\s*(?:units|vehicles|deliveries)",
        "L7": r"L[i\s]*7[^\d]*?([\d,]+)\s*(?:units|vehicles|deliveries)",
        "L8": r"L[i\s]*8[^\d]*?([\d,]+)\s*(?:units|vehicles|deliveries)",
        "L9": r"L[i\s]*9[^\d]*?([\d,]+)\s*(?:units|vehicles|deliveries)",
        "MEGA": r"MEGA[^\d]*?([\d,]+)\s*(?:units|vehicles|deliveries)",
    }

    found_any = False
    for model, pattern in model_patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            sales = int(re.sub(r"[^\d]", "", m.group(1)))
            rows.append({"model": model, "year_month": year_month, "sales": sales})
            found_any = True

    # If no model breakdown found, extract total and store as "TOTAL"
    if not found_any:
        m = re.search(r"delivered\s+([\d,]+)\s+vehicles", text, re.IGNORECASE)
        if m:
            total = int(re.sub(r"[^\d]", "", m.group(1)))
            rows.append({"model": "TOTAL", "year_month": year_month, "sales": total})

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
            if df.empty:
                raise ValueError(
                    "Cached delivery data is empty — the HTML parser found no rows. "
                    "Check that your HTML files match the expected table structure."
                )
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
