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
