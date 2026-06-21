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
print(f"  -> {len(listings)} listings cached")

print("\nNOTE: Li Auto IR and CPCA data must be fetched manually:")
print("  1. Download Li Auto monthly delivery press releases from ir.lixiang.com")
print("     and save HTML files to data/raw/li_auto_ir/pages/")
print("  2. Download CPCA monthly ranking pages and save to data/raw/cpca/pages/")
print("  3. Run scripts/build_features.py to parse and featurize")
print("\nFor Autohome specs and reviews, save model spec/review pages to:")
print("  data/raw/autohome/specs/<model>.html")
print("  data/raw/autohome/reviews/<model>/<YYYY-MM>.html")
