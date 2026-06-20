import json
import time
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from src.fetch.carapis import fetch_listings, LI_AUTO_SOURCES


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
        "fetched_at": time.time() - 8 * 86400,
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
