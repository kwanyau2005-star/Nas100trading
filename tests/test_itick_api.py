import json
from unittest.mock import patch

import pytest

from itick_api import ItickAPI, ItickAPIError


class _FakeResponse:
    def __init__(self, payload: dict | list):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_fetch_klines_dict_records_normalizes_close():
    payload = {
        "data": [
            {"timestamp": "2026-01-01T00:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": "1.5"},
            {"timestamp": "2026-01-01T00:01:00Z", "open": 1.5, "high": 2, "low": 1, "close": "bad"},
        ]
    }
    api = ItickAPI(base_url="https://itick.org", klines_endpoint="/api/klines")

    with patch("itick_api.urlopen", return_value=_FakeResponse(payload)):
        out = api.fetch_klines(symbol="NAS100", interval="1m", limit=2)

    assert "close" in out.columns
    assert len(out) == 1
    assert float(out["close"].iloc[0]) == 1.5


def test_fetch_klines_array_records_maps_columns():
    payload = {"klines": [[1710000000, 10, 12, 9, 11, 100], [1710000060, 11, 12, 10, 10.5, 120]]}
    api = ItickAPI(base_url="https://itick.org", klines_endpoint="/api/klines")

    with patch("itick_api.urlopen", return_value=_FakeResponse(payload)):
        out = api.fetch_klines(symbol="NAS100")

    assert list(out.columns[:5]) == ["timestamp", "open", "high", "low", "close"]
    assert float(out["close"].iloc[-1]) == 10.5


def test_fetch_klines_raises_when_close_missing():
    payload = {"data": [{"timestamp": "2026-01-01T00:00:00Z", "open": 1, "high": 2, "low": 0.5}]}
    api = ItickAPI(base_url="https://itick.org", klines_endpoint="/api/klines")

    with patch("itick_api.urlopen", return_value=_FakeResponse(payload)):
        with pytest.raises(ItickAPIError, match="close"):
            api.fetch_klines(symbol="NAS100")
