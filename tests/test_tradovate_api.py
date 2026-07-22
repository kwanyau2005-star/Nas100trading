import json
from unittest.mock import patch

import pytest

from tradovate_api import TradovateAPI, TradovateAPIError


class _FakeResponse:
    def __init__(self, payload: dict | list):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_fetch_klines_auth_then_chart_normalizes_close():
    auth_payload = {"accessToken": "token123"}
    chart_payload = {
        "bars": [
            {"timestamp": "2026-01-01T00:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": "1.2"},
            {"timestamp": "2026-01-01T00:01:00Z", "open": 1.2, "high": 2, "low": 1, "close": "bad"},
        ]
    }
    api = TradovateAPI(
        base_url="https://demo-api.tradovate.com",
        auth_endpoint="/v1/auth/accesstokenrequest",
        chart_endpoint="/md/getchart",
        username="user",
        **{"password": "pw123"},
    )

    with patch("tradovate_api.urlopen", side_effect=[_FakeResponse(auth_payload), _FakeResponse(chart_payload)]):
        out = api.fetch_klines(symbol="MNQ", interval="1m", limit=2)

    assert len(out) == 1
    assert float(out["close"].iloc[0]) == 1.2


def test_fetch_klines_uses_existing_access_token_without_auth():
    chart_payload = {"bars": [[1710000000, 10, 12, 9, 11, 100]]}
    api = TradovateAPI(
        base_url="https://demo-api.tradovate.com",
        auth_endpoint="/v1/auth/accesstokenrequest",
        chart_endpoint="/md/getchart",
        access_token="ready-token",
    )

    with patch("tradovate_api.urlopen", return_value=_FakeResponse(chart_payload)) as mocked:
        out = api.fetch_klines(symbol="MNQ", interval="5m", limit=1)

    assert mocked.call_count == 1
    assert float(out["close"].iloc[0]) == 11


def test_fetch_klines_raises_when_no_auth_credentials():
    api = TradovateAPI(
        base_url="https://demo-api.tradovate.com",
        auth_endpoint="/v1/auth/accesstokenrequest",
        chart_endpoint="/md/getchart",
    )
    with pytest.raises(TradovateAPIError, match="required"):
        api.fetch_klines(symbol="MNQ")
