"""Tradovate futures API provider."""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd


class TradovateAPIError(RuntimeError):
    """Raised when Tradovate auth/data requests fail."""


class TradovateAPI:
    def __init__(
        self,
        base_url: str,
        auth_endpoint: str,
        chart_endpoint: str,
        username: str | None = None,
        password: str | None = None,
        app_id: str | None = None,
        app_version: str | None = None,
        cid: str | None = None,
        sec: str | None = None,
        device_id: str | None = None,
        access_token: str | None = None,
        timeout: int = 15,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.auth_endpoint = auth_endpoint.lstrip("/")
        self.chart_endpoint = chart_endpoint.lstrip("/")
        self.username = username
        self.password = password
        self.app_id = app_id
        self.app_version = app_version
        self.cid = cid
        self.sec = sec
        self.device_id = device_id
        self.access_token = access_token
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "TradovateAPI":
        return cls(
            os.getenv("TRADOVATE_BASE_URL", "https://demo-api.tradovate.com"),
            os.getenv("TRADOVATE_AUTH_ENDPOINT", "/v1/auth/accesstokenrequest"),
            os.getenv("TRADOVATE_CHART_ENDPOINT", "/md/getchart"),
            os.getenv("TRADOVATE_USERNAME"),
            os.getenv("TRADOVATE_PASSWORD"),
            app_id=os.getenv("TRADOVATE_APP_ID"),
            app_version=os.getenv("TRADOVATE_APP_VERSION"),
            cid=os.getenv("TRADOVATE_CID"),
            sec=os.getenv("TRADOVATE_SECRET"),
            device_id=os.getenv("TRADOVATE_DEVICE_ID"),
            access_token=os.getenv("TRADOVATE_ACCESS_TOKEN"),
            timeout=int(os.getenv("TRADOVATE_TIMEOUT_SECONDS", "15")),
        )

    def _request(self, endpoint: str, method: str = "GET", payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        data = None
        req_headers = {"Accept": "application/json"}
        if headers:
            req_headers.update(headers)
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        url = urljoin(self.base_url, endpoint.lstrip("/"))
        req = Request(url=url, data=data, headers=req_headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as exc:
            raise TradovateAPIError(f"Tradovate request failed with HTTP {exc.code}.") from exc
        except URLError as exc:
            raise TradovateAPIError(f"Tradovate request failed: {exc.reason}.") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TradovateAPIError("Tradovate response is not valid JSON.") from exc

    @staticmethod
    def _parse_interval(interval: str) -> int:
        text = str(interval or "1m").strip().lower()
        if text.endswith("m"):
            text = text[:-1]
        try:
            minutes = int(text)
        except ValueError as exc:
            raise TradovateAPIError("Tradovate interval numeric value is invalid (examples: '1m', '5m').") from exc
        if minutes <= 0:
            raise TradovateAPIError("Tradovate interval must be greater than zero.")
        return minutes

    def _get_access_token(self) -> str:
        if self.access_token:
            return self.access_token
        if not self.username or not self.password:
            raise TradovateAPIError("Tradovate username/password or TRADOVATE_ACCESS_TOKEN is required.")

        payload: dict[str, Any] = {"name": self.username, "password": self.password}
        optional = {
            "appId": self.app_id,
            "appVersion": self.app_version,
            "cid": self.cid,
            "sec": self.sec,
            "deviceId": self.device_id,
        }
        payload.update({k: v for k, v in optional.items() if v})
        auth = self._request(self.auth_endpoint, method="POST", payload=payload)
        token = auth.get("accessToken") if isinstance(auth, dict) else None
        if not token:
            raise TradovateAPIError("Tradovate auth response missing accessToken.")
        self.access_token = token
        return token

    def fetch_klines(self, symbol: str, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
        if not symbol:
            raise TradovateAPIError("Tradovate symbol is required.")
        minutes = self._parse_interval(interval)

        token = self._get_access_token()
        payload = {
            "symbol": symbol,
            "chartDescription": {"underlyingType": "MinuteBar", "elementSize": minutes},
            "timeRange": {"asMuchAsElements": int(limit)},
        }
        data = self._request(
            self.chart_endpoint,
            method="POST",
            payload=payload,
            headers={"Authorization": "Bearer " + token},
        )
        records = self._extract_records(data)
        if not records:
            raise TradovateAPIError("Tradovate returned empty candle data.")

        df = self._to_dataframe(records)
        if "close" not in df.columns:
            raise TradovateAPIError("Tradovate candle data must include 'close'.")

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            raise TradovateAPIError("Tradovate candle data has no valid close values.")

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            if ts.notna().any():
                df.index = ts
        return df

    @staticmethod
    def _extract_records(payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("bars", "data", "candles", "klines", "items", "rows"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
        raise TradovateAPIError("Unable to locate candle list in Tradovate response.")

    @staticmethod
    def _to_dataframe(records: list[Any]) -> pd.DataFrame:
        first = records[0]
        if isinstance(first, dict):
            return pd.DataFrame(records)
        if isinstance(first, (list, tuple)):
            cols = ["timestamp", "open", "high", "low", "close", "volume"]
            width = len(first)
            if width < 5:
                raise TradovateAPIError("Tradovate array candle format needs at least 5 columns.")
            return pd.DataFrame(records, columns=cols[:width])
        raise TradovateAPIError("Unsupported Tradovate candle format.")
