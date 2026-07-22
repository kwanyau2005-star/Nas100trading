"""itick futures API provider.

This module isolates network/auth/data normalization so strategy code stays decoupled.
"""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

import pandas as pd


class ItickAPIError(RuntimeError):
    """Raised when itick API requests or payload normalization fails."""


class ItickAPI:
    def __init__(
        self,
        base_url: str,
        klines_endpoint: str,
        api_key: str | None = None,
        api_secret: str | None = None,
        account: str | None = None,
        timeout: int = 15,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.klines_endpoint = klines_endpoint.lstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.account = account
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "ItickAPI":
        return cls(
            base_url=os.getenv("ITICK_BASE_URL", "https://itick.org"),
            klines_endpoint=os.getenv("ITICK_KLINES_ENDPOINT", "/api/v1/futures/klines"),
            api_key=os.getenv("ITICK_API_KEY"),
            api_secret=os.getenv("ITICK_API_SECRET"),
            account=os.getenv("ITICK_ACCOUNT"),
            timeout=int(os.getenv("ITICK_TIMEOUT_SECONDS", "15")),
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-ITICK-API-KEY"] = self.api_key
        if self.api_secret:
            headers["X-ITICK-API-SECRET"] = self.api_secret
        if self.account:
            headers["X-ITICK-ACCOUNT"] = self.account
        return headers

    def fetch_klines(self, symbol: str, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
        if not symbol:
            raise ItickAPIError("ITICK symbol is required.")

        query = urlencode({"symbol": symbol, "interval": interval, "limit": int(limit)})
        url = f"{urljoin(self.base_url, self.klines_endpoint)}?{query}"
        req = Request(url=url, headers=self._headers(), method="GET")

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as exc:
            raise ItickAPIError(f"itick request failed with HTTP {exc.code}.") from exc
        except URLError as exc:
            raise ItickAPIError(f"itick request failed: {exc.reason}.") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ItickAPIError("itick response is not valid JSON.") from exc

        records = self._extract_records(payload)
        if not records:
            raise ItickAPIError("itick returned empty candle data.")

        df = self._to_dataframe(records)
        if "close" not in df.columns:
            raise ItickAPIError("itick candle data must include 'close'.")

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            raise ItickAPIError("itick candle data has no valid close values.")

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
            for key in ("data", "results", "klines", "candles"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
                if key in payload and isinstance(payload[key], dict):
                    nested = payload[key]
                    for nested_key in ("items", "rows", "klines", "candles", "data"):
                        if nested_key in nested and isinstance(nested[nested_key], list):
                            return nested[nested_key]

        raise ItickAPIError("Unable to locate candle list in itick response.")

    @staticmethod
    def _to_dataframe(records: list[Any]) -> pd.DataFrame:
        first = records[0]
        if isinstance(first, dict):
            return pd.DataFrame(records)

        if isinstance(first, (list, tuple)):
            cols = ["timestamp", "open", "high", "low", "close", "volume"]
            width = len(first)
            if width < 5:
                raise ItickAPIError("itick array candle format needs at least 5 columns.")
            return pd.DataFrame(records, columns=cols[:width])

        raise ItickAPIError("Unsupported itick candle format.")
