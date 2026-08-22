import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol, Tuple
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


def _to_datetime(value) -> pd.Timestamp:
    if isinstance(value, (int, float)):
        if value > 1e12:
            return pd.to_datetime(value, unit="ms", utc=True)
        return pd.to_datetime(value, unit="s", utc=True)
    return pd.to_datetime(value, utc=True)


class DeepchartsNQFeed:
    """Polling data feed for Deepcharts-style K-line endpoints."""

    def __init__(
        self,
        base_url: str,
        symbol: str = "NQ",
        interval: str = "1m",
        timeout: int = 10,
        retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        self.base_url = base_url
        self.symbol = symbol
        self.interval = interval
        self.timeout = timeout
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds

    def fetch_latest(self, limit: int = 300) -> pd.DataFrame:
        params = urlencode(
            {"symbol": self.symbol, "interval": self.interval, "limit": limit}
        )
        url = f"{self.base_url}?{params}"
        last_error = None
        for _ in range(self.retries):
            try:
                req = Request(url, headers={"Accept": "application/json"})
                with urlopen(req, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return self._parse_payload(payload)
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                last_error = exc
                time.sleep(self.retry_delay_seconds)
        raise RuntimeError(f"Unable to fetch K-lines from Deepcharts endpoint: {last_error}")

    def _parse_payload(self, payload) -> pd.DataFrame:
        candles = payload.get("candles", payload) if isinstance(payload, dict) else payload
        if not isinstance(candles, list):
            raise ValueError("Deepcharts response must contain a candle list")

        rows = []
        for candle in candles:
            ts = candle.get("timestamp", candle.get("time", candle.get("t")))
            close = candle.get("close", candle.get("c"))
            if ts is None or close is None:
                continue
            closed = candle.get("is_closed", candle.get("closed", candle.get("x", True)))
            rows.append(
                {
                    "timestamp": _to_datetime(ts),
                    "close": float(close),
                    "is_closed": bool(closed),
                }
            )

        if not rows:
            return pd.DataFrame(columns=["close", "is_closed"])

        df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df = df.set_index("timestamp")
        return df[["close", "is_closed"]]


@dataclass
class PropFirmRiskRules:
    max_daily_loss: float = 1_000.0
    max_drawdown: float = 2_000.0
    max_positions: int = 1


class PropFirmRiskManager:
    def __init__(
        self,
        rules: Optional[PropFirmRiskRules] = None,
        starting_equity: float = 100_000.0,
    ):
        self.rules = rules or PropFirmRiskRules()
        self.starting_equity = starting_equity
        self.peak_equity = starting_equity
        self.current_equity = starting_equity
        self.open_positions = 0
        self.daily_pnl = 0.0
        self._pnl_date = datetime.now(timezone.utc).date()

    def _ensure_day(self, now: Optional[datetime]) -> None:
        date = (now or datetime.now(timezone.utc)).date()
        if date != self._pnl_date:
            self._pnl_date = date
            self.daily_pnl = 0.0

    def update_equity(self, equity: float, now: Optional[datetime] = None) -> None:
        self._ensure_day(now)
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def register_closed_trade(self, pnl: float, now: Optional[datetime] = None) -> None:
        self._ensure_day(now)
        self.daily_pnl += pnl
        self.update_equity(self.current_equity + pnl, now)
        if self.open_positions > 0:
            self.open_positions -= 1

    def register_open_trade(self) -> None:
        self.open_positions += 1

    def can_open_long(self, now: Optional[datetime] = None) -> Tuple[bool, str]:
        self._ensure_day(now)
        if self.open_positions >= self.rules.max_positions:
            return False, "Position limit reached"
        if -self.daily_pnl >= self.rules.max_daily_loss:
            return False, "Daily loss limit reached"
        drawdown = self.peak_equity - self.current_equity
        if drawdown >= self.rules.max_drawdown:
            return False, "Max drawdown reached"
        return True, "OK"


class TelegramNotifier:
    def __init__(self, token: Optional[str], chat_id: Optional[str]):
        self.token = token
        self.chat_id = chat_id

    def send(self, message: str) -> bool:
        if not self.token or not self.chat_id:
            return False
        endpoint = f"https://api.telegram.org/bot{self.token}/sendMessage"
        body = json.dumps({"chat_id": self.chat_id, "text": message}).encode("utf-8")
        req = Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=10):
                return True
        except Exception as exc:
            logging.warning("Telegram send failed: %s", exc)
            return False

    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)


class FeedLike(Protocol):
    def fetch_latest(self, limit: int) -> pd.DataFrame:
        ...


class SignalGeneratorLike(Protocol):
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class NotifierLike(Protocol):
    def send(self, message: str) -> bool:
        ...


class RealtimeSignalMonitor:
    def __init__(
        self,
        feed: FeedLike,
        signal_generator: SignalGeneratorLike,
        risk_manager: PropFirmRiskManager,
        notifier: Optional[NotifierLike] = None,
        lookback: int = 300,
    ):
        self.feed = feed
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.notifier = notifier
        self.lookback = lookback
        self.last_closed_ts = None

    def run_once(self) -> Optional[Dict[str, Any]]:
        """Process the latest closed candle once and return the resulting event."""
        candles = self.feed.fetch_latest(self.lookback)
        if candles.empty:
            return None
        closed = candles[candles["is_closed"]]
        if closed.empty:
            return None

        latest_ts = closed.index[-1]
        if self.last_closed_ts is not None and latest_ts <= self.last_closed_ts:
            return None

        signal_df = self.signal_generator.generate_signals(closed[["close"]])
        latest = signal_df.iloc[-1]
        signal = int(latest["signal"])
        event = {
            "timestamp": latest_ts.isoformat(),
            "signal": signal,
            "price": float(latest["close"]),
            "status": "no_signal",
        }

        if signal == 1:
            allowed, reason = self.risk_manager.can_open_long()
            if allowed:
                self.risk_manager.register_open_trade()
                message = f"BUY NQ signal @ {event['price']} ({event['timestamp']})"
                event["status"] = "triggered"
            else:
                message = f"BUY NQ blocked by risk rules: {reason}"
                event["status"] = "blocked"
                event["reason"] = reason
            if self.notifier:
                self.notifier.send(message)

        self.last_closed_ts = latest_ts
        return event

    def run_forever(self, poll_seconds: int = 5):
        while True:
            try:
                self.run_once()
            except Exception as exc:
                logging.warning("Realtime monitor poll failed: %s", exc)
            time.sleep(poll_seconds)
