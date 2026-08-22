from datetime import datetime, timezone

import pandas as pd

from live_monitor import PropFirmRiskManager, PropFirmRiskRules, RealtimeSignalMonitor


class DummyFeed:
    def __init__(self, df):
        self.df = df

    def fetch_latest(self, limit):
        return self.df


class DummySignalGenerator:
    def generate_signals(self, df):
        out = df.copy()
        out["signal"] = 0
        out.iat[-1, out.columns.get_loc("signal")] = 1
        return out


class DummyNotifier:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)
        return True


def _closed_candle_frame():
    idx = pd.to_datetime(
        ["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z", "2026-01-01T00:02:00Z"]
    )
    return pd.DataFrame(
        {"close": [100.0, 101.0, 102.0], "is_closed": [True, True, True]}, index=idx
    )


def test_monitor_triggers_buy_once_per_closed_candle():
    risk = PropFirmRiskManager()
    notifier = DummyNotifier()
    monitor = RealtimeSignalMonitor(
        feed=DummyFeed(_closed_candle_frame()),
        signal_generator=DummySignalGenerator(),
        risk_manager=risk,
        notifier=notifier,
    )

    first = monitor.run_once()
    second = monitor.run_once()

    assert first is not None
    assert first["signal"] == 1
    assert first["status"] == "triggered"
    assert risk.open_positions == 1
    assert len(notifier.messages) == 1
    assert second is None


def test_risk_manager_blocks_buy_after_daily_loss_limit():
    rules = PropFirmRiskRules(max_daily_loss=100, max_drawdown=5000, max_positions=1)
    risk = PropFirmRiskManager(rules=rules, starting_equity=100000)
    risk.register_closed_trade(
        pnl=-120, now=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    allowed, reason = risk.can_open_long(now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert allowed is False
    assert reason == "Daily loss limit reached"


def test_monitor_marks_signal_as_blocked_when_risk_rejects():
    rules = PropFirmRiskRules(max_daily_loss=10000, max_drawdown=100, max_positions=1)
    risk = PropFirmRiskManager(rules=rules, starting_equity=100000)
    risk.update_equity(99800)
    notifier = DummyNotifier()
    monitor = RealtimeSignalMonitor(
        feed=DummyFeed(_closed_candle_frame()),
        signal_generator=DummySignalGenerator(),
        risk_manager=risk,
        notifier=notifier,
    )

    event = monitor.run_once()
    assert event["status"] == "blocked"
    assert event["reason"] == "Max drawdown reached"
    assert len(notifier.messages) == 1
