import os
import time

from dotenv import load_dotenv

from live_monitor import (
    DeepchartsNQFeed,
    PropFirmRiskManager,
    PropFirmRiskRules,
    RealtimeSignalMonitor,
    TelegramNotifier,
)
from signal import SignalGenerator


def build_monitor_from_env() -> RealtimeSignalMonitor:
    load_dotenv()

    base_url = os.getenv("DEEPCHARTS_KLINE_URL", "").strip()
    if not base_url:
        raise ValueError("DEEPCHARTS_KLINE_URL is required")

    feed = DeepchartsNQFeed(
        base_url=base_url,
        symbol=os.getenv("DEEPCHARTS_SYMBOL", "NQ"),
        interval=os.getenv("DEEPCHARTS_INTERVAL", "1m"),
        timeout=int(os.getenv("DEEPCHARTS_TIMEOUT", "10")),
        retries=int(os.getenv("DEEPCHARTS_RETRIES", "3")),
    )
    signal_generator = SignalGenerator(
        ema_fast=int(os.getenv("EMA_FAST", "5")),
        ema_slow=int(os.getenv("EMA_SLOW", "20")),
        rsi_period=int(os.getenv("RSI_PERIOD", "14")),
        rsi_long=int(os.getenv("RSI_LONG", "60")),
        rsi_exit=int(os.getenv("RSI_EXIT", "80")),
    )
    risk_rules = PropFirmRiskRules(
        max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "1000")),
        max_drawdown=float(os.getenv("MAX_DRAWDOWN", "2000")),
        max_positions=int(os.getenv("MAX_POSITIONS", "1")),
    )
    risk_manager = PropFirmRiskManager(rules=risk_rules)
    notifier = TelegramNotifier(
        token=os.getenv("TELEGRAM_TOKEN"),
        chat_id=os.getenv("TELEGRAM_CHAT_ID"),
    )
    return RealtimeSignalMonitor(
        feed=feed,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        notifier=notifier,
        lookback=int(os.getenv("KLINE_LOOKBACK", "300")),
    )


def main():
    monitor = build_monitor_from_env()
    poll_seconds = int(os.getenv("POLL_SECONDS", "5"))
    print("Realtime NQ monitor started.")
    while True:
        event = monitor.run_once()
        if event:
            print(event)
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
