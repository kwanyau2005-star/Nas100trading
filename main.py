
# main.py
# Integrates ws_handler tick aggregation (if available) + polling fallback + news buffer + signal logic

import os
import asyncio
import time
import datetime as dt
from threading import Thread
import pandas as pd
from ws_handler import TickAggregator, connect_and_aggregate
from signal_demo import generate_signal, alert, fetch_recent_minutes, fetch_recent_news

SYMBOL = os.getenv('SYMBOL', 'QQQ.US')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '15'))
USE_WS = os.getenv('USE_WS', 'true').lower() in ('1','true','yes')

# global news buffer is in webhook_server.NEWS_BUFFER but we fallback to polling if not using webhook

# run websocket client in background
def start_ws_agg(agg):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(connect_and_aggregate(agg, symbol_filter=SYMBOL))


def main_loop():
    last_checked_minute = None
    agg = TickAggregator()

    # start ws thread if enabled
    if USE_WS:
        t = Thread(target=start_ws_agg, args=(agg,), daemon=True)
        t.start()
        print('WS aggregator thread started')

    while True:
        try:
            # attempt to get aggregated bars from tick aggregator
            bars = agg.aggregate_minute()
            df = None
            if bars is not None and not bars.empty:
                df = bars.sort_values('minute')
                # rename to expected columns
                df = df.rename(columns={'minute': 'trade_time', 'open':'open','high':'high','low':'low','close':'close','volume':'volume'})
            else:
                # fallback to polling 1-min bars
                df = fetch_recent_minutes(SYMBOL, minutes=60)
            if df is None or df.empty:
                print(dt.datetime.utcnow(), 'no data this cycle')
                time.sleep(POLL_INTERVAL)
                continue

            # determine latest minute
            if 'trade_time' in df.columns:
                latest_time = pd.to_datetime(df['trade_time'].iloc[-1])
            else:
                latest_time = dt.datetime.utcnow().replace(second=0, microsecond=0)

            if last_checked_minute is None or latest_time > last_checked_minute:
                last_checked_minute = latest_time
                # prepare df for signal_demo.generate_signal
                df_proc = df.copy()
                # ensure lowercase columns
                df_proc.columns = [c.lower() for c in df_proc.columns]
                cols = ['open','high','low','close','volume']
                for c in cols:
                    if c not in df_proc.columns:
                        df_proc[c] = 0
                df_proc = df_proc[cols].astype(float)
                # fetch news from polling method as fallback
                news = fetch_recent_news(minutes=10)
                sig = generate_signal(df_proc, latest_news=news)
                if sig.get('signal'):
                    alert(sig)
                else:
                    print(dt.datetime.utcnow(), 'no signal:', sig.get('reason'))
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            print('main loop exception', e)
            time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main_loop()
