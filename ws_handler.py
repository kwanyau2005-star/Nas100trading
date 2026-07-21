
# ws_handler.py
# WebSocket client for minidoc rt-bid-ms or etf-mins tick stream

import os
import json
import asyncio
import websockets
import datetime as dt
from collections import deque
import pandas as pd

MINIDOC_WS_URL = os.getenv('MINIDOC_WS_URL', 'wss://minidoc.example/rt-bid-ms?token=')
MINIDOC_TOKEN = os.getenv('MINIDOC_TOKEN', '')

# sliding window for ticks to aggregate to 1-min bars
class TickAggregator:
    def __init__(self):
        self.ticks = []

    def add_tick(self, tick):
        # expected tick fields: {'symbol','price','size','timestamp'} timestamp in ms or us
        self.ticks.append(tick)

    def aggregate_minute(self):
        if not self.ticks:
            return None
        df = pd.DataFrame(self.ticks)
        # convert timestamp
        if 'timestamp' in df.columns:
            # normalize to seconds
            df['ts'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        else:
            df['ts'] = pd.Timestamp.now()
        df = df.sort_values('ts')
        # group by minute
        df['minute'] = df['ts'].dt.floor('T')
        groups = df.groupby('minute')
        bars = []
        for minute, g in groups:
            open_p = g['price'].iloc[0]
            high_p = g['price'].max()
            low_p = g['price'].min()
            close_p = g['price'].iloc[-1]
            vol = g['size'].sum() if 'size' in g.columns else len(g)
            bars.append({'minute': minute, 'open': open_p, 'high': high_p, 'low': low_p, 'close': close_p, 'volume': vol})
        # keep ticks for the last unfinished minute
        last_minute = df['minute'].max()
        self.ticks = df[df['minute'] == last_minute].to_dict('records')
        return pd.DataFrame(bars)


async def connect_and_aggregate(agg: TickAggregator, symbol_filter=None):
    url = MINIDOC_WS_URL
    if MINIDOC_TOKEN and '?' not in url:
        url = url + f"?token={MINIDOC_TOKEN}"
    async with websockets.connect(url) as ws:
        print(dt.datetime.utcnow(), "connected to ws", url)
        async for message in ws:
            try:
                data = json.loads(message)
                # sample structure: {"event": "etf-mins", "symbol": "QQQ.US", "price": 400.12, "size": 100, "timestamp": 169xxx}
                # filter symbol
                if symbol_filter and data.get('symbol') != symbol_filter:
                    continue
                # only handle price ticks
                tick = {'symbol': data.get('symbol'), 'price': data.get('price'), 'size': data.get('size', 0), 'timestamp': data.get('timestamp')}
                agg.add_tick(tick)
            except Exception as e:
                print('ws message error', e)


if __name__ == '__main__':
    agg = TickAggregator()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(connect_and_aggregate(agg, symbol_filter=None))
    except KeyboardInterrupt:
        pass
