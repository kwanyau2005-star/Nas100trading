"""Quick runner for the scalping SignalGenerator.

Supports two data sources:
- demo: synthetic random-walk prices (default)
- itick: remote futures klines from configured itick endpoint
"""
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from itick_api import ItickAPI, ItickAPIError
from signal_generator import SignalGenerator


def generate_synthetic_prices(n=500, start=100.0, seed=42):
    np.random.seed(seed)
    steps = np.random.normal(loc=0.0, scale=0.5, size=n)
    price = start + np.cumsum(steps)
    idx = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='min')
    return pd.DataFrame({'close': price}, index=idx)


def load_prices():
    source = os.getenv('DATA_SOURCE', 'demo').strip().lower()
    if source == 'itick':
        api = ItickAPI.from_env()
        symbol = os.getenv('ITICK_SYMBOL', 'NAS100')
        interval = os.getenv('ITICK_INTERVAL', '1m')
        limit = int(os.getenv('ITICK_LIMIT', '500'))
        return api.fetch_klines(symbol=symbol, interval=interval, limit=limit), source
    return generate_synthetic_prices(), 'demo'


def run_demo():
    load_dotenv()
    try:
        df, source = load_prices()
    except ItickAPIError as exc:
        raise SystemExit(f'Failed to load itick data: {exc}') from exc

    sg = SignalGenerator(ema_fast=5, ema_slow=20, rsi_period=14, rsi_long=60, rsi_exit=80)
    out = sg.generate_signals(df)
    out.to_csv('signals.csv')
    # print summary
    entries = (out['signal'] == 1).sum()
    shorts = (out['signal'] == -1).sum()
    print(f'Data source: {source}')
    print(f'Generated {len(out)} rows — long entries: {entries}, short entries: {shorts}')
    print('Sample signals:')
    print(out[['close','ema_fast','ema_slow','rsi','signal']].tail(20))


if __name__ == '__main__':
    run_demo()
