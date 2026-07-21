"""Quick demo runner for the scalping SignalGenerator.
Generates synthetic price data, runs the signal generator, and writes signals.csv for inspection.
"""
import pandas as pd
import numpy as np
from signal import SignalGenerator


def generate_synthetic_prices(n=500, start=100.0, seed=42):
    np.random.seed(seed)
    steps = np.random.normal(loc=0.0, scale=0.5, size=n)
    price = start + np.cumsum(steps)
    idx = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='T')
    return pd.DataFrame({'close': price}, index=idx)


def run_demo():
    df = generate_synthetic_prices()
    sg = SignalGenerator(ema_fast=5, ema_slow=20, rsi_period=14, rsi_long=60, rsi_exit=80)
    out = sg.generate_signals(df)
    out.to_csv('signals.csv')
    # print summary
    entries = (out['signal'] == 1).sum()
    shorts = (out['signal'] == -1).sum()
    print(f'Generated {len(out)} rows — long entries: {entries}, short entries: {shorts}')
    print('Sample signals:')
    print(out[['close','ema_fast','ema_slow','rsi','signal']].tail(20))


if __name__ == '__main__':
    run_demo()
