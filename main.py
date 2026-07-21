"""Quick demo runner for the scalping SignalGenerator.
Generates synthetic price data, runs the signal generator, runs a simple backtest, and writes outputs for inspection.
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
    sg = SignalGenerator(ema_fast=5, ema_slow=20, rsi_period=14, rsi_long=60, rsi_exit=80,
                         atr_period=14, sl_atr=1.5, tp_atr=3.0, max_holding_bars=240)

    signals = sg.generate_signals(df)
    signals.to_csv('signals.csv')

    trades = sg.simulate_trades(df)
    if not trades.empty:
        trades.to_csv('trades.csv', index=False)

    # summary
    entries = (signals['signal'] == 1).sum()
    shorts = (signals['signal'] == -1).sum()
    print(f'Generated {len(signals)} rows — long entries: {entries}, short entries: {shorts}')
    print('Sample signals:')
    print(signals[['close','ema_fast','ema_slow','rsi','atr','signal']].tail(10))

    print('\nBacktest summary:')
    if trades.empty:
        print('No trades generated')
    else:
        total_pnl = trades['pnl'].sum()
        win_rate = (trades['pnl'] > 0).mean()
        avg_ret = trades['return'].mean()
        print(f"Trades: {len(trades)}, Total PnL: {total_pnl:.4f}, Win rate: {win_rate:.2%}, Avg return: {avg_ret:.2%}")


if __name__ == '__main__':
    run_demo()
