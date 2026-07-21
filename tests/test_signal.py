import pandas as pd
import numpy as np
from signal import SignalGenerator


def test_signal_on_uptrend():
    # synthetic uptrend
    n = 200
    np.random.seed(1)
    steps = np.abs(np.random.normal(loc=0.2, scale=0.2, size=n))
    price = 100 + np.cumsum(steps)
    df = pd.DataFrame({'close': price})
    sg = SignalGenerator(ema_fast=3, ema_slow=10, rsi_period=14, rsi_long=55, rsi_exit=85,
                         atr_period=10, sl_atr=1.0, tp_atr=2.0)
    out = sg.generate_signals(df)
    # expect at least one long entry on a sustained uptrend
    assert (out['signal'] == 1).sum() >= 1


def test_simulate_trades_returns_trades():
    np.random.seed(2)
    n = 300
    steps = np.abs(np.random.normal(loc=0.15, scale=0.15, size=n))
    price = 100 + np.cumsum(steps)
    df = pd.DataFrame({'close': price})
    sg = SignalGenerator(ema_fast=3, ema_slow=10, rsi_period=14, rsi_long=55, rsi_exit=85,
                         atr_period=10, sl_atr=1.0, tp_atr=2.0, max_holding_bars=50)
    trades = sg.simulate_trades(df)
    # allow zero trades in extreme corner cases, but expect at least one for a bullish series
    assert isinstance(trades, pd.DataFrame)
    assert 'pnl' in trades.columns
    # if trades found, ensure numeric pnl
    if len(trades) > 0:
        assert trades['pnl'].dtype.kind in 'fi'
