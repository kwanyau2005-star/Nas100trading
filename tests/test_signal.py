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
    sg = SignalGenerator(ema_fast=3, ema_slow=10, rsi_period=14, rsi_long=55, rsi_exit=85)
    out = sg.generate_signals(df)
    # expect at least one long entry on a sustained uptrend
    assert (out['signal'] == 1).sum() >= 1
