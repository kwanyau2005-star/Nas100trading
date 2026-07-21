import pandas as pd
import numpy as np
from typing import List

class SignalGenerator:
    """Scalping signal generator with EMA/RSI entries and ATR + key-level SL/TP.

    Features:
    - EMA crossover + RSI threshold for entries
    - ATR proxy (rolling abs diff) for volatility
    - Stop-loss / take-profit in multiples of ATR
    - simulate_trades to run a simple first-touch backtest
    """

    def __init__(
        self,
        ema_fast=5,
        ema_slow=20,
        rsi_period=14,
        rsi_long=60,
        rsi_exit=80,
        atr_period=14,
        sl_atr=1.5,
        tp_atr=3.0,
        max_holding_bars=240,
        key_lookback=120,
        key_buffer_atr=0.2,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_long = rsi_long
        self.rsi_exit = rsi_exit
        self.atr_period = atr_period
        self.sl_atr = sl_atr
        self.tp_atr = tp_atr
        self.max_holding_bars = max_holding_bars
        self.key_lookback = key_lookback
        self.key_buffer_atr = key_buffer_atr

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        price = df['close']
        df = df.copy()
        df['ema_fast'] = price.ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = price.ewm(span=self.ema_slow, adjust=False).mean()

        # RSI (simple smoothed method)
        delta = price.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(self.rsi_period, min_periods=1).mean()
        ma_down = down.rolling(self.rsi_period, min_periods=1).mean()
        rs = ma_up / (ma_down.replace(0, np.nan))
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)

        # ATR proxy: rolling mean of absolute close changes
        df['tr'] = price.diff().abs()
        df['atr'] = df['tr'].rolling(self.atr_period, min_periods=1).mean()

        return df

    def _find_key_levels(self, df: pd.DataFrame, i: int, entry_price: float):
        if i <= 2:
            return None, None

        start = max(0, i - self.key_lookback)
        window = df.iloc[start:i].copy()
        if len(window) < 5:
            return None, None

        if 'high' not in window.columns:
            window['high'] = window['close']
        if 'low' not in window.columns:
            window['low'] = window['close']

        # Simple local pivots (1-bar neighborhood)
        pivot_lows = window[(window['low'].shift(1) > window['low']) & (window['low'].shift(-1) > window['low'])]['low']
        pivot_highs = window[(window['high'].shift(1) < window['high']) & (window['high'].shift(-1) < window['high'])]['high']

        support = None
        resistance = None

        lows_below = pivot_lows[pivot_lows < entry_price]
        highs_above = pivot_highs[pivot_highs > entry_price]
        if len(lows_below) > 0:
            support = float(lows_below.max())
        if len(highs_above) > 0:
            resistance = float(highs_above.min())

        # Fallback quantile keys when pivots are sparse
        if support is None:
            q_low = window['low'].quantile(0.25)
            if q_low < entry_price:
                support = float(q_low)
        if resistance is None:
            q_high = window['high'].quantile(0.75)
            if q_high > entry_price:
                resistance = float(q_high)

        return support, resistance

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return dataframe with indicators and 'signal' column: 1 long entry, -1 short entry, 0 neutral/exit."""
        df = self.compute_indicators(df)
        df['signal'] = 0

        long = False
        short = False
        for i in range(len(df)):
            ema_f = df.iloc[i]['ema_fast']
            ema_s = df.iloc[i]['ema_slow']
            rsi = df.iloc[i]['rsi']

            if not long and not short:
                if ema_f > ema_s and rsi < self.rsi_long:
                    df.iat[i, df.columns.get_loc('signal')] = 1
                    long = True
                elif ema_f < ema_s and rsi > (100 - self.rsi_long):
                    df.iat[i, df.columns.get_loc('signal')] = -1
                    short = True
            elif long:
                # exit long
                if ema_f < ema_s or rsi > self.rsi_exit:
                    df.iat[i, df.columns.get_loc('signal')] = 0
                    long = False
            elif short:
                if ema_f > ema_s or rsi < (100 - self.rsi_exit):
                    df.iat[i, df.columns.get_loc('signal')] = 0
                    short = False

        return df

    def simulate_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple first-touch simulator using close prices and ATR for SL/TP.

        Returns a DataFrame of trades with columns: side, entry_idx, exit_idx, entry_price, exit_price, pnl, return
        """
        signals = self.generate_signals(df)
        trades: List[dict] = []

        position = None
        entry_idx = None
        entry_price = None
        sl = None
        tp = None
        hold_counter = 0

        for i in range(len(signals)):
            row = signals.iloc[i]
            price = row['close']
            atr = max(row.get('atr', 0.0), 1e-9)

            # open
            if position is None and row['signal'] == 1:
                position = 'long'
                entry_idx = signals.index[i]
                entry_price = price
                atr_sl = entry_price - self.sl_atr * atr
                atr_tp = entry_price + self.tp_atr * atr
                support, resistance = self._find_key_levels(signals, i, entry_price)

                if support is not None:
                    sl = min(atr_sl, support - self.key_buffer_atr * atr)
                    sl_source = 'key_level'
                else:
                    sl = atr_sl
                    sl_source = 'atr'

                if resistance is not None and resistance > entry_price:
                    tp = max(atr_tp * 0.7 + entry_price * 0.3, resistance)
                    tp_source = 'key_level'
                else:
                    tp = atr_tp
                    tp_source = 'atr'
                hold_counter = 0
                continue
            if position is None and row['signal'] == -1:
                position = 'short'
                entry_idx = signals.index[i]
                entry_price = price
                atr_sl = entry_price + self.sl_atr * atr
                atr_tp = entry_price - self.tp_atr * atr
                support, resistance = self._find_key_levels(signals, i, entry_price)

                if resistance is not None:
                    sl = max(atr_sl, resistance + self.key_buffer_atr * atr)
                    sl_source = 'key_level'
                else:
                    sl = atr_sl
                    sl_source = 'atr'

                if support is not None and support < entry_price:
                    tp = min(atr_tp * 0.7 + entry_price * 0.3, support)
                    tp_source = 'key_level'
                else:
                    tp = atr_tp
                    tp_source = 'atr'
                hold_counter = 0
                continue

            # manage open position
            if position == 'long':
                hold_counter += 1
                # first-touch exit using close price as proxy
                if price <= sl:
                    exit_price = sl
                    exit_reason = 'stop_loss'
                elif price >= tp:
                    exit_price = tp
                    exit_reason = 'take_profit'
                elif row['signal'] == 0 and row['ema_fast'] < row['ema_slow']:
                    exit_price = price
                    exit_reason = 'signal_exit'
                elif hold_counter >= self.max_holding_bars:
                    exit_price = price
                    exit_reason = 'timeout'
                else:
                    continue

                pnl = exit_price - entry_price
                ret = pnl / entry_price
                trades.append({'side': 'long', 'entry_idx': entry_idx, 'exit_idx': signals.index[i],
                               'entry_price': entry_price, 'exit_price': exit_price,
                               'sl_price': sl, 'tp_price': tp,
                               'sl_source': sl_source, 'tp_source': tp_source,
                               'exit_reason': exit_reason,
                               'pnl': pnl, 'return': ret})
                position = None
                entry_idx = None
                entry_price = None
                sl = None
                tp = None

            elif position == 'short':
                hold_counter += 1
                if price >= sl:
                    exit_price = sl
                    exit_reason = 'stop_loss'
                elif price <= tp:
                    exit_price = tp
                    exit_reason = 'take_profit'
                elif row['signal'] == 0 and row['ema_fast'] > row['ema_slow']:
                    exit_price = price
                    exit_reason = 'signal_exit'
                elif hold_counter >= self.max_holding_bars:
                    exit_price = price
                    exit_reason = 'timeout'
                else:
                    continue

                pnl = entry_price - exit_price
                ret = pnl / entry_price
                trades.append({'side': 'short', 'entry_idx': entry_idx, 'exit_idx': signals.index[i],
                               'entry_price': entry_price, 'exit_price': exit_price,
                               'sl_price': sl, 'tp_price': tp,
                               'sl_source': sl_source, 'tp_source': tp_source,
                               'exit_reason': exit_reason,
                               'pnl': pnl, 'return': ret})
                position = None
                entry_idx = None
                entry_price = None
                sl = None
                tp = None

        return pd.DataFrame(trades)


if __name__ == '__main__':
    print('SignalGenerator: import and use simulate_trades for quick backtest')
