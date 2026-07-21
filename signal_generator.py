import pandas as pd
import numpy as np
from typing import List

class SignalGenerator:
    """Scalping signal generator with EMA/RSI entries and ATR-based SL/TP.

    Features:
    - EMA crossover + RSI threshold for entries
    - ATR proxy (rolling abs diff) for volatility
    - Stop-loss / take-profit in multiples of ATR
    - simulate_trades to run a simple first-touch backtest
    """

    def __init__(self,
                 ema_fast=5,
                 ema_slow=20,
                 rsi_period=14,
                 rsi_long=60,
                 rsi_exit=80,
                 atr_period=14,
                 sl_atr=1.5,
                 tp_atr=3.0,
                 max_holding_bars=240):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_long = rsi_long
        self.rsi_exit = rsi_exit
        self.atr_period = atr_period
        self.sl_atr = sl_atr
        self.tp_atr = tp_atr
        self.max_holding_bars = max_holding_bars

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
        df = self.compute_indicators(df)
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
                sl = entry_price - self.sl_atr * atr
                tp = entry_price + self.tp_atr * atr
                hold_counter = 0
                continue
            if position is None and row['signal'] == -1:
                position = 'short'
                entry_idx = signals.index[i]
                entry_price = price
                sl = entry_price + self.sl_atr * atr
                tp = entry_price - self.tp_atr * atr
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
