import pandas as pd
import numpy as np

class SignalGenerator:
    """Simple scalping signal generator using short/long EMAs and RSI.

    - Long entry: ema_fast > ema_slow and rsi < rsi_long
    - Long exit: ema_fast < ema_slow or rsi > rsi_exit
    - Short entry/exit symmetric
    This is a lightweight test implementation for local testing/backtest.
    """
    def __init__(self, ema_fast=5, ema_slow=20, rsi_period=14, rsi_long=60, rsi_exit=80):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_long = rsi_long
        self.rsi_exit = rsi_exit

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        price = df['close']
        df = df.copy()
        df['ema_fast'] = price.ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = price.ewm(span=self.ema_slow, adjust=False).mean()
        delta = price.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(self.rsi_period, min_periods=1).mean()
        ma_down = down.rolling(self.rsi_period, min_periods=1).mean()
        rs = ma_up / (ma_down.replace(0, np.nan))
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.compute_indicators(df)
        df['signal'] = 0  # 1 = long entry, -1 = short entry, 0 = neutral / possible exit

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


if __name__ == '__main__':
    print('Import this module and use SignalGenerator. Run demo via main.py')
