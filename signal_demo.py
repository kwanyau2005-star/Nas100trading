# signal_demo.py
# Requirements:
# pip install minishare pandas requests python-dotenv

import os
import time
import datetime as dt
import pandas as pd
import requests
from dotenv import load_dotenv
from threading import Lock
try:
    import minishare as ms
except ImportError:
    ms = None

load_dotenv()

# config from env
TOKEN = os.getenv("MINIDOC_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SYMBOL = os.getenv("SYMBOL", "QQQ.US")   # adapt to minishare ts_code
EQUITY = float(os.getenv("EQUITY", "50000"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.005"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "15"))

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("Warning: Telegram not fully configured (alerts will print to console)")

api = None
api_lock = Lock()

def get_api():
    """Lazily create and return the minishare API client."""
    global api
    if api is not None:
        return api
    with api_lock:
        if api is not None:
            return api
        if ms is None:
            raise RuntimeError("minishare is not installed; run `pip install minishare`")
        if not TOKEN:
            raise RuntimeError("MINIDOC_TOKEN not set in environment")
        api = ms.pro_api(TOKEN)
    return api

# ---- helpers: indicators ----
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def true_range(df):
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(df, n=14):
    tr = true_range(df)
    return tr.rolling(n, min_periods=1).mean()

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    pv = (tp * df['volume']).cumsum()
    v = df['volume'].cumsum()
    return (pv / v).fillna(method='ffill')

# simple structure detectors
def is_bos_up(df, lookback=5):
    if len(df) < lookback + 2:
        return False
    prev_max = df['high'][-(lookback+1):-1].max()
    return df['high'].iloc[-1] > prev_max

def wick_sweep_up(df, wick_pct=0.002):
    if len(df) < 3:
        return False
    prev_low = df['low'].iloc[-2]
    curr_low = df['low'].iloc[-1]
    curr_close = df['close'].iloc[-1]
    return (curr_low < prev_low * (1 - wick_pct)) and (curr_close > prev_low)

def is_bos_down(df, lookback=5):
    if len(df) < lookback + 2:
        return False
    prev_min = df['low'][-(lookback+1):-1].min()
    return df['low'].iloc[-1] < prev_min

def wick_sweep_down(df, wick_pct=0.002):
    if len(df) < 3:
        return False
    prev_high = df['high'].iloc[-2]
    curr_high = df['high'].iloc[-1]
    curr_close = df['close'].iloc[-1]
    return (curr_high > prev_high * (1 + wick_pct)) and (curr_close < prev_high)

# generate signal (same logic as earlier)
def generate_signal(df, latest_news=None, require_vwap_on=True):
    if df is None or len(df) < 20:
        return {'signal': None, 'reason': 'insufficient data'}
    df = df.copy().reset_index(drop=True)
    df['ema8'] = ema(df['close'], 8)
    df['ema21'] = ema(df['close'], 21)
    df['atr14'] = atr(df, 14)
    df['rsi14'] = rsi(df['close'], 14)
    df['vwap'] = vwap(df)

    last = df.iloc[-1]

    # news filter
    if latest_news:
        for n in latest_news:
            if n.get('impact', 'low') in ('high', 'critical'):
                return {'signal': None, 'reason': 'high_impact_news'}

    cond_trend = last['ema8'] > last['ema21']
    cond_rsi = last['rsi14'] > 45
    cond_bos = is_bos_up(df, lookback=5)
    cond_wick = wick_sweep_up(df, wick_pct=0.002)
    cond_vwap = True
    if require_vwap_on:
        cond_vwap = last['close'] > last['vwap']

    long_entry = cond_trend and cond_rsi and (cond_bos or cond_wick) and cond_vwap
    if long_entry:
        sl_price = last['close'] - last['atr14'] * 1.2
        tp_price = last['close'] + last['atr14'] * 1.8
        risk_per_share = last['close'] - sl_price
        if risk_per_share <= 0:
            return {'signal': None, 'reason': 'invalid SL'}
        risk_amount = EQUITY * RISK_PCT
        size = max(1, int(risk_amount / risk_per_share))
        return {
            'signal': 'long',
            'reason': 'ema8>ema21 rsi>45 and (bos|wick) and vwap ok',
            'price': float(last['close']),
            'stop': round(float(sl_price), 4),
            'tp': round(float(tp_price), 4),
            'size_suggest': int(size)
        }

    cond_trend_s = last['ema8'] < last['ema21']
    cond_rsi_s = last['rsi14'] < 55
    cond_bos_s = is_bos_down(df, lookback=5)
    cond_wick_s = wick_sweep_down(df, wick_pct=0.002)
    cond_vwap_s = True
    if require_vwap_on:
        cond_vwap_s = last['close'] < last['vwap']

    short_entry = cond_trend_s and cond_rsi_s and (cond_bos_s or cond_wick_s) and cond_vwap_s
    if short_entry:
        sl_price = last['close'] + last['atr14'] * 1.2
        tp_price = last['close'] - last['atr14'] * 1.8
        risk_per_share = sl_price - last['close']
        if risk_per_share <= 0:
            return {'signal': None, 'reason': 'invalid SL'}
        risk_amount = EQUITY * RISK_PCT
        size = max(1, int(risk_amount / risk_per_share))
        return {
            'signal': 'short',
            'reason': 'ema8<ema21 rsi<55 and (bos|wick) and vwap ok',
            'price': float(last['close']),
            'stop': round(float(sl_price), 4),
            'tp': round(float(tp_price), 4),
            'size_suggest': int(size)
        }

    return {'signal': None, 'reason': 'no_setup'}

# ---- notifier ----
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("TG not configured:", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=3)
    except Exception as e:
        print("telegram send error:", e)

def alert(payload):
    text = f"[{SYMBOL}] SIGNAL: {payload.get('signal')} price:{payload.get('price')} stop:{payload.get('stop')} tp:{payload.get('tp')} size:{payload.get('size_suggest')} reason:{payload.get('reason')}"
    print(dt.datetime.utcnow(), text)
    send_telegram(text)

# ---- data fetch / runner ----
def fetch_recent_minutes(ts_code, minutes=30):
    """
    Use idx_mins to fetch recent minutes. Adjust format to provider requirements.
    """
    end_dt = dt.datetime.utcnow()
    start_dt = end_dt - dt.timedelta(minutes=minutes)
    start_s = start_dt.strftime("%Y%m%d %H:%M:%S")
    end_s = end_dt.strftime("%Y%m%d %H:%M:%S")
    try:
        api = get_api()
        df = api.idx_mins(ts_code=ts_code, freq='1min', start_date=start_s, end_date=end_s)
        # minishare returns DataFrame with columns likely: trade_time, open, high, low, close, volume
        if df is None or df.empty:
            return None
        # normalize column names if necessary
        # ensure numeric types
        df = df.rename(columns={c: c.lower() for c in df.columns})
        # require columns: open,high,low,close,volume,trade_time
        # convert trade_time to datetime if exists
        if 'trade_time' in df.columns:
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df = df.sort_values('trade_time')
        return df
    except Exception as e:
        print("fetch_recent_minutes error:", e)
        return None

def fetch_recent_news(minutes=10):
    # simple pull of latest news in last N minutes
    try:
        api = get_api()
        now = dt.datetime.utcnow()
        start = (now - dt.timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
        # using ms.pro_api news() with start_date/end_date; adjust format to provider
        dfn = api.news(start_date=start, end_date=now.strftime("%Y-%m-%d %H:%M:%S"), limit=200)
        if dfn is None or dfn.empty:
            return []
        # normalize: assume dfn has 'pub_time','impact' or similar
        news_list = []
        for _, r in dfn.iterrows():
            # adapt to actual columns: title, source, pub_time, impact
            news_list.append({
                'headline': r.get('title') or r.get('headline') or '',
                'impact': r.get('impact') or r.get('level') or 'low'
            })
        return news_list
    except Exception as e:
        print("fetch_recent_news error:", e)
        return []

def main_loop():
    last_checked_minute = None
    while True:
        try:
            df = fetch_recent_minutes(SYMBOL, minutes=60)
            if df is None or df.empty:
                print("no minute data")
                time.sleep(POLL_INTERVAL)
                continue
            # determine the latest complete minute key (use trade_time or index)
            if 'trade_time' in df.columns:
                latest_time = pd.to_datetime(df['trade_time'].iloc[-1])
            else:
                latest_time = dt.datetime.utcnow().replace(second=0, microsecond=0)  # fallback

            # if new minute bar (not processed)
            if last_checked_minute is None or latest_time > last_checked_minute:
                last_checked_minute = latest_time
                # build df for indicators
                # ensure df has columns open, high, low, close, volume
                df_proc = df.rename(columns={c: c.lower() for c in df.columns})
                # if necessary, select only required cols:
                cols = ['open','high','low','close','volume']
                for c in cols:
                    if c not in df_proc.columns:
                        print(f"missing column {c} in data, abort this cycle")
                        df_proc[c] = 0
                df_proc = df_proc[cols].astype(float)
                news = fetch_recent_news(minutes=10)
                sig = generate_signal(df_proc, latest_news=news)
                if sig.get('signal'):
                    alert(sig)
                else:
                    print(dt.datetime.utcnow(), "no signal:", sig.get('reason'))
            else:
                # no new minute yet
                pass
        except Exception as e:
            print("main_loop exception:", e)
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    print("Starting signal demo for", SYMBOL)
    main_loop()
