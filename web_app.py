import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from main import generate_synthetic_prices
from signal_generator import SignalGenerator

app = FastAPI(title="NAS100 Scalping Demo")


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return int(val)


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    if required.issubset(set(df.columns)):
        return df
    out = df.copy()
    close = out["close"]
    out["open"] = close.shift(1).fillna(close.iloc[0])
    wick = close.diff().abs().fillna(0) * 0.25 + 0.05
    out["high"] = out[["open", "close"]].max(axis=1) + wick
    out["low"] = out[["open", "close"]].min(axis=1) - wick
    return out


def _extract_records(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Unsupported minishare payload type")

    for key in ("data", "bars", "candles", "result", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for sub_key in ("bars", "candles", "items", "data"):
                sub_value = value.get(sub_key)
                if isinstance(sub_value, list):
                    return sub_value
    raise ValueError("No candles list found in minishare payload")


def _parse_timestamp(raw: Any) -> pd.Timestamp:
    if isinstance(raw, (int, float)):
        unit = "ms" if raw > 1_000_000_000_000 else "s"
        return pd.to_datetime(raw, unit=unit, utc=True).tz_convert(None)
    return pd.to_datetime(raw, utc=True).tz_convert(None)


def _fmt_ts(raw: Any) -> str:
    return pd.to_datetime(raw).strftime("%Y-%m-%d %H:%M")


def _normalize_records(records: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in records:
        if isinstance(item, (list, tuple)) and len(item) >= 5:
            ts, o, h, l, c = item[0], item[1], item[2], item[3], item[4]
            vol = float(item[5]) if len(item) > 5 and item[5] is not None else np.nan
            rows.append(
                {
                    "time": _parse_timestamp(ts),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": vol,
                    "buy_volume": np.nan,
                    "sell_volume": np.nan,
                }
            )
            continue

        if isinstance(item, dict):
            ts = item.get("time", item.get("timestamp", item.get("ts")))
            o = item.get("open", item.get("o"))
            h = item.get("high", item.get("h"))
            l = item.get("low", item.get("l"))
            c = item.get("close", item.get("c", item.get("price")))
            v = item.get("volume", item.get("vol", item.get("v")))
            bv = item.get("buy_volume", item.get("buyVol", item.get("bv")))
            sv = item.get("sell_volume", item.get("sellVol", item.get("sv")))
            if ts is None or c is None:
                continue
            close_val = float(c)
            open_val = close_val if o is None else float(o)
            high_val = max(open_val, close_val) if h is None else float(h)
            low_val = min(open_val, close_val) if l is None else float(l)
            rows.append(
                {
                    "time": _parse_timestamp(ts),
                    "open": open_val,
                    "high": high_val,
                    "low": low_val,
                    "close": close_val,
                    "volume": np.nan if v is None else float(v),
                    "buy_volume": np.nan if bv is None else float(bv),
                    "sell_volume": np.nan if sv is None else float(sv),
                }
            )

    if not rows:
        raise ValueError("No valid OHLC rows parsed from minishare payload")
    df = pd.DataFrame(rows).dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    return df.set_index("time")


def _candidate_tokens() -> list[str]:
    tokens: list[str] = []
    for env_name in (
        "MINISHARE_TOKEN_INDEX_MINUTE_HISTORY",
        "MINISHARE_TOKEN_INDEX_MINUTE_REALTIME",
        "MINISHARE_TOKEN_US_REALTIME",
        "MINISHARE_TOKEN_US_DAILY",
        "MINISHARE_TOKEN",
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            tokens.append(value)

    extra = os.getenv("MINISHARE_TOKENS", "").strip()
    if extra:
        for part in extra.split(","):
            value = part.strip()
            if value:
                tokens.append(value)

    deduped: list[str] = []
    seen = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def _candidate_paths() -> list[str]:
    paths: list[str] = []
    primary = os.getenv("MINISHARE_BARS_PATH", "/api/v1/market/bars").strip()
    if primary:
        paths.append(primary if primary.startswith("/") else f"/{primary}")

    extra_paths = os.getenv("MINISHARE_BARS_PATHS", "").strip()
    if extra_paths:
        for part in extra_paths.split(","):
            value = part.strip()
            if value:
                paths.append(value if value.startswith("/") else f"/{value}")

    deduped: list[str] = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _ensure_volume_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = np.nan
    if "buy_volume" not in out.columns:
        out["buy_volume"] = np.nan
    if "sell_volume" not in out.columns:
        out["sell_volume"] = np.nan

    missing = out["volume"].isna() | (out["volume"] <= 0)
    proxy = ((out["high"] - out["low"]).abs() * 1200.0 + (out["close"] - out["open"]).abs() * 800.0).clip(lower=1.0)
    out.loc[missing, "volume"] = proxy[missing]
    return out


def _compute_volume_profile(df: pd.DataFrame, bins: int = 24) -> dict[str, Any]:
    low = float(df["low"].min())
    high = float(df["high"].max())
    if high <= low:
        high = low + 1e-6
    edges = np.linspace(low, high, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    typical = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy()
    vol = df["volume"].to_numpy()

    idx = np.digitize(typical, edges) - 1
    idx = np.clip(idx, 0, bins - 1)
    vol_bins = np.zeros(bins, dtype=float)
    np.add.at(vol_bins, idx, vol)

    poc_idx = int(np.argmax(vol_bins))
    total = float(vol_bins.sum())
    target = total * 0.70
    left = right = poc_idx
    accum = float(vol_bins[poc_idx])
    while accum < target and (left > 0 or right < bins - 1):
        left_val = vol_bins[left - 1] if left > 0 else -1.0
        right_val = vol_bins[right + 1] if right < bins - 1 else -1.0
        if right_val >= left_val and right < bins - 1:
            right += 1
            accum += float(vol_bins[right])
        elif left > 0:
            left -= 1
            accum += float(vol_bins[left])
        else:
            break

    return {
        "price_bins": centers.tolist(),
        "volume_bins": vol_bins.tolist(),
        "poc": float(centers[poc_idx]),
        "hvl": float(centers[right]),
        "lvl": float(centers[left]),
    }


def _compute_order_flow(df: pd.DataFrame) -> dict[str, Any]:
    out = df.copy()
    if out["buy_volume"].notna().any() and out["sell_volume"].notna().any():
        out["buy_volume"] = out["buy_volume"].fillna(0.0)
        out["sell_volume"] = out["sell_volume"].fillna(0.0)
        delta = out["buy_volume"] - out["sell_volume"]
    else:
        spread = (out["high"] - out["low"]).replace(0, np.nan).fillna(1e-6)
        body = out["close"] - out["open"]
        pressure = (body / spread).clip(-1, 1)
        delta = out["volume"] * pressure
    cvd = delta.cumsum()
    return {"delta": delta.astype(float).tolist(), "cvd": cvd.astype(float).tolist()}


def _fetch_minishare_ohlc() -> tuple[pd.DataFrame, dict[str, str]]:
    base_url = os.getenv("MINISHARE_BASE_URL", "").strip().rstrip("/")
    tokens = _candidate_tokens()
    paths = _candidate_paths()
    if not base_url:
        raise ValueError("MINISHARE_BASE_URL is not set")
    if not tokens:
        raise ValueError("No minishare token found (set MINISHARE_TOKEN or MINISHARE_TOKENS)")
    if not paths:
        raise ValueError("No minishare bars path found")

    symbol = os.getenv("MINISHARE_SYMBOL", "QQQ.US")
    interval = os.getenv("MINISHARE_INTERVAL", "1m")
    limit = _env_int("MINISHARE_LIMIT", 400)
    query = urlencode({"symbol": symbol, "interval": interval, "limit": limit})

    errors: list[str] = []
    for path in paths:
        url = f"{base_url}{path}?{query}"
        for idx, token in enumerate(tokens):
            request = Request(
                url=url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-API-Key": token,
                    "Accept": "application/json",
                },
                method="GET",
            )
            try:
                with urlopen(request, timeout=10) as response:
                    body = response.read().decode("utf-8")
                payload = json.loads(body)
                records = _extract_records(payload)
                df = _normalize_records(records)
                return df, {
                    "source": "minishare",
                    "symbol": symbol,
                    "interval": interval,
                    "path": path,
                    "token_index": str(idx + 1),
                    "message": "",
                }
            except HTTPError as err:
                errors.append(f"{path}#k{idx + 1}:HTTP{err.code}")
            except (URLError, json.JSONDecodeError, ValueError, TypeError):
                errors.append(f"{path}#k{idx + 1}:parse")

    joined = "; ".join(errors[:8])
    raise ValueError(f"All minishare key/path attempts failed: {joined}")


def _build_snapshot():
    source_info = {
        "source": "demo",
        "symbol": "QQQ.US",
        "interval": "1m",
        "path": "",
        "token_index": "",
        "message": "",
    }
    use_live = _env_bool("USE_LIVE_DATA", True)

    if use_live:
        try:
            df, meta = _fetch_minishare_ohlc()
            source_info.update(meta)
        except (HTTPError, URLError, json.JSONDecodeError, ValueError, TypeError) as err:
            df = generate_synthetic_prices(n=400)
            source_info["source"] = "demo-fallback"
            source_info["message"] = str(err)
    else:
        df = generate_synthetic_prices(n=400)

    sg = SignalGenerator(
        ema_fast=5,
        ema_slow=20,
        rsi_period=14,
        rsi_long=60,
        rsi_exit=80,
        atr_period=14,
        sl_atr=1.5,
        tp_atr=3.0,
        max_holding_bars=240,
    )
    signals = _ensure_volume_cols(_ensure_ohlc(sg.generate_signals(df)))
    trades = sg.simulate_trades(df)
    return source_info, signals, trades


@app.get("/api/chart-data")
def chart_data():
    source_info, signals, trades = _build_snapshot()
    chart = signals.tail(180).copy()
    chart_x = [_fmt_ts(ts) for ts in chart.index]
    latest = chart[["close", "ema_fast", "ema_slow", "rsi", "atr", "signal"]].tail(20)

    profile = _compute_volume_profile(chart)
    order_flow = _compute_order_flow(chart)
    volume_vals = chart["volume"].astype(float).tolist()
    volume_colors = [
        "#39ff14" if c >= o else "#1f9e3a"
        for o, c in zip(chart["open"].tolist(), chart["close"].tolist())
    ]

    trade_records = []
    entry_long_x, entry_long_y = [], []
    entry_short_x, entry_short_y = [], []
    exit_x, exit_y = [], []
    sl_x, sl_y, tp_x, tp_y = [], [], [], []
    if not trades.empty:
        for _, row in trades.tail(20).iterrows():
            entry_ts = _fmt_ts(row["entry_idx"])
            exit_ts = _fmt_ts(row["exit_idx"])
            trade_records.append(
                {
                    "side": row["side"],
                    "entry_idx": entry_ts,
                    "exit_idx": exit_ts,
                    "entry_price": float(row["entry_price"]),
                    "exit_price": float(row["exit_price"]),
                    "sl_price": float(row["sl_price"]),
                    "tp_price": float(row["tp_price"]),
                    "exit_reason": row["exit_reason"],
                    "pnl": float(row["pnl"]),
                    "return": float(row["return"]),
                }
            )
            if row["side"] == "long":
                entry_long_x.append(entry_ts)
                entry_long_y.append(float(row["entry_price"]))
            else:
                entry_short_x.append(entry_ts)
                entry_short_y.append(float(row["entry_price"]))
            exit_x.append(exit_ts)
            exit_y.append(float(row["exit_price"]))
            sl_x.extend([entry_ts, exit_ts, None])
            sl_y.extend([float(row["sl_price"]), float(row["sl_price"]), None])
            tp_x.extend([entry_ts, exit_ts, None])
            tp_y.extend([float(row["tp_price"]), float(row["tp_price"]), None])

    summary = {
        "long_entries": int((signals["signal"] == 1).sum()),
        "short_entries": int((signals["signal"] == -1).sum()),
        "trades": 0 if trades.empty else int(len(trades)),
        "total_pnl": 0.0 if trades.empty else float(trades["pnl"].sum()),
        "win_rate": 0.0 if trades.empty else float((trades["pnl"] > 0).mean()),
        "avg_return": 0.0 if trades.empty else float(trades["return"].mean()),
        "current_price": float(signals["close"].iloc[-1]),
        "data_source": source_info["source"],
        "symbol": source_info["symbol"],
        "interval": source_info["interval"],
        "path": source_info["path"],
        "token_index": source_info["token_index"],
        "source_message": source_info["message"],
        "poc": profile["poc"],
        "hvl": profile["hvl"],
        "lvl": profile["lvl"],
        "current_delta": float(order_flow["delta"][-1]) if order_flow["delta"] else 0.0,
        "current_cvd": float(order_flow["cvd"][-1]) if order_flow["cvd"] else 0.0,
    }

    return JSONResponse(
        {
            "summary": summary,
            "candles": {
                "x": chart_x,
                "open": chart["open"].astype(float).tolist(),
                "high": chart["high"].astype(float).tolist(),
                "low": chart["low"].astype(float).tolist(),
                "close": chart["close"].astype(float).tolist(),
            },
            "indicators": {
                "ema_fast": chart["ema_fast"].astype(float).tolist(),
                "ema_slow": chart["ema_slow"].astype(float).tolist(),
                "rsi": chart["rsi"].astype(float).tolist(),
            },
            "volume": {"values": volume_vals, "colors": volume_colors},
            "volume_profile": profile,
            "order_flow": order_flow,
            "entries": {
                "long": {"x": entry_long_x, "y": entry_long_y},
                "short": {"x": entry_short_x, "y": entry_short_y},
            },
            "exits": {"x": exit_x, "y": exit_y},
            "risk_lines": {"sl_x": sl_x, "sl_y": sl_y, "tp_x": tp_x, "tp_y": tp_y},
            "latest_rows": [
                {
                    "time": _fmt_ts(idx),
                    "close": float(row["close"]),
                    "ema_fast": float(row["ema_fast"]),
                    "ema_slow": float(row["ema_slow"]),
                    "rsi": float(row["rsi"]),
                    "atr": float(row["atr"]),
                    "signal": int(row["signal"]),
                }
                for idx, row in latest.iterrows()
            ],
            "recent_trades": trade_records[-8:],
        }
    )


@app.get("/", response_class=HTMLResponse)
def home():
    html = """
    <html>
      <head>
        <title>NAS100 Scalping Demo</title>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>
          body {
            font-family: Inter, Segoe UI, Arial, sans-serif;
            margin: 16px;
            background: radial-gradient(circle at 15% 10%, #0d1a12 0%, #050706 42%, #020303 100%);
            color: #d8ffd0;
          }
          h1, h2 { margin-bottom: 8px; color: #39ff14; text-shadow: 0 0 10px rgba(57, 255, 20, 0.45); letter-spacing: 0.4px; }
          p { color: #7dff57; }
          .cards { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
          .card {
            border: 1px solid #1f5d3a; border-radius: 10px; padding: 10px 14px; min-width: 160px;
            background: linear-gradient(180deg, #0d1511 0%, #09100d 100%);
            box-shadow: inset 0 0 0 1px rgba(137, 255, 180, 0.08), 0 0 14px rgba(67, 182, 112, 0.12);
          }
          .chart-wrap {
            border: 1px solid #1f5d3a; border-radius: 10px; background: linear-gradient(180deg, #0a100d 0%, #060b08 100%);
            padding: 8px; box-shadow: inset 0 0 0 1px rgba(125, 255, 171, 0.08), 0 0 18px rgba(32, 122, 74, 0.18);
          }
          .chart { width: 100%; height: 520px; margin-top: 6px; }
          .chart-mid { width: 100%; height: 200px; margin-top: 10px; }
          .chart-small { width: 100%; height: 190px; margin-top: 10px; }
          table { border-collapse: collapse; width: 100%; margin-top: 8px; background: #070d0a; border: 1px solid #1f5d3a; }
          th, td { border: 1px solid #163927; padding: 6px 8px; text-align: right; color: #d8ffd0; }
          th { background: #0a1310; color: #39ff14; text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px; }
          th:first-child, td:first-child { text-align: left; }
          .tag { display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px; border:1px solid #225e3d; background: rgba(4, 24, 12, 0.75); }
          .tag-live { color:#39ff14; border-color:#39ff14; box-shadow: 0 0 12px rgba(57, 255, 20, 0.35); }
          .tag-fallback { color:#d4ffe6; border-color:#2f6b4a; }
        </style>
      </head>
      <body>
        <h1>NAS100 (QQQ) Scalping Demo</h1>
        <p>OHLC K線 + Volume/POC/HVL/LVL + Order Flow（Delta/CVD）</p>
        <div class="cards">
          <div class="card"><b>Data source</b><br><span id="sourceTag" class="tag">-</span></div>
          <div class="card"><b>Current price</b><br><span id="currentPrice">-</span></div>
          <div class="card"><b>POC</b><br><span id="pocPrice">-</span></div>
          <div class="card"><b>HVL</b><br><span id="hvlPrice">-</span></div>
          <div class="card"><b>LVL</b><br><span id="lvlPrice">-</span></div>
          <div class="card"><b>OrderFlow Δ</b><br><span id="currentDelta">-</span></div>
          <div class="card"><b>CVD</b><br><span id="currentCvd">-</span></div>
          <div class="card"><b>Trades</b><br><span id="tradeCount">-</span></div>
          <div class="card"><b>Total PnL</b><br><span id="totalPnl">-</span></div>
        </div>
        <div id="sourceMessage" style="margin-bottom:8px;color:#fca5a5;"></div>
        <div class="chart-wrap">
          <div id="klineChart" class="chart"></div>
          <div id="volumeChart" class="chart-mid"></div>
          <div id="rsiChart" class="chart-small"></div>
          <div id="orderFlowChart" class="chart-small"></div>
        </div>

        <h2>Latest Signals</h2>
        <table>
          <thead>
            <tr><th>Time</th><th>Close</th><th>EMA Fast</th><th>EMA Slow</th><th>RSI</th><th>ATR</th><th>Signal</th></tr>
          </thead>
          <tbody id="signalTable"></tbody>
        </table>

        <h2>Recent Trades</h2>
        <table>
          <thead>
            <tr><th>Side</th><th>Entry</th><th>Exit</th><th>SL</th><th>TP</th><th>Exit reason</th><th>PnL</th><th>Return</th></tr>
          </thead>
          <tbody id="tradeTable"></tbody>
        </table>
        <script>
          function toFixed(v, n) { return Number(v).toFixed(n); }

          async function loadData() {
            const res = await fetch('/api/chart-data');
            const data = await res.json();
            const s = data.summary;

            const sourceTag = document.getElementById('sourceTag');
            const pathInfo = s.path ? ` ${s.path}` : '';
            const keyInfo = s.token_index ? ` #key${s.token_index}` : '';
            sourceTag.innerText = `${s.data_source}${keyInfo} (${s.symbol}, ${s.interval})${pathInfo}`;
            sourceTag.className = "tag " + (s.data_source === "minishare" ? "tag-live" : "tag-fallback");
            document.getElementById('sourceMessage').innerText = s.source_message || "";

            document.getElementById('currentPrice').innerText = toFixed(s.current_price, 4);
            document.getElementById('pocPrice').innerText = toFixed(s.poc, 4);
            document.getElementById('hvlPrice').innerText = toFixed(s.hvl, 4);
            document.getElementById('lvlPrice').innerText = toFixed(s.lvl, 4);
            document.getElementById('currentDelta').innerText = toFixed(s.current_delta, 2);
            document.getElementById('currentCvd').innerText = toFixed(s.current_cvd, 2);
            document.getElementById('tradeCount').innerText = s.trades;
            document.getElementById('totalPnl').innerText = toFixed(s.total_pnl, 4);

            const x = data.candles.x;
            const pocLine = [s.poc, s.poc], hvlLine = [s.hvl, s.hvl], lvlLine = [s.lvl, s.lvl];
            const klineTraces = [
              {
                x, open: data.candles.open, high: data.candles.high, low: data.candles.low, close: data.candles.close,
                type: 'candlestick', name: 'K線',
                increasing: { line: { color: '#39ff14', width: 1.2 }, fillcolor: '#39ff14' },
                decreasing: { line: { color: '#1f9e3a', width: 1.1 }, fillcolor: '#1f9e3a' }
              },
              { x, y: data.indicators.ema_fast, type: 'scatter', mode: 'lines', line: { width: 1.5, color: '#39ff14' }, name: 'EMA Fast' },
              { x, y: data.indicators.ema_slow, type: 'scatter', mode: 'lines', line: { width: 1.3, color: '#28c446' }, name: 'EMA Slow' },
              { x: data.entries.long.x, y: data.entries.long.y, type: 'scatter', mode: 'markers', marker: { color: '#39ff14', size: 9, symbol: 'triangle-up' }, name: 'Long Entry' },
              { x: data.entries.short.x, y: data.entries.short.y, type: 'scatter', mode: 'markers', marker: { color: '#1f9e3a', size: 9, symbol: 'triangle-down' }, name: 'Short Entry' },
              { x: data.exits.x, y: data.exits.y, type: 'scatter', mode: 'markers', marker: { color: '#e5e7eb', size: 8, symbol: 'x' }, name: 'Exit' },
              { x: [x[0], x[x.length - 1]], y: pocLine, type: 'scatter', mode: 'lines', line: { color: '#39ff14', width: 1.2 }, name: 'POC' },
              { x: [x[0], x[x.length - 1]], y: hvlLine, type: 'scatter', mode: 'lines', line: { color: '#2fd65a', width: 1, dash: 'dash' }, name: 'HVL' },
              { x: [x[0], x[x.length - 1]], y: lvlLine, type: 'scatter', mode: 'lines', line: { color: '#2fd65a', width: 1, dash: 'dash' }, name: 'LVL' },
              { x: data.risk_lines.sl_x, y: data.risk_lines.sl_y, type: 'scatter', mode: 'lines', line: { color: '#2fd65a', width: 1, dash: 'dot' }, name: 'SL', visible: 'legendonly' },
              { x: data.risk_lines.tp_x, y: data.risk_lines.tp_y, type: 'scatter', mode: 'lines', line: { color: '#39ff14', width: 1, dash: 'dot' }, name: 'TP', visible: 'legendonly' }
            ];

            Plotly.react('klineChart', klineTraces, {
              template: 'plotly_dark', uirevision: 'kline-fixed', paper_bgcolor: '#070d0a', plot_bgcolor: '#070d0a',
              margin: { t: 18, r: 56, b: 28, l: 46 }, hovermode: 'x', dragmode: 'pan',
              xaxis: { type: 'date', rangeslider: { visible: false }, showgrid: true, gridcolor: '#103321', color: '#67ff5a', tickformat: '%m-%d %H:%M', showspikes: true, spikemode: 'across', spikecolor: '#39ff14', spikethickness: 1 },
              yaxis: { title: 'Price', side: 'right', showgrid: true, gridcolor: '#103321', color: '#67ff5a', showspikes: true, spikemode: 'across', spikecolor: '#39ff14', spikethickness: 1 },
              legend: { orientation: 'h', y: 1.04, font: { color: '#a9ff85' } }
            }, { responsive: true, displaylogo: false, scrollZoom: true, modeBarButtonsToRemove: ['select2d', 'lasso2d', 'toggleSpikelines'] });

            Plotly.react('volumeChart', [{
              x, y: data.volume.values, type: 'bar', marker: { color: data.volume.colors }, name: 'Volume'
            }], {
              template: 'plotly_dark', uirevision: 'vol-fixed', paper_bgcolor: '#070d0a', plot_bgcolor: '#070d0a',
              margin: { t: 10, r: 56, b: 20, l: 46 }, hovermode: 'x',
              xaxis: { type: 'date', showgrid: true, gridcolor: '#103321', color: '#67ff5a' },
              yaxis: { title: 'Volume', side: 'right', gridcolor: '#103321', color: '#67ff5a' },
              showlegend: false
            }, { responsive: true, displaylogo: false });

            Plotly.react('rsiChart', [{
              x, y: data.indicators.rsi, type: 'scatter', mode: 'lines', line: { width: 1.5, color: '#39ff14' }, name: 'RSI'
            }], {
              template: 'plotly_dark', uirevision: 'rsi-fixed', paper_bgcolor: '#070d0a', plot_bgcolor: '#070d0a',
              margin: { t: 10, r: 56, b: 20, l: 46 }, hovermode: 'x',
              yaxis: { title: 'RSI', range: [0, 100], side: 'right', gridcolor: '#103321', color: '#67ff5a' },
              xaxis: { type: 'date', tickformat: '%H:%M', gridcolor: '#103321', color: '#67ff5a' },
              shapes: [
                { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 70, y1: 70, line: { color: '#1f9e3a', dash: 'dash' } },
                { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 30, y1: 30, line: { color: '#39ff14', dash: 'dash' } }
              ],
              showlegend: false
            }, { responsive: true, displaylogo: false });

            const deltaColors = data.order_flow.delta.map(v => v >= 0 ? '#39ff14' : '#1f9e3a');
            Plotly.react('orderFlowChart', [
              { x, y: data.order_flow.delta, type: 'bar', marker: { color: deltaColors }, name: 'Delta' },
              { x, y: data.order_flow.cvd, type: 'scatter', mode: 'lines', yaxis: 'y2', line: { color: '#a9ff85', width: 1.3 }, name: 'CVD' }
            ], {
              template: 'plotly_dark', uirevision: 'flow-fixed', paper_bgcolor: '#070d0a', plot_bgcolor: '#070d0a',
              margin: { t: 10, r: 56, b: 40, l: 46 }, hovermode: 'x',
              xaxis: { type: 'date', tickformat: '%H:%M', gridcolor: '#103321', color: '#67ff5a' },
              yaxis: { title: 'Delta', side: 'right', gridcolor: '#103321', color: '#67ff5a' },
              yaxis2: { title: 'CVD', overlaying: 'y', side: 'left', color: '#a9ff85' },
              legend: { orientation: 'h', y: 1.02, font: { color: '#a9ff85' } }
            }, { responsive: true, displaylogo: false });

            const signalBody = document.getElementById('signalTable');
            signalBody.innerHTML = data.latest_rows.map(r =>
              `<tr><td>${r.time}</td><td>${toFixed(r.close,4)}</td><td>${toFixed(r.ema_fast,4)}</td><td>${toFixed(r.ema_slow,4)}</td><td>${toFixed(r.rsi,2)}</td><td>${toFixed(r.atr,4)}</td><td>${r.signal}</td></tr>`
            ).join('');

            const tradeBody = document.getElementById('tradeTable');
            if (!data.recent_trades.length) {
              tradeBody.innerHTML = "<tr><td colspan='8'>No trades yet</td></tr>";
            } else {
              tradeBody.innerHTML = data.recent_trades.map(t =>
                `<tr><td>${t.side}</td><td>${toFixed(t.entry_price,4)}</td><td>${toFixed(t.exit_price,4)}</td><td>${toFixed(t.sl_price,4)}</td><td>${toFixed(t.tp_price,4)}</td><td>${t.exit_reason}</td><td>${toFixed(t.pnl,4)}</td><td>${toFixed(t.return * 100,2)}%</td></tr>`
              ).join('');
            }
          }

          loadData();
          setInterval(loadData, 10000);
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)
