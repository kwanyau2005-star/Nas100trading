import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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
            rows.append(
                {
                    "time": _parse_timestamp(ts),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                }
            )
            continue
        if isinstance(item, dict):
            ts = item.get("time", item.get("timestamp", item.get("ts")))
            o = item.get("open", item.get("o"))
            h = item.get("high", item.get("h"))
            l = item.get("low", item.get("l"))
            c = item.get("close", item.get("c", item.get("price")))
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
                }
            )
    if not rows:
        raise ValueError("No valid OHLC rows parsed from minishare payload")
    df = pd.DataFrame(rows).dropna()
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
            except (URLError, json.JSONDecodeError, ValueError, TypeError) as err:
                errors.append(f"{path}#k{idx + 1}:{type(err).__name__}")

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
    signals = _ensure_ohlc(sg.generate_signals(df))
    trades = sg.simulate_trades(df)

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
    }
    return summary, signals, trades


@app.get("/api/chart-data")
def chart_data():
    summary, signals, trades = _build_snapshot()

    chart = signals.tail(180).copy()
    chart_x = [_fmt_ts(ts) for ts in chart.index]
    latest = chart[["close", "ema_fast", "ema_slow", "rsi", "atr", "signal"]].tail(20)

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
            "entries": {
                "long": {
                    "x": entry_long_x,
                    "y": entry_long_y,
                },
                "short": {
                    "x": entry_short_x,
                    "y": entry_short_y,
                },
            },
            "exits": {
                "x": exit_x,
                "y": exit_y,
            },
            "risk_lines": {
                "sl_x": sl_x,
                "sl_y": sl_y,
                "tp_x": tp_x,
                "tp_y": tp_y,
            },
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
          body { font-family: Inter, Segoe UI, Arial, sans-serif; margin: 16px; background: #0b1220; color: #d1d5db; }
          h1, h2 { margin-bottom: 8px; color: #e5e7eb; }
          p { color: #9ca3af; }
          .cards { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
          .card { border: 1px solid #1f2937; border-radius: 8px; padding: 10px 14px; min-width: 170px; background: #111827; }
          .chart-wrap { border: 1px solid #1f2937; border-radius: 8px; background: #111827; padding: 8px; }
          .chart { width: 100%; height: 560px; margin-top: 6px; }
          .chart-small { width: 100%; height: 220px; margin-top: 10px; }
          table { border-collapse: collapse; width: 100%; margin-top: 8px; background: #0f172a; }
          th, td { border: 1px solid #1f2937; padding: 6px 8px; text-align: right; color: #d1d5db; }
          th { background: #111827; color: #e5e7eb; }
          th:first-child, td:first-child { text-align: left; }
          .tag { display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px; border:1px solid #334155; }
          .tag-live { color:#34d399; border-color:#065f46; }
          .tag-fallback { color:#fbbf24; border-color:#92400e; }
        </style>
      </head>
      <body>
        <h1>NAS100 (QQQ) Scalping Demo</h1>
        <p>OHLC K線 + EMA 指標 + 入場/離場 + 止損/止盈（每 10 秒更新）</p>
        <div class="cards">
          <div class="card"><b>Data source</b><br><span id="sourceTag" class="tag">-</span></div>
          <div class="card"><b>Current price</b><br><span id="currentPrice">-</span></div>
          <div class="card"><b>Long entries</b><br><span id="longEntries">-</span></div>
          <div class="card"><b>Short entries</b><br><span id="shortEntries">-</span></div>
          <div class="card"><b>Trades</b><br><span id="tradeCount">-</span></div>
          <div class="card"><b>Total PnL</b><br><span id="totalPnl">-</span></div>
          <div class="card"><b>Win rate</b><br><span id="winRate">-</span></div>
          <div class="card"><b>Avg return</b><br><span id="avgReturn">-</span></div>
        </div>
        <div id="sourceMessage" style="margin-bottom:8px;color:#fca5a5;"></div>
        <div class="chart-wrap">
          <div id="klineChart" class="chart"></div>
          <div id="rsiChart" class="chart-small"></div>
        </div>

        <h2>Latest Signals</h2>
        <table>
          <thead>
            <tr>
              <th>Time</th><th>Close</th><th>EMA Fast</th><th>EMA Slow</th><th>RSI</th><th>ATR</th><th>Signal</th>
            </tr>
          </thead>
          <tbody id="signalTable"></tbody>
        </table>

        <h2>Recent Trades</h2>
        <table>
          <thead>
            <tr>
              <th>Side</th><th>Entry</th><th>Exit</th><th>SL</th><th>TP</th><th>Exit reason</th><th>PnL</th><th>Return</th>
            </tr>
          </thead>
          <tbody id="tradeTable"></tbody>
        </table>
        <script>
          function toFixed(v, n) {
            return Number(v).toFixed(n);
          }

          async function loadData() {
            const res = await fetch('/api/chart-data');
            const data = await res.json();

            const s = data.summary;
            const sourceTag = document.getElementById('sourceTag');
            const pathInfo = s.path ? ` ${s.path}` : '';
            const keyInfo = s.token_index ? ` #key${s.token_index}` : '';
            const sourceLabel = `${s.data_source}${keyInfo} (${s.symbol}, ${s.interval})${pathInfo}`;
            sourceTag.innerText = sourceLabel;
            sourceTag.className = "tag " + (s.data_source === "minishare" ? "tag-live" : "tag-fallback");
            document.getElementById('sourceMessage').innerText = s.source_message || "";

            document.getElementById('currentPrice').innerText = toFixed(s.current_price, 4);
            document.getElementById('longEntries').innerText = s.long_entries;
            document.getElementById('shortEntries').innerText = s.short_entries;
            document.getElementById('tradeCount').innerText = s.trades;
            document.getElementById('totalPnl').innerText = toFixed(s.total_pnl, 4);
            document.getElementById('winRate').innerText = toFixed(s.win_rate * 100, 2) + '%';
            document.getElementById('avgReturn').innerText = toFixed(s.avg_return * 100, 2) + '%';

            const x = data.candles.x;
            const traces = [
              {
                x,
                open: data.candles.open,
                high: data.candles.high,
                low: data.candles.low,
                close: data.candles.close,
                type: 'candlestick',
                name: 'K線',
                increasing: { line: { color: '#22c55e', width: 1 }, fillcolor: '#22c55e' },
                decreasing: { line: { color: '#ef4444', width: 1 }, fillcolor: '#ef4444' }
              },
              {
                x,
                y: data.indicators.ema_fast,
                type: 'scatter',
                mode: 'lines',
                line: { width: 1.4, color: '#60a5fa' },
                name: 'EMA Fast'
              },
              {
                x,
                y: data.indicators.ema_slow,
                type: 'scatter',
                mode: 'lines',
                line: { width: 1.4, color: '#fbbf24' },
                name: 'EMA Slow'
              },
              {
                x: data.entries.long.x,
                y: data.entries.long.y,
                type: 'scatter',
                mode: 'markers',
                marker: { color: '#22c55e', size: 9, symbol: 'triangle-up' },
                name: 'Long Entry'
              },
              {
                x: data.entries.short.x,
                y: data.entries.short.y,
                type: 'scatter',
                mode: 'markers',
                marker: { color: '#ef4444', size: 9, symbol: 'triangle-down' },
                name: 'Short Entry'
              },
              {
                x: data.exits.x,
                y: data.exits.y,
                type: 'scatter',
                mode: 'markers',
                marker: { color: '#e5e7eb', size: 8, symbol: 'x' },
                name: 'Exit'
              },
              {
                x: data.risk_lines.sl_x,
                y: data.risk_lines.sl_y,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#f87171', width: 1, dash: 'dot' },
                name: 'SL',
                visible: 'legendonly'
              },
              {
                x: data.risk_lines.tp_x,
                y: data.risk_lines.tp_y,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#34d399', width: 1, dash: 'dot' },
                name: 'TP',
                visible: 'legendonly'
              }
            ];

            Plotly.react('klineChart', traces, {
              template: 'plotly_dark',
              uirevision: 'kline-fixed',
              paper_bgcolor: '#111827',
              plot_bgcolor: '#111827',
              margin: { t: 18, r: 56, b: 28, l: 46 },
              hovermode: 'x',
              dragmode: 'pan',
              xaxis: {
                type: 'date',
                rangeslider: { visible: false },
                showgrid: true,
                gridcolor: '#1f2937',
                color: '#9ca3af',
                tickformat: '%m-%d %H:%M',
                showspikes: true,
                spikemode: 'across',
                spikecolor: '#6b7280',
                spikethickness: 1
              },
              yaxis: {
                title: 'Price',
                side: 'right',
                showgrid: true,
                gridcolor: '#1f2937',
                color: '#9ca3af',
                showspikes: true,
                spikemode: 'across',
                spikecolor: '#6b7280',
                spikethickness: 1
              },
              legend: { orientation: 'h', y: 1.04, font: { color: '#cbd5e1' } }
            }, {
              responsive: true,
              displaylogo: false,
              scrollZoom: true,
              modeBarButtonsToRemove: ['select2d', 'lasso2d', 'toggleSpikelines']
            });

            Plotly.react('rsiChart', [{
              x,
              y: data.indicators.rsi,
              type: 'scatter',
              mode: 'lines',
              line: { width: 1.5, color: '#a78bfa' },
              name: 'RSI'
            }], {
              template: 'plotly_dark',
              uirevision: 'rsi-fixed',
              paper_bgcolor: '#111827',
              plot_bgcolor: '#111827',
              margin: { t: 10, r: 56, b: 40, l: 46 },
              hovermode: 'x',
              yaxis: { title: 'RSI', range: [0, 100], side: 'right', gridcolor: '#1f2937', color: '#9ca3af' },
              xaxis: { title: 'Time', type: 'date', tickformat: '%H:%M', gridcolor: '#1f2937', color: '#9ca3af' },
              shapes: [
                { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 70, y1: 70, line: { color: '#f97316', dash: 'dash' } },
                { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 30, y1: 30, line: { color: '#22c55e', dash: 'dash' } }
              ],
              showlegend: false
            }, { responsive: true, displaylogo: false });

            const signalBody = document.getElementById('signalTable');
            signalBody.innerHTML = data.latest_rows.map(r =>
              `<tr>
                <td>${r.time}</td>
                <td>${toFixed(r.close,4)}</td>
                <td>${toFixed(r.ema_fast,4)}</td>
                <td>${toFixed(r.ema_slow,4)}</td>
                <td>${toFixed(r.rsi,2)}</td>
                <td>${toFixed(r.atr,4)}</td>
                <td>${r.signal}</td>
              </tr>`
            ).join('');

            const tradeBody = document.getElementById('tradeTable');
            if (!data.recent_trades.length) {
              tradeBody.innerHTML = "<tr><td colspan='8'>No trades yet</td></tr>";
            } else {
              tradeBody.innerHTML = data.recent_trades.map(t =>
                `<tr>
                  <td>${t.side}</td>
                  <td>${toFixed(t.entry_price,4)}</td>
                  <td>${toFixed(t.exit_price,4)}</td>
                  <td>${toFixed(t.sl_price,4)}</td>
                  <td>${toFixed(t.tp_price,4)}</td>
                  <td>${t.exit_reason}</td>
                  <td>${toFixed(t.pnl,4)}</td>
                  <td>${toFixed(t.return * 100,2)}%</td>
                </tr>`
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
