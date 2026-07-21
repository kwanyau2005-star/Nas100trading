from html import escape

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from main import generate_synthetic_prices
from signal_generator import SignalGenerator

app = FastAPI(title="NAS100 Scalping Demo")


def _build_snapshot():
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
    signals = sg.generate_signals(df)
    trades = sg.simulate_trades(df)

    long_entries = int((signals["signal"] == 1).sum())
    short_entries = int((signals["signal"] == -1).sum())
    latest = signals[["close", "ema_fast", "ema_slow", "rsi", "atr", "signal"]].tail(12)

    if trades.empty:
        summary = {"trades": 0, "total_pnl": 0.0, "win_rate": 0.0, "avg_return": 0.0}
        recent_trades = []
    else:
        summary = {
            "trades": int(len(trades)),
            "total_pnl": float(trades["pnl"].sum()),
            "win_rate": float((trades["pnl"] > 0).mean()),
            "avg_return": float(trades["return"].mean()),
        }
        recent_trades = trades.tail(8).to_dict(orient="records")

    return long_entries, short_entries, latest, summary, recent_trades


@app.get("/", response_class=HTMLResponse)
def home():
    long_entries, short_entries, latest, summary, recent_trades = _build_snapshot()

    signal_rows = []
    for idx, row in latest.iterrows():
        signal_rows.append(
            "<tr>"
            f"<td>{escape(str(idx))}</td>"
            f"<td>{row['close']:.4f}</td>"
            f"<td>{row['ema_fast']:.4f}</td>"
            f"<td>{row['ema_slow']:.4f}</td>"
            f"<td>{row['rsi']:.2f}</td>"
            f"<td>{row['atr']:.4f}</td>"
            f"<td>{int(row['signal'])}</td>"
            "</tr>"
        )

    trade_rows = []
    for trade in recent_trades:
        trade_rows.append(
            "<tr>"
            f"<td>{escape(str(trade['side']))}</td>"
            f"<td>{trade['entry_price']:.4f}</td>"
            f"<td>{trade['exit_price']:.4f}</td>"
            f"<td>{trade['pnl']:.4f}</td>"
            f"<td>{trade['return'] * 100:.2f}%</td>"
            "</tr>"
        )
    if not trade_rows:
        trade_rows.append("<tr><td colspan='5'>No trades yet</td></tr>")

    html = f"""
    <html>
      <head>
        <title>NAS100 Scalping Demo</title>
        <meta http-equiv="refresh" content="10">
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          h1, h2 {{ margin-bottom: 8px; }}
          .cards {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
          .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px 14px; min-width: 180px; }}
          table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
          th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
          th:first-child, td:first-child {{ text-align: left; }}
        </style>
      </head>
      <body>
        <h1>NAS100 (QQQ) Scalping Demo</h1>
        <p>Auto-refresh every 10 seconds (synthetic data demo).</p>
        <div class="cards">
          <div class="card"><b>Long entries</b><br>{long_entries}</div>
          <div class="card"><b>Short entries</b><br>{short_entries}</div>
          <div class="card"><b>Trades</b><br>{summary["trades"]}</div>
          <div class="card"><b>Total PnL</b><br>{summary["total_pnl"]:.4f}</div>
          <div class="card"><b>Win rate</b><br>{summary["win_rate"] * 100:.2f}%</div>
          <div class="card"><b>Avg return</b><br>{summary["avg_return"] * 100:.2f}%</div>
        </div>

        <h2>Latest Signals</h2>
        <table>
          <thead>
            <tr>
              <th>Time</th><th>Close</th><th>EMA Fast</th><th>EMA Slow</th><th>RSI</th><th>ATR</th><th>Signal</th>
            </tr>
          </thead>
          <tbody>
            {''.join(signal_rows)}
          </tbody>
        </table>

        <h2>Recent Trades</h2>
        <table>
          <thead>
            <tr>
              <th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Return</th>
            </tr>
          </thead>
          <tbody>
            {''.join(trade_rows)}
          </tbody>
        </table>
      </body>
    </html>
    """
    return HTMLResponse(content=html)
