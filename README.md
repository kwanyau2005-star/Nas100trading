# Nas100trading

Lightweight demo for NAS100 (QQQ) scalping signals.

Quick start

1) Create and activate a Python venv:
   python3 -m venv venv && source venv/bin/activate
2) Install dependencies:
   pip install -r requirements.txt
3) Run demo to generate synthetic signals:
   python main.py
   - outputs signals.csv and prints summary
4) Run tests:
   pytest -q

Use itick futures data source

1) Copy env template and fill your credentials/settings:
   cp .env.example .env
2) Set:
   - DATA_SOURCE=itick
   - ITICK_API_KEY / ITICK_API_SECRET / ITICK_ACCOUNT (if required by your endpoint)
   - ITICK_KLINES_ENDPOINT to your real kline API path from itick
   - ITICK_SYMBOL / ITICK_INTERVAL / ITICK_LIMIT
3) Run:
   python main.py
   - main.py will fetch klines from itick and feed them into SignalGenerator.
   - if DATA_SOURCE is not `itick`, it falls back to demo synthetic prices.

Use Tradovate futures data source

1) Copy env template and fill your credentials/settings:
   cp .env.example .env
2) Set:
   - DATA_SOURCE=tradovate
   - TRADOVATE_USERNAME / TRADOVATE_PASSWORD (or TRADOVATE_ACCESS_TOKEN)
   - TRADOVATE_BASE_URL (demo or live), TRADOVATE_AUTH_ENDPOINT, TRADOVATE_CHART_ENDPOINT
   - TRADOVATE_SYMBOL / TRADOVATE_INTERVAL / TRADOVATE_LIMIT
3) Run:
   python main.py
   - main.py will authenticate with Tradovate, fetch candles, and feed them into SignalGenerator.
   - if DATA_SOURCE is not `tradovate` or `itick`, it falls back to demo synthetic prices.

Continuous Integration

- A GitHub Actions workflow (/.github/workflows/ci.yml) runs pytest on pull requests and pushes to main.

Notes
- This is a test/demo implementation. Replace data source with real market data before live use.
- Keep secrets out of the repo; use .env and .env.example as templates.
