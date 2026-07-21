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
5) Run simple web preview:
   uvicorn web_app:app --host 0.0.0.0 --port 8000
   - open http://127.0.0.1:8000

Live data (minishare)

Set env before starting web app:

MINISHARE_BASE_URL=<your-minishare-api-base-url>
MINISHARE_TOKEN=<your-token>
MINISHARE_SYMBOL=QQQ.US
MINISHARE_INTERVAL=1m
MINISHARE_LIMIT=400
MINISHARE_BARS_PATH=/api/v1/market/bars
USE_LIVE_DATA=true

Notes:
- Web app reads minishare first. If API fails, it falls back to demo data and shows reason in page.
- If your minishare endpoint path/query is different, update MINISHARE_BARS_PATH and keep symbol/interval/limit env values aligned with your API.

Continuous Integration

- A GitHub Actions workflow (/.github/workflows/ci.yml) runs pytest on pull requests and pushes to main.

Notes
- This is a test/demo implementation. Replace data source with real market data before live use.
- Keep secrets out of the repo; use .env and .env.example as templates.
