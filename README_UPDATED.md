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

Notes
- This is a test/demo implementation. Replace data source with real market data before live use.
- Keep secrets out of the repo; use .env and .env.example as templates.
