# Nas100trading - Signal demo

This repository contains a simple signal-only demo for a 1-minute QQQ/NDX scalping strategy using the `minishare` (minidoc) data APIs. The demo fetches 1-minute bars and recent news, computes indicators (EMA, VWAP, ATR, RSI), applies a simplified ICT/SMC-style signal logic, and sends alerts to Telegram.

Important: This project only generates signals and does NOT place live orders. Do NOT put real API keys into the repo. Use environment variables or a secrets manager.

Files included
- signal_demo.py - main demo script (polling idx_mins)
- .env.example - environment variables example (fill and rename to .env locally)
- requirements.txt - Python dependencies
- Dockerfile - optional containerization

Quick start (local)
1. Create a Python virtualenv and install deps:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Copy `.env.example` to `.env` and fill your keys (on your machine or VPS):
   cp .env.example .env
   # edit .env with your keys

3. Run the demo:
   python signal_demo.py

Telegram
- Configure a bot token and chat id, set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in your .env. The script will post signals to the chat.

Security
- Rotate/revoke any keys accidentally exposed. Do NOT commit `.env` or real keys. Use Docker secrets or a secrets manager for production.

Notes & next steps
- This demo uses polling of `idx_mins`. For lower latency, consider switching to the realtime WebSocket (rt-bid-ms) and aggregating ticks to 1-minute bars.
- If you want me to switch to WebSocket or add a FastAPI webhook for news, open a PR and request changes.
