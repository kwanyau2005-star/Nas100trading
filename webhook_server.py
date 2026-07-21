
# webhook_server.py
# FastAPI webhook to receive news push

from fastapi import FastAPI, Request, Header
import uvicorn
import os
import datetime as dt
from collections import deque

app = FastAPI()

# simple in-memory buffer for recent news (last 30 mins)
NEWS_BUFFER = deque(maxlen=1000)

@app.post('/webhook/news')
async def news_webhook(request: Request, authorization: str = Header(None)):
    # optional basic auth check
    expected = os.getenv('NEWS_WEBHOOK_SECRET')
    if expected and authorization != f"Bearer {expected}":
        return {"error": "unauthorized"}
    payload = await request.json()
    # expected payload includes: title, impact (low/medium/high), symbol, timestamp
    item = {
        'title': payload.get('title'),
        'impact': payload.get('impact', 'low'),
        'symbol': payload.get('symbol'),
        'timestamp': payload.get('timestamp', dt.datetime.utcnow().isoformat())
    }
    NEWS_BUFFER.append(item)
    return {"status": "ok"}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('WEBHOOK_PORT', 8000)))
