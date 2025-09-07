# Venice Proxy for ElevenLabs

A FastAPI-based OpenAI-compatible proxy that wraps Venice AI's chat completion API.

## Local Run

```bash
pip install -r requirements.txt
uvicorn venice_proxy:app --host=0.0.0.0 --port=8000
```

## Render Deployment

- **Build command**: `pip install -r requirements.txt`
- **Start command**: `uvicorn venice_proxy:app --host=0.0.0.0 --port=10000`
- **Environment variable**: `VENICE_API_KEY=<your_key>`