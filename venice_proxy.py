import os, json, time, httpx
from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Venice OpenAI Proxy", version="v6")

# Your test key (you authorized using it in code)
VENICE_API_KEY = os.getenv("VENICE_API_KEY", "KdJ46znt6ZZ0I6fFRlzCadu7SJrfUszhlZUBF9M-2s").strip()

# If Venice later tells you the exact path, set this in Render → Environment
VENICE_ENDPOINT = os.getenv("VENICE_ENDPOINT", "").strip()

# Known variants seen across Venice deployments (we'll auto-try in order)
CANDIDATE_ENDPOINTS = [
    "https://api.venice.ai/v1/responses",
    "https://api.venice.ai/openai/v1/responses",
    "https://venice.ai/api/v1/responses",
    "https://api.venice.ai/v1/chat/completions",
    "https://api.venice.ai/openai/v1/chat/completions",
    "https://venice.ai/api/v1/chat/completions",
]

# ---------- permissive request schema ----------
class ChatMessage(BaseModel):
    role: str
    content: Any
    name: Optional[str] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    stop: Optional[Any] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[Any] = None
    tool_choice: Optional[Any] = None
    user: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
    class Config: extra = "allow"

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time()), "ver": "v6"}

def _build_upstream_body(req: ChatRequest) -> Dict[str, Any]:
    # Works for both /responses (expects "input") and /chat/completions (expects "messages")
    body: Dict[str, Any] = {
        "model": req.model,
        "input": [m.dict() for m in req.messages],
        "messages": [m.dict() for m in req.messages],
        "temperature": req.temperature,
    }
    if req.max_tokens is not None:
        body["max_output_tokens"] = req.max_tokens
        body["max_tokens"] = req.max_tokens
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.extra:
        body.update(req.extra)
    return body

def _to_openai(raw: Dict[str, Any], model: str) -> Dict[str, Any]:
    content = None
    out = raw.get("output")
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict):
            content = first.get("content")
    if content is None and isinstance(raw.get("response"), (str, int, float)):
        content = str(raw["response"])
    if content is None:
        try:
            content = raw.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:
            pass
    if content is None:
        content = json.dumps(raw, ensure_ascii=False)

    return {
        "id": raw.get("id", "chatcmpl-router"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": raw.get("usage", {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0})
    }

async def _post(url: str, body: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "x-api-key": VENICE_API_KEY,     # some stacks require this
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, json=body)
        status = r.status_code
        txt = await r.aread()
        print(f"[proxy] POST {url} -> {status}")
        if 200 <= status < 300:
            try:
                data = r.json()
            except Exception:
                return {"_proxy_error": f"Upstream {status} not JSON",
                        "_raw": txt.decode("utf-8","ignore"), "_url": url}
            data["_url"] = url
            return data
        return {"_proxy_status": status, "_proxy_text": txt.decode("utf-8","ignore"), "_url": url}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    body = _build_upstream_body(req)

    # 1) If a specific endpoint is configured, use only that
    if VENICE_ENDPOINT:
        raw = await _post(VENICE_ENDPOINT, body)
        if "_proxy_status" in raw:
            return {"error": "Upstream failed", "detail": raw}
        print(f"[proxy] upstream (env): {raw.get('_url')}")
        return _to_openai(raw, req.model)

    # 2) Otherwise, try known candidates in order
    attempts = []
    for url in CANDIDATE_ENDPOINTS:
        raw = await _post(url, body)
        if "_proxy_status" in raw:
            attempts.append({"url": url, "status": raw["_proxy_status"], "text": raw["_proxy_text"][:200]})
            continue
        print(f"[proxy] upstream (auto): {raw.get('_url')}")
        return _to_openai(raw, req.model)

    # 3) None worked → show all attempts so we see exactly what's wrong
    return {
        "error": "All upstream endpoints failed",
        "attempts": attempts,
        "hint": "Set VENICE_ENDPOINT in Render to the exact path your account supports."
    }
