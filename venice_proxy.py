import os, json, time, httpx
from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Venice OpenAI Proxy", version="v4")

VENICE_API_KEY = os.getenv("VENICE_API_KEY", "KdJ46znt6ZZ0I6fFRlzCadu7SJrfUszhlZUBF9M-2s").strip()

# Candidate bases/paths seen across Venice deployments
CANDIDATE_ENDPOINTS = [
    "https://api.venice.ai/v1/responses",          # new Responses API (most likely)
    "https://api.venice.ai/openai/v1/responses",   # nested under /openai
    "https://venice.ai/api/v1/responses",          # alt domain + /api
    "https://api.venice.ai/v1/chat/completions",   # classic chat route (some tenants)
]

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
    return {"ok": True, "ts": int(time.time()), "ver": "v4"}

async def call_venice(body: Dict[str, Any]) -> Dict[str, Any]:
    """Try each candidate endpoint; return first successful JSON response, with info."""
    headers = {"Authorization": f"Bearer {VENICE_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        last = None
        for url in CANDIDATE_ENDPOINTS:
            try:
                r = await client.post(url, headers=headers, json=body)
                status = r.status_code
                txt = await r.aread()
                # print which url we used (visible in Render logs)
                print(f"[proxy] POST {url} -> {status}")
                if 200 <= status < 300:
                    try:
                        data = r.json()
                    except Exception:
                        return {"_proxy_error": f"Upstream {status} not JSON", "_raw": txt.decode("utf-8","ignore"), "_url": url}
                    data["_url"] = url
                    return data
                last = {"status": status, "text": txt.decode("utf-8","ignore"), "url": url}
            except Exception as e:
                last = {"status": 0, "text": str(e), "url": url}
        # none worked
        return {"_proxy_error": "All endpoints failed", "_last": last}

def map_openai_to_responses(req: ChatRequest) -> Dict[str, Any]:
    """Build a Venice-compatible request that works for both /responses and chat routes."""
    body = {
        "model": req.model,
        # /responses expects "input" (list of role/content dicts); chat expects "messages"
        "input": [m.dict() for m in req.messages],
        "messages": [m.dict() for m in req.messages],  # include both for compatibility
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

def map_responses_to_openai(raw: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Map Venice Responses/chat JSON to OpenAI Chat Completions format."""
    content = None
    out = raw.get("output")
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict):
            content = first.get("content")
    if content is None and isinstance(raw.get("response"), (str, int, float)):
        content = str(raw["response"])
    if content is None:
        # if it's already an OpenAI-shaped completion, try to read it
        try:
            ch = raw.get("choices", [{}])[0].get("message", {}).get("content")
            if ch:
                content = ch
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

# ElevenLabs will call this path:
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    venice_body = map_openai_to_responses(req)
    raw = await call_venice(venice_body)
    # bubble up a clearer error if none of the candidates worked
    if "_proxy_error" in raw:
        return {"error": raw["_proxy_error"], "detail": raw, "hint": "Check which /v1/* route your Venice account exposes."}
    # include which upstream URL succeeded for debugging
