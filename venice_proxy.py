import json, time
from typing import Any, Dict, List, Optional
import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Venice OpenAI Proxy", version="v5")

VENICE_API_KEY = "KdJ46znt6ZZ0I6fFRlzCadu7SJrfUszhlZUBF9M-2s"

# Try Venice Responses endpoint first
VENICE_ENDPOINT = "https://api.venice.ai/v1/responses"

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
    extra: Dict[str, Any] = Field(default_factory=dict)
    class Config: extra = "allow"

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time()), "ver": "v5"}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    body: Dict[str, Any] = {
        "model": req.model,
        "input": [m.dict() for m in req.messages],       # for /responses
        "messages": [m.dict() for m in req.messages],    # for chat routes
        "temperature": req.temperature,
    }
    if req.max_tokens is not None:
        body["max_output_tokens"] = req.max_tokens
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.extra:
        body.update(req.extra)

    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "x-api-key": VENICE_API_KEY,  # some stacks expect this
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(VENICE_ENDPOINT, headers=headers, json=body)
        try:
            raw = r.json()
        except Exception:
            return {"error": f"Upstream {r.status_code}", "text": await r.aread()}

    # Map back to OpenAI format
    content = None
    if isinstance(raw.get("output"), list) and raw["output"]:
        first = raw["output"][0]
        if isinstance(first, dict):
            content = first.get("content")
    if content is None and isinstance(raw.get("response"), (str, int, float)):
        content = str(raw["response"])
    if content is None:
        content = json.dumps(raw, ensure_ascii=False)

    return {
        "id": raw.get("id", "chatcmpl-router"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": raw.get("usage", {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0})
    }
