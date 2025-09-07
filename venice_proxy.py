import os, json, time, httpx
from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Venice OpenAI Proxy", version="v5")

# --- AUTH (you said this test key is okay to hard-code) ---
VENICE_API_KEY = os.getenv("VENICE_API_KEY", "KdJ46znt6ZZ0I6fFRlzCadu7SJrfUszhlZUBF9M-2s").strip()

# --- You can hard-set the exact upstream endpoint via env in Render if you learn it later.
# Example: VENICE_ENDPOINT=https://api.venice.ai/openai/v1/chat/completions
VENICE_ENDPOINT = os.getenv("VENICE_ENDPOINT", "").strip()

# Common Venice variants we will auto-try (in this order)
CANDIDATE_ENDPOINTS = [
    "https://api.venice.ai/v1/responses",             # Responses API (new)
    "https://api.venice.ai/openai/v1/responses",      # Responses under /openai
    "https://venice.ai/api/v1/responses",             # Alt domain/base
    "https://api.venice.ai/v1/chat/completions",      # Classic chat
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
    return {"ok": True, "ts": int(time.time()), "ver": "v5"}

def _make_body(req: ChatRequest) -> Dict[str, Any]:
    """Build a body that works for both Responses and Chat endpoints."""
    body: Dict[str, Any] = {
        "model": req.model,
        "input": [m.dict() for m in req.messages],       # for /responses
        "messages": [m.dict() for m in req.messages],    # for /chat/completions
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

def _to_openai(raw: D
