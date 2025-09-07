import json, time, httpx
from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Venice OpenAI Proxy", version="v3")

# --- Your experimental Venice API key (hard-coded here) ---
VENICE_API_KEY = "KdJ46znt6ZZ0I6fFRlzCadu7SJrfUszhlZUBF9M-2s"

# Venice base
VENICE_BASE    = "https://api.venice.ai/v1"
VENICE_RESP    = f"{VENICE_BASE}/responses"  # your account supports this

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
    return {"ok": True, "ts": int(time.time()), "ver": "v3"}

# --- OpenAI-compatib
