import json, time
from typing import Any, Dict, List, Optional
import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Venice OpenAI Proxy", version="v3")

# You told me this is a test key and authorized its use in code.
VENICE_API_KEY = "KdJ46znt6ZZ0I6fFRlzCadu7SJrfUszhlZUBF9M-2s"

# Your Venice tenant does NOT serve /v1/chat/completions; it serves /v1/responses.
VENICE_BASE = "https://api.venice.ai/v1"
VENICE_RESP = f"{VENICE_BASE}/responses"

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

# >>> This route MUST exist; Eleven Labs calls /v1/chat/completions
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Map OpenAI Chat body -> Venice Res
