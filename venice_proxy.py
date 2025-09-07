from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import httpx, os, re, time

app = FastAPI()

# --- CONFIG ---
VENICE_API_KEY = os.getenv("VENICE_API_KEY", "").strip()
# Your tenant returns 404 on /chat/completions; use Responses API:
VENICE_BASE = "https://api.venice.ai/v1"
VENICE_RESPONSES_URL = f"{VENICE_BASE}/responses"

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

    class Config:
        extra = "allow"

# ---------- simple router (optional) ----------
NSFW_HINTS = re.compile(
    r"(nsfw|explicit|porn|sexual|domme|findom|humiliat|degrad|edg(?:e|ing)|"
    r"orgasm|chastit|cbt|sissy|feet|heels|submiss|mistress|obey|kneel|pathetic|"
    r"worthless|good\s+girl)",
    re.IGNORECASE,
)
ANALYTIC_HINTS = re.compile(
    r"(reason|analy|explain|compare|contrast|evaluate|step-by-step|tradeoff|"
    r"limitations|design)",
    re.IGNORECASE,
)

DEFAULT_MODEL = "llama-3.3-70b"
NSFW_MODEL   = "venice-uncensored"
SMART_MODEL  = "qwen3-235b"

def _pull_user_text(messages: List[ChatMessage]) -> str:
    chunks: List[str] = []
    for m in messages:
        if m.role != "user":
            continue
        if isinstance(m.content, str):
            chunks.append(m.content)
        elif isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict) and "text" in part:
                    chunks.append(str(part["text"]))
    return " ".join(chunks)[:10000]

def choose_model(req: ChatRequest) -> str:
    if req.model and req.model.lower() != "auto":
        return req.model
    txt = _pull_user_text(req.messages)
    if NSFW_HINTS.search(txt):
        return NSFW_MODEL
    if ANALYTIC_HINTS.search(txt) or len(txt) > 1200:
        return SMART_MODEL
    return DEFAULT_MODEL

@app.get("/health")
async def health():
    return {"ok": True, "ts": int(time.time()), "ver": "sanitized-v3"}

# ------
