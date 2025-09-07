from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import httpx, os, re, time

# Create FastAPI app
app = FastAPI()

VENICE_API_KEY = os.getenv("VENICE_API_KEY")
VENICE_BASE_URL = "https://api.venice.ai/api/v1/chat/completions"

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

# ---------- simple router ----------
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
    chunks = []
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
    return {"ok": True, "ts": int(time.time()), "ver": "sanitized-v2"}

# ---------- sanitized handler ----------
@app.post("/chat/completions")
async def chat_completion(req: ChatRequest):
    target_model = choose_model(req)

    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "x-api-key": VENICE_API_KEY,
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": target_model,
        "messages": [m.dict() for m in req.messages],
        "temperature": req.temperature,
    }
    for k in ("max_tokens","top_p","stream","stop",
              "response_format","tools","tool_choice","user"):
        v = getattr(req, k, None)
        if v is not None:
            payload[k] = v
    if req.extra:
        payload.update(req.extra)

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(VENICE_BASE_URL, headers=headers, json=payload)
        try:
            raw = r.json()
        except Exception:
            return {"error": f"Upstream status {r.status_code}", "text": r.text}

    # sanitize to OpenAI spec
    clean: Dict[str, Any] = {
        "id": raw.get("id"),
        "object": raw.get("object", "chat.completion"),
        "created": raw.get("created"),
        "model": raw.get("model", target_model),
        "choices": raw.get("choices", []),
        "usage": raw.get("usage", {}),
    }
    return clean
