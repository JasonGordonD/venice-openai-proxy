from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import httpx, os, time, re

app = FastAPI()

VENICE_API_KEY = os.getenv("VENICE_API_KEY")
VENICE_BASE_URL = "https://api.venice.ai/api/v1/chat/completions"

# ---------- permissive request schema ----------
class ChatMessage(BaseModel):
    role: str
    content: Any            # allow string OR structured content
    name: Optional[str] = None

class ChatRequest(BaseModel):
    model: str              # "auto" enables routing; any other value is respected
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
        extra = "allow"     # ignore unknown fields instead of failing

# ---------- simple heuristics ----------
NSFW_HINTS = re.compile(
    r"(nsfw|explicit|porn|sexual|domme|findom|humiliat|degrad|edg(?:e|ing)|"
    r"orgasm|chastit|cbt|slut|cuck|ruin(?:ed)?\s*orgasm|sissy|feet|heels|"
    r"submiss|dominant|mistress|obey|kneel|pathetic|worthless|good\s+girl)",
    re.IGNORECASE,
)

ANALYTIC_HINTS = re.compile(
    r"(reason|analy|explain|compare|contrast|evaluate|step-by-step|chain of thought|"
    r"derive|proof|why|how|tradeoff|limitations|design)",
    re.IGNORECASE,
)

DEFAULT_MODEL = "llama-3.3-70b"
NSFW_MODEL   = "venice-uncensored"
SMART_MODEL  = "qwen3-235b"

def choose_model(req: ChatRequest) -> str:
    # respect explicit model unless "auto"
    if req.model and req.model.lower() != "auto":
        return req.model

    # pull last user content
    user_texts = []
    for m in req.messages[::-1]:  # reverse scan for recent emphasis
        if m.role == "user" and isinstance(m.content, str):
            user_texts.append(m.content)
        elif m.role == "user" and isinstance(m.content, list):
            # tolerate structured content: pick text parts
            for part in m.content:
                if isinstance(part, dict) and "text" in part:
                    user_texts.append(str(part["text"]))
    joined = " ".join(user_texts)[:10000]

    # heuristics
    if NSFW_HINTS.search(joined):
        return NSFW_MODEL
    if ANALYTIC_HINTS.search(joined) or len(joined) > 1200:
        return SMART_MODEL
    return DEFAULT_MODEL

@app.get("/health")
async def health():
    return {"ok": True, "ts": int(time.time())}

@app.post("/chat/completions")
async def chat_completion(req: ChatRequest):
    # pick model (routing)
    target_model = choose_model(req)

    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "x-api-key": VENICE_API_KEY,           # some Venice stacks prefer this
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": target_model,
        "messages": [m.dict() for m in req.messages],
        "temperature": req.temperature,
    }
    # pass through optional fields if present
    for k in ("max_tokens","top_p","stream","stop","response_format","tools","tool_choice","user"):
        v = getattr(req, k, None)
        if v is not None:
            payload[k] = v
    if req.extra:
        payload.update(req.extra)

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(VENICE_BASE_URL, headers=headers, json=payload)
        try:
            data = r.json()
        except Exception:
            return {"error": f"Upstream status {r.status_code}", "text": r.text}

    # attach which model was used (handy for debugging)
    if isinstance(data, dict):
        data.setdefault("router", {})["selected_model"] = target_model
    return data
