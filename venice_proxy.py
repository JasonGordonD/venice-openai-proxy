# venice_proxy.py
# FINAL (v13): OpenAI-compatible proxy to Venice for ElevenLabs Agents.
# - Cleans ElevenLabs payloads (drops/normalizes tools, stream_options, input, etc.).
# - Coerces invalid roles (e.g., "string") to "user"; drops assistant turns before first user.
# - Coerces non-string message content to strings.
# - Forces NON-STREAMING replies (stable for ElevenLabs, which often sets stream:true).
# - Adds safe defaults: model fallback to DEFAULT_MODEL (venice-uncensored).
# - Returns strict OpenAI chat.completions JSON (or wraps Venice responses into that shape).
# - Forwards real upstream HTTP status codes, logs upstream error text & duration.
# - Root (/) + HEAD and /health + HEAD return 200 to avoid probe “server error” banners.
# - Optional: disable Venice default system prompt via VENICE_DISABLE_SYSTEM_PROMPT=1.

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Response, Request
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# App & Config
# ------------------------------------------------------------------------------
app = FastAPI(title="Venice OpenAI Proxy", version="v13")

VENICE_API_KEY: str = os.getenv("VENICE_API_KEY", "").strip()
VENICE_ENDPOINT: str = os.getenv(
    "VENICE_ENDPOINT", "https://api.venice.ai/api/v1/chat/completions"
).strip()

# Model fallback if incoming model is missing/placeholder/invalid
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "venice-uncensored").strip()

# Timeout for upstream HTTP calls (seconds)
HTTPX_TIMEOUT: float = float(os.getenv("HTTPX_TIMEOUT", "60"))

# Optional: set to 1/true to drop Venice default system prompt
VENICE_DISABLE_SYSTEM_PROMPT: bool = os.getenv(
    "VENICE_DISABLE_SYSTEM_PROMPT", "0"
).strip() not in ("0", "false", "False")

if not VENICE_API_KEY:
    logging.warning("VENICE_API_KEY is not set; upstream calls will fail with 401")

if not VENICE_ENDPOINT:
    VENICE_ENDPOINT = "https://api.venice.ai/api/v1/chat/completions"

_client: Optional[httpx.AsyncClient] = None
logging.basicConfig(level=logging.INFO)

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


@app.on_event("startup")
async def _startup() -> None:
    global _client
    _client = httpx.AsyncClient(timeout=HTTPX_TIMEOUT)


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        finally:
            _client = None


# ------------------------------------------------------------------------------
# Models (permissive; keep unknown keys)
# ------------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Any
    name: Optional[str] = None

    model_config = {"extra": "allow"}  # retain unknown per-message keys


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False   # will be forced False for EL stability
    stop: Optional[Any] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[Any] = None
    tool_choice: Optional[Any] = None
    user: Optional[str] = None
    # Allow arbitrary extra keys (e.g., ElevenLabs may add stream_options)
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}  # retain unknown top-level keys


# ------------------------------------------------------------------------------
# Root, Health, HEADs (avoid probe failures)
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "service": "Venice OpenAI Proxy", "version": "v13"}


@app.head("/")
async def head_root():
    return Response(status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok", "ts": int(time.time()), "version": "v13"}


@app.head("/health")
async def head_health():
    return Response(status_code=200)


# ------------------------------------------------------------------------------
# Sanitizers & Builders
# ------------------------------------------------------------------------------
def _normalize_model(model: Any) -> str:
    """
    Use DEFAULT_MODEL if model is missing, not a string, blank, or the swagger placeholder "string".
    """
    if not isinstance(model, str):
        return DEFAULT_MODEL
    m = model.strip()
    if not m or m.lower() == "string":
        return DEFAULT_MODEL
    return m


def _coerce_messages_to_strings(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every message.content is a string; stringify objects/arrays safely."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except Exception:
                content = str(content)
        entry: Dict[str, Any] = {"role": role, "content": content}
        if "name" in m and m["name"] is not None:
            entry["name"] = m["name"]
        out.append(entry)
    return out


def _normalize_roles_and_prune(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    - Coerce invalid roles to 'user'.
    - Always keep 'system'.
    - Keep 'assistant' only after the first 'user' has appeared.
    """
    # Coerce invalid roles
    tmp: List[Dict[str, Any]] = []
    for m in msgs:
        role = m.get("role")
        role = role.lower().strip() if isinstance(role, str) else "user"
        if role not in ALLOWED_ROLES:
            role = "user"
        m2 = dict(m)
        m2["role"] = role
        tmp.append(m2)

    # Prune assistant before first user
    cleaned: List[Dict[str, Any]] = []
    seen_user = False
    for m in tmp:
        role = m["role"]
        if role == "system":
            cleaned.append(m)
            continue
        if role == "user":
            seen_user = True
            cleaned.append(m)
            continue
        if role == "assistant":
            if seen_user:
                cleaned.append(m)
            continue
        # allow 'tool' or unknown future roles to pass
        cleaned.append(m)
    return cleaned


def _strip_problem_fields(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize incoming payload so Venice /chat/completions accepts it
    AND ElevenLabs UI doesn't error on tool/stream mismatches.
    """
    # Remove fields Venice doesn't accept on chat/completions
    for k in ("stream_options", "tool_choice", "input", "max_output_tokens"):
        body.pop(k, None)

    # tools: keep only if it's a non-empty list; otherwise remove
    tools_val = body.get("tools", None)
    if not isinstance(tools_val, list) or len(tools_val) == 0:
        body.pop("tools", None)

    # response_format: null or dict schema -> remove (avoid Venice validation issues)
    rf = body.get("response_format")
    if rf is None or isinstance(rf, dict):
        body.pop("response_format", None)

    # user: null -> remove
    if body.get("user") is None:
        body.pop("user", None)

    # Normalize/clean messages
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs:
        msgs = _normalize_roles_and_prune(msgs)
        msgs = _coerce_messages_to_strings(msgs)
        body["messages"] = msgs

    # Drop explicit None keys
    for k in [k for k, v in body.items() if v is None]:
        body.pop(k, None)

    return body


def _build_chat_body(req: ChatRequest) -> Dict[str, Any]:
    """
    Build the OpenAI-style request body for Venice chat/completions.
    Always force non-streaming (stable for ElevenLabs).
    """
    body: Dict[str, Any] = {
        "model": _normalize_model(req.model),
        "messages": [m.model_dump() for m in req.messages] if req.messages else [],
        "temperature": req.temperature,
        "stream": False,  # force non-streaming for EL stability
    }
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if isinstance(req.stop, (list, str)):
        body["stop"] = req.stop

    # Merge unknown top-level fields retained by pydantic (extra=allow)
    as_dict = req.model_dump()
    for k, v in as_dict.items():
        if k not in body and k not in (
            "messages", "temperature", "max_tokens", "top_p", "stop", "stream", "model", "extra"
        ):
            body[k] = v

    # Merge explicit "extra" bag
    if req.extra:
        body.update(req.extra)

    # Optional: disable Venice default system prompt via env
    if VENICE_DISABLE_SYSTEM_PROMPT:
        body["venice_parameters"] = {"include_venice_system_prompt": False}

    # Normalize & coerce
    return _strip_problem_fields(body)


def _wrap_minimal(up_json: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Minimal wrapper to strict OpenAI chat.completion shape."""
    content: Optional[str] = None

    if "output" in up_json:
        out = up_json["output"]
        if isinstance(out, list) and out and isinstance(out[0], dict):
            content = out[0].get("content")

    if content is None and "response" in up_json:
        content = str(up_json["response"])

    if content is None:
        content = json.dumps(up_json, ensure_ascii=False)

    return {
        "id": up_json.get("id", f"chatcmpl-proxy-{int(time.time())}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": up_json.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
    }


def _as_openai(up_json: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Sanitize upstream OpenAI-shaped responses; otherwise wrap minimally.
    Ensures choices[].message has only role/content and removes Venice extras.
    """
    choices = up_json.get("choices")
    if isinstance(choices, list):
        try:
            for c in choices:
                if "message" in c and isinstance(c["message"], dict):
                    msg = c["message"]
                    role = msg.get("role", "assistant")
                    content = msg.get("content")
                    if content is None:
                        content = json.dumps(up_json, ensure_ascii=False)
                    c["message"] = {"role": role, "content": content}
                # Remove non-OpenAI keys if present
                for bad in ("refusal", "annotations", "audio", "function_call", "tool_calls", "reasoning_content", "stop_reason"):
                    c.pop(bad, None)
            # Remove Venice-specific top-level extras if present
            for bad in ("service_tier", "system_fingerprint", "prompt_logprobs", "kv_transfer_params", "venice_parameters"):
                up_json.pop(bad, None)
        except Exception:
            return _wrap_minimal(up_json, model)
        return up_json
    return _wrap_minimal(up_json, model)


async def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json",
    }


# ------------------------------------------------------------------------------
# Main route
# ------------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    global _client
    assert _client is not None, "HTTP client not initialized"

    # Build & sanitize body
    body = _build_chat_body(req)

    # Log a concise, safe summary of what we received vs forward
    try:
        incoming = await request.json()
    except Exception:
        incoming = {}
    first_role = None
    imsgs = incoming.get("messages")
    if isinstance(imsgs, list) and imsgs:
        first_role = imsgs[0].get("role")
    logging.info(
        "[proxy] incoming stream=%s tools=%s first_role=%s  -> forward model=%s stream=%s tools=%s",
        incoming.get("stream", None),
        "present" if incoming.get("tools") else ("null" if "tools" in incoming else "absent"),
        first_role,
        body.get("model"),
        body.get("stream"),
        "present" if body.get("tools") else "absent",
    )

    headers = await _headers()

    t0 = time.perf_counter()
    try:
        r = await _client.post(VENICE_ENDPOINT, headers=headers, json=body)
        dt_ms = int((time.perf_counter() - t0) * 1000)
        status = r.status_code
        logging.info("[proxy] POST %s -> %s in %dms", VENICE_ENDPOINT, status, dt_ms)
        ctype = r.headers.get("content-type", "application/json")

        if 200 <= status < 300:
            try:
                up_json = r.json()
            except Exception:
                raw = await r.aread()
                return Response(content=raw, status_code=status, media_type=ctype)
            out_json = _as_openai(up_json, body["model"])
            return Response(
                content=json.dumps(out_json, ensure_ascii=False).encode("utf-8"),
                status_code=200,
                media_type="application/json",
            )

        # Error: log upstream text & return real status
        txt = await r.aread()
        logging.error(
            "[proxy] upstream error %s in %dms: %s",
            status, dt_ms, txt[:500].decode("utf-8", "ignore")
        )
        err = {
            "error": "Upstream failed",
            "detail": {
                "_proxy_status": status,
                "_proxy_text": txt.decode("utf-8", "ignore"),
                "_url": VENICE_ENDPOINT,
            },
        }
        return Response(
            content=json.dumps(err, ensure_ascii=False).encode("utf-8"),
            status_code=status,
            media_type="application/json",
        )

    except httpx.HTTPError as e:
        logging.exception("Upstream call failed: %s", e)
        err = {"error": "Upstream failed", "detail": {"exception": str(e), "_url": VENICE_ENDPOINT}}
        return Response(
            content=json.dumps(err, ensure_ascii=False).encode("utf-8"),
            status_code=502,
            media_type="application/json",
        )


# ------------------------------------------------------------------------------
# Local dev entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("venice_proxy:app", host="0.0.0.0", port=int(os.getenv("PORT", "8013")))
