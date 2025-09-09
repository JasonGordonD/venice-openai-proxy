# venice_proxy.py
# FINAL (v16) — OpenAI-compatible proxy to Venice for ElevenLabs Agents.
# - Streaming: pass-through SSE when stream=true, JSON when stream=false
# - Payload hardening: clamp top_p/temperature, drop unknown keys, tools:null -> drop,
#   response_format null/object -> drop, map user_id->user, fix roles, content->string
# - Fallback model if "string"/blank, clean HEAD/GET / and /health to 200

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, AsyncIterator

import httpx
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# App & Config
# ------------------------------------------------------------------------------
app = FastAPI(title="Venice OpenAI Proxy", version="v16")

VENICE_API_KEY: str = os.getenv("VENICE_API_KEY", "").strip()
VENICE_ENDPOINT: str = os.getenv(
    "VENICE_ENDPOINT", "https://api.venice.ai/api/v1/chat/completions"
).strip()

# If the incoming model is missing/placeholder, use this one:
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "venice-uncensored").strip()

# Timeout for upstream HTTP calls (seconds)
HTTPX_TIMEOUT: float = float(os.getenv("HTTPX_TIMEOUT", "60"))

# Optional: set to 1/true to drop Venice default system prompt
VENICE_DISABLE_SYSTEM_PROMPT: bool = os.getenv(
    "VENICE_DISABLE_SYSTEM_PROMPT", "0"
).strip() not in ("0", "false", "False")

if not VENICE_API_KEY:
    logging.warning("VENICE_API_KEY is not set; upstream calls will fail with 401")

logging.basicConfig(level=logging.INFO)
_client: Optional[httpx.AsyncClient] = None

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}

# Only allow these top-level keys to reach Venice
ALLOWED_TOP_KEYS = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "top_p",
    "stream",
    "stop",
    "user",
    "tools",
    # Venice-specific:
    "venice_parameters",
    # OpenAI-compatible extras (safe if present):
    "n",
    "seed",
    "logprobs",
    "logit_bias",
    "presence_penalty",
    "frequency_penalty",
    "metadata",
}


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
# Models (permissive; we sanitize before forwarding)
# ------------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Any
    name: Optional[str] = None

    model_config = {"extra": "allow"}


class ChatRequest(BaseModel):
    model: Optional[str] = None
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
    # ElevenLabs sometimes sends this; map to OpenAI "user"
    user_id: Optional[str] = None
    # Allow arbitrary extra keys (e.g., stream_options, elevenlabs_extra_body)
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


# ------------------------------------------------------------------------------
# Root & Health & HEAD (keep probes green)
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "service": "Venice OpenAI Proxy", "version": "v16"}


@app.head("/")
async def head_root():
    return Response(status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok", "ts": int(time.time()), "version": "v16"}


@app.head("/health")
async def head_health():
    return Response(status_code=200)


# ------------------------------------------------------------------------------
# Sanitizers & Builders
# ------------------------------------------------------------------------------
def _normalize_model(model: Any) -> str:
    if not isinstance(model, str):
        return DEFAULT_MODEL
    m = model.strip()
    if not m or m.lower() == "string":
        return DEFAULT_MODEL
    return m


def _coerce_messages_to_strings(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    # Coerce invalid roles → "user"
    tmp: List[Dict[str, Any]] = []
    for m in msgs:
        role = m.get("role")
        role = role.lower().strip() if isinstance(role, str) else "user"
        if role not in ALLOWED_ROLES:
            role = "user"
        m2 = dict(m)
        m2["role"] = role
        tmp.append(m2)

    # Drop assistant turns that come before the first user
    cleaned: List[Dict[str, Any]] = []
    seen_user = False
    for m in tmp:
        role = m["role"]
        if role == "system":
            cleaned.append(m)
        elif role == "user":
            seen_user = True
            cleaned.append(m)
        elif role == "assistant":
            if seen_user:
                cleaned.append(m)
        else:  # 'tool' etc. — allow
            cleaned.append(m)
    return cleaned


def _normalize_sampling(body: Dict[str, Any]) -> None:
    # temperature clamp to [0, 2] (default 0.7 if invalid)
    if "temperature" in body:
        t = body["temperature"]
        if not isinstance(t, (int, float)):
            body["temperature"] = 0.7
        else:
            if t < 0:
                body["temperature"] = 0.0
            elif t > 2:
                body["temperature"] = 2.0

    # top_p must be in (0, 1]; set to 0.95 if invalid (e.g., 0.0 or > 1)
    if "top_p" in body:
        p = body["top_p"]
        if not isinstance(p, (int, float)) or p <= 0 or p > 1:
            body["top_p"] = 0.95

    # max_tokens must be positive int; else drop
    if "max_tokens" in body:
        mt = body["max_tokens"]
        if not isinstance(mt, int) or mt <= 0:
            body.pop("max_tokens", None)

    # stop must be str or list; else drop
    if "stop" in body and not isinstance(body["stop"], (str, list)):
        body.pop("stop", None)


def _filter_allowed_top_keys(body: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in body.items() if k in ALLOWED_TOP_KEYS}


def _strip_problem_fields(body: Dict[str, Any]) -> Dict[str, Any]:
    # Remove fields Venice doesn't accept on chat/completions
    for k in ("stream_options", "tool_choice", "input", "max_output_tokens",
              "elevenlabs_extra_body", "response_format"):
        body.pop(k, None)

    # tools: keep only non-empty list; otherwise remove
    tools_val = body.get("tools", None)
    if not isinstance(tools_val, list) or len(tools_val) == 0:
        body.pop("tools", None)

    # user: null → remove
    if body.get("user") is None:
        body.pop("user", None)

    # Normalize & clean messages
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs:
        msgs = _normalize_roles_and_prune(msgs)
        msgs = _coerce_messages_to_strings(msgs)
        body["messages"] = msgs

    # Fix sampling
    _normalize_sampling(body)

    # Drop explicit None keys
    for k in [k for k, v in body.items() if v is None]:
        body.pop(k, None)

    # Keep only allowed top-level keys (removes unknowns like "additionalProp1")
    body = _filter_allowed_top_keys(body)
    return body


def _build_chat_body(req: ChatRequest) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": _normalize_model(req.model),
        "messages": [m.model_dump() for m in req.messages] if req.messages else [],
        "temperature": req.temperature,
        "stream": bool(req.stream),  # pass through streaming intent
    }

    # Map user_id -> user if provided (EL pattern)
    if req.user:
        body["user"] = req.user
    elif req.user_id:
        body["user"] = req.user_id

    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if isinstance(req.stop, (list, str)):
        body["stop"] = req.stop

    # Optionally disable Venice default system prompt via env
    if VENICE_DISABLE_SYSTEM_PROMPT:
        body["venice_parameters"] = {"include_venice_system_prompt": False}

    return _strip_problem_fields(body)


def _wrap_minimal(up_json: Dict[str, Any], model: str) -> Dict[str, Any]:
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
                for bad in ("refusal", "annotations", "audio", "function_call", "tool_calls",
                            "reasoning_content", "stop_reason"):
                    c.pop(bad, None)
            for bad in ("service_tier", "system_fingerprint", "prompt_logprobs",
                        "kv_transfer_params", "venice_parameters"):
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
    assert _client is not None, "HTTP client not initialized"

    body = _build_chat_body(req)

    # Minimal safe logging of inbound vs forwarded
    try:
        incoming = await request.json()
    except Exception:
        incoming = {}
    first_role = None
    imsgs = incoming.get("messages")
    if isinstance(imsgs, list) and imsgs:
        first_role = imsgs[0].get("role")
    logging.info(
        "[proxy] incoming stream=%s top_p=%s first_role=%s -> forward model=%s stream=%s top_p=%s",
        incoming.get("stream", None),
        incoming.get("top_p", None),
        first_role,
        body.get("model"),
        body.get("stream"),
        body.get("top_p", None),
    )

    headers = await _headers()
    t0 = time.perf_counter()

    # If EL wants streaming, pass SSE through verbatim
    if body.get("stream") is True:
        async def stream_gen() -> AsyncIterator[bytes]:
            try:
                async with _client.stream("POST", VENICE_ENDPOINT, headers=headers, json=body) as r:
                    logging.info("[proxy] POST %s -> %s (stream)", VENICE_ENDPOINT, r.status_code)
                    async for chunk in r.aiter_raw():
                        # Upstream emits SSE; pass through untouched
                        yield chunk
            except httpx.HTTPError as e:
                logging.exception("Upstream stream error: %s", e)
                # Emit a final SSE error line so EL can end the stream gracefully
                yield b'data: {"error":"Upstream stream failed"}\n\n'
        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    # Non-streaming path (single JSON)
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

        txt = await r.aread()
        logging.error("[proxy] upstream error %s in %dms: %s",
                      status, dt_ms, txt[:500].decode("utf-8", "ignore"))
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
