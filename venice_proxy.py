# venice_proxy.py
# ROV-M + RIGOR: OpenAI-compatible proxy to Venice for ElevenLabs Agents.
# - Cleans ElevenLabs payloads (tools:null -> [], strips stream_options/input/etc.).
# - Removes any 'assistant' turns before the first 'user' turn (keep 'system').
# - Coerces non-string message content to string.
# - Returns strict OpenAI chat.completions JSON.
# - Forwards real upstream error status codes (no 200-wrapping on errors).
# - Defaults to NON-STREAMING for safer ElevenLabs integration (override via PROXY_FORCE_NON_STREAM=0).

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, AsyncIterator

import httpx
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# App & Config
# ------------------------------------------------------------------------------
app = FastAPI(title="Venice OpenAI Proxy", version="v9.1")

VENICE_API_KEY: str = os.getenv("VENICE_API_KEY", "").strip()
VENICE_ENDPOINT: str = os.getenv(
    "VENICE_ENDPOINT",
    "https://api.venice.ai/api/v1/chat/completions",
).strip()

# Force non-streaming replies by default (safer with some clients).
PROXY_FORCE_NON_STREAM: bool = os.getenv("PROXY_FORCE_NON_STREAM", "1").strip() not in ("0", "false", "False")

HTTPX_TIMEOUT: float = float(os.getenv("HTTPX_TIMEOUT", "60"))

if not VENICE_API_KEY:
    logging.warning("VENICE_API_KEY is not set; upstream calls will fail with 401")

if not VENICE_ENDPOINT:
    VENICE_ENDPOINT = "https://api.venice.ai/api/v1/chat/completions"

_client: Optional[httpx.AsyncClient] = None


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
# Models (accept permissive input from clients)
# ------------------------------------------------------------------------------
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
    # Allow arbitrary extra keys (e.g., ElevenLabs may add stream_options)
    extra: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"  # keep unknown top-level keys in the dict


# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "ts": int(time.time()), "version": "v9.1"}


# ------------------------------------------------------------------------------
# Sanitizers & Builders
# ------------------------------------------------------------------------------
def _coerce_messages_to_strings(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every message.content is a string; stringify objects/arrays for safety."""
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


def _strip_problem_fields(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the incoming payload so Venice /chat/completions accepts it
    AND ElevenLabs UI doesn't error on tool serialization mismatches.
    """
    # Remove fields Venice doesn't accept on chat/completions
    for k in ("stream_options", "tool_choice", "input", "max_output_tokens"):
        body.pop(k, None)

    # Normalize tools: null -> [] (keep array if already present)
    if body.get("tools") is None:
        body["tools"] = []

    # response_format: null or dict schema -> remove while debugging
    rf = body.get("response_format")
    if rf is None or isinstance(rf, dict):
        body.pop("response_format", None)

    # user: null -> remove
    if body.get("user") is None:
        body.pop("user", None)

    # Enforce: any 'assistant' messages BEFORE the first 'user' are dropped (keep 'system')
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs:
        cleaned: List[Dict[str, Any]] = []
        seen_user = False
        for m in msgs:
            role = m.get("role")
            if role == "system":
                cleaned.append(m)  # always keep system preface
                continue
            if role == "user":
                seen_user = True
                cleaned.append(m)
                continue
            if role == "assistant":
                if seen_user:
                    cleaned.append(m)  # assistant replies only after first user turn
                # else: drop assistant seed before any user
                continue
            # pass through unknown roles defensively
            cleaned.append(m)
        body["messages"] = cleaned

    # Drop any top-level keys with explicit None
    for k in [k for k, v in body.items() if v is None]:
        body.pop(k, None)

    # Coerce message contents to strings
    msgs2 = body.get("messages")
    if isinstance(msgs2, list):
        body["messages"] = _coerce_messages_to_strings(msgs2)

    return body


def _build_chat_body(req: ChatRequest) -> Dict[str, Any]:
    """
    Build the OpenAI-style request body for Venice chat/completions.
    """
    body: Dict[str, Any] = {
        "model": req.model,
        "messages": [m.dict() for m in req.messages] if req.messages else [],
        "temperature": req.temperature,
    }

    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if isinstance(req.stop, (list, str)):
        body["stop"] = req.stop
    if req.stream is not None:
        # Optionally force non-streaming to simplify ElevenLabs integration
        body["stream"] = False if PROXY_FORCE_NON_STREAM else bool(req.stream)

    # Merge unknown top-level fields from pydantic dict
    as_dict = req.dict()
    for k, v in as_dict.items():
        if k not in body and k not in (
            "messages", "temperature", "max_tokens", "top_p", "stop",
            "stream", "model", "extra"
        ):
            body[k] = v

    # Merge explicit "extra" bag
    if req.extra:
        body.update(req.extra)

    # Normalize & coerce
    body = _strip_problem_fields(body)
    return body


def _wrap_minimal(up_json: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Minimal wrapper to OpenAI chat.completion shape."""
    content: Optional[str] = None

    # Try common Venice shapes
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
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": up_json.get(
            "usage",
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        ),
    }


def _as_openai(up_json: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Always return strict OpenAI-compatible JSON (chat.completion).
    """
    choices = up_json.get("choices")
    if isinstance(choices, list):
        # Sanitize choices/messages and remove non-OpenAI keys
        try:
            for c in choices:
                if "message" in c and isinstance(c["message"], dict):
                    msg = c["message"]
                    role = msg.get("role", "assistant")
                    content = msg.get("content")
                    if content is None:
                        content = json.dumps(up_json, ensure_ascii=False)
                    c["message"] = {"role": role, "content": content}
                for bad in (
                    "refusal",
                    "annotations",
                    "audio",
                    "function_call",
                    "tool_calls",
                    "reasoning_content",
                    "stop_reason",
                ):
                    c.pop(bad, None)
            for bad in (
                "service_tier",
                "system_fingerprint",
                "prompt_logprobs",
                "kv_transfer_params",
                "venice_parameters",
            ):
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
async def chat_completions(req: ChatRequest):
    global _client
    assert _client is not None, "HTTP client not initialized"

    body = _build_chat_body(req)
    headers = await _headers()

    logging.info("[proxy] forwarding to %s", VENICE_ENDPOINT)

    # Streaming pass-through (optional; disabled by default via PROXY_FORCE_NON_STREAM)
    if body.get("stream") is True:
        async def stream_gen() -> AsyncIterator[bytes]:
            try:
                async with _client.stream(
                    "POST", VENICE_ENDPOINT, headers=headers, json=body
                ) as r:
                    logging.info("[proxy] POST %s -> %s (stream)", VENICE_ENDPOINT, r.status_code)
                    async for chunk in r.aiter_raw():
                        yield chunk  # pass upstream SSE through unchanged
            except httpx.HTTPError as e:
                logging.exception("Upstream stream error: %s", e)
                yield b'data: {"error":"Upstream stream failed"}\n\n'
        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    # Non-streaming path (default/safe)
    try:
        r = await _client.post(VENICE_ENDPOINT, headers=headers, json=body)
        status = r.status_code
        logging.info("[proxy] POST %s -> %s", VENICE_ENDPOINT, status)
        ctype = r.headers.get("content-type", "application/json")

        if 200 <= status < 300:
            try:
                up_json = r.json()
            except Exception:
                # Forward raw if upstream is non-JSON (unlikely for Venice)
                return Response(content=await r.aread(), status_code=status, media_type=ctype)
            # Strict OpenAI JSON out
            out_json = _as_openai(up_json, req.model)
            return Response(
                content=json.dumps(out_json, ensure_ascii=False).encode("utf-8"),
                status_code=200,
                media_type="application/json",
            )

        # Forward real status code on errors (no 200-wrapping)
        txt = await r.aread()
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
        err = {
            "error": "Upstream failed",
            "detail": {"exception": str(e), "_url": VENICE_ENDPOINT},
        }
        return Response(
            content=json.dumps(err, ensure_ascii=False).encode("utf-8"),
            status_code=502,
            media_type="application/json",
        )
