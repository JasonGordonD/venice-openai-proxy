import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# App & config
# ------------------------------------------------------------------------------
app = FastAPI(title="Venice OpenAI Proxy", version="v7")

VENICE_API_KEY = os.getenv("VENICE_API_KEY", "").strip()
VENICE_ENDPOINT = os.getenv("VENICE_ENDPOINT", "https://api.venice.ai/api/v1/chat/completions").strip()

if not VENICE_API_KEY:
    logging.warning("VENICE_API_KEY is not set; upstream calls will fail with 401")

if not VENICE_ENDPOINT:
    VENICE_ENDPOINT = "https://api.venice.ai/api/v1/chat/completions"

HTTPX_TIMEOUT = float(os.getenv("HTTPX_TIMEOUT", "60"))

client = httpx.AsyncClient(timeout=HTTPX_TIMEOUT)

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
    # allow arbitrary extra keys (e.g., ElevenLabs may add stream_options)
    extra: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"  # keep unknown keys in .__dict__


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "ts": int(time.time()), "version": "v7"}

def _strip_problem_fields(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the incoming OpenAI-style payload so Venice /chat/completions accepts it.
    Removes or fixes fields that caused 400 in your logs:
      - tools: null -> remove (or keep [] if already list)
      - remove stream_options/tool_choice/input/max_output_tokens
      - optionally drop an initial assistant message on first turn
    """
    # Drop fields Venice doesn't accept on /chat/completions
    for k in ("stream_options", "tool_choice", "input", "max_output_tokens"):
        if k in body:
            body.pop(k, None)

    # tools: null → remove
    if "tools" in body and body["tools"] is None:
        body.pop("tools", None)

    # If messages exist and the very first message is "assistant" (before any user), drop it.
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs:
        if msgs[0].get("role") == "assistant":
            msgs.pop(0)

    # If response_format is a dict with schema, Venice may reject; remove while debugging.
    rf = body.get("response_format")
    if isinstance(rf, dict) and rf:
        # leave JSON mode etc. for later; safest is to drop for now
        body.pop("response_format", None)

    return body

def _build_chat_body(req: ChatRequest) -> Dict[str, Any]:
    # Base OpenAI chat/completions shape
    body: Dict[str, Any] = {
        "model": req.model,
        "messages": [m.dict() for m in req.messages] if req.messages else [],
        "temperature": req.temperature,
    }
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if isinstance(req.stop, (list, str)) or req.stop is None:
        # leave stop as-is if list/str/None; Venice follows OpenAI here
        if req.stop is not None:
            body["stop"] = req.stop
    if req.stream is not None:
        body["stream"] = bool(req.stream)

    # Merge any unknown fields, then strip the problem ones
    # (req.extra collects only top-level "extra", but Config.extra="allow" preserves others in __dict__)
    # So also merge anything else that might have been set at top-level.
    as_dict = req.dict()
    # Known top-level keys already handled; pick up potential extras like stream_options if they landed there
    for k, v in as_dict.items():
        if k not in body and k not in ("messages", "temperature", "max_tokens", "top_p", "stop", "stream", "model", "extra"):
            body[k] = v
    # Also merge explicit "extra" bag
    if req.extra:
        body.update(req.extra)

    body = _strip_problem_fields(body)
    return body

def _as_openai_if_needed(up_json: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    If upstream already returns an OpenAI-style response, pass it through.
    Otherwise, wrap minimally into OpenAI chat.completion shape.
    """
    if "choices" in up_json and isinstance(up_json["choices"], list):
        # Already OpenAI-compatible; pass through
        return up_json

    # Minimal adapter if Venice responds in a different shape
    # Try to find content
    content = None
    if "output" in up_json:
        out = up_json["output"]
        if isinstance(out, list) and out and isinstance(out[0], dict):
            content = out[0].get("content")
    if content is None:
        try:
            content = up_json.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:
            content = None
    if content is None and "response" in up_json:
        content = str(up_json["response"])
    if content is None:
        content = json.dumps(up_json, ensure_ascii=False)

    return {
        "id": up_json.get("id", "chatcmpl-proxy"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": up_json.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    }

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
    body = _build_chat_body(req)
    headers = await _headers()

    # Log a concise line so you can see status in Render logs
    logging.info("[proxy] forwarding to %s", VENICE_ENDPOINT)

    # If client asked for streaming, stream upstream SSE back to client.
    if body.get("stream") is True:
        async def stream_gen() -> AsyncIterator[bytes]:
            try:
                async with client.stream("POST", VENICE_ENDPOINT, headers=headers, json=body) as r:
                    logging.info("[proxy] POST %s -> %s (stream)", VENICE_ENDPOINT, r.status_code)
                    async for chunk in r.aiter_raw():
                        # pass upstream SSE as-is
                        yield chunk
            except httpx.HTTPError as e:
                logging.exception("Upstream stream error: %s", e)
                # Emit a minimal SSE error
                yield b'data: {"error":"Upstream stream failed"}\n\n'

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    # Non-streaming request
    try:
        r = await client.post(VENICE_ENDPOINT, headers=headers, json=body)
        status = r.status_code
        logging.info("[proxy] POST %s -> %s", VENICE_ENDPOINT, status)

        # Forward upstream content-type when possible
        ctype = r.headers.get("content-type", "application/json")

        # On success, return JSON (wrapping if needed)
        if 200 <= status < 300:
            try:
                up_json = r.json()
            except Exception:
                # If upstream returned non-JSON, forward raw
                return Response(content=await r.aread(), status_code=status, media_type=ctype)
            return Response(
                content=json.dumps(_as_openai_if_needed(up_json, req.model)).encode("utf-8"),
                status_code=200,
                media_type="application/json"
            )

        # On error, forward Venice error with small envelope
        txt = await r.aread()
        err = {
            "error": "Upstream failed",
            "detail": {
                "_proxy_status": status,
                "_proxy_text": txt.decode("utf-8", "ignore"),
                "_url": VENICE_ENDPOINT
            }
        }
        return Response(content=json.dumps(err).encode("utf-8"), status_code=200, media_type="application/json")

    except httpx.HTTPError as e:
        logging.exception("Upstream call failed: %s", e)
        err = {"error": "Upstream failed", "detail": {"exception": str(e), "_url": VENICE_ENDPOINT}}
        return Response(content=json.dumps(err).encode("utf-8"), status_code=200, media_type="application/json")
