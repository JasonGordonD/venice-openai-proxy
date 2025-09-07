@app.post("/chat/completions")
async def chat_completion(req: ChatRequest):
    target_model = choose_model(req)

    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "x-api-key": VENICE_API_KEY,     # supports both auth styles
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": target_model,
        "messages": [m.dict() for m in req.messages],
        "temperature": req.temperature,
    }
    for k in ("max_tokens","top_p","stream","stop","response_format","tools","tool_choice","user"):
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

    # ---- sanitize to pure OpenAI schema ----
    clean: Dict[str, Any] = {
        "id": raw.get("id"),
        "object": raw.get("object", "chat.completion"),
        "created": raw.get("created"),
        "model": raw.get("model", target_model),
        "choices": raw.get("choices", []),
        "usage": raw.get("usage", {}),
    }
    return clean
