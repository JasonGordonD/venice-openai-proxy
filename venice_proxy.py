from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

VENICE_API_KEY = os.getenv("VENICE_API_KEY")
VENICE_BASE_URL = "https://api.venice.ai/api/v1/chat/completions"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7

@app.post("/chat/completions")
async def chat_completion(request: ChatRequest):
    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": request.model,
        "messages": [msg.dict() for msg in request.messages],
        "temperature": request.temperature,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(VENICE_BASE_URL, headers=headers, json=payload)
    return response.json()