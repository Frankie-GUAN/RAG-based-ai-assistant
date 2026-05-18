import json
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.services.chat_service import run_agent

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    question: str
    history: list[dict] = []
    has_docs: bool = False


async def _stream_chat(request: ChatRequest) -> AsyncGenerator[dict, None]:
    # Send start event
    yield {"event": "status", "data": json.dumps({"status": "thinking"})}

    # Run agent in thread pool (LangGraph is sync)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, run_agent, request.question, request.history, request.has_docs
    )

    yield {"event": "route", "data": json.dumps({"route": result["route"]})}

    # Stream answer token-by-token (4 char chunks simulate token streaming)
    answer = result["answer"]
    for i in range(0, len(answer), 4):
        chunk = answer[i:i + 4]
        yield {"event": "token", "data": json.dumps({"content": chunk})}
        await asyncio.sleep(0.02)

    yield {"event": "done", "data": json.dumps({"answer": answer, "route": result["route"]})}


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    return EventSourceResponse(_stream_chat(request))
