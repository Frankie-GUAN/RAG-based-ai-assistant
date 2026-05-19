import json
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.services.chat_service import run_agent
from app.services.conversation_service import (
    create_conversation,
    save_message,
    get_conversation_with_context,
)
from app.db.session import SessionLocal
from app.rag.vector_store import has_persisted_index

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    question: str
    history: list[dict] = []
    has_docs: bool = False


async def _stream_chat(request: ChatRequest) -> AsyncGenerator[dict, None]:
    db = SessionLocal()
    loop = asyncio.get_event_loop()
    try:
        if request.conversation_id:
            # Offload compression (LLM call) to thread to avoid blocking event loop
            ctx = await loop.run_in_executor(
                None, get_conversation_with_context, db, request.conversation_id
            )
            conversation_id = request.conversation_id
            recent_msgs = ctx["messages"][-6:]
            history_for_agent = [
                {"role": m["role"], "content": m["content"]}
                for m in recent_msgs
            ]
            summary = ctx.get("summary")
        else:
            title = request.question[:30] + ("..." if len(request.question) > 30 else "")
            conv = create_conversation(db, title=title)
            conversation_id = conv.id
            history_for_agent = request.history
            summary = None

        # Save user message
        save_message(db, conversation_id, "user", request.question)

        yield {"event": "status", "data": json.dumps({"status": "thinking"})}

        # Auto-detect document availability for RAG routing
        effective_has_docs = request.has_docs or has_persisted_index()

        # Run agent in thread pool (LangGraph is sync)
        result = await loop.run_in_executor(
            None, run_agent, request.question, history_for_agent,
            effective_has_docs, summary
        )

        # Save assistant message
        save_message(
            db, conversation_id, "assistant", result["answer"],
            source_type=result.get("route"),
        )

        yield {"event": "route", "data": json.dumps({"route": result["route"]})}

        # Stream answer token-by-token (4 char chunks simulate token streaming)
        answer = result["answer"]
        for i in range(0, len(answer), 4):
            chunk = answer[i:i + 4]
            yield {"event": "token", "data": json.dumps({"content": chunk})}
            await asyncio.sleep(0.02)

        yield {
            "event": "done",
            "data": json.dumps({
                "answer": answer,
                "route": result["route"],
                "conversation_id": conversation_id,
            }),
        }

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    return EventSourceResponse(_stream_chat(request))
