"""对话 API：把 Agent 的事件流转成 SSE。

事件格式统一为 `data: {"type": "...", ...}`，前端按 `event:` 名分发。
另有两个非协议事件：`status`（处理中）与 `done`（携带 conversation_id 与最终答案）。

**真流式**：token 直接来自 `model.astream()`，到达即转发。旧实现等 LLM 全量返回后
按 4 个字符切片、每次 sleep 20ms 伪造成流式 —— 所有帧几乎同时到达，首 token 延迟等于
整段生成耗时。

关于 `has_docs`：请求体里不再声明。ReAct 路径下由模型自主决定是否调用检索工具，
客户端声明「有没有文档」不再影响任何决策 —— 留着它只会让人以为它还有作用。
（前端停止发送它属于同一次改动的配套修改。）
"""
import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.agent.protocol import ErrorEvent, FinalEvent, to_sse
from app.db.session import SessionLocal
from app.services.chat_service import stream_agent
from app.services.conversation_service import (
    create_conversation,
    get_conversation_with_context,
    save_message,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    question: str
    history: list[dict] = []
    max_iterations: int = 8


def _sse(event_type: str, payload: dict) -> dict[str, str]:
    return {"event": event_type, "data": json.dumps(payload, ensure_ascii=False, default=str)}


async def _stream_chat(request: ChatRequest, http_request: Request) -> AsyncGenerator[dict, None]:
    db = SessionLocal()
    loop = asyncio.get_event_loop()

    try:
        if request.conversation_id:
            # 摘要会调 LLM，必须挪出事件循环
            ctx = await loop.run_in_executor(
                None, get_conversation_with_context, db, request.conversation_id
            )
            conversation_id = request.conversation_id
            recent = ctx["messages"][-6:]
            history_for_agent = [{"role": m["role"], "content": m["content"]} for m in recent]
            summary = ctx.get("summary")
        else:
            title = request.question[:30] + ("..." if len(request.question) > 30 else "")
            conv = create_conversation(db, title=title)
            conversation_id = conv.id
            history_for_agent = request.history
            summary = None

        save_message(db, conversation_id, "user", request.question)
        yield _sse("status", {"status": "thinking"})

        tools = http_request.app.state.mcp_client
        answer = ""
        had_error = False

        async for event in stream_agent(
            question=request.question,
            history=history_for_agent,
            summary=summary,
            tools=tools,
            max_iterations=request.max_iterations,
        ):
            if isinstance(event, FinalEvent):
                answer = event.answer
            elif isinstance(event, ErrorEvent):
                had_error = True
            yield to_sse(event)

        # 只在真的拿到答案时才落助手消息：Agent 以 ErrorEvent 收尾时（空响应 /
        # 达到迭代上限）写一条空记录，只会让历史里出现无法解释的空洞。
        if answer:
            save_message(db, conversation_id, "assistant", answer, source_type="agent")

        # 用户消息无论如何都要留下：这一轮问过就是问过了，即便回答失败。
        db.commit()

        yield _sse("done", {"conversation_id": conversation_id, "answer": answer})
        if had_error:
            logger.info("本轮以 ErrorEvent 结束：conversation_id=%s", conversation_id)

    except Exception as exc:  # noqa: BLE001 —— SSE 已开始，无法再改 HTTP 状态码
        db.rollback()
        # 结构化回报而不是让异常逃逸：EventSourceResponse 只会把异常变成一个断掉的流，
        # 前端拿不到任何可读原因。
        yield _sse("error", {
            "type": "error",
            "code": "server_error",
            "message": f"{type(exc).__name__}: {exc}",
            "recoverable": False,
        })
    finally:
        db.close()


@router.post("/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    return EventSourceResponse(_stream_chat(request, http_request))