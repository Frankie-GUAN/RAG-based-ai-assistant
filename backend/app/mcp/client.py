"""MCP 客户端封装。

与 MCP Server **同进程**运行（InMemoryTransport）。
不可改用 stdio 子进程：BGE-M3 约 2GB，子进程会再加载一份模型。
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from mcp.client._memory import InMemoryTransport
from mcp.client.session import ClientSession
from mcp.server import MCPServer


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    ms: int = 0


class MCPToolClient:
    def __init__(self, session: ClientSession) -> None:
        self._session = session
        self._specs: list[ToolSpec] = []

    @property
    def specs(self) -> list[ToolSpec]:
        return self._specs

    async def refresh_tools(self) -> list[ToolSpec]:
        listed = await self._session.list_tools()
        self._specs = [
            ToolSpec(name=t.name, description=t.description or "", input_schema=t.input_schema)
            for t in listed.tools
        ]
        return self._specs

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()

        try:
            result = await self._session.call_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001 — 传输层自身出错（连接 / 协议层）
            # 注意这里**不是**「工具名不存在」那条路：未知工具走的是 is_error=True 与
            # "Unknown tool: X"，由下面那条分支返回（InMemoryTransport 的
            # raise_exceptions 默认 False）。实测本分支未被触发过。
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}", ms=_elapsed_ms(started))

        ms = _elapsed_ms(started)

        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            return ToolResult(ok=False, error=text or "工具调用失败", ms=ms)

        payload = result.structured_content
        if payload is None:
            payload = _json_from_text(result.content)
        # 仅当返回注解不是 dict 时，mcp 才把结构化结果包在 {"result": ...} 里
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]

        # 工具用 {"ok", "data", "error"} 信封报错（见 app/mcp/server.py）。信封必须在此
        # 拆开：调用本身是成功的（is_error=False），不上报 ok=False 的话，工具失败与空结果
        # 在 Agent 眼里无从区分 —— 那正是本任务要消灭的哑错误。
        if isinstance(payload, dict) and set(payload) == {"ok", "data", "error"}:
            return ToolResult(ok=bool(payload["ok"]), data=payload, error=payload["error"], ms=ms)

        return ToolResult(ok=True, data=payload, ms=ms)


def _json_from_text(content: list[Any]) -> Any:
    """`structured_content` 为 None 时的兜底：从文本内容里还原 JSON。

    工具的返回注解是**裸** `dict` 时 mcp 不生成 output schema，也就没有
    structured_content，JSON 只存在于文本内容里。注解写成 `dict[str, Any]`
    时 structured_content 才是那个 dict 本身。
    """
    texts = [c.text for c in content if getattr(c, "text", None)]
    if not texts:
        return None
    raw = texts[0] if len(texts) == 1 else " ".join(texts)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


@asynccontextmanager
async def open_tool_client(server: MCPServer) -> AsyncIterator[MCPToolClient]:
    """打开一个与指定 MCP Server 同进程连接的客户端，并在退出时清理。"""
    async with InMemoryTransport(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            client = MCPToolClient(session)
            await client.refresh_tools()
            yield client
