"""MCP 工具服务端 —— 项目中工具定义的唯一来源。

关键设计：所有工具返回 {"ok": bool, "data": Any, "error": str | None}，
**绝不向客户端抛异常**。

原因（实测确认）：MCP 协议会向客户端隐藏工具异常细节。工具抛异常时，
客户端只拿到 is_error=True 与通用文案 "Error executing tool X"，
真实异常仅记录在服务端。若工具靠抛异常报错，ReAct Agent 将收到
一个无法据以决策的哑错误，无法判断是重试、换工具还是降级。

工具内部用 asyncio.to_thread 包装同步实现，避免 BGE-M3 推理阻塞事件循环。
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from mcp.server import MCPServer

from app.tools.document_parser import parse_document as _parse_document
from app.tools.document_search import search_documents as _search_documents
from app.tools.web_search import search_web as _search_web

mcp_server = MCPServer(
    name="rag-agent-tools",
    version="0.1.0",
    description="智能问答平台的工具集：私有知识库检索、联网搜索、文档解析。",
)


def _envelope(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """把「抛异常」的同步函数转成结构化结果。"""
    try:
        return {"ok": True, "data": fn(*args, **kwargs), "error": None}
    except Exception as exc:  # noqa: BLE001 — 工具边界必须兜住一切
        return {"ok": False, "data": None, "error": f"{type(exc).__name__}: {exc}"}


@mcp_server.tool()
async def search_documents(query: str, top_k: int = 4) -> dict[str, Any]:
    """在私有文档知识库中做混合检索（语义 + 关键词），返回最相关的文档片段。"""
    return await asyncio.to_thread(_envelope, _search_documents, query, top_k)


@mcp_server.tool()
async def search_web(query: str, num: int = 5) -> dict[str, Any]:
    """在互联网上搜索最新信息，用于时效性问题和实时数据。"""
    return await asyncio.to_thread(_envelope, _search_web, query, num)


@mcp_server.tool()
async def parse_document(file_path: str) -> dict[str, Any]:
    """解析上传的文档（PDF/TXT），提取文本内容并建立索引。"""
    return await asyncio.to_thread(_envelope, _parse_document, file_path)
