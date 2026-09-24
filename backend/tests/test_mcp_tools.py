import pytest
from mcp.server import MCPServer

from app.mcp.client import MCPToolClient, open_tool_client
from app.mcp.server import _envelope


def _boom() -> dict:
    raise ValueError("数据库连接失败")


def make_stub_server() -> MCPServer:
    server = MCPServer(name="stub-tools", version="0.1.0")

    @server.tool()
    async def search_documents(query: str, top_k: int = 4) -> dict:
        """检索私有文档库。"""
        return {"ok": True, "data": [{"doc_id": "d1", "content": f"关于 {query} 的片段"}], "error": None}

    @server.tool()
    async def always_fails() -> dict:
        """用 _envelope 包装必抛异常的函数，模拟工具内部错误。"""
        return _envelope(_boom)

    return server


def test_envelope_turns_exception_into_structured_failure():
    result = _envelope(_boom)

    assert result["ok"] is False
    assert result["data"] is None
    assert "ValueError" in result["error"]
    assert "数据库连接失败" in result["error"]


def test_envelope_wraps_success():
    result = _envelope(lambda: {"answer": 42})

    assert result == {"ok": True, "data": {"answer": 42}, "error": None}


@pytest.mark.anyio
async def test_client_lists_tools_with_generated_schemas():
    async with open_tool_client(make_stub_server()) as client:
        names = {spec.name for spec in client.specs}

    assert names == {"search_documents", "always_fails"}


@pytest.mark.anyio
async def test_openai_schema_shape_matches_langchain_expectation():
    async with open_tool_client(make_stub_server()) as client:
        schema = next(s for s in client.specs if s.name == "search_documents").to_openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search_documents"
    assert "query" in schema["function"]["parameters"]["properties"]


@pytest.mark.anyio
async def test_successful_call_unwraps_structured_content():
    """mcp 把结构化返回值包在 {"result": ...} 里，客户端必须拆包。"""
    async with open_tool_client(make_stub_server()) as client:
        result = await client.call("search_documents", {"query": "合同解除"})

    assert result.ok is True
    assert result.data["data"][0]["doc_id"] == "d1"
    assert result.ms >= 0


@pytest.mark.anyio
async def test_tool_internal_failure_is_visible_to_the_agent():
    """这是本任务的核心设计点。

    MCP 协议会向客户端隐藏工具抛出的异常细节（只剩 "Error executing tool X"），
    所以工具必须把错误作为**正常返回值**回传，Agent 才能据此决策重试或降级。
    """
    async with open_tool_client(make_stub_server()) as client:
        result = await client.call("always_fails", {})

    assert result.ok is False
    assert "数据库连接失败" in result.error


@pytest.mark.anyio
async def test_unknown_tool_returns_ok_false_instead_of_raising():
    async with open_tool_client(make_stub_server()) as client:
        result = await client.call("no_such_tool", {})

    assert result.ok is False
    assert result.error
