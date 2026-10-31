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

    @server.tool()
    async def returns_a_string() -> str:
        """非 dict 注解：只有这种注解下 mcp 才把结果包进 {"result": ...}。"""
        return "纯文本结果"

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


def test_envelope_key_set_is_a_wire_contract():
    """把信封的键集钉死 —— 客户端正是靠它判断「这是不是一个信封」。

    判别式是**失败开放**的：`set(payload) == {"ok","data","error"}` 一旦因为
    _envelope 多出（或少了）一个键而失配，工具失败就会被当成 ok=True 正常返回，
    也就是本任务要消灭的哑错误又回来了。钉住键集，这类改动会在这里响亮地失败。
    两个分支都要钉 —— 原先只有成功分支有断言。
    """
    assert set(_envelope(lambda: "值").keys()) == {"ok", "data", "error"}
    assert set(_envelope(_boom).keys()) == {"ok", "data", "error"}


@pytest.mark.anyio
async def test_client_lists_tools_with_generated_schemas():
    async with open_tool_client(make_stub_server()) as client:
        names = {spec.name for spec in client.specs}

    assert names == {"search_documents", "always_fails", "returns_a_string"}


@pytest.mark.anyio
async def test_openai_schema_shape_matches_langchain_expectation():
    async with open_tool_client(make_stub_server()) as client:
        schema = next(s for s in client.specs if s.name == "search_documents").to_openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search_documents"
    assert "query" in schema["function"]["parameters"]["properties"]


@pytest.mark.anyio
async def test_successful_call_returns_the_full_envelope_as_data():
    """成功调用：`ok` 为 True，`data` 里是**完整信封**（下游要用 `data["data"]`）。

    注意这条**不走** `{"result": ...}` 拆包分支：stub 的注解是裸 `-> dict`，mcp 不为它
    生成 output schema，`structured_content` 为 None，JSON 只存在于文本内容里，
    客户端走的是 `_json_from_text` 兜底。生产实际走的路径见下面那条真实服务器的用例。
    """
    async with open_tool_client(make_stub_server()) as client:
        result = await client.call("search_documents", {"query": "合同解除"})

    assert result.ok is True
    assert result.data["data"][0]["doc_id"] == "d1"
    assert result.ms >= 0


@pytest.mark.anyio
async def test_non_dict_return_is_unwrapped_from_the_result_key():
    """非 dict 注解时 mcp 会把结果包进 `{"result": ...}`，客户端必须拆开。

    只有非 dict 注解走这条路 —— 三个真实工具都是 `dict[str, Any]`，到不了这里。
    保留它是为了客户端作为通用设施的完整性：将来加一个返回 str / list 的工具，
    这条路就会启用。
    """
    async with open_tool_client(make_stub_server()) as client:
        result = await client.call("returns_a_string", {})

    assert result.ok is True
    assert result.data == "纯文本结果"


@pytest.mark.anyio
async def test_real_server_takes_the_primary_structured_content_path():
    """用**真实**的 mcp_server 与其真实工具跑一次端到端。

    上面所有用例都建在 stub 上：真实服务器的接线、三个工具包装器、以及
    `asyncio.to_thread` 那一跳，此前**完全没有测试覆盖**。这条补上，并且验证真实异常
    文本确实穿到了客户端。

    它**不**用来区分「主动 structured_content 路径」与「文本兜底路径」—— 对
    `dict[str, Any]` 注解，两条路给出的是同一份 payload（`structured_content` 就是那个
    dict 本身，文本内容里也是同一段 JSON），所以那种区分在这里做不出来。

    刻意用不存在的路径调 `parse_document`：它在碰任何模型或索引**之前**就抛
    FileNotFoundError，所以既不加载约 2GB 的模型、也不写 data/。
    """
    from app.mcp.server import mcp_server

    async with open_tool_client(mcp_server) as client:
        result = await client.call("parse_document", {"file_path": "E:/tmp/__definitely_missing__.pdf"})

    assert result.ok is False
    assert "FileNotFoundError" in result.error, "真实异常文本必须穿到客户端"
    assert "文件不存在" in result.error
    assert result.data["ok"] is False, "data 里保留的是完整信封"


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
