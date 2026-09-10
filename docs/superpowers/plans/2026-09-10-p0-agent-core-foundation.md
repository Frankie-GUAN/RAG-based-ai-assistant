# P0 Agent Core Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把「带路由的 RAG pipeline」改造成真正的 ReAct Agent，并兑现 README 里关于 MCP 协议与 token 流式的两处虚假宣称。

**Architecture:** 先定义 `AgentEvent` 事件协议（Python 端），再依次落地上下文工程引擎、真 MCP 工具层、ReAct 核心循环、真流式 SSE。工具通过 `MCPServer` 在同进程内暴露（`InMemoryTransport`），Agent 作为 MCP Client 调用；LLM 调用用 `astream` 累积 `AIMessageChunk`，同时获得 token 级流式与 tool calling。

**Tech Stack:** Python 3.11 · FastAPI · LangGraph/LangChain 1.x · `mcp==2.2.0` · pytest + anyio · Vue 3 + TypeScript

**依赖关系：** Task 1 → 2 → 3 → 4 → 5 → 6。Task 3 与 4 无相互依赖，可并行。

---

## 关键技术前提（2026-09-10 已实测验证）

写这份计划前，以下事实都在本机实际跑通验证过，**不要凭记忆改**：

| 事实 | 结论 |
|---|---|
| `mcp` 版本 | **2.2.0**。高层类叫 `MCPServer`，**不是 `FastMCP`** |
| 导入路径 | `from mcp.server import MCPServer`；`from mcp.client.session import ClientSession`；`from mcp.client._memory import InMemoryTransport` |
| 字段命名 | mcp 2.x 全 snake_case：`input_schema` / `is_error` / `structured_content`。用 camelCase 会 `AttributeError` |
| `structured_content` | 返回值被包在 `{"result": ...}` 里，需拆包 |
| **MCP 隐藏异常细节** | 工具抛异常时，客户端只拿到 `is_error=True` 与通用文案 `"Error executing tool X"`，真实异常仅在服务端日志。**因此工具必须返回 `{ok, data, error}` 而非抛异常** |
| 同进程运行 | `InMemoryTransport` 可用。**不可用 stdio 子进程**——BGE-M3 约 2GB，子进程会再加载一份 |
| 流式 + 工具调用 | `AIMessageChunk + AIMessageChunk` 能正确合并 `tool_call_chunks`。实测 `{"name":...,"args":""}` + `{"name":None,"args":"{\"query\":\"x\"}"}` → `tool_calls=[{'name':'search_documents','args':{'query':'x'},'id':'call_1'}]` |
| pytest | **项目当前未安装 pytest**，现有 4 个测试文件跑不起来。Task 1 必须补上 |
| anyio | 已安装 4.13.0（传递依赖）。用 `@pytest.mark.anyio` + `anyio_backend` fixture，无需 pytest-asyncio |

---

## File Structure

**新建：**

| 文件 | 职责 |
|---|---|
| `backend/requirements-dev.txt` | 开发依赖（pytest 等） |
| `backend/app/agent/protocol.py` | `AgentEvent` 事件协议 + SSE 编码 |
| `backend/app/context/__init__.py` | 上下文工程引擎包 |
| `backend/app/context/tokens.py` | 可替换的 token 计数器 |
| `backend/app/context/engine.py` | 分层预算 + 消息拼装 |
| `backend/app/mcp/__init__.py` | MCP 包 |
| `backend/app/mcp/server.py` | MCP 工具服务端（唯一工具定义源） |
| `backend/app/mcp/client.py` | MCP 客户端封装 + 生命周期 |
| `backend/app/agent/llm.py` | LLM 工厂（支持测试注入） |
| `backend/app/agent/core.py` | ReAct 核心循环 |
| `backend/tests/fakes.py` | 脚本化假模型 |
| `backend/tests/test_protocol.py` | 协议测试 |
| `backend/tests/test_context_engine.py` | 上下文引擎测试 |
| `backend/tests/test_mcp_tools.py` | MCP 工具层测试 |
| `backend/tests/test_agent_core.py` | ReAct 循环测试 |
| `backend/tests/test_streaming.py` | SSE 事件流测试 |

**修改：**

| 文件 | 改动 |
|---|---|
| `backend/requirements.txt` | 删死依赖、加 `mcp` |
| `backend/tests/conftest.py` | 加 `anyio_backend` fixture |
| `backend/app/tools/document_search.py` | 返回结构化 list，移除 registry 注册 |
| `backend/app/tools/web_search.py` | 同上 |
| `backend/app/tools/document_parser.py` | 同上 |
| `backend/app/config.py` | 加上下文预算配置 |
| `backend/app/api/chat.py` | 真流式，转发 AgentEvent |
| `backend/app/services/chat_service.py` | 改为 async，走 Agent Core |
| `frontend/src/composables/useChat.ts` | 适配新事件格式 |

**删除：**

| 文件 | 原因 |
|---|---|
| `backend/app/tools/registry.py` | 被 MCP 取代（这正是 README 谎称 MCP 的地方） |
| `backend/app/tools/schemas.py` | schema 改由 MCP 协议自动生成 |
| `backend/app/agent/graph.py` | 被 `core.py` 取代 |
| `backend/app/agent/nodes.py` | 同上 |

---

## Task 1: 清理死依赖 + 锁定版本

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/requirements-dev.txt`

- [ ] **Step 1: 确认死依赖确实没被引用**

```bash
cd backend
grep -rn "google.generativeai\|langchain_google_genai\|ragas" app/ tests/ || echo "确认：无任何引用"
```

Expected: 输出 `确认：无任何引用`

- [ ] **Step 2: 确认 `ToolRegistry` 只被将要删除的文件引用**

```bash
cd backend
grep -rn "tool_registry\|tools.registry\|tools.schemas\|TOOL_SCHEMAS" app/ tests/
```

Expected: 命中 `app/tools/registry.py`、`app/tools/schemas.py`、`app/tools/document_search.py`、`app/tools/web_search.py`、`app/tools/document_parser.py`、`app/tools/__init__.py`、`app/api/tools.py`。

记下 `app/api/tools.py` —— 它对外暴露工具列表接口，Task 4 需要改它。

- [ ] **Step 3: 重写 `backend/requirements.txt`**

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
sse-starlette>=2.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Agent / LLM
langchain>=1.0.0
langchain-community>=0.4.0
langchain-huggingface>=1.0.0
langchain-openai>=1.0.0
langgraph>=1.0.0

# MCP 工具协议
mcp==2.2.0

# 检索
chromadb>=1.0.0
rank-bm25>=0.2.2
sentence-transformers>=3.0.0
pypdf>=4.0.0

# 存储
sqlalchemy>=2.0.0
pymysql>=1.1.0
cryptography>=42.0.0

# 其他
python-multipart>=0.0.12
google-search-results>=2.4.2
```

注意：`langchain` 已从 `>=0.3.0` 提升到 `>=1.0.0`（实际安装的是 1.2.18，原下限与实装严重脱节）；`ragas` 与两个 gemini 包移除。

- [ ] **Step 4: 新建 `backend/requirements-dev.txt`**

```
-r requirements.txt
pytest>=8.0.0
```

- [ ] **Step 5: 安装并验证依赖可解析**

```bash
cd backend
../.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

Expected: 成功，无冲突。`mcp==2.2.0` 被安装。

- [ ] **Step 6: 验证 pytest 与 mcp 可用**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest --version
../.venv/Scripts/python.exe -c "from mcp.server import MCPServer; print('mcp ok')"
```

Expected: 打印 pytest 版本号，以及 `mcp ok`

- [ ] **Step 7: 验证应用仍能导入**

```bash
cd backend
../.venv/Scripts/python.exe -c "from app.main import app; print('app imports ok')"
```

Expected: `app imports ok`（此时尚未删除 registry，仍可导入）

- [ ] **Step 8: Commit**

```bash
git add backend/requirements.txt backend/requirements-dev.txt
git commit -m "chore: drop dead deps (gemini, ragas), add mcp, add pytest

langchain lower bound was >=0.3.0 but 1.2.18 is installed.
pytest was never installed, so the 4 existing test files could not run."
```

---

## Task 2: Agent 协议（Python 端）

**Files:**
- Create: `backend/app/agent/protocol.py`
- Create: `backend/tests/test_protocol.py`
- Modify: `backend/tests/conftest.py`

- [ ] **Step 1: 在 `backend/tests/conftest.py` 追加 anyio fixture**

在文件**末尾**追加（保留已有的 `sample_documents` fixture 不动）：

```python
@pytest.fixture
def anyio_backend():
    """让 @pytest.mark.anyio 测试跑在 asyncio 上（pytest 的 anyio 插件要求此 fixture）。"""
    return "asyncio"
```

- [ ] **Step 2: 写失败的测试 `backend/tests/test_protocol.py`**

```python
import json

import pytest

from app.agent.protocol import (
    Citation,
    CitationEvent,
    ErrorEvent,
    FinalEvent,
    HandoffEvent,
    PlanEvent,
    PlanStep,
    ReflectEvent,
    ThoughtEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
    to_sse,
)

# 与 TS 端 frontend/src/agent/protocol.ts 共享的契约。
# 任何一边增删事件类型，此集合必须同步更新。
EXPECTED_EVENT_TYPES = {
    "plan",
    "thought",
    "tool_call",
    "tool_result",
    "token",
    "citation",
    "memory_write",
    "handoff",
    "reflect",
    "final",
    "error",
}


def test_token_event_serializes_with_type_key():
    assert TokenEvent(text="你好").to_dict() == {"type": "token", "text": "你好"}


def test_handoff_wire_key_is_from_not_from_underscore():
    """协议里字段名是 from，但 from 是 Python 关键字，实现用 from_。"""
    payload = HandoffEvent(from_="edge", to="cloud", reason="complex-plan", latency_budget_ms=200).to_dict()

    assert payload["from"] == "edge"
    assert payload["to"] == "cloud"
    assert payload["latency_budget_ms"] == 200
    assert "from_" not in payload


def test_tool_call_carries_exec_location():
    payload = ToolCallEvent(id="call_1", name="search_documents", args={"query": "规则"}).to_dict()

    assert payload["type"] == "tool_call"
    assert payload["exec"] == "cloud"
    assert payload["args"] == {"query": "规则"}


def test_tool_result_records_failure_without_raising():
    payload = ToolResultEvent(id="call_1", ok=False, error="TimeoutError: boom", ms=1200).to_dict()

    assert payload["ok"] is False
    assert payload["error"] == "TimeoutError: boom"
    assert payload["ms"] == 1200


def test_final_event_nests_citations_and_usage():
    payload = FinalEvent(
        answer="转化率最低的是 8 月 3 日那场",
        citations=(Citation(doc_id="rules#0", chunk="...", score=0.91, source="rules.pdf"),),
        usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
    ).to_dict()

    assert payload["answer"].startswith("转化率")
    assert payload["citations"][0]["doc_id"] == "rules#0"
    assert payload["usage"]["total_tokens"] == 120


def test_plan_event_nests_steps():
    payload = PlanEvent(steps=(PlanStep(id="s1", goal="查场次数据", depends_on=()),)).to_dict()

    assert payload["steps"][0]["goal"] == "查场次数据"


def test_every_declared_event_type_is_covered():
    """防止新增事件类型后忘记同步 TS 端契约。"""
    declared = {
        PlanEvent.type,
        ThoughtEvent.type,
        ToolCallEvent.type,
        ToolResultEvent.type,
        TokenEvent.type,
        CitationEvent.type,
        HandoffEvent.type,
        ReflectEvent.type,
        FinalEvent.type,
        ErrorEvent.type,
    }

    assert declared <= EXPECTED_EVENT_TYPES, f"出现了未登记的事件类型: {declared - EXPECTED_EVENT_TYPES}"


def test_to_sse_keeps_chinese_readable():
    frame = to_sse(TokenEvent(text="直播"))

    assert frame["event"] == "token"
    assert "直播" in frame["data"]  # ensure_ascii=False
    assert json.loads(frame["data"]) == {"type": "token", "text": "直播"}


def test_to_sse_data_is_single_line():
    """SSE 是行协议，data 里出现裸换行会破坏前端按行解析。"""
    frame = to_sse(FinalEvent(answer="第一行\n第二行"))

    assert "\n" not in frame["data"]
```

- [ ] **Step 3: 运行测试，确认失败**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_protocol.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.agent.protocol'`

- [ ] **Step 4: 实现 `backend/app/agent/protocol.py`**

```python
"""Agent 事件协议（Python 端）。

契约与 TS 端 frontend/src/agent/protocol.ts 一致。
设计依据见 docs/superpowers/specs/2026-09-10-live-agent-platform-design.md §4.1。

所有事件继承 AgentEvent，to_dict() 序列化为 {"type": <事件名>, ...payload}。
新增事件类型时，必须同步：
  1. 本文件
  2. frontend/src/agent/protocol.ts
  3. tests/test_protocol.py 的 EXPECTED_EVENT_TYPES
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

MemoryScope = Literal["session", "long_term"]
ExecLocation = Literal["edge", "cloud"]
ReflectVerdict = Literal["pass", "retry"]


@dataclass(frozen=True)
class PlanStep:
    id: str
    goal: str
    tool_hint: str | None = None
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Citation:
    doc_id: str
    chunk: str
    score: float
    source: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class AgentEvent:
    """所有事件的基类。子类用 ClassVar 声明 type 名（ClassVar 不参与 asdict）。"""

    type: ClassVar[str] = "event"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = self.type
        return payload


@dataclass(frozen=True)
class PlanEvent(AgentEvent):
    type: ClassVar[str] = "plan"
    steps: tuple[PlanStep, ...] = ()


@dataclass(frozen=True)
class ThoughtEvent(AgentEvent):
    type: ClassVar[str] = "thought"
    text: str = ""
    step_id: str = ""


@dataclass(frozen=True)
class ToolCallEvent(AgentEvent):
    type: ClassVar[str] = "tool_call"
    id: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    exec: ExecLocation = "cloud"


@dataclass(frozen=True)
class ToolResultEvent(AgentEvent):
    type: ClassVar[str] = "tool_result"
    id: str = ""
    ok: bool = True
    data: Any = None
    ms: int = 0
    error: str | None = None


@dataclass(frozen=True)
class TokenEvent(AgentEvent):
    type: ClassVar[str] = "token"
    text: str = ""


@dataclass(frozen=True)
class CitationEvent(AgentEvent):
    type: ClassVar[str] = "citation"
    doc_id: str = ""
    chunk: str = ""
    score: float = 0.0
    source: str = ""


@dataclass(frozen=True)
class MemoryWriteEvent(AgentEvent):
    type: ClassVar[str] = "memory_write"
    scope: MemoryScope = "session"
    content: str = ""


@dataclass(frozen=True)
class HandoffEvent(AgentEvent):
    type: ClassVar[str] = "handoff"
    from_: str = "edge"          # from 是 Python 关键字，线上字段名仍是 from
    to: str = "cloud"
    reason: str = ""
    latency_budget_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["from"] = payload.pop("from_")
        return payload


@dataclass(frozen=True)
class ReflectEvent(AgentEvent):
    type: ClassVar[str] = "reflect"
    verdict: ReflectVerdict = "pass"
    critique: str = ""


@dataclass(frozen=True)
class FinalEvent(AgentEvent):
    type: ClassVar[str] = "final"
    answer: str = ""
    citations: tuple[Citation, ...] = ()
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class ErrorEvent(AgentEvent):
    type: ClassVar[str] = "error"
    code: str = ""
    message: str = ""
    recoverable: bool = False


def to_sse(event: AgentEvent) -> dict[str, str]:
    """转成 sse_starlette 接受的 dict 形式。ensure_ascii=False 保持中文可读。"""
    return {"event": event.type, "data": json.dumps(event.to_dict(), ensure_ascii=False, default=str)}
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_protocol.py -v
```

Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add backend/app/agent/protocol.py backend/tests/test_protocol.py backend/tests/conftest.py
git commit -m "feat: add AgentEvent protocol shared with the future TS edge runtime"
```

---

## Task 3: 上下文工程引擎

**Files:**
- Create: `backend/app/context/__init__.py`
- Create: `backend/app/context/tokens.py`
- Create: `backend/app/context/engine.py`
- Create: `backend/tests/test_context_engine.py`
- Modify: `backend/app/config.py`

- [ ] **Step 1: 在 `backend/app/config.py` 的 `Settings` 类中追加配置**

插入到 `reranker_model` 之后、`history_window_size` 之前：

```python
    # Context engineering (§4.7)
    context_total_tokens: int = 32_000
    context_system_ratio: float = 0.15
    context_memory_ratio: float = 0.10
    context_history_ratio: float = 0.20
    context_retrieved_ratio: float = 0.35
    context_scratchpad_ratio: float = 0.20
    context_summary_max_chars: int = 2_000
```

- [ ] **Step 2: 写失败的测试 `backend/tests/test_context_engine.py`**

```python
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.context.engine import ContextBudget, ContextEngine
from app.context.tokens import HeuristicTokenCounter


@pytest.fixture
def counter():
    return HeuristicTokenCounter()


@pytest.fixture
def engine(counter):
    # 总预算刻意调小，便于断言截断行为
    return ContextEngine(
        budget=ContextBudget(total_tokens=1_000),
        counter=counter,
    )


def test_counter_is_monotonic_and_positive():
    assert counter.count("") == 0 or counter.count("") == 1
    assert counter.count("短") < counter.count("这是一段明显更长的中文文本内容")


def test_counter_treats_cjk_as_denser_than_ascii(counter):
    """同样字符数下中文占更多 token。"""
    cjk = "中" * 60
    ascii_text = "a" * 60

    assert counter.count(cjk) > counter.count(ascii_text)


def test_build_puts_system_message_first(engine):
    result = engine.build(system="你是助手", question="你好")

    assert isinstance(result.messages[0], SystemMessage)
    assert "你是助手" in result.messages[0].content


def test_build_ends_with_the_user_question(engine):
    result = engine.build(system="sys", question="上周哪场转化率最低？")

    assert isinstance(result.messages[-1], HumanMessage)
    assert "上周哪场转化率最低？" in result.messages[-1].content


def test_history_is_mapped_to_message_roles(engine):
    result = engine.build(
        system="sys",
        question="继续",
        history=[HumanMessage(content="问题一"), AIMessage(content="回答一")],
    )

    roles = [type(m).__name__ for m in result.messages]
    assert "HumanMessage" in roles
    assert "AIMessage" in roles


def test_each_layer_respects_its_budget(engine):
    """retrieved 层给 50 条超长片段，必须被截到预算内。"""
    huge = ["片段" * 500] * 50

    result = engine.build(system="sys", question="q", retrieved=huge)

    retrieved_stat = next(s for s in result.stats if s.name == "retrieved")
    assert retrieved_stat.used <= retrieved_stat.budget
    assert retrieved_stat.truncated is True
    assert retrieved_stat.kept_count < retrieved_stat.source_count


def test_retrieved_layer_keeps_highest_ranked_first(engine):
    """传入顺序即相关度顺序（reranker 已排好），截断必须从头保留。"""
    retrieved = [f"排名第{i}的片段" for i in range(50)]

    result = engine.build(system="sys", question="q", retrieved=retrieved)

    joined = "\n".join(m.content for m in result.messages)
    assert "排名第0的片段" in joined


def test_summary_is_truncated_to_configured_char_limit(engine):
    result = engine.build(system="sys", question="q", summary="摘要内容" * 5_000)

    memory_stat = next(s for s in result.stats if s.name == "memory")
    assert memory_stat.truncated is True


def test_stats_cover_every_layer(engine):
    result = engine.build(system="sys", question="q", summary="s", retrieved=["r"], scratchpad="sp")

    assert {s.name for s in result.stats} == {
        "system", "memory", "history", "retrieved", "scratchpad"
    }


def test_empty_layers_are_omitted_from_messages(engine):
    result = engine.build(system="sys", question="q")

    contents = "\n".join(m.content for m in result.messages)
    assert "None" not in contents
```

- [ ] **Step 3: 运行测试，确认失败**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_context_engine.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.context'`

- [ ] **Step 4: 实现 `backend/app/context/tokens.py`**

```python
"""可替换的 token 计数。

P0 用启发式估算（不引入 tokenizer 依赖）；后续如需精确计数，
实现 TokenCounter 协议换成 tiktoken / DeepSeek tokenizer 即可。
"""
from __future__ import annotations

import math
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF      # 基本汉字
        or 0x3400 <= code <= 0x4DBF   # 扩展 A
        or 0x3000 <= code <= 0x303F   # CJK 标点
        or 0xFF00 <= code <= 0xFFEF   # 全角
    )


class HeuristicTokenCounter:
    """CJK 约 1 token / 1.5 字符，其余约 1 token / 4 字符。

    实测 DeepSeek 对中文的切分接近该比例，用于预算控制足够准确。
    """

    def count(self, text: str) -> int:
        if not text:
            return 0
        cjk = sum(1 for ch in text if _is_cjk(ch))
        other = len(text) - cjk
        return max(1, math.ceil(cjk / 1.5 + other / 4))
```

- [ ] **Step 5: 实现 `backend/app/context/engine.py`**

```python
"""上下文工程引擎：分层 token 预算 + 消息拼装。

设计见 docs/superpowers/specs/2026-09-10-live-agent-platform-design.md §4.7。
每层独立预算，层内超限先截断，**不跨层抢占**。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.context.tokens import TokenCounter

# 截断时在层内容尾部补的提示，避免模型误以为内容完整
_TRUNCATION_NOTE = "\n…（内容因长度限制被截断）"


@dataclass(frozen=True)
class ContextBudget:
    total_tokens: int = 32_000
    system_ratio: float = 0.15
    memory_ratio: float = 0.10
    history_ratio: float = 0.20
    retrieved_ratio: float = 0.35
    scratchpad_ratio: float = 0.20

    def layer_budget(self, ratio: float) -> int:
        return int(self.total_tokens * ratio)

    @classmethod
    def from_settings(cls, settings) -> "ContextBudget":
        return cls(
            total_tokens=settings.context_total_tokens,
            system_ratio=settings.context_system_ratio,
            memory_ratio=settings.context_memory_ratio,
            history_ratio=settings.context_history_ratio,
            retrieved_ratio=settings.context_retrieved_ratio,
            scratchpad_ratio=settings.context_scratchpad_ratio,
        )


@dataclass(frozen=True)
class LayerStat:
    name: str
    budget: int
    used: int
    source_count: int
    kept_count: int
    truncated: bool


@dataclass(frozen=True)
class ContextResult:
    messages: list[BaseMessage]
    stats: list[LayerStat]


class ContextEngine:
    def __init__(self, budget: ContextBudget, counter: TokenCounter) -> None:
        self._budget = budget
        self._counter = counter

    def build(
        self,
        *,
        system: str,
        question: str,
        summary: str = "",
        history: Sequence[BaseMessage] = (),
        retrieved: Sequence[str] = (),
        scratchpad: str = "",
    ) -> ContextResult:
        stats: list[LayerStat] = []

        system_text, stat = self._fit_text("system", [system], self._budget.system_ratio)
        stats.append(stat)

        memory_text, stat = self._fit_text("memory", [summary] if summary else [], self._budget.memory_ratio)
        stats.append(stat)

        kept_history = self._fit_messages("history", list(history), self._budget.history_ratio)
        stats.append(
            LayerStat(
                name="history",
                budget=self._budget.layer_budget(self._budget.history_ratio),
                used=sum(self._counter.count(str(m.content)) for m in kept_history),
                source_count=len(history),
                kept_count=len(kept_history),
                truncated=len(kept_history) < len(history),
            )
        )

        retrieved_text, stat = self._fit_text("retrieved", list(retrieved), self._budget.retrieved_ratio)
        stats.append(stat)

        scratchpad_text, stat = self._fit_text("scratchpad", [scratchpad] if scratchpad else [], self._budget.scratchpad_ratio)
        stats.append(stat)

        messages: list[BaseMessage] = [SystemMessage(content=self._compose_system(system_text, memory_text))]

        # 历史按时间正序保留最近的部分
        messages.extend(kept_history)

        messages.append(HumanMessage(content=self._compose_user(question, retrieved_text, scratchpad_text)))

        return ContextResult(messages=messages, stats=stats)

    # --- internals ---

    def _compose_system(self, system_text: str, memory_text: str) -> str:
        parts = [system_text]
        if memory_text:
            parts.append(f"\n\n【长期记忆】\n{memory_text}")
        return "".join(parts)

    def _compose_user(self, question: str, retrieved_text: str, scratchpad_text: str) -> str:
        parts: list[str] = []
        if retrieved_text:
            parts.append(f"【参考资料】\n{retrieved_text}\n\n")
        if scratchpad_text:
            parts.append(f"【已获取的中间结果】\n{scratchpad_text}\n\n")
        parts.append(f"【用户问题】\n{question}")
        return "".join(parts)

    def _fit_text(self, name: str, items: Sequence[str], ratio: float) -> tuple[str, LayerStat]:
        """按顺序累积条目直到预算耗尽。条目本身超预算时硬截断。"""
        budget = self._budget.layer_budget(ratio)
        kept: list[str] = []
        used = 0
        truncated = False

        for item in items:
            if not item:
                continue
            remaining = budget - used
            if remaining <= 0:
                truncated = True
                break

            cost = self._counter.count(item)
            if cost <= remaining:
                kept.append(item)
                used += cost
                continue

            # 单条超预算：按比例硬截断
            allowed_chars = max(1, int(len(item) * (remaining / cost)))
            clipped = item[:allowed_chars] + _TRUNCATION_NOTE
            kept.append(clipped)
            used += self._counter.count(clipped)
            truncated = True
            break

        return "\n\n".join(kept), LayerStat(
            name=name,
            budget=budget,
            used=used,
            source_count=len([i for i in items if i]),
            kept_count=len(kept),
            truncated=truncated or len(kept) < len([i for i in items if i]),
        )

    def _fit_messages(self, name: str, messages: list[BaseMessage], ratio: float) -> list[BaseMessage]:
        """从最近的消息往前保留，直到预算耗尽（保留时间连续的一段）。"""
        budget = self._budget.layer_budget(ratio)
        kept_reversed: list[BaseMessage] = []
        used = 0

        for message in reversed(messages):
            cost = self._counter.count(str(message.content))
            if used + cost > budget:
                break
            kept_reversed.append(message)
            used += cost

        return list(reversed(kept_reversed))
```

- [ ] **Step 6: 实现 `backend/app/context/__init__.py`**

```python
from app.context.engine import ContextBudget, ContextEngine, ContextResult, LayerStat
from app.context.tokens import HeuristicTokenCounter, TokenCounter

__all__ = [
    "ContextBudget",
    "ContextEngine",
    "ContextResult",
    "LayerStat",
    "HeuristicTokenCounter",
    "TokenCounter",
]
```

- [ ] **Step 7: 运行测试，确认通过**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_context_engine.py -v
```

Expected: 10 passed

- [ ] **Step 8: Commit**

```bash
git add backend/app/context/ backend/app/config.py backend/tests/test_context_engine.py
git commit -m "feat: add layered context engineering engine with per-layer token budgets"
```

---

## Task 4: 真 MCP 工具层

**Files:**
- Create: `backend/app/mcp/__init__.py`
- Create: `backend/app/mcp/server.py`
- Create: `backend/app/mcp/client.py`
- Create: `backend/tests/test_mcp_tools.py`
- Modify: `backend/app/tools/document_search.py`
- Modify: `backend/app/tools/web_search.py`
- Modify: `backend/app/tools/document_parser.py`
- Modify: `backend/app/tools/__init__.py`
- Modify: `backend/app/api/tools.py`
- Delete: `backend/app/tools/registry.py`, `backend/app/tools/schemas.py`

- [ ] **Step 1: 改写 `backend/app/tools/document_search.py`**

返回结构化数据（不再拼字符串、不再注册 registry），供 Agent 与引用溯源使用：

```python
from app.rag.bm25_index import BM25Index
from app.rag.hybrid_search import hybrid_search


def search_documents(query: str, top_k: int = 4) -> list[dict]:
    """混合检索私有文档库，返回结构化片段列表（相关度降序）。"""
    bm25 = BM25Index()
    if bm25.document_count == 0:
        bm25.load()

    results = hybrid_search(query, bm25, final_top_k=top_k)

    return [
        {
            "doc_id": f"{doc.metadata.get('source', 'unknown')}#{i}",
            "content": doc.page_content,
            "source": doc.metadata.get("source", "未知"),
            "score": round(float(score), 4),
        }
        for i, (doc, score) in enumerate(results)
    ]
```

- [ ] **Step 2: 改写 `backend/app/tools/web_search.py`**

```python
import os

from langchain_community.utilities import SerpAPIWrapper

from app.config import settings


def search_web(query: str, num: int = 5) -> list[dict]:
    """联网搜索，返回结构化结果列表。未配置 key 时抛 RuntimeError（由 MCP 层转成 ok=False）。"""
    if not settings.serpapi_key:
        raise RuntimeError("联网搜索未配置 SERPAPI_KEY")

    os.environ["SERPAPI_API_KEY"] = settings.serpapi_key
    wrapper = SerpAPIWrapper(params={"num": num, "engine": "google", "hl": "zh-cn"})
    results = wrapper.results(query)

    return [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item["link"],
        }
        for item in results.get("organic_results", [])
        if item.get("link")
    ]
```

- [ ] **Step 3: 改写 `backend/app/tools/document_parser.py`**

```python
from pathlib import Path

from app.rag.bm25_index import BM25Index
from app.rag.loader import load_file, split_documents
from app.rag.vector_store import add_documents


def parse_document(file_path: str) -> dict:
    """解析上传文档并建立索引。文件不存在时抛 FileNotFoundError。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    chunks = split_documents(load_file(path))
    add_documents(chunks)

    bm25 = BM25Index()
    bm25.load()
    all_docs = list(bm25._documents) if bm25._documents else []
    all_docs.extend(chunks)
    bm25.build(all_docs)
    bm25.save()

    return {"file": path.name, "chunks": len(chunks)}
```

- [ ] **Step 4: 改写 `backend/app/tools/__init__.py`**

```python
from app.tools import document_parser  # noqa: F401
from app.tools import document_search  # noqa: F401
from app.tools import web_search  # noqa: F401
```

- [ ] **Step 5: 删除被取代的文件**

```bash
cd backend
git rm app/tools/registry.py app/tools/schemas.py
```

- [ ] **Step 6: 写失败的测试 `backend/tests/test_mcp_tools.py`**

```python
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
        result = await client.call("search_documents", {"query": "直播规则"})

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
```

- [ ] **Step 7: 运行测试，确认失败**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_mcp_tools.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.mcp'`

- [ ] **Step 8: 实现 `backend/app/mcp/server.py`**

```python
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
    description="直播主播经营助手的工具集：私有知识库检索、联网搜索、文档解析。",
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
```

- [ ] **Step 9: 实现 `backend/app/mcp/client.py`**

```python
"""MCP 客户端封装。

与 MCP Server **同进程**运行（InMemoryTransport）。
不可改用 stdio 子进程：BGE-M3 约 2GB，子进程会再加载一份模型。
"""
from __future__ import annotations

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
        except Exception as exc:  # noqa: BLE001 — 传输层错误（工具名不存在等）
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}", ms=_elapsed_ms(started))

        ms = _elapsed_ms(started)

        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            return ToolResult(ok=False, error=text or "工具调用失败", ms=ms)

        payload = result.structured_content
        # mcp 把结构化返回值包在 {"result": ...} 里
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]

        return ToolResult(ok=True, data=payload, ms=ms)


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
```

- [ ] **Step 10: 实现 `backend/app/mcp/__init__.py`**

```python
from app.mcp.client import MCPToolClient, ToolResult, ToolSpec, open_tool_client
from app.mcp.server import mcp_server

__all__ = ["MCPToolClient", "ToolResult", "ToolSpec", "open_tool_client", "mcp_server"]
```

- [ ] **Step 11: 运行测试，确认通过**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_mcp_tools.py -v
```

Expected: 7 passed

若 `from mcp.client._memory import InMemoryTransport` 报错（该模块在下划线前缀的私有路径下，未来版本可能变动），
替代方案是自行基于 `mcp.shared.memory.create_client_server_memory_streams()` 构造，见该函数源码。

- [ ] **Step 12: 改写 `backend/app/api/tools.py` 以走 MCP**

先查看现有内容：

```bash
cd backend
cat app/api/tools.py
```

将其中的数据来源从 `tool_registry.list_tools()` 改为 MCP。若该文件只做「列出可用工具」，改成：

```python
from fastapi import APIRouter

from app.mcp.server import mcp_server

router = APIRouter(prefix="/api/tools", tags=["tools"])


@router.get("")
async def list_tools():
    """列出 Agent 可用的 MCP 工具。"""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in await mcp_server.list_tools()
    ]
```

注意：`mcp_server.list_tools()` 的确切返回结构需在实现时用
`../.venv/Scripts/python.exe -c "from mcp.server import MCPServer; print([m for m in dir(MCPServer) if 'tool' in m.lower()])"`
确认；若签名不同，以实际为准并同步更新本步骤。

- [ ] **Step 13: 验证应用可导入且工具能列出**

```bash
cd backend
../.venv/Scripts/python.exe -c "
import anyio
from app.mcp.server import mcp_server
from app.mcp.client import open_tool_client

async def main():
    async with open_tool_client(mcp_server) as c:
        for s in c.specs:
            print(s.name, '->', s.description[:30])

anyio.run(main)
"
```

Expected: 打印 `search_documents` / `search_web` / `parse_document` 三行

- [ ] **Step 14: Commit**

```bash
git add -A backend/app/tools backend/app/mcp backend/app/api/tools.py backend/tests/test_mcp_tools.py
git commit -m "feat: replace fake MCP with a real MCP server over in-process transport

README claimed MCP but there was no mcp dependency at all — tools were a
plain dict registry. Now tools are served by mcp.server.MCPServer and the
agent is a real MCP client.

Tools return {ok, data, error} instead of raising: MCP hides exception
details from clients, so a raised error reaches the agent as an
undecidable 'Error executing tool X'."
```

---

## Task 5: Agent Core（ReAct 循环）

**Files:**
- Create: `backend/app/agent/llm.py`
- Create: `backend/app/agent/core.py`
- Create: `backend/tests/fakes.py`
- Create: `backend/tests/test_agent_core.py`
- Modify: `backend/app/services/chat_service.py`
- Delete: `backend/app/agent/graph.py`, `backend/app/agent/nodes.py`

- [ ] **Step 1: 实现 `backend/app/agent/llm.py`**

```python
"""LLM 工厂。

生产环境返回 DeepSeek 的 ChatOpenAI；测试通过 set_chat_model_factory
注入脚本化模型，使 Agent 循环测试不需要 API key 与网络。
"""
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import settings

_override: Callable[[], BaseChatModel] | None = None


def set_chat_model_factory(factory: Callable[[], BaseChatModel] | None) -> None:
    """测试专用。传 None 恢复默认 DeepSeek。"""
    global _override
    _override = factory


def get_chat_model() -> BaseChatModel:
    if _override is not None:
        return _override()

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.2,
    )
```

- [ ] **Step 2: 实现 `backend/tests/fakes.py`**

```python
"""测试替身：按脚本产出 AIMessageChunk 的假模型。

不用 LangChain 自带的 FakeListChatModel —— 它只支持纯文本，
无法模拟流式 tool_call_chunks 的增量合并。
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessageChunk, BaseMessage


class ScriptedChatModel:
    """script 是「每一轮 LLM 调用产出哪些 chunk」的列表。

    例如两轮：先调工具，再出答案 →
        ScriptedChatModel([
            tool_turn("search_documents", {"query": "x"}),
            text_turn("答案", "是……"),
        ])
    """

    def __init__(self, script: list[list[AIMessageChunk]]) -> None:
        self._script = [list(turn) for turn in script]
        self.calls: list[list[BaseMessage]] = []
        self.bound_tools: list[dict[str, Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]]) -> "ScriptedChatModel":
        self.bound_tools = list(tools)
        return self

    async def astream(self, messages: list[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        self.calls.append(list(messages))

        if not self._script:
            raise AssertionError(
                f"脚本已耗尽：模型被调用了第 {len(self.calls)} 次，但没有预设响应"
            )

        for chunk in self._script.pop(0):
            yield chunk


def text_turn(*pieces: str) -> list[AIMessageChunk]:
    """一次纯文本回答，拆成多个 chunk 以模拟真实流式。"""
    return [AIMessageChunk(content=piece) for piece in pieces]


def tool_turn(name: str, args: dict[str, Any], call_id: str = "call_1") -> list[AIMessageChunk]:
    """一次工具调用。刻意拆成两个 chunk，复现真实的流式参数增量合并。"""
    return [
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": name, "args": "", "id": call_id, "index": 0}
        ]),
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": None, "args": json.dumps(args), "id": None, "index": 0}
        ]),
    ]


def two_tool_turn(
    first: tuple[str, dict[str, Any], str],
    second: tuple[str, dict[str, Any], str],
) -> list[AIMessageChunk]:
    """一轮内并发发起两个工具调用。"""
    return [
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": first[0], "args": json.dumps(first[1]), "id": first[2], "index": 0},
            {"name": second[0], "args": json.dumps(second[1]), "id": second[2], "index": 1},
        ]),
    ]
```

- [ ] **Step 3: 写失败的测试 `backend/tests/test_agent_core.py`**

```python
import pytest
from mcp.server import MCPServer

from app.agent.core import AgentDeps, AgentTask, run_agent_stream
from app.context.engine import ContextBudget, ContextEngine
from app.context.tokens import HeuristicTokenCounter
from app.mcp.client import open_tool_client
from tests.fakes import ScriptedChatModel, text_turn, tool_turn, two_tool_turn

SLOW_TOOL_MARKER = "slow"


def make_stub_server() -> MCPServer:
    server = MCPServer(name="stub", version="0.1.0")

    @server.tool()
    async def search_documents(query: str, top_k: int = 4) -> dict:
        """检索私有文档库。"""
        return {"ok": True, "data": [{"doc_id": "d1", "content": f"{query} 的结果"}], "error": None}

    @server.tool()
    async def broken_tool(query: str = "") -> dict:
        """总是失败，用于验证工具失败不中断循环。"""
        return {"ok": False, "data": None, "error": "RuntimeError: 服务不可用"}

    return server


def make_engine() -> ContextEngine:
    return ContextEngine(
        budget=ContextBudget(total_tokens=4_000),
        counter=HeuristicTokenCounter(),
    )


def events_of(events):
    return [e.type for e in events]


@pytest.mark.anyio
async def test_agent_answers_directly_without_tools():
    model = ScriptedChatModel([text_turn("你好", "！")])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="你好"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    assert events_of(events) == ["token", "token", "final"]
    assert events[-1].answer == "你好！"


@pytest.mark.anyio
async def test_agent_executes_tool_then_streams_the_answer():
    model = ScriptedChatModel([
        tool_turn("search_documents", {"query": "直播规则"}),
        text_turn("根据", "资料", "答案是……"),
    ])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="直播有哪些红线？"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    assert events_of(events) == [
        "tool_call", "tool_result", "token", "token", "token", "final",
    ]
    call = events[0]
    assert call.name == "search_documents"
    assert call.args == {"query": "直播规则"}
    assert events[1].ok is True


@pytest.mark.anyio
async def test_tool_result_is_fed_back_to_the_model():
    model = ScriptedChatModel([
        tool_turn("search_documents", {"query": "直播规则"}),
        text_turn("完成"),
    ])

    async with open_tool_client(make_stub_server()) as tools:
        _ = [
            e async for e in run_agent_stream(
                AgentTask(question="q"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    second_call_messages = model.calls[1]
    tool_messages = [m for m in second_call_messages if type(m).__name__ == "ToolMessage"]

    assert len(tool_messages) == 1
    assert "直播规则 的结果" in tool_messages[0].content
    assert tool_messages[0].tool_call_id == "call_1"


@pytest.mark.anyio
async def test_failing_tool_does_not_abort_the_loop():
    """工具失败要以 ok=false 回灌给模型，让模型自行决定重试或降级。"""
    model = ScriptedChatModel([
        tool_turn("broken_tool", {"query": "x"}),
        text_turn("工具不可用，我根据已有知识回答"),
    ])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="q"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    assert events_of(events)[-1] == "final"
    assert events[1].ok is False
    assert "服务不可用" in events[1].error

    tool_messages = [m for m in model.calls[1] if type(m).__name__ == "ToolMessage"]
    assert '"ok": false' in tool_messages[0].content


@pytest.mark.anyio
async def test_independent_tool_calls_run_in_parallel():
    """同一轮内的多个工具调用必须并发，且全部 tool_call 先于任何 tool_result。"""
    model = ScriptedChatModel([
        two_tool_turn(
            ("search_documents", {"query": "a"}, "call_a"),
            ("search_documents", {"query": "b"}, "call_b"),
        ),
        text_turn("好了"),
    ])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="q"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    types = events_of(events)
    assert types[:2] == ["tool_call", "tool_call"]
    assert types[2:4] == ["tool_result", "tool_result"]
    assert {events[0].id, events[1].id} == {"call_a", "call_b"}


@pytest.mark.anyio
async def test_loop_stops_at_max_iterations():
    """模型一直调工具不收敛时，必须止步而不是无限循环。"""
    model = ScriptedChatModel([
        tool_turn("search_documents", {"query": f"q{i}"}, f"call_{i}")
        for i in range(10)
    ])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="q"),
                AgentDeps(model=model, tools=tools, context=make_engine(), max_iterations=3),
            )
        ]

    assert events[-1].type == "error"
    assert events[-1].code == "max_iterations"
    assert len(model.calls) == 3


@pytest.mark.anyio
async def test_tool_schemas_are_bound_to_the_model():
    model = ScriptedChatModel([text_turn("hi")])

    async with open_tool_client(make_stub_server()) as tools:
        _ = [
            e async for e in run_agent_stream(
                AgentTask(question="q"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    names = {t["function"]["name"] for t in model.bound_tools}
    assert names == {"search_documents", "broken_tool"}
```

- [ ] **Step 4: 运行测试，确认失败**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_agent_core.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.agent.core'`

- [ ] **Step 5: 实现 `backend/app/agent/core.py`**

```python
"""Agent 核心循环：ReAct + 并行工具调用 + 迭代上限。

用 model.astream() 累积 AIMessageChunk，同时获得：
  · token 级真流式（文本增量立即外发）
  · tool calling（tool_call_chunks 在流结束后已合并为完整 tool_calls）

已知取舍：若模型先输出一段文字再发起工具调用，那段文字已经作为
token 事件外发了。通过 system prompt 约束「调用工具时不要输出解释文字」来规避。
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.agent.protocol import (
    AgentEvent,
    ErrorEvent,
    FinalEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.context.engine import ContextEngine
from app.mcp.client import MCPToolClient

SYSTEM_PROMPT = (
    "你是直播主播经营助手，服务主播及其运营人员。"
    "你可以调用工具检索平台规则与商品知识库、查询直播数据、联网搜索最新信息。\n"
    "规则：\n"
    "1. 需要事实依据时先调用工具，不要凭记忆回答平台规则或数据。\n"
    "2. 调用工具时不要输出解释性文字。\n"
    "3. 资料不足时明确说明不确定，不要编造。\n"
    "4. 引用检索结果时说明来源。"
)


@dataclass
class AgentTask:
    question: str
    history: list[BaseMessage] = field(default_factory=list)
    summary: str = ""


@dataclass
class AgentDeps:
    model: BaseChatModel
    tools: MCPToolClient
    context: ContextEngine
    max_iterations: int = 8


async def run_agent_stream(task: AgentTask, deps: AgentDeps) -> AsyncIterator[AgentEvent]:
    """执行 Agent 循环并产出事件流。

    保证：最后一个事件必定是 FinalEvent 或 ErrorEvent。
    """
    context = deps.context.build(
        system=SYSTEM_PROMPT,
        question=task.question,
        summary=task.summary,
        history=task.history,
    )
    messages: list[BaseMessage] = list(context.messages)

    schemas = [spec.to_openai_schema() for spec in deps.tools.specs]
    bound = deps.model.bind_tools(schemas) if schemas else deps.model

    for _ in range(deps.max_iterations):
        accumulated: AIMessage | None = None

        async for chunk in bound.astream(messages):
            accumulated = chunk if accumulated is None else accumulated + chunk
            text = _as_text(chunk.content)
            if text:
                yield TokenEvent(text=text)

        if accumulated is None:
            yield ErrorEvent(code="empty_response", message="模型未返回任何内容", recoverable=False)
            return

        messages.append(accumulated)

        if not accumulated.tool_calls:
            yield FinalEvent(answer=_as_text(accumulated.content))
            return

        # 先外发全部 tool_call，让前端立刻看到并发意图，再并行执行
        for call in accumulated.tool_calls:
            yield ToolCallEvent(id=call["id"], name=call["name"], args=call["args"])

        results = await asyncio.gather(
            *(deps.tools.call(call["name"], call["args"]) for call in accumulated.tool_calls)
        )

        for call, result in zip(accumulated.tool_calls, results):
            yield ToolResultEvent(
                id=call["id"], ok=result.ok, data=result.data, ms=result.ms, error=result.error
            )
            messages.append(ToolMessage(
                content=json.dumps(
                    {"ok": result.ok, "data": result.data, "error": result.error},
                    ensure_ascii=False,
                    default=str,
                ),
                tool_call_id=call["id"],
            ))

    yield ErrorEvent(
        code="max_iterations",
        message=f"达到迭代上限 {deps.max_iterations}，未能得出结论",
        recoverable=True,
    )


def _as_text(content: Any) -> str:
    """兼容 str 与 LangChain 的多模态 content block 列表。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)
```

- [ ] **Step 6: 运行测试，确认通过**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_agent_core.py -v
```

Expected: 7 passed

- [ ] **Step 7: 改写 `backend/app/services/chat_service.py`**

```python
"""对话服务：把会话历史交给 Agent Core，产出事件流。"""
from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent.core import AgentDeps, AgentTask, run_agent_stream
from app.agent.llm import get_chat_model
from app.agent.protocol import AgentEvent
from app.context.engine import ContextBudget, ContextEngine
from app.context.tokens import HeuristicTokenCounter
from app.config import settings
from app.mcp.client import MCPToolClient


def build_context_engine() -> ContextEngine:
    return ContextEngine(
        budget=ContextBudget.from_settings(settings),
        counter=HeuristicTokenCounter(),
    )


def to_langchain_messages(history: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    return messages


async def stream_agent(
    *,
    question: str,
    history: list[dict],
    summary: str | None,
    tools: MCPToolClient,
    max_iterations: int = 8,
) -> AsyncIterator[AgentEvent]:
    task = AgentTask(
        question=question,
        history=to_langchain_messages(history),
        summary=summary or "",
    )
    deps = AgentDeps(
        model=get_chat_model(),
        tools=tools,
        context=build_context_engine(),
        max_iterations=max_iterations,
    )

    async for event in run_agent_stream(task, deps):
        yield event
```

- [ ] **Step 8: 删除被取代的编排文件**

```bash
cd backend
git rm app/agent/graph.py app/agent/nodes.py
```

同时删除 `backend/tests/test_agent.py`（它直接调用 `agent_graph.invoke`，且依赖真实 DeepSeek API key 才能跑，已被 `test_agent_core.py` 全面取代）：

```bash
cd backend
git rm tests/test_agent.py
```

- [ ] **Step 9: 验证应用可导入**

```bash
cd backend
../.venv/Scripts/python.exe -c "from app.main import app; print('app imports ok')"
```

Expected: `app imports ok`

注意：`app/api/chat.py` 目前仍 import `run_agent`（已删除），此步会失败。
这是预期的 —— Task 6 会重写它。若想保持每步可导入，可在此步临时把
`chat.py` 的 import 改为 `from app.services.chat_service import stream_agent`。

- [ ] **Step 10: Commit**

```bash
git add -A backend/app backend/tests
git commit -m "feat: replace the 3-node DAG with a real ReAct loop

graph.py was router -> retrieve -> generate with a fixed topology; the
AgentState.iteration field existed but was never read. Now the model
decides which tools to call, results are fed back, and independent calls
in one turn run concurrently via asyncio.gather.

Tests use a scripted fake model, so the loop is verifiable without an
API key or network."
```

---

## Task 6: 真流式 SSE

**Files:**
- Modify: `backend/app/api/chat.py`
- Modify: `frontend/src/composables/useChat.ts`
- Create: `backend/tests/test_streaming.py`

- [ ] **Step 1: 写失败的测试 `backend/tests/test_streaming.py`**

```python
import json

import pytest

from app.agent.protocol import (
    ErrorEvent,
    FinalEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    to_sse,
)


def test_token_event_maps_to_sse_frame():
    frame = to_sse(TokenEvent(text="直播"))

    assert frame["event"] == "token"
    assert json.loads(frame["data"])["type"] == "token"


def test_frames_can_be_reassembled_into_the_full_answer():
    """前端按顺序拼接 token 事件必须还原出完整答案。"""
    pieces = ["根据", "平台", "规则", "……"]
    reconstructed = "".join(
        json.loads(to_sse(TokenEvent(text=p))["data"])["text"] for p in pieces
    )

    assert reconstructed == "根据平台规则……"


def test_tool_events_carry_enough_for_a_trace_view():
    call = json.loads(to_sse(ToolCallEvent(id="c1", name="search_documents", args={"query": "x"}))["data"])
    result = json.loads(to_sse(ToolResultEvent(id="c1", ok=True, data={"n": 1}, ms=42))["data"])

    assert call["name"] == "search_documents"
    assert call["args"] == {"query": "x"}
    assert result["ms"] == 42
    assert result["id"] == call["id"]


def test_error_event_is_terminal_and_machine_readable():
    payload = json.loads(to_sse(ErrorEvent(code="max_iterations", message="超限", recoverable=True))["data"])

    assert payload["type"] == "error"
    assert payload["code"] == "max_iterations"
    assert payload["recoverable"] is True


def test_final_event_carries_the_answer_without_chunking():
    """最终答案必须是完整字符串，前端不必再拼接 done 事件。"""
    payload = json.loads(to_sse(FinalEvent(answer="完整答案"))["data"])

    assert payload["answer"] == "完整答案"
```

- [ ] **Step 2: 运行测试，确认通过（协议层已就绪）**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_streaming.py -v
```

Expected: 5 passed（这些测试锁定的是 SSE 编码契约，实现已在 Task 2 完成）

- [ ] **Step 3: 重写 `backend/app/api/chat.py`**

```python
"""对话 API：把 Agent 的事件流转成 SSE。

事件格式统一为 data: {"type": "...", ...}。
另有两个非协议事件：status（处理中）与 done（携带 conversation_id）。
"""
import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.agent.protocol import FinalEvent, to_sse
from app.db.session import SessionLocal
from app.rag.vector_store import has_persisted_index
from app.services.chat_service import stream_agent
from app.services.conversation_service import (
    create_conversation,
    get_conversation_with_context,
    save_message,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    question: str
    history: list[dict] = []
    has_docs: bool = False
    max_iterations: int = 8


def _sse(event_type: str, payload: dict) -> dict[str, str]:
    return {"event": event_type, "data": json.dumps(payload, ensure_ascii=False, default=str)}


async def _stream_chat(request: ChatRequest, app_request: Request) -> AsyncGenerator[dict, None]:
    db = SessionLocal()
    loop = asyncio.get_event_loop()

    try:
        if request.conversation_id:
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

        tools = app_request.app.state.mcp_client
        answer = ""

        async for event in stream_agent(
            question=request.question,
            history=history_for_agent,
            summary=summary,
            tools=tools,
            max_iterations=request.max_iterations,
        ):
            if isinstance(event, FinalEvent):
                answer = event.answer
            yield to_sse(event)

        save_message(db, conversation_id, "assistant", answer, source_type="agent")
        db.commit()

        yield _sse("done", {"conversation_id": conversation_id, "answer": answer})

    except Exception as exc:  # noqa: BLE001 — 必须回滚并回报结构化错误
        db.rollback()
        yield _sse("error", {"type": "error", "code": "server_error",
                             "message": f"{type(exc).__name__}: {exc}", "recoverable": False})
    finally:
        db.close()


@router.post("/stream")
async def chat_stream(request: ChatRequest, app_request: Request):
    return EventSourceResponse(_stream_chat(request, app_request))
```

- [ ] **Step 4: 在 `backend/app/main.py` 的 lifespan 中挂载 MCP 工具客户端**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.base import Base
from app.db.session import engine
from app.mcp.client import open_tool_client
from app.mcp.server import mcp_server
from app.api.chat import router as chat_router
from app.api.knowledge import router as knowledge_router
from app.api.tools import router as tools_router
from app.api.evaluation import router as eval_router
from app.api.conversations import router as conversations_router

import app.models.conversation  # noqa: F401
import app.models.message       # noqa: F401
import app.models.document      # noqa: F401
import app.models.evaluation    # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    # MCP 工具客户端与 Server 同进程，整个应用生命周期内复用同一个会话
    async with open_tool_client(mcp_server) as tool_client:
        app.state.mcp_client = tool_client
        yield


app = FastAPI(title="Live Agent Platform API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(tools_router)
app.include_router(eval_router)
app.include_router(conversations_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 5: 验证应用可导入**

```bash
cd backend
../.venv/Scripts/python.exe -c "from app.main import app; print('app imports ok')"
```

Expected: `app imports ok`

- [ ] **Step 6: 前端适配 —— 改写 `frontend/src/composables/useChat.ts` 的事件分发**

只替换 `sendMessage` 中的解析循环部分（`const reader = ...` 到 `fetchConversations()` 之前）：

```ts
    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let finalAnswer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = JSON.parse(line.slice(6))

        switch (data.type) {
          case 'token':
            streamingContent.value += data.text
            break
          case 'tool_call':
            currentRoute.value = `调用工具：${data.name}`
            break
          case 'tool_result':
            currentRoute.value = data.ok
              ? `工具返回（${data.ms}ms）`
              : `工具失败：${data.error}`
            break
          case 'final':
            finalAnswer = data.answer
            break
          case 'error':
            currentRoute.value = `出错：${data.message}`
            break
          default:
            // status / done 等非协议事件
            if (data.status === 'thinking') currentRoute.value = '思考中...'
            if (data.conversation_id && !store.conversationId) {
              store.setConversationId(data.conversation_id)
            }
        }
      }
    }

    if (finalAnswer) {
      store.addMessage({ role: 'assistant', content: finalAnswer })
    }
```

- [ ] **Step 7: 端到端验证真流式（不可跳过）**

启动后端：

```bash
cd backend
../.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

另开终端，观察 token 是否**逐条到达**而非一次性到达：

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"你好，简单介绍一下你自己"}'
```

Expected: 输出中 `event: token` 帧随时间陆续出现（而非所有帧在最后一瞬间一起刷出）。
这是本次改造要兑现的核心承诺 —— 改造前 `chat.py` 是等 LLM 全量返回后按 4 字符切片，
所有帧会几乎同时到达。

- [ ] **Step 8: Commit**

```bash
git add -A backend/app/api/chat.py backend/app/main.py frontend/src/composables/useChat.ts backend/tests/test_streaming.py
git commit -m "feat: real token streaming via LLM astream, replacing fake 4-char chunking

chat.py generated the whole answer, then sliced it into 4-character
pieces with a 20ms sleep to simulate streaming. Tokens now come from
model.astream() directly."
```

---

## 完成标准

全部 6 个任务完成后，以下命令必须全绿：

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/ -v
```

以及手工验收：

- [ ] `curl -N` 能看到 token 逐条到达（Task 6 Step 7）
- [ ] MCP 工具被真实列出：`open_tool_client(mcp_server)` 返回 3 个工具
- [ ] 代码中已无 `tool_registry` / `agent_graph` / `sleep(0.02)` 残留：

```bash
cd backend
grep -rn "tool_registry\|agent_graph\|sleep(0.02)\|TOOL_SCHEMAS" app/ tests/ || echo "干净"
```
Expected: `干净`

- [ ] README 的「MCP 协议」「SSE 流式输出」两条宣称此时**已为真**，可保留；`项目亮点` 一节待 P1 完成后重写

---

## Self-Review 记录

**Spec 覆盖检查：** 本计划覆盖 spec §6 P0 的全部 6 项任务（依赖清理、Agent 协议、上下文引擎、MCP 工具层、Agent Core、真流式）。
spec §4.9 的评测体系属 P1/P2，不在本计划范围。

**与 spec 的两处偏离（已回写 spec §6）：**
1. **Agent 协议从 P2 提前到 P0 任务 2**。原计划排在 P2 任务 11，但 P0 的 MCP、Agent Core、流式三者都要发事件，协议必须先有。P2 任务 11 现改为「实现协议的 TS 端」。
2. **上下文引擎从任务 6 提前到任务 3**。原顺序下 Agent Core 要调用尚不存在的上下文引擎，构成循环依赖。

**类型一致性：** `AgentEvent` / `ToolSpec` / `ToolResult` / `AgentTask` / `AgentDeps` / `ContextEngine.build()` 的签名在 Task 2–6 中一致引用，无重命名漂移。

**已知未决项（实施时需现场确认，已在对应步骤标注）：**
- Task 4 Step 12：`mcp_server.list_tools()` 的确切返回结构未实测，`app/api/tools.py` 的改写需以实际签名为准
- Task 4 Step 11：`InMemoryTransport` 位于 `mcp.client._memory` 私有路径，版本升级可能失效，已给出替代方案
