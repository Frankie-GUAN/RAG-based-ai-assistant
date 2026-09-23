# P0 Agent Core Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把「带路由的 RAG pipeline」改造成真正的 ReAct Agent，并兑现 README 里关于 MCP 协议与 token 流式的两处虚假宣称。

**第二版修订（2026-09-23）：** 在动手前对该仓库做了一次缺陷排查，发现 4 个计划外但会削弱 P0 成果的问题（检索链路零缓存、评测阻塞事件循环、历史摘要永久丢上下文、消息时序不确定）。它们已作为 Task 2 / 3 / 9 / 10 并入本计划，原 Task 2–6 顺延为 Task 4–8。详见「与原始六任务版本的偏离」。

**Architecture:** 先定义 `AgentEvent` 事件协议（Python 端），再依次落地上下文工程引擎、真 MCP 工具层、ReAct 核心循环、真流式 SSE。工具通过 `MCPServer` 在同进程内暴露（`InMemoryTransport`），Agent 作为 MCP Client 调用；LLM 调用用 `astream` 累积 `AIMessageChunk`，同时获得 token 级流式与 tool calling。

**Tech Stack:** Python 3.11 · FastAPI · LangGraph/LangChain 1.x · `mcp==2.2.0` · pytest + anyio · Vue 3 + TypeScript

**依赖关系：** Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10。

其中真正的硬约束只有三条：

- **Task 2 必须先于 Task 6/7。** 新 Agent Core 会并发调用工具，不先做单例化会导致 N 份 2GB 模型同时驻留。
- **Task 3 必须先于 Task 9。** 滚动摘要要靠 `Message.id` 定义窗口边界，而 id 排序正是 Task 3 的产物。
- **Task 9、10 排在 Task 8 之后。** Task 10 依赖 Task 7 产出的 Agent Core。

其余任务彼此独立，顺序可按需调整。Task 5 与 Task 6 无相互依赖，可并行。

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

### 2026-09-23 缺陷排查补充验证

| 事实 | 结论 |
|---|---|
| **检索链路零缓存** | `get_embeddings()` 每次调用都新建 `HuggingFaceEmbeddings`。实测 `load_vectorstore()` 连续两次耗时 **9.82s / 2.29s**。全项目无 `lru_cache`、无单例 |
| `lru_cache` 不可用于此处 | 官方文档：cache miss 时被包装函数**可能被多个线程同时进入**。对 2GB 模型等于偶发双份常驻内存 |
| **消息时序不确定** | `messages.created_at` 是秒精度 `datetime`；`save_message` 只 `db.add` 不 flush，两条消息在 `db.commit()` 时才一起 INSERT。实测每对 user/assistant 时间戳完全相同（id 35/36 均为 `2026-06-03 17:58:52`），而 `Conversation.messages` 正按 `created_at` 排序 |
| **历史摘要只生成一次** | 判据是 `if not conv.summary`，其后永不更新。窗口=10 时第 11 条触发摘要（只摘要第 1 条），之后第 2 条至第 N-10 条既不在摘要里也不在最近窗口里，静默丢失 |
| **摘要失败会被永久缓存** | `_generate_summary` 的 `except` 返回 `"历史对话 (N 条消息)"`，被写入 `conv.summary` 后不再重试 |
| **评测阻塞事件循环** | `api/evaluation.py:18` 是 `async def` 却直接调用同步的 `run_evaluation`（内含 `agent_graph.invoke` + 每查询 2 次 LLM + `time.sleep(3)`）。3 条 query 即冻结整个服务器 30~60s。对照 `chat.py:34,60` 用了 `run_in_executor` |
| 无迁移工具 | 仓库无 alembic。`Base.metadata.create_all` 只建表不改表，新增列在已有库上会静默不生效 |
| `evaluation_service` 是计划缺口 | 它 import `agent_graph` 与 `AgentState`，但不在原计划「修改」清单里。Task 7 删除 `graph.py` 后它会 ImportError，且原计划只标注了 `api/chat.py` |
| reranker 未接入 | `Reranker` 已实现且有测试，但 `hybrid_search` 从不调用它；README 架构图画了这一级 |

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
| `backend/app/db/migrate.py` | 轻量幂等列检查（不引入 alembic，见 Task 9） |
| `backend/tests/test_retrieval_singletons.py` | 检索链路单例化测试（Task 2） |
| `backend/tests/test_message_order.py` | 消息时序确定性测试（Task 3） |
| `backend/tests/test_summary_rollover.py` | 滚动增量摘要测试（Task 9） |
| `backend/tests/test_eval_nonblocking.py` | 评测不阻塞事件循环的回归测试（Task 10） |

**修改：**

| 文件 | 改动 |
|---|---|
| `backend/requirements.txt` | 删死依赖、加 `mcp` |
| `backend/tests/conftest.py` | 加 `anyio_backend` fixture |
| `backend/app/tools/document_search.py` | 读路径改用 `get_bm25_index()`（Task 2）；返回结构化 list、移除 registry 注册（Task 6） |
| `backend/app/tools/web_search.py` | 同上 |
| `backend/app/tools/document_parser.py` | 同上 |
| `backend/app/config.py` | 加上下文预算配置 |
| `backend/app/api/chat.py` | 真流式，转发 AgentEvent |
| `backend/app/services/chat_service.py` | 改为 async，走 Agent Core |
| `frontend/src/composables/useChat.ts` | 适配新事件格式 |
| `backend/app/rag/vector_store.py` | embeddings 与 Chroma 句柄单例化（Task 2） |
| `backend/app/rag/hybrid_search.py` | 向量侧改用 `hybrid_top_k` 召回候选（Task 2） |
| `backend/app/rag/bm25_index.py` | 单例化 + 原子替换 `(documents, index)`（Task 2） |
| `backend/app/rag/reranker.py` | 单例化，但**本轮不接入查询链路**（Task 2） |
| `backend/app/services/knowledge_service.py` | 改用 `get_bm25_index().add()`（Task 2） |
| `backend/app/config.py` | 追加 `preload_models`（Task 2）与摘要相关配置（Task 9） |
| `backend/app/models/conversation.py` | 加 `summary_upto_message_id`（Task 9） |
| `backend/app/services/conversation_service.py` | 滚动增量摘要，失败不缓存（Task 9） |
| `backend/app/models/evaluation.py` | 移除两个从未写入的死列（Task 10） |
| `backend/app/services/evaluation_service.py` | 改走 Agent Core，清理 Gemini 残留（Task 7 / 10） |
| `backend/app/api/evaluation.py` | 改用专属 executor，不阻塞事件循环（Task 10） |
| `backend/app/main.py` | lifespan 挂 MCP client（Task 8）+ 预热模型与列检查（Task 2 / 9） |
| `frontend/src/composables/useEvaluation.ts` | 移除已无意义的 `has_docs`（Task 10） |
| `README.md` | 修正 reranker 措辞（Task 2） |
| `backend/tests/test_reranker.py` | 移出默认收集（需下载 2GB 模型） |

**删除：**

| 文件 | 原因 |
|---|---|
| `backend/app/tools/registry.py` | 被 MCP 取代（这正是 README 谎称 MCP 的地方） |
| `backend/app/tools/schemas.py` | schema 改由 MCP 协议自动生成 |
| `backend/app/agent/graph.py` | 被 `core.py` 取代 |
| `backend/app/agent/nodes.py` | 同上 |
| `backend/tests/test_agent.py` | 直接调 `agent_graph.invoke`，且需真实 DeepSeek key；已被 `test_agent_core.py` 全面取代（Task 7） |

---

## Task 1: 清理死依赖 + 锁定版本

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/requirements-dev.txt`

- [x] **Step 1: 确认死依赖确实没被引用**

```bash
cd backend
grep -rn "google.generativeai\|langchain_google_genai\|ragas" app/ tests/ || echo "确认：无任何引用"
```

Expected: 输出 `确认：无任何引用`

- [x] **Step 2: 确认 `ToolRegistry` 只被将要删除的文件引用**

```bash
cd backend
grep -rn "tool_registry\|tools.registry\|tools.schemas\|TOOL_SCHEMAS" app/ tests/
```

Expected: 命中 `app/tools/registry.py`、`app/tools/schemas.py`、`app/tools/document_search.py`、`app/tools/web_search.py`、`app/tools/document_parser.py`、`app/tools/__init__.py`、`app/api/tools.py`、**`app/agent/nodes.py`**。

记下这两个「任务外但会受牵连」的文件：

- `app/api/tools.py` —— 对外暴露工具列表接口，Task 6 Step 12 改它。
- `app/agent/nodes.py` —— `retrieve_node` / `web_search_node` 直接调 `tool_registry.execute(...)`。它会在 Task 7 Step 8 被删除，所以本任务不动它；但这也说明**在 Task 6 删掉 registry 之后、Task 7 删掉 nodes.py 之前，应用处于无法导入的中间态**。这是计划内的已知状态，Task 7 Step 11 已标注。

补充（2026-09-23）：`app/services/evaluation_service.py` 同样 import 了 `agent_graph` 与 `AgentState`，原计划漏了它。Task 7 Step 9 / Step 10 已补上对应的处理步骤。

- [x] **Step 3: 重写 `backend/requirements.txt`**

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

- [x] **Step 4: 新建 `backend/requirements-dev.txt`**

```
-r requirements.txt
pytest>=8.0.0
```

- [x] **Step 5: 安装并验证依赖可解析**

```bash
cd backend
../.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

Expected: 成功，无冲突。`mcp==2.2.0` 被安装。

- [x] **Step 6: 验证 pytest 与 mcp 可用**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest --version
../.venv/Scripts/python.exe -c "from mcp.server import MCPServer; print('mcp ok')"
```

Expected: 打印 pytest 版本号，以及 `mcp ok`

- [x] **Step 7: 验证应用仍能导入**

```bash
cd backend
../.venv/Scripts/python.exe -c "from app.main import app; print('app imports ok')"
```

Expected: `app imports ok`（此时尚未删除 registry，仍可导入）

- [x] **Step 8: Commit**

```bash
git add backend/requirements.txt backend/requirements-dev.txt
git commit -m "chore: drop dead deps (gemini, ragas), add mcp, add pytest

langchain lower bound was >=0.3.0 but 1.2.18 is installed.
pytest was never installed, so the 4 existing test files could not run."
```

---

## Task 2: 检索链路单例化

**为什么排在 MCP 工具层之前（不是可选的优化）：** Task 7 的 Agent Core 会用 `asyncio.gather` 并发发起多个工具调用，每个工具经 `asyncio.to_thread` 落到线程池 —— 也就是说 `search_documents` 会被**并发**调用。当前「每次调用新建一个 `HuggingFaceEmbeddings`」在并发下不是慢，而是 N 个线程各加载一份约 2GB 的模型，直接 OOM。

**实测数据（2026-09-23）：** `load_vectorstore()` 连续两次调用耗时 **9.82s / 2.29s**。链路 `hybrid_search → _vector_search → get_retriever → load_vectorstore → get_embeddings()` 全无缓存，全项目无 `lru_cache`、无单例。

**为什么不能用 `functools.lru_cache`：** 官方文档明确写明 cache miss 时被包装函数**可能被多个线程同时进入**：

> It is possible for the wrapped function to be called more than once if another thread makes an additional call before the initial call has been completed and cached.

对 2GB 模型，这就是偶发的双份常驻内存 —— 正是要防的那件事。必须用显式双检锁。

**本任务同时修第二个缺陷：向量侧只召回 4 个候选。** `get_retriever()` 把 k 硬编码成 `settings.top_k`（=4），于是 `hybrid_search(vector_top_k=10)` 传进来的值被无声忽略，而 `settings.hybrid_top_k = 10` **在整个代码库里从未被任何地方引用**。结果 RRF 融合的是 4 路向量结果 + 10 路 BM25 结果，而非 10+10 —— 这直接削弱了混合检索的召回率。它和单例化改的是同一个文件（`get_retriever` 与 `_vector_search`），因此并入本任务。见 Step 8–11。

**Files:**
- Modify: `backend/app/rag/vector_store.py`
- Modify: `backend/app/rag/bm25_index.py`
- Modify: `backend/app/rag/hybrid_search.py`
- Modify: `backend/app/rag/reranker.py`
- Modify: `backend/app/services/knowledge_service.py`
- Modify: `backend/app/tools/document_parser.py`
- Modify: `backend/app/tools/document_search.py`
- Modify: `backend/app/config.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_hybrid_search.py`
- Modify: `README.md`
- Create: `backend/tests/test_retrieval_singletons.py`
- Modify: `backend/tests/conftest.py`（追加 autouse fixture，关掉 `preload_models`）

> **本任务已交付（2026-09-24），下面的代码块是**原始规范**，不要照抄。**
> 最终代码以 `backend/app/rag/*.py` 与 `backend/tests/*` 为准 —— 实现经过两轮评审加固，照抄下面的代码块会**重新引入已修复的问题**：
>
> | 位置 | 原始规范里的问题 | 已改为 |
> |---|---|---|
> | Step 4 `get_retriever` | `k or settings.hybrid_top_k` 把 0 当「没传」 | `if k is None`；并注明 chromadb 1.5.9 接受 k=0 构造、但在 `invoke` 时抛 `TypeError` |
> | Step 4 `add_documents` | `assert vectorstore is not None`（`python -O` 下被剥掉） | 显式 `raise RuntimeError`；`get_retriever` 补返回标注 |
> | Step 5 `load()` | 只捕 `UnpicklingError/EOFError/OSError` | 宽到 `Exception` —— pickle 里嵌的是**活的** BM25Okapi/Document，rank_bm25 升级、类改名、dict 结构变更抛的是 `AttributeError`/`KeyError`，窄元组接不住 |
> | Step 5 `load()` | 静默 `return False` 之后单例把**空索引固化**，下次上传的 `add()` 会用空基础覆盖磁盘 —— 2196 篇变 1 篇，而仓库里没有从 Chroma 重建的路径 | `add()` 在 `_load_failed` 时拒绝落盘；`get_bm25_index()` 以 30s 节流重试 |
> | Step 5 `save()` | 直接 `open(index_path, "wb")` 就地覆盖 | 先写 `<name>.<pid>.tmp` 再 `os.replace`（原子；带 pid 才不会两个进程互 rename） |
> | Step 5 `build()` | `if corpus else None` —— `corpus == [[]]` 是**真值**，`BM25Okapi` 会 `ZeroDivisionError`（空 PDF 页走上传接口可达） | `if any(corpus)` |
> | Step 5 `_documents` / `_index` 属性 | 死代码，且每次访问都重读 `_pair`，看着原子其实不是 | 已删除（保留 `document_count`） |
> | Step 5/10 的 `top_k` | 负数走切片 `[:-1]`，含义是「除最后一个之外的全部」 | 三处（`_vector_search`/`BM25Index.search`/`_rrf_fuse`）统一为 `top_k <= 0 → []` |
> | Step 10 `hybrid_search` | `or` 默认值同样把 0 当「没传」 | `if x is None` |
> | Step 12 `main.py` | 预载无异常处理 —— 检索层故障会让整个 API 起不来 | `try/except Exception`，降级为「首次请求时重试」 |
> | Step 3 的 Expected | 声称测试全红；实际 `test_bm25_search_is_safe_while_rebuilding` 修复前**就是绿的** | 见 Step 3 的修订说明 |
> | Step 2 的那个用例 | 只跑增长式重建，触发不了旧实现的失配 | 追加**收缩式**重建（失配方向才会翻转），并补了损坏文件恢复、空白语料、负数 `top_k` 等用例 |
> | Step 8 的测试 | 把 `hybrid_top_k` patch 成 **10**，而 10 就是默认值 —— 测不出写死字面量 | patch 成 **7** |

- [x] **Step 1: 在 `backend/app/config.py` 的 `Settings` 类中追加配置**

插入到 `reranker_model` 之后：

```python
    # 启动时预载检索链路（embeddings + Chroma + BM25）。
    # 测试里若用 `with TestClient(app)` 会触发 lifespan，设 PRELOAD_MODELS=false 跳过。
    preload_models: bool = True
```

- [x] **Step 2: 写失败的测试 `backend/tests/test_retrieval_singletons.py`**

**全部 monkeypatch 掉真实模型类**，因此不下载 BGE-M3（约 2GB），属于可 CI 的轻量层 —— 与 Task 7 用 `ScriptedChatModel` 替代真实 LLM 是同一套思路。

```python
"""检索链路单例化测试。

最关键的一条是 test_embeddings_singleton_holds_under_concurrency：
它锁死「不许退回 functools.lru_cache」—— lru_cache 在 miss 时允许多线程同时进入
被包装函数，对 2GB 模型就是偶发双份常驻内存。
"""
import threading

import pytest

from app.rag import bm25_index as bm25_mod
from app.rag import reranker as reranker_mod
from app.rag import vector_store as vs_mod


class _CountingEmbeddings:
    """替身：记录构造次数，不加载任何权重。"""

    constructions = []

    def __init__(self, model_name=None, model_kwargs=None, encode_kwargs=None):
        type(self).constructions.append(model_name)
        self.model_name = model_name

    def embed_query(self, text):
        return [0.0]


class _SlowCountingEmbeddings(_CountingEmbeddings):
    """构造时故意停顿，放大并发窗口，让「构造了多次」必然暴露出来。"""

    def __init__(self, *args, **kwargs):
        import time

        time.sleep(0.05)
        super().__init__(*args, **kwargs)


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """每个用例前把模块级单例清空，避免相互污染。"""
    monkeypatch.setattr(vs_mod, "_embeddings", None, raising=False)
    monkeypatch.setattr(vs_mod, "_vectorstore", None, raising=False)
    monkeypatch.setattr(bm25_mod, "_bm25", None, raising=False)
    monkeypatch.setattr(reranker_mod, "_reranker", None, raising=False)


def test_embeddings_constructed_once_and_reused(monkeypatch):
    _CountingEmbeddings.constructions = []
    monkeypatch.setattr(vs_mod, "HuggingFaceEmbeddings", _CountingEmbeddings)

    first = vs_mod.get_embeddings()
    second = vs_mod.get_embeddings()

    assert first is second
    assert len(_CountingEmbeddings.constructions) == 1


def test_embeddings_singleton_holds_under_concurrency(monkeypatch):
    """8 个线程同时抢 —— 构造必须只发生一次。"""
    _SlowCountingEmbeddings.constructions = []
    monkeypatch.setattr(vs_mod, "HuggingFaceEmbeddings", _SlowCountingEmbeddings)

    barrier = threading.Barrier(8)
    results = []
    errors = []

    def worker():
        try:
            barrier.wait()
            results.append(vs_mod.get_embeddings())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(_SlowCountingEmbeddings.constructions) == 1
    assert len({id(r) for r in results}) == 1


def test_bm25_index_is_shared(monkeypatch, tmp_path):
    monkeypatch.setattr(bm25_mod.settings, "data_dir", tmp_path, raising=False)

    assert bm25_mod.get_bm25_index() is bm25_mod.get_bm25_index()


def test_bm25_search_is_safe_while_rebuilding():
    """build() 整体替换 (documents, index)，search() 快照一次 —— 二者交错不得抛异常。

    真实触发场景：同一个 ReAct 轮次里模型并发发起 parse_document 与 search_documents。
    若 build 分别替换两个属性，search 会读到长度不匹配的一对，直接 IndexError。
    """
    from langchain_core.documents import Document

    index = bm25_mod.BM25Index()
    index.build([Document(page_content=f"旧片段{i}") for i in range(3)])

    stop = threading.Event()
    failures = []

    def searcher():
        while not stop.is_set():
            try:
                index.search("旧片段", top_k=5)
            except Exception as exc:  # noqa: BLE001
                failures.append(exc)
                return

    threads = [threading.Thread(target=searcher) for _ in range(4)]
    for t in threads:
        t.start()

    for round_no in range(50):
        index.build([Document(page_content=f"新片段{round_no}-{i}") for i in range(100)])

    stop.set()
    for t in threads:
        t.join()

    assert failures == []


def test_reranker_is_singleton(monkeypatch):
    constructions = []

    class _FakeCrossEncoder:
        def __init__(self, model_name=None):
            constructions.append(model_name)

    monkeypatch.setattr(reranker_mod, "CrossEncoder", _FakeCrossEncoder)

    assert reranker_mod.get_reranker() is reranker_mod.get_reranker()
    assert len(constructions) == 1
```

- [x] **Step 3: 运行测试，确认失败**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_retrieval_singletons.py -v
```

Expected: **4 failed, 1 passed** —— 未实现的 `get_bm25_index` / `get_reranker` 报 AttributeError，并发单例断言失败。

**注意那 1 passed**：`test_bm25_search_is_safe_while_rebuilding` 修复前就是绿的，不是先红后绿的回归测试。旧实现先写 `_documents` 再写 `_index`，而 `add()` 只追加、索引只增不减，所以「index 比 documents 长」这个致命失配方向不可达。它是不变量护栏。真正让它变红需要在重建里加入**收缩**段 —— 文档数变少时失配才翻转，实测增长段 0/9 触发、加上收缩段 9/9 触发 `IndexError`。本步之后的测试文件已按此加强。

- [x] **Step 4: 重写 `backend/app/rag/vector_store.py` 的加载部分**

```python
import threading
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

_embeddings: HuggingFaceEmbeddings | None = None
_embeddings_lock = threading.Lock()

_vectorstore: Chroma | None = None
_vectorstore_lock = threading.Lock()


def get_embeddings() -> HuggingFaceEmbeddings:
    """BGE-M3 的进程内单例。

    不用 functools.lru_cache：cache miss 时它允许多个线程同时进入被包装函数，
    对约 2GB 的模型等于偶发双份常驻内存。双检锁保证只构造一次。
    """
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                model = HuggingFaceEmbeddings(
                    model_name=settings.embed_model_path,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                model.embed_query("ping")  # 预热，确保权重真正就绪
                _embeddings = model
    return _embeddings


def has_persisted_index() -> bool:
    return (settings.chroma_dir / "chroma.sqlite3").exists()


def get_vectorstore(create: bool = False) -> Optional[Chroma]:
    """Chroma 句柄的进程内单例。

    读写必须共用同一句柄 —— 旧实现每次 `Chroma(persist_directory=...)` 都新建客户端，
    导致「上传后立刻检索」是碰运气成立的。
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    if not create and not has_persisted_index():
        return None

    with _vectorstore_lock:
        if _vectorstore is None:
            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            _vectorstore = Chroma(
                persist_directory=str(settings.chroma_dir),
                embedding_function=get_embeddings(),
            )
    return _vectorstore


def add_documents(documents: List[Document]) -> Chroma:
    vectorstore = get_vectorstore(create=True)
    assert vectorstore is not None
    vectorstore.add_documents(documents)
    return vectorstore


def get_retriever(k: int | None = None):
    """k 默认取 hybrid_top_k（每路召回的候选数），**不是** top_k（最终返回数）。

    此前这里硬编码 settings.top_k=4，导致 hybrid_search 的 vector_top_k 参数
    被无声忽略、settings.hybrid_top_k 成为死配置 —— RRF 实际只融合了 4 路
    向量结果。修正见 Step 8–11。
    """
    vectorstore = get_vectorstore()
    if vectorstore is None:
        raise ValueError("No persisted vector store found")
    return vectorstore.as_retriever(search_kwargs={"k": k or settings.hybrid_top_k})
```

注意：`Chroma.from_documents(...)` 的旧分支已消失 —— 对空集合调用 `add_documents` 等价且更简单。同时删掉 `load_vectorstore`（`knowledge_service.py` 曾导入它但从未调用）。

- [x] **Step 5: 重写 `backend/app/rag/bm25_index.py` 的并发与单例部分**

要点是 `build()` **整体替换** `(documents, index)` 元组，`search()` 开头快照一次。引用赋值在 CPython 下是原子的，因此读路径无需加锁（加了会把并发检索串行化）。

```python
import pickle
import threading
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings


class BM25Index:
    def __init__(self):
        # (documents, index) 作为**一对**整体替换，search 永远读不到长度不匹配的组合
        self._pair: tuple[List[Document], BM25Okapi | None] = ([], None)
        self._write_lock = threading.Lock()
        self._index_path = settings.data_dir / "bm25_index.pkl"

    @property
    def _documents(self) -> List[Document]:
        return self._pair[0]

    @property
    def _index(self) -> BM25Okapi | None:
        return self._pair[1]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re

        return [t.lower() for t in re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text)]

    def build(self, documents: List[Document]) -> None:
        corpus = [self._tokenize(doc.page_content) for doc in documents]
        index = BM25Okapi(corpus) if corpus else None
        self._pair = (list(documents), index)  # 原子替换

    def add(self, chunks: List[Document]) -> None:
        """把新片段并入索引并落盘。两条建索引路径共用此方法，避免 Chroma 与 BM25 分叉。"""
        with self._write_lock:
            self.build([*self._pair[0], *chunks])
            self.save()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        documents, index = self._pair  # 快照：整个 search 用这一对
        if index is None:
            return []
        tokens = self._tokenize(query)
        scores = index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        max_score = scores[ranked[0][0]] if ranked and scores[ranked[0][0]] > 0 else 1.0
        return [(documents[i], score / max_score) for i, score in ranked if score > 0]

    def save(self) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        documents, index = self._pair
        with open(self._index_path, "wb") as f:
            pickle.dump({"documents": documents, "index": index}, f)

    def load(self) -> bool:
        if not self._index_path.exists():
            return False
        with open(self._index_path, "rb") as f:
            data = pickle.load(f)
        self._pair = (data["documents"], data["index"])
        return True

    @property
    def document_count(self) -> int:
        return len(self._pair[0])


_bm25: BM25Index | None = None
_bm25_lock = threading.Lock()


def get_bm25_index() -> BM25Index:
    """BM25 索引的进程内单例，懒加载磁盘上的 pickle。"""
    global _bm25
    if _bm25 is None:
        with _bm25_lock:
            if _bm25 is None:
                index = BM25Index()
                index.load()
                _bm25 = index
    return _bm25
```

- [x] **Step 6: 给 `backend/app/rag/reranker.py` 加单例**

```python
_reranker: "Reranker | None" = None
_reranker_lock = threading.Lock()


def get_reranker() -> "Reranker":
    """Cross-Encoder 的进程内单例。

    本轮不接入查询链路（CPU 上精排 10 个候选约需 1~3 秒，会顶掉「首 token < 800ms」
    的验收目标）。此处只把底座准备好，接入与检索指标（Hit Rate / NDCG）一起留到 P1。
    """
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = Reranker()
    return _reranker
```

并在文件顶部补 `import threading`。

- [x] **Step 7: 让两条建索引路径共用单例**

`backend/app/services/knowledge_service.py` 里把

```python
    bm25 = BM25Index()
    if bm25.load():
        all_docs = list(bm25._documents)
    else:
        all_docs = []
    all_docs.extend(chunks)
    bm25.build(all_docs)
    bm25.save()
```

替换为

```python
    get_bm25_index().add(chunks)
```

并同步调整 import：删掉未使用的 `load_vectorstore`，把 `BM25Index` 换成 `get_bm25_index`。

`backend/app/tools/document_parser.py`（写路径）与 `backend/app/tools/document_search.py`（**读路径，也就是每次检索都走的热路径**）做同样的替换：

```python
# document_search.py
def search_documents(query: str, top_k: int = 4) -> str:
    results = hybrid_search(query, get_bm25_index(), final_top_k=top_k)
    ...
```

**这两个文件在 Task 6 会被整体改写，改写时必须沿用 `get_bm25_index()`** —— 否则 Task 2 对 BM25 的修复会被 Task 6 悄悄改回去，读路径重新变成「每次检索都重建索引 + 重读 pickle」。Task 6 Step 1 / Step 3 的代码已按此写好。

顺带把 `backend/app/services/knowledge_service.py` 与 `backend/app/tools/document_parser.py` 里重复的「load → extend → build → save」逻辑删掉 —— 这正是 CLAUDE.md 里记的「两条建索引路径可能分叉」的根源，现在只剩 `add()` 一处。

- [x] **Step 8: 在 `backend/tests/test_hybrid_search.py` 追加失败的测试**

该文件已存在（3 条 `_rrf_fuse` 用例），直接追加：

```python
import pytest

from app.rag import hybrid_search as hybrid_mod


class _EmptyRetriever:
    def invoke(self, query):
        return []


class _EmptyBM25:
    def search(self, query, top_k=10):
        return []


def test_vector_side_requests_hybrid_top_k(monkeypatch):
    """回归：向量侧过去固定只召回 settings.top_k=4 个候选。

    get_retriever() 把 k 硬编码成 settings.top_k，于是 hybrid_search 的
    vector_top_k 参数被无声忽略、settings.hybrid_top_k 成为死配置 ——
    RRF 实际融合的是 4 路向量结果 + 10 路 BM25 结果，而非 10+10。
    """
    requested = {}

    def _fake_get_retriever(k=None):
        requested["k"] = k
        return _EmptyRetriever()

    monkeypatch.setattr(hybrid_mod, "get_retriever", _fake_get_retriever)
    # 刻意用 7，而不是 app/config.py 里的默认值 10：否则 `vector_top_k or 10`
    # 这种写死字面量的回归也能骗过断言
    monkeypatch.setattr(hybrid_mod.settings, "hybrid_top_k", 7, raising=False)

    hybrid_mod.hybrid_search("劳动合同如何解除", _EmptyBM25())

    assert requested["k"] == 7
```

- [x] **Step 9: 运行测试，确认失败**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_hybrid_search.py -v
```

Expected: 1 failed, 3 passed —— 新用例拿到 `k=None`，因为 `_vector_search` 从未把 `top_k` 传给 retriever

- [x] **Step 10: 修正 `backend/app/rag/hybrid_search.py`**

`_vector_search` 必须把 `top_k` 传下去；`hybrid_search` 的三个默认值改从配置读取，让 `hybrid_top_k`（每路召回的候选数）与 `top_k`（最终返回数）各司其职。

```python
def _vector_search(query: str, top_k: int) -> List[Tuple[Document, float]]:
    retriever = get_retriever(top_k)          # ← 此前这个 k 被忽略
    docs = retriever.invoke(query)
    # 这里是名次占位分，不是相似度。RRF 只用名次，故不影响融合结果，
    # 但不要把它当相关度读。
    return [(doc, 1.0 - i * 0.05) for i, doc in enumerate(docs)][:top_k]


def hybrid_search(
    query: str,
    bm25_index: BM25Index,
    vector_top_k: int | None = None,
    bm25_top_k: int | None = None,
    final_top_k: int | None = None,
) -> List[Tuple[Document, float]]:
    """两路各召回 hybrid_top_k 个候选，RRF 融合后返回 top_k 个。"""
    vector_top_k = vector_top_k or settings.hybrid_top_k
    bm25_top_k = bm25_top_k or settings.hybrid_top_k
    final_top_k = final_top_k or settings.top_k

    vector_results = _vector_search(query, top_k=vector_top_k)
    bm25_results = bm25_index.search(query, top_k=bm25_top_k)
    return _rrf_fuse(vector_results, bm25_results, top_k=final_top_k)
```

`get_retriever` 的签名已在 Step 4 改好（`k: int | None = None`，默认取 `hybrid_top_k`）。

注意调用方 `document_search.search_documents(query, top_k=4)` 仍显式传 `final_top_k`，返回条数不变 —— 变的只是**候选池**从 4 扩到 10。

- [x] **Step 11: 运行测试，确认通过**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_hybrid_search.py -v
```

Expected: 4 passed

- [x] **Step 12: 在 `backend/app/main.py` 的 lifespan 中预热**

```python
import asyncio
import logging

logger = logging.getLogger(__name__)


def _preload_retrieval() -> None:
    """在启动线程里把检索链路拉起来，避免第一个请求付 10s 冷启动、
    也避免并发请求一起去加载。"""
    from app.rag.bm25_index import get_bm25_index
    from app.rag.vector_store import get_embeddings, get_vectorstore, has_persisted_index

    get_embeddings()
    if has_persisted_index():
        get_vectorstore()
    get_bm25_index()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    if settings.preload_models:
        import time

        started = time.perf_counter()
        await asyncio.to_thread(_preload_retrieval)
        logger.info("检索链路预载完成，耗时 %.2fs", time.perf_counter() - started)
    yield
```

- [x] **Step 13: 修正 README 关于 reranker 的措辞**

README 的架构图画了 `Cross-Encoder Reranker → Top-4` 这一级，但查询链路从不调用它。**本轮不接线**（理由见 Step 6），所以必须把这条宣称改成实话 —— 在 README 的架构图与「项目亮点」里把它标注为「已实现、未启用，P1 接入并附检索指标」。

- [x] **Step 14: 运行测试，确认通过**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_retrieval_singletons.py -v
```

Expected: 5 passed

- [x] **Step 15: 手工确认收益**

```bash
cd backend
../.venv/Scripts/python.exe -c "
import time
from app.rag.vector_store import get_vectorstore
t0 = time.perf_counter(); get_vectorstore(); t1 = time.perf_counter()
get_vectorstore(); t2 = time.perf_counter()
print(f'首次 {t1-t0:.2f}s / 二次 {t2-t1:.4f}s')
"
```

Expected: 二次调用接近 0（改造前实测 2.29s）

- [x] **Step 16: Commit**

```bash
git add -A backend/app backend/tests README.md
git commit -m "perf: make the retrieval stack process-wide singletons

get_embeddings() rebuilt HuggingFaceEmbeddings on every call, so every
search reloaded BGE-M3: measured 9.82s cold / 2.29s warm. The new Agent
Core calls tools concurrently, so the old behaviour would have loaded N
copies of a 2GB model at once.

Deliberately not functools.lru_cache: it may invoke the wrapped function
from several threads on a cache miss, which is the double-load we are
trying to prevent.

BM25Index.build() now swaps (documents, index) as one tuple so a search
racing a rebuild cannot read a mismatched pair.

Retrieval also only ever fused four vector candidates: get_retriever()
hardcoded k=settings.top_k, so hybrid_search's vector_top_k was silently
ignored and settings.hybrid_top_k was dead config. The two settings now
mean what their names say — hybrid_top_k per source, top_k for the
final result count."
```

---

## Task 3: 消息时序确定性

**问题（2026-09-23 实测确认）：** `messages.created_at` 是秒精度 `datetime`；`save_message` 只 `db.add` 不 flush，两条消息在 `db.commit()` 时才一起 INSERT，因此**同一次问答的 user 与 assistant 消息时间戳完全相同**：

```
id 35  conversation_id 17  user       2026-06-03 17:58:52
id 36  conversation_id 17  assistant  2026-06-03 17:58:52
```

而 `Conversation.messages` 正按 `created_at` 排序 —— 并列值的返回顺序不保证，重载历史时可能颠倒，LLM 会收到「助手先答、用户后问」。

**为什么修法是改排序键，而不是把列升到 `DATETIME(6)`：** `AUTO_INCREMENT` 天然反映真实插入顺序且严格单调；改列精度是 schema 变更、要迁移，换来的时序信息反而不如 id 直接。**本任务零 schema 改动。**

**波及面比看上去大：** `get_conversation_with_context` 与 `_generate_summary` 都用 `conv.messages` —— 摘要的 transcript 顺序此前可能也是错的。`list_conversations` 已在用 `func.max(Message.id)`（正确做法），改完后两处终于一致。

**Files:**
- Modify: `backend/app/models/conversation.py`
- Create: `backend/tests/test_message_order.py`
- Modify: `backend/tests/conftest.py`

- [ ] **Step 1: 新建 `backend/pytest.ini`，并在 `backend/tests/conftest.py` 追加 SQLite fixture**

先解决「一套命令全绿」的前提：`tests/test_reranker.py` 会下载并加载约 2GB 的 Cross-Encoder，不能进默认收集。仓库当前没有任何 pytest 配置，正好一并建立。

`backend/pytest.ini`：

```ini
[pytest]
testpaths = tests
markers =
    requires_model: 需要下载或加载本地大模型（BGE-Reranker 约 2GB），默认不收集
addopts = -m "not requires_model"
```

并给 `tests/test_reranker.py` 的用例加上 `@pytest.mark.requires_model`（同时补 `import pytest`）。需要显式运行它时：

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_reranker.py -m requires_model -v
```

然后是 SQLite fixture：

```python
@pytest.fixture
def sqlite_session():
    """独立的内存 SQLite 会话。

    用于验证依赖 ORM 配置（而非 MySQL 具体行为）的逻辑。只有 SQLite 内存库能让这类
    测试进 CI —— 仓库其余测试分别依赖真实 MySQL、真实 API key、2GB 模型。
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db.base import Base

    import app.models.conversation  # noqa: F401
    import app.models.message       # noqa: F401

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
```

- [ ] **Step 2: 写失败的测试 `backend/tests/test_message_order.py`**

```python
"""消息时序回归测试。

真实触发场景是秒精度的 created_at 打平（实测同一次问答的两条消息时间戳完全相同）。
但「时间戳打平」在 SQLite 上恰好会按 rowid 返回，即碰巧等于正确顺序，测不出差异。

所以这里让 created_at 的顺序与插入顺序**相反** —— 这样「按时间排」与「按 id 排」
必然给出不同结果，测试才能确定性地把两者区分开。
"""
from datetime import datetime, timedelta

from app.models.conversation import Conversation
from app.models.message import Message


def _add(session, conversation_id, role, content, created_at):
    session.add(
        Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=created_at,
        )
    )
    session.flush()


def _conversation(session):
    conv = Conversation(title="t")
    session.add(conv)
    session.flush()
    return conv


def test_ordering_is_by_id_not_timestamp(sqlite_session):
    session = sqlite_session
    conv = _conversation(session)

    base = datetime(2026, 9, 23, 10, 0, 0)
    # 插入顺序：user → assistant；时间戳顺序相反
    _add(session, conv.id, "user", "问题", base + timedelta(seconds=5))
    _add(session, conv.id, "assistant", "回答", base)

    session.expire_all()
    reloaded = session.get(Conversation, conv.id)

    assert [m.role for m in reloaded.messages] == ["user", "assistant"]


def test_ordering_survives_identical_timestamps(sqlite_session):
    """回归的实际形态：秒精度下同一轮问答的两条消息时间戳相同，只有 id 能定序。"""
    session = sqlite_session
    conv = _conversation(session)

    same_second = datetime(2026, 9, 23, 10, 0, 0)
    _add(session, conv.id, "user", "问题", same_second)
    _add(session, conv.id, "assistant", "回答", same_second)

    session.expire_all()
    reloaded = session.get(Conversation, conv.id)

    assert [m.role for m in reloaded.messages] == ["user", "assistant"]
```

- [ ] **Step 3: 运行测试，确认失败**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_message_order.py -v
```

Expected: `test_ordering_is_by_id_not_timestamp` FAIL —— 得到 `["assistant", "user"]`（按 created_at 排出来的顺序）。第二条用例可能碰巧通过，这正是需要第一条的原因。

- [ ] **Step 4: 修改 `backend/app/models/conversation.py`**

```python
    # 按自增 id 排序，不按 created_at：created_at 是秒精度，同一次问答的
    # user/assistant 两条消息时间戳完全相同（实测确认），无法定序。
    messages = relationship("Message", back_populates="conversation", order_by="Message.id")
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_message_order.py -v
```

Expected: 2 passed

- [ ] **Step 6: 在真实库上复核**

SQLite 不能完全代表 MySQL 的返回顺序，所以在开发库上再确认一次：

```bash
mysql -u root -p rag_agent -e "SELECT conversation_id, GROUP_CONCAT(role ORDER BY id) FROM messages GROUP BY conversation_id ORDER BY conversation_id DESC LIMIT 5;"
```

然后启动后端，对一个已有会话调 `GET /api/conversations/{id}`，确认 `messages` 里 user 在 assistant 之前。

- [ ] **Step 7: Commit**

```bash
git add backend/app/models/conversation.py backend/tests/conftest.py backend/tests/test_message_order.py
git commit -m "fix: order conversation messages by id, not by second-precision timestamp

messages.created_at is a MySQL DATETIME with no fractional seconds, and
save_message only db.add()s — both rows INSERT at commit, so a question
and its answer share a timestamp exactly (verified: ids 35/36 are both
2026-06-03 17:58:52). Ordering by created_at left reload order undefined,
so history replay could hand the model an answer before its question.

This also fixes the transcript fed to _generate_summary, whose ordering
was wrong for the same reason."
```

---

## Task 4: Agent 协议（Python 端）

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

from app.agent.protocol import (
    ErrorEvent,
    FinalEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    to_sse,
)

# 与前端事件分发（frontend/src/composables/useChat.ts）共享的契约。
# 任何一边增删事件类型，此集合必须同步更新。
EXPECTED_EVENT_TYPES = {
    "tool_call",
    "tool_result",
    "token",
    "final",
    "error",
}


def test_token_event_serializes_with_type_key():
    assert TokenEvent(text="你好").to_dict() == {"type": "token", "text": "你好"}


def test_tool_call_carries_args():
    payload = ToolCallEvent(id="call_1", name="search_documents", args={"query": "合同解除"}).to_dict()

    assert payload["type"] == "tool_call"
    assert payload["args"] == {"query": "合同解除"}


def test_tool_result_records_failure_without_raising():
    payload = ToolResultEvent(id="call_1", ok=False, error="TimeoutError: boom", ms=1200).to_dict()

    assert payload["ok"] is False
    assert payload["error"] == "TimeoutError: boom"
    assert payload["ms"] == 1200


def test_final_event_carries_the_full_answer():
    payload = FinalEvent(answer="劳动合同经双方协商一致可以解除").to_dict()

    assert payload["answer"] == "劳动合同经双方协商一致可以解除"


def test_error_event_is_machine_readable():
    payload = ErrorEvent(code="max_iterations", message="达到迭代上限", recoverable=True).to_dict()

    assert payload["type"] == "error"
    assert payload["code"] == "max_iterations"
    assert payload["recoverable"] is True


def test_every_declared_event_type_is_covered():
    """防止新增事件类型后忘记同步前端契约。"""
    declared = {
        ToolCallEvent.type,
        ToolResultEvent.type,
        TokenEvent.type,
        FinalEvent.type,
        ErrorEvent.type,
    }

    assert declared <= EXPECTED_EVENT_TYPES, f"出现了未登记的事件类型: {declared - EXPECTED_EVENT_TYPES}"


def test_to_sse_keeps_chinese_readable():
    frame = to_sse(TokenEvent(text="检索"))

    assert frame["event"] == "token"
    assert "检索" in frame["data"]  # ensure_ascii=False
    assert json.loads(frame["data"]) == {"type": "token", "text": "检索"}


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

只定义 P0 真正会发出的事件 —— 不留「定义了但无生产者」的类型，
那正是本计划要消灭的那类问题。所有事件继承 AgentEvent，
to_dict() 序列化为 {"type": <事件名>, ...payload}。

新增事件类型时，必须同步：
  1. 本文件
  2. tests/test_protocol.py 的 EXPECTED_EVENT_TYPES
  3. frontend/src/composables/useChat.ts 的事件分发
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True)
class AgentEvent:
    """所有事件的基类。子类用 ClassVar 声明 type 名（ClassVar 不参与 asdict）。"""

    type: ClassVar[str] = "event"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = self.type
        return payload


@dataclass(frozen=True)
class ToolCallEvent(AgentEvent):
    type: ClassVar[str] = "tool_call"
    id: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)


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
class FinalEvent(AgentEvent):
    type: ClassVar[str] = "final"
    answer: str = ""


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

Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add backend/app/agent/protocol.py backend/tests/test_protocol.py backend/tests/conftest.py
git commit -m "feat: add the AgentEvent protocol shared with the frontend

Defines only the five events P0 actually emits — tool_call, tool_result,
token, final, error. Types with no producer are deliberately absent,
since 'declared but never sent' is the same class of problem as the
README claims this work exists to remove."
```

---

## Task 5: 上下文工程引擎

**Files:**
- Create: `backend/app/context/__init__.py`
- Create: `backend/app/context/tokens.py`
- Create: `backend/app/context/engine.py`
- Create: `backend/tests/test_context_engine.py`
- Modify: `backend/app/config.py`

- [ ] **Step 1: 在 `backend/app/config.py` 的 `Settings` 类中追加配置**

插入到 `reranker_model` 之后、`history_window_size` 之前：

```python
    # 上下文工程：分层 token 预算
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
    result = engine.build(system="sys", question="劳动合同如何解除？")

    assert isinstance(result.messages[-1], HumanMessage)
    assert "劳动合同如何解除？" in result.messages[-1].content


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

每层独立预算，层内超限先截断，**不跨层抢占**。
各层比例由 config.py 的 context_* 配置项控制。
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

## Task 6: 真 MCP 工具层

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
from app.rag.bm25_index import get_bm25_index
from app.rag.hybrid_search import hybrid_search


def search_documents(query: str, top_k: int = 4) -> list[dict]:
    """混合检索私有文档库，返回结构化片段列表（相关度降序）。"""
    # 必须用共享单例：此处若写 BM25Index()，每次检索都会重建索引并重读 pickle，
    # 把 Task 2 对 BM25 的修复整体作废。
    results = hybrid_search(query, get_bm25_index(), final_top_k=top_k)

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

from app.rag.bm25_index import get_bm25_index
from app.rag.loader import load_file, split_documents
from app.rag.vector_store import add_documents


def parse_document(file_path: str) -> dict:
    """解析上传文档并建立索引。文件不存在时抛 FileNotFoundError。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    chunks = split_documents(load_file(path))
    add_documents(chunks)
    get_bm25_index().add(chunks)          # Task 2 引入，取代原先的 load/extend/build/save

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

**已知中间态：从本步到 Step 12 之间，应用无法导入。** `app/api/tools.py` 仍 import 被删掉的 `tool_registry`（Step 12 才改），`app/agent/nodes.py` 同样（Task 7 Step 8 才删）。这与 Task 1 Step 2 里记的是同一件事，Step 13 会验证最终可导入。若想每一步都保持可导入，可把 Step 12 提前到本步之前。

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

## Task 7: Agent Core（ReAct 循环）

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
        tool_turn("search_documents", {"query": "合同解除"}),
        text_turn("根据", "资料", "答案是……"),
    ])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="劳动合同解除需要什么条件？"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    assert events_of(events) == [
        "tool_call", "tool_result", "token", "token", "token", "final",
    ]
    call = events[0]
    assert call.name == "search_documents"
    assert call.args == {"query": "合同解除"}
    assert events[1].ok is True


@pytest.mark.anyio
async def test_tool_result_is_fed_back_to_the_model():
    model = ScriptedChatModel([
        tool_turn("search_documents", {"query": "合同解除"}),
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
    assert "合同解除 的结果" in tool_messages[0].content
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
    "你是智能问答助手，基于私有文档知识库与联网搜索回答用户问题。"
    "你可以调用工具检索私有文档、联网搜索、解析上传的文档。\n"
    "规则：\n"
    "1. 需要事实依据时先调用工具，不要凭记忆回答文档中的内容或数据。\n"
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

- [ ] **Step 9: 改写 `backend/app/services/evaluation_service.py`，改走 Agent Core**

**原计划漏了这一处。** `evaluation_service.py` 第 8、9 行 import 了 `agent_graph` 与 `AgentState`，Step 8 删掉 `graph.py` 后它会 ImportError —— 原计划只标注了 `api/chat.py`。而且 `api/evaluation.py` 是 `async def` 却直接调用**同步**的 `run_evaluation`（内含 `time.sleep(3)`），3 条 query 就冻结整个服务器 30~60s。既然要重写，一次写成真异步。

Agent Core 的 `model.astream` 本就是异步的，所以这里只需 `await`，不需要线程池。

```python
"""评测服务：走 Agent Core，全程异步。

原实现是同步的，却被 async 端点直接调用（含 time.sleep(3)），会把整个事件循环冻住。
"""
from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.agent.core import AgentDeps, AgentTask, run_agent_stream
from app.agent.llm import get_chat_model
from app.agent.protocol import FinalEvent, ToolResultEvent
from app.mcp.client import MCPToolClient
from app.services.chat_service import build_context_engine

# 查询之间的人工间隔，用于规避限流。
# 并发执行 + 信号量限流属于评测体系的后续优化，本轮保持串行。
_QUERY_INTERVAL_SECONDS = 3


async def _run_query(
    question: str, tools: MCPToolClient, max_iterations: int = 8
) -> tuple[str, str]:
    """跑一次问答，返回 (答案, 检索到的上下文)。

    **上下文口径（重要）：** 忠实度判官本应吃「真正进入 prompt 的检索片段」。
    P0 的 Agent Core 只发出 5 种事件，其中不含引用溯源，因此这里用工具返回的
    data 近似。将来给 Core 加上引用溯源时，必须同步复核此处的口径，
    否则 faithfulness 分数的含义会悄悄漂移。
    """
    answer = ""
    retrieved: list[str] = []

    async for event in run_agent_stream(
        AgentTask(question=question),
        AgentDeps(
            model=get_chat_model(),
            tools=tools,
            context=build_context_engine(),
            max_iterations=max_iterations,
        ),
    ):
        if isinstance(event, ToolResultEvent) and event.ok and event.data:
            retrieved.append(_flatten(event.data))
        elif isinstance(event, FinalEvent):
            answer = event.answer

    return answer, "\n\n".join(retrieved)


def _flatten(data: Any) -> str:
    """把工具返回的结构化片段压成判官可读的文本。"""
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or item.get("snippet") or item))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    if isinstance(data, dict):
        return str(data.get("content") or data)
    return str(data)


async def _score(chain, payload: dict) -> float:
    try:
        return float((await chain.ainvoke(payload)).strip())
    except (ValueError, TypeError):
        return 0.0


async def _evaluate_single(query: str, answer: str, context: str) -> dict:
    llm = get_chat_model()

    faith_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "评分0-1：回答是否完全基于给定上下文？1=完全基于上下文。只输出数字。"),
            ("human", "上下文: {context}\n\n回答: {answer}"),
        ])
        | llm
        | StrOutputParser()
    )
    relev_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "评分0-1：回答与问题的相关程度。1=高度相关。只输出数字。"),
            ("human", "问题: {query}\n\n回答: {answer}"),
        ])
        | llm
        | StrOutputParser()
    )

    return {
        "faithfulness": await _score(faith_chain, {"context": context, "answer": answer}),
        "answer_relevancy": await _score(relev_chain, {"query": query, "answer": answer}),
    }


async def run_evaluation(queries: list[str], tools: MCPToolClient) -> list[dict]:
    results = []
    for i, query in enumerate(queries):
        try:
            answer, context = await _run_query(query, tools)
            if i < len(queries) - 1:
                await asyncio.sleep(_QUERY_INTERVAL_SECONDS)
            scores = await _evaluate_single(query, answer, context)
            results.append({
                "query": query, "answer": answer, "context": context,
                "error": None, **scores,
            })
        except Exception as e:  # noqa: BLE001 —— 单条失败不应中断整批评测
            results.append({
                "query": query, "answer": "", "context": "",
                "faithfulness": None, "answer_relevancy": None,
                "error": str(e),
            })
    return results
```

**顺带删掉两处：**

- **`has_docs` 参数**：工具可用性由 MCP client 的 specs 决定、路由由模型决定，该参数已无意义。
- **`RESOURCE_EXHAUSTED` 判断**：那是 Gemini 的错误码。本项目已迁到 DeepSeek（`docker-compose.yml` 里残留的 `GEMINI_API_KEY` 同源），属于迁移未清尾。

`evaluation_service` 也不再直连 `ChatOpenAI`，一律经 `get_chat_model()` 与 Agent Core 同源，测试可注入。

同时改 `backend/app/api/evaluation.py`：

```python
@router.post("/run")
async def run_eval(
    request: EvalRequest, app_request: Request, db: Session = Depends(get_db)
):
    results = await run_evaluation(request.queries, app_request.app.state.mcp_client)
    ...
```

`EvalRequest` 去掉 `has_docs` 字段，`frontend/src/composables/useEvaluation.ts` 同步去掉。

- [ ] **Step 10: 在 `backend/app/main.py` 的 lifespan 中挂载 MCP 工具客户端**

**这一步从原 Task 6（现 Task 8）前移至此** —— Task 7 的评测链路是第一个需要 `app.state.mcp_client` 的消费者，放在 Task 8 会导致 Task 7 结束时应用不可用。

与 Task 2 Step 8 的预热合并后，`main.py` 的 lifespan 终态为：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    if settings.preload_models:
        started = time.perf_counter()
        await asyncio.to_thread(_preload_retrieval)
        logger.info("检索链路预载完成，耗时 %.2fs", time.perf_counter() - started)

    # MCP 工具客户端与 Server 同进程，整个应用生命周期内复用同一个会话
    async with open_tool_client(mcp_server) as tool_client:
        app.state.mcp_client = tool_client
        yield
```

并在文件顶部补 `from app.mcp.client import open_tool_client` 与 `from app.mcp.server import mcp_server`。

- [ ] **Step 11: 验证应用可导入**

```bash
cd backend
../.venv/Scripts/python.exe -c "from app.main import app; print('app imports ok')"
```

Expected: `app imports ok`

注意：`app/api/chat.py` 目前仍 import `run_agent`（已删除），此步会失败。
这是预期的 —— Task 8 会重写它。若想保持每步可导入，可在此步临时把
`chat.py` 的 import 改为 `from app.services.chat_service import stream_agent`。

- [ ] **Step 12: Commit**

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

## Task 8: 真流式 SSE

**Files:**
- Modify: `backend/app/api/chat.py`
- Modify: `frontend/src/composables/useChat.ts`
- Create: `backend/tests/test_streaming.py`

- [ ] **Step 1: 写失败的测试 `backend/tests/test_streaming.py`**

```python
import json

from app.agent.protocol import (
    ErrorEvent,
    FinalEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    to_sse,
)


def test_token_event_maps_to_sse_frame():
    frame = to_sse(TokenEvent(text="检索"))

    assert frame["event"] == "token"
    assert json.loads(frame["data"])["type"] == "token"


def test_frames_can_be_reassembled_into_the_full_answer():
    """前端按顺序拼接 token 事件必须还原出完整答案。"""
    pieces = ["根据", "文档", "内容", "……"]
    reconstructed = "".join(
        json.loads(to_sse(TokenEvent(text=p))["data"])["text"] for p in pieces
    )

    assert reconstructed == "根据文档内容……"


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

Expected: 5 passed（这些测试锁定的是 SSE 编码契约，实现已在 Task 4 完成）

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

- [ ] **Step 4: 确认 MCP 客户端已挂载（Task 7 Step 10 已完成）**

此步原为「在 lifespan 中挂载 MCP 工具客户端」，已前移到 Task 7 Step 10 —— 因为 Task 7 的评测链路是第一个需要 `app.state.mcp_client` 的消费者，若留在这里，Task 7 结束时应用处于不可用状态。

此处只需确认 `main.py` 的 lifespan 里已有 `async with open_tool_client(mcp_server) as tool_client: app.state.mcp_client = tool_client`。同时顺手把 `FastAPI` 的 `version` 从 `"0.1.0"` 改为 `"0.2.0"`（功能已显著变化，版本号值得跟进）；`title` 保持 `"Agentic RAG API"` 不变，它与本仓库一致。

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

## Task 9: 滚动增量摘要

**问题（读码确认）：** `get_conversation_with_context` 只在 `conv.summary` 为空时摘要一次，其后**永不更新**：

```python
if len(messages) > settings.history_window_size:
    if not conv.summary:                                     # ← 只进一次
        old = messages[:-settings.history_window_size]
        conv.summary = _generate_summary(old)
```

窗口为 10 时：第 11 条消息触发摘要，`messages[:-10]` 只摘要了第 1 条；此后再增长，**第 2 条到第 N-10 条既不在摘要里、也不在最近窗口里，静默消失**。

雪上加霜的是 `_generate_summary` 用 `except Exception` 兜底返回 `"历史对话 (N 条消息)"` —— LLM 抖动一次，这个空壳摘要就被永久钉死在那个会话上。

**为什么必须加字段：** 要「把新滑出窗口的那批追加进已有摘要」，就必须知道**上次摘要覆盖到第几条**。消息是追加写的，上次的窗口位置不可推导，这个边界不落库就无法恢复。备选方案（把水位线藏进 `summary` 文本做哨兵）会让字段语义变脏，且 `summary` 本来就经 `get_conversation_with_context` 直接吐给前端，剥离逻辑漏一处就泄漏脏字符串。

**Files:**
- Create: `backend/app/db/migrate.py`
- Modify: `backend/app/models/conversation.py`
- Modify: `backend/app/services/conversation_service.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_summary_rollover.py`

依赖：`settings.context_summary_max_chars`（Task 5 Step 1 已加）与 `Message.id` 排序（Task 3）。

- [ ] **Step 1: 新建 `backend/app/db/migrate.py`**

`Base.metadata.create_all` 只建表、不改表 —— 对已存在的库新增列会**静默不生效**。在引入 alembic 之前，用这个幂等检查兜住少量必需的新列。它同时是将来上 alembic 的种子。

```python
"""轻量幂等列检查。

create_all 只建表、不改表：对已存在的库新增列会静默不生效。
仅登记**必需**的列。这是过渡方案，不是迁移框架 —— 列一多就应当换成 alembic。
"""
from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# (表名, 列名, 列定义)
REQUIRED_COLUMNS: list[tuple[str, str, str]] = [
    ("conversations", "summary_upto_message_id", "INT NULL"),
]


def ensure_columns(engine: Engine) -> None:
    """补齐缺失的列。对已存在的列是 no-op，可重复调用。

    必须在 create_all **之后**调用：新库由 create_all 直接带上这些列（此处 no-op），
    旧库才走 ALTER。
    """
    for table, column, ddl in REQUIRED_COLUMNS:
        inspector = inspect(engine)
        if table not in set(inspector.get_table_names()):
            continue
        if column in {c["name"] for c in inspector.get_columns(table)}:
            continue

        logger.info("补列: %s.%s", table, column)
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
```

注意 `inspect(engine)` 每次循环都重建 —— inspector 会缓存列信息，同一张表补第二列时会读到过期快照。

- [ ] **Step 2: 给 `backend/app/models/conversation.py` 加水位线字段**

```python
    # 摘要已覆盖的消息 id 上界（含）。NULL 表示「尚未记录」，
    # 下次遇到超窗会话时会自愈式地合并全部窗口外消息并落上水位线。
    summary_upto_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
```

并在 import 行补 `Integer`。

- [ ] **Step 3: 在 `backend/app/main.py` 的 lifespan 中调用列检查**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    ensure_columns(engine)          # 必须在 create_all 之后
    ...
```

补 `from app.db.migrate import ensure_columns`。

- [ ] **Step 4: 写失败的测试 `backend/tests/test_summary_rollover.py`**

复用 Task 3 的 `sqlite_session` fixture；假模型让断言可以检查「到底喂了什么」。

```python
"""滚动增量摘要测试。SQLite 内存库 + 假模型，不依赖 MySQL / API key。"""
from datetime import datetime

import pytest

from app.models.conversation import Conversation
from app.models.message import Message
from app.services import conversation_service as svc

WINDOW = 4


class _RecordingModel:
    """假模型：记录 prompt，可被指定失败一次。"""

    prompts: list[str] = []
    fail_next = False

    def invoke(self, prompt):
        type(self).prompts.append(prompt)
        if type(self).fail_next:
            type(self).fail_next = False
            raise RuntimeError("LLM 不可用")

        class _R:
            content = "合并后的摘要"

        return _R()


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    _RecordingModel.prompts = []
    _RecordingModel.fail_next = False
    monkeypatch.setattr(svc, "get_chat_model", lambda: _RecordingModel())
    monkeypatch.setattr(svc.settings, "history_window_size", WINDOW, raising=False)
    monkeypatch.setattr(svc.settings, "context_summary_max_chars", 500, raising=False)


def _seed(session, count):
    conv = Conversation(title="t")
    session.add(conv)
    session.flush()
    base = datetime(2026, 9, 23, 10, 0, 0)
    for i in range(count):
        session.add(Message(
            conversation_id=conv.id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"消息{i}",
            created_at=base,
        ))
    session.flush()
    return conv


def _add(session, conv, role, content):
    session.add(Message(conversation_id=conv.id, role=role, content=content))
    session.flush()


def test_summary_advances_watermark(sqlite_session):
    conv = _seed(sqlite_session, WINDOW + 2)          # 2 条在窗口外
    svc.get_conversation_with_context(sqlite_session, conv.id)

    reloaded = sqlite_session.get(Conversation, conv.id)
    assert reloaded.summary == "合并后的摘要"
    assert reloaded.summary_upto_message_id == 2


def test_only_newly_slid_out_messages_are_sent(sqlite_session):
    """第二次只喂「新滑出窗口」的那批，不重复喂已覆盖的。"""
    conv = _seed(sqlite_session, WINDOW + 2)
    svc.get_conversation_with_context(sqlite_session, conv.id)
    assert "消息0" in _RecordingModel.prompts[0]

    _add(sqlite_session, conv, "user", "消息6")
    _add(sqlite_session, conv, "assistant", "消息7")
    svc.get_conversation_with_context(sqlite_session, conv.id)

    second = _RecordingModel.prompts[1]
    assert "消息2" in second and "消息3" in second      # 新滑出的
    assert "消息0" not in second                        # 已覆盖，不再重复喂


def test_failure_does_not_advance_watermark(sqlite_session):
    """失败时不覆盖摘要、不推进水位线 —— 下个请求会重试（旧实现会永久钉死兜底文案）。"""
    conv = _seed(sqlite_session, WINDOW + 2)
    _RecordingModel.fail_next = True

    svc.get_conversation_with_context(sqlite_session, conv.id)
    reloaded = sqlite_session.get(Conversation, conv.id)
    assert reloaded.summary is None
    assert reloaded.summary_upto_message_id is None

    svc.get_conversation_with_context(sqlite_session, conv.id)      # 重试
    reloaded = sqlite_session.get(Conversation, conv.id)
    assert reloaded.summary == "合并后的摘要"
    assert reloaded.summary_upto_message_id == 2


def test_legacy_conversation_without_watermark_self_heals(sqlite_session):
    """存量会话有摘要但无水位线：一次合并全部窗口外消息并落上水位线。"""
    conv = _seed(sqlite_session, WINDOW + 2)
    conv.summary = "旧的摘要"
    sqlite_session.flush()

    svc.get_conversation_with_context(sqlite_session, conv.id)

    reloaded = sqlite_session.get(Conversation, conv.id)
    assert reloaded.summary_upto_message_id == 2
    assert "旧的摘要" in _RecordingModel.prompts[0]


def test_no_llm_call_when_nothing_new_slid_out(sqlite_session):
    conv = _seed(sqlite_session, WINDOW + 2)
    svc.get_conversation_with_context(sqlite_session, conv.id)
    assert len(_RecordingModel.prompts) == 1

    svc.get_conversation_with_context(sqlite_session, conv.id)

    assert len(_RecordingModel.prompts) == 1            # 稳态下不再调用 LLM
```

- [ ] **Step 5: 运行测试，确认失败**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_summary_rollover.py -v
```

Expected: FAIL —— `AttributeError: ... has no attribute 'summary_upto_message_id'` 或断言 `summary_upto_message_id == 2` 处得到 `None`

- [ ] **Step 6: 改写 `backend/app/services/conversation_service.py` 的摘要部分**

删掉 `_generate_summary`，替换为：

```python
def _roll_summary(
    db: Session, conv: Conversation, messages: list[Message], window: int
) -> None:
    """把新滑出窗口的消息合并进已有摘要。

    旧实现只在 summary 为空时摘要一次，其后永不更新 —— 窗口=10 时，第 11 条触发摘要
    （只摘要第 1 条），之后第 2 条至第 N-10 条既不在摘要里也不在最近窗口里，静默丢失。

    失败时**不推进水位线**，下个请求自然重试。旧实现把兜底文案写进 conv.summary，
    一次抖动就被永久钉死。
    """
    cutoff = messages[-window].id                    # 窗口起点
    upto = conv.summary_upto_message_id
    pending = [
        m for m in messages
        if m.id < cutoff and (upto is None or m.id > upto)
    ]
    if not pending:
        return

    merged = _merge_summary(conv.summary, pending)
    if merged is None:
        return                                       # 交给下个请求重试

    conv.summary = merged
    conv.summary_upto_message_id = pending[-1].id
    db.flush()


def _merge_summary(previous: str | None, pending: list[Message]) -> str | None:
    """合并摘要。失败返回 None（调用方据此不推进水位线）。"""
    transcript = "\n".join(
        f"{'用户' if m.role == 'user' else '助手'}: {m.content}" for m in pending
    )
    prior = f"已有摘要：\n{previous}\n\n" if previous else ""
    prompt = (
        f"{prior}以下是新增的对话片段：\n{transcript}\n\n"
        "请输出一份更新后的对话摘要，保留关键事实、决策与用户意图，"
        f"总长度不超过 {settings.context_summary_max_chars} 字。只输出摘要本身。"
    )
    try:
        return get_chat_model().invoke(prompt).content
    except Exception:  # noqa: BLE001 —— 失败不缓存，交由下个请求重试
        logger.warning("摘要合并失败，本次不推进水位线", exc_info=True)
        return None
```

`get_conversation_with_context` 里的判定改为：

```python
    messages = conv.messages                         # 按 Message.id 排序（Task 3）
    window = settings.history_window_size

    if len(messages) > window:
        _roll_summary(db, conv, messages, window)
        recent = messages[-window:]
    else:
        recent = messages
```

补 `import logging`、`logger = logging.getLogger(__name__)`、`from app.agent.llm import get_chat_model`；删掉不再需要的 `from langchain_openai import ChatOpenAI`。

- [ ] **Step 7: 运行测试，确认通过**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_summary_rollover.py -v
```

Expected: 5 passed

- [ ] **Step 8: 在真实库上验证列已补上**

```bash
cd backend
../.venv/Scripts/python.exe -c "
from app.db.session import engine
from app.db.migrate import ensure_columns
ensure_columns(engine); ensure_columns(engine)   # 幂等：连跑两次不报错
print('ok')
"
mysql -u root -p rag_agent -e "SHOW COLUMNS FROM conversations LIKE 'summary_upto%';"
```

Expected: 打印 `ok`，且 SQL 显示 `summary_upto_message_id int YES NULL`

- [ ] **Step 9: Commit**

```bash
git add backend/app/db/migrate.py backend/app/models/conversation.py backend/app/services/conversation_service.py backend/app/main.py backend/tests/test_summary_rollover.py
git commit -m "fix: roll the conversation summary forward instead of freezing it

The summary was generated once, gated on 'if not conv.summary'. With a
window of 10, message 11 triggered a summary covering only message 1;
every message after that until N-10 then lived neither in the summary
nor in the recent window and was silently dropped from context.

A watermark column is required to know which messages are already
covered — append-only messages make the previous window position
underivable. Since create_all never ALTERs an existing table, this adds
a small idempotent column check rather than pretending the column will
appear on its own.

_summary no longer swallows failures into a cached placeholder string:
it returns None, and the watermark only advances on success."
```

---

## Task 10: 评测链路收尾与回归

Task 7 Step 9 已把评测改成真异步（`ainvoke` + `asyncio.sleep`）。本任务负责清理残留，并用一条**能真失败**的测试把「不阻塞事件循环」锁住。

**Files:**
- Modify: `backend/app/models/evaluation.py`
- Modify: `backend/app/api/evaluation.py`
- Modify: `frontend/src/composables/useEvaluation.ts`
- Create: `backend/tests/test_eval_nonblocking.py`

- [ ] **Step 1: 移除死列与 Gemini 迁移残留**

**1a. `backend/app/models/evaluation.py` 的两个死列。** `context_precision` 与 `context_recall` 从未被任何代码写入或读取（`api/evaluation.py` 的 `EvaluationModel(...)` 构造里没有它们，`/history` 也不返回它们）。它们让模型暗示存在四层指标，实际只有两层。从 ORM 模型里删掉这两行即可 —— 库里已有的列留着无害，删列是破坏性操作，不值得为整洁冒这个险。

**1b. `backend/app/services/evaluation_service.py` 的 `RESOURCE_EXHAUSTED`。** 那是 Gemini 的错误码，Task 7 Step 9 重写该文件时一并删除。

**1c. `docker-compose.yml` 的 `GEMINI_API_KEY`。** compose 给后端传的是 `GEMINI_API_KEY`，而 `config.py` 读的是 `DEEPSEEK_API_KEY` —— **Docker 部署会拿不到任何 LLM key**。同一处还漏传了 `SERPAPI_KEY`，导致 compose 环境下联网搜索恒为关闭。改为：

```yaml
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-https://api.deepseek.com}
      - SERPAPI_KEY=${SERPAPI_KEY:-}
      - MYSQL_HOST=mysql
      - MYSQL_PORT=3306
      - MYSQL_USER=root
      - MYSQL_PASSWORD=${MYSQL_ROOT_PASSWORD:-ragagent123}
      - MYSQL_DATABASE=rag_agent
```

（MySQL root 密码硬编码默认值 `ragagent123` 属 backlog，本轮不动。）

- [ ] **Step 2: 从 `EvalRequest` 与前端移除 `has_docs`**

`backend/app/api/evaluation.py` 的 `EvalRequest` 去掉 `has_docs: bool = False`；`frontend/src/composables/useEvaluation.ts` 的请求体同步去掉该字段。

理由：工具可用性由 MCP client 的 specs 决定、路由由模型决定，客户端传 `has_docs` 已不产生任何效果。留着它就是留一个「参数名与行为不符」的东西 —— 正是本计划要消灭的那类问题。

- [ ] **Step 3: 写回归测试 `backend/tests/test_eval_nonblocking.py`**

这条测试必须**能真失败**：把 `run_evaluation` 的调用改回同步形式，它会立刻变红。

```python
"""评测链路不阻塞事件循环的回归测试。

旧实现（async 端点直接调用同步 run_evaluation）会让并发请求一直等到整批评测结束。
"""
import asyncio

import httpx
import pytest

from app.api import evaluation as eval_api
from app.db.session import get_db
from app.main import app
from app.services import evaluation_service as eval_svc


class _StubSession:
    """评测端点只用到 add/commit。"""

    def add(self, obj):
        pass

    def commit(self):
        pass


@pytest.mark.anyio
async def test_evaluation_does_not_block_other_requests(monkeypatch):
    started = asyncio.Event()

    async def _slow_run_evaluation(queries, tools):
        started.set()
        await asyncio.sleep(0.5)                  # 模拟慢评测
        return [
            {
                "query": q, "answer": "", "context": "",
                "faithfulness": 0.0, "answer_relevancy": 0.0, "error": None,
            }
            for q in queries
        ]

    # 注意：api 层是 `from ... import run_evaluation`，名字已绑定到 eval_api 的命名空间，
    # 只 patch eval_svc 不生效，两处都要打补丁。
    monkeypatch.setattr(eval_svc, "run_evaluation", _slow_run_evaluation)
    monkeypatch.setattr(eval_api, "run_evaluation", _slow_run_evaluation)

    # ASGITransport 不跑 lifespan，因此 app.state.mcp_client 不存在，需要手工补上
    monkeypatch.setattr(app.state, "mcp_client", None, raising=False)
    app.dependency_overrides[get_db] = lambda: _StubSession()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            eval_task = asyncio.create_task(
                client.post("/api/evaluation/run", json={"queries": ["q1"]})
            )
            await asyncio.wait_for(started.wait(), timeout=2)

            # 评测还在跑，health 必须立刻回来
            health = await asyncio.wait_for(client.get("/api/health"), timeout=0.3)

            assert health.status_code == 200
            assert not eval_task.done()           # 评测尚未结束 —— health 却已经返回了
            await eval_task
    finally:
        app.dependency_overrides.pop(get_db, None)
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_eval_nonblocking.py -v
```

Expected: 1 passed

**验证它确实是一条有效测试**（不可跳过）：临时把 `api/evaluation.py` 改回

```python
    results = asyncio.run(run_evaluation(...))   # 或任何同步等待形式
```

再跑一次，Expected: FAIL —— 若仍然通过，说明这条测试没测到东西。

- [ ] **Step 5: 手工确认并发不再互相影响**

```bash
# 终端 A：发起一次评测（会串行跑，含 3s 间隔）
curl -X POST http://localhost:8000/api/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"queries":["劳动合同如何解除","试用期最长多久"]}'

# 终端 B：评测进行中，反复打 health 并计时
time curl -s http://localhost:8000/api/health
```

Expected: health 在几十毫秒内返回，且**不随评测进行而变慢**。改造前它会被卡住 30~60s。

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/evaluation.py backend/app/models/evaluation.py frontend/src/composables/useEvaluation.ts backend/tests/test_eval_nonblocking.py
git commit -m "test: lock in that evaluation no longer blocks the event loop

Adds a regression test that fails if run_evaluation is awaited
synchronously again.

Also drops the two Evaluation columns that were never written or read,
drops the now-meaningless has_docs request field, and removes the
RESOURCE_EXHAUSTED branch left over from the Gemini era."
```

---

## 完成标准

全部 10 个任务完成后，以下命令必须全绿，且**不需要 API key、不需要 MySQL、不下载任何模型**：

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/ -v
```

以及手工验收：

- [ ] `curl -N` 能看到 token 逐条到达（Task 8 Step 7）
- [ ] MCP 工具被真实列出：`open_tool_client(mcp_server)` 返回 3 个工具
- [ ] 检索链路二次调用接近 0 耗时（Task 2 Step 11）
- [ ] 评测进行中 `/api/health` 依然秒回（Task 10 Step 5）
- [ ] 重载一个已有会话，`messages` 里 user 在 assistant 之前（Task 3 Step 6）
- [ ] 代码中已无 `tool_registry` / `agent_graph` / `sleep(0.02)` / `TOOL_SCHEMAS` 残留：

```bash
cd backend
grep -rn "tool_registry\|agent_graph\|sleep(0.02)\|TOOL_SCHEMAS" app/ tests/ || echo "干净"
```
Expected: `干净`

- [ ] 代码与 compose 中已无 Gemini 残留：

```bash
cd backend
grep -rn "RESOURCE_EXHAUSTED\|GEMINI" app/ tests/ ../docker-compose.yml || echo "干净"
```
Expected: `干净`

- [ ] README 的三条宣称已与实现对得上：「MCP 协议」「SSE 流式输出」此时**已为真**；reranker 一条已按 Task 2 Step 9 改成「已实现、未启用」的实话。`项目亮点` 一节待 P1 完成后重写

---

## Self-Review 记录

**覆盖检查：** 本计划覆盖 P0 的全部 6 项技术任务（依赖清理、Agent 协议、上下文引擎、MCP 工具层、Agent Core、真流式），外加 4 项缺陷修复。

**任务集相对最初六任务版本的六处调整：**

1. **Agent 协议提前到 P0（现 Task 4）**。最初排在后续阶段，但 MCP、Agent Core、流式三者都要发事件，协议必须先有。
2. **上下文引擎从第 6 位提前到第 3 位（现 Task 5）**。原顺序下 Agent Core 要调用尚不存在的上下文引擎，构成循环依赖。
3. **新增 Task 2「检索链路单例化」**。Task 7 的 Agent Core 会用 `asyncio.gather` 并发调用工具，而 `get_embeddings()` 每次调用都重建 BGE-M3（实测 9.82s 冷 / 2.29s 温）。不改则并发下会同时加载 N 份约 2GB 的模型 —— 这是 Task 6/7 的前置条件，不是可选优化。同任务顺带修了向量侧召回被截断的问题：`get_retriever()` 硬编码 `k=settings.top_k`，使 `hybrid_search` 的 `vector_top_k` 参数失效、`settings.hybrid_top_k` 成为死配置，RRF 实际只融合 4 路向量结果。
4. **新增 Task 3「消息时序确定性」**。`messages.created_at` 秒精度 + `save_message` 不 flush，导致同一次问答的两条消息时间戳完全相同（实测 id 35/36 均为 `2026-06-03 17:58:52`），而关系正按 `created_at` 排序。它同时影响喂给摘要的 transcript 顺序，因此也是 Task 9 的前置。
5. **新增 Task 9「滚动增量摘要」**。原实现只在 `summary` 为空时摘要一次，之后窗口外的中间消息静默丢失。修复需要记录摘要的覆盖边界，因而新增 `summary_upto_message_id` 字段，并附带一个幂等列检查（`app/db/migrate.py`）—— 因为 `create_all` 不会给已有表加列。
6. **新增 Task 10「评测链路收尾与回归」**。最初的「修改」清单遗漏了 `app/services/evaluation_service.py`（它 import 了将被删除的 `agent_graph`），且该模块是 `async` 端点里直接调用的同步函数、内含 `time.sleep(3)`，3 条 query 即冻结整个服务器 30~60s。相应地，「在 lifespan 挂载 MCP 客户端」一步前移到了 Task 7。

第 3–6 项均为 2026-09-23 的缺陷排查所发现。

**范围说明：** 本计划只修本仓库（Agentic RAG 智能问答平台）自身的问题。仓库中曾存在一份描述「端云协同的直播主播经营 Agent 平台」的 PRD，那是另一个项目的规划 —— 与本仓库的实现、数据、验收标准均无关，且其场景在本仓库代码里从未落地（`backend/app`、`frontend/src`、README 里零处提及），已从本仓库移除。计划原先从该 PRD 引用的技术前提（事件协议、上下文分层预算）现已自包含。

**类型一致性：** `AgentEvent` / `ToolSpec` / `ToolResult` / `AgentTask` / `AgentDeps` / `ContextEngine.build()` 的签名在 Task 4–8 中一致引用，无重命名漂移。

**已知未决项（实施时需现场确认，已在对应步骤标注）：**
- Task 6 Step 12：`mcp_server.list_tools()` 的确切返回结构未实测，`app/api/tools.py` 的改写需以实际签名为准
- Task 6 Step 11：`InMemoryTransport` 位于 `mcp.client._memory` 私有路径，版本升级可能失效，已给出替代方案

以下为 2026-09-23 排查新增：

- **reranker 已实现但未接入查询链路**（Task 2 Step 6 / Step 9）。README 措辞已修正为「已实现、未启用」。接入留到后续，与检索指标（Hit Rate / NDCG）一起做 —— 在 CPU 上精排 10 个候选约需 1~3 秒，会明显拖慢首 token 延迟，需要先有数据证明它值这个代价
- **评测的 `context` 口径是近似的**（Task 7 Step 9）：用工具返回的 data 代替「真正进入 prompt 的检索片段」，因为 P0 的 Core 尚不发出 citation 事件。今后改动 Core 的引用机制时，必须同步复核此处，否则 faithfulness 分数的含义会悄悄漂移
- **评测仍是串行 + 3s 间隔**（Task 7 Step 9）。并发执行 + 信号量限流属于评测体系的后续优化
- **仍无认证与用户隔离**：`/api/conversations` 返回全库会话，任何人都能消耗 DeepSeek 额度。属 backlog
- **依赖仍全部是 `>=` 下限**，仅 `mcp==2.2.0` 被钉死。属 backlog
- **`docker-compose.yml` 的 MySQL root 密码硬编码默认值 `ragagent123`**（Task 10 Step 1c 只修了 LLM key 与 SerpAPI key 的透传）。属 backlog
- **`.env` 内的 DeepSeek key 建议轮换**：它从未被提交（`.gitignore` 已覆盖），但在排查过程中被读取过
- **「BM25 索引是可重建的缓存」目前是空头支票**（Task 2 交付时发现）：仓库里**没有**从 Chroma 重建 BM25 的路径。Task 2 为此加了护栏（加载失败时 `add()` 拒绝落盘，避免用空基础覆盖唯一副本），但真正的修复是写出重建路径 —— 它天然属于跨存储一致性那件事
- **Chroma 与 BM25 之间没有事务**：`vector_store.add_documents()` 先写 Chroma，随后的 BM25 写入若失败，就留下「Chroma 有、BM25 无」的分叉，而此时 `DocumentModel` 行尚未插入，没有可据以对账的记录。Task 7 起工具会被并发调用，这个窗口会变成常态
- **`os.replace` 在 Win32 上遇到被占用的目标文件会抛 `PermissionError`**（实测 150/150），此时旧索引完好、只是本次落盘失败。单进程不可达（`get_bm25_index` 在同一把锁内 load）；跑 `uvicorn --workers N` 共享 `data/` 时才需要重试与退避
- **`rank-bm25` 未钉版本**（`>=0.2.2`，实装 0.2.2）：pickle 里嵌着活的 `BM25Okapi`，升级会让磁盘索引失效（Task 2 已让这条路径降级而不是崩掉）；测试里对 BM25 分数的精确值断言也依赖它的 idf 数学实现
- **重复上传同一份文件会重复入库**：Chroma 与 BM25 双向都会再加一遍，无去重
- **`langchain_community.vectorstores.Chroma` 已是弃用导入**（提示迁移到 `langchain-chroma`），Task 2 动了那处构造函数但未迁移
- **BM25 小语料陷阱**（写测试时反复踩到，已在测试注释里记）：`idf = log((N-df+0.5)/(df+0.5))`，只有 `df < N/2` 才为正；`df == N` 或 `N == 1` 时 idf 为负，被 epsilon 地板压成非正，`search()` 里 `score > 0` 的过滤会把结果全丢掉 —— 是 BM25 的数学，不是索引坏了
- **降级状态下每 30s 会打一次完整堆栈日志**（约 2880 条/天，且来自请求路径），每次重试也真读一遍那 2MB pickle。可改为「首次记 exception、其后记一行 warning」+ 指数退避
- **护栏触发时 Chroma 已经被写过**：上传会 500，而 Chroma 里已有 chunk、`DocumentModel` 行却没有。失败得响亮仍然是对的（跳过落盘会留下**错误**的内存索引，不只是没持久化），但在写 Chroma **之前**先探测一下 BM25 索引状态不花什么代价
- **pid 后缀的临时文件会留下孤儿**：`pickle.dump` 中途失败（正是本次引用的 MemoryError 场景）会留下一个与索引同尺寸的 `.tmp`，而固定名字原本会被下次 save 复用。可在启动时清扫 `data/*.tmp`
- **降级状态下每次 `get_bm25_index()` 都要拿 `_bm25_lock`**（即便节流命中直接返回），真正重试期间锁会持有整个 `pickle.load`，并发检索会串行排队。pickle 小可忽略，大则不然
- `test_bm25_load_returns_false_when_a_pickled_class_is_gone` 是**特征化测试**而非先红后绿的回归测试（它在改造前后都绿）；它的价值在于断言比原来更强（`_pair[1] is not None` 能抓到「赋了一半」），但别把它当作覆盖了窄异常元组那个洞的证据
