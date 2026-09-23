# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 常用命令

项目在仓库根目录只有一个虚拟环境：`.venv/Scripts/python.exe`（Windows，Python 3.11）。后端命令必须**在 `backend/` 目录下**执行，否则 `app` 包无法解析 —— 仓库没有 `pytest.ini`/`pyproject.toml`，pytest 依赖 rootdir 插入机制来找到包。

```bash
# 后端 —— 开发服务器（需要 MySQL 8 在运行；lifespan 里会调用 Base.metadata.create_all）
cd backend && ../.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000

# 后端 —— 依赖（requirements-dev.txt 即 `-r requirements.txt` 加 pytest）
cd backend && ../.venv/Scripts/python.exe -m pip install -r requirements-dev.txt

# 测试
cd backend && ../.venv/Scripts/python.exe -m pytest tests/ -v
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_hybrid_search.py::test_rrf_empty_inputs -v

# 前端 —— 开发服务器 :5173，把 /api 代理到 http://localhost:8000
cd frontend && npm run dev

# 前端 —— 构建会先跑 `vue-tsc -b`，所以类型错误会直接阻断构建
cd frontend && npm run build

# 全栈
docker compose up --build
```

测试用例的依赖情况 —— 只有 `test_hybrid_search.py` 是完全自包含的：

- `test_agent.py` 会调用真实的 DeepSeek API（需要 `DEEPSEEK_API_KEY` 与网络）。
- `test_reranker.py` 首次运行会下载/加载 BGE-Reranker（约 2GB）。
- `test_chat_api.py` 走的是落库的对话链路，因此需要 MySQL。
- 仅收集用例就要约 35 秒，因为导入 `app.main` 会加载 LangChain/Chroma。

前端没有 lint 脚本，也没有单元测试运行器。`playwright` 是 devDependency，用于临时驱动浏览器做视觉验证，产物落在 `screenshots/`。

## 架构

Vue 3 单页应用 → FastAPI → LangGraph Agent → 混合检索 → ChromaDB + MySQL。LLM 走 DeepSeek（OpenAI 兼容 API）；Embedding 与 Reranker 是完全本地的 BGE 模型（CPU 推理）。

### 请求链路

`app/api/chat.py` 的 `POST /api/chat/stream` → `app/services/chat_service.py` 的 `run_agent()` → `app/agent/graph.py`（导入时即编译为模块级单例 `agent_graph`）→ `app/tools/registry.py` → `app/rag/hybrid_search.py`。

图是固定拓扑：`router`（由 LLM 分类到 `web`/`rag`/`direct`）→ `retrieve` | `web_search` → `generate`。`AgentState.iteration` 会被 router 自增，但没有任何地方读取它 —— 这里没有循环。

### 并发模型

`chat.py` 是唯一的异步入口。所有阻塞调用都通过 `loop.run_in_executor(None, ...)` 挪出事件循环 —— 尤其是 `get_conversation_with_context`（它会调用 LLM 压缩历史）和 `run_agent`（LangGraph 是同步的）。SQLAlchemy session 是同步的，在同一个生成器内共享；最后由调用方 `db.commit()`，异常时回滚。

### 检索栈

`rag/loader.py`（PyPDF/TextLoader + `RecursiveCharacterTextSplitter`，800/120）把内容喂给**两个彼此独立、必须同步更新的存储**：

- `rag/vector_store.py` —— Chroma 持久化在 `data/chroma`，BGE-M3（1024 维、归一化、CPU）。
- `rag/bm25_index.py` —— pickle 文件 `data/bm25_index.pkl`，内含 `{documents, index}`，分词器对中文做了处理。

`hybrid_search()` 用 RRF（k=60）融合两路结果，以 `page_content[:200]` 作为去重键。两路各召回 `settings.hybrid_top_k`（10）个候选，最终返回 `settings.top_k`（4）个。**`rag/reranker.py` 已实现且有单元测试，但没有接入查询链路** —— `hybrid_search` 的结果直接进 prompt。README 已按实情修正为「已实现、未启用」，并注明接入需先付 1~3 秒 CPU 精排的代价。

检索链路（embeddings、Chroma 句柄、BM25 索引、reranker）都是**进程内单例**，用显式双检锁而非 `lru_cache`（后者在 cache miss 时允许多线程同时进入被包装函数，对 2GB 的 BGE-M3 等于偶发双份常驻内存）。启动时 lifespan 会经 `asyncio.to_thread` 预热，受 `settings.preload_models` 控制；`tests/conftest.py` 有一个 autouse fixture 把它关掉，避免测试里加载真实模型。

两个建索引的调用点（`services/knowledge_service.process_document` 与 `tools/document_parser.parse_document`）都走 `get_bm25_index().add(chunks)`，BM25 这一侧已经不会再分叉。但 **Chroma 与 BM25 之间仍无事务**：`vector_store.add_documents()` 先写 Chroma，随后 BM25 写入若失败，就留下「Chroma 有、BM25 无」的不一致，而此时 `DocumentModel` 行尚未插入，没有可据以对账的记录。跨存储原子性是 Task 6/7 的待办。

BM25 索引加载失败时**降级而不是崩溃**，但要理解这套语义才能安全地碰它：`load()` 宽捕所有异常并返回 `False`（pickle 里嵌着活的 `BM25Okapi`/`Document`，rank_bm25 升级或类改名会抛 `AttributeError`，窄元组接不住）；实例记住 `_load_failed`，`get_bm25_index()` 每 30s 重试一次，`save()` 在降级状态下**拒绝落盘** —— 磁盘上那份读不出来的索引是唯一副本，用内存里的空索引覆盖它会把 2196 篇变成新上传的几篇。恢复手段：修好或删除该文件（删除后重试视其为全新库），或 `save(force=True)`（调用方须已重建出完整集合，它会清掉标志）。注意**「索引是可重建的缓存」目前还不是事实** —— 仓库里没有从 Chroma 重建 BM25 的路径。

### 其他状态

- `config.py` 暴露一个 `settings` 单例（pydantic-settings，读取**仓库根目录**的 `.env`）。`embed_model_path` / `reranker_model_path` 会优先使用 `data/models/` 下的本地副本，找不到才回退到 HuggingFace 模型 id。
- 历史压缩是一次性的：`conversation_service.get_conversation_with_context` 只在 `Conversation.summary` 为空时，对超出 `settings.history_window_size` 的消息做摘要，然后永久缓存。
- `api/chat.py` 不信任客户端传来的 `has_docs`，而是用 `has_persisted_index()`（判断 `data/chroma/chroma.sqlite3` 是否存在）自动探测 —— 这才是真正决定能否走 `rag` 路由的条件。

### 前端结构

状态与 I/O 被刻意分开：Pinia store（`stores/chat.ts`、`stores/knowledge.ts`，setup 风格）**只存状态**，所有请求都写在 composable 里（`composables/useChat.ts`、`useConversations.ts`、`useKnowledge.ts`、`useEvaluation.ts`）。这里没有统一的 API 客户端 —— 全部用裸 `fetch` 硬编码相对路径 `/api/...`（`axios` 是依赖但没被使用）。base URL 只存在于 `vite.config.ts`（开发）和 `nginx.conf`（生产）两处。

`useChat.ts` 用 `fetch` + `ReadableStream` reader 消费 SSE（不是 `EventSource`，因为请求是 POST），并且按 **JSON 载荷的形状分发，而不是按 SSE 的 `event:` 名**。

GSAP 动画统一使用同一套写法：模块作用域的 `let ctx`，`onMounted` 里 `ctx = gsap.context(() => {...}, scopeEl)`，`onUnmounted` 里 `ctx?.revert()`。`composables/useGSAP.ts` 的 `initGSAP()` 在 `main.ts` 里调用一次，负责全局安装 `prefers-reduced-motion` 处理。

样式是 Tailwind v4 的 CSS-first 方式 —— `src/styles/main.css` 首行是 `@import "tailwindcss"`，**没有 `tailwind.config.js`**。设计 token 是手写在 `:root` 里的 CSS 变量（`--paper`、`--ink`、`--clay`、`--r-md` 等），通过 `style="color: var(--ink)"` 使用，并与真正的 Tailwind 工具类混用。

## 当前工作：P0 agent-core 重构（分支 `feat/p0-agent-core`）

仓库正处于重构中途。动手改 `backend/app/agent/`、`backend/app/tools/` 或 `backend/app/api/chat.py` 之前，请先读这两份文档：

- `docs/superpowers/specs/2026-09-10-live-agent-platform-design.md` —— PRD/设计的事实来源（目标架构、协议、P0/P1/P2 计划）。
- `docs/superpowers/plans/2026-09-10-p0-agent-core-foundation.md` —— 逐任务的实施计划，为 `superpowers:subagent-driven-development` 编写。

该计划会**删除** `agent/graph.py`、`agent/nodes.py`、`tools/registry.py`、`tools/schemas.py`，并**新建**真正的 ReAct 循环（`agent/core.py`）、真正的 MCP 工具层（`app/mcp/`，`mcp==2.2.0`，同进程 `InMemoryTransport`）、分层上下文引擎（`app/context/`），以及与未来 TypeScript 端侧 runtime 共享的 `AgentEvent` 协议（`agent/protocol.py`）。目前只完成了 Task 1（清理依赖）—— `backend/requirements.txt` / `requirements-dev.txt` 的未提交改动就是这一步。计划里记录了几条已实测验证、但凭记忆很容易写错的硬事实（MCP 类是 `MCPServer` 而非 `FastMCP`；MCP 会向客户端隐藏工具异常细节，所以工具必须返回 `{ok, data, error}` 而不能抛异常）。

### README 与实现不符之处

这些正是 P0 计划要消灭的「虚假宣称」—— 不要把 README 当作规格说明：

- README 宣称使用 MCP 协议；但 `tools/registry.py` 只是一个普通 dict 加手写 schema，`app/` 里没有任何地方 import `mcp`。
- README 宣称 SSE 流式输出；但 `chat.py` 会等完整答案生成后，再按 4 个字符切片并 `await asyncio.sleep(0.02)` 伪造成流式（`chat.py:75-78`）。
- `docker-compose.yml` 给后端传的是 `GEMINI_API_KEY`，而 `config.py` 读的是 `DEEPSEEK_API_KEY` —— 用 Docker 部署会拿不到任何 LLM key。compose 也从未透传 `SERPAPI_KEY`，所以那里的联网搜索始终是关闭的。

（README 关于 reranker 的那处宣称已由 Task 2 修正，不再是言行不符项。上面两条要等 Task 6 和 Task 8 落地后才消。）

## 约定

- 后端是同步 SQLAlchemy 2.0（`select()` 风格），用 `get_db()` 作为依赖；ORM 模型在 `main.py` 里被导入以产生副作用，好让 `create_all` 能看到它们。
- 后端的注释/文档字符串与所有文档都是中文；前端代码注释是英文，但所有 UI 文案是中文。
- 测试 fixture 用中文劳动法/PDF 文档问答作为示例数据（`tests/conftest.py` 的 `sample_documents`）。
- Vue 组件是单词 PascalCase，放在 `components/{chat,knowledge,evaluation,layout}/` 下；页面是 `views/` 下的 `*View.vue`。
