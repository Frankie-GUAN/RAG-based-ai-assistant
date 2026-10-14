# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 常用命令

项目在仓库根目录只有一个虚拟环境：`.venv/Scripts/python.exe`（Windows，**Python 3.13**）。后端命令必须**在 `backend/` 目录下**执行，否则 `app` 包无法解析 —— `backend/pytest.ini` 把 `testpaths` 设为 `tests`，并注册了 `requires_model` 标记。

```bash
# 后端 —— 开发服务器（需要 MySQL 8 在运行；lifespan 里会调用 Base.metadata.create_all）
cd backend && ../.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000

# 后端 —— 依赖（requirements-dev.txt 即 `-r requirements.txt` 加 pytest）
cd backend && ../.venv/Scripts/python.exe -m pip install -r requirements-dev.txt

# 测试（默认排除 requires_model —— 见下方「测试用例的依赖情况」）
cd backend && ../.venv/Scripts/python.exe -m pytest tests/ -v
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_hybrid_search.py::test_rrf_empty_inputs -v

# 需要下载大模型的用例**必须显式指定标记**，否则跑 0 个用例：
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_reranker.py -m requires_model -v

# 前端 —— 开发服务器 :5173，把 /api 代理到 http://localhost:8000
cd frontend && npm run dev

# 前端 —— 构建会先跑 `vue-tsc -b`，所以类型错误会直接阻断构建
cd frontend && npm run build

# 全栈
docker compose up --build
```

测试用例的依赖情况。`backend/pytest.ini` 的 `addopts` 默认排除标了 `requires_model` 的用例：

- `test_reranker.py` 需要 BGE-Reranker（约 2GB），已标 `requires_model` 而**默认不跑**。要跑必须显式加 `-m requires_model`（命令见上）—— 注意 `pytest tests/test_reranker.py` 会**跑 0 个用例并以退出码 5 结束**，不打印任何提示，脚本无法从退出码区分「通过」与「根本没跑」。仓库没有 CI，所以这个测试实际上无人覆盖。
- `test_chat_api.py` 走的是落库的对话链路，因此需要 MySQL。
- **自包含、裸跑即可**：`test_hybrid_search.py`、`test_bm25_tokenize.py`、`test_reranker_wiring.py`、`test_reranker_availability.py`、`test_citations.py`、`test_streaming.py`、`test_message_order.py`、`test_retrieval_singletons.py`、`test_ingest.py`。
- 仅收集用例就要约 20~35 秒，因为导入 `app.main` 会加载 LangChain/Chroma。

**给新用例的约定**：任何要下载模型或加载 BGE-M3/Chroma 的用例都必须自己打 `@pytest.mark.requires_model`。`-m "not requires_model"` 也会匹配**没有**标记的用例 —— 忘了打标记，它就会在默认跑里执行，这正是 `test_reranker.py` 在 Task 3 之前的行为。

前端没有 lint 脚本，也没有单元测试运行器。`playwright` 是 devDependency，用于临时驱动浏览器做视觉验证，产物落在 `screenshots/`。

## 架构

Vue 3 单页应用 → FastAPI → LangGraph Agent → 混合检索 → ChromaDB + MySQL。LLM 走 DeepSeek（OpenAI 兼容 API）；Embedding 与 Reranker 是完全本地的 BGE 模型（CPU 推理）。

### 请求链路

`app/api/chat.py` 的 `POST /api/chat/stream` → `app/services/chat_service.py` 的 `stream_agent()`
→ `app/agent/core.py` 的 `run_agent_stream()`（ReAct 循环，`model.astream()` 边生成边外发）
→ `app/mcp/client.py` 的 `MCPToolClient` → `app/mcp/server.py` 的工具实现 → `app/rag/`。

事件流是这条链路的骨架：`token` / `tool_call` / `tool_result` / `citation` / `final` / `error`，
定义在 `app/agent/protocol.py`，`api/chat.py` 只做转发与落库。
**2026-10-14 起本链路已跑通**（此前 `chat.py` 调用的 `run_agent` 在 ReAct 改造后已不存在，`app.main` 不可导入）。

工具层是真 MCP：`mcp==2.2.0`，`MCPServer` 与 `MCPToolClient` 同进程经 `InMemoryTransport` 连接，
整个应用生命周期内复用同一会话（挂载在 lifespan 的 `app.state.mcp_client`）。
工具一律返回 `{ok, data, error}` 而不抛异常——MCP 会向客户端隐藏异常细节，抛异常的话
Agent 只会收到一个无法据以决策的哑错误。

### 并发模型

`chat.py` 是唯一的异步入口。会阻塞的调用只有一处：`get_conversation_with_context`（会调 LLM 压缩历史），
用 `loop.run_in_executor`挪出事件循环。Agent 循环本身是 async 的，工具调用在 `app/mcp/server.py`
里用 `asyncio.to_thread` 包装同步实现，因此 BGE-M3 推理不阻塞事件循环，工具之间可并发。
SQLAlchemy session 是同步的，在生成器内共享；最后由 `chat.py` 末尾 `db.commit()`，异常时回滚——
注意用户消息在异常路径上也会一起回滚。

### 检索栈

`rag/loader.py`（PyPDF/TextLoader + `RecursiveCharacterTextSplitter`，800/120）→ `split_documents`
会给每个片段打上稳定的 `chunk_id`（`来源#p页码#序号`），并把 `metadata["source"]` 降级成文件名
（绝对路径会同时泄露给前端和模型）。内容喂给**两个彼此独立、必须同步更新的存储**：

- `rag/vector_store.py` —— Chroma 持久化在 `data/chroma`，BGE-M3（1024 维、归一化、CPU）。
  相似度用 `similarity_search_with_relevance_scores` 取**真实分数**，不再用名次占位。
- `rag/bm25_index.py` —— pickle 文件 `data/bm25_index.pkl`，内含 `{documents, index}`，
  分词器是 `jieba.lcut_for_search`（搜索模式，保留复合词）+ 一张很小的停用词表。

`hybrid_search()` 用 RRF（k=60）融合两路各 `settings.hybrid_top_k`（10）个候选，
**去重键是 `chunk_id`**（不是 `page_content[:200]`——那会让同文档开头相似的相邻分片互相覆盖）。
融合结果交给 `rag/reranker.py` 的 Cross-Encoder 精排出 `settings.top_k`（4）条：
- `settings.enable_reranker`（默认开）关掉即回到纯 RRF 顺序；
- **精排失败自动退回 RRF 顺序**，不让整条检索路径失败；
- **精排只改顺序不改分数** —— CE logit 没有可解释量纲，当成「相关度」展示给模型等于重演当年伪造分数的错误。

CPU 上精排 10 个候选约 1~3 秒；模型缺失时 `get_reranker()` 会进入 10 分钟冷却，
不会让每条查询都去重试一次下载（该失败模式已实测踩过：Windows venv 缺 CA 证书 →
`CERTIFICATE_VERIFY_FAILED` → huggingface_hub 重试 5 次 → 每条查询平白多几秒）。

检索链路（embeddings、Chroma 句柄、BM25 索引、reranker）都是**进程内单例**，
用显式双检锁而非 `lru_cache`（后者在 cache miss 时允许多线程同时进入被包装函数，
对 2GB 的 BGE-M3 等于偶发双份常驻内存）。启动时 lifespan 会经 `asyncio.to_thread` 预热，
受 `settings.preload_models` 控制；`tests/conftest.py` 有一个 autouse fixture 把它关掉。

**换分词器后必须重建 BM25**：`python -m app.rag.bm25_index rebuild`（或 `app.rag.ingest --fresh`
重建全部）。pickle 里存的是**已分词**的语料，改分词器不会让旧文件失效，只会让它变成
「新切法查旧粒度」的静默错误。`rebuild_from_chroma()` 会拒绝空语料（否则会用 0 篇覆盖几千篇），
并以 `save(force=True)` 落盘——那是 `_load_failed` 唯一的自愈入口。

**运维入口**：

```bash
python -m app.rag.ingest --fresh        # 从 data/uploads 重建全部索引（先改名备份旧目录）
python -m app.rag.bm25_index rebuild    # 仅重建 BM25
python -m app.rag.search "查询" --no-rerank   # 命令行检索，核对检索质量用
```

两个建索引的调用点（`services/knowledge_service.process_document` 与
`tools/document_parser.parse_document`）都走 `get_bm25_index().add(chunks)`，BM25 这一侧不会分叉。
但 **Chroma 与 BM25 之间仍无事务**：`vector_store.add_documents()` 先写 Chroma，
随后的 BM25 写入若失败，就留下「Chroma 有、BM25 无」的分叉，而此时 `DocumentModel` 行尚未插入，
没有可据以对账的记录。

BM25 索引加载失败时**降级而不是崩溃**，但要理解这套语义才能安全地碰它：`load()` 宽捕所有异常并返回
`False`（pickle 里嵌着活的 `BM25Okapi`/`Document`，类改名会抛 `AttributeError`）；
实例记住 `_load_failed`，`get_bm25_index()` 每 30s 重试一次，`save()` 在降级状态下**拒绝落盘**。
注意「索引是可重建的缓存」现在**已经是事实**——`rebuild_from_chroma()` 就是那条路径。

### 引用溯源

`CitationEvent` 由 Agent Core 在 `final` 之前外发，来源是 `search_documents` 的返回载荷
（`app/agent/core.py` 的 `_citations_from`，会拆 MCP 信封、只认 `search_documents`、按 `doc_id` 去重、
载荷形状异常时返回空而不抛错）。`api/chat.py` 把它们 JSON 化写进 `Message.source_data`，
并把该轮的 `source_type` 标成 `doc`；`api/conversations.py` 再解析回来，所以**重开对话仍能看到依据**。
前端映射到 `components/chat/SourcePanel.vue`（无 `link` 时渲染成 `<span>` 而不是死链接）。

评估端的忠实度判官读的也是 `CitationEvent` 的片段（此前用工具原始返回近似，口径已对齐）。

### 其他状态

- `config.py` 暴露一个 `settings` 单例（pydantic-settings，读取**仓库根目录**的 `.env`）。
  `embed_model_path` / `reranker_model_path` 会优先使用 `data/models/` 下的本地副本，找不到才回退到
  HuggingFace 模型 id。**目前只有 bge-m3 有本地副本**，精排模型没有，因此精排恒定降级为 RRF 顺序。
- 历史压缩是**一次性的**（`conversation_service.get_conversation_with_context` 只在
  `Conversation.summary` 为空时摘要一次，然后永久缓存）—— 这是已知缺陷，修法见旧 P0 计划 Task 9。
- `api/chat.py` 不信任客户端传来的 `has_docs`，该字段已删除：ReAct 下由模型自主决定是否检索。

### 前端结构

状态与 I/O 被刻意分开：Pinia store（`stores/chat.ts`、`stores/knowledge.ts`，setup 风格）**只存状态**，所有请求都写在 composable 里（`composables/useChat.ts`、`useConversations.ts`、`useKnowledge.ts`、`useEvaluation.ts`）。这里没有统一的 API 客户端 —— 全部用裸 `fetch` 硬编码相对路径 `/api/...`（`axios` 是依赖但没被使用）。base URL 只存在于 `vite.config.ts`（开发）和 `nginx.conf`（生产）两处。

`useChat.ts` 用 `fetch` + `ReadableStream` reader 消费 SSE（不是 `EventSource`，因为请求是 POST），
并且按 **payload 的 `type` 字段**分发，不再靠数据形状猜 —— 旧实现用 `if (data.answer)` 先命中，
把 `done` 帧里的 `route` 吃掉，路由展示因此是死的。`sendMessage` 的收尾放在 `finally`：
任何异常（网络中断、非 2xx、坏帧）都必须复位 `isStreaming`，否则输入框永久禁用。

GSAP 动画统一使用同一套写法：模块作用域的 `let ctx`，`onMounted` 里 `ctx = gsap.context(() => {...}, scopeEl)`，
`onUnmounted` 里 `ctx?.revert()`。`composables/useGSAP.ts` 的 `initGSAP()` 在 `main.ts` 里调用一次，
负责全局安装 `prefers-reduced-motion` 处理。

样式是 Tailwind v4 的 CSS-first 方式 —— `src/styles/main.css` 首行是 `@import "tailwindcss"`，**没有 `tailwind.config.js`**。设计 token 是手写在 `:root` 里的 CSS 变量（`--paper`、`--ink`、`--clay`、`--r-md` 等），通过 `style="color: var(--ink)"` 使用，并与真正的 Tailwind 工具类混用。

## 当前工作：个人知识库方向（2026-10-14 起）

本仓库定位为**个人使用的知识库问答助手**（直播主播经营 Agent 平台已另立项目，其 PRD 已从本仓库移除）。
动手改代码前请先读这两份文档：

- `docs/tech-notes/verified-facts.md` —— **改代码前必查**：本机实测过的技术前提
  （含 pyarrow/torch 加载顺序这个会段错误的坑、MCP 2.2.0 的类名与字段、pickle 兼容性），
  以及带行号的已确认缺陷清单与已知未修项。
- `docs/plans/2026-10-14-personal-kb-revival.md` —— 当前计划。P0（检索质量、Docker）与
  P1（真流式、引用溯源、前端错误处理）已交付；P2（滚动摘要、评估收尾、知识库管理、
  解析增强）待排期。

旧计划 `docs/superpowers/plans/2026-09-10-p0-agent-core-foundation.md` 的 Task 1–7 已交付、
Task 8（真流式）已由新计划吸收；**Task 9（滚动增量摘要）与 Task 10（评测收尾）仍然有效**，
它们的规格是现成的，重做时照着那份计划走即可。

### 环境上的硬前提（踩过才知道）

venv 是 **Python 3.13.2**（此前本文件写的是 3.11，已订正；Docker 镜像已对齐）。
依赖已在 `requirements.txt` 全部钉死 —— 此前全是 `>=` 下限，两天内因此出了两次事故：
chromadb 0.5.x → 1.5.9 让旧持久化索引读不出来（903 条向量形同消失且不报错）；
pyarrow 24 在 torch 之后加载会让进程崩在 `create_module`（exit 139，pytest 只显示
`Segmentation fault`）。`app/__init__.py` 顶部那句 `import pyarrow` 是针对后者的权宜之计，
升级依赖时**先跑测试再决定是否删掉它**。

`data/chroma.bak.20261014-163233/` 是 chromadb 0.5.x 写的旧格式索引，当前版本读不了，
但它是唯一一份旧格式副本，确认不再需要前不要删。

### 测试情况

`pytest tests/ -q` → 112 passed / 1 deselected，约 25~65 秒（收集阶段就要加载 chroma + torch）。
`test_agent.py`（调真实 DeepSeek）已删除。`test_reranker.py` 需要 2GB 模型，默认不跑。
仓库**没有 CI**，测试要手动执行。

## 约定

- 后端是同步 SQLAlchemy 2.0（`select()` 风格），用 `get_db()` 作为依赖；ORM 模型在 `main.py` 里被导入以产生副作用，好让 `create_all` 能看到它们。
- 后端的注释/文档字符串与所有文档都是中文；前端代码注释是英文，但所有 UI 文案是中文。
- 测试 fixture 用中文劳动法/PDF 文档问答作为示例数据（`tests/conftest.py` 的 `sample_documents`）。
- Vue 组件是单词 PascalCase，放在 `components/{chat,knowledge,evaluation,layout}/` 下；页面是 `views/` 下的 `*View.vue`。
