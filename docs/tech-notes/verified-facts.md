# 已验证技术事实（Verified Facts）

> 本文件保存**在本机实际跑通过**的技术结论与已确认缺陷。改代码前先查这里，不要凭记忆。
> 最后更新：2026-10-14（检索质量 + 流式 + 溯源 + 环境修复之后）

---

## 1. 环境事实

| 事实 | 结论 |
|---|---|
| venv 位置 | 仓库根 `.venv/Scripts/python.exe`（Windows）。后端命令必须在 `backend/` 下执行 |
| **Python 版本** | **3.13.2**（此前 CLAUDE.md 记的是 3.11，已订正）。Docker 镜像已对齐为 `python:3.13-slim` |
| 依赖版本 | 已在 `requirements.txt` 全部钉死（`==`）。2026-10-14 之前全是 `>=`，两天内因此出了两次事故（见 §5） |
| pytest | 已装。默认 `addopts` 排除 `requires_model` 标记的用例（见 pytest.ini） |
| anyio | 异步测试用 `@pytest.mark.anyio` + `conftest.py` 的 module 级 `anyio_backend` fixture，无需 pytest-asyncio |
| 测试规模 | `pytest tests/ -q` → **144 passed, 1 deselected**，约 20~30 秒（收集阶段要加载 chroma/torch） |
| 模型 | BGE-M3 本地在 `data/models/BAAI/bge-m3`；**精排模型本地没有**（下载失败，见 §5.3），检索自动退回 RRF 顺序 |
| 未安装的依赖 | `google-search-results` 没装 → `search_web` 一调就失败。要联网搜索先 `pip install google-search-results==2.4.2` |
| 测试分级 | 自包含可裸跑：test_hybrid_search / test_bm25_tokenize / test_reranker_wiring / test_reranker_availability / test_citations / test_message_order / test_retrieval_singletons / test_ingest；需要 MySQL：test_chat_api；需要真实 key：test_agent（已删）；需要 2GB 模型：test_reranker（`-m requires_model`） |

## 2. ⚠️ pyarrow 与 torch 的加载顺序（会段错误）

**症状**：`import app.main` 或跑 pytest 时进程直接崩，`exit 139`，
只有一行 `Segmentation fault`；`python -c "import app.main"` 毫无输出。
**不是 Python 异常**，try/except 接不住。

**根因**：`sentence_transformers` 先加载 torch，再经 `datasets` 导入 `pyarrow.dataset`。
torch 的原生内存已映射之后，pyarrow 24.0.0 的原生扩展 `create_module` 失败（Windows access violation）。

**处置**：`backend/app/__init__.py` 在任何业务 import 之前先 `import pyarrow`。
**这是权宜之计**，根治要靠锁版本（已做）与必要时重建 venv。验证：

```bash
cd backend
../.venv/Scripts/python.exe -c "import app.main; print('ok')"   # 必须有输出
```

## 3. 运维命令

```bash
cd backend

# 从 data/uploads 重新入库（分词器变更 / 索引损坏 / 版本迁移后重建）
../.venv/Scripts/python.exe -m app.rag.ingest --fresh

# 仅从 Chroma 重建 BM25
../.venv/Scripts/python.exe -m app.rag.bm25_index rebuild

# 命令行检索：核对检索质量用，--no-rerank 看纯 RRF 顺序
../.venv/Scripts/python.exe -m app.rag.search "指针分析的别名集是什么" --no-rerank
```

`ingest --fresh` 会把旧 Chroma 目录**改名备份**（`data/chroma.bak.<时间戳>/`）而不是删除。
2026-10-14 留有一份 `data/chroma.bak.20261014-163233`，是 chromadb 0.5.x 写的原始索引
（1.5.9 读不了，但它是唯一一份旧格式副本，暂留）。

## 4. MCP / 流式预研结论（2026-09-10 实测，仍然有效）

| 事实 | 结论 |
|---|---|
| `mcp` 版本 | **2.2.0**，高层类叫 `MCPServer`，**不是** `FastMCP` |
| 导入路径 | `from mcp.server import MCPServer`；`from mcp.client.session import ClientSession`；`from mcp.client._memory import InMemoryTransport` |
| 字段命名 | mcp 2.x 全 snake_case：`input_schema` / `is_error` / `structured_content` |
| `structured_content` | **取决于返回注解**：`-> dict` 时它是 `None`（JSON 只在文本内容里）；`-> dict[str, Any]` 时才是那个 dict 本身；只有非 dict 注解才包成 `{"result": ...}` |
| **MCP 隐藏异常细节** | 工具抛异常时客户端只拿到 `is_error=True` 与通用文案，真实异常仅在服务端日志 → **工具必须返回 `{ok, data, error}` 而不是抛异常** |
| 同进程运行 | 必须用 `InMemoryTransport`，**不可用 stdio 子进程**（BGE-M3 约 2GB，子进程会再加载一份） |
| 流式 + 工具调用 | `AIMessageChunk` 累积可同时得到 token 流式与 tool calling，`tool_call_chunks` 会自动合并 |
| SSE 行协议 | `data` 里不能出现裸换行；序列化用 `ensure_ascii=False` |

## 5. 本次修复掉的问题（2026-10-14，均有回归测试）

| # | 原症状 | 位置 | 现状 |
|---|---|---|---|
| 1 | `import app.main` 段错误 | `app/__init__.py` | 已修（见 §2） |
| 2 | `chat.py` 调用已不存在的 `run_agent`，**应用无法导入** | `app/api/chat.py` | 已重写为事件流转发 |
| 3 | 假流式（全量返回后 4 字符切片 + sleep） | `app/api/chat.py` | 已改为 `astream` 真流式 |
| 4 | 前端靠 payload 形状猜事件，`route` 永远丢失 | `useChat.ts` | 已按 `data.type` 分发 |
| 5 | 前端无错误处理，失败后输入框永久禁用 | `useChat.ts` | 已加 try/catch/finally + 错误展示 |
| 6 | BM25 中文单字切分 | `rag/bm25_index.py` | 已换 jieba（`lcut_for_search`）+ 小停用词表 |
| 7 | 无 BM25 重建路径（"可重建的缓存"是空头支票） | `rag/bm25_index.py` | 新增 `rebuild_from_chroma()` |
| 8 | 向量侧伪造相似度 `1.0 - i*0.05`，误导 LLM 判断资料相关性 | `rag/hybrid_search.py` | 改用 `similarity_search_with_relevance_scores` |
| 9 | RRF 去重键 `page_content[:200]`，相邻分片互相覆盖 | `rag/hybrid_search.py` | 改用 loader 打的稳定 `chunk_id` |
| 10 | Reranker 从未接入 | `rag/hybrid_search.py` | 已接入，`settings.enable_reranker` 可关 |
| 11 | 精排模型缺失时每次检索都重试下载 | `rag/reranker.py` | 10 分钟冷却，失败降级为 RRF 顺序 |
| 12 | 引用溯源链路断裂（`source_data` 从未写入） | `agent/core.py` / `api/chat.py` | 已打通到前端 |
| 13 | `metadata["source"]` 是绝对路径，泄露目录结构给用户与 LLM | `rag/loader.py` | 降级为文件名 |
| 14 | compose 传 `GEMINI_API_KEY`，容器内拿不到 LLM key | `docker-compose.yml` | 已修正 + 透传 SerpAPI + 模型缓存卷 |
| 15 | nginx 默认 60s 读超时切断长回答 | `frontend/nginx.conf` | 提到 300s |
| 16 | 依赖全是 `>=` 下限 | `requirements.txt` | 全部钉死 |

## 6. 已知未修（有意留下）

| # | 问题 | 说明 / 建议 |
|---|---|---|
| 1 | **BM25 小语料陷阱** | `idf` 只在 `df < N/2` 时为正，语料 ≤2 篇时**任何查询都返回空**。这是数学不是 bug。换 `BM25Plus` 可解（未做） |
| 2 | 精排模型本地缺失 | HuggingFace 下载失败；已装 `certifi` 但本机仍下不动（疑似无外网）。把模型放进 `data/models/BAAI/bge-reranker-v2-m3` 即启用，否则保持自动降级 |
| 3 | Chroma 与 BM25 之间无事务 | 写 Chroma 成功、写 BM25 失败会留下分叉，且此时 `DocumentModel` 行尚未插入，无从对账。`remove_source` 与删除已尽量缩小窗口，但跨存储原子性未解决 |
| 4 | `tools/document_parser.parse_document`（MCP 工具）不建 `DocumentModel` 行 | 经 API 上传会建行，经 MCP 工具解析不会。因此这类片段无法从 UI 删除。工具需要 DB session 才能修，未做 |
| 5 | `torch` 未钉版本 | 它是 sentence-transformers 的传递依赖、体积大且平台相关；Docker 镜像里装到的版本可能与本机不同 |
| 6 | `langchain_community.vectorstores.Chroma` 已弃用 | 官方建议迁移到 `langchain-chroma` 包（未迁） |
| 7 | 来源名带 uuid 后缀 | `metadata["source"]` 用的是存储文件名（`原始名_uuid8位.扩展名`），因为它是删除/去重时与 Chroma、BM25 对得上的键。代价是界面上的来源名不够干净 |
| 8 | 无认证与用户隔离 | `/api/conversations` 返回全库会话，任何人都能消耗 DeepSeek 额度。个人自用可接受 |
| 9 | 仓库无 CI | 测试要手动跑 |
| 10 | 答案正文没有 `[1][2]` 引用角标 | 来源以标签形式展示在答案下方；正文内联角标未做（认为标签够用） |

## 7. 其他容易踩的点（血泪）

- **pickle 兼容性**：`data/bm25_index.pkl` 里存的是**已分词**的语料。改 `_tokenize` 不会让旧文件失效，只会让它变成「新切法查旧粒度」的静默错误 → 必须 rebuild。
- **`BM25Okapi` 的类名会变**：pickle 里嵌着活的 `BM25Okapi`，`rank_bm25` 升级会让索引直接读不出来（已钉死 0.2.2）。加载失败是**降级而非崩溃**，但 `save()` 会拒绝在空白基础上覆盖磁盘。
- **jieba 首次加载会打 stderr**（`Building prefix dict…`），已在 `bm25_index` 里 `setLogLevel` 静音；另有一句 `pkg_resources is deprecated` 的 warning 来自 jieba 自身。
- **`has_persisted_index()` 判断 `chroma.sqlite3` 是否存在**，硬编码内部实现文件名，Chroma 大版本升级时可能失灵。
- **`Base.metadata.create_all` 不改表**：新增列对已存在的库静默不生效（Task 9 会引入 `app/db/migrate.py`）。
- **Windows 上 `os.replace` 遇到被占用的目标文件会抛 PermissionError**（实测 150/150），此时旧索引完好、只是本次落盘失败。单进程不可达；`uvicorn --workers N` 共享 `data/` 时才会遇到。