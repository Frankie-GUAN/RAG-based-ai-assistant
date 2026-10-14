# 个人知识库助手（Agentic RAG）

基于 **ReAct Agent + MCP 工具层 + 混合检索（含精排）+ DeepSeek** 的个人知识库问答系统。前后端分离，支持私有文档问答、联网搜索、多轮对话与**引用溯源**（回答可点开看依据哪一段）。

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Vue 3 + Vite + TypeScript + Pinia + Tailwind CSS v4 |
| 后端 | FastAPI + LangChain（ReAct 循环自研，见 `app/agent/core.py`） |
| LLM | DeepSeek (OpenAI 兼容 API) |
| Embedding | BAAI/bge-m3（本地部署，1024 维） |
| 向量存储 | ChromaDB（本地持久化） |
| 关键词检索 | BM25（rank-bm25，中文用 jieba 分词） |
| 混合检索 | RRF 融合（向量 + BM25，各召回 10 个） |
| 重排序 | BAAI/bge-reranker-v2-m3（Cross-Encoder，本地；**已接入**，可用 `ENABLE_RERANKER=false` 关闭） |
| 工具层 | MCP（`mcp==2.2.0`，同进程 `InMemoryTransport`） |
| 评估 | LLM-as-Judge（忠实度 + 相关性评分） |
| 数据库 | MySQL 8.0 + SQLAlchemy ORM |
| 部署 | Docker Compose |

## 架构

```
┌──────────────────────────────────────────────────┐
│                  Vue 3 Frontend                   │
│   Chat UI · Knowledge Base · Evaluation          │
│         SSE（真流式，按 type 分发 + 错误处理）      │
└────────────────────┬─────────────────────────────┘
                     │ HTTP + SSE
┌────────────────────▼─────────────────────────────┐
│               FastAPI Backend                     │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │   Agent Core：ReAct 循环（astream 真流式）   │ │
│  │   逐 token 外发 · 并行工具调用 · 迭代上限     │ │
│  └──────────────────┬──────────────────────────┘ │
│                     │                              │
│  ┌──────────────────▼──────────────────────────┐ │
│  │        MCP 工具层（MCPServer/Client）        │ │
│  │  search_documents · search_web · parse_doc  │ │
│  │  工具统一返回 {ok, data, error}，不抛异常     │ │
│  └──────────────────┬──────────────────────────┘ │
│                     │                              │
│  ┌──────────────────▼──────────────────────────┐ │
│  │      混合检索 + Cross-Encoder 精排           │ │
│  │  BGE-M3(Dense) + BM25(Sparse) → RRF →       │ │
│  │  精排(10 选 4，失败自动退回 RRF 顺序)         │ │
│  └──────────────────────────────────────────────┘ │
│                                                   │
│  ChromaDB · MySQL · BGE-M3 · BGE-Reranker        │
└──────────────────────────────────────────────────┘
```

**事件协议**：后端 `agent/protocol.py` 定义 `tool_call` / `tool_result` / `token` / `citation` / `final` / `error`，前端按 payload 的 `type` 分发。`tests/test_protocol.py` 会遍历所有事件子类，任何新增类型未登记都会让测试失败。

## 目录结构

```
├── backend/
│   ├── app/
│   │   ├── agent/        # Agent Core（ReAct 循环、事件协议、LLM 工厂）
│   │   ├── api/          # FastAPI 路由（chat, knowledge, evaluation, tools）
│   │   ├── context/      # 上下文工程引擎（分层 token 预算）
│   │   ├── db/           # SQLAlchemy 连接与 Base
│   │   ├── mcp/          # MCP 工具服务端 / 客户端封装
│   │   ├── models/       # ORM 模型（Conversation, Message, Document, Evaluation）
│   │   ├── rag/          # RAG 核心（加载器、向量存储、BM25、混合检索、精排、ingest/search CLI）
│   │   ├── services/     # 业务逻辑（chat, knowledge, evaluation）
│   │   ├── tools/        # 工具实现（由 app/mcp 注册为 MCP 工具）
│   │   ├── config.py     # 集中配置（pydantic-settings）
│   │   └── main.py       # FastAPI 入口
│   ├── tests/
│   ├── pytest.ini        # 默认排除 requires_model 用例
│   ├── Dockerfile
│   └── requirements.txt  # 版本已钉死（见文件顶部说明）
├── frontend/
│   ├── src/
│   │   ├── components/   # Vue 组件（chat, knowledge, evaluation, layout）
│   │   ├── composables/  # 组合式 API（useChat, useKnowledge, useEvaluation）
│   │   ├── stores/       # Pinia 状态管理
│   │   ├── views/        # 页面（ChatView, KnowledgeView, EvaluationView）
│   │   └── styles/       # Tailwind + 设计系统 CSS 变量
│   ├── Dockerfile
│   └── package.json
├── docs/                 # 计划、技术事实与索引
├── docker-compose.yml
├── .env                  # 环境变量（API Key 等）
└── data/                 # uploads 源文件 / ChromaDB / BM25 索引 / 模型（本地持久化）
```

## 环境要求

- Python 3.11+
- Node.js 20+
- MySQL 8.0

## 安装与运行

### 1. 配置环境变量

在项目根目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=你的DeepSeek密钥
MYSQL_PASSWORD=你的MySQL密码
SERPAPI_KEY=你的SerpAPI密钥（可选，用于联网搜索）
```

### 2. 后端

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

首次启动会自动创建 MySQL 表，并在后台预热检索链路（BGE-M3 + Chroma + BM25），首个请求因此不必再付冷启动的代价。Embedding 模型首次预热时需联网下载；Reranker 模型在 P1 接入前不会被加载。如需跳过启动预热（例如测试），设 `PRELOAD_MODELS=false`。

### 3. 前端

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 `http://localhost:5173`。

### 4. Docker 部署

```bash
docker compose up --build
```

- 前端：`http://localhost:3000`
- 后端：`http://localhost:8000`
- MySQL：`localhost:3306`

## RAG 评估

评估功能使用 LLM-as-Judge 对每个查询打分：

- **忠实度 (Faithfulness)**：回答是否完全基于检索到的上下文，有无编造
- **相关性 (Answer Relevancy)**：回答与问题的匹配程度

结果持久化到 MySQL，可在评估页面查看历史记录。

## 项目亮点

- **真流式输出**：token 直接来自 `model.astream()`，到达即转发（此前是等答案生成完再按 4 字符切片伪造的）
- **ReAct Agent 循环**：模型自主决定是否检索、调用哪个工具；无依赖步骤并发执行，迭代上限 8
- **MCP 工具层**：工具统一以 MCP 协议暴露，Agent 作 Client 调用；工具失败以 `{ok, data, error}` 返回而非抛异常，Agent 据此决策
- **混合检索 + 精排**：BM25（jieba 分词）+ BGE-M3 语义 → RRF 融合 → Cross-Encoder 精排 10 选 4，**精排失败自动退回 RRF 顺序**
- **引用溯源**：回答下方的来源标签指向具体文件的具体片段，随会话持久化，重开对话仍可核对
- **本地 Embedding**：BGE-M3 完全本地运行，零 API 成本
- **可重建的索引**：`python -m app.rag.ingest --fresh` 可从 `data/uploads` 重建全部索引
- **前后端分离**：FastAPI + Vue 3

## 运维命令

```bash
cd backend

# 全部测试（默认跳过需要 2GB 模型的用例）
../.venv/Scripts/python.exe -m pytest tests/ -q

# 从 data/uploads 重建全部索引（--fresh 会把旧索引改名备份）
../.venv/Scripts/python.exe -m app.rag.ingest --fresh

# 仅重建 BM25（换分词器后必须执行）
../.venv/Scripts/python.exe -m app.rag.bm25_index rebuild

# 命令行检索，核对检索质量（--no-rerank 看纯 RRF 顺序）
../.venv/Scripts/python.exe -m app.rag.search "如何解除劳动合同"
```

## 当前状态与已知问题

已修复的虚假宣称（MCP、真流式、reranker 启用、引用溯源）与仍未修的问题，都记录在
[docs/tech-notes/verified-facts.md](docs/tech-notes/verified-facts.md)。
修改计划见 [docs/plans/2026-10-14-personal-kb-revival.md](docs/plans/2026-10-14-personal-kb-revival.md)。

**改代码前请先读 verified-facts.md** —— 那里有本机实测过的技术前提和带行号的缺陷清单。

尚未修复的主要问题：

- **会话摘要只生成一次**：窗口 10 时第 11 条消息触发摘要，此后中间的消息既不在摘要也不在窗口，会静默丢失（规格见旧 P0 计划 Task 9）
- **评估页仍挂在主导航**：个人使用场景下它对用户零价值；且 Judge 无评分锚点、打分失败被静默计为 0 分
- **精排模型本地没有**：HuggingFace 下载在本机失败，检索会自动退回 RRF 顺序（功能正常，只是没有精排）。把模型放到 `data/models/BAAI/bge-reranker-v2-m3` 即可启用
- **联网搜索不可用**：`google-search-results` 未安装
- **语料 ≤2 篇时 BM25 必然返回空**：BM25 的 idf 数学性质，非 bug
