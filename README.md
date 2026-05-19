# Agentic RAG — 智能问答平台

基于 **LangGraph Agent + MCP 协议 + 混合检索 + DeepSeek** 的全栈智能问答系统。前后端分离架构，Agent 自主决策检索策略，支持私有文档问答、联网搜索、多轮对话和 RAG 质量评估。

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Vue 3 + Vite + TypeScript + Pinia + Tailwind CSS v4 |
| 后端 | FastAPI + LangGraph + LangChain |
| LLM | DeepSeek (OpenAI 兼容 API) |
| Embedding | BAAI/bge-m3（本地部署，1024 维） |
| 向量存储 | ChromaDB（本地持久化） |
| 关键词检索 | BM25（rank-bm25） |
| 混合检索 | RRF 融合（向量 + BM25） |
| 重排序 | BAAI/bge-reranker-v2-m3（Cross-Encoder，本地） |
| 评估 | LLM-as-Judge（忠实度 + 相关性评分） |
| 数据库 | MySQL 8.0 + SQLAlchemy ORM |
| 部署 | Docker Compose |

## 架构

```
┌──────────────────────────────────────────────────┐
│                  Vue 3 Frontend                   │
│   Chat UI · Knowledge Base · Evaluation Dashboard │
│              SSE Streaming                        │
└────────────────────┬─────────────────────────────┘
                     │ HTTP + SSE
┌────────────────────▼─────────────────────────────┐
│               FastAPI Backend                     │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │         LangGraph Agent 编排层               │ │
│  │  Router → Retrieve/Search → Generate → END  │ │
│  └──────────────────┬──────────────────────────┘ │
│                     │                              │
│  ┌──────────────────▼──────────────────────────┐ │
│  │            MCP 工具层                        │ │
│  │  search_documents · search_web · parse_doc  │ │
│  └──────────────────┬──────────────────────────┘ │
│                     │                              │
│  ┌──────────────────▼──────────────────────────┐ │
│  │            混合检索 + Reranker               │ │
│  │  BGE-M3 (Dense) + BM25 (Sparse) → RRF →     │ │
│  │  Cross-Encoder Reranker → Top-4              │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  ChromaDB · MySQL · BGE-M3 · BGE-Reranker        │
└──────────────────────────────────────────────────┘
```

## 目录结构

```
├── backend/
│   ├── app/
│   │   ├── agent/        # LangGraph Agent（状态、节点、图）
│   │   ├── api/          # FastAPI 路由（chat, knowledge, evaluation, tools）
│   │   ├── db/           # SQLAlchemy 连接与 Base
│   │   ├── models/       # ORM 模型（Conversation, Message, Document, Evaluation）
│   │   ├── rag/          # RAG 核心（加载器、向量存储、BM25、混合检索、Reranker）
│   │   ├── services/     # 业务逻辑（chat, knowledge, evaluation）
│   │   ├── tools/        # MCP 工具注册与实现
│   │   ├── config.py     # 集中配置（pydantic-settings）
│   │   └── main.py       # FastAPI 入口
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/   # Vue 组件（chat, knowledge, evaluation, layout）
│   │   ├── composables/  # 组合式 API（useChat, useKnowledge, useEvaluation）
│   │   ├── stores/       # Pinia 状态管理
│   │   ├── views/        # 页面（ChatView, KnowledgeView, EvaluationView）
│   │   └── styles/       # Tailwind + 设计系统 CSS 变量
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
├── .env                  # 环境变量（API Key 等）
└── data/                 # ChromaDB / BM25 索引 / 模型文件（本地持久化）
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

首次启动会自动创建 MySQL 表。Embedding 和 Reranker 模型首次使用时会自动下载（需网络）。

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

- **Agentic RAG**：LangGraph 驱动的多步自主决策，非一次性检索
- **MCP 协议**：标准化工具接口，Agent 与工具解耦
- **混合检索 + 精排**：BM25 关键词 + BGE-M3 语义 + RRF 融合 + Cross-Encoder
- **本地 Embedding**：BGE-M3 完全本地运行，零 API 成本
- **SSE 流式输出**：实时推送，逐 token 显示
- **评估体系**：RAG 质量可量化、可追踪
- **前后端分离**：FastAPI + Vue 3，现代化全栈架构
