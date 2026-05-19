# Agentic RAG 全栈重构实施方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 Streamlit 单体 RAG 应用重构为 FastAPI + Vue 3 + LangGraph Agent + MCP 协议 + 混合检索的全栈 AI 产品。

**Architecture:** 前端 Vue 3 通过 SSE 与后端 FastAPI 通信，后端使用 LangGraph 编排 Agent 决策循环，Agent 通过 MCP 标准化接口调用混合检索（BM25+向量+RRF+Reranker）、联网搜索等工具，数据层使用 ChromaDB（向量）和 MySQL（业务数据）。

**Tech Stack:** FastAPI, LangGraph, LangChain, ChromaDB, rank-bm25, BGE-Reranker, MySQL 8.0 + SQLAlchemy, Vue 3 + Vite + TypeScript + Pinia, Docker Compose

---

## 文件结构总览

```
project/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI 入口 + CORS + 生命周期
│   │   ├── config.py                  # 集中配置
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py                # POST /api/chat (SSE)
│   │   │   ├── knowledge.py           # 知识库 CRUD
│   │   │   ├── evaluation.py          # RAGAS 评估
│   │   │   └── tools.py               # MCP 工具列表
│   │   ├── agent/
│   │   │   ├── __init__.py
│   │   │   ├── state.py               # AgentState TypedDict
│   │   │   ├── nodes.py               # 各节点函数
│   │   │   └── graph.py               # StateGraph 构建
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── registry.py            # MCP 工具注册中心
│   │   │   ├── schemas.py             # 工具 JSON Schema
│   │   │   ├── document_search.py     # 混合检索工具
│   │   │   ├── web_search.py          # 联网搜索工具
│   │   │   └── document_parser.py     # 文档解析工具
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py              # 文档加载 + 切片
│   │   │   ├── vector_store.py        # ChromaDB 操作
│   │   │   ├── bm25_index.py          # BM25 关键词索引
│   │   │   ├── hybrid_search.py       # RRF 融合
│   │   │   └── reranker.py            # Cross-Encoder 重排序
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── conversation.py        # 会话 ORM
│   │   │   ├── message.py             # 消息 ORM
│   │   │   ├── document.py            # 文档元数据 ORM
│   │   │   └── evaluation.py          # 评估记录 ORM
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── session.py             # engine + SessionLocal
│   │   │   └── base.py               # declarative_base
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── chat_service.py        # 对话编排
│   │       ├── knowledge_service.py   # 知识库逻辑
│   │       └── evaluation_service.py  # 评估逻辑
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_hybrid_search.py
│   │   ├── test_reranker.py
│   │   ├── test_agent.py
│   │   ├── test_chat_api.py
│   │   └── test_knowledge_api.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.ts
│   │   ├── App.vue
│   │   ├── router/index.ts
│   │   ├── views/
│   │   │   ├── ChatView.vue
│   │   │   ├── KnowledgeView.vue
│   │   │   └── EvaluationView.vue
│   │   ├── components/
│   │   │   ├── layout/AppSidebar.vue
│   │   │   ├── chat/ChatMessage.vue
│   │   │   ├── chat/ChatInput.vue
│   │   │   ├── chat/SourcePanel.vue
│   │   │   ├── chat/AgentThinking.vue
│   │   │   ├── knowledge/FileUploader.vue
│   │   │   ├── knowledge/DocList.vue
│   │   │   └── evaluation/EvalChart.vue
│   │   ├── composables/
│   │   │   ├── useChat.ts
│   │   │   ├── useKnowledge.ts
│   │   │   └── useEvaluation.ts
│   │   ├── stores/
│   │   │   ├── chat.ts
│   │   │   └── knowledge.ts
│   │   ├── types/index.ts
│   │   └── styles/main.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── Dockerfile
├── docker-compose.yml
└── data/                              # ChromaDB + 上传文件 (保留现有)
```

---

## Phase 1: 项目骨架搭建

### Task 1.1: 后端项目目录初始化

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/__init__.py`

- [ ] **Step 1: 创建 requirements.txt**

```txt
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
sse-starlette>=2.0.0
langchain>=0.3.0
langchain-community>=0.3.0
langchain-google-genai>=2.0.0
langgraph>=0.2.0
google-generativeai>=0.8.0
chromadb>=0.5.0
pypdf>=4.0.0
google-search-results>=2.4.2
sqlalchemy>=2.0.0
pymysql>=1.1.0
cryptography>=42.0.0
rank-bm25>=0.2.2
ragas>=0.2.0
sentence-transformers>=3.0.0
python-multipart>=0.0.12
pydantic>=2.0.0
pydantic-settings>=2.0.0
```

- [ ] **Step 2: 创建 config.py**

```python
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Gemini
    gemini_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    embed_model: str = "gemini-embedding-001"

    # MySQL
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "rag_agent"

    # SerpAPI
    serpapi_key: str = ""

    # Paths
    data_dir: Path = Path(__file__).resolve().parent.parent.parent / "data"
    upload_dir: Path = data_dir / "uploads"
    chroma_dir: Path = data_dir / "chroma"

    # RAG
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    hybrid_top_k: int = 10

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
```

- [ ] **Step 3: 创建 main.py 骨架**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agentic RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 4: 安装依赖并启动验证**

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

访问 `http://localhost:8000/api/health`，确认返回 `{"status": "ok"}`。

- [ ] **Step 5: Commit**

```bash
git add backend/requirements.txt backend/app/__init__.py backend/app/config.py backend/app/main.py backend/tests/__init__.py
git commit -m "feat: scaffold backend project with FastAPI entry point and config"
```

---

### Task 1.2: 前端项目初始化 (Vue 3 + Vite + TypeScript)

**Files:**
- Create: `frontend/` (Vite scaffold)

- [ ] **Step 1: 使用 Vite 创建 Vue 3 + TS 项目**

```bash
cd E:/project/RAG智能问答机器人
npm create vite@latest frontend -- --template vue-ts
cd frontend
npm install
```

- [ ] **Step 2: 安装额外依赖**

```bash
npm install vue-router@4 pinia axios
npm install -D tailwindcss @tailwindcss/vite
```

- [ ] **Step 3: 配置 vite.config.ts**

```typescript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

- [ ] **Step 4: 创建最小目录结构**

```bash
mkdir -p src/router src/views src/components/layout src/components/chat src/components/knowledge src/components/evaluation src/composables src/stores src/types src/styles
```

- [ ] **Step 5: 创建 router/index.ts**

```typescript
import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'chat', component: () => import('../views/ChatView.vue') },
    { path: '/knowledge', name: 'knowledge', component: () => import('../views/KnowledgeView.vue') },
    { path: '/evaluation', name: 'evaluation', component: () => import('../views/EvaluationView.vue') },
  ],
})

export default router
```

- [ ] **Step 6: 创建 main.ts**

```typescript
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import './styles/main.css'

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
```

- [ ] **Step 7: 创建 App.vue 骨架**

```vue
<template>
  <div class="flex h-screen bg-gray-50">
    <AppSidebar />
    <main class="flex-1 overflow-hidden">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
import AppSidebar from './components/layout/AppSidebar.vue'
</script>
```

- [ ] **Step 8: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold Vue 3 + Vite + TypeScript frontend"
```

---

### Task 1.3: Docker Compose + MySQL 配置

**Files:**
- Create: `docker-compose.yml`
- Create: `backend/.env.example`
- Create: `backend/Dockerfile`
- Create: `frontend/Dockerfile`

- [ ] **Step 1: 创建 docker-compose.yml**

```yaml
version: "3.8"

services:
  mysql:
    image: mysql:8.0
    container_name: rag-mysql
    restart: unless-stopped
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD:-ragagent123}
      MYSQL_DATABASE: rag_agent
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend:
    build: ./backend
    container_name: rag-backend
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - MYSQL_HOST=mysql
      - MYSQL_PORT=3306
      - MYSQL_USER=root
      - MYSQL_PASSWORD=${MYSQL_ROOT_PASSWORD:-ragagent123}
      - MYSQL_DATABASE=rag_agent
    volumes:
      - ./data:/app/data
    depends_on:
      mysql:
        condition: service_healthy

  frontend:
    build: ./frontend
    container_name: rag-frontend
    restart: unless-stopped
    ports:
      - "3000:80"
    depends_on:
      - backend

volumes:
  mysql_data:
```

- [ ] **Step 2: 创建 backend/.env.example**

```
GEMINI_API_KEY=your_gemini_key
MYSQL_ROOT_PASSWORD=ragagent123
```

- [ ] **Step 3: 创建 backend/Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: 创建 frontend/Dockerfile**

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

- [ ] **Step 5: 创建 frontend/nginx.conf**

```nginx
server {
    listen 80;
    server_name localhost;

    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_buffering off;
    }
}
```

- [ ] **Step 6: 验证 MySQL 连接可用**

```bash
docker compose up mysql -d
# 用 Navicat 连接 localhost:3306, root / ragagent123, 确认 rag_agent 库已创建
```

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml backend/.env.example backend/Dockerfile frontend/Dockerfile frontend/nginx.conf
git commit -m "feat: add Docker Compose with MySQL, backend and frontend services"
```

---

## Phase 2: 数据库层

### Task 2.1: SQLAlchemy 基础 + MySQL 连接

**Files:**
- Create: `backend/app/db/__init__.py`
- Create: `backend/app/db/base.py`
- Create: `backend/app/db/session.py`

- [ ] **Step 1: 创建 base.py**

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

- [ ] **Step 2: 创建 session.py**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings

database_url = (
    f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}"
    f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
    "?charset=utf8mb4"
)

engine = create_engine(database_url, pool_size=10, max_overflow=20, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- [ ] **Step 3: 在 main.py 添加启动建表**

```python
from contextlib import asynccontextmanager

from app.db.base import Base
from app.db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Agentic RAG API", version="0.1.0", lifespan=lifespan)
```

- [ ] **Step 4: 启动验证建表**

```bash
uvicorn app.main:app --reload --port 8000
# 确认启动日志无报错，Navicat 查看表已自动创建（暂时为空）
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/db/ backend/app/main.py
git commit -m "feat: add SQLAlchemy MySQL connection with auto-migration"
```

---

### Task 2.2: 业务 ORM 模型

**Files:**
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/models/conversation.py`
- Create: `backend/app/models/message.py`
- Create: `backend/app/models/document.py`
- Create: `backend/app/models/evaluation.py`

- [ ] **Step 1: 创建 conversation.py**

```python
from datetime import datetime
from sqlalchemy import String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), default="新对话")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    messages = relationship("Message", back_populates="conversation", order_by="Message.created_at")
```

- [ ] **Step 2: 创建 message.py**

```python
from datetime import datetime
from sqlalchemy import Integer, String, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(Integer, ForeignKey("conversations.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user / assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str | None] = mapped_column(String(20), nullable=True)  # web / doc / direct
    source_data: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")
```

- [ ] **Step 3: 创建 document.py**

```python
from datetime import datetime
from sqlalchemy import Integer, String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- [ ] **Step 4: 创建 evaluation.py**

```python
from datetime import datetime
from sqlalchemy import Integer, String, Float, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str] = mapped_column(Text, nullable=False)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- [ ] **Step 5: 更新 main.py 的 import 确保建表**

```python
# 在 lifespan 之前添加 import，确保所有模型被 Base.metadata 感知
import app.models.conversation  # noqa: F401
import app.models.message        # noqa: F401
import app.models.document       # noqa: F401
import app.models.evaluation     # noqa: F401
```

- [ ] **Step 6: 启动验证**

```bash
uvicorn app.main:app --reload --port 8000
# Navicat 确认四张表均已创建
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/models/ backend/app/main.py
git commit -m "feat: add ORM models for conversations, messages, documents, evaluations"
```

---

## Phase 3: RAG 核心 — 混合检索 + Reranker

### Task 3.1: 文档加载器（迁移 + 增强）

**Files:**
- Create: `backend/app/rag/__init__.py`
- Create: `backend/app/rag/loader.py`

- [ ] **Step 1: 创建 loader.py**

```python
from pathlib import Path
from typing import List
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from app.config import settings


def save_uploaded_file(uploaded_bytes: bytes, filename: str) -> Path:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    safe_name = f"{Path(filename).stem}_{uuid4().hex}{suffix}"
    file_path = settings.upload_dir / safe_name
    file_path.write_bytes(uploaded_bytes)
    return file_path


def load_file(file_path: Path) -> List[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".txt":
        loader = TextLoader(str(file_path), autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/rag/__init__.py backend/app/rag/loader.py
git commit -m "feat: add document loader with PDF/TXT support and chunking"
```

---

### Task 3.2: ChromaDB 向量存储

**Files:**
- Create: `backend/app/rag/vector_store.py`

- [ ] **Step 1: 创建 vector_store.py**

```python
import os
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embed_model,
        google_api_key=settings.gemini_api_key,
    )
    embeddings.embed_query("ping")
    return embeddings


def has_persisted_index() -> bool:
    return (settings.chroma_dir / "chroma.sqlite3").exists()


def load_vectorstore() -> Optional[Chroma]:
    if not has_persisted_index():
        return None
    return Chroma(persist_directory=str(settings.chroma_dir), embedding_function=get_embeddings())


def add_documents(documents: List[Document]) -> Chroma:
    embeddings = get_embeddings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    if has_persisted_index():
        vectorstore = Chroma(persist_directory=str(settings.chroma_dir), embedding_function=embeddings)
        vectorstore.add_documents(documents)
    else:
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=str(settings.chroma_dir))
    return vectorstore


def get_retriever():
    vectorstore = load_vectorstore()
    if vectorstore is None:
        raise ValueError("No persisted vector store found")
    return vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/rag/vector_store.py
git commit -m "feat: add ChromaDB vector store with Gemini embeddings"
```

---

### Task 3.3: BM25 关键词索引

**Files:**
- Create: `backend/app/rag/bm25_index.py`

- [ ] **Step 1: 创建 bm25_index.py**

```python
import pickle
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings


class BM25Index:
    def __init__(self):
        self._index: BM25Okapi | None = None
        self._documents: List[Document] = []
        self._index_path = settings.data_dir / "bm25_index.pkl"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re

        return [t.lower() for t in re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text)]

    def build(self, documents: List[Document]) -> None:
        self._documents = documents
        corpus = [self._tokenize(doc.page_content) for doc in documents]
        self._index = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        if self._index is None:
            return []
        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        max_score = scores[ranked[0][0]] if ranked and scores[ranked[0][0]] > 0 else 1.0
        return [(self._documents[i], score / max_score) for i, score in ranked if score > 0]

    def save(self) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "wb") as f:
            pickle.dump({"documents": self._documents, "index": self._index}, f)

    def load(self) -> bool:
        if not self._index_path.exists():
            return False
        with open(self._index_path, "rb") as f:
            data = pickle.load(f)
        self._documents = data["documents"]
        self._index = data["index"]
        return True

    @property
    def document_count(self) -> int:
        return len(self._documents)
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/rag/bm25_index.py
git commit -m "feat: add BM25 keyword index for hybrid search"
```

---

### Task 3.4: 混合检索 + RRF 融合

**Files:**
- Create: `backend/app/rag/hybrid_search.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_hybrid_search.py`

- [ ] **Step 1: 创建 hybrid_search.py**

```python
from typing import List, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.rag.bm25_index import BM25Index
from app.rag.vector_store import get_retriever


def _vector_search(query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return [(doc, 1.0 - i * 0.05) for i, doc in enumerate(docs)][:top_k]


def _rrf_fuse(
    vector_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    k: int = 60,
    top_k: int = 10,
) -> List[Tuple[Document, float]]:
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(vector_results):
        key = doc.page_content[:200]
        doc_map[key] = doc
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)

    for rank, (doc, _) in enumerate(bm25_results):
        key = doc.page_content[:200]
        doc_map[key] = doc
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_map[key], score) for key, score in ranked]


def hybrid_search(
    query: str,
    bm25_index: BM25Index,
    vector_top_k: int = 10,
    bm25_top_k: int = 10,
    final_top_k: int = 10,
) -> List[Tuple[Document, float]]:
    vector_results = _vector_search(query, top_k=vector_top_k)
    bm25_results = bm25_index.search(query, top_k=bm25_top_k)
    return _rrf_fuse(vector_results, bm25_results, top_k=final_top_k)
```

- [ ] **Step 2: 创建 conftest.py**

```python
import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="劳动合同法第三十六条规定用人单位与劳动者协商一致可以解除劳动合同", metadata={"source": "labor_law.pdf"}),
        Document(page_content="2025年新修订的劳动法增加了远程办公相关条款", metadata={"source": "amendments.pdf"}),
        Document(page_content="Python是一种广泛使用的高级编程语言", metadata={"source": "python.pdf"}),
        Document(page_content="劳动者提前三十日书面通知用人单位可以解除劳动合同", metadata={"source": "labor_law.pdf"}),
        Document(page_content="FastAPI是现代Python Web框架支持异步处理", metadata={"source": "python.pdf"}),
    ]
```

- [ ] **Step 3: 创建 test_hybrid_search.py（测试 RRF 融合逻辑）**

```python
from langchain_core.documents import Document
from app.rag.hybrid_search import _rrf_fuse


def test_rrf_fusion_ranks_both_sources():
    docs = [
        Document(page_content="A"),
        Document(page_content="B"),
        Document(page_content="C"),
    ]
    vector = [(docs[0], 0.9), (docs[1], 0.5)]
    bm25 = [(docs[1], 0.8), (docs[2], 0.3)]

    result = _rrf_fuse(vector, bm25, top_k=3)
    ids = [doc.page_content for doc, _ in result]
    # B appears in both lists → should rank highest
    assert ids[0] == "B"
    assert len(result) == 3


def test_rrf_empty_inputs():
    result = _rrf_fuse([], [], top_k=5)
    assert result == []


def test_rrf_handles_duplicate_content():
    doc = Document(page_content="same")
    result = _rrf_fuse([(doc, 1.0)], [(doc, 1.0)], top_k=5)
    assert len(result) == 1
```

- [ ] **Step 4: 运行测试**

```bash
cd backend
pytest tests/test_hybrid_search.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/rag/hybrid_search.py backend/tests/conftest.py backend/tests/test_hybrid_search.py
git commit -m "feat: add hybrid search with RRF fusion of vector and BM25 results"
```

---

### Task 3.5: Cross-Encoder Reranker

**Files:**
- Create: `backend/app/rag/reranker.py`
- Create: `backend/tests/test_reranker.py`

- [ ] **Step 1: 创建 reranker.py**

```python
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config import settings


class Reranker:
    def __init__(self, model_name: str | None = None):
        self._model = CrossEncoder(model_name or settings.reranker_model)

    def rerank(
        self, query: str, candidates: List[Tuple[Document, float]], top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc, _ in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for (doc, _), score in ranked[:top_k]]
```

- [ ] **Step 2: 创建 test_reranker.py**

```python
from langchain_core.documents import Document
from app.rag.reranker import Reranker


def test_reranker_returns_top_k():
    reranker = Reranker()
    docs = [
        (Document(page_content="Python编程语言入门教程"), 0.5),
        (Document(page_content="劳动法劳动合同解除"), 0.5),
        (Document(page_content="Django和FastAPI对比"), 0.5),
        (Document(page_content="劳动争议仲裁流程"), 0.5),
        (Document(page_content="机器学习基础"), 0.5),
    ]
    query = "劳动合同如何解除"
    result = reranker.rerank(query, docs, top_k=3)
    assert len(result) == 3
    # 劳动法相关文档应该排在前面
    assert "劳动法" in result[0][0].page_content or "劳动合同" in result[0][0].page_content
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_reranker.py -v
# 注意：首次运行会下载 BGE-Reranker 模型 (~1GB)，耗时较长
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/rag/reranker.py backend/tests/test_reranker.py
git commit -m "feat: add Cross-Encoder reranker with BGE-Reranker-v2-m3"
```

---

## Phase 4: MCP 工具层 + Agent

### Task 4.1: MCP 工具注册中心 + JSON Schema

**Files:**
- Create: `backend/app/tools/__init__.py`
- Create: `backend/app/tools/schemas.py`
- Create: `backend/app/tools/registry.py`

- [ ] **Step 1: 创建 schemas.py**

```python
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    type: str
    description: str


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, ToolParameter]


TOOL_SCHEMAS: dict[str, ToolSchema] = {
    "search_documents": ToolSchema(
        name="search_documents",
        description="在私有文档知识库中进行混合检索（语义+关键词），返回最相关的文档片段",
        parameters={
            "query": ToolParameter(type="string", description="搜索查询"),
            "top_k": ToolParameter(type="integer", description="返回结果数量，默认4"),
        },
    ),
    "search_web": ToolSchema(
        name="search_web",
        description="在互联网上搜索最新信息，用于时效性问题和实时数据",
        parameters={
            "query": ToolParameter(type="string", description="搜索查询"),
            "num": ToolParameter(type="integer", description="返回结果数量，默认5"),
        },
    ),
    "parse_document": ToolSchema(
        name="parse_document",
        description="解析上传的文档（PDF/TXT），提取文本内容并建立索引",
        parameters={
            "file_path": ToolParameter(type="string", description="上传文件的本地路径"),
        },
    ),
}
```

- [ ] **Step 2: 创建 registry.py**

```python
from typing import Any, Callable

from app.tools.schemas import TOOL_SCHEMAS


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._tools[name] = fn

    def get(self, name: str) -> Callable | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "description": schema.description, "parameters": schema.parameters}
            for name, schema in TOOL_SCHEMAS.items()
        ]

    def execute(self, name: str, **kwargs) -> str:
        fn = self.get(name)
        if fn is None:
            return f"Tool '{name}' not found"
        return fn(**kwargs)


tool_registry = ToolRegistry()
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/tools/
git commit -m "feat: add MCP tool registry with JSON Schema definitions"
```

---

### Task 4.2: 工具实现（混合检索 / 联网搜索 / 文档解析）

**Files:**
- Create: `backend/app/tools/document_search.py`
- Create: `backend/app/tools/web_search.py`
- Create: `backend/app/tools/document_parser.py`

- [ ] **Step 1: 创建 document_search.py**

```python
from app.rag.hybrid_search import hybrid_search
from app.rag.bm25_index import BM25Index
from app.tools.registry import tool_registry


def search_documents(query: str, top_k: int = 4) -> str:
    bm25 = BM25Index()
    if bm25.document_count == 0:
        bm25.load()

    results = hybrid_search(query, bm25, final_top_k=top_k)
    if not results:
        return "未找到相关文档"

    lines = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "未知")
        lines.append(f"[{i}] (相关度: {score:.2f}) 来源: {source}\n{doc.page_content[:500]}")
    return "\n\n".join(lines)


tool_registry.register("search_documents", search_documents)
```

- [ ] **Step 2: 创建 web_search.py**

```python
import os
from typing import List

from langchain_community.utilities import SerpAPIWrapper

from app.config import settings
from app.tools.registry import tool_registry


def _format_items(items: List[dict]) -> str:
    lines = []
    for i, item in enumerate(items, 1):
        title = item.get("title", item.get("link", ""))
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        lines.append(f"[{i}] {title}\n{snippet}\n{link}")
    return "\n\n".join(lines)


def search_web(query: str, num: int = 5) -> str:
    if not settings.serpapi_key:
        return "联网搜索未配置 SerpAPI_KEY"

    os.environ["SERPAPI_API_KEY"] = settings.serpapi_key
    wrapper = SerpAPIWrapper(params={"num": num, "engine": "google", "hl": "zh-cn"})
    results = wrapper.results(query)

    items = []
    for item in results.get("organic_results", []):
        if item.get("link"):
            items.append(item)
    return _format_items(items)


tool_registry.register("search_web", search_web)
```

- [ ] **Step 3: 创建 document_parser.py**

```python
from app.rag.loader import save_uploaded_file, load_file, split_documents
from app.rag.vector_store import add_documents
from app.rag.bm25_index import BM25Index
from app.tools.registry import tool_registry


def parse_document(file_path: str) -> str:
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        return f"文件不存在: {file_path}"

    documents = load_file(path)
    chunks = split_documents(documents)

    # 写入向量存储
    add_documents(chunks)

    # 更新 BM25 索引
    bm25 = BM25Index()
    bm25.load()
    all_docs = list(bm25._documents) if bm25._documents else []
    all_docs.extend(chunks)
    bm25.build(all_docs)
    bm25.save()

    return f"成功解析文档 {path.name}，共生成 {len(chunks)} 个片段"


tool_registry.register("parse_document", parse_document)
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/tools/document_search.py backend/app/tools/web_search.py backend/app/tools/document_parser.py
git commit -m "feat: implement search_documents, search_web, parse_document tools"
```

---

### Task 4.3: LangGraph Agent — State + Nodes

**Files:**
- Create: `backend/app/agent/__init__.py`
- Create: `backend/app/agent/state.py`
- Create: `backend/app/agent/nodes.py`

- [ ] **Step 1: 创建 state.py**

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    has_docs: bool
    route: str
    context: str
    iteration: int
    final_answer: str
```

- [ ] **Step 2: 创建 nodes.py**

```python
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agent.state import AgentState
from app.config import settings
from app.tools.registry import tool_registry


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )


def router_node(state: AgentState) -> dict:
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是路由分类器。根据用户问题判断应该走哪条路径。"
         "输出: web (需要联网搜索最新信息) / rag (需要检索私有文档) / direct (直接回答)。"
         "只输出这三个词之一。"),
        ("human", "问题: {question}\n是否有私有文档: {has_docs}"),
    ])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"question": state["question"], "has_docs": "是" if state.get("has_docs") else "否"})
    decision = raw.strip().lower()

    if "web" in decision:
        route = "web"
    elif "rag" in decision and state.get("has_docs"):
        route = "rag"
    else:
        route = "direct"

    return {"route": route, "iteration": state.get("iteration", 0) + 1}


def retrieve_node(state: AgentState) -> dict:
    result = tool_registry.execute("search_documents", query=state["question"], top_k=4)
    return {"context": result}


def web_search_node(state: AgentState) -> dict:
    result = tool_registry.execute("search_web", query=state["question"])
    return {"context": result}


def generate_node(state: AgentState) -> dict:
    llm = _get_llm()
    context = state.get("context", "")
    history = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in state["messages"][-6:]
    )

    if context:
        system_text = (
            "你是专业中文助手，基于提供的参考资料回答问题。若资料不足，说明不确定。"
            f"\n\n参考资料:\n{context}"
        )
    else:
        system_text = "你是专业中文助手，基于自身知识回答问题。若不知道，说明不确定。"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "对话历史:\n{history}\n\n用户问题: {question}\n\n请用简洁中文回答。"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"history": history, "question": state["question"]})
    return {"final_answer": answer}
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/agent/
git commit -m "feat: add LangGraph Agent state and node definitions"
```

---

### Task 4.4: LangGraph 图构建

**Files:**
- Create: `backend/app/agent/graph.py`
- Create: `backend/tests/test_agent.py`

- [ ] **Step 1: 创建 graph.py**

```python
from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import router_node, retrieve_node, web_search_node, generate_node


def _route_decision(state: AgentState) -> str:
    route = state.get("route", "direct")
    if route == "web":
        return "web_search"
    elif route == "rag":
        return "retrieve"
    return "generate"


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", _route_decision, {
        "web_search": "web_search",
        "retrieve": "retrieve",
        "generate": "generate",
    })
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


agent_graph = build_graph()
```

- [ ] **Step 2: 创建 test_agent.py**

```python
from langchain_core.messages import HumanMessage
from app.agent.graph import agent_graph
from app.agent.state import AgentState


def test_agent_direct_route():
    state: AgentState = {
        "messages": [HumanMessage(content="Python中如何读取文件？")],
        "question": "Python中如何读取文件？",
        "has_docs": False,
        "route": "",
        "context": "",
        "iteration": 0,
        "final_answer": "",
    }
    result = agent_graph.invoke(state)
    assert "final_answer" in result
    assert result["final_answer"] != ""
    assert isinstance(result["final_answer"], str)
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_agent.py -v
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/agent/graph.py backend/tests/test_agent.py
git commit -m "feat: build LangGraph Agent with router -> retrieve/search -> generate flow"
```

---

## Phase 5: API 层

### Task 5.1: Chat 服务 + SSE 流式端点

**Files:**
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/services/chat_service.py`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/chat.py`
- Create: `backend/tests/test_chat_api.py`

- [ ] **Step 1: 创建 chat_service.py**

```python
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.graph import agent_graph
from app.agent.state import AgentState


def run_agent(question: str, history: list[dict], has_docs: bool) -> dict:
    lc_messages = []
    for m in history:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))
    lc_messages.append(HumanMessage(content=question))

    initial_state: AgentState = {
        "messages": lc_messages,
        "question": question,
        "has_docs": has_docs,
        "route": "",
        "context": "",
        "iteration": 0,
        "final_answer": "",
    }
    result = agent_graph.invoke(initial_state)
    return {
        "answer": result["final_answer"],
        "route": result.get("route", "direct"),
        "context": result.get("context", ""),
    }
```

- [ ] **Step 2: 创建 chat.py（SSE 流式端点）**

```python
import json
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.chat_service import run_agent

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    question: str
    history: list[dict] = []
    has_docs: bool = False


async def _stream_chat(request: ChatRequest) -> AsyncGenerator[dict, None]:
    # Send start event
    yield {"event": "status", "data": json.dumps({"status": "thinking"})}

    # Run agent (in thread pool since LangGraph is sync)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, run_agent, request.question, request.history, request.has_docs
    )

    yield {"event": "route", "data": json.dumps({"route": result["route"]})}

    # Simulate token-by-token streaming
    answer = result["answer"]
    for i in range(0, len(answer), 4):
        chunk = answer[i:i + 4]
        yield {"event": "token", "data": json.dumps({"content": chunk})}
        await asyncio.sleep(0.02)

    yield {"event": "done", "data": json.dumps({"answer": answer, "route": result["route"]})}


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    return EventSourceResponse(_stream_chat(request))
```

- [ ] **Step 3: 创建 test_chat_api.py**

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_stream_endpoint():
    response = client.post("/api/chat/stream", json={
        "question": "你好",
        "history": [],
        "has_docs": False,
    })
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_chat_api.py -v
```

- [ ] **Step 5: 注册路由到 main.py**

```python
from app.api.chat import router as chat_router

app.include_router(chat_router)
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/ backend/app/api/ backend/app/main.py backend/tests/test_chat_api.py
git commit -m "feat: add SSE streaming chat endpoint with Agent orchestration"
```

---

### Task 5.2: 知识库管理 API

**Files:**
- Create: `backend/app/services/knowledge_service.py`
- Create: `backend/app/api/knowledge.py`
- Create: `backend/tests/test_knowledge_api.py`

- [ ] **Step 1: 创建 knowledge_service.py**

```python
from pathlib import Path
from typing import List

from sqlalchemy.orm import Session

from app.models.document import Document as DocumentModel
from app.rag.loader import save_uploaded_file, load_file, split_documents
from app.rag.vector_store import add_documents, load_vectorstore, has_persisted_index
from app.rag.bm25_index import BM25Index


def process_document(db: Session, filename: str, content: bytes) -> DocumentModel:
    file_path = save_uploaded_file(content, filename)
    documents = load_file(file_path)
    chunks = split_documents(documents)

    vectorstore = add_documents(chunks)

    bm25 = BM25Index()
    if bm25.load():
        all_docs = list(bm25._documents)
    else:
        all_docs = []
    all_docs.extend(chunks)
    bm25.build(all_docs)
    bm25.save()

    doc_record = DocumentModel(
        filename=filename,
        file_type=Path(filename).suffix.lower().lstrip("."),
        chunk_count=len(chunks),
        file_path=str(file_path),
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)
    return doc_record


def list_documents(db: Session) -> List[DocumentModel]:
    return db.query(DocumentModel).order_by(DocumentModel.created_at.desc()).all()
```

- [ ] **Step 2: 创建 knowledge.py**

```python
from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.knowledge_service import process_document, list_documents

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    doc = process_document(db, file.filename, content)
    return {"id": doc.id, "filename": doc.filename, "chunk_count": doc.chunk_count}


@router.get("/documents")
async def get_documents(db: Session = Depends(get_db)):
    docs = list_documents(db)
    return [
        {"id": d.id, "filename": d.filename, "file_type": d.file_type, "chunk_count": d.chunk_count, "created_at": d.created_at.isoformat()}
        for d in docs
    ]
```

- [ ] **Step 3: 注册路由到 main.py**

```python
from app.api.knowledge import router as knowledge_router

app.include_router(knowledge_router)
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/knowledge_service.py backend/app/api/knowledge.py backend/app/main.py
git commit -m "feat: add knowledge base upload and list APIs"
```

---

### Task 5.3: MCP 工具列表 API

**Files:**
- Create: `backend/app/api/tools.py`

- [ ] **Step 1: 创建 tools.py**

```python
from fastapi import APIRouter
from pydantic import BaseModel

from app.tools.registry import tool_registry

router = APIRouter(prefix="/api/tools", tags=["tools"])


class ToolExecuteRequest(BaseModel):
    name: str
    params: dict = {}


@router.get("/")
async def list_tools():
    return tool_registry.list_tools()


@router.post("/execute")
async def execute_tool(request: ToolExecuteRequest):
    result = tool_registry.execute(request.name, **request.params)
    return {"result": result}
```

- [ ] **Step 2: 注册路由到 main.py**

```python
from app.api.tools import router as tools_router

app.include_router(tools_router)
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/api/tools.py backend/app/main.py
git commit -m "feat: add MCP tools listing and execution API"
```

---

## Phase 6: RAGAS 评估

### Task 6.1: 评估服务 + API

**Files:**
- Create: `backend/app/services/evaluation_service.py`
- Create: `backend/app/api/evaluation.py`

- [ ] **Step 1: 创建 evaluation_service.py**

```python
from typing import List

from langchain_core.documents import Document

from app.agent.graph import agent_graph
from app.agent.state import AgentState
from app.rag.hybrid_search import hybrid_search
from app.rag.bm25_index import BM25Index
from app.tools.registry import tool_registry


def _run_query(question: str, has_docs: bool) -> tuple[str, str]:
    state: AgentState = {
        "messages": [],
        "question": question,
        "has_docs": has_docs,
        "route": "",
        "context": "",
        "iteration": 0,
        "final_answer": "",
    }
    result = agent_graph.invoke(state)
    return result["final_answer"], result["context"]


def _evaluate_single(query: str, answer: str, context: str) -> dict:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from app.config import settings

    llm = ChatGoogleGenerativeAI(model=settings.llm_model, google_api_key=settings.gemini_api_key, temperature=0)

    # Faithfulness: is the answer supported by the context?
    faith_prompt = ChatPromptTemplate.from_messages([
        ("system", "评分0-1：回答是否完全基于给定上下文？1=完全基于上下文。只输出数字。"),
        ("human", "上下文: {context}\n\n回答: {answer}"),
    ])
    faith_chain = faith_prompt | llm | StrOutputParser()
    try:
        faithfulness = float(faith_chain.invoke({"context": context, "answer": answer}).strip())
    except ValueError:
        faithfulness = 0.0

    # Answer Relevancy
    relev_prompt = ChatPromptTemplate.from_messages([
        ("system", "评分0-1：回答与问题的相关程度。1=高度相关。只输出数字。"),
        ("human", "问题: {query}\n\n回答: {answer}"),
    ])
    relev_chain = relev_prompt | llm | StrOutputParser()
    try:
        answer_relevancy = float(relev_chain.invoke({"query": query, "answer": answer}).strip())
    except ValueError:
        answer_relevancy = 0.0

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
    }


def run_evaluation(queries: List[str], has_docs: bool) -> List[dict]:
    results = []
    for query in queries:
        answer, context = _run_query(query, has_docs)
        scores = _evaluate_single(query, answer, context)
        results.append({
            "query": query,
            "answer": answer,
            "context": context,
            **scores,
        })
    return results
```

- [ ] **Step 2: 创建 evaluation.py**

```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.evaluation import Evaluation as EvaluationModel
from app.services.evaluation_service import run_evaluation

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


class EvalRequest(BaseModel):
    queries: list[str]
    has_docs: bool = False


@router.post("/run")
async def run_eval(request: EvalRequest, db: Session = Depends(get_db)):
    results = run_evaluation(request.queries, request.has_docs)

    for r in results:
        record = EvaluationModel(
            query=r["query"],
            answer=r["answer"],
            context=r["context"],
            faithfulness=r["faithfulness"],
            answer_relevancy=r["answer_relevancy"],
        )
        db.add(record)
    db.commit()

    return {"results": results}


@router.get("/history")
async def get_eval_history(db: Session = Depends(get_db)):
    records = db.query(EvaluationModel).order_by(EvaluationModel.created_at.desc()).limit(50).all()
    return [
        {
            "id": r.id,
            "query": r.query,
            "answer": r.answer[:200],
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]
```

- [ ] **Step 3: 注册路由到 main.py**

```python
from app.api.evaluation import router as eval_router

app.include_router(eval_router)
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/evaluation_service.py backend/app/api/evaluation.py backend/app/main.py
git commit -m "feat: add RAGAS-style evaluation with faithfulness and relevancy scoring"
```

---

## Phase 7: Vue 3 前端

### Task 7.1: 前端基础布局 + 侧边栏 + 路由

**Files:**
- Create: `frontend/src/components/layout/AppSidebar.vue`
- Modify: `frontend/src/App.vue`
- Create: `frontend/src/views/ChatView.vue` (placeholder)
- Create: `frontend/src/views/KnowledgeView.vue` (placeholder)
- Create: `frontend/src/views/EvaluationView.vue` (placeholder)

- [ ] **Step 1: 创建 types/index.ts**

```typescript
export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  sourceType?: 'web' | 'doc'
}

export interface Source {
  title: string
  link: string
  snippet?: string
}

export interface DocumentRecord {
  id: number
  filename: string
  file_type: string
  chunk_count: number
  created_at: string
}

export interface EvalResult {
  query: string
  answer: string
  faithfulness: number
  answer_relevancy: number
}
```

- [ ] **Step 2: 创建 AppSidebar.vue**

```vue
<template>
  <aside class="w-56 bg-white border-r border-gray-200 flex flex-col">
    <div class="p-4 border-b border-gray-100">
      <h1 class="text-lg font-semibold text-gray-800">Agentic RAG</h1>
      <p class="text-xs text-gray-400 mt-0.5">智能问答平台</p>
    </div>
    <nav class="flex-1 p-3 space-y-1">
      <router-link
        v-for="item in navItems"
        :key="item.path"
        :to="item.path"
        class="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-gray-600 hover:bg-gray-100 transition-colors"
        active-class="bg-amber-50 text-amber-700 font-medium"
      >
        <span>{{ item.icon }}</span>
        <span>{{ item.label }}</span>
      </router-link>
    </nav>
    <div class="p-3 border-t border-gray-100">
      <div class="text-xs text-gray-400">Agentic RAG v0.1</div>
    </div>
  </aside>
</template>

<script setup lang="ts">
const navItems = [
  { path: '/', label: '对话', icon: '💬' },
  { path: '/knowledge', label: '知识库', icon: '📚' },
  { path: '/evaluation', label: '评估', icon: '📊' },
]
</script>
```

- [ ] **Step 3: 更新 App.vue**

```vue
<template>
  <div class="flex h-screen bg-gray-50">
    <AppSidebar />
    <main class="flex-1 overflow-hidden">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
import AppSidebar from './components/layout/AppSidebar.vue'
</script>
```

- [ ] **Step 4: 创建三个占位 View**

`ChatView.vue`:
```vue
<template>
  <div class="flex flex-col h-full">
    <header class="px-6 py-4 border-b border-gray-100 bg-white">
      <h2 class="text-lg font-semibold text-gray-800">对话</h2>
    </header>
    <!-- Chat messages area -->
    <div class="flex-1 overflow-y-auto p-6" ref="messagesContainer">
      <!-- Will be implemented in Task 7.2 -->
    </div>
    <!-- Chat input -->
    <div class="p-4 border-t border-gray-100 bg-white">
      <!-- Will be implemented in Task 7.2 -->
    </div>
  </div>
</template>
```

`KnowledgeView.vue` and `EvaluationView.vue` follow the same header + body pattern (full implementation deferred to their respective tasks).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/
git commit -m "feat: add frontend layout with sidebar navigation and route views"
```

---

### Task 7.2: ChatView — SSE 流式对话

**Files:**
- Create: `frontend/src/composables/useChat.ts`
- Create: `frontend/src/stores/chat.ts`
- Modify: `frontend/src/views/ChatView.vue`
- Create: `frontend/src/components/chat/ChatMessage.vue`
- Create: `frontend/src/components/chat/ChatInput.vue`
- Create: `frontend/src/components/chat/SourcePanel.vue`
- Create: `frontend/src/components/chat/AgentThinking.vue`

- [ ] **Step 1: 创建 useChat.ts**

```typescript
import { ref } from 'vue'
import type { ChatMessage } from '../types'

export function useChat() {
  const messages = ref<ChatMessage[]>([])
  const isStreaming = ref(false)
  const currentRoute = ref('')
  const streamingContent = ref('')

  async function sendMessage(question: string, history: any[], hasDocs: boolean) {
    messages.value.push({ role: 'user', content: question })
    isStreaming.value = true
    streamingContent.value = ''

    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history, has_docs: hasDocs }),
    })

    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6))
          if (data.status === 'thinking') {
            currentRoute.value = '思考中...'
          } else if (data.route) {
            currentRoute.value = data.route
          } else if (data.content) {
            streamingContent.value += data.content
          } else if (data.answer) {
            messages.value.push({
              role: 'assistant',
              content: data.answer,
              sourceType: data.route === 'web' ? 'web' : data.route === 'rag' ? 'doc' : undefined,
            })
          }
        }
      }
    }

    isStreaming.value = false
    currentRoute.value = ''
    streamingContent.value = ''
  }

  function clearMessages() {
    messages.value = []
  }

  return { messages, isStreaming, currentRoute, streamingContent, sendMessage, clearMessages }
}
```

- [ ] **Step 2: 创建 pinia store stores/chat.ts**

```typescript
import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { ChatMessage } from '../types'

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMessage[]>([])
  const conversationId = ref<number | null>(null)

  function addMessage(msg: ChatMessage) {
    messages.value.push(msg)
  }

  function clearMessages() {
    messages.value = []
    conversationId.value = null
  }

  return { messages, conversationId, addMessage, clearMessages }
})
```

- [ ] **Step 3: 创建 ChatMessage.vue**

```vue
<template>
  <div class="py-3" :class="message.role === 'user' ? 'flex justify-end' : 'flex justify-start'">
    <div
      class="max-w-[75%] rounded-xl px-4 py-3 text-sm leading-relaxed"
      :class="message.role === 'user'
        ? 'bg-amber-500 text-white'
        : 'bg-white border border-gray-200 text-gray-800'"
    >
      <div v-html="renderedContent"></div>
      <SourcePanel v-if="message.sources?.length" :sources="message.sources" :source-type="message.sourceType" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ChatMessage } from '../../types'
import SourcePanel from './SourcePanel.vue'

const props = defineProps<{ message: ChatMessage }>()

const renderedContent = computed(() => {
  return props.message.content.replace(/\n/g, '<br>')
})
</script>
```

- [ ] **Step 4: 创建 ChatInput.vue**

```vue
<template>
  <form @submit.prevent="submit" class="flex gap-3">
    <input
      v-model="input"
      type="text"
      placeholder="输入你的问题..."
      class="flex-1 rounded-xl border border-gray-200 px-4 py-3 text-sm focus:outline-none focus:border-amber-400 focus:ring-2 focus:ring-amber-100 transition-shadow"
      :disabled="disabled"
    />
    <button
      type="submit"
      :disabled="disabled || !input.trim()"
      class="px-6 py-3 bg-amber-500 text-white rounded-xl text-sm font-medium hover:bg-amber-600 disabled:opacity-40 transition-colors"
    >
      发送
    </button>
  </form>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ disabled: boolean }>()
const emit = defineEmits<{ send: [text: string] }>()

const input = ref('')

function submit() {
  if (!input.value.trim()) return
  emit('send', input.value.trim())
  input.value = ''
}
</script>
```

- [ ] **Step 5: 创建 SourcePanel.vue**

```vue
<template>
  <details class="mt-2">
    <summary class="text-xs text-gray-400 cursor-pointer hover:text-gray-500">
      来源链接 ({{ sources.length }} 条)
    </summary>
    <div class="mt-2 space-y-1">
      <a
        v-for="(source, i) in sources"
        :key="i"
        :href="source.link"
        target="_blank"
        class="block text-xs text-amber-600 hover:text-amber-700 truncate"
      >
        {{ source.title }}
      </a>
    </div>
  </details>
</template>

<script setup lang="ts">
import type { Source } from '../../types'

defineProps<{ sources: Source[]; sourceType?: string }>()
</script>
```

- [ ] **Step 6: 创建 AgentThinking.vue**

```vue
<template>
  <div v-if="isStreaming" class="flex items-center gap-2 px-4 py-2 text-xs text-gray-400">
    <span class="inline-block w-2 h-2 bg-amber-400 rounded-full animate-pulse"></span>
    <span>{{ label }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ route: string; isStreaming: boolean }>()

const label = computed(() => {
  if (!props.isStreaming) return ''
  const map: Record<string, string> = {
    'web': '正在联网搜索...',
    'rag': '正在检索文档...',
    'direct': '正在生成回答...',
  }
  return map[props.route] || '思考中...'
})
</script>
```

- [ ] **Step 7: 组装 ChatView.vue**

```vue
<template>
  <div class="flex flex-col h-full">
    <header class="px-6 py-4 border-b border-gray-100 bg-white flex items-center justify-between">
      <h2 class="text-lg font-semibold text-gray-800">对话</h2>
      <button @click="clearMessages" class="text-xs text-gray-400 hover:text-gray-600 transition-colors">清空对话</button>
    </header>

    <div class="flex-1 overflow-y-auto px-6 py-4" ref="msgContainer">
      <ChatMessage v-for="(msg, i) in messages" :key="i" :message="msg" />
      <AgentThinking :route="currentRoute" :is-streaming="isStreaming" />
      <div v-if="streamingContent" class="flex justify-start py-3">
        <div class="max-w-[75%] rounded-xl px-4 py-3 bg-white border border-gray-200 text-sm">
          {{ streamingContent }}
        </div>
      </div>
    </div>

    <div class="p-4 border-t border-gray-100 bg-white">
      <ChatInput :disabled="isStreaming" @send="handleSend" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { useChatStore } from '../stores/chat'
import { useChat } from '../composables/useChat'
import ChatMessage from '../components/chat/ChatMessage.vue'
import ChatInput from '../components/chat/ChatInput.vue'
import AgentThinking from '../components/chat/AgentThinking.vue'

const store = useChatStore()
const { messages, isStreaming, currentRoute, streamingContent, sendMessage, clearMessages } = useChat()
const msgContainer = ref<HTMLElement>()

watch(messages, async () => {
  await nextTick()
  if (msgContainer.value) {
    msgContainer.value.scrollTop = msgContainer.value.scrollHeight
  }
}, { deep: true })

async function handleSend(text: string) {
  const history = messages.value.slice(-6).map(m => ({ role: m.role, content: m.content }))
  await sendMessage(text, history, false)
}
</script>
```

- [ ] **Step 8: Commit**

```bash
git add frontend/src/
git commit -m "feat: add SSE streaming chat with Agent thinking visualization"
```

---

### Task 7.3: KnowledgeView — 知识库管理

**Files:**
- Create: `frontend/src/composables/useKnowledge.ts`
- Create: `frontend/src/stores/knowledge.ts`
- Modify: `frontend/src/views/KnowledgeView.vue`
- Create: `frontend/src/components/knowledge/FileUploader.vue`
- Create: `frontend/src/components/knowledge/DocList.vue`

- [ ] **Step 1: 创建 useKnowledge.ts**

```typescript
import { ref } from 'vue'
import type { DocumentRecord } from '../types'

export function useKnowledge() {
  const documents = ref<DocumentRecord[]>([])
  const isUploading = ref(false)

  async function upload(file: File) {
    isUploading.value = true
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch('/api/knowledge/upload', {
      method: 'POST',
      body: formData,
    })
    const result = await response.json()
    await fetchDocuments()
    isUploading.value = false
    return result
  }

  async function fetchDocuments() {
    const response = await fetch('/api/knowledge/documents')
    documents.value = await response.json()
  }

  return { documents, isUploading, upload, fetchDocuments }
}
```

- [ ] **Step 2: 创建 FileUploader.vue**

```vue
<template>
  <div
    class="border-2 border-dashed border-gray-200 rounded-xl p-8 text-center hover:border-amber-300 hover:bg-amber-50/50 transition-colors cursor-pointer"
    @dragover.prevent
    @drop.prevent="handleDrop"
    @click="triggerInput"
  >
    <input ref="fileInput" type="file" accept=".pdf,.txt" class="hidden" @change="handleFileChange" />
    <div class="text-3xl mb-2">📄</div>
    <p class="text-sm text-gray-500">拖拽 PDF 或 TXT 文件到此处，或点击上传</p>
    <p v-if="isUploading" class="text-xs text-amber-500 mt-2">上传中...</p>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ isUploading: boolean }>()
const emit = defineEmits<{ upload: [file: File] }>()

const fileInput = ref<HTMLInputElement>()

function triggerInput() {
  fileInput.value?.click()
}

function handleFileChange(e: Event) {
  const target = e.target as HTMLInputElement
  if (target.files?.[0]) {
    emit('upload', target.files[0])
    target.value = ''
  }
}

function handleDrop(e: DragEvent) {
  if (e.dataTransfer?.files[0]) {
    emit('upload', e.dataTransfer.files[0])
  }
}
</script>
```

- [ ] **Step 3: 创建 DocList.vue**

```vue
<template>
  <div class="space-y-2">
    <div
      v-for="doc in documents"
      :key="doc.id"
      class="flex items-center justify-between px-4 py-3 bg-white border border-gray-100 rounded-lg"
    >
      <div>
        <p class="text-sm font-medium text-gray-700">{{ doc.filename }}</p>
        <p class="text-xs text-gray-400">{{ doc.chunk_count }} 个片段 · {{ formatDate(doc.created_at) }}</p>
      </div>
      <span class="text-xs px-2 py-1 bg-gray-100 rounded text-gray-500">{{ doc.file_type.toUpperCase() }}</span>
    </div>
    <div v-if="!documents.length" class="text-center py-12 text-sm text-gray-400">
      暂无文档，请上传 PDF 或 TXT 文件
    </div>
  </div>
</template>

<script setup lang="ts">
import type { DocumentRecord } from '../../types'

defineProps<{ documents: DocumentRecord[] }>()

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('zh-CN')
}
</script>
```

- [ ] **Step 4: 组装 KnowledgeView.vue**

```vue
<template>
  <div class="flex flex-col h-full">
    <header class="px-6 py-4 border-b border-gray-100 bg-white">
      <h2 class="text-lg font-semibold text-gray-800">知识库管理</h2>
    </header>
    <div class="flex-1 overflow-y-auto p-6">
      <FileUploader :is-uploading="isUploading" @upload="handleUpload" />
      <div class="mt-6">
        <h3 class="text-sm font-medium text-gray-500 mb-3">已上传文档</h3>
        <DocList :documents="documents" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useKnowledge } from '../composables/useKnowledge'
import FileUploader from '../components/knowledge/FileUploader.vue'
import DocList from '../components/knowledge/DocList.vue'

const { documents, isUploading, upload, fetchDocuments } = useKnowledge()

onMounted(() => fetchDocuments())

async function handleUpload(file: File) {
  await upload(file)
}
</script>
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/
git commit -m "feat: add knowledge base management with file upload and document list"
```

---

### Task 7.4: EvaluationView — 评估可视化

**Files:**
- Create: `frontend/src/composables/useEvaluation.ts`
- Modify: `frontend/src/views/EvaluationView.vue`
- Create: `frontend/src/components/evaluation/EvalChart.vue`

- [ ] **Step 1: 创建 useEvaluation.ts**

```typescript
import { ref } from 'vue'
import type { EvalResult } from '../types'

export function useEvaluation() {
  const results = ref<EvalResult[]>([])
  const isRunning = ref(false)
  const history = ref<any[]>([])

  async function run(queries: string[], hasDocs: boolean) {
    isRunning.value = true
    const response = await fetch('/api/evaluation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ queries, has_docs: hasDocs }),
    })
    const data = await response.json()
    results.value = data.results
    isRunning.value = false
  }

  async function fetchHistory() {
    const response = await fetch('/api/evaluation/history')
    history.value = await response.json()
  }

  return { results, isRunning, history, run, fetchHistory }
}
```

- [ ] **Step 2: 创建 EvalChart.vue**

```vue
<template>
  <div class="space-y-3">
    <div v-for="(result, i) in results" :key="i" class="bg-white border border-gray-100 rounded-lg p-4">
      <p class="text-sm font-medium text-gray-700 mb-1">Q: {{ result.query }}</p>
      <p class="text-xs text-gray-400 mb-3 line-clamp-2">A: {{ result.answer }}</p>
      <div class="flex gap-4">
        <div class="flex items-center gap-2">
          <span class="text-xs text-gray-400">忠实度</span>
          <div class="w-20 h-2 bg-gray-100 rounded-full overflow-hidden">
            <div class="h-full bg-emerald-500 rounded-full transition-all" :style="{ width: (result.faithfulness * 100) + '%' }"></div>
          </div>
          <span class="text-xs font-medium" :class="result.faithfulness >= 0.7 ? 'text-emerald-600' : 'text-red-500'">
            {{ (result.faithfulness * 100).toFixed(0) }}%
          </span>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-xs text-gray-400">相关性</span>
          <div class="w-20 h-2 bg-gray-100 rounded-full overflow-hidden">
            <div class="h-full bg-blue-500 rounded-full transition-all" :style="{ width: (result.answer_relevancy * 100) + '%' }"></div>
          </div>
          <span class="text-xs font-medium" :class="result.answer_relevancy >= 0.7 ? 'text-blue-600' : 'text-red-500'">
            {{ (result.answer_relevancy * 100).toFixed(0) }}%
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { EvalResult } from '../../types'

defineProps<{ results: EvalResult[] }>()
</script>
```

- [ ] **Step 3: 组装 EvaluationView.vue**

```vue
<template>
  <div class="flex flex-col h-full">
    <header class="px-6 py-4 border-b border-gray-100 bg-white">
      <h2 class="text-lg font-semibold text-gray-800">RAG 评估</h2>
    </header>
    <div class="flex-1 overflow-y-auto p-6">
      <div class="mb-6">
        <label class="block text-sm font-medium text-gray-600 mb-1">测试查询（每行一个）</label>
        <textarea
          v-model="queryText"
          rows="4"
          class="w-full border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-amber-400 resize-none"
          placeholder="输入测试查询，每行一个..."
        ></textarea>
        <button
          @click="handleRun"
          :disabled="isRunning || !queryText.trim()"
          class="mt-3 px-5 py-2.5 bg-amber-500 text-white rounded-xl text-sm font-medium hover:bg-amber-600 disabled:opacity-40 transition-colors"
        >
          {{ isRunning ? '评估中...' : '开始评估' }}
        </button>
      </div>

      <EvalChart v-if="results.length" :results="results" />

      <div v-if="history.length && !results.length" class="space-y-2">
        <h3 class="text-sm font-medium text-gray-500 mb-2">历史评估记录</h3>
        <div v-for="h in history" :key="h.id" class="bg-white border border-gray-100 rounded-lg p-3">
          <p class="text-xs text-gray-500 truncate">{{ h.query }}</p>
          <div class="flex gap-4 mt-1">
            <span class="text-xs text-gray-400">忠实度: {{ (h.faithfulness * 100).toFixed(0) }}%</span>
            <span class="text-xs text-gray-400">相关性: {{ (h.answer_relevancy * 100).toFixed(0) }}%</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useEvaluation } from '../composables/useEvaluation'
import EvalChart from '../components/evaluation/EvalChart.vue'

const { results, isRunning, history, run, fetchHistory } = useEvaluation()
const queryText = ref('')

onMounted(() => fetchHistory())

async function handleRun() {
  const queries = queryText.value.split('\n').filter(q => q.trim())
  if (!queries.length) return
  await run(queries, false)
}
</script>
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/
git commit -m "feat: add RAG evaluation view with scoring visualization"
```

---

### Task 7.5: 前端样式完善 + Tailwind 配置

**Files:**
- Create: `frontend/src/styles/main.css`

- [ ] **Step 1: 创建 main.css**

```css
@import "tailwindcss";

@layer base {
  body {
    font-family: 'Inter', 'Noto Sans SC', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
}

.line-clamp-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 5px;
  height: 5px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: #e5e7eb;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: #d1d5db;
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/styles/main.css
git commit -m "feat: add Tailwind CSS base styles and custom scrollbar"
```

---

## Phase 8: Docker 部署 + 集成验证

### Task 8.1: 端到端集成验证

- [ ] **Step 1: 启动所有服务**

```bash
docker compose up --build
```

- [ ] **Step 2: 验证前端可访问**

浏览器打开 `http://localhost:3000`，确认：
- 侧边栏三个导航工作正常
- 对话页面可以输入问题并收到 SSE 流式回答
- 知识库页面可以上传 PDF 文件并查看列表
- 评估页面可以运行评估并查看结果

- [ ] **Step 3: 验证 API 端点**

```bash
# Health check
curl http://localhost:8000/api/health

# List tools
curl http://localhost:8000/api/tools/

# Upload document
curl -X POST http://localhost:8000/api/knowledge/upload -F "file=@test.pdf"

# Chat stream
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "你好", "history": [], "has_docs": false}'
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: finalize Docker Compose integration and document verification steps"
```

---

## 任务总览

| Phase | Task | 内容 | 涉及文件数 |
|-------|------|------|------------|
| 1 | 1.1 | 后端骨架 | 5 |
| 1 | 1.2 | 前端初始化 | 8+ |
| 1 | 1.3 | Docker Compose + MySQL | 5 |
| 2 | 2.1 | SQLAlchemy 连接 | 3 |
| 2 | 2.2 | ORM 模型 | 4 |
| 3 | 3.1 | 文档加载器 | 2 |
| 3 | 3.2 | ChromaDB | 1 |
| 3 | 3.3 | BM25 索引 | 1 |
| 3 | 3.4 | 混合检索 + RRF | 3 |
| 3 | 3.5 | Reranker | 2 |
| 4 | 4.1 | MCP 工具注册中心 | 2 |
| 4 | 4.2 | 工具实现 | 3 |
| 4 | 4.3 | Agent State + Nodes | 2 |
| 4 | 4.4 | LangGraph 图 | 2 |
| 5 | 5.1 | Chat SSE 端点 | 4 |
| 5 | 5.2 | 知识库 API | 3 |
| 5 | 5.3 | MCP 工具 API | 1 |
| 6 | 6.1 | RAGAS 评估 | 2 |
| 7 | 7.1 | 前端布局 | 6 |
| 7 | 7.2 | ChatView | 7 |
| 7 | 7.3 | KnowledgeView | 5 |
| 7 | 7.4 | EvaluationView | 3 |
| 7 | 7.5 | 样式完善 | 1 |
| 8 | 8.1 | 集成验证 | - |

**总计: 约 25 个 Task, 70+ 个文件**

---

## 执行顺序

Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8

同 Phase 内 Task 顺序执行。Phase 7（前端）可在 Phase 3 完成后部分并行启动，因为 API 已基本成型。

---

## 关键技术决策记录

1. **MCP 协议**: 采用 MCP 接口设计思想（JSON Schema 定义 + 注册中心），不引入完整 MCP Server 框架，保持轻量
2. **流式输出**: 使用 `sse-starlette` 实现 SSE，而非 WebSocket，因为对话场景是单向推送
3. **BM25 持久化**: 使用 pickle 序列化到磁盘，避免引入 Elasticsearch 等重型组件
4. **Reranker**: 首次运行下载 ~1GB 模型，后续使用本地缓存
5. **评估简化**: 使用 LLM-as-Judge 方式计算 faithfulnes 和 relevancy，而非完整 RAGAS 数据集评估流程，保证可用性
