import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.base import Base
from app.db.session import engine
from app.api.chat import router as chat_router
from app.api.knowledge import router as knowledge_router
from app.api.tools import router as tools_router
from app.api.evaluation import router as eval_router
from app.api.conversations import router as conversations_router

import app.models.conversation  # noqa: F401
import app.models.message       # noqa: F401
import app.models.document      # noqa: F401
import app.models.evaluation    # noqa: F401

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
        try:
            await asyncio.to_thread(_preload_retrieval)
        except Exception:
            # 预载只是延迟优化，绝不能用它换掉可用性：模型缺失、pickle 损坏都只该让
            # 首次检索付代价。单例保持 None，懒加载路径照常工作。
            logger.exception("检索链路预载失败，将在首次请求时重试")
        else:
            logger.info("检索链路预载完成，耗时 %.2fs", time.perf_counter() - started)
    yield


app = FastAPI(title="Agentic RAG API", version="0.1.0", lifespan=lifespan)

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
