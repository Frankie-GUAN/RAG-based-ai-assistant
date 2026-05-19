from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
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
