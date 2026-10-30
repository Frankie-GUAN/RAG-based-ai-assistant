import pytest
from langchain_core.documents import Document

from app.config import settings


@pytest.fixture(autouse=True)
def _disable_model_preload(monkeypatch):
    """关掉 lifespan 里的检索链路预载。

    `with TestClient(app)` 会跑 lifespan，预载要加载约 2GB 的 BGE-M3；
    一个只想测接口的用例不该为此付几分钟。想测预载本身的用例自己覆盖这个值即可。
    """
    monkeypatch.setattr(settings, "preload_models", False)


@pytest.fixture(autouse=True)
def _stub_query_embedding(monkeypatch):
    """把 query 编码换成固定向量，**不让测试碰真实模型**。

    **为什么需要。** 检索链路现在用「预计算的 query 向量」做检索
    （见 app/rag/query_cache.py，那是性能优化的关键）。
    于是每个走检索的用例都会去调 BGE-M3 编码一条 query——
    CPU 上要好几秒，而且让单元测试依赖一个 2GB 的模型文件。

    桩函数返回**确定性**的固定向量（由 query 文本哈希生成），
    这样：
    - 同一个 query 永远得到同一个向量（缓存语义可测）
    - 不同 query 得到不同向量（不会让所有检索结果都一样）
    - 完全不碰模型
    """
    import hashlib

    from app.rag import hybrid_search as hybrid_mod

    def _fake_embed_query(query: str):
        digest = hashlib.sha256(query.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:1024].ljust(1024, b"\x00")]

    monkeypatch.setattr(hybrid_mod, "cached_embed_query", _fake_embed_query, raising=False)


@pytest.fixture(autouse=True)
def _disable_query_rewrite(monkeypatch):
    """默认关掉跨语言 query 改写。

    **为什么。** 改写要扫语料、嵌术语、跑 BM25 校验——单元测试不该为此
    加载真实语料与模型。需要测改写本身的用例自己把它打开。

    这条也顺带保证了「检索层测试」测的是纯检索行为，
    不会被改写悄悄改变结果（那会让断言变得难以理解）。
    """
    monkeypatch.setattr(settings, "enable_query_rewrite", False, raising=False)


@pytest.fixture(autouse=True)
def _disable_memory_layers(monkeypatch):
    """默认关掉语义缓存与长期记忆。

    **为什么必须关。** 这两个功能一开就会去连真实依赖：
    · 语义缓存 → MySQL（`semantic_cache_entries` 表）+ Chroma collection
    · 长期记忆 → MySQL（`user_memories` 表）+ Chroma collection + embedding 模型

    不关的话，每个走 `stream_agent` 的用例都会尝试连库——
    连不上时异常被 `except` 吞掉、测试仍然「通过」，但那只是把
    「依赖缺失」伪装成了「功能正常」，比直接失败更难发现。

    需要测这两层的用例自己打开开关，并注入替身。
    """
    monkeypatch.setattr(settings, "enable_semantic_cache", False, raising=False)
    monkeypatch.setattr(settings, "enable_long_term_memory", False, raising=False)


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="劳动合同法第三十六条规定用人单位与劳动者协商一致可以解除劳动合同", metadata={"source": "labor_law.pdf"}),
        Document(page_content="2025年新修订的劳动法增加了远程办公相关条款", metadata={"source": "amendments.pdf"}),
        Document(page_content="Python是一种广泛使用的高级编程语言", metadata={"source": "python.pdf"}),
        Document(page_content="劳动者提前三十日书面通知用人单位可以解除劳动合同", metadata={"source": "labor_law.pdf"}),
        Document(page_content="FastAPI是现代Python Web框架支持异步处理", metadata={"source": "python.pdf"}),
    ]


@pytest.fixture
def sqlite_session():
    """独立的内存 SQLite 会话。

    用于验证依赖 ORM 配置（而非 MySQL 具体行为）的逻辑，让这类测试在本地裸跑
    `pytest` 时就有覆盖（仓库没有 CI；其余测试分别依赖真实 MySQL、真实 API key、
    2GB 模型）。

    两个必须讲清楚的坑：

    · `:memory:` 默认走 SingletonThreadPool，**每个线程一个连接、也就是一个独立的空库**。
      摘要路径经 `loop.run_in_executor` 跑在工作线程里（Task 9 要用这个 fixture 测它），
      所以必须换 StaticPool 并放开 check_same_thread，否则工作线程看到的是一个没有表的库。
    · `Base.metadata` 是**进程级**的：不显式给 tables，建出来的表取决于那一刻导入了哪些
      模型 —— 单独跑是 conversations+messages，一旦 `app.main` 被导入就变成四张表。
      显式列出这两张，schema 才不会随收集顺序漂移。
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db.base import Base
    from app.models.conversation import Conversation
    from app.models.message import Message

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine, tables=[Conversation.__table__, Message.__table__])
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture(scope="module")
def anyio_backend():
    """让 @pytest.mark.anyio 测试只跑在 asyncio 上。

    作用域必须与 anyio 插件自带的 fixture 一致（module）。插件把它声明为 module
    作用域，若这里覆盖成 function，那么**任何 module 作用域的 fixture 依赖它都会在
    setup 期抛 ScopeMismatch**（pytest 把这类错误报成 ERROR at setup，收集本身不中断；
    Task 7 很可能会写那种 fixture），而依赖插件原版则正常 ——
    等于用一个「固定后端」的小便利换掉了 module 作用域的可组合性。

    另外要清楚：插件本身已提供这个 fixture，参数取自 get_available_backends()。
    这里显式覆盖的唯一作用是固定为 asyncio —— 将来若装了 trio，也不会让每个 async
    测试参数化翻倍。当前环境未装 trio，所以这层覆盖在今天是行为中性的。
    """
    return "asyncio"
