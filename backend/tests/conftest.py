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
    收集期抛 ScopeMismatch**（Task 7 很可能会写那种 fixture），而依赖插件原版则正常 ——
    等于用一个「固定后端」的小便利换掉了 module 作用域的可组合性。

    另外要清楚：插件本身已提供这个 fixture，参数取自 get_available_backends()。
    这里显式覆盖的唯一作用是固定为 asyncio —— 将来若装了 trio，也不会让每个 async
    测试参数化翻倍。当前环境未装 trio，所以这层覆盖在今天是行为中性的。
    """
    return "asyncio"
