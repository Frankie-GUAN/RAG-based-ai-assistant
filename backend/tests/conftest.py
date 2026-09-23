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
