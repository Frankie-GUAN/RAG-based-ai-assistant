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
