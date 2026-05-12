import os
from pathlib import Path

import streamlit as st

try:
    # 兼容 LangChain 新旧版本的 Memory 位置
    from langchain.memory import ConversationBufferMemory
except ModuleNotFoundError:
    from langchain_classic.memory import ConversationBufferMemory

from src.chat_chain import (
    answer_direct,
    answer_with_rag,
    answer_with_web,
    get_llm,
    route_question,
)
from src.config import GEMINI_API_KEY_ENV
from src.document_loader import load_documents, save_uploaded_files, split_documents
from src.rag_retriever import build_retriever, format_documents, retrieve_documents
from src.vector_store import add_documents, get_embeddings, has_persisted_index, load_vectorstore
from src.web_search import search_web


st.set_page_config(page_title="智能问答机器人", page_icon="🤖", layout="wide")

st.markdown(
        """
<style>
/* ====== 主题与颜色（灰白、扁平） ====== */
:root {
    --bg: #f7f7f8;
    --panel: #ffffff;
    --panel-muted: #f2f3f5;
    --border: #e5e7eb;
    --text: #111827;
    --muted: #6b7280;
    --accent: #10a37f;
    --user-bubble: #e9eef6;
}

/* ====== 全局背景与布局 ====== */
.stApp {
    background: var(--bg);
    color: var(--text);
}

.block-container {
    padding-top: 1.6rem;
    max-width: 840px;
}

/* ====== 顶部标题卡片 ====== */
.hero {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 6px 16px rgba(17, 24, 39, 0.06);
    margin-bottom: 1rem;
}

.hero-title {
    font-family: "Inter", "Noto Sans SC", sans-serif;
    font-size: 1.6rem;
    font-weight: 600;
}

.hero-subtitle {
    color: var(--muted);
    margin-top: 0.3rem;
    font-size: 0.92rem;
}

/* ====== 侧边栏 ====== */
section[data-testid="stSidebar"] > div {
    background: var(--panel);
    border-right: 1px solid var(--border);
}

/* ====== 聊天气泡 ====== */
div[data-testid="stChatMessage"] {
    display: flex;
    padding: 0.2rem 0;
}

div[data-testid="stChatMessage"] > div {
    max-width: 76%;
    border-radius: 12px;
    padding: 0.65rem 0.85rem;
    border: 1px solid var(--border);
    background: var(--panel);
    box-shadow: 0 4px 10px rgba(17, 24, 39, 0.06);
}

div[data-testid="stChatMessage"][data-message-author-role="assistant"] {
    justify-content: flex-start;
}

div[data-testid="stChatMessage"][data-message-author-role="assistant"] > div {
    background: var(--panel);
}

div[data-testid="stChatMessage"][data-message-author-role="user"] {
    justify-content: flex-end;
}

div[data-testid="stChatMessage"][data-message-author-role="user"] > div {
    background: var(--user-bubble);
    border-color: #dbe2ea;
}

/* ====== 输入框与按钮 ====== */
div[data-testid="stChatInput"] textarea,
div[data-testid="stTextInput"] input {
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--text);
}

div[data-testid="stChatInput"] textarea:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: #c7d2fe;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12);
}

.stButton > button {
    background: var(--accent);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 0.9rem;
    font-weight: 600;
    box-shadow: 0 6px 14px rgba(16, 163, 127, 0.2);
}

/* ====== 上传组件 ====== */
div[data-testid="stFileUploaderDropzone"] {
    border-radius: 10px;
    border: 1px dashed #d1d5db;
    background: var(--panel-muted);
}

/* ====== 滚动条 ====== */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #d1d5db;
    border-radius: 8px;
}
</style>
""",
        unsafe_allow_html=True,
)


def init_session_state() -> None:
    # 初始化对话状态与缓存
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=False,
        )
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "has_docs" not in st.session_state:
        st.session_state.has_docs = False


def ensure_vectorstore() -> None:
    # 尝试加载已持久化的本地知识库
    if st.session_state.vectorstore is None and has_persisted_index():
        embeddings = get_embeddings()
        st.session_state.vectorstore = load_vectorstore(embeddings)
        st.session_state.has_docs = st.session_state.vectorstore is not None


def render_web_sources(items) -> None:
    if not items:
        return
    st.markdown("**来源链接：**")
    for item in items:
        title = item.get("title") or item.get("link")
        link = item.get("link")
        if link:
            st.markdown(f"- [{title}]({link})")


def render_doc_sources(docs) -> None:
    if not docs:
        return
    st.markdown("**文档引用：**")
    for doc in docs:
        source = doc.metadata.get("source") or "文档"
        page = doc.metadata.get("page")
        label = Path(source).name if source else "文档"
        if page is not None:
            label = f"{label} - 第{page + 1}页"
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = f"{snippet[:200]}..."
        st.markdown(f"- {label}: {snippet}")


init_session_state()

if not os.getenv(GEMINI_API_KEY_ENV):
    st.error("未检测到 GEMINI_API_KEY 环境变量，请先配置后再启动应用。")
    st.stop()

ensure_vectorstore()

with st.sidebar:
    st.header("配置")
    serpapi_key = st.text_input("SerpAPI_KEY", type="password")
    uploaded_files = st.file_uploader(
        "上传 PDF/TXT 文档",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("构建/更新知识库"):
        if not uploaded_files:
            st.warning("请先上传 PDF 或 TXT 文件")
        else:
            with st.spinner("正在处理文档并写入向量库..."):
                saved_paths = save_uploaded_files(uploaded_files)
                documents = load_documents(saved_paths)
                chunks = split_documents(documents)
                embeddings = get_embeddings()
                st.session_state.vectorstore = add_documents(chunks, embeddings)
                st.session_state.has_docs = True
            st.success("知识库更新完成")

    if st.button("清空对话"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.success("对话已清空")

    if st.session_state.has_docs:
        st.caption("已加载本地知识库")
    else:
        st.caption("暂无本地知识库")

st.markdown(
        """
<div class="hero">
    <div class="hero-title">RAG 智能问答机器人</div>
    <div class="hero-subtitle">私有文档 + 实时搜索 + 多轮对话，一站式智能问答</div>
</div>
""",
        unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            if message.get("source_type") == "web":
                render_web_sources(message["sources"])
            elif message.get("source_type") == "doc":
                render_doc_sources(message["sources"])

if prompt := st.chat_input("请输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            llm = get_llm()
            history = st.session_state.memory.load_memory_variables({}).get("history", "")
            route = route_question(llm, prompt, history, st.session_state.has_docs)

            sources = []
            source_type = None

            if route == "web":
                try:
                    items, web_context = search_web(prompt, serpapi_key)
                    answer = answer_with_web(llm, prompt, history, web_context)
                    sources = items
                    source_type = "web"
                except ValueError:
                    answer = answer_direct(llm, prompt, history)
                    st.warning("未填写 SerpAPI_KEY，已改为直接回答")
            elif route == "rag" and st.session_state.vectorstore:
                retriever = build_retriever(st.session_state.vectorstore)
                docs = retrieve_documents(retriever, prompt)
                context = format_documents(docs)
                answer = answer_with_rag(llm, prompt, history, context)
                sources = docs
                source_type = "doc"
            else:
                answer = answer_direct(llm, prompt, history)

            st.markdown(answer)
            if source_type == "web":
                render_web_sources(sources)
            elif source_type == "doc":
                render_doc_sources(sources)

    st.session_state.memory.save_context({"input": prompt}, {"output": answer})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "source_type": source_type,
        }
    )
