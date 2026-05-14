import os
from pathlib import Path

import streamlit as st

try:
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


st.set_page_config(page_title="RAG 智能问答机器人", page_icon="◈", layout="wide")

st.markdown(
    """
<style>
/* ============================================================
   RAG 智能问答机器人 — 「Editorial Refined」
   温暖克制 · 编辑级排版 · Claude 风格
   ============================================================ */

/* ---------- 字体 ---------- */
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400;1,6..72,500&family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Noto+Sans+SC:wght@300;400;500;600;700&family=Noto+Serif+SC:wght@400;500;600;700&display=swap');

/* ---------- CSS 变量 ---------- */
:root {
    --color-bg:              #fafaf7;
    --color-surface:         #ffffff;
    --color-surface-muted:   #f5f4f0;
    --color-surface-warm:    #fefbf6;
    --color-border:          #e8e6e1;
    --color-border-light:    #f0ede8;
    --color-text:            #1a1a18;
    --color-text-secondary:  #6b6b63;
    --color-text-muted:      #9d9d94;
    --color-accent:          #d97706;
    --color-accent-soft:     #fffbeb;
    --color-accent-hover:    #b45309;
    --color-accent-muted:    #fef3c7;
    --color-user-bg:         #faf9f5;
    --color-user-border:     #e8e6d8;
    --font-display:          "Newsreader", "Noto Serif SC", serif;
    --font-body:             "Plus Jakarta Sans", "Noto Sans SC", sans-serif;
    --font-mono:             "Plus Jakarta Sans", "Courier New", monospace;
    --radius-sm:             6px;
    --radius-md:             10px;
    --radius-lg:             16px;
    --radius-xl:             24px;
    --shadow-xs:             0 1px 2px rgba(0,0,0,0.03);
    --shadow-sm:             0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03);
    --shadow-md:             0 2px 8px rgba(0,0,0,0.05);
    --shadow-lg:             0 4px 20px rgba(0,0,0,0.06);
    --transition-fast:       180ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-smooth:     280ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-spring:     400ms cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* ---------- 全局 ---------- */
.stApp {
    background: var(--color-bg);
    color: var(--color-text);
    font-family: var(--font-body);
    font-weight: 400;
    font-size: 0.94rem;
    letter-spacing: 0.005em;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* 极淡肌理覆盖 */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 9998;
    opacity: 0.025;
    background-image: url("data:image/svg+xml,%3Csvg width='200' height='200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* ---------- 主内容区 ---------- */
.block-container {
    padding: 1.6rem 2rem 2.5rem;
    max-width: min(740px, 100%);
    margin: 0 auto;
}

/* ---------- 侧边栏 ---------- */
[data-testid="stSidebar"] {
    background: var(--color-surface);
    border-right: 1px solid var(--color-border-light);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    font-family: var(--font-body);
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--color-text);
    letter-spacing: 0.01em;
}

[data-testid="stSidebar"] label {
    font-family: var(--font-body);
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--color-text-secondary);
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    font-family: var(--font-body);
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    padding: 0.5rem 0.9rem;
    border-radius: var(--radius-md);
    transition: all var(--transition-fast);
}

[data-testid="stSidebar"] hr,
[data-testid="stSidebar"] [data-testid="stDivider"] {
    border-color: var(--color-border-light);
}

/* ---------- Hero ---------- */
.hero {
    background: var(--color-surface);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-lg);
    padding: clamp(1.4rem, 2.8vw, 1.8rem) clamp(1.6rem, 3.2vw, 2.2rem);
    margin-bottom: 1.6rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow var(--transition-smooth);
}

.hero:hover {
    box-shadow: var(--shadow-md);
}

.hero-title {
    font-family: var(--font-display);
    font-size: clamp(1.5rem, 3vw, 2rem);
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--color-text);
    line-height: 1.25;
}

.hero-subtitle {
    font-family: var(--font-body);
    color: var(--color-text-secondary);
    margin-top: 0.4rem;
    font-size: clamp(0.85rem, 1.4vw, 0.94rem);
    line-height: 1.55;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* ---------- 聊天气泡 ---------- */
div[data-testid="stChatMessage"] {
    display: flex;
    padding: 0.16rem 0;
    animation: msgReveal 340ms cubic-bezier(0.22, 0.61, 0.36, 1) both;
}

div[data-testid="stChatMessage"] > div {
    border-radius: var(--radius-md);
    padding: 0.8rem 1.15rem;
    border: 1px solid var(--color-border-light);
    background: var(--color-surface);
    box-shadow: var(--shadow-xs);
    transition: box-shadow var(--transition-fast);
}

div[data-testid="stChatMessage"] > div:hover {
    box-shadow: var(--shadow-sm);
}

/* 助手消息 */
div[data-testid="stChatMessage"][data-message-author-role="assistant"] {
    justify-content: flex-start;
}
div[data-testid="stChatMessage"][data-message-author-role="assistant"] > div {
    background: var(--color-surface);
    border-left: 3px solid var(--color-accent);
}

/* 用户消息 */
div[data-testid="stChatMessage"][data-message-author-role="user"] {
    justify-content: flex-end;
}
div[data-testid="stChatMessage"][data-message-author-role="user"] > div {
    background: var(--color-user-bg);
    border: 1px solid var(--color-user-border);
    border-right: 3px solid var(--color-border);
}

/* ---------- 输入框 ---------- */
div[data-testid="stChatInput"] textarea {
    border: 1.5px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-surface);
    color: var(--color-text);
    font-family: var(--font-body);
    font-size: 0.92rem;
    padding: 0.65rem 1rem;
    transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
    caret-color: var(--color-accent);
    box-shadow: var(--shadow-xs);
}

div[data-testid="stChatInput"] textarea:focus {
    border-color: var(--color-accent);
    box-shadow: 0 0 0 3px rgba(217, 119, 6, 0.1);
    outline: none;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: var(--color-text-muted);
    font-family: var(--font-body);
    font-size: 0.88rem;
}

/* ---------- 按钮 ---------- */
.stButton > button {
    background: var(--color-accent);
    color: #ffffff;
    border: none;
    border-radius: var(--radius-md);
    padding: 0.5rem 1.2rem;
    font-family: var(--font-body);
    font-weight: 500;
    font-size: 0.84rem;
    letter-spacing: 0.02em;
    box-shadow: var(--shadow-xs);
    transition: all var(--transition-fast);
    cursor: pointer;
}

.stButton > button:hover {
    background: var(--color-accent-hover);
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}

.stButton > button:active {
    background: var(--color-accent-hover);
    transform: translateY(0);
    box-shadow: var(--shadow-xs);
}

/* 侧边栏次要按钮 */
[data-testid="stSidebar"] .stButton:last-of-type > button {
    background: transparent;
    color: var(--color-text-secondary);
    border: 1.5px solid var(--color-border);
    box-shadow: none;
}

[data-testid="stSidebar"] .stButton:last-of-type > button:hover {
    background: var(--color-surface-muted);
    border-color: var(--color-text-muted);
    color: var(--color-text);
    transform: none;
    box-shadow: none;
}

/* ---------- 文件上传 ---------- */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-surface-muted);
    transition: border-color var(--transition-fast), background var(--transition-fast);
    padding: 1rem;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--color-accent);
    border-style: solid;
    background: var(--color-accent-soft);
}

/* ---------- Spinner ---------- */
[data-testid="stSpinner"] {
    color: var(--color-accent);
}

/* ---------- 通知 ---------- */
[data-testid="stNotification"] {
    border-radius: var(--radius-md);
    font-size: 0.86rem;
    font-family: var(--font-body);
    border-left: 3px solid currentColor;
}

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
    border: none;
    border-left: 2px solid var(--color-border-light);
    border-radius: 0;
}

[data-testid="stExpander"]:hover {
    border-left-color: var(--color-accent);
}

[data-testid="stExpander"] summary {
    font-family: var(--font-body);
    font-size: 0.84rem;
    font-weight: 500;
    color: var(--color-text-secondary);
}

/* ---------- Caption ---------- */
.stCaption {
    color: var(--color-text-muted);
    font-family: var(--font-body);
    font-size: 0.8rem;
    font-style: italic;
}

/* ---------- 滚动条 ---------- */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: var(--color-border);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--color-text-muted);
}

/* ---------- Markdown 内容 ---------- */
[data-testid="stChatMessage"] p {
    margin: 0.15rem 0;
    line-height: 1.75;
    font-family: var(--font-body);
    font-size: 0.92rem;
}

[data-testid="stChatMessage"] strong {
    font-weight: 600;
    color: var(--color-text);
}

[data-testid="stChatMessage"] a {
    color: var(--color-accent);
    text-decoration: none;
    border-bottom: 1px solid rgba(217, 119, 6, 0.25);
    transition: border-color var(--transition-fast);
}

[data-testid="stChatMessage"] a:hover {
    border-bottom-color: var(--color-accent);
}

[data-testid="stChatMessage"] code {
    background: var(--color-accent-soft);
    border-radius: var(--radius-sm);
    padding: 0.1em 0.45em;
    font-family: var(--font-mono);
    font-size: 0.85em;
    color: var(--color-accent-hover);
    font-weight: 500;
}

[data-testid="stChatMessage"] pre {
    border-radius: var(--radius-md);
    border: 1px solid var(--color-border-light);
    background: var(--color-surface-muted);
    padding: 0.8rem 1rem;
    overflow-x: auto;
}

/* ---------- 侧边栏文本输入 ---------- */
div[data-testid="stTextInput"] input {
    border: 1.5px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-surface);
    color: var(--color-text);
    font-family: var(--font-body);
    font-size: 0.84rem;
    padding: 0.5rem 0.7rem;
    transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
    caret-color: var(--color-accent);
}

div[data-testid="stTextInput"] input:focus {
    border-color: var(--color-accent);
    box-shadow: 0 0 0 3px rgba(217, 119, 6, 0.1);
    outline: none;
}

div[data-testid="stTextInput"] input::placeholder {
    color: var(--color-text-muted);
}

/* ---------- 分隔线 ---------- */
hr,
[data-testid="stDivider"] {
    border-color: var(--color-border-light);
}

/* ============================================================
   动画
   ============================================================ */

@keyframes msgReveal {
    from {
        opacity: 0;
        transform: translateY(6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ============================================================
   响应式
   ============================================================ */

@media screen and (max-width: 640px) {
    .block-container {
        padding: 0.7rem 0.9rem 1.5rem;
        max-width: 100%;
    }

    div[data-testid="stChatMessage"] > div {
        max-width: 90% !important;
        padding: 0.6rem 0.8rem;
    }

    .hero {
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
    }

    .stButton > button {
        width: 100%;
        padding: 0.5rem 0.9rem;
    }

    div[data-testid="stChatInput"] textarea {
        font-size: 0.86rem;
        padding: 0.55rem 0.8rem;
    }
}

@media screen and (min-width: 641px) and (max-width: 1024px) {
    .block-container {
        padding: 1.2rem 1.5rem 2rem;
        max-width: 92%;
    }

    div[data-testid="stChatMessage"] > div {
        max-width: 82%;
    }
}

@media screen and (min-width: 1025px) {
    div[data-testid="stChatMessage"] > div {
        max-width: 70%;
    }
}

@media screen and (min-width: 1600px) {
    .block-container {
        max-width: 760px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state() -> None:
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
    if st.session_state.vectorstore is None and has_persisted_index():
        embeddings = get_embeddings()
        st.session_state.vectorstore = load_vectorstore(embeddings)
        st.session_state.has_docs = st.session_state.vectorstore is not None


def render_web_sources(items) -> None:
    if not items:
        return
    with st.expander(f"来源链接（{len(items)} 条）"):
        for item in items:
            title = item.get("title") or item.get("link")
            link = item.get("link")
            if link:
                st.markdown(f"- [{title}]({link})")


def render_doc_sources(docs) -> None:
    if not docs:
        return
    with st.expander(f"文档引用（{len(docs)} 条）"):
        for doc in docs:
            source = doc.metadata.get("source") or "文档"
            page = doc.metadata.get("page")
            label = Path(source).name if source else "文档"
            if page is not None:
                label = f"{label} — 第 {page + 1} 页"
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = f"{snippet[:200]}..."
            st.markdown(f"- **{label}**")
            st.caption(snippet)


init_session_state()

if not os.getenv(GEMINI_API_KEY_ENV):
    st.error("未检测到 GEMINI_API_KEY 环境变量，请先配置后再启动应用。")
    st.stop()

ensure_vectorstore()

with st.sidebar:
    st.markdown("### 配置")

    with st.expander("API 密钥", expanded=True):
        serpapi_key = st.text_input(
            "SerpAPI_KEY",
            type="password",
            placeholder="输入 SerpAPI 密钥以启用联网搜索",
        )

    with st.expander("知识库管理", expanded=True):
        uploaded_files = st.file_uploader(
            "上传 PDF / TXT 文档",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="支持上传多个 PDF 或 TXT 文件，系统会自动分词并构建向量索引",
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("构建知识库", use_container_width=True):
                if not uploaded_files:
                    st.warning("请先上传 PDF 或 TXT 文件")
                else:
                    with st.spinner("正在处理文档..."):
                        saved_paths = save_uploaded_files(uploaded_files)
                        documents = load_documents(saved_paths)
                        chunks = split_documents(documents)
                        embeddings = get_embeddings()
                        st.session_state.vectorstore = add_documents(chunks, embeddings)
                        st.session_state.has_docs = True
                    st.success("知识库更新完成")
                    st.rerun()
        with col2:
            if st.button("清空对话", use_container_width=True):
                st.session_state.messages = []
                st.session_state.memory.clear()
                st.success("对话已清空")
                st.rerun()

    st.divider()
    if st.session_state.has_docs:
        st.caption("知识库已就绪")
    else:
        st.caption("暂无本地知识库")

st.markdown(
    """
<div class="hero">
    <div class="hero-title">RAG 智能问答机器人</div>
    <div class="hero-subtitle">私有文档 · 实时搜索 · 多轮对话 — 一站式智能问答</div>
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

if prompt := st.chat_input("输入你的问题..."):
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
