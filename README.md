# RAG 智能问答机器人

基于 LangChain + Google Gemini + ChromaDB + SerpAPI + Streamlit 的联网搜索增强 RAG 智能问答机器人。

## 环境要求

- Python 3.10+

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境变量配置

请在系统环境变量中配置 Gemini 密钥（必须）：

- GEMINI_API_KEY

如需联网搜索，请在应用侧边栏输入 SerpAPI_KEY（无需写入环境变量）。

示例（PowerShell）：

```powershell
setx GEMINI_API_KEY "你的Gemini密钥"
```

设置后重新打开终端再运行程序。

## 运行方式

```bash
streamlit run app.py
```

## 使用说明

1. 侧边栏上传 PDF/TXT 文档并点击“构建/更新知识库”。
2. 如需联网搜索，在侧边栏输入 SerpAPI_KEY。
3. 在主界面输入问题开始对话。

## 目录结构

```
.
├─ app.py
├─ requirements.txt
├─ README.md
└─ src
   ├─ __init__.py
   ├─ chat_chain.py
   ├─ config.py
   ├─ document_loader.py
   ├─ rag_retriever.py
   ├─ vector_store.py
   └─ web_search.py
```
