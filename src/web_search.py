from __future__ import annotations

import os
from typing import Dict, List, Tuple

from langchain_community.utilities import SerpAPIWrapper

from .config import SERPAPI_API_KEY_ENV


def search_web(query: str, serpapi_key: str, top_k: int = 5) -> Tuple[List[Dict[str, str]], str]:
    # 使用 SerpAPI 进行联网检索，返回结构化结果与可读上下文
    if not serpapi_key:
        raise ValueError("SerpAPI_KEY 不能为空")

    os.environ[SERPAPI_API_KEY_ENV] = serpapi_key
    wrapper = SerpAPIWrapper(params={"num": top_k, "engine": "google", "hl": "zh-cn"})
    results = wrapper.results(query)

    items: List[Dict[str, str]] = []
    for item in results.get("organic_results", []):
        link = item.get("link")
        title = item.get("title")
        snippet = item.get("snippet")
        if not snippet:
            snippet = item.get("snippet_highlighted_words")
        if link:
            items.append(
                {
                    "title": title or link,
                    "link": link,
                    "snippet": str(snippet or ""),
                }
            )

    context_lines: List[str] = []
    for idx, item in enumerate(items, start=1):
        context_lines.append(
            f"[{idx}] {item['title']}\n{item['snippet']}\n{item['link']}"
        )
    context = "\n\n".join(context_lines)
    return items, context
