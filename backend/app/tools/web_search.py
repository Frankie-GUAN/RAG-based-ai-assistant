import os
from typing import List

from langchain_community.utilities import SerpAPIWrapper

from app.config import settings
from app.tools.registry import tool_registry


def _format_items(items: List[dict]) -> str:
    lines = []
    for i, item in enumerate(items, 1):
        title = item.get("title", item.get("link", ""))
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        lines.append(f"[{i}] {title}\n{snippet}\n{link}")
    return "\n\n".join(lines)


def search_web(query: str, num: int = 5) -> str:
    if not settings.serpapi_key:
        return "联网搜索未配置 SerpAPI_KEY"

    os.environ["SERPAPI_API_KEY"] = settings.serpapi_key
    wrapper = SerpAPIWrapper(params={"num": num, "engine": "google", "hl": "zh-cn"})
    results = wrapper.results(query)

    items = []
    for item in results.get("organic_results", []):
        if item.get("link"):
            items.append(item)
    return _format_items(items)


tool_registry.register("search_web", search_web)
