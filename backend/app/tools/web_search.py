import os

from langchain_community.utilities import SerpAPIWrapper

from app.config import settings


def search_web(query: str, num: int = 5) -> list[dict]:
    """联网搜索，返回结构化结果列表。未配置 key 时抛 RuntimeError（由 MCP 层转成 ok=False）。"""
    if not settings.serpapi_key:
        raise RuntimeError("联网搜索未配置 SERPAPI_KEY")

    os.environ["SERPAPI_API_KEY"] = settings.serpapi_key
    wrapper = SerpAPIWrapper(params={"num": num, "engine": "google", "hl": "zh-cn"})
    results = wrapper.results(query)

    return [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item["link"],
        }
        for item in results.get("organic_results", [])
        if item.get("link")
    ]
