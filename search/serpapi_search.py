"""
SerpAPI 搜索引擎实现

SerpAPI 提供 Google 搜索结果的结构化 API，
适合需要 Google 搜索结果的场景。
"""

import httpx

from search.base_search import BaseSearch, SearchResult


class SerpAPISearch(BaseSearch):
    """
    SerpAPI 搜索引擎实现

    通过 SerpAPI 获取 Google 搜索结果，
    返回结构化的搜索数据。
    """

    BASE_URL = "https://serpapi.com/search"

    def __init__(self, api_key: str = "", max_results: int = 5):
        super().__init__(api_key, max_results)

    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """
        调用 SerpAPI 执行 Google 搜索

        Args:
            query: 搜索关键词
            max_results: 最大返回结果数

        Returns:
            SearchResult 列表
        """
        num_results = max_results or self.max_results

        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": num_results,
            "hl": "zh-cn",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        results = []
        # SerpAPI 返回的有机搜索结果在 "organic_results" 字段中
        for item in data.get("organic_results", [])[:num_results]:
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                content=item.get("snippet", ""),
                source="serpapi",
                score=0.0,
            )
            results.append(result)

        return results
