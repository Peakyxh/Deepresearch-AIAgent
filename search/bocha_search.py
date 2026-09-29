"""
Bocha 搜索引擎实现

Bocha（博查）是国内可用的搜索 API，
适合需要国内搜索结果的场景。
"""

import httpx

from search.base_search import BaseSearch, SearchResult


class BochaSearch(BaseSearch):
    """
    Bocha 搜索引擎实现

    Bocha 是国内可用的 AI 搜索 API，
    返回结构化的搜索结果。
    """

    BASE_URL = "https://api.bochaai.com/v1/web-search"

    def __init__(self, api_key: str = "", max_results: int = 5):
        super().__init__(api_key, max_results)

    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """
        调用 Bocha API 执行搜索

        Args:
            query: 搜索关键词
            max_results: 最大返回结果数

        Returns:
            SearchResult 列表
        """
        num_results = max_results or self.max_results

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "count": num_results,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.BASE_URL, headers=headers, json=payload
            )
            response.raise_for_status()
            data = response.json()

        results = []
        # Bocha 返回结果在 "data" -> "webPages" -> "value" 中
        for item in data.get("data", {}).get("webPages", {}).get("value", []):
            result = SearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                content=item.get("snippet", ""),
                source="bocha",
                score=0.0,
            )
            results.append(result)

        return results
