"""
Bing 搜索引擎实现

使用 Bing Web Search API 获取搜索结果。
"""

import httpx

from search.base_search import BaseSearch, SearchResult


class BingSearch(BaseSearch):
    """
    Bing 搜索引擎实现

    通过 Microsoft Bing Web Search API 获取搜索结果。
    """

    BASE_URL = "https://api.bing.microsoft.com/v7.0/search"

    def __init__(self, api_key: str = "", max_results: int = 5):
        super().__init__(api_key, max_results)

    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """
        调用 Bing Web Search API 执行搜索

        Args:
            query: 搜索关键词
            max_results: 最大返回结果数

        Returns:
            SearchResult 列表
        """
        num_results = max_results or self.max_results

        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": num_results,
            "mkt": "zh-CN",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                self.BASE_URL, headers=headers, params=params
            )
            response.raise_for_status()
            data = response.json()

        results = []
        # Bing 返回的网页结果在 "webPages" -> "value" 字段中
        for item in data.get("webPages", {}).get("value", [])[:num_results]:
            result = SearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                content=item.get("snippet", ""),
                source="bing",
                score=0.0,
            )
            results.append(result)

        return results
