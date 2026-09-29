"""
Tavily 搜索引擎实现

Tavily 是专为 AI Agent 设计的搜索 API，
返回结果已经过清洗和摘要，非常适合 Agent 场景。
"""

import httpx

from search.base_search import BaseSearch, SearchResult


class TavilySearch(BaseSearch):
    """
    Tavily 搜索引擎实现

    Tavily API 特点：
    - 专为 AI Agent 设计，返回结果已清洗
    - 支持搜索深度控制（basic / advanced）
    - 自动提取网页内容摘要
    - 返回相关度评分
    """

    # Tavily API 端点
    BASE_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str = "",
        max_results: int = 5,
        search_depth: str = "advanced",
    ):
        """
        初始化 Tavily 搜索

        Args:
            api_key: Tavily API 密钥
            max_results: 最大返回结果数
            search_depth: 搜索深度，"basic" 或 "advanced"
        """
        super().__init__(api_key, max_results)
        self.search_depth = search_depth

    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """
        调用 Tavily API 执行搜索

        Args:
            query: 搜索关键词
            max_results: 最大返回结果数

        Returns:
            SearchResult 列表
        """
        num_results = max_results or self.max_results

        # 构造请求体
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": num_results,
            "search_depth": self.search_depth,
            "include_answer": False,
            "include_raw_content": False,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.BASE_URL, json=payload)
            response.raise_for_status()
            data = response.json()

        # 将 Tavily 返回结果转换为统一的 SearchResult 格式
        results = []
        for item in data.get("results", []):
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                source="tavily",
                score=float(item.get("score", 0.0)),
            )
            results.append(result)

        return results
