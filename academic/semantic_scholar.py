"""
Semantic Scholar 学术搜索实现

使用 Semantic Scholar API 搜索论文，
支持引用次数查询，适合评估论文影响力。
"""

import httpx

from academic.base_academic import BaseAcademic, PaperResult


class SemanticScholarSearch(BaseAcademic):
    """
    Semantic Scholar 学术搜索实现

    Semantic Scholar 特点：
    - 免费学术搜索 API
    - 支持引用次数查询
    - 覆盖多学科论文
    - 可选 API Key（提高速率限制）
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str = "", max_results: int = 5):
        super().__init__(api_key, max_results)

    async def search_papers(
        self, query: str, max_results: int | None = None
    ) -> list[PaperResult]:
        """
        搜索 Semantic Scholar 论文

        Args:
            query: 搜索关键词
            max_results: 最大返回结果数

        Returns:
            PaperResult 列表
        """
        num_results = max_results or self.max_results

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        params = {
            "query": query,
            "limit": num_results,
            "fields": "title,authors,abstract,url,year,citationCount",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/paper/search",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("data", []):
            authors = []
            for author in item.get("authors", []):
                if author.get("name"):
                    authors.append(author["name"])

            result = PaperResult(
                title=item.get("title", ""),
                authors=authors,
                abstract=item.get("abstract", "") or "",
                url=item.get("url", ""),
                year=item.get("year"),
                citation_count=item.get("citationCount"),
                source="semantic_scholar",
            )
            results.append(result)

        return results

    async def get_paper_detail(self, paper_id: str) -> PaperResult | None:
        """
        获取论文详情

        Args:
            paper_id: Semantic Scholar 论文 ID

        Returns:
            论文详情
        """
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        params = {
            "fields": "title,authors,abstract,url,year,citationCount",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/paper/{paper_id}",
                headers=headers,
                params=params,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            item = response.json()

        authors = []
        for author in item.get("authors", []):
            if author.get("name"):
                authors.append(author["name"])

        return PaperResult(
            title=item.get("title", ""),
            authors=authors,
            abstract=item.get("abstract", "") or "",
            url=item.get("url", ""),
            year=item.get("year"),
            citation_count=item.get("citationCount"),
            source="semantic_scholar",
        )

    async def get_paper_abstract(self, paper_id: str) -> str:
        """
        获取论文摘要

        Args:
            paper_id: Semantic Scholar 论文 ID

        Returns:
            论文摘要文本
        """
        paper = await self.get_paper_detail(paper_id)
        return paper.abstract if paper else ""
