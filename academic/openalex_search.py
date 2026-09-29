"""
OpenAlex 学术搜索实现

OpenAlex 是开放学术元数据平台，
覆盖全球学术出版物，数据量巨大。
"""

import httpx

from academic.base_academic import BaseAcademic, PaperResult


class OpenAlexSearch(BaseAcademic):
    """
    OpenAlex 学术搜索实现

    OpenAlex 特点：
    - 完全免费开放
    - 覆盖 2.5 亿+ 学术作品
    - 支持引用次数、机构等元数据查询
    - 无需 API Key（但建议加 email 提高速率）
    """

    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, api_key: str = "", max_results: int = 5):
        # OpenAlex 的 api_key 实际上是 email，用于提高速率限制
        super().__init__(api_key, max_results)

    async def search_papers(
        self, query: str, max_results: int | None = None
    ) -> list[PaperResult]:
        """
        搜索 OpenAlex 论文

        Args:
            query: 搜索关键词
            max_results: 最大返回结果数

        Returns:
            PaperResult 列表
        """
        num_results = max_results or self.max_results

        params = {
            "search": query,
            "per_page": num_results,
            "sort": "relevance_score:desc",
        }
        # 如果提供了 email，加入参数以提高速率限制
        if self.api_key and "@" in self.api_key:
            params["mailto"] = self.api_key

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("results", []):
            # 提取作者信息
            authors = []
            for authorship in item.get("authorships", []):
                author = authorship.get("author", {})
                if author.get("display_name"):
                    authors.append(author["display_name"])

            # 提取年份
            year = item.get("publication_year")

            # 提取引用次数
            citation_count = item.get("cited_by_count")

            result = PaperResult(
                title=item.get("title", ""),
                authors=authors,
                abstract=self._extract_abstract(item.get("abstract_inverted_index")),
                url=item.get("id", ""),
                year=year,
                citation_count=citation_count,
                source="openalex",
            )
            results.append(result)

        return results

    async def get_paper_detail(self, paper_id: str) -> PaperResult | None:
        """
        获取论文详情

        Args:
            paper_id: OpenAlex 作品 ID（如 "W1234567890"）

        Returns:
            论文详情
        """
        params = {}
        if self.api_key and "@" in self.api_key:
            params["mailto"] = self.api_key

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/{paper_id}", params=params
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            item = response.json()

        authors = []
        for authorship in item.get("authorships", []):
            author = authorship.get("author", {})
            if author.get("display_name"):
                authors.append(author["display_name"])

        return PaperResult(
            title=item.get("title", ""),
            authors=authors,
            abstract=self._extract_abstract(item.get("abstract_inverted_index")),
            url=item.get("id", ""),
            year=item.get("publication_year"),
            citation_count=item.get("cited_by_count"),
            source="openalex",
        )

    async def get_paper_abstract(self, paper_id: str) -> str:
        """
        获取论文摘要

        Args:
            paper_id: OpenAlex 作品 ID

        Returns:
            论文摘要文本
        """
        paper = await self.get_paper_detail(paper_id)
        return paper.abstract if paper else ""

    @staticmethod
    def _extract_abstract(
        inverted_index: dict[str, list[int]] | None,
    ) -> str:
        """
        从 OpenAlex 的倒排索引格式中提取摘要文本

        OpenAlex 返回的摘要是倒排索引格式：
        {"word": [position1, position2], ...}
        需要将其还原为正常文本。

        Args:
            inverted_index: 倒排索引字典

        Returns:
            还原后的摘要文本
        """
        if not inverted_index:
            return ""

        # 找出最大位置值
        max_pos = 0
        for positions in inverted_index.values():
            for pos in positions:
                if pos > max_pos:
                    max_pos = pos

        # 创建位置到词的映射
        words = [""] * (max_pos + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word

        return " ".join(words)
