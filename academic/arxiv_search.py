"""
Arxiv 学术搜索实现

使用 arxiv 官方 API 搜索论文，
无需 API Key，是最容易上手的学术搜索方式。
"""

import arxiv

from academic.base_academic import BaseAcademic, PaperResult


class ArxivSearch(BaseAcademic):
    """
    Arxiv 学术搜索实现

    Arxiv 特点：
    - 免费、无需 API Key
    - 覆盖物理、数学、计算机科学等领域
    - 支持全文搜索
    - 返回论文元数据（标题、作者、摘要等）
    """

    def __init__(self, max_results: int = 5):
        # Arxiv 不需要 API Key
        super().__init__(api_key="", max_results=max_results)

    async def search_papers(
        self, query: str, max_results: int | None = None
    ) -> list[PaperResult]:
        """
        搜索 Arxiv 论文

        使用 arxiv 库的同步 API（因为 arxiv 库不支持异步），
        在实际使用中可以通过 asyncio.to_thread 包装为异步调用。

        Args:
            query: 搜索关键词
            max_results: 最大返回结果数

        Returns:
            PaperResult 列表
        """
        num_results = max_results or self.max_results

        # 使用 arxiv 库搜索
        # arxiv.Client() 是同步的，在 async 函数中需要用 to_thread 包装
        import asyncio

        def _sync_search():
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=num_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results = []
            for paper in client.results(search):
                result = PaperResult(
                    title=paper.title,
                    authors=[author.name for author in paper.authors],
                    abstract=paper.summary,
                    url=paper.entry_id,
                    year=paper.published.year if paper.published else None,
                    citation_count=None,
                    source="arxiv",
                )
                results.append(result)
            return results

        # 将同步调用包装为异步
        return await asyncio.to_thread(_sync_search)

    async def get_paper_detail(self, paper_id: str) -> PaperResult | None:
        """
        获取 Arxiv 论文详情

        Args:
            paper_id: Arxiv 论文 ID（如 "2301.01234"）

        Returns:
            论文详情
        """
        import asyncio

        def _sync_get():
            client = arxiv.Client()
            search = arxiv.Search(id_list=[paper_id])
            for paper in client.results(search):
                return PaperResult(
                    title=paper.title,
                    authors=[author.name for author in paper.authors],
                    abstract=paper.summary,
                    url=paper.entry_id,
                    year=paper.published.year if paper.published else None,
                    citation_count=None,
                    source="arxiv",
                )
            return None

        return await asyncio.to_thread(_sync_get)

    async def get_paper_abstract(self, paper_id: str) -> str:
        """
        获取 Arxiv 论文摘要

        Args:
            paper_id: Arxiv 论文 ID

        Returns:
            论文摘要文本
        """
        paper = await self.get_paper_detail(paper_id)
        return paper.abstract if paper else ""
