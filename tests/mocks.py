"""
测试用 Mock 对象

提供 MockLLM、MockSearch、MockAcademic、MockContextManager，
用于 Agent 单元测试，避免依赖外部 API。

注意：Mock 对象不继承基类，避免触发 llm/search/academic 包的 __init__.py
级联导入（需要 openai 等外部依赖），而是直接实现所需接口。
"""

import json


class MockLLM:
    """
    Mock LLM

    根据预设的响应映射返回固定回复，
    支持按 prompt 关键词匹配不同的响应。
    """

    def __init__(
        self,
        default_response: str = '{"result": "mock"}',
        responses: dict[str, str] | None = None,
    ):
        self.model = "mock-model"
        self.api_key = "mock-key"
        self.base_url = "http://mock"
        self.temperature = 0.3
        self.max_tokens = 4096
        self.default_response = default_response
        self.responses = responses or {}
        self.call_log: list[dict] = []

    async def generate(
        self, prompt: str, system_prompt: str = "", max_tokens: int | None = None
    ) -> str:
        self.call_log.append({
            "prompt": prompt[:200],
            "system_prompt": system_prompt[:100],
        })

        for keyword, response in self.responses.items():
            if keyword in prompt or keyword in system_prompt:
                return response

        return self.default_response

    async def stream_generate(self, prompt: str, system_prompt: str = ""):
        response = await self.generate(prompt, system_prompt)
        yield response

    async def embedding(self, text: str) -> list[float]:
        return [0.1] * 128

    def __repr__(self) -> str:
        return f"MockLLM(model={self.model})"


class MockSearchResult:
    """Mock 搜索结果"""

    def __init__(
        self,
        title: str = "",
        url: str = "",
        content: str = "",
        source: str = "mock",
        score: float = 0.9,
    ):
        self.title = title
        self.url = url
        self.content = content
        self.source = source
        self.score = score

    def model_dump(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "source": self.source,
            "score": self.score,
        }


class MockSearch:
    """
    Mock 搜索引擎

    返回固定的搜索结果列表。
    """

    def __init__(self, results: list[MockSearchResult] | None = None):
        self.api_key = "mock-key"
        self.max_results = 5
        self._results = results or [
            MockSearchResult(
                title="Mock搜索结果1",
                url="https://example.com/1",
                content="这是模拟的搜索结果内容，用于单元测试。" * 5,
                source="mock",
                score=0.9,
            ),
            MockSearchResult(
                title="Mock搜索结果2",
                url="https://example.com/2",
                content="这是另一个模拟的搜索结果内容，包含更多详细信息。" * 5,
                source="mock",
                score=0.8,
            ),
        ]
        self.search_log: list[str] = []

    async def search(
        self, query: str, max_results: int | None = None
    ) -> list[MockSearchResult]:
        self.search_log.append(query)
        return self._results[: (max_results or self.max_results)]

    def __repr__(self) -> str:
        return f"MockSearch(max_results={self.max_results})"


class MockPaperResult:
    """Mock 论文搜索结果"""

    def __init__(
        self,
        title: str = "",
        authors: list[str] | None = None,
        abstract: str = "",
        url: str = "",
        year: int | None = None,
        citation_count: int | None = None,
        source: str = "mock",
    ):
        self.title = title
        self.authors = authors or []
        self.abstract = abstract
        self.url = url
        self.year = year
        self.citation_count = citation_count
        self.source = source

    def model_dump(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "url": self.url,
            "year": self.year,
            "citation_count": self.citation_count,
            "source": self.source,
        }


class MockAcademic:
    """
    Mock 学术搜索引擎

    返回固定的论文搜索结果。
    """

    def __init__(self, papers: list[MockPaperResult] | None = None):
        self.api_key = "mock-key"
        self.max_results = 5
        self._papers = papers or [
            MockPaperResult(
                title="Mock论文1: A Survey on RAG",
                authors=["Author A", "Author B"],
                abstract="This paper presents a comprehensive survey on Retrieval-Augmented Generation.",
                url="https://arxiv.org/abs/2401.00001",
                year=2024,
                citation_count=100,
                source="mock",
            ),
            MockPaperResult(
                title="Mock论文2: Advanced RAG Techniques",
                authors=["Author C"],
                abstract="We propose advanced techniques for improving RAG systems.",
                url="https://arxiv.org/abs/2401.00002",
                year=2023,
                citation_count=50,
                source="mock",
            ),
        ]
        self.search_log: list[str] = []

    async def search_papers(
        self, query: str, max_results: int | None = None
    ) -> list[MockPaperResult]:
        self.search_log.append(query)
        return self._papers[: (max_results or self.max_results)]

    async def get_paper_detail(self, paper_id: str) -> MockPaperResult | None:
        return self._papers[0] if self._papers else None

    async def get_paper_abstract(self, paper_id: str) -> str:
        return self._papers[0].abstract if self._papers else ""

    def __repr__(self) -> str:
        return f"MockAcademic(max_results={self.max_results})"


class MockContextManager:
    """
    Mock 上下文管理器

    不做实际压缩，直接返回原文。
    """

    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.context: list[dict] = []

    def add_context(self, role: str, content: str, is_key: bool = False, query: str = ""):
        self.context.append({
            "role": role,
            "content": content,
            "is_key": is_key,
            "query": query,
        })

    def get_key_info(self) -> str:
        return "\n".join(
            item["content"] for item in self.context if item.get("is_key", False)
        )

    def estimate_text_tokens(self, text: str) -> int:
        return len(text) // 3

    def get_compress_threshold(self, text_type: str) -> int:
        ratios = {
            "search": 0.5,
            "paper": 0.5,
            "findings": 0.5,
            "prior": 0.25,
            "sources": 0.33,
        }
        return int(self.max_tokens * ratios.get(text_type, 0.5))

    async def maybe_compress(self):
        return self.get_context()

    async def compress_search_results(self, text: str) -> str:
        return text[: self.max_tokens * 3]

    async def compress_paper_results(self, text: str) -> str:
        return text[: self.max_tokens * 3]

    async def compress_text(self, text: str, instruction: str = "") -> str:
        return text[: self.max_tokens * 3]

    async def compress_findings(self, text: str, threshold_tokens: int | None = None) -> str:
        return text[: self.max_tokens * 3]

    async def get_rounds_findings(self) -> str:
        """提取历轮研究发现（与 ContextManager.get_rounds_findings 行为一致，不压缩）"""
        findings_items = [
            item["content"]
            for item in self.context
            if item.get("role") == "system" and item["content"].startswith("研究发现")
        ]
        return "\n\n".join(findings_items)

    def get_context(self) -> list[dict]:
        return [
            {"role": item["role"], "content": item["content"]}
            for item in self.context
        ]
