"""
学术搜索工厂 —— 根据配置自动创建对应的学术搜索引擎实例
"""

from academic.arxiv_search import ArxivSearch
from academic.base_academic import BaseAcademic, PaperResult
from academic.openalex_search import OpenAlexSearch
from academic.semantic_scholar import SemanticScholarSearch
from config import settings


class AcademicFactory:
    """
    学术搜索工厂类

    根据配置中的 academic_provider 字段，自动创建对应的学术搜索引擎实例。
    支持的搜索引擎：arxiv、semantic_scholar、openalex
    """

    _providers: dict[str, type[BaseAcademic]] = {
        "arxiv": ArxivSearch,
        "semantic_scholar": SemanticScholarSearch,
        "openalex": OpenAlexSearch,
    }

    @classmethod
    def create(cls, provider: str | None = None) -> BaseAcademic:
        """
        创建学术搜索引擎实例

        Args:
            provider: 学术搜索引擎名称，如果为 None 则从配置读取

        Returns:
            对应的学术搜索引擎实例

        Raises:
            ValueError: 不支持的学术搜索引擎名称
        """
        provider = provider or settings.academic_provider
        provider = provider.lower()

        if provider not in cls._providers:
            raise ValueError(
                f"不支持的学术搜索引擎: {provider}，"
                f"可选值: {list(cls._providers.keys())}"
            )

        if provider == "arxiv":
            return ArxivSearch(max_results=settings.max_paper_results)
        elif provider == "semantic_scholar":
            return SemanticScholarSearch(
                api_key=settings.semantic_scholar_api_key,
                max_results=settings.max_paper_results,
            )
        elif provider == "openalex":
            return OpenAlexSearch(
                api_key="",
                max_results=settings.max_paper_results,
            )

        raise ValueError(f"未知的学术搜索引擎: {provider}")

    @classmethod
    def register(cls, name: str, academic_class: type[BaseAcademic]) -> None:
        """
        注册新的学术搜索引擎

        Args:
            name: 学术搜索引擎名称
            academic_class: 对应的学术搜索引擎类
        """
        cls._providers[name] = academic_class
