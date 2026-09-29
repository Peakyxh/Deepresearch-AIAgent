"""
搜索引擎工厂 —— 根据配置自动创建对应的搜索引擎实例

使用工厂模式，上层代码只需调用 SearchFactory.create()，
无需关心具体使用的是哪个搜索引擎。
"""

from config import settings
from search.base_search import BaseSearch, SearchResult
from search.bing_search import BingSearch
from search.bocha_search import BochaSearch
from search.serpapi_search import SerpAPISearch
from search.tavily_search import TavilySearch


class SearchFactory:
    """
    搜索引擎工厂类

    根据配置中的 search_provider 字段，自动创建对应的搜索引擎实例。
    支持的搜索引擎：tavily、serpapi、bing、bocha
    """

    _providers: dict[str, type[BaseSearch]] = {
        "tavily": TavilySearch,
        "serpapi": SerpAPISearch,
        "bing": BingSearch,
        "bocha": BochaSearch,
    }

    @classmethod
    def create(cls, provider: str | None = None) -> BaseSearch:
        """
        创建搜索引擎实例

        Args:
            provider: 搜索引擎名称，如果为 None 则从配置读取

        Returns:
            对应的搜索引擎实例

        Raises:
            ValueError: 不支持的搜索引擎名称
        """
        provider = provider or settings.search_provider
        provider = provider.lower()

        if provider not in cls._providers:
            raise ValueError(
                f"不支持的搜索引擎: {provider}，"
                f"可选值: {list(cls._providers.keys())}"
            )

        # 根据不同的搜索引擎，传入对应的配置参数
        if provider == "tavily":
            return TavilySearch(
                api_key=settings.tavily_api_key,
                max_results=settings.max_search_results,
            )
        elif provider == "serpapi":
            return SerpAPISearch(
                api_key=settings.serpapi_api_key,
                max_results=settings.max_search_results,
            )
        elif provider == "bing":
            return BingSearch(
                api_key=settings.bing_api_key,
                max_results=settings.max_search_results,
            )
        elif provider == "bocha":
            return BochaSearch(
                api_key=settings.bocha_api_key,
                max_results=settings.max_search_results,
            )

        raise ValueError(f"未知的搜索引擎: {provider}")

    @classmethod
    def register(cls, name: str, search_class: type[BaseSearch]) -> None:
        """
        注册新的搜索引擎

        Args:
            name: 搜索引擎名称
            search_class: 对应的搜索引擎类
        """
        cls._providers[name] = search_class
