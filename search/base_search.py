"""
搜索引擎基类 —— 定义所有搜索引擎必须实现的统一接口

所有搜索引擎（Tavily、SerpAPI、Bing、Bocha）都继承此基类，
确保返回统一的 SearchResult 格式。
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """
    搜索结果统一数据模型

    所有搜索引擎返回的结果都必须符合此格式，
    确保上层代码可以无差别地处理不同搜索引擎的结果。
    """

    title: str = Field(default="", description="搜索结果标题")
    url: str = Field(default="", description="搜索结果链接")
    content: str = Field(default="", description="搜索结果内容摘要")
    source: str = Field(default="", description="来源搜索引擎名称")
    score: float = Field(default=0.0, description="相关度评分，0-1 之间")


class BaseSearch(ABC):
    """
    搜索引擎抽象基类

    定义统一接口：
    - search(): 执行搜索，返回 SearchResult 列表
    """

    def __init__(self, api_key: str = "", max_results: int = 5):
        """
        初始化搜索引擎基类

        Args:
            api_key: 搜索引擎 API 密钥
            max_results: 最大返回结果数
        """
        self.api_key = api_key
        self.max_results = max_results

    @abstractmethod
    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """
        执行搜索

        Args:
            query: 搜索关键词
            max_results: 最大返回结果数，如果为 None 则使用默认值

        Returns:
            SearchResult 列表
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_results={self.max_results})"
