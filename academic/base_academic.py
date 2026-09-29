"""
学术搜索基类 —— 定义所有学术搜索引擎必须实现的统一接口

所有学术搜索（Arxiv、Semantic Scholar、OpenAlex）都继承此基类，
确保返回统一的 PaperResult 格式。
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class PaperResult(BaseModel):
    """
    论文搜索结果统一数据模型

    所有学术搜索引擎返回的结果都必须符合此格式。
    """

    title: str = Field(default="", description="论文标题")
    authors: list[str] = Field(default_factory=list, description="作者列表")
    abstract: str = Field(default="", description="论文摘要")
    url: str = Field(default="", description="论文链接")
    year: int | None = Field(default=None, description="发表年份")
    citation_count: int | None = Field(default=None, description="引用次数")
    source: str = Field(default="", description="来源学术搜索引擎名称")


class BaseAcademic(ABC):
    """
    学术搜索引擎抽象基类

    定义三个核心方法：
    - search_papers(): 搜索论文
    - get_paper_detail(): 获取论文详情
    - get_paper_abstract(): 获取论文摘要
    """

    def __init__(self, api_key: str = "", max_results: int = 5):
        """
        初始化学术搜索引擎基类

        Args:
            api_key: API 密钥（部分学术搜索不需要）
            max_results: 最大返回结果数
        """
        self.api_key = api_key
        self.max_results = max_results

    @abstractmethod
    async def search_papers(
        self, query: str, max_results: int | None = None
    ) -> list[PaperResult]:
        """
        搜索论文

        Args:
            query: 搜索关键词
            max_results: 最大返回结果数

        Returns:
            PaperResult 列表
        """
        ...

    @abstractmethod
    async def get_paper_detail(self, paper_id: str) -> PaperResult | None:
        """
        获取论文详情

        Args:
            paper_id: 论文唯一标识符

        Returns:
            论文详情，如果未找到则返回 None
        """
        ...

    @abstractmethod
    async def get_paper_abstract(self, paper_id: str) -> str:
        """
        获取论文摘要

        Args:
            paper_id: 论文唯一标识符

        Returns:
            论文摘要文本
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_results={self.max_results})"
