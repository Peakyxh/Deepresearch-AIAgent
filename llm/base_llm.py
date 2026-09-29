"""
LLM 基类 —— 定义所有 LLM 提供者必须实现的统一接口

所有 LLM 实现（OpenAI、DeepSeek、vLLM）都继承此基类，
确保上层代码可以无差别地切换不同的 LLM 提供者。
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseLLM(ABC):
    """
    LLM 抽象基类

    定义三个核心方法：
    - generate(): 同步生成完整回复
    - stream_generate(): 流式生成回复（逐 token 输出）
    - embedding(): 获取文本的向量表示
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        """
        初始化 LLM 基类

        Args:
            model: 模型名称，如 "gpt-4o"、"deepseek-chat"
            api_key: API 密钥
            base_url: API 基础 URL
            temperature: 温度参数，控制输出随机性
            max_tokens: 最大输出 token 数
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = "", max_tokens: int | None = None) -> str:
        """
        生成完整回复

        Args:
            prompt: 用户输入的提示词
            system_prompt: 系统提示词，用于设定 AI 角色和行为
            max_tokens: 本次调用的最大输出 token 数，为 None 则使用默认值

        Returns:
            LLM 生成的完整文本回复
        """
        ...

    @abstractmethod
    async def stream_generate(
        self, prompt: str, system_prompt: str = ""
    ) -> AsyncIterator[str]:
        """
        流式生成回复（逐 token 输出）

        适用于需要实时显示生成过程的场景，如聊天界面。

        Args:
            prompt: 用户输入的提示词
            system_prompt: 系统提示词

        Yields:
            每次产出的文本片段（通常是单个 token 或几个 token）
        """
        ...
        # 让 mypy 知道这是一个异步生成器
        if False:
            yield ""

    @abstractmethod
    async def embedding(self, text: str) -> list[float]:
        """
        获取文本的向量表示

        用于语义搜索、相似度计算等场景。

        Args:
            text: 需要向量化的文本

        Returns:
            浮点数列表，表示文本的向量表示
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
