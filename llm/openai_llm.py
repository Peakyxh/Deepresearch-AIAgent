"""
OpenAI LLM 实现

通过 OpenAI API（或兼容接口）调用大语言模型，
支持 generate / stream_generate / embedding 三大核心方法。
"""

from typing import AsyncIterator

from openai import AsyncOpenAI

from llm.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM 实现

    使用 openai 官方异步客户端，支持：
    - GPT-4o、GPT-4、GPT-3.5 等模型
    - 任何 OpenAI 兼容接口（如 Azure OpenAI）
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        super().__init__(model, api_key, base_url, temperature, max_tokens)
        # 创建异步 OpenAI 客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def generate(self, prompt: str, system_prompt: str = "", max_tokens: int | None = None) -> str:
        """
        调用 OpenAI API 生成完整回复

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            max_tokens: 本次调用的最大输出 token 数，为 None 则使用默认值

        Returns:
            模型生成的完整文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream_generate(
        self, prompt: str, system_prompt: str = ""
    ) -> AsyncIterator[str]:
        """
        流式调用 OpenAI API，逐 token 返回

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词

        Yields:
            每次产出的文本片段
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            # 每个 chunk 的 choices[0].delta.content 可能是 None
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def embedding(self, text: str) -> list[float]:
        """
        调用 OpenAI Embedding API 获取文本向量

        默认使用 text-embedding-3-small 模型，
        性价比高，1536 维向量。

        Args:
            text: 需要向量化的文本

        Returns:
            浮点数列表
        """
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
