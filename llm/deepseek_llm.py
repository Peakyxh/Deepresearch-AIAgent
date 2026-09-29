"""
DeepSeek LLM 实现

DeepSeek API 与 OpenAI API 完全兼容，
因此实现逻辑与 OpenAILLM 基本一致，只需切换 base_url 和 model。
"""

from typing import AsyncIterator

from openai import AsyncOpenAI

from llm.base_llm import BaseLLM


class DeepSeekLLM(BaseLLM):
    """
    DeepSeek LLM 实现

    DeepSeek 提供 OpenAI 兼容接口，可以直接使用 openai 客户端调用。
    支持模型：deepseek-chat、deepseek-reasoner 等。
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str = "",
        base_url: str = "https://api.deepseek.com/v1",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        super().__init__(model, api_key, base_url, temperature, max_tokens)
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def generate(self, prompt: str, system_prompt: str = "", max_tokens: int | None = None) -> str:
        """调用 DeepSeek API 生成完整回复"""
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
        """流式调用 DeepSeek API，逐 token 返回"""
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
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def embedding(self, text: str) -> list[float]:
        """
        获取文本向量

        DeepSeek 目前不提供独立的 Embedding API，
        这里回退到 OpenAI 的 embedding 接口。
        如果需要使用其他 embedding 服务，可以覆盖此方法。
        """
        # 使用 OpenAI 兼容接口获取 embedding
        # 如果 DeepSeek 后续提供 embedding 接口，可以替换此处
        from config import settings

        client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
