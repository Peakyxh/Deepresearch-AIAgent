"""
Qwen LLM 实现

通义千问 DashScope API 与 OpenAI API 完全兼容，
因此实现逻辑与 DeepSeekLLM 基本一致，只需切换 base_url 和 model。
"""

from typing import AsyncIterator

from openai import AsyncOpenAI

from llm.base_llm import BaseLLM


class QwenLLM(BaseLLM):
    """
    Qwen LLM 实现

    通义千问提供 OpenAI 兼容接口（DashScope），可以直接使用 openai 客户端调用。
    支持模型：qwen3.6-flash、qwen-plus、qwen-turbo 等。
    """

    def __init__(
        self,
        model: str = "qwen3.6-flash",
        api_key: str = "",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ):
        super().__init__(model, api_key, base_url, temperature, max_tokens)
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def generate(self, prompt: str, system_prompt: str = "", max_tokens: int | None = None) -> str:
        """调用 Qwen API 生成完整回复"""
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
        """流式调用 Qwen API，逐 token 返回"""
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

        Qwen DashScope 提供 text-embedding-v3 模型，
        通过 OpenAI 兼容接口调用。
        """
        response = await self.client.embeddings.create(
            model="text-embedding-v3",
            input=text,
        )
        return response.data[0].embedding
