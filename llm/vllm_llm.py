"""
vLLM LLM 实现

vLLM 是高性能的本地 LLM 推理引擎，提供 OpenAI 兼容接口。
适用于本地部署大模型，无需依赖外部 API。
"""

from typing import AsyncIterator

from openai import AsyncOpenAI

from llm.base_llm import BaseLLM


class VLLMLLM(BaseLLM):
    """
    vLLM LLM 实现

    vLLM 部署后提供 OpenAI 兼容的 HTTP 接口，
    因此同样使用 openai 客户端调用，只需修改 base_url 指向本地服务。
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-3-8B",
        api_key: str = "EMPTY",
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        super().__init__(model, api_key, base_url, temperature, max_tokens)
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def generate(self, prompt: str, system_prompt: str = "", max_tokens: int | None = None) -> str:
        """调用本地 vLLM 服务生成完整回复"""
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
        """流式调用本地 vLLM 服务，逐 token 返回"""
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

        vLLM 如果部署了 embedding 模型（如 bge-large），
        可以通过 OpenAI 兼容接口调用。
        如果未部署，则回退到 OpenAI embedding。
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except Exception:
            # vLLM 未部署 embedding 模型时，回退到 OpenAI
            from config import settings

            fallback_client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
            response = await fallback_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
