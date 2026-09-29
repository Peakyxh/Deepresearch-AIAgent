"""
LLM 工厂 —— 根据配置自动创建对应的 LLM 实例

使用工厂模式，上层代码只需调用 LLMFactory.create()，
无需关心具体使用的是哪个 LLM 提供者。
"""

from config import settings
from llm.base_llm import BaseLLM
from llm.deepseek_llm import DeepSeekLLM
from llm.openai_llm import OpenAILLM
from llm.qwen_llm import QwenLLM
from llm.vllm_llm import VLLMLLM


class LLMFactory:
    """
    LLM 工厂类

    根据配置中的 llm_provider 字段，自动创建对应的 LLM 实例。
    支持的提供者：openai、deepseek、vllm
    """

    # 注册所有可用的 LLM 提供者
    # key: 提供者名称, value: 对应的 LLM 类
    _providers: dict[str, type[BaseLLM]] = {
        "openai": OpenAILLM,
        "deepseek": DeepSeekLLM,
        "vllm": VLLMLLM,
        "qwen": QwenLLM,
    }

    @classmethod
    def create(cls, provider: str | None = None) -> BaseLLM:
        """
        创建 LLM 实例

        Args:
            provider: LLM 提供者名称，如果为 None 则从配置读取

        Returns:
            对应提供者的 LLM 实例

        Raises:
            ValueError: 不支持的提供者名称
        """
        provider = provider or settings.llm_provider
        provider = provider.lower()

        if provider not in cls._providers:
            raise ValueError(
                f"不支持的 LLM 提供者: {provider}，"
                f"可选值: {list(cls._providers.keys())}"
            )

        # 根据不同的提供者，传入对应的配置参数
        if provider == "openai":
            return OpenAILLM(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        elif provider == "deepseek":
            return DeepSeekLLM(
                model=settings.deepseek_model,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        elif provider == "vllm":
            return VLLMLLM(
                model=settings.vllm_model,
                api_key=settings.vllm_api_key,
                base_url=settings.vllm_base_url,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        elif provider == "qwen":
            return QwenLLM(
                model=settings.qwen_model,
                api_key=settings.qwen_api_key,
                base_url=settings.qwen_base_url,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

        # 理论上不会走到这里，因为上面已经校验了 provider
        raise ValueError(f"未知的 LLM 提供者: {provider}")

    @classmethod
    def register(cls, name: str, llm_class: type[BaseLLM]) -> None:
        """
        注册新的 LLM 提供者

        用于扩展支持新的 LLM 提供者，无需修改工厂代码。

        Args:
            name: 提供者名称
            llm_class: 对应的 LLM 类（必须继承 BaseLLM）
        """
        cls._providers[name] = llm_class
