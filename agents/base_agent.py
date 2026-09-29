"""
Agent 基类 —— 所有 Agent 的公共父类

提供 LLM 调用、日志输出、上下文压缩等公共能力，
各具体 Agent（Planner、Researcher、Critic、Writer）继承此基类。
"""

import logging
from typing import Any, TYPE_CHECKING

from llm.base_llm import BaseLLM
from llm.llm_factory import LLMFactory

if TYPE_CHECKING:
    from memory.context_manager import ContextManager


class BaseAgent:
    """
    Agent 抽象基类

    每个 Agent 都拥有：
    - 一个 LLM 实例（用于生成回复）
    - 一个 name（用于日志标识）
    - 一个 logger（用于日志输出）
    - 一个 context_manager（可选，用于语义压缩）

    子类需要实现 run() 方法，定义具体的 Agent 行为。
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM | None = None,
        context_manager: "ContextManager | None" = None,
    ):
        """
        初始化 Agent 基类

        Args:
            name: Agent 名称，如 "Planner"、"Researcher"
            llm: LLM 实例，如果为 None 则从工厂创建
            context_manager: 上下文管理器，用于语义压缩。
                            为 None 时 Agent 内部无法进行语义压缩。
        """
        self.name = name
        # 如果没有传入 LLM，则从工厂创建默认实例
        self.llm = llm or LLMFactory.create()
        # 上下文管理器（可选）
        self.context_manager = context_manager
        # 每个 Agent 拥有独立的 logger
        self.logger = logging.getLogger(f"Agent.{name}")

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        执行 Agent 逻辑（模板方法）

        在子类 _execute() 前后自动检查上下文是否需要压缩。

        Args:
            state: 当前工作流状态字典

        Returns:
            更新后的状态字典
        """
        await self._check_context()
        result = await self._execute(state)
        await self._check_context()
        return result

    async def _execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        执行 Agent 具体逻辑

        子类必须实现此方法，定义具体的 Agent 行为。

        Args:
            state: 当前工作流状态字典

        Returns:
            更新后的状态字典
        """
        raise NotImplementedError(f"Agent {self.name} 必须实现 _execute() 方法")

    async def _check_context(self) -> None:
        """
        检查上下文是否需要压缩

        如果 Agent 拥有 context_manager，则在执行前后
        自动检查并触发渐进式压缩。
        """
        if self.context_manager:
            await self.context_manager.maybe_compress()

    def context_prompt_prefix(self) -> str:
        """
        把上下文管理器中的关键信息渲染为 prompt 前缀

        上游阶段（Clarifier / Planner）会把"意图澄清""研究计划"等关键事实写入
        ContextManager 并标记 is_key=True，下游 Agent（Critic / Writer）通过本方法
        取得这些事实，从而保证研究不偏离用户真实意图与研究计划。

        可压缩的过程性内容（研究发现等）不在其中：它们体积大且已经在 state 里，
        重复注入只会浪费 token。

        Returns:
            以 Markdown 小节形式呈现的关键背景；没有关键信息或没有
            context_manager 时返回空字符串。
        """
        if not self.context_manager:
            return ""

        get_key_info = getattr(self.context_manager, "get_key_info", None)
        if not callable(get_key_info):
            return ""

        key_info = get_key_info()
        if not key_info:
            return ""

        return (
            "## 研究背景（来自上游阶段，必须遵守）\n"
            f"{key_info}\n\n"
            "---\n\n"
        )

    def log(self, message: str) -> None:
        """
        输出日志信息

        同时打印到控制台和写入 logger，
        方便调试和追踪 Agent 执行过程。

        Args:
            message: 日志消息
        """
        formatted = f"[{self.name}] {message}"
        self.logger.info(formatted)
        print(formatted)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, llm={self.llm})"
