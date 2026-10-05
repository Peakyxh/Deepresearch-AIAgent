"""
研究工作流

使用自研 DAG 引擎构建研究工作流，
实现 Planner → Orchestrator → Critic → Writer 的完整研究流程。

Orchestrator 负责根据 Planner 的子问题依赖关系进行调度：
- 无依赖的子问题并行执行
- 有依赖的子问题按拓扑排序串行执行
- 前置子问题的发现传递给后续子问题
- 子问题数 ≤ 1 或依赖解析失败时回退到串行 Researcher

如果 Critic 审查不通过，则回退到 Orchestrator/Researcher 重新搜索（最多重试 N 次）。

优势（相比 LangGraph）：
- 零外部依赖，完全可控的调度逻辑
- 支持 DAG 拓扑并发执行（fan-out / fan-in）
- 支持执行快照，可断点续跑和调试
- 支持最大步数限制，防止条件边导致的无限循环
"""

import logging
from typing import Any, Callable, Literal

from agents.clarifier_agent import ClarifierAgent
from agents.critic_agent import CriticAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.planner_agent import PlannerAgent
from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent
from config import settings
from llm.llm_factory import LLMFactory
from memory.context_manager import ContextManager
from memory.memory_manager import MemoryManager
from memory.vector_store import VectorStore
from workflows.dag_engine import END, Workflow
from workflows.state import ResearchState
from workflows.interrupts import WorkflowPaused

logger = logging.getLogger(__name__)


class ResearchWorkflow:
    """
    研究工作流

    使用自研 DAG 引擎构建如下工作流：

    ┌───────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐
    │ Clarifier │───>│ Planner  │───>│Orchestrator  │───>│  Critic  │───>│  Writer  │──> END
    └───────────┘    └──────────┘    │(并行调度)    │    └────┬─────┘    └──────────┘
                                     │  ┌─────────┐│         │
                                     │  │Sub-Agent1││         │ 未通过
                                     │  │Sub-Agent2││         v
                                     │  │Sub-Agent3││    ┌──────────────┐
                                     │  └─────────┘│    │Orchestrator  │ (重试)
                                     └──────────────┘    └──────────────┘

    状态流转：
    - clarifier → planner: 意图澄清完成后进入规划阶段
    - planner → orchestrator: 规划完成后进入调度阶段
    - orchestrator → critic: 所有子问题研究完成后进入审查阶段
    - critic → writer: 审查通过，进入写作阶段
    - critic → orchestrator: 审查未通过，回退到调度阶段（重试）
    - writer → END: 报告生成完成
    """

    def __init__(
        self,
        session_id: str | None = None,
        interaction_handler: Any = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        """
        初始化研究工作流

        Args:
            session_id: 会话 ID，用于记忆隔离。
                        为 None 则从配置读取（默认自动生成 UUID）。
        """
        self.logger = logging.getLogger("ResearchWorkflow")
        self.event_callback = event_callback

        self.session_id = session_id or settings.effective_session_id
        self.logger.info(f"会话 ID: {self.session_id}")

        self.llm = LLMFactory.create()

        # Critic 使用独立 LLM，避免同模型自审自导致偏执
        critic_provider = settings.critic_llm_provider or settings.llm_provider
        self.critic_llm = LLMFactory.create(critic_provider)
        self.logger.info(f"Critic 使用独立 LLM: {critic_provider}")

        self.context_manager = ContextManager(llm=self.llm)

        self.clarifier = ClarifierAgent(
            llm=self.llm, interaction_handler=interaction_handler
        )
        self.planner = PlannerAgent(llm=self.llm)
        self.researcher = ResearcherAgent(llm=self.llm, context_manager=self.context_manager)
        self.orchestrator = OrchestratorAgent(
            llm=self.llm,
            context_manager=self.context_manager,
            researcher=self.researcher,
        )
        self.critic = CriticAgent(llm=self.critic_llm, context_manager=self.context_manager)
        self.writer = WriterAgent(
            llm=self.llm,
            context_manager=self.context_manager,
            interaction_handler=interaction_handler,
        )

        for agent in (
            self.clarifier,
            self.planner,
            self.researcher,
            self.orchestrator,
            self.critic,
            self.writer,
        ):
            agent.event_callback = event_callback

        self.vector_store = VectorStore(
            persist_dir=settings.chroma_persist_dir,
            session_id=self.session_id,
        )
        self.memory_manager = MemoryManager(vector_store=self.vector_store)

        self.workflow = self._build_workflow()

        self.logger.info(f"ResearchWorkflow 初始化完成，会话ID: {self.session_id}")

    def _emit_event(self, event_type: str, **payload: Any) -> None:
        if not self.event_callback:
            return
        try:
            self.event_callback({"type": event_type, **payload})
        except Exception as exc:
            self.logger.debug("工作流事件回调失败: %s", exc)

    def _collect_token_usage(self) -> dict[str, Any]:
        """Collect approximate prompt/output usage from workflow agents."""
        agents = [
            self.clarifier,
            self.planner,
            self.researcher,
            self.orchestrator,
            self.critic,
            self.writer,
        ]
        if self.orchestrator._sub_agent is not None:
            agents.append(self.orchestrator._sub_agent)

        by_agent: dict[str, dict[str, int]] = {}
        for agent in agents:
            usage = dict(agent.token_usage)
            if not usage.get("calls"):
                continue
            bucket = by_agent.setdefault(
                agent.name,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0},
            )
            for key in bucket:
                bucket[key] += int(usage.get(key, 0))

        compression_usage = dict(self.context_manager.token_usage)
        if compression_usage.get("calls"):
            by_agent["ContextCompression"] = {
                key: int(compression_usage.get(key, 0))
                for key in ("calls", "input_tokens", "output_tokens")
            }

        return {
            "estimated": True,
            "by_agent": by_agent,
            "total_calls": sum(item["calls"] for item in by_agent.values()),
            "input_tokens": sum(item["input_tokens"] for item in by_agent.values()),
            "output_tokens": sum(item["output_tokens"] for item in by_agent.values()),
        }

    def _reset_token_usage(self) -> None:
        for agent in (
            self.clarifier,
            self.planner,
            self.researcher,
            self.orchestrator,
            self.critic,
            self.writer,
        ):
            agent.token_usage = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        if self.orchestrator._sub_agent is not None:
            self.orchestrator._sub_agent.token_usage = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        self.context_manager.token_usage = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def _build_workflow(self) -> Workflow:
        """
        构建 DAG 工作流

        定义节点和边：
        - 节点：clarifier, planner, orchestrator, critic, writer
        - 条件边：critic 的输出决定下一步是 writer 还是 orchestrator

        Returns:
            配置好的 Workflow 实例
        """
        workflow = Workflow(max_steps=20)

        workflow.add_node("clarifier", self._clarifier_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("critic", self._critic_node)
        workflow.add_node("writer", self._writer_node)

        workflow.set_entry_point("clarifier")

        workflow.add_edge("clarifier", "planner")
        workflow.add_edge("planner", "orchestrator")
        workflow.add_edge("orchestrator", "critic")

        workflow.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {
                "writer": "writer",
                "orchestrator": "orchestrator",
            },
        )

        workflow.add_edge("writer", END)

        self.logger.info("DAG 工作流构建完成")
        return workflow

    async def _clarifier_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Clarifier 节点

        调用 Clarifier Agent 与用户交互，澄清研究意图。

        Args:
            state: 当前工作流状态

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 40)
        self.logger.info("📍 进入 Clarifier 阶段")
        self.logger.info("=" * 40)
        self._emit_event("phase_started", phase="clarifier", label="意图澄清")

        state = await self.clarifier.run(state)

        clarified_intent = state.get("clarified_intent", "")
        if clarified_intent:
            # 意图澄清是"小而不可再生"的关键事实：跨阶段保留并注入 Critic/Writer 的 prompt
            self.context_manager.add_context(
                role="system",
                content=f"意图澄清: {clarified_intent}",
                is_key=True,
                query=state.get("query", ""),
            )

        await self.context_manager.maybe_compress()
        self._emit_event("phase_completed", phase="clarifier")
        return state

    async def _planner_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Planner 节点

        调用 Planner Agent 拆解研究问题，输出带依赖关系的结构化子问题。

        Args:
            state: 当前工作流状态

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 40)
        self.logger.info("📍 进入 Planner 阶段")
        self.logger.info("=" * 40)
        self._emit_event("phase_started", phase="planner", label="研究规划")

        try:
            await self.memory_manager.initialize()
            history = await self.memory_manager.recall_research(
                query=state.get("query", ""), top_k=2
            )
            if history:
                self.logger.info(f"找到 {len(history)} 条历史研究记录")
                history_text = "\n".join(
                    [h.get("content", "")[:500] for h in history]
                )
                state["history_context"] = history_text
        except Exception as e:
            self.logger.warning(f"历史研究检索失败: {e}")

        if not settings.orchestrator_enabled:
            state["use_orchestrator"] = False

        state = await self.planner.run(state)

        # 研究计划同样是关键事实：体积小、不可再生，后续 Critic/Writer 都要遵守
        self.context_manager.add_context(
            role="system",
            content=f"研究计划: {state.get('research_plan', '')}",
            is_key=True,
            query=state.get("query", ""),
        )

        structured = state.get("structured_sub_questions", [])
        if structured:
            self.logger.info(f"Planner 输出 {len(structured)} 个结构化子问题")
            for sq in structured:
                deps = f"依赖{sq.depends_on}" if sq.depends_on else "无依赖"
                self.logger.info(f"  [{sq.id}] {sq.question} ({deps})")

        await self.context_manager.maybe_compress()
        self._emit_event(
            "phase_completed",
            phase="planner",
            sub_questions=[
                sq.model_dump() if hasattr(sq, "model_dump") else sq
                for sq in state.get("structured_sub_questions", [])
            ],
            research_plan=state.get("research_plan", ""),
            plan_coverage=state.get("plan_coverage", []),
            plan_assumptions=state.get("plan_assumptions", []),
            planner_self_check=state.get("planner_self_check", {}),
        )
        return state

    async def _orchestrator_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Orchestrator 节点

        调用 Orchestrator Agent 进行子问题调度和并行研究。

        Args:
            state: 当前工作流状态

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 40)
        self.logger.info("📍 进入 Orchestrator 阶段")
        self.logger.info("=" * 40)
        self._emit_event("phase_started", phase="research", label="并行研究")

        # 在 orchestrator 运行前，检索缓存的搜索结果和论文，避免重复搜索
        try:
            cached_search = await self.memory_manager.recall_search_results(
                query=state.get("query", ""), top_k=5
            )
            if cached_search:
                self.logger.info(f"找到 {len(cached_search)} 条缓存搜索结果")
                cached_search_text = "\n".join(
                    [c.get("content", "")[:300] for c in cached_search]
                )
                state["cached_search_context"] = cached_search_text

            cached_papers = await self.memory_manager.recall_papers(
                query=state.get("query", ""), top_k=5
            )
            if cached_papers:
                self.logger.info(f"找到 {len(cached_papers)} 篇缓存论文")
                cached_papers_text = "\n".join(
                    [c.get("content", "")[:300] for c in cached_papers]
                )
                state["cached_paper_context"] = cached_papers_text
        except Exception as e:
            self.logger.warning(f"缓存检索失败: {e}")

        critique_feedback = state.get("critique_feedback", "")
        failed_sub_question_ids = state.get("failed_sub_question_ids", [])
        if critique_feedback and state.get("retry_count", 0) > 0:
            # 重试时从 context_manager 提取历轮研究发现（self.context 的真正消费点），
            # 让重试的 sub-agent 在上一轮基础上修正补充，而不是从零重做
            state["prior_round_findings"] = await self.context_manager.get_rounds_findings()
            if state["prior_round_findings"]:
                self.logger.info(
                    f"已提取历轮研究发现供重试参考 "
                    f"({len(state['prior_round_findings'])} 字符)"
                )
            if failed_sub_question_ids:
                self.logger.info(f"携带逐子问题审查反馈（{len(failed_sub_question_ids)} 个不合格子问题）")
            else:
                self.logger.info(f"携带上一轮审查反馈（{len(critique_feedback)} 字符）")
                compressed_feedback = await self.context_manager.compress_text(
                    critique_feedback,
                    instruction=(
                        "请将以下审查反馈压缩为更短的版本，保留：\n"
                        "1. 具体的问题描述\n"
                        "2. 修正建议\n"
                        "3. 建议搜索的关键词\n"
                        "去除笼统的评价，用中文输出"
                    ),
                )
                state["critique_feedback_for_researcher"] = compressed_feedback

        state = await self.orchestrator.run(state)

        try:
            search_results = state.get("search_results", [])
            paper_results = state.get("paper_results", [])
            if search_results:
                await self.memory_manager.save_search_results(
                    query=state.get("query", ""),
                    results=search_results,
                )
            if paper_results:
                await self.memory_manager.save_papers(
                    query=state.get("query", ""),
                    papers=paper_results,
                )
        except Exception as e:
            self.logger.warning(f"搜索结果缓存失败: {e}")

        findings_text = state.get('findings', '')
        if findings_text and self.context_manager:
            findings_tokens = self.context_manager.estimate_text_tokens(findings_text)
            if findings_tokens > self.context_manager.get_compress_threshold("findings"):
                self.logger.info(f"研究发现过长 ({findings_tokens} tokens)，压缩后存入上下文...")
                findings_text = await self.context_manager.compress_findings(findings_text)

        # 研究发现体积最大、且 state["findings"] 已持有同一份内容，
        # 因此标记为非关键信息：允许被去重/修剪/摘要，避免上下文只增不减。
        self.context_manager.add_context(
            role="system",
            content=f"研究发现: {findings_text}",
            is_key=False,
            query=state.get("query", ""),
        )

        use_orchestrator = state.get("use_orchestrator", True)
        sub_results = state.get("sub_question_results", [])
        if use_orchestrator and sub_results:
            self.logger.info(f"Orchestrator 完成：{len(sub_results)} 个子问题结果")
        else:
            self.logger.info("串行 Researcher 完成")

        await self.context_manager.maybe_compress()
        self._emit_event(
            "phase_completed",
            phase="research",
            source_count=len(state.get("sources", [])),
            sub_question_results=[
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in state.get("sub_question_results", [])
            ],
        )
        return state

    async def _critic_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Critic 节点

        调用 Critic Agent 审查研究发现。

        Args:
            state: 当前工作流状态

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 40)
        self.logger.info("📍 进入 Critic 阶段")
        self.logger.info("=" * 40)
        self._emit_event("phase_started", phase="critic", label="质量审查")

        state = await self.critic.run(state)

        self._emit_event(
            "phase_completed",
            phase="critic",
            score=state.get("critique_total_score", 0),
            passed=state.get("critique_passed", False),
            outcome=state.get("critique_outcome", ""),
            feedback=state.get("critique_feedback", ""),
            writer_revision_suggestions=state.get(
                "writer_revision_suggestions", []
            ),
        )

        return state

    async def _writer_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Writer 节点

        调用 Writer Agent 生成研究报告。

        Args:
            state: 当前工作流状态

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 40)
        self.logger.info("📍 进入 Writer 阶段")
        self.logger.info("=" * 40)
        self._emit_event("phase_started", phase="writer", label="报告撰写")

        state = await self.writer.run(state)

        try:
            await self.memory_manager.save_research(
                topic=state.get("query", ""),
                content=state.get("report", ""),
                metadata={
                    "critique_passed": state.get("critique_passed", False),
                    "retry_count": state.get("retry_count", 0),
                },
            )
        except Exception as e:
            self.logger.warning(f"研究记录保存失败: {e}")

        self.logger.info("=" * 40)
        self.logger.info("✅ 研究工作流完成！")
        self.logger.info("=" * 40)
        self._emit_event(
            "phase_completed",
            phase="writer",
            report=state.get("report", ""),
        )

        return state

    def _route_after_critic(self, state: dict[str, Any]) -> Literal["writer", "orchestrator"]:
        """
        Critic 节点后的路由函数

        根据审查结果决定下一步：
        - 审查通过 → writer
        - 审查未通过 → orchestrator（重试）

        Args:
            state: 当前工作流状态

        Returns:
            下一个节点名称
        """
        current_step = state.get("current_step", "writer")

        if current_step == "writer":
            self.logger.info("🔀 路由: Critic → Writer（审查通过）")
            return "writer"
        else:
            self.logger.info("🔀 路由: Critic → Orchestrator（审查未通过，重试）")
            return "orchestrator"

    @staticmethod
    def initial_state(query: str, session_id: str) -> dict[str, Any]:
        """Build the serializable state used by both CLI and durable workers."""
        return {
            "query": query,
            "session_id": session_id,
            "clarified_intent": "",
            "intent_profile": {},
            "clarification_qa": [],
            "clarification_round": 0,
            "pending_clarification_questions": [],
            "history_context": "",
            "sub_questions": [],
            "structured_sub_questions": [],
            "search_keywords": [],
            "research_plan": "",
            "plan_coverage": [],
            "plan_assumptions": [],
            "planner_self_check": {},
            "orchestrator_plan": {},
            "sub_question_results": [],
            "search_results": [],
            "paper_results": [],
            "findings": "",
            "claims": [],
            "sources": [],
            "evidence_cards": [],
            "token_usage": {},
            "critique_passed": False,
            "critique_feedback": "",
            "critique_feedback_for_researcher": "",
            "prior_round_findings": "",
            "revision_suggestions": [],
            "critique_scores": {},
            "critique_score_details": {},
            "critique_total_score": 0,
            "per_sub_question_critiques": {},
            "failed_sub_question_ids": [],
            "overall_coherence_score": 0,
            "overall_coherence_feedback": "",
            "overall_coherence_passed": False,
            "critique_outcome": "",
            "writer_revision_suggestions": [],
            "missing_research_topics": [],
            "report": "",
            "citation_integrity_issues": [],
            "report_outline": "",
            "report_outline_display": "",
            "report_outline_payload": {},
            "user_outline_feedback": "",
            "current_step": "clarifier",
            "retry_count": 0,
            "max_retries": settings.critique_max_retries,
            "use_orchestrator": settings.orchestrator_enabled,
            "error": "",
        }

    def restore_context(self, state: dict[str, Any]) -> None:
        """Rebuild ephemeral prompt context from a durable state snapshot."""
        self.context_manager.clear()
        clarified_intent = state.get("clarified_intent", "")
        if clarified_intent:
            self.context_manager.add_context(
                role="system",
                content=f"意图澄清: {clarified_intent}",
                is_key=True,
                query=state.get("query", ""),
            )
        research_plan = state.get("research_plan", "")
        if research_plan:
            self.context_manager.add_context(
                role="system",
                content=f"研究计划: {research_plan}",
                is_key=True,
                query=state.get("query", ""),
            )
        findings = state.get("findings", "")
        if findings:
            self.context_manager.add_context(
                role="system",
                content=f"研究发现: {findings}",
                is_key=False,
                query=state.get("query", ""),
            )

    async def execute_phase(
        self, state: dict[str, Any], phase: str
    ) -> tuple[dict[str, Any], str | None]:
        """Execute exactly one durable phase and return its successor."""
        nodes = {
            "clarifier": self._clarifier_node,
            "planner": self._planner_node,
            "orchestrator": self._orchestrator_node,
            "critic": self._critic_node,
            "writer": self._writer_node,
        }
        if phase not in nodes:
            raise ValueError(f"unknown workflow phase: {phase}")
        self.restore_context(state)
        self._reset_token_usage()
        state["current_step"] = phase
        state = await nodes[phase](state)

        usage = self._collect_token_usage()
        previous_usage = state.get("token_usage", {}) or {}
        by_agent = {
            name: dict(values)
            for name, values in previous_usage.get("by_agent", {}).items()
        }
        for name, values in usage["by_agent"].items():
            bucket = by_agent.setdefault(
                name, {"calls": 0, "input_tokens": 0, "output_tokens": 0}
            )
            for key in ("calls", "input_tokens", "output_tokens"):
                bucket[key] = int(bucket.get(key, 0)) + int(values.get(key, 0))
        state["token_usage"] = {
            "estimated": True,
            "total_calls": int(previous_usage.get("total_calls", 0)) + usage["total_calls"],
            "input_tokens": int(previous_usage.get("input_tokens", 0)) + usage["input_tokens"],
            "output_tokens": int(previous_usage.get("output_tokens", 0)) + usage["output_tokens"],
            "by_agent": by_agent,
        }

        if phase == "critic":
            next_phase = self._route_after_critic(state)
        else:
            next_phase = {
                "clarifier": "planner",
                "planner": "orchestrator",
                "orchestrator": "critic",
                "writer": None,
            }[phase]
        state["current_step"] = next_phase or "completed"
        return state, next_phase

    async def run(self, query: str) -> ResearchState:
        """
        执行完整研究工作流

        Args:
            query: 用户研究问题

        Returns:
            最终的研究状态（包含报告）
        """
        self.logger.info(f"🔬 开始深度研究: {query}")
        self._reset_token_usage()

        # 会话隔离：清空上一轮研究遗留的上下文与关键信息，避免跨问题串味
        self.context_manager.clear()

        initial_state = self.initial_state(query, self.session_id)

        try:
            final_state = await self.workflow.run(initial_state)
            final_state["token_usage"] = self._collect_token_usage()

            result = ResearchState(**final_state)
            return result

        except WorkflowPaused:
            raise
        except Exception as e:
            self.logger.error(f"工作流执行失败: {e}")
            return ResearchState(
                query=query,
                error=str(e),
                current_step="error",
            )
