"""
Orchestrator Agent —— 主控调度

负责根据 Planner 输出的带依赖关系的子问题，
进行拓扑排序、并行调度、上下文传递和结果汇总。

核心能力：
- 依赖分析：解析子问题依赖关系，构建执行 DAG
- 拓扑排序：确定执行层级（同一层可并行，不同层串行）
- 并行调度：同层无依赖的子问题并发执行
- 上下文传递：前置子问题的发现作为后续子问题的输入
- 动态调整：根据中间结果判断是否需要追加子问题（LLM 判断）
- 结果汇总：收集所有子 Agent 结果，按子问题维度组织
- 回退兼容：子问题数为1或依赖解析失败时回退到串行 Researcher
"""

import asyncio
import json
import re
from collections import defaultdict
from typing import Any

from agents.base_agent import BaseAgent
from agents.researcher_agent import ResearcherAgent
from agents.researcher_sub_agent import ResearcherSubAgent
from llm.base_llm import BaseLLM
from memory.context_manager import ContextManager
from search.search_factory import SearchFactory
from academic.academic_factory import AcademicFactory
from config import settings
from workflows.state import SubQuestion, SubQuestionResult
from workflows.evidence import deduplicate_cards


ORCHESTRATOR_ADJUST_PROMPT = """你是一个研究调度专家。根据以下已完成子问题的研究结果，判断是否需要追加新的子问题。

原始研究问题：{query}
研究计划：{research_plan}

已完成的子问题及发现：
{completed_findings}

剩余待研究的子问题：
{remaining_questions}

请判断：
1. 当前已完成的研究是否足以回答原始问题？
2. 是否存在重要的信息缺口需要追加新的子问题？

你的输出必须严格遵循以下 JSON 格式，不要输出任何其他内容：
{{
    "need_more": true/false,
    "reason": "判断理由",
    "new_sub_questions": [
        {{
            "id": "sq_extra_1",
            "question": "追加的子问题",
            "depends_on": ["依赖的子问题id"],
            "priority": 优先级数字,
            "keywords_zh": ["中文关键词"],
            "keywords_en": ["英文关键词"]
        }}
    ]
}}

如果不需要追加子问题，new_sub_questions 为空列表即可。

重要规则：
- 追加的子问题不得与已完成或剩余待研究的子问题语义重复
- 追加的子问题的搜索关键词应与已有子问题的关键词有所区别
- 严格控制追加数量，仅在存在关键信息缺口时追加，每次最多追加 {max_new_per_round} 个子问题
"""


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent —— 主控调度

    工作流程：
    1. 读取 Planner 输出的结构化子问题
    2. 分析依赖关系，拓扑排序确定执行层级
    3. 按层级调度 Researcher Sub-Agent（同层并行）
    4. 前置子问题的发现传递给后续子问题
    5. 可选：根据中间结果动态追加子问题
    6. 汇总所有结果，写入工作流状态
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        context_manager: ContextManager | None = None,
        researcher: ResearcherAgent | None = None,
    ):
        super().__init__(name="Orchestrator", llm=llm, context_manager=context_manager)
        self.researcher = researcher
        self._sub_agent: ResearcherSubAgent | None = None
 
    def _get_sub_agent(self) -> ResearcherSubAgent:
        if self._sub_agent is None:
            search_engine = SearchFactory.create()
            academic_search = AcademicFactory.create()
            self._sub_agent = ResearcherSubAgent(
                llm=self.llm,
                search_engine=search_engine,
                academic_search=academic_search,
                context_manager=self.context_manager,
            )
            self._sub_agent.event_callback = self.event_callback
        return self._sub_agent

    async def _execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        执行主控调度

        Args:
            state: 工作流状态字典

        Returns:
            更新后的状态字典
        """
        failed_sub_question_ids = state.get("failed_sub_question_ids", [])
        is_retry = state.get("retry_count", 0) > 0 and failed_sub_question_ids

        if is_retry:
            return await self._selective_retry(state, failed_sub_question_ids)

        use_orchestrator = state.get("use_orchestrator", True)
        structured_sub_questions = state.get("structured_sub_questions", [])

        if settings.sub_question_overlap_threshold > 0 and len(structured_sub_questions) > 1:
            original_count = len(structured_sub_questions)
            structured_sub_questions = self._filter_overlapping_sqs(
                structured_sub_questions, []
            )
            if len(structured_sub_questions) < original_count:
                self.log(f"子问题去重: {original_count} → {len(structured_sub_questions)}")

        if not use_orchestrator or len(structured_sub_questions) <= 1:
            self.log("子问题数 ≤ 1 或未启用 Orchestrator，回退到串行 Researcher")
            return await self._fallback_to_serial_researcher(state)

        self.log(f"开始 Orchestrator 调度，共 {len(structured_sub_questions)} 个子问题")

        try:
            execution_layers = self._topological_sort(structured_sub_questions)
        except ValueError as e:
            self.log(f"⚠️ 依赖解析失败: {e}，回退到串行 Researcher")
            return await self._fallback_to_serial_researcher(state)

        self._log_execution_plan(execution_layers)

        state["orchestrator_plan"] = {
            "layers": [
                {"layer": i, "sub_question_ids": [sq.id for sq in layer]}
                for i, layer in enumerate(execution_layers)
            ],
            "total_sub_questions": len(structured_sub_questions),
            "total_layers": len(execution_layers),
        }

        completed_results: dict[str, SubQuestionResult] = {}
        max_concurrent = settings.orchestrator_max_concurrent
        dynamic_adjust_rounds = 0
        max_dynamic_rounds = settings.orchestrator_dynamic_adjust_max_rounds
        max_new_per_round = settings.orchestrator_dynamic_adjust_max_new_per_round
        max_total_appends = settings.orchestrator_dynamic_adjust_max_total_appends
        total_appended = 0
        search_cache: dict[str, list] = {}

        layer_idx = 0
        while layer_idx < len(execution_layers):
            layer = execution_layers[layer_idx]
            self.log(f"执行第 {layer_idx + 1}/{len(execution_layers)} 层，"
                     f"共 {len(layer)} 个子问题")

            tasks = []
            for sq in layer:
                prior_findings = self._collect_prior_findings(sq, completed_results)
                tasks.append((sq, prior_findings))

            if len(tasks) == 1:
                sq, prior = tasks[0]
                result = await self._execute_single_sub_question(
                    sq, state, prior, search_cache
                )
                completed_results[sq.id] = result
            else:
                semaphore = asyncio.Semaphore(max_concurrent)

                async def _run_with_semaphore(
                    sq: SubQuestion, prior: dict[str, str]
                ) -> tuple[str, SubQuestionResult]:
                    async with semaphore:
                        result = await self._execute_single_sub_question(
                            sq, state, prior, search_cache
                        )
                        return sq.id, result

                layer_results = await asyncio.gather(
                    *[_run_with_semaphore(sq, prior) for sq, prior in tasks],
                    return_exceptions=True,
                )

                for lr in layer_results:
                    if isinstance(lr, Exception):
                        self.log(f"⚠️ 子问题执行失败: {lr}")
                        continue
                    sq_id, result = lr
                    completed_results[sq_id] = result

            is_last_planned_layer = layer_idx == len(execution_layers) - 1
            if (
                settings.orchestrator_dynamic_adjust
                and is_last_planned_layer
                and dynamic_adjust_rounds < max_dynamic_rounds
                and total_appended < max_total_appends
            ):
                new_sqs = await self._dynamic_adjust(
                    state, completed_results, execution_layers, layer_idx
                )
                if new_sqs:
                    # 单次截断：每次最多追加 max_new_per_round 个
                    new_sqs = new_sqs[:max_new_per_round]
                    # 总量截断：不超过剩余可追加数量
                    remaining = max_total_appends - total_appended
                    new_sqs = new_sqs[:remaining]
                    if new_sqs:
                        dynamic_adjust_rounds += 1
                        total_appended += len(new_sqs)
                        self.log(f"  本次追加 {len(new_sqs)} 个子问题，"
                                 f"累计追加 {total_appended}/{max_total_appends}")
                        self._rebuild_remaining_layers(
                            execution_layers, layer_idx + 1, new_sqs, completed_results
                        )
            elif total_appended >= max_total_appends:
                self.log(f"动态调整已达总追加上限 {max_total_appends}，不再追加子问题")
            elif dynamic_adjust_rounds >= max_dynamic_rounds:
                self.log(f"动态调整已达最大追加次数 {max_dynamic_rounds}，不再追加子问题")

            layer_idx += 1

        self._merge_results_to_state(state, completed_results)

        state["current_step"] = "critic"
        return state

    async def _execute_single_sub_question(
        self,
        sub_question: SubQuestion,
        state: dict[str, Any],
        prior_findings: dict[str, str],
        search_cache: dict[str, list] | None = None,
        critique_feedback: str = "",
        extra_keywords: dict[str, list[str]] | None = None,
    ) -> SubQuestionResult:
        """
        执行单个子问题的研究

        Args:
            sub_question: 子问题
            state: 工作流状态
            prior_findings: 前置子问题的发现
            search_cache: 跨子问题的搜索缓存，避免重复搜索相同关键词
            critique_feedback: 上一轮审查反馈，用于重试时改进
            extra_keywords: 重试时根据评审反馈生成的补充搜索关键词

        Returns:
            SubQuestionResult
        """
        sub_agent = self._get_sub_agent()

        # 构造缓存上下文
        cached_parts = []
        cached_search = state.get("cached_search_context", "")
        cached_papers = state.get("cached_paper_context", "")
        if cached_search:
            cached_parts.append(f"【缓存网页结果】\n{cached_search}")
        if cached_papers:
            cached_parts.append(f"【缓存论文结果】\n{cached_papers}")
        cached_context = "\n\n".join(cached_parts) if cached_parts else None

        # 历轮研究发现（重试时由工作流从 ContextManager 提取）
        prior_round_findings = state.get("prior_round_findings", "")

        result = await sub_agent.run(
            sub_question=sub_question,
            query=state.get("query", ""),
            research_plan=state.get("research_plan", ""),
            prior_findings=prior_findings if prior_findings else None,
            search_cache=search_cache,
            critique_feedback=critique_feedback if critique_feedback else None,
            cached_context=cached_context,
            extra_keywords=extra_keywords,
            prior_round_findings=prior_round_findings if prior_round_findings else None,
        )
        return result

    def _topological_sort(
        self,
        sub_questions: list[SubQuestion],
        satisfied_dependency_ids: set[str] | None = None,
    ) -> list[list[SubQuestion]]:
        """
        对子问题进行拓扑排序，返回按层级组织的执行计划

        同一层内的子问题互相无依赖，可以并行执行。
        不同层之间有依赖关系，必须按顺序执行。

        Args:
            sub_questions: 子问题列表
            satisfied_dependency_ids: 已在前序阶段完成的依赖 ID。这些依赖
                不参与本次排序，但应视为已经满足。

        Returns:
            按层级组织的子问题列表，外层列表表示执行层级

        Raises:
            ValueError: 存在循环依赖
        """
        sq_map = {sq.id: sq for sq in sub_questions}
        satisfied_dependency_ids = satisfied_dependency_ids or set()
        in_degree: dict[str, int] = defaultdict(int)
        dependents: dict[str, list[str]] = defaultdict(list)

        for sq in sub_questions:
            if sq.id not in in_degree:
                in_degree[sq.id] = 0
            for dep_id in sq.depends_on:
                if dep_id not in sq_map:
                    if dep_id in satisfied_dependency_ids:
                        continue
                    self.log(f"⚠️ 子问题 {sq.id} 依赖的 {dep_id} 不存在，忽略该依赖")
                    continue
                in_degree[sq.id] += 1
                dependents[dep_id].append(sq.id)

        layers: list[list[SubQuestion]] = []
        remaining = set(sq.id for sq in sub_questions)
        processed: set[str] = set()

        while remaining:
            current_layer_ids = [
                sq_id for sq_id in remaining if in_degree[sq_id] == 0
            ]

            if not current_layer_ids:
                raise ValueError(
                    f"检测到循环依赖，未处理的子问题: {remaining}"
                )

            current_layer = [sq_map[sq_id] for sq_id in current_layer_ids]
            current_layer.sort(key=lambda sq: (sq.priority, sq.id))
            layers.append(current_layer)

            for sq_id in current_layer_ids:
                processed.add(sq_id)
                remaining.discard(sq_id)
                for dependent_id in dependents[sq_id]:
                    if dependent_id in in_degree:
                        in_degree[dependent_id] -= 1

        return layers

    def _collect_prior_findings(
        self,
        sub_question: SubQuestion,
        completed_results: dict[str, SubQuestionResult],
    ) -> dict[str, str]:
        """
        收集前置子问题的发现

        Args:
            sub_question: 当前子问题
            completed_results: 已完成的子问题结果

        Returns:
            前置子问题发现的字典，key 为子问题 id，value 为发现文本
        """
        prior_findings: dict[str, str] = {}
        for dep_id in sub_question.depends_on:
            if dep_id in completed_results:
                prior_findings[dep_id] = completed_results[dep_id].findings
        return prior_findings

    def _tokenize(self, text: str) -> set[str]:
        tokens = set()
        text_lower = text.lower()
        for word in re.findall(r'[a-z0-9]{2,}', text_lower):
            tokens.add(word)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text_lower)
        for i in range(len(chinese_chars) - 1):
            tokens.add(chinese_chars[i] + chinese_chars[i + 1])
        return tokens

    def _compute_overlap(self, sq1: SubQuestion, sq2: SubQuestion) -> float:
        kw1 = set(k.lower() for k in sq1.keywords_zh + sq1.keywords_en)
        kw2 = set(k.lower() for k in sq2.keywords_zh + sq2.keywords_en)
        keyword_jaccard = 0.0
        if kw1 and kw2:
            keyword_jaccard = len(kw1 & kw2) / len(kw1 | kw2)

        tokens1 = self._tokenize(sq1.question)
        tokens2 = self._tokenize(sq2.question)
        text_jaccard = 0.0
        if tokens1 and tokens2:
            text_jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)

        return max(keyword_jaccard, text_jaccard)

    def _filter_overlapping_sqs(
        self,
        new_sqs: list[SubQuestion],
        existing_sqs: list[SubQuestion],
    ) -> list[SubQuestion]:
        all_existing = list(existing_sqs)
        keep: list[SubQuestion] = []
        for sq in new_sqs:
            is_duplicate = False
            for existing in all_existing:
                if self._compute_overlap(sq, existing) >= settings.sub_question_overlap_threshold:
                    is_duplicate = True
                    self.log(f"  过滤重叠子问题: [{sq.id}] {sq.question} "
                             f"(与 [{existing.id}] {existing.question} 重叠)")
                    break
            if not is_duplicate:
                keep.append(sq)
                all_existing.append(sq)
        return keep

    def _collect_all_existing_sqs(
        self,
        execution_layers: list[list[SubQuestion]],
        completed_results: dict[str, SubQuestionResult],
    ) -> list[SubQuestion]:
        existing: list[SubQuestion] = []
        for sq_id, result in completed_results.items():
            existing.append(SubQuestion(
                id=sq_id,
                question=result.sub_question,
                keywords_zh=[],
                keywords_en=[],
            ))
        for layer in execution_layers:
            for sq in layer:
                existing.append(sq)
        return existing

    async def _dynamic_adjust(
        self,
        state: dict[str, Any],
        completed_results: dict[str, SubQuestionResult],
        execution_layers: list[list[SubQuestion]],
        current_layer_idx: int,
    ) -> list[SubQuestion]:
        """
        动态调整：根据已完成的研究结果，判断是否需要追加子问题

        使用 LLM 判断是否存在重要信息缺口需要补充研究。

        Args:
            state: 工作流状态
            completed_results: 已完成的子问题结果
            execution_layers: 执行层级
            current_layer_idx: 当前完成的层级索引

        Returns:
            需要追加的新子问题列表（空列表表示无需追加）
        """
        completed_findings_parts = []
        for sq_id, result in completed_results.items():
            completed_findings_parts.append(
                f"- [{sq_id}] {result.sub_question}: {result.findings[:200]}"
            )
        completed_findings = "\n".join(completed_findings_parts)

        remaining_parts = []
        for layer_idx in range(current_layer_idx + 1, len(execution_layers)):
            for sq in execution_layers[layer_idx]:
                remaining_parts.append(f"- [{sq.id}] {sq.question}")
        remaining_questions = "\n".join(remaining_parts) if remaining_parts else "无"

        prompt = ORCHESTRATOR_ADJUST_PROMPT.format(
            query=state.get("query", ""),
            research_plan=state.get("research_plan", ""),
            completed_findings=completed_findings,
            remaining_questions=remaining_questions,
            max_new_per_round=settings.orchestrator_dynamic_adjust_max_new_per_round,
        )

        try:
            self.log("动态调整：调用 LLM 判断是否需要追加子问题...")
            response = await self.generate(
                prompt=prompt,
                system_prompt="你是一个研究调度专家，只输出 JSON。",
                max_tokens=settings.orchestrator_max_tokens,
            )

            adjust_data = self._parse_json_response(response)
            new_sub_questions: list[SubQuestion] = []
            if adjust_data and adjust_data.get("need_more"):
                new_sqs = adjust_data.get("new_sub_questions", [])
                for sq_data in new_sqs:
                    new_sq = SubQuestion(
                        id=sq_data.get("id", f"sq_extra_{len(completed_results) + len(new_sub_questions) + 1}"),
                        question=sq_data.get("question", ""),
                        depends_on=sq_data.get("depends_on", []),
                        priority=sq_data.get("priority", 99),
                        keywords_zh=sq_data.get("keywords_zh", []),
                        keywords_en=sq_data.get("keywords_en", []),
                    )
                    if new_sq.question:
                        new_sub_questions.append(new_sq)

                if settings.sub_question_overlap_threshold > 0 and new_sub_questions:
                    existing_sqs = self._collect_all_existing_sqs(
                        execution_layers, completed_results
                    )
                    before_count = len(new_sub_questions)
                    new_sub_questions = self._filter_overlapping_sqs(
                        new_sub_questions, existing_sqs
                    )
                    if len(new_sub_questions) < before_count:
                        self.log(f"  动态追加去重: {before_count} → {len(new_sub_questions)}")

                for sq in new_sub_questions:
                    self.log(f"  追加子问题: [{sq.id}] {sq.question}")
            else:
                self.log("动态调整：无需追加子问题")

            return new_sub_questions

        except Exception as e:
            self.log(f"⚠️ 动态调整失败: {e}，继续执行原计划")
            return []

    def _rebuild_remaining_layers(
        self,
        execution_layers: list[list[SubQuestion]],
        next_layer_idx: int,
        new_sub_questions: list[SubQuestion],
        completed_results: dict[str, SubQuestionResult],
    ) -> None:
        """
        重建剩余执行层级：将剩余子问题与新追加的子问题合并后重新拓扑排序

        确保新子问题根据其 depends_on 被放置到正确的层级，
        而不是简单地追加到最后一层。

        Args:
            execution_layers: 当前执行层级列表（会被原地修改）
            next_layer_idx: 下一个待执行层的索引
            new_sub_questions: 动态追加的新子问题
            completed_results: 已完成的子问题结果
        """
        remaining_sqs: list[SubQuestion] = []
        for layer in execution_layers[next_layer_idx:]:
            remaining_sqs.extend(layer)

        remaining_sqs.extend(new_sub_questions)

        try:
            new_layers = self._topological_sort(
                remaining_sqs,
                satisfied_dependency_ids=set(completed_results),
            )
            execution_layers[next_layer_idx:] = new_layers
            self.log(f"  重建执行计划：剩余 {len(new_layers)} 层，"
                     f"共 {len(remaining_sqs)} 个子问题（含 {len(new_sub_questions)} 个追加）")
            self._log_execution_plan(execution_layers)
        except ValueError as e:
            self.log(f"⚠️ 重建执行计划失败（可能存在循环依赖）: {e}，"
                     f"新子问题追加到新的一层")
            execution_layers[next_layer_idx:] = execution_layers[next_layer_idx:]
            execution_layers.append(new_sub_questions)

    async def _selective_retry(
        self,
        state: dict[str, Any],
        failed_sub_question_ids: list[str],
    ) -> dict[str, Any]:
        """
        选择性重试：只重新执行评审未通过的子问题，保留已通过的结果

        Args:
            state: 工作流状态
            failed_sub_question_ids: 评审未通过的子问题 id 列表

        Returns:
            更新后的状态
        """
        self.log(f"选择性重试：{len(failed_sub_question_ids)} 个不合格子问题: {failed_sub_question_ids}")

        completed_results: dict[str, SubQuestionResult] = {}
        sub_question_results = state.get("sub_question_results", [])

        for result in sub_question_results:
            sq_id = result.sub_question_id if isinstance(result, SubQuestionResult) else result.get("sub_question_id", "")
            if sq_id not in failed_sub_question_ids:
                completed_results[sq_id] = result if isinstance(result, SubQuestionResult) else SubQuestionResult(**result)

        self.log(f"  保留 {len(completed_results)} 个已通过的子问题结果")

        structured_sub_questions = state.get("structured_sub_questions", [])
        sq_map = {sq.id: sq for sq in structured_sub_questions}

        failed_sqs: list[SubQuestion] = []
        for sq_id in failed_sub_question_ids:
            if sq_id in sq_map:
                failed_sqs.append(sq_map[sq_id])
            else:
                for result in sub_question_results:
                    r_id = result.sub_question_id if isinstance(result, SubQuestionResult) else result.get("sub_question_id", "")
                    r_question = result.sub_question if isinstance(result, SubQuestionResult) else result.get("sub_question", "")
                    if r_id == sq_id:
                        failed_sqs.append(SubQuestion(id=sq_id, question=r_question))
                        break

        per_critiques = state.get("per_sub_question_critiques", {})
        overall_coherence_feedback = state.get("overall_coherence_feedback", "")
        search_cache: dict[str, list] = {}

        if failed_sqs:
            try:
                retry_layers = self._topological_sort(
                    failed_sqs,
                    satisfied_dependency_ids=set(completed_results),
                )
            except ValueError:
                retry_layers = [failed_sqs]

            for layer in retry_layers:
                for sq in layer:
                    prior_findings = self._collect_prior_findings(sq, completed_results)
                    critique_feedback = self._format_sq_critique_feedback(
                        sq.id, per_critiques, overall_coherence_feedback
                    )
                    # 根据评审反馈生成补充搜索关键词
                    extra_keywords = await self._generate_extra_keywords(
                        sq, critique_feedback
                    )
                    self.log(f"  重试子问题 [{sq.id}]: {sq.question}")
                    if extra_keywords:
                        self.log(f"    补充关键词: 中文={extra_keywords.get('zh', [])}, "
                                 f"英文={extra_keywords.get('en', [])}")
                    result = await self._execute_single_sub_question(
                        sq, state, prior_findings, search_cache,
                        critique_feedback=critique_feedback,
                        extra_keywords=extra_keywords,
                    )
                    completed_results[sq.id] = result

        self._merge_results_to_state(state, completed_results)
        state["current_step"] = "critic"
        return state

    def _format_sq_critique_feedback(
        self,
        sq_id: str,
        per_critiques: dict,
        overall_coherence_feedback: str,
    ) -> str:
        """
        格式化指定子问题的审查反馈，供重试时参考

        Args:
            sq_id: 子问题 id
            per_critiques: 逐子问题评审结果
            overall_coherence_feedback: 整体一致性反馈

        Returns:
            格式化后的反馈文本
        """
        critique = per_critiques.get(sq_id, {})
        if not critique:
            return overall_coherence_feedback

        parts = []
        feedback = critique.get("feedback", "")
        if feedback:
            parts.append(f"审查意见: {feedback}")

        issues = critique.get("issues", [])
        if issues:
            issue_texts = [f"- {i.get('description', '')}" for i in issues]
            parts.append("存在的问题:\n" + "\n".join(issue_texts))

        suggestions = critique.get("revision_suggestions", [])
        if suggestions:
            parts.append("修正建议:\n" + "\n".join(f"- {s}" for s in suggestions))

        if overall_coherence_feedback:
            parts.append(f"整体一致性反馈: {overall_coherence_feedback}")

        return "\n\n".join(parts)

    async def _generate_extra_keywords(
        self,
        sub_question: SubQuestion,
        critique_feedback: str,
    ) -> dict[str, list[str]] | None:
        """
        根据评审反馈生成补充搜索关键词

        让 LLM 分析评审反馈中指出的不足，生成针对性的搜索关键词，
        使重试时能获取到不同的搜索结果，从而实质性改进研究结果。

        Args:
            sub_question: 子问题
            critique_feedback: 格式化后的审查反馈

        Returns:
            补充关键词字典 {"zh": [...], "en": [...]}，失败时返回 None
        """
        if not critique_feedback:
            return None

        from prompts.researcher_prompt import (
            EXTRA_KEYWORDS_SYSTEM_PROMPT,
            EXTRA_KEYWORDS_USER_PROMPT,
        )

        original_keywords_zh = "、".join(sub_question.keywords_zh)
        original_keywords_en = ", ".join(sub_question.keywords_en)

        try:
            user_prompt = EXTRA_KEYWORDS_USER_PROMPT.format(
                sub_question=sub_question.question,
                original_keywords_zh=original_keywords_zh,
                original_keywords_en=original_keywords_en,
                critique_feedback=critique_feedback,
            )
            response = await self.generate(
                prompt=user_prompt,
                system_prompt=EXTRA_KEYWORDS_SYSTEM_PROMPT,
                max_tokens=settings.orchestrator_max_tokens,
            )

            data = self._parse_json_response(response)
            if data:
                extra_zh = data.get("extra_keywords_zh", [])
                extra_en = data.get("extra_keywords_en", [])
                if extra_zh or extra_en:
                    return {"zh": extra_zh, "en": extra_en}
            return None
        except Exception as e:
            self.log(f"  ⚠️ 生成补充关键词失败: {e}")
            return None

    def _parse_json_response(self, response: str) -> dict | None:
        """解析 LLM 返回的 JSON 响应"""
        text = response.strip()

        if "```json" in text:
            start = text.find("```json") + len("```json")
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + len("```")
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        import json
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    def _merge_results_to_state(
        self,
        state: dict[str, Any],
        completed_results: dict[str, SubQuestionResult],
    ) -> None:
        """
        将所有子问题的结果合并到工作流状态

        合并策略：
        - search_results / paper_results: 合并所有子问题的搜索结果
        - findings: 按子问题维度组织，生成综合研究发现
        - sources: 合并所有参考文献来源（去重）

        Args:
            state: 工作流状态
            completed_results: 所有已完成的子问题结果
        """
        all_search_results = []
        all_paper_results = []
        all_sources = []
        all_evidence_cards = []
        findings_parts = []

        seen_urls = set()
        seen_paper_urls = set()
        seen_source_urls = set()

        for sq_id, result in completed_results.items():
            findings_parts.append(
                f"## 子问题: {result.sub_question}\n\n{result.findings}"
            )

            for sr in result.search_results:
                url = sr.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_search_results.append(sr)

            for pr in result.paper_results:
                url = pr.get("url", "")
                if url and url not in seen_paper_urls:
                    seen_paper_urls.add(url)
                    all_paper_results.append(pr)

            for src in result.sources:
                url = src.get("url", "")
                if url and url not in seen_source_urls:
                    seen_source_urls.add(url)
                    all_sources.append(src)

            all_evidence_cards.extend(result.evidence_cards)

        state["sub_question_results"] = list(completed_results.values())
        state["search_results"] = all_search_results
        state["paper_results"] = all_paper_results
        state["findings"] = "\n\n---\n\n".join(findings_parts)
        state["sources"] = all_sources
        state["evidence_cards"] = deduplicate_cards(all_evidence_cards)

        self.log(f"结果汇总：{len(completed_results)} 个子问题，"
                 f"{len(all_search_results)} 条网页，"
                 f"{len(all_paper_results)} 篇论文，"
                 f"{len(all_sources)} 条参考文献")

    async def _fallback_to_serial_researcher(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        回退到串行 Researcher 模式

        当子问题数 ≤ 1 或依赖解析失败时使用。

        Args:
            state: 工作流状态

        Returns:
            更新后的状态
        """
        self.log("回退到串行 Researcher 模式")

        if self.researcher is None:
            self.log("⚠️ 串行 Researcher 未注入，尝试创建")
            self.researcher = ResearcherAgent(
                llm=self.llm,
                context_manager=self.context_manager,
            )

        structured_sub_questions = state.get("structured_sub_questions", [])
        if len(structured_sub_questions) == 1:
            sq = structured_sub_questions[0]
            state["search_keywords"] = [
                {
                    "sub_question": sq.question,
                    "keywords_zh": sq.keywords_zh,
                    "keywords_en": sq.keywords_en,
                }
            ]
            state["sub_questions"] = [sq.question]

        state = await self.researcher.run(state)
        state["use_orchestrator"] = False
        return state

    def _log_execution_plan(self, layers: list[list[SubQuestion]]) -> None:
        """
        打印执行计划

        Args:
            layers: 按层级组织的子问题
        """
        self.log(f"执行计划：共 {len(layers)} 层")
        for i, layer in enumerate(layers):
            ids = [sq.id for sq in layer]
            deps_info = []
            for sq in layer:
                deps = f"依赖{sq.depends_on}" if sq.depends_on else "无依赖"
                deps_info.append(f"{sq.id}({deps})")
            self.log(f"  第 {i + 1} 层 [{', '.join(ids)}]: {', '.join(deps_info)}")

    def _parse_json_response(self, response: str) -> dict | None:
        text = response.strip()

        if "```json" in text:
            start = text.find("```json") + len("```json")
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + len("```")
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None
