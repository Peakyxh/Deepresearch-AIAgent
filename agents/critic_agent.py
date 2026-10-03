"""
Critic Agent —— 研究审查员

负责对研究结果进行两级评审：
1. 逐子问题评审：对每个子问题的研究发现单独评分，精确定位不合格的 Sub-Agent
2. 整体一致性评审：评估所有子问题之间的逻辑一致性、完整性和衔接性

评审未通过时，将不合格的子问题 id 列表传给 Orchestrator 进行选择性重试。
重试后只对重试的子问题重新评审，已通过的不再重复评审。
"""

import json
from typing import Any

from agents.base_agent import BaseAgent
from llm.base_llm import BaseLLM
from memory.context_manager import ContextManager
from config import settings
from workflows.state import SubQuestion, SubQuestionResult
from workflows.evidence import (
    deduplicate_cards,
    format_evidence_catalog,
    summarize_evidence_coverage,
)
from prompts.critic_prompt import (
    CRITIC_SYSTEM_PROMPT,
    CRITIC_USER_PROMPT,
    SUB_QUESTION_CRITIC_SYSTEM_PROMPT,
    SUB_QUESTION_CRITIC_USER_PROMPT,
    OVERALL_COHERENCE_CRITIC_SYSTEM_PROMPT,
    OVERALL_COHERENCE_CRITIC_USER_PROMPT,
)


class CriticAgent(BaseAgent):
    """
    Critic Agent

    工作流程：
    1. 读取 Orchestrator 输出的子问题研究结果
    2. 逐子问题评审：对每个子问题单独评分
    3. 整体一致性评审：评估子问题间的一致性
    4. 识别不合格的子问题，供 Orchestrator 选择性重试
    5. 重试时只对重试的子问题重新评审（增量评审）
    """

    def __init__(self, llm: BaseLLM | None = None, context_manager: ContextManager | None = None):
        super().__init__(name="Critic", llm=llm, context_manager=context_manager)

    async def _execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        执行研究审查

        Args:
            state: 工作流状态字典

        Returns:
            更新后的状态字典
        """
        findings = state.get("findings", "")
        sub_question_results = state.get("sub_question_results", [])

        if not findings:
            self.log("⚠️ 研究发现为空，跳过审查")
            state["critique_passed"] = False
            state["critique_feedback"] = "研究发现为空，无法审查"
            state["critique_scores"] = {}
            state["critique_total_score"] = 0
            state["current_step"] = "orchestrator"
            return state

        if sub_question_results:
            return await self._run_two_level_critique(state)
        else:
            return await self._run_overall_critique(state)

    async def _run_two_level_critique(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        两级评审：逐子问题评审 + 整体一致性评审

        支持增量评审：重试时只对重试的子问题重新评审
        """
        query = state.get("query", "")
        sub_question_results = state.get("sub_question_results", [])
        search_results = state.get("search_results", [])
        paper_results = state.get("paper_results", [])

        existing_critiques = state.get("per_sub_question_critiques", {})
        is_incremental = bool(existing_critiques)

        # 上轮失败的子问题（包括逐子问题评审不合格和整体一致性标记的），必须重新评审
        previous_failed_ids = set(state.get("failed_sub_question_ids", []))

        per_critiques = dict(existing_critiques)

        sqs_to_critique = []
        for sq_result in sub_question_results:
            sq_id = self._get_sq_id(sq_result)
            # 上轮失败列表中的子问题必须重新评审（无论逐子问题评审是否通过）
            if sq_id in previous_failed_ids:
                sqs_to_critique.append(sq_result)
                continue
            existing = existing_critiques.get(sq_id)
            if existing and existing.get("passed", False):
                continue
            sqs_to_critique.append(sq_result)

        if is_incremental:
            self.log(f"增量评审：{len(sqs_to_critique)} 个子问题需要重新评审 "
                     f"（共 {len(sub_question_results)} 个，{len(existing_critiques) - len(sqs_to_critique)} 个已通过跳过）")
        else:
            self.log(f"完整评审：{len(sqs_to_critique)} 个子问题")

        for sq_result in sqs_to_critique:
            sq_id = self._get_sq_id(sq_result)
            self.log(f"  评审子问题 [{sq_id}]...")
            critique = await self._critique_single_sub_question(query, sq_result)
            per_critiques[sq_id] = critique

            total = critique.get("total_score", 0)
            passed = critique.get("passed", False)
            status = "✅ 通过" if passed else "❌ 不合格"
            self.log(f"  子问题 [{sq_id}] {status}（总分 {total}/100）")
            self._log_sq_score_details(sq_id, critique)

        failed_ids = []
        for sq_id, critique in per_critiques.items():
            if not critique.get("passed", False):
                failed_ids.append(sq_id)

        # 计划覆盖是确定性约束，不交给 LLM 猜测。只有实际进入执行计划、
        # 却没有产出结果的子问题才视为阻断性缺失。
        missing_planned_ids = self._find_missing_planned_ids(
            state, sub_question_results
        )
        for sq_id in missing_planned_ids:
            if sq_id not in failed_ids:
                failed_ids.append(sq_id)
                self.log(f"  计划覆盖缺失 [{sq_id}]，加入补充研究列表")

        if failed_ids:
            self.log(f"不合格子问题: {failed_ids}")
        else:
            self.log("所有子问题评审通过 ✅")

        self.log("进行整体一致性评审...")
        overall_critique = await self._critique_overall_coherence(
            query, sub_question_results, search_results, paper_results
        )

        overall_score = overall_critique.get("total_score", 0)
        direct_threshold = settings.overall_coherence_pass_threshold
        conditional_threshold = settings.overall_coherence_conditional_threshold

        known_ids = {
            self._get_sq_id(item) for item in sub_question_results
            if self._get_sq_id(item)
        }
        known_ids.update(
            sq.id if isinstance(sq, SubQuestion) else sq.get("id", "")
            for sq in state.get("structured_sub_questions", [])
        )
        problematic_ids = self._valid_target_ids(
            overall_critique.get("problematic_sub_question_ids", []), known_ids
        )
        blocking_ids = self._valid_target_ids(
            overall_critique.get("blocking_sub_question_ids", []), known_ids
        )

        is_direct_pass = overall_score >= direct_threshold
        is_conditional_band = overall_score >= conditional_threshold
        added_topic_ids: list[str] = []
        if not is_direct_pass:
            added_topic_ids = self._append_missing_research_topics(
                state, overall_critique.get("missing_research_topics", [])
            )

        if is_direct_pass:
            # 高分时，整体评审列出的普通问题只进入 Writer 修订清单。
            overall_accepted = True
            critique_outcome = "passed"
            overall_status = "✅ 直接通过"
        elif is_conditional_band:
            # 中间分只允许明确标记为 blocking 的问题触发补研。
            for sq_id in [*blocking_ids, *added_topic_ids]:
                if sq_id not in failed_ids:
                    failed_ids.append(sq_id)
            overall_accepted = not blocking_ids and not added_topic_ids
            critique_outcome = (
                "conditional_pass" if overall_accepted else "research_retry"
            )
            overall_status = (
                "⚠️ 有条件通过" if overall_accepted else "❌ 存在阻断问题"
            )
        else:
            # 低分时优先使用阻断目标；兼容旧模型输出时回退到 problematic IDs。
            retry_targets = blocking_ids or problematic_ids
            for sq_id in [*retry_targets, *added_topic_ids]:
                if sq_id not in failed_ids:
                    failed_ids.append(sq_id)
            if retry_targets or added_topic_ids:
                overall_accepted = False
                critique_outcome = "research_retry"
                overall_status = "❌ 需要补充研究"
            else:
                # 没有可执行目标时禁止退回并重跑整个研究，交给 Writer 明示局限。
                overall_accepted = True
                critique_outcome = "conditional_pass"
                overall_status = "⚠️ 有条件通过（无可执行补研目标）"

        self.log(
            f"整体一致性: {overall_status}（总分 {overall_score}/100；"
            f"直接通过阈值 {direct_threshold}，有条件通过阈值 {conditional_threshold}）"
        )
        self._log_overall_score_details(overall_critique)

        if problematic_ids:
            self.log(
                f"  诊断涉及子问题 {problematic_ids}；"
                "非阻断问题仅交给 Writer，不自动补研"
            )
        for sq_id in blocking_ids:
            if not is_direct_pass:
                self.log(f"  阻断性问题涉及子问题 [{sq_id}]，加入补充研究列表")

        all_sq_passed = len(failed_ids) == 0
        critique_passed = all_sq_passed and overall_accepted

        avg_sq_score = 0
        if per_critiques:
            avg_sq_score = sum(c.get("total_score", 0) for c in per_critiques.values()) / len(per_critiques)
        aggregate_total_score = int(avg_sq_score * 0.6 + overall_score * 0.4)

        feedback_parts = []
        if failed_ids:
            feedback_parts.append(f"不合格子问题: {', '.join(failed_ids)}")
        for sq_id in failed_ids:
            sq_feedback = per_critiques.get(sq_id, {}).get("feedback", "")
            if sq_feedback:
                feedback_parts.append(f"[{sq_id}] {sq_feedback}")
        overall_feedback = overall_critique.get("feedback", "")
        if overall_feedback:
            feedback_parts.append(f"整体一致性: {overall_feedback}")
        combined_feedback = "\n".join(feedback_parts)

        all_suggestions = []
        for sq_id in failed_ids:
            sq_suggestions = per_critiques.get(sq_id, {}).get("revision_suggestions", [])
            for s in sq_suggestions:
                all_suggestions.append(f"[{sq_id}] {s}")
        overall_suggestions = overall_critique.get("revision_suggestions", [])
        all_suggestions.extend(overall_suggestions)
        writer_suggestions = list(
            overall_critique.get("writer_revision_suggestions", [])
        )
        for suggestion in overall_suggestions:
            if suggestion not in writer_suggestions:
                writer_suggestions.append(suggestion)
        if writer_suggestions:
            feedback_parts.append(
                "写作阶段修订建议:\n"
                + "\n".join(f"- {item}" for item in writer_suggestions)
            )
            combined_feedback = "\n".join(feedback_parts)

        state["per_sub_question_critiques"] = per_critiques
        state["failed_sub_question_ids"] = failed_ids
        state["overall_coherence_score"] = overall_score
        state["overall_coherence_feedback"] = overall_critique.get("feedback", "")
        state["overall_coherence_passed"] = overall_accepted
        state["critique_outcome"] = critique_outcome
        state["writer_revision_suggestions"] = writer_suggestions

        state["critique_scores"] = overall_critique.get("scores", {})
        state["critique_score_details"] = overall_critique.get("score_details", {})
        state["critique_total_score"] = aggregate_total_score
        state["critique_feedback"] = combined_feedback
        state["revision_suggestions"] = all_suggestions

        if critique_passed:
            state["critique_passed"] = True
            self.log(f"审查结果: ✅ 全部通过（综合分 {aggregate_total_score}/100）")
        else:
            state["critique_passed"] = False
            fail_reasons = []
            if not all_sq_passed:
                fail_reasons.append(f"{len(failed_ids)} 个阻断性问题")
            if not overall_accepted:
                fail_reasons.append("需要补充研究")
            self.log(f"审查结果: ❌ 未通过（{'，'.join(fail_reasons)}）")

        if state["critique_passed"]:
            state["current_step"] = "writer"
        else:
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", 2)
            if retry_count < max_retries:
                state["retry_count"] = retry_count + 1
                state["current_step"] = "orchestrator"
                self.log(f"回退到 Orchestrator 进行选择性重试（第 {retry_count + 1} 次重试）")
                if state.get("revision_suggestions"):
                    self.log("修正建议:")
                    for i, s in enumerate(state["revision_suggestions"], 1):
                        self.log(f"  {i}. {s}")
            else:
                state["current_step"] = "writer"
                state["critique_outcome"] = "forced_writer"
                self.log(f"已达到最大重试次数 ({max_retries})，强制进入 Writer 阶段")

        return state

    def _find_missing_planned_ids(
        self,
        state: dict[str, Any],
        sub_question_results: list,
    ) -> list[str]:
        """Find required execution-plan items that produced no result."""
        planned_ids: list[str] = []
        for layer in state.get("orchestrator_plan", {}).get("layers", []):
            planned_ids.extend(layer.get("sub_question_ids", []))
        result_ids = {
            self._get_sq_id(result) for result in sub_question_results
            if self._get_sq_id(result)
        }
        return [sq_id for sq_id in planned_ids if sq_id not in result_ids]

    @staticmethod
    def _valid_target_ids(candidate_ids: list, known_ids: set[str]) -> list[str]:
        """Keep stable, known IDs and remove hallucinated/duplicate targets."""
        result: list[str] = []
        for candidate in candidate_ids:
            sq_id = str(candidate).strip()
            if sq_id and sq_id in known_ids and sq_id not in result:
                result.append(sq_id)
        return result

    def _append_missing_research_topics(
        self,
        state: dict[str, Any],
        topics: list,
    ) -> list[str]:
        """Turn blocking coverage gaps into explicit targeted sub-questions."""
        structured = state.setdefault("structured_sub_questions", [])
        existing_ids = {
            sq.id if isinstance(sq, SubQuestion) else sq.get("id", "")
            for sq in structured
        }
        existing_questions = {
            (sq.question if isinstance(sq, SubQuestion) else sq.get("question", ""))
            .strip()
            .lower()
            for sq in structured
        }
        state_topics = state.setdefault("missing_research_topics", [])
        added_ids: list[str] = []

        for topic in topics:
            if not isinstance(topic, dict):
                continue
            question = str(topic.get("question", "")).strip()
            if not question or question.lower() in existing_questions:
                continue
            index = 1
            while f"sq_gap_{index}" in existing_ids:
                index += 1
            sq_id = f"sq_gap_{index}"
            structured.append(
                SubQuestion(
                    id=sq_id,
                    question=question,
                    priority=0,
                    keywords_zh=topic.get("keywords_zh", []),
                    keywords_en=topic.get("keywords_en", []),
                )
            )
            normalized_topic = {**topic, "id": sq_id}
            state_topics.append(normalized_topic)
            existing_ids.add(sq_id)
            existing_questions.add(question.lower())
            added_ids.append(sq_id)
            self.log(f"  新增阻断性缺失主题 [{sq_id}]: {question}")

        return added_ids

    async def _run_overall_critique(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        整体评审（向后兼容，无子问题结果时使用）
        """
        query = state.get("query", "")
        findings = state.get("findings", "")
        search_results = state.get("search_results", [])
        paper_results = state.get("paper_results", [])
        evidence_cards = state.get("evidence_cards", [])

        self.log("开始多维度量化评分审查...")

        if evidence_cards:
            search_text, paper_text = format_evidence_catalog(evidence_cards)
        else:
            search_text = self._format_search_results_for_verification(search_results)
            paper_text = self._format_paper_results_for_verification(paper_results)

        if self.context_manager:
            combined_text = search_text + "\n\n" + paper_text
            combined_tokens = self.context_manager.estimate_text_tokens(combined_text)

            if combined_tokens > self.context_manager.max_tokens:
                self.log(f"验证文本过长 ({combined_tokens} tokens)，进行语义压缩...")

                search_tokens = self.context_manager.estimate_text_tokens(search_text)
                if search_tokens > self.context_manager.get_compress_threshold("search"):
                    self.log(f"  压缩网页验证文本 ({search_tokens} tokens)...")
                    search_text = await self.context_manager.compress_search_results(search_text)
                    self.log(f"  网页验证文本压缩完成: {search_tokens} → "
                             f"{self.context_manager.estimate_text_tokens(search_text)} tokens")

                paper_tokens = self.context_manager.estimate_text_tokens(paper_text)
                if paper_tokens > self.context_manager.get_compress_threshold("paper"):
                    self.log(f"  压缩论文验证文本 ({paper_tokens} tokens)...")
                    paper_text = await self.context_manager.compress_paper_results(
                        paper_text,
                    )
                    self.log(f"  论文验证文本压缩完成: {paper_tokens} → "
                             f"{self.context_manager.estimate_text_tokens(paper_text)} tokens")

        user_prompt = self.context_prompt_prefix() + CRITIC_USER_PROMPT.format(
            query=query,
            findings=findings,
            search_results=search_text,
            paper_results=paper_text,
        )

        self.log("正在调用 LLM 进行多维度评分审查...")
        response = await self.generate(
            prompt=user_prompt,
            system_prompt=CRITIC_SYSTEM_PROMPT,
            max_tokens=settings.critic_max_tokens,
        )

        critique_data = self._parse_json_response(response)

        if critique_data:
            scores = critique_data.get("scores", {})
            score_details = critique_data.get("score_details", {})
            total_score = sum(scores.values())

            state["critique_scores"] = scores
            state["critique_score_details"] = score_details
            state["critique_total_score"] = total_score
            state["critique_feedback"] = critique_data.get("overall_feedback", "")
            state["revision_suggestions"] = critique_data.get("revision_suggestions", [])

            pass_threshold = settings.critique_pass_threshold
            conditional_threshold = settings.critique_conditional_threshold

            if total_score >= pass_threshold:
                state["critique_passed"] = True
                self.log(f"审查结果: ✅ 通过（总分 {total_score}/100，阈值 {pass_threshold}）")
            elif total_score >= conditional_threshold:
                state["critique_passed"] = True
                self.log(f"审查结果: ⚠️ 有条件通过（总分 {total_score}/100，阈值 {pass_threshold}）")
                state["critique_feedback"] = (
                    f"[有条件通过，总分{total_score}分] "
                    + state["critique_feedback"]
                )
            else:
                state["critique_passed"] = False
                self.log(f"审查结果: ❌ 未通过（总分 {total_score}/100，阈值 {pass_threshold}）")

            self._log_score_details(scores, score_details, total_score)

            hallucination_count = len(critique_data.get("hallucination_issues", []))
            logic_count = len(critique_data.get("logic_issues", []))
            completeness_count = len(critique_data.get("completeness_issues", []))
            source_count = len(critique_data.get("source_issues", []))
            timeliness_count = len(critique_data.get("timeliness_issues", []))

            self.log(f"  问题汇总: 幻觉={hallucination_count} 逻辑={logic_count} "
                     f"完整性={completeness_count} 来源={source_count} 时效={timeliness_count}")
        else:
            self.log("⚠️ 审查结果 JSON 解析失败，有条件通过")
            state["critique_passed"] = True
            state["critique_scores"] = {}
            state["critique_score_details"] = {}
            state["critique_total_score"] = 0
            state["critique_feedback"] = "审查结果解析失败，有条件通过"
            state["revision_suggestions"] = []

        if state["critique_passed"]:
            state["current_step"] = "writer"
        else:
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", 2)
            if retry_count < max_retries:
                state["retry_count"] = retry_count + 1
                state["current_step"] = "orchestrator"
                self.log(f"回退到 Orchestrator（第 {retry_count + 1} 次重试）")
                if state.get("revision_suggestions"):
                    self.log("修正建议:")
                    for i, s in enumerate(state["revision_suggestions"], 1):
                        self.log(f"  {i}. {s}")
            else:
                state["current_step"] = "writer"
                self.log(f"已达到最大重试次数 ({max_retries})，强制进入 Writer 阶段")

        return state

    async def _critique_single_sub_question(
        self,
        query: str,
        sq_result,
    ) -> dict:
        """
        对单个子问题的研究结果进行评审

        Args:
            query: 原始研究问题
            sq_result: 子问题研究结果（SubQuestionResult 或 dict）

        Returns:
            评审结果字典
        """
        sq_question = self._get_sq_field(sq_result, "sub_question", "")
        sq_findings = self._get_sq_field(sq_result, "findings", "")
        sq_search_results = self._get_sq_field(sq_result, "search_results", [])
        sq_paper_results = self._get_sq_field(sq_result, "paper_results", [])

        evidence_cards = self._get_sq_field(sq_result, "evidence_cards", [])
        if evidence_cards:
            search_text, paper_text = format_evidence_catalog(evidence_cards)
        else:
            search_text = self._format_search_results_for_verification(sq_search_results)
            paper_text = self._format_paper_results_for_verification(sq_paper_results)

        if self.context_manager:
            combined_text = search_text + "\n\n" + paper_text
            combined_tokens = self.context_manager.estimate_text_tokens(combined_text)

            if combined_tokens > self.context_manager.max_tokens:
                search_tokens = self.context_manager.estimate_text_tokens(search_text)
                if search_tokens > self.context_manager.get_compress_threshold("search"):
                    search_text = await self.context_manager.compress_search_results(search_text)

                paper_tokens = self.context_manager.estimate_text_tokens(paper_text)
                if paper_tokens > self.context_manager.get_compress_threshold("paper"):
                    paper_text = await self.context_manager.compress_paper_results(
                        paper_text,
                    )

        user_prompt = self.context_prompt_prefix() + SUB_QUESTION_CRITIC_USER_PROMPT.format(
            query=query,
            sub_question=sq_question,
            findings=sq_findings,
            search_results=search_text,
            paper_results=paper_text,
        )

        response = await self.generate(
            prompt=user_prompt,
            system_prompt=SUB_QUESTION_CRITIC_SYSTEM_PROMPT,
            max_tokens=settings.critic_max_tokens,
        )

        critique_data = self._parse_json_response(response)

        if critique_data:
            scores = critique_data.get("scores", {})
            total_score = sum(scores.values())
            pass_threshold = settings.sub_question_critique_pass_threshold

            return {
                "scores": scores,
                "score_details": critique_data.get("score_details", {}),
                "total_score": total_score,
                "passed": total_score >= pass_threshold,
                "issues": critique_data.get("issues", []),
                "feedback": critique_data.get("feedback", ""),
                "revision_suggestions": critique_data.get("revision_suggestions", []),
            }
        else:
            return {
                "scores": {},
                "score_details": {},
                "total_score": 0,
                "passed": False,
                "issues": [],
                "feedback": "审查结果解析失败",
                "revision_suggestions": [],
            }

    async def _critique_overall_coherence(
        self,
        query: str,
        sub_question_results,
        search_results: list[dict],
        paper_results: list[dict],
    ) -> dict:
        """
        对所有子问题的研究结果进行整体一致性评审

        Args:
            query: 原始研究问题
            sub_question_results: 所有子问题研究结果
            search_results: 合并的网页搜索结果
            paper_results: 合并的论文搜索结果

        Returns:
            整体一致性评审结果字典
        """
        all_findings_parts = []
        for sq_result in sub_question_results:
            sq_id = self._get_sq_id(sq_result)
            sq_question = self._get_sq_field(sq_result, "sub_question", "")
            sq_findings = self._get_sq_field(sq_result, "findings", "")
            key_insights = self._get_sq_field(sq_result, "key_insights", [])
            insight_text = "\n".join(f"- {item}" for item in key_insights)
            finding_excerpt = sq_findings[
                : settings.critic_findings_chars_per_sub_question
            ]
            if len(sq_findings) > len(finding_excerpt):
                finding_excerpt += "\n（其余细节已在逐子问题评审中核验）"
            all_findings_parts.append(
                f"## 子问题 [{sq_id}]: {sq_question}\n\n"
                f"关键洞察:\n{insight_text or '无结构化洞察'}\n\n"
                f"研究发现摘要:\n{finding_excerpt}"
            )

        all_findings = "\n\n---\n\n".join(all_findings_parts)

        evidence_cards = []
        for result in sub_question_results:
            evidence_cards.extend(self._get_sq_field(result, "evidence_cards", []))
        evidence_summary = summarize_evidence_coverage(
            deduplicate_cards(evidence_cards)
        )
        if not evidence_cards:
            evidence_summary = (
                f"网页证据 {len(search_results)} 条，论文证据 {len(paper_results)} 条。"
                "详细事实核验已在逐子问题评审完成。"
            )

        user_prompt = self.context_prompt_prefix() + OVERALL_COHERENCE_CRITIC_USER_PROMPT.format(
            query=query,
            all_findings=all_findings,
            evidence_summary=evidence_summary,
        )

        response = await self.generate(
            prompt=user_prompt,
            system_prompt=OVERALL_COHERENCE_CRITIC_SYSTEM_PROMPT,
            max_tokens=settings.critic_max_tokens,
        )

        critique_data = self._parse_json_response(response)

        if critique_data:
            scores = critique_data.get("scores", {})
            total_score = sum(scores.values())
            pass_threshold = settings.overall_coherence_pass_threshold

            return {
                "scores": scores,
                "score_details": critique_data.get("score_details", {}),
                "total_score": total_score,
                "passed": total_score >= pass_threshold,
                "issues": critique_data.get("issues", []),
                "feedback": critique_data.get("feedback", ""),
                "revision_suggestions": critique_data.get("revision_suggestions", []),
                "blocking_sub_question_ids": critique_data.get(
                    "blocking_sub_question_ids", []
                ),
                "writer_revision_suggestions": critique_data.get(
                    "writer_revision_suggestions", []
                ),
                "missing_research_topics": critique_data.get(
                    "missing_research_topics", []
                ),
                "problematic_sub_question_ids": critique_data.get("problematic_sub_question_ids", []),
            }
        else:
            return {
                "scores": {},
                "score_details": {},
                "total_score": 0,
                "passed": False,
                "issues": [],
                "feedback": "整体一致性审查结果解析失败",
                "revision_suggestions": [],
                "blocking_sub_question_ids": [],
                "writer_revision_suggestions": [],
                "missing_research_topics": [],
                "problematic_sub_question_ids": [],
            }

    def _get_sq_id(self, sq_result) -> str:
        if isinstance(sq_result, SubQuestionResult):
            return sq_result.sub_question_id
        return sq_result.get("sub_question_id", "")

    def _get_sq_field(self, sq_result, field: str, default=None):
        if isinstance(sq_result, SubQuestionResult):
            return getattr(sq_result, field, default)
        return sq_result.get(field, default)

    def _log_sq_score_details(self, sq_id: str, critique: dict) -> None:
        scores = critique.get("scores", {})
        score_details = critique.get("score_details", {})
        dimension_names = {
            "factual_accuracy": "事实准确性",
            "completeness": "完整性",
            "source_sufficiency": "来源充分性",
            "relevance": "相关性",
        }
        dimension_max = {
            "factual_accuracy": 30,
            "completeness": 25,
            "source_sufficiency": 25,
            "relevance": 20,
        }

        for key, name in dimension_names.items():
            score = scores.get(key, 0)
            max_score = dimension_max.get(key, 0)
            detail = score_details.get(key, "")
            self.log(f"    {name}: {score}/{max_score} - {detail[:50]}...")

    def _log_overall_score_details(self, critique: dict) -> None:
        scores = critique.get("scores", {})
        score_details = critique.get("score_details", {})
        total_score = critique.get("total_score", 0)

        dimension_names = {
            "logic_consistency": "逻辑一致性",
            "completeness": "完整性",
            "coherence": "逻辑衔接",
            "source_diversity": "来源多样性",
        }
        dimension_max = {
            "logic_consistency": 30,
            "completeness": 25,
            "coherence": 25,
            "source_diversity": 20,
        }

        self.log(f"  整体一致性评分详情（总分 {total_score}/100）:")
        for key, name in dimension_names.items():
            score = scores.get(key, 0)
            max_score = dimension_max.get(key, 0)
            detail = score_details.get(key, "")
            self.log(f"    {name}: {score}/{max_score} - {detail[:50]}...")

    def _log_score_details(
        self,
        scores: dict[str, int],
        score_details: dict[str, str],
        total_score: int,
    ) -> None:
        dimension_names = {
            "factual_accuracy": "事实准确性",
            "logic_consistency": "逻辑一致性",
            "completeness": "信息完整性",
            "source_sufficiency": "来源充分性",
            "timeliness": "时效性",
        }
        dimension_max = {
            "factual_accuracy": 30,
            "logic_consistency": 20,
            "completeness": 20,
            "source_sufficiency": 15,
            "timeliness": 15,
        }

        self.log(f"  评分详情（总分 {total_score}/100）:")
        for key, name in dimension_names.items():
            score = scores.get(key, 0)
            max_score = dimension_max.get(key, 0)
            detail = score_details.get(key, "")
            self.log(f"    {name}: {score}/{max_score} - {detail[:60]}...")

    def _format_search_results_for_verification(
        self, search_results: list[dict]
    ) -> str:
        if not search_results:
            return "（无搜索结果可供验证）"

        formatted = []
        for i, r in enumerate(search_results, 1):
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            source = r.get("source", "")
            text = f"[{i}] {title}\n    来源: {source} | URL: {url}\n    内容: {content}"
            formatted.append(text)

        return "\n\n".join(formatted)

    def _format_paper_results_for_verification(
        self, paper_results: list[dict]
    ) -> str:
        if not paper_results:
            return "（无论文结果可供验证）"

        formatted = []
        for i, p in enumerate(paper_results, 1):
            title = p.get("title", "")
            url = p.get("url", "")
            authors = ", ".join(p.get("authors", [])[:3])
            year = p.get("year", "未知")
            abstract = p.get("abstract", "")
            citation_count = p.get("citation_count")

            text = f"[论文{i}] {title}\n"
            text += f"    作者: {authors}\n"
            text += f"    年份: {year} | URL: {url}\n"
            if citation_count is not None:
                text += f"    引用: {citation_count} 次\n"
            text += f"    摘要: {abstract[:300]}{'...' if len(abstract) > 300 else ''}"
            formatted.append(text)

        return "\n\n".join(formatted)

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
