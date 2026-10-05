"""
Planner Agent —— 研究规划师

负责将复杂的研究问题拆解为可执行的子任务，
提取搜索关键词，制定系统的研究计划。
支持参考历史研究记录，避免重复搜索。
支持输出带依赖关系的结构化子问题。
"""

import json
from typing import Any

from agents.base_agent import BaseAgent
from llm.base_llm import BaseLLM
from prompts.planner_prompt import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    PLANNER_HISTORY_SECTION,
    PLANNER_CLARIFIED_INTENT_SECTION,
    PLANNER_CLARIFICATION_QA_SECTION,
)
from workflows.state import SubQuestion
from config import settings


class PlannerAgent(BaseAgent):
    """
    Planner Agent

    工作流程：
    1. 接收用户的原始研究问题
    2. 读取历史研究记录作为参考上下文（不覆盖研究计划）
    3. 调用 LLM 拆解问题为带依赖关系的子问题
    4. 提取每个子问题的搜索关键词
    5. 制定研究计划
    6. 将结果写入工作流状态（同时输出新格式和兼容旧格式）
    """

    def __init__(self, llm: BaseLLM | None = None):
        super().__init__(name="Planner", llm=llm)

    async def _execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        执行研究规划

        Args:
            state: 工作流状态字典，必须包含 "query" 字段
                   可选包含 "history_context" 字段（历史研究参考）

        Returns:
            更新后的状态字典，包含 sub_questions、structured_sub_questions、
            search_keywords、research_plan
        """
        query = state.get("query", "")
        if not query:
            self.log("错误：研究问题为空")
            state["error"] = "研究问题不能为空"
            return state

        self.log(f"开始规划研究：{query}")

        clarified_intent_section = ""
        intent_profile = state.get("intent_profile")
        clarified_intent = state.get("clarified_intent", "")
        if isinstance(intent_profile, dict) and intent_profile:
            clarified_intent_section = PLANNER_CLARIFIED_INTENT_SECTION.format(
                clarified_intent=json.dumps(intent_profile, ensure_ascii=False, indent=2)
            )
            self.log("参考了结构化意图档案")
        elif clarified_intent:
            clarified_intent_section = PLANNER_CLARIFIED_INTENT_SECTION.format(
                clarified_intent=clarified_intent
            )
            self.log(f"参考了意图澄清信息（{len(clarified_intent)} 字符）")

        clarification_qa_section = ""
        clarification_qa = state.get("clarification_qa", [])
        if clarification_qa:
            qa_lines = []
            for qa in clarification_qa:
                question = str(qa.get("question", "")).strip()
                answer = str(qa.get("answer", "")).strip()
                if question and answer:
                    qa_lines.append(f"Q: {question}\nA: {answer}")
            if qa_lines:
                clarification_qa_section = PLANNER_CLARIFICATION_QA_SECTION.format(
                    clarification_qa="\n\n".join(qa_lines)
                )

        history_section = ""
        history_context = state.get("history_context", "")
        if history_context:
            history_section = PLANNER_HISTORY_SECTION.format(history_text=history_context)
            self.log(f"参考了历史研究记录（{len(history_context)} 字符）")

        user_prompt = PLANNER_USER_PROMPT.format(
            query=query,
            clarified_intent_section=(
                clarification_qa_section + clarified_intent_section
            ),
            history_section=history_section,
        )

        self.log("正在调用 LLM 生成研究计划...")
        response = await self.generate(
            prompt=user_prompt,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            max_tokens=settings.planner_max_tokens,
        )

        self.log(f"LLM 原始回复长度: {len(response)} 字符")

        plan_data = self._parse_json_response(response)

        if plan_data:
            self._parse_plan_data(plan_data, state, query)
            self_check = state.get("planner_self_check", {})
            if not self_check.get("fully_answers_original_query", False):
                retry_prompt = self._build_self_correction_prompt(
                    user_prompt, self_check
                )
                self.log("规划覆盖自检未通过，正在进行一次内部修订...")
                revised_response = await self.generate(
                    prompt=retry_prompt,
                    system_prompt=PLANNER_SYSTEM_PROMPT,
                    max_tokens=settings.planner_max_tokens,
                )
                revised_plan = self._parse_json_response(revised_response)
                if revised_plan:
                    self._parse_plan_data(revised_plan, state, query)
                else:
                    self.log("⚠️ 修订后的研究计划解析失败，保留首次规划结果")
        else:
            self.log("⚠️ JSON 解析失败，使用回退策略")
            self._fallback_plan(state, query)

        state["current_step"] = "orchestrator"
        return state

    @staticmethod
    def _build_self_correction_prompt(
        original_prompt: str, self_check: dict[str, Any]
    ) -> str:
        """要求同一个 Planner 针对代码侧发现的覆盖问题修订一次。"""
        issues = json.dumps(self_check, ensure_ascii=False, indent=2)
        return (
            f"{original_prompt}\n\n"
            "你上一次生成的计划未通过覆盖一致性检查：\n"
            f"{issues}\n\n"
            "请修订完整研究计划，确保 original_query 和每个 "
            "explicit_requirement_N 都有有效 coverage，且 covered_by 只引用实际存在的子问题 id。"
            "请重新输出完整 JSON，不要解释。"
        )

    def _parse_plan_data(
        self, plan_data: dict, state: dict[str, Any], query: str
    ) -> None:
        """
        解析 LLM 返回的研究计划数据

        支持两种格式：
        1. 新格式：sub_questions 是对象列表，包含 id、depends_on 等
        2. 旧格式：sub_questions 是字符串列表（向后兼容）

        Args:
            plan_data: 解析后的 JSON 字典
            state: 工作流状态字典
            query: 原始研究问题
        """
        raw_sub_questions = plan_data.get("sub_questions", [])

        if not raw_sub_questions:
            self.log("⚠️ 子问题列表为空，使用回退策略")
            self._fallback_plan(state, query)
            return

        first_item = raw_sub_questions[0]

        if isinstance(first_item, dict) and "id" in first_item:
            self._parse_structured_sub_questions(raw_sub_questions, state)
        else:
            self._parse_legacy_sub_questions(raw_sub_questions, state, query)

        research_plan = plan_data.get("research_plan", "")
        state["research_plan"] = research_plan
        self._parse_planning_metadata(plan_data, state, query)
        self.log(f"研究计划: {research_plan[:100]}...")

    def _parse_planning_metadata(
        self, plan_data: dict[str, Any], state: dict[str, Any], query: str
    ) -> None:
        """归一化覆盖映射、规划假设和 Planner 自检结果。"""
        structured = state.get("structured_sub_questions", [])
        valid_ids = {
            sq.id if hasattr(sq, "id") else str(sq.get("id", ""))
            for sq in structured
        }
        valid_ids.discard("")

        intent_profile = state.get("intent_profile", {})
        explicit_requirements = (
            intent_profile.get("explicit_requirements", [])
            if isinstance(intent_profile, dict)
            else []
        )
        required_coverage = {"original_query": query}
        for index, requirement in enumerate(explicit_requirements, 1):
            requirement = str(requirement).strip()
            if requirement:
                required_coverage[f"explicit_requirement_{index}"] = requirement

        coverage = []
        invalid_references = []
        covered_requirement_ids = set()
        for item in plan_data.get("coverage", []):
            if not isinstance(item, dict):
                continue
            requirement = str(item.get("user_requirement", "")).strip()
            requirement_id = str(item.get("requirement_id", "")).strip()
            covered_by = item.get("covered_by", [])
            if not isinstance(covered_by, list):
                covered_by = []
            normalized_ids = []
            for sq_id in covered_by:
                sq_id = str(sq_id).strip()
                if sq_id in valid_ids and sq_id not in normalized_ids:
                    normalized_ids.append(sq_id)
                elif sq_id:
                    invalid_references.append(sq_id)
            if requirement:
                coverage.append(
                    {
                        "requirement_id": requirement_id,
                        "user_requirement": requirement,
                        "covered_by": normalized_ids,
                        "explanation": str(item.get("explanation", "")).strip(),
                    }
                )
                if requirement_id in required_coverage and normalized_ids:
                    covered_requirement_ids.add(requirement_id)

        assumptions = plan_data.get("assumptions", [])
        if not isinstance(assumptions, list):
            assumptions = []
        assumptions = [str(item).strip() for item in assumptions if str(item).strip()]

        raw_self_check = plan_data.get("self_check", {})
        if not isinstance(raw_self_check, dict):
            raw_self_check = {}
        uncovered = raw_self_check.get("uncovered_requirements", [])
        if not isinstance(uncovered, list):
            uncovered = []
        uncovered = [str(item).strip() for item in uncovered if str(item).strip()]

        empty_extra_coverage = [
            item["user_requirement"]
            for item in coverage
            if not item["covered_by"]
            and item["requirement_id"] not in required_coverage
        ]
        for requirement in empty_extra_coverage:
            if requirement not in uncovered:
                uncovered.append(requirement)
        for requirement_id, requirement in required_coverage.items():
            if requirement_id not in covered_requirement_ids:
                missing = f"{requirement_id}：{requirement}"
                if missing not in uncovered:
                    uncovered.append(missing)

        fully_answers = bool(raw_self_check.get("fully_answers_original_query", False))
        fully_answers = fully_answers and not uncovered and not invalid_references
        self_check = {
            "fully_answers_original_query": fully_answers,
            "uncovered_requirements": uncovered,
            "invalid_sub_question_references": sorted(set(invalid_references)),
        }

        state["plan_coverage"] = coverage
        state["plan_assumptions"] = assumptions
        state["planner_self_check"] = self_check

        if fully_answers:
            self.log(f"规划自检通过：{len(coverage)} 项需求均有覆盖")
        else:
            details = uncovered or invalid_references
            self.log(f"⚠️ 规划自检未完全通过：{details}")

    def _parse_structured_sub_questions(
        self, raw_sub_questions: list[dict], state: dict[str, Any]
    ) -> None:
        """
        解析新格式的子问题（带依赖关系）

        Args:
            raw_sub_questions: 子问题字典列表
            state: 工作流状态字典
        """
        structured = []
        legacy_questions = []
        legacy_keywords = []

        for i, sq_data in enumerate(raw_sub_questions):
            sq_id = sq_data.get("id", f"sq_{i + 1}")
            question = sq_data.get("question", "")
            depends_on = sq_data.get("depends_on", [])
            priority = sq_data.get("priority", i)
            keywords_zh = sq_data.get("keywords_zh", [])
            keywords_en = sq_data.get("keywords_en", [])

            if not question:
                continue

            sub_q = SubQuestion(
                id=sq_id,
                question=question,
                depends_on=depends_on,
                priority=priority,
                keywords_zh=keywords_zh,
                keywords_en=keywords_en,
            )
            structured.append(sub_q)
            legacy_questions.append(question)
            legacy_keywords.append({
                "sub_question": question,
                "keywords_zh": keywords_zh,
                "keywords_en": keywords_en,
            })

            deps_str = f" (依赖: {', '.join(depends_on)})" if depends_on else " (无依赖)"
            self.log(f"  子问题 {sq_id}: {question}{deps_str}")

        state["structured_sub_questions"] = structured
        state["sub_questions"] = legacy_questions
        state["search_keywords"] = legacy_keywords
        state["use_orchestrator"] = True

        self.log(f"拆解出 {len(structured)} 个结构化子问题（含依赖关系）")

    def _parse_legacy_sub_questions(
        self,
        raw_sub_questions: list[str],
        state: dict[str, Any],
        query: str,
    ) -> None:
        """
        解析旧格式的子问题（纯字符串列表），自动补充 id 和依赖关系

        Args:
            raw_sub_questions: 子问题字符串列表
            state: 工作流状态字典
            query: 原始研究问题
        """
        structured = []
        legacy_keywords = []

        search_keywords = state.get("search_keywords", [])

        for i, sq_text in enumerate(raw_sub_questions):
            sq_id = f"sq_{i + 1}"
            kw_group = search_keywords[i] if i < len(search_keywords) else {}

            sub_q = SubQuestion(
                id=sq_id,
                question=sq_text,
                depends_on=[],
                priority=i,
                keywords_zh=kw_group.get("keywords_zh", [sq_text]),
                keywords_en=kw_group.get("keywords_en", [sq_text]),
            )
            structured.append(sub_q)
            legacy_keywords.append({
                "sub_question": sq_text,
                "keywords_zh": sub_q.keywords_zh,
                "keywords_en": sub_q.keywords_en,
            })

            self.log(f"  子问题 {sq_id}: {sq_text} (无依赖，旧格式)")

        state["structured_sub_questions"] = structured
        state["sub_questions"] = raw_sub_questions
        state["search_keywords"] = legacy_keywords
        state["use_orchestrator"] = len(structured) > 1

        self.log(f"拆解出 {len(structured)} 个子问题（旧格式，无依赖关系）")

    def _fallback_plan(self, state: dict[str, Any], query: str) -> None:
        """
        JSON 解析失败时的回退策略

        将原始问题作为唯一子问题，使用串行模式。

        Args:
            state: 工作流状态字典
            query: 原始研究问题
        """
        sub_q = SubQuestion(
            id="sq_1",
            question=query,
            depends_on=[],
            priority=0,
            keywords_zh=[query],
            keywords_en=[query],
        )

        state["structured_sub_questions"] = [sub_q]
        state["sub_questions"] = [query]
        state["search_keywords"] = [
            {
                "sub_question": query,
                "keywords_zh": [query],
                "keywords_en": [query],
            }
        ]
        state["research_plan"] = f"直接研究原始问题：{query}"
        state["plan_coverage"] = [
            {
                "user_requirement": query,
                "requirement_id": "original_query",
                "covered_by": ["sq_1"],
                "explanation": "回退计划直接研究用户原始问题",
            }
        ]
        state["plan_assumptions"] = []
        state["planner_self_check"] = {
            "fully_answers_original_query": True,
            "uncovered_requirements": [],
            "invalid_sub_question_references": [],
        }
        state["use_orchestrator"] = False

    def _parse_json_response(self, response: str) -> dict | None:
        """
        解析 LLM 返回的 JSON 响应

        LLM 可能返回带有 markdown 代码块标记的 JSON，
        需要先清理再解析。

        Args:
            response: LLM 的原始回复文本

        Returns:
            解析后的字典，如果解析失败则返回 None
        """
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
        except json.JSONDecodeError as e:
            self.log(f"JSON 解析错误: {e}")
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None
