"""
Clarifier Agent —— 用户意图澄清师

负责分析用户的研究问题，识别模糊点和缺失上下文，
通过与用户交互获取澄清信息，最终总结结构化的意图描述。
"""

import json
from typing import Any

from agents.base_agent import BaseAgent
from config import settings
from llm.base_llm import BaseLLM
from prompts.clarifier_prompt import (
    CLARIFIER_SYSTEM_PROMPT,
    CLARIFIER_USER_PROMPT,
    CLARIFIER_SUMMARY_SYSTEM_PROMPT,
    CLARIFIER_SUMMARY_USER_PROMPT,
)


class ClarifierAgent(BaseAgent):
    """
    Clarifier Agent

    工作流程：
    1. 接收用户的原始研究问题
    2. 调用 LLM 分析问题，识别模糊点和缺失上下文
    3. 生成澄清问题，通过 CLI 与用户交互获取回答
    4. 根据用户回答，调用 LLM 总结结构化意图描述
    5. 将意图描述和澄清问答对写入工作流状态
    """

    def __init__(self, llm: BaseLLM | None = None, interaction_handler: Any = None):
        super().__init__(name="Clarifier", llm=llm)
        self.max_rounds = settings.clarifier_max_rounds
        self.enabled = settings.clarifier_enabled
        self.interaction_handler = interaction_handler

    async def _execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        执行意图澄清

        Args:
            state: 工作流状态字典，必须包含 "query" 字段

        Returns:
            更新后的状态字典，包含 clarified_intent 和 clarification_qa
        """
        query = state.get("query", "")
        if not query:
            self.log("错误：研究问题为空")
            state["error"] = "研究问题不能为空"
            return state

        if not self.enabled:
            self.log("Clarifier 已禁用，跳过意图澄清")
            state["clarified_intent"] = ""
            state["clarification_qa"] = []
            state["current_step"] = "planner"
            return state

        self.log(f"开始分析用户意图：{query}")

        clarification_qa = list(state.get("clarification_qa", []))
        start_round = int(state.get("clarification_round", 0) or 0) + 1

        for round_num in range(start_round, self.max_rounds + 1):
            self.log(f"澄清交互第 {round_num}/{self.max_rounds} 轮")

            questions = state.get("pending_clarification_questions", [])
            if not questions:
                questions = await self._generate_clarification_questions(
                    query, clarification_qa
                )
                state["pending_clarification_questions"] = questions
                state["clarification_round"] = round_num - 1

            if not questions:
                self.log("问题已足够清晰，无需进一步澄清")
                break

            round_answers = await self._interact_with_user(questions, round_num)
            state["pending_clarification_questions"] = []
            state["clarification_round"] = round_num

            if round_answers is None:
                self.log("用户跳过所有澄清问题")
                break

            clarification_qa.extend(round_answers)

            if self._should_stop_clarifying(clarification_qa):
                break

        clarified_intent = ""
        if clarification_qa:
            self.log("正在总结结构化意图描述...")
            clarified_intent = await self._summarize_intent(query, clarification_qa)
            self.log(f"意图描述: {clarified_intent[:100]}...")
        else:
            self.log("未收集到澄清信息，将使用原始问题进行规划")

        state["clarified_intent"] = clarified_intent
        state["clarification_qa"] = clarification_qa
        state["pending_clarification_questions"] = []
        state["current_step"] = "planner"

        return state

    async def _generate_clarification_questions(
        self, query: str, existing_qa: list[dict]
    ) -> list[dict]:
        """
        调用 LLM 生成澄清问题

        Args:
            query: 用户原始问题
            existing_qa: 已有的澄清问答对

        Returns:
            澄清问题列表，每个元素包含 id、question、options、purpose
        """
        qa_context = ""
        if existing_qa:
            qa_lines = []
            for qa in existing_qa:
                qa_lines.append(f"Q: {qa['question']}\nA: {qa['answer']}")
            qa_context = "\n\n已有的澄清问答：\n" + "\n".join(qa_lines)

        user_prompt = CLARIFIER_USER_PROMPT.format(query=query) + qa_context

        response = await self.generate(
            prompt=user_prompt,
            system_prompt=CLARIFIER_SYSTEM_PROMPT,
            max_tokens=settings.clarifier_max_tokens,
        )

        parsed = self._parse_json_response(response)
        if not parsed:
            self.log("⚠️ 澄清问题解析失败，跳过本轮")
            return []

        questions = parsed.get("clarification_questions", [])
        analysis = parsed.get("analysis", "")
        if analysis:
            self.log(f"问题分析: {analysis[:80]}...")

        return questions

    async def _interact_with_user(
        self, questions: list[dict], round_num: int
    ) -> list[dict] | None:
        """
        通过 CLI 与用户交互，获取澄清回答

        Args:
            questions: 澄清问题列表
            round_num: 当前轮次

        Returns:
            问答对列表，如果用户跳过所有问题则返回 None
        """
        if self.interaction_handler:
            response = await self.interaction_handler.request(
                kind="clarification",
                payload={"round": round_num, "questions": questions},
            )
            answers = response.get("answers", []) if isinstance(response, dict) else []
            normalized = []
            for item in answers:
                if not isinstance(item, dict):
                    continue
                question = str(item.get("question", "")).strip()
                answer = str(item.get("answer", "")).strip()
                if question and answer:
                    normalized.append({"question": question, "answer": answer})
            return normalized or None

        return self._interact_with_user_cli(questions, round_num)

    def _interact_with_user_cli(
        self, questions: list[dict], round_num: int
    ) -> list[dict] | None:
        """CLI 兼容路径：保留原有终端交互。"""
        print(f"\n🤔 为了更好地理解您的研究需求，我有几个问题（第{round_num}轮）：")
        print("━" * 50)

        qa_pairs = []
        all_skipped = True

        for i, q in enumerate(questions, 1):
            question_text = q.get("question", "")
            options = q.get("options", [])
            purpose = q.get("purpose", "")

            if not question_text:
                continue

            print(f"\n  [{i}/{len(questions)}] {question_text}")
            if options:
                options_text = "、".join([f"「{opt}」" for opt in options])
                print(f"       参考: {options_text}")
            if purpose:
                print(f"       (目的: {purpose})")
            print("       输入 'skip' 跳过此问题，'skip all' 跳过所有问题")

            try:
                answer = input("  > ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return None

            if answer.lower() == "skip all":
                return None

            if answer.lower() == "skip" or not answer:
                self.log(f"用户跳过了问题: {question_text[:50]}")
                continue

            all_skipped = False
            qa_pairs.append({
                "question": question_text,
                "answer": answer,
            })
            self.log(f"用户回答: {answer[:50]}")

        print("━" * 50)

        if all_skipped:
            return None

        return qa_pairs

    async def _summarize_intent(
        self, query: str, clarification_qa: list[dict]
    ) -> str:
        """
        根据用户回答总结结构化意图描述

        Args:
            query: 用户原始问题
            clarification_qa: 澄清问答对

        Returns:
            结构化意图描述文本
        """
        qa_lines = []
        for qa in clarification_qa:
            qa_lines.append(f"Q: {qa['question']}\nA: {qa['answer']}")
        qa_section = "\n".join(qa_lines)

        user_prompt = CLARIFIER_SUMMARY_USER_PROMPT.format(
            query=query,
            qa_section=qa_section,
        )

        response = await self.generate(
            prompt=user_prompt,
            system_prompt=CLARIFIER_SUMMARY_SYSTEM_PROMPT,
            max_tokens=settings.clarifier_max_tokens,
        )

        return response.strip()

    def _should_stop_clarifying(self, clarification_qa: list[dict]) -> bool:
        """
        判断是否应该停止澄清

        当已经收集到足够的信息时，提前结束澄清。

        Args:
            clarification_qa: 当前已收集的问答对

        Returns:
            True 表示应该停止
        """
        if len(clarification_qa) >= 4:
            return True
        return False

    def _parse_json_response(self, response: str) -> dict | None:
        """
        解析 LLM 返回的 JSON 响应

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
