"""Offline tests for intent fidelity and Planner coverage metadata."""

import asyncio
import json

from agents.clarifier_agent import ClarifierAgent
from agents.planner_agent import PlannerAgent
from tests.mocks import MockLLM


def test_clarifier_profile_keeps_assumptions_separate() -> None:
    profile = ClarifierAgent._normalize_intent_profile(
        {
            "original_goal": "模型错误改写的目标",
            "explicit_requirements": ["关注成本"],
            "assumptions": ["默认研究中国市场"],
            "unresolved_items": ["时间范围未明确"],
        },
        "比较 A 和 B",
        [{"question": "关注什么？", "answer": "成本"}],
    )

    rendered = ClarifierAgent._render_intent_profile(profile)

    assert profile["original_goal"] == "比较 A 和 B"
    assert profile["explicit_requirements"] == ["关注什么？：成本"]
    assert profile["assumptions"] == ["默认研究中国市场"]
    assert "默认假设（未经用户确认）" in rendered
    assert "尚未明确" in rendered


def test_planner_records_valid_coverage_and_self_check() -> None:
    planner = PlannerAgent(llm=MockLLM())
    state: dict = {}
    planner._parse_plan_data(
        {
            "sub_questions": [
                {
                    "id": "sq_1",
                    "question": "A 和 B 的成本分别是多少？",
                    "depends_on": [],
                    "priority": 0,
                    "keywords_zh": ["A 成本", "B 成本"],
                    "keywords_en": ["A cost", "B cost"],
                }
            ],
            "research_plan": "比较成本后给出结论",
            "coverage": [
                {
                    "requirement_id": "original_query",
                    "user_requirement": "比较 A 和 B 的成本",
                    "covered_by": ["sq_1"],
                    "explanation": "sq_1 直接比较成本",
                }
            ],
            "assumptions": ["默认比较当前公开价格"],
            "self_check": {
                "fully_answers_original_query": True,
                "uncovered_requirements": [],
            },
        },
        state,
        "比较 A 和 B 的成本",
    )

    assert state["planner_self_check"]["fully_answers_original_query"] is True
    assert state["plan_coverage"][0]["covered_by"] == ["sq_1"]
    assert state["plan_assumptions"] == ["默认比较当前公开价格"]


def test_planner_rejects_unknown_coverage_references() -> None:
    planner = PlannerAgent(llm=MockLLM())
    state: dict = {}
    planner._parse_plan_data(
        {
            "sub_questions": [
                {
                    "id": "sq_1",
                    "question": "研究 A",
                    "depends_on": [],
                    "priority": 0,
                    "keywords_zh": [],
                    "keywords_en": [],
                }
            ],
            "research_plan": "研究 A",
            "coverage": [
                {
                    "requirement_id": "original_query",
                    "user_requirement": "研究 A",
                    "covered_by": ["sq_missing"],
                    "explanation": "错误引用",
                }
            ],
            "self_check": {
                "fully_answers_original_query": True,
                "uncovered_requirements": [],
            },
        },
        state,
        "研究 A",
    )

    self_check = state["planner_self_check"]
    assert self_check["fully_answers_original_query"] is False
    assert self_check["invalid_sub_question_references"] == ["sq_missing"]
    assert self_check["uncovered_requirements"] == ["original_query：研究 A"]


def test_planner_prompt_preserves_source_priority() -> None:
    response = json.dumps(
        {
            "sub_questions": [
                {
                    "id": "sq_1",
                    "question": "比较 A 和 B 的成本",
                    "depends_on": [],
                    "priority": 0,
                    "keywords_zh": ["A 成本", "B 成本"],
                    "keywords_en": ["A cost", "B cost"],
                }
            ],
            "research_plan": "比较成本",
            "coverage": [
                {
                    "requirement_id": "original_query",
                    "user_requirement": "比较 A 和 B",
                    "covered_by": ["sq_1"],
                    "explanation": "直接覆盖",
                }
            ],
            "assumptions": [],
            "self_check": {
                "fully_answers_original_query": True,
                "uncovered_requirements": [],
            },
        },
        ensure_ascii=False,
    )
    llm = MockLLM(default_response=response)
    planner = PlannerAgent(llm=llm)
    state = {
        "query": "比较 A 和 B",
        "intent_profile": {
            "original_goal": "比较 A 和 B",
            "explicit_requirements": ["关注成本"],
            "assumptions": [],
            "unresolved_items": [],
        },
        "clarification_qa": [
            {"question": "重点是什么？", "answer": "成本"}
        ],
    }

    asyncio.run(planner.run(state))
    prompt = llm.call_log[0]["prompt"]

    assert "研究问题：比较 A 和 B" in prompt
    assert "用户澄清原话" in prompt
    assert "用户意图档案" in prompt


def test_planner_revises_once_when_coverage_is_missing() -> None:
    missing_coverage = {
        "sub_questions": [
            {
                "id": "sq_1",
                "question": "研究 A",
                "depends_on": [],
                "priority": 0,
                "keywords_zh": ["A"],
                "keywords_en": ["A"],
            }
        ],
        "research_plan": "初始计划",
        "coverage": [],
        "assumptions": [],
        "self_check": {
            "fully_answers_original_query": True,
            "uncovered_requirements": [],
        },
    }
    revised = {
        **missing_coverage,
        "research_plan": "修订计划",
        "coverage": [
            {
                "requirement_id": "original_query",
                "user_requirement": "研究 A",
                "covered_by": ["sq_1"],
                "explanation": "直接覆盖",
            }
        ],
    }

    class SequentialLLM(MockLLM):
        def __init__(self) -> None:
            super().__init__()
            self.responses_in_order = [missing_coverage, revised]

        async def generate(self, prompt, system_prompt="", max_tokens=None):
            self.call_log.append({"prompt": prompt, "system_prompt": system_prompt})
            return json.dumps(self.responses_in_order.pop(0), ensure_ascii=False)

    llm = SequentialLLM()
    state = {"query": "研究 A"}

    asyncio.run(PlannerAgent(llm=llm).run(state))

    assert len(llm.call_log) == 2
    assert state["research_plan"] == "修订计划"
    assert state["planner_self_check"]["fully_answers_original_query"] is True

