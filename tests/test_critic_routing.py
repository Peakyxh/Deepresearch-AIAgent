"""Regression tests for cost-aware Critic routing."""

import asyncio
from typing import Any

from agents.critic_agent import CriticAgent
from prompts.critic_prompt import (
    CRITIC_USER_PROMPT,
    OVERALL_COHERENCE_CRITIC_USER_PROMPT,
    SUB_QUESTION_CRITIC_USER_PROMPT,
)
from tests.mocks import MockLLM
from workflows.state import SubQuestion, SubQuestionResult


class ScriptedCritic(CriticAgent):
    def __init__(self, overall: dict[str, Any]):
        super().__init__(llm=MockLLM())
        self.overall = overall

    async def _critique_single_sub_question(self, query: str, sq_result) -> dict:
        return {
            "scores": {
                "factual_accuracy": 28,
                "completeness": 24,
                "source_sufficiency": 24,
                "relevance": 20,
            },
            "total_score": 96,
            "passed": True,
            "feedback": "",
            "revision_suggestions": [],
        }

    async def _critique_overall_coherence(
        self, query: str, sub_question_results, search_results, paper_results
    ) -> dict:
        return {
            "scores": {},
            "score_details": {},
            "issues": [],
            "feedback": "改善报告结构",
            "revision_suggestions": [],
            "blocking_sub_question_ids": [],
            "writer_revision_suggestions": [],
            "missing_research_topics": [],
            "problematic_sub_question_ids": [],
            **self.overall,
        }


def make_state(ids: tuple[str, ...] = ("sq_1",)) -> dict[str, Any]:
    questions = [SubQuestion(id=sq_id, question=f"Question {sq_id}") for sq_id in ids]
    results = [
        SubQuestionResult(
            sub_question_id=sq_id,
            sub_question=f"Question {sq_id}",
            findings=f"Findings {sq_id}",
        )
        for sq_id in ids
    ]
    return {
        "query": "test",
        "sub_question_results": results,
        "structured_sub_questions": questions,
        "orchestrator_plan": {
            "layers": [{"layer": 0, "sub_question_ids": list(ids)}]
        },
        "search_results": [],
        "paper_results": [],
        "per_sub_question_critiques": {},
        "failed_sub_question_ids": [],
        "retry_count": 0,
        "max_retries": 1,
    }


def run_critique(overall: dict[str, Any], state: dict[str, Any] | None = None):
    critic = ScriptedCritic(overall)
    return asyncio.run(critic._run_two_level_critique(state or make_state()))


def test_high_score_problem_ids_go_to_writer_not_research() -> None:
    state = run_critique(
        {
            "total_score": 82,
            "passed": True,
            "problematic_sub_question_ids": ["sq_1"],
            "blocking_sub_question_ids": ["sq_1"],
            "writer_revision_suggestions": ["在报告中说明证据局限"],
        }
    )

    assert state["current_step"] == "writer"
    assert state["critique_outcome"] == "passed"
    assert state["failed_sub_question_ids"] == []
    assert state["writer_revision_suggestions"] == ["在报告中说明证据局限"]


def test_conditional_score_without_blocker_goes_to_writer() -> None:
    state = run_critique(
        {
            "total_score": 75,
            "passed": False,
            "problematic_sub_question_ids": ["sq_1"],
        }
    )

    assert state["current_step"] == "writer"
    assert state["critique_outcome"] == "conditional_pass"
    assert state["retry_count"] == 0


def test_conditional_score_with_blocker_retries_only_target() -> None:
    state = run_critique(
        {
            "total_score": 75,
            "passed": False,
            "problematic_sub_question_ids": ["sq_1"],
            "blocking_sub_question_ids": ["sq_1"],
        }
    )

    assert state["current_step"] == "orchestrator"
    assert state["critique_outcome"] == "research_retry"
    assert state["failed_sub_question_ids"] == ["sq_1"]
    assert state["retry_count"] == 1


def test_missing_execution_plan_result_is_always_blocking() -> None:
    state = make_state(("sq_1",))
    state["orchestrator_plan"]["layers"][0]["sub_question_ids"].append("sq_2")

    result = run_critique({"total_score": 85, "passed": True}, state)

    assert result["current_step"] == "orchestrator"
    assert result["failed_sub_question_ids"] == ["sq_2"]


def test_blocking_missing_topic_becomes_targeted_sub_question() -> None:
    state = run_critique(
        {
            "total_score": 75,
            "passed": False,
            "missing_research_topics": [
                {
                    "question": "制造业软件工程岗位受到什么影响？",
                    "reason": "原始范围完全缺失",
                    "keywords_zh": ["制造业 软件工程 AI"],
                    "keywords_en": ["manufacturing software jobs AI"],
                }
            ],
        }
    )

    assert state["current_step"] == "orchestrator"
    assert state["failed_sub_question_ids"] == ["sq_gap_1"]
    assert state["structured_sub_questions"][-1].id == "sq_gap_1"


def test_overall_prompt_uses_compact_evidence_summary() -> None:
    prompt = OVERALL_COHERENCE_CRITIC_USER_PROMPT.format(
        query="test query",
        all_findings="findings",
        evidence_summary="3 web sources and 2 papers",
    )

    assert "3 web sources and 2 papers" in prompt
    assert "{search_results}" not in prompt
    assert "{paper_results}" not in prompt


def test_fact_check_prompts_keep_detailed_evidence_inputs() -> None:
    overall_prompt = CRITIC_USER_PROMPT.format(
        query="test query",
        findings="findings",
        search_results="web evidence",
        paper_results="paper evidence",
    )
    sub_question_prompt = SUB_QUESTION_CRITIC_USER_PROMPT.format(
        query="test query",
        sub_question="sub question",
        findings="findings",
        search_results="web evidence",
        paper_results="paper evidence",
    )

    for prompt in (overall_prompt, sub_question_prompt):
        assert "web evidence" in prompt
        assert "paper evidence" in prompt
