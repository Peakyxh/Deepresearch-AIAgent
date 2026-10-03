"""Regression tests for Critic -> Orchestrator selective retry."""

import asyncio

from agents.orchestrator_agent import OrchestratorAgent
from tests.mocks import MockLLM
from workflows.state import SubQuestion


def test_extra_keyword_prompt_keeps_literal_json_braces() -> None:
    llm = MockLLM(
        default_response=(
            '{"extra_keywords_zh":["官方就业统计"],'
            '"extra_keywords_en":["official employment statistics"]}'
        )
    )
    orchestrator = OrchestratorAgent(llm=llm)
    question = SubQuestion(
        id="sq_2",
        question="生成式 AI 如何影响软件工程岗位？",
        keywords_zh=["软件工程岗位"],
        keywords_en=["software engineering jobs"],
    )

    result = asyncio.run(
        orchestrator._generate_extra_keywords(question, "补充官方就业数据")
    )

    assert result == {
        "zh": ["官方就业统计"],
        "en": ["official employment statistics"],
    }


def test_retry_sort_treats_completed_dependencies_as_satisfied() -> None:
    orchestrator = OrchestratorAgent(llm=MockLLM())
    questions = [
        SubQuestion(id="sq_2", question="岗位数量变化"),
        SubQuestion(
            id="sq_3",
            question="岗位结构变化",
            depends_on=["sq_1", "sq_2"],
        ),
    ]

    layers = orchestrator._topological_sort(
        questions,
        satisfied_dependency_ids={"sq_1"},
    )

    assert [[item.id for item in layer] for layer in layers] == [
        ["sq_2"],
        ["sq_3"],
    ]
