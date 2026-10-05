import asyncio
import json

from academic.base_academic import BaseAcademic, PaperResult
from agents.researcher_sub_agent import ResearcherSubAgent
from config import settings
from search.base_search import BaseSearch, SearchResult
from workflows.state import SubQuestion


class ScriptedLLM:
    def __init__(self, actions: list[dict]):
        self.actions = list(actions)
        self.prompts: list[str] = []

    async def generate(self, prompt: str, system_prompt: str = "", max_tokens=None) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.actions.pop(0), ensure_ascii=False)


class RecordingWebSearch(BaseSearch):
    def __init__(self):
        super().__init__()
        self.queries: list[str] = []

    async def search(self, query: str, max_results=None) -> list[SearchResult]:
        self.queries.append(query)
        return [
            SearchResult(
                title=f"Web result for {query}",
                url=f"https://example.com/web/{len(self.queries)}",
                content="A sufficiently detailed web result used as research evidence.",
                source="fake",
                score=0.9,
            )
        ]


class RecordingAcademicSearch(BaseAcademic):
    def __init__(self):
        super().__init__()
        self.queries: list[str] = []

    async def search_papers(self, query: str, max_results=None) -> list[PaperResult]:
        self.queries.append(query)
        return [
            PaperResult(
                title=f"Paper for {query}",
                authors=["Researcher"],
                abstract="A detailed abstract supporting the selected research direction.",
                url=f"https://example.com/paper/{len(self.queries)}",
                year=2026,
                source="fake",
            )
        ]

    async def get_paper_detail(self, paper_id: str) -> PaperResult | None:
        return None

    async def get_paper_abstract(self, paper_id: str) -> str:
        return ""


def make_sub_question() -> SubQuestion:
    return SubQuestion(
        id="sq_1",
        question="What evidence supports adaptive research agents?",
        keywords_zh=["自适应研究代理"],
        keywords_en=["adaptive research agents"],
    )


def test_llm_selects_tools_and_finishes(monkeypatch) -> None:
    monkeypatch.setattr(settings, "researcher_tool_max_rounds", 5)
    monkeypatch.setattr(settings, "researcher_tool_min_calls", 2)
    monkeypatch.setattr(settings, "researcher_tool_max_web_calls", 3)
    monkeypatch.setattr(settings, "researcher_tool_max_paper_calls", 2)

    llm = ScriptedLLM(
        [
            {"action": "web_search", "query": "adaptive agents evidence", "reason": "baseline"},
            {"action": "academic_search", "query": "adaptive agents evaluation", "reason": "verify"},
            {"action": "finish", "query": "", "reason": "enough evidence"},
        ]
    )
    web = RecordingWebSearch()
    academic = RecordingAcademicSearch()
    agent = ResearcherSubAgent(llm=llm, search_engine=web, academic_search=academic)

    web_results, paper_results = asyncio.run(
        agent._collect_evidence_autonomously(
            sub_question=make_sub_question(),
            query="How should deep research agents work?",
            research_plan="Collect and cross-check evidence.",
            prior_findings=None,
            search_cache={},
            critique_feedback=None,
            extra_keywords=None,
            prior_round_findings=None,
            cached_context=None,
        )
    )

    assert web.queries == ["adaptive agents evidence"]
    assert academic.queries == ["adaptive agents evaluation"]
    assert len(web_results) == 1
    assert len(paper_results) == 1
    assert len(llm.prompts) == 3
    assert "adaptive agents evidence" in llm.prompts[1]


def test_premature_finish_and_repeated_query_use_safe_fallback(monkeypatch) -> None:
    monkeypatch.setattr(settings, "researcher_tool_max_rounds", 3)
    monkeypatch.setattr(settings, "researcher_tool_min_calls", 2)
    monkeypatch.setattr(settings, "researcher_tool_max_web_calls", 2)
    monkeypatch.setattr(settings, "researcher_tool_max_paper_calls", 1)

    llm = ScriptedLLM(
        [
            {"action": "finish", "query": "", "reason": "too early"},
            {"action": "web_search", "query": "自适应研究代理", "reason": "repeat"},
            {"action": "finish", "query": "", "reason": "done"},
        ]
    )
    web = RecordingWebSearch()
    academic = RecordingAcademicSearch()
    agent = ResearcherSubAgent(llm=llm, search_engine=web, academic_search=academic)

    web_results, paper_results = asyncio.run(
        agent._collect_evidence_autonomously(
            sub_question=make_sub_question(),
            query="Deep research",
            research_plan="Research safely.",
            prior_findings=None,
            search_cache={},
            critique_feedback=None,
            extra_keywords=None,
            prior_round_findings=None,
            cached_context=None,
        )
    )

    # The first premature finish falls back to the first seed. The repeated
    # decision then falls back to the next unused web seed.
    assert web.queries == ["自适应研究代理", "adaptive research agents"]
    assert len(web_results) == 2
    assert paper_results == []


def test_invalid_action_respects_round_and_tool_budgets(monkeypatch) -> None:
    monkeypatch.setattr(settings, "researcher_tool_max_rounds", 2)
    monkeypatch.setattr(settings, "researcher_tool_min_calls", 1)
    monkeypatch.setattr(settings, "researcher_tool_max_web_calls", 1)
    monkeypatch.setattr(settings, "researcher_tool_max_paper_calls", 1)

    llm = ScriptedLLM(
        [
            {"action": "unknown", "query": "ignored", "reason": "invalid"},
            {"action": "unknown", "query": "ignored again", "reason": "invalid"},
        ]
    )
    web = RecordingWebSearch()
    academic = RecordingAcademicSearch()
    agent = ResearcherSubAgent(llm=llm, search_engine=web, academic_search=academic)

    asyncio.run(
        agent._collect_evidence_autonomously(
            sub_question=make_sub_question(),
            query="Deep research",
            research_plan="Research safely.",
            prior_findings=None,
            search_cache={},
            critique_feedback=None,
            extra_keywords=None,
            prior_round_findings=None,
            cached_context=None,
        )
    )

    assert len(llm.prompts) == 2
    assert web.queries == ["自适应研究代理"]
    assert academic.queries == ["自适应研究代理"]
