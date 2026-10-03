"""Tests for compact evidence selection and reuse."""

from academic.base_academic import PaperResult
from search.base_search import SearchResult
from workflows.evidence import (
    build_evidence_cards,
    deduplicate_cards,
    format_evidence_catalog,
    select_paper_evidence,
    select_web_evidence,
    summarize_evidence_coverage,
)


def test_web_budget_enforces_limit_and_domain_diversity() -> None:
    results = [
        SearchResult(
            title=f"AI software jobs report {index}",
            url=f"https://example.com/{index}",
            content="AI software engineering employment evidence " * 20,
            score=0.9 - index * 0.01,
        )
        for index in range(5)
    ]
    results.extend(
        [
            SearchResult(
                title="Official employment data",
                url="https://bls.gov/data",
                content="software engineering employment official statistics " * 20,
                score=0.8,
            ),
            SearchResult(
                title="University study",
                url="https://stanford.edu/study",
                content="AI software jobs academic research " * 20,
                score=0.8,
            ),
        ]
    )

    selected = select_web_evidence(
        results,
        "AI software engineering employment",
        limit=4,
        per_domain_limit=1,
    )

    assert len(selected) <= 4
    assert sum("example.com" in item.url for item in selected) == 1
    assert any("bls.gov" in item.url for item in selected)


def test_evidence_cards_are_compact_deduplicated_and_citable() -> None:
    web = [
        SearchResult(
            title="Official report",
            url="https://example.gov/report",
            content="evidence " * 500,
            score=0.9,
        )
    ]
    papers = [
        PaperResult(
            title="Research paper",
            url="https://arxiv.org/abs/1",
            abstract="abstract " * 500,
            year=2025,
            citation_count=10,
        )
    ]
    selected_papers = select_paper_evidence(papers, "research", limit=1)
    first = build_evidence_cards(web, selected_papers, "sq_1", excerpt_chars=120)
    second = build_evidence_cards(web, [], "sq_2", excerpt_chars=120)
    cards = deduplicate_cards([*first, *second])

    assert len(cards) == 2
    assert len(cards[0]["excerpt"]) <= 121
    assert cards[0]["sub_question_ids"] == ["sq_1", "sq_2"]

    web_text, paper_text = format_evidence_catalog(cards)
    assert "[网页1]" in web_text
    assert "[论文1]" in paper_text
    assert "各子问题证据数" in summarize_evidence_coverage(cards)
