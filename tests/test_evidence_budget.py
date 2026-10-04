"""Tests for compact evidence selection and reuse."""

from academic.base_academic import PaperResult
from search.base_search import SearchResult
from workflows.evidence import (
    build_evidence_cards,
    deduplicate_cards,
    format_evidence_catalog,
    format_claims,
    normalize_claims,
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


def test_explicit_year_range_excludes_out_of_scope_papers() -> None:
    papers = [
        PaperResult(
            title=f"Pupil segmentation {year}",
            url=f"https://arxiv.org/abs/{year}",
            abstract="pupil segmentation benchmark",
            year=year,
        )
        for year in (2019, 2023, 2026)
    ]

    selected = select_paper_evidence(
        papers, "2020-2025 pupil segmentation progress", limit=5
    )

    assert [paper.year for paper in selected] == [2023]


def test_paper_selection_rejects_generic_but_off_topic_match() -> None:
    papers = [
        PaperResult(
            title="Robust pupil segmentation for eye tracking",
            url="https://arxiv.org/abs/pupil",
            abstract="A pupil and gaze segmentation benchmark.",
            year=2024,
        ),
        PaperResult(
            title="Efficient brain tumor segmentation",
            url="https://arxiv.org/abs/tumor",
            abstract="A deep learning medical image segmentation benchmark.",
            year=2024,
        ),
    ]

    selected = select_paper_evidence(
        papers, "pupil segmentation eye tracking", limit=5
    )

    assert [paper.url for paper in selected] == ["https://arxiv.org/abs/pupil"]


def test_claims_keep_only_real_evidence_ids_and_final_citations() -> None:
    cards = build_evidence_cards(
        [
            SearchResult(
                title="Official report",
                url="https://example.gov/report",
                content="verified finding",
                score=0.9,
            )
        ],
        [],
        "sq_1",
        excerpt_chars=120,
    )
    valid_id = cards[0]["source_id"]

    claims = normalize_claims(
        [
            {
                "claim": "A supported claim",
                "source_ids": [valid_id, "src_invented"],
                "confidence": "high",
                "scope": "2025",
            },
            {
                "claim": "An unsupported claim",
                "source_ids": ["src_invented"],
                "confidence": "high",
            },
        ],
        cards,
    )

    assert claims[0]["source_ids"] == [valid_id]
    assert claims[1]["source_ids"] == []
    assert claims[1]["confidence"] == "low"
    rendered = format_claims(claims, cards)
    assert "[网页1]" in rendered
    assert "需进一步验证" in rendered
