"""Deterministic evidence selection and compact source cards.

Raw search payloads are expensive to send through every workflow stage. This
module selects a diverse evidence set once and turns it into reusable cards.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse

from academic.base_academic import PaperResult
from search.base_search import SearchResult


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _terms(text: str) -> set[str]:
    latin = re.findall(r"[a-z0-9]{2,}", text.lower())
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    chinese_bigrams = [
        chinese_chars[index] + chinese_chars[index + 1]
        for index in range(len(chinese_chars) - 1)
    ]
    return set(latin + chinese_bigrams)


_GENERIC_RESEARCH_TERMS = {
    "analysis", "approach", "application", "applications", "based",
    "benchmark", "comparison", "data", "dataset", "deep", "detection",
    "domain", "evaluation", "framework", "learning", "machine", "method",
    "methods", "model", "models", "performance", "progress", "recent",
    "research", "review", "segmentation", "study", "system", "systems",
    "technology", "using",
}


def _topic_anchor_terms(text: str) -> set[str]:
    return {
        term for term in _terms(text)
        if term not in _GENERIC_RESEARCH_TERMS and not term.isdigit()
    }


def _authority_score(domain: str) -> float:
    weak_hosts = (
        "blog.csdn.net",
        "medium.com",
        "researchgate.net",
        "academia.edu",
        "catalyzex.com",
        "alphaxiv.org",
    )
    academic_hosts = (
        "doi.org",
        "arxiv.org",
        "ieee.org",
        "acm.org",
        "springer.com",
        "nature.com",
        "sciencedirect.com",
        "wiley.com",
        "ncbi.nlm.nih.gov",
        "semanticscholar.org",
    )
    if any(domain == host or domain.endswith(f".{host}") for host in weak_hosts):
        return 0.15
    if domain.endswith((".gov", ".gov.cn", ".edu", ".edu.cn")):
        return 1.0
    if any(domain == host or domain.endswith(f".{host}") for host in academic_hosts):
        return 0.85
    if domain.endswith(".org"):
        return 0.6
    return 0.35


def _quality_label(authority: float) -> str:
    if authority >= 0.85:
        return "high"
    if authority >= 0.3:
        return "medium"
    return "low"


def _explicit_year_range(text: str) -> tuple[int, int] | None:
    """Extract an explicit research range such as 2020-2025.

    A single mentioned year is not treated as a hard boundary.  This keeps the
    selector conservative while preventing clearly out-of-scope papers from
    entering reports that explicitly request a bounded period.
    """
    years = [int(value) for value in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text)]
    if len(years) < 2:
        return None
    lower, upper = min(years), max(years)
    return (lower, upper) if lower < upper else None


def select_web_evidence(
    results: list[SearchResult],
    query_text: str,
    limit: int,
    per_domain_limit: int,
) -> list[SearchResult]:
    """Rank web results and enforce a domain-diversity budget."""
    query_terms = _terms(query_text)

    def rank(item: SearchResult) -> float:
        item_terms = _terms(f"{item.title} {item.content[:1000]}")
        overlap = len(query_terms & item_terms) / max(1, len(query_terms))
        return float(item.score or 0) * 0.45 + overlap * 0.3 + _authority_score(_domain(item.url)) * 0.25

    selected: list[SearchResult] = []
    domains: Counter[str] = Counter()
    weak_source_count = 0
    for result in sorted(results, key=rank, reverse=True):
        domain = _domain(result.url) or result.source or "unknown"
        if domains[domain] >= per_domain_limit:
            continue
        if _quality_label(_authority_score(domain)) == "low":
            if weak_source_count >= 1:
                continue
            weak_source_count += 1
        selected.append(result)
        domains[domain] += 1
        if len(selected) >= limit:
            break
    return selected


def select_paper_evidence(
    papers: list[PaperResult], query_text: str, limit: int
) -> list[PaperResult]:
    """Rank papers by relevance, citations, and recency without another LLM call."""
    query_terms = _terms(query_text)
    year_range = _explicit_year_range(query_text)

    if year_range:
        lower, upper = year_range
        papers = [
            item for item in papers
            if item.year is None or lower <= item.year <= upper
        ]

    # Require at least one topic-specific anchor when the query provides one.
    # This removes superficially similar papers (for example, an unrelated
    # medical segmentation paper) without spending another model call.
    anchors = _topic_anchor_terms(query_text)
    if anchors:
        papers = [
            item for item in papers
            if anchors & _terms(f"{item.title} {item.abstract[:1000]}")
        ]

    def rank(item: PaperResult) -> float:
        item_terms = _terms(f"{item.title} {item.abstract[:1000]}")
        overlap = len(query_terms & item_terms) / max(1, len(query_terms))
        citations = math.log1p(max(0, item.citation_count or 0)) / 10
        recency = max(0, min(1, ((item.year or 2000) - 2018) / 8))
        return overlap * 0.6 + citations * 0.25 + recency * 0.15

    return sorted(papers, key=rank, reverse=True)[:limit]


def build_evidence_cards(
    web_results: list[SearchResult],
    paper_results: list[PaperResult],
    sub_question_id: str,
    excerpt_chars: int,
) -> list[dict[str, Any]]:
    """Build stable, compact evidence cards for downstream agents."""
    cards: list[dict[str, Any]] = []
    for result in web_results:
        authority = _authority_score(_domain(result.url))
        cards.append(
            {
                "source_id": _source_id(result.url, result.title),
                "sub_question_id": sub_question_id,
                "type": "web",
                "title": result.title,
                "url": result.url,
                "domain": _domain(result.url),
                "authority": authority,
                "quality": _quality_label(authority),
                "excerpt": _compact(result.content, excerpt_chars),
            }
        )
    for paper in paper_results:
        authority = 0.85
        cards.append(
            {
                "source_id": _source_id(paper.url, paper.title),
                "sub_question_id": sub_question_id,
                "type": "paper",
                "title": paper.title,
                "url": paper.url,
                "domain": _domain(paper.url),
                "authority": authority,
                "quality": _quality_label(authority),
                "excerpt": _compact(paper.abstract, excerpt_chars),
                "authors": paper.authors[:3],
                "year": paper.year,
                "citation_count": paper.citation_count,
            }
        )
    return cards


def deduplicate_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate cards while retaining every related sub-question id."""
    merged: dict[str, dict[str, Any]] = {}
    for card in cards:
        key = str(card.get("source_id") or card.get("url") or card.get("title"))
        if key not in merged:
            merged[key] = dict(card)
            merged[key]["sub_question_ids"] = [card.get("sub_question_id", "")]
            continue
        sq_id = card.get("sub_question_id", "")
        if sq_id and sq_id not in merged[key]["sub_question_ids"]:
            merged[key]["sub_question_ids"].append(sq_id)
    return list(merged.values())


def format_evidence_catalog(cards: list[dict[str, Any]]) -> tuple[str, str]:
    """Format global web/paper citation catalogs with short excerpts."""
    web_lines: list[str] = []
    paper_lines: list[str] = []
    for index, card in enumerate((c for c in cards if c.get("type") == "web"), 1):
        web_lines.append(
            f"[网页{index}] {card.get('title', '')} "
            f"(证据ID: {card.get('source_id', '')}; 质量: {card.get('quality', 'unknown')})\n"
            f"URL: {card.get('url', '')}\n"
            f"证据摘录: {card.get('excerpt', '')}"
        )
    for index, card in enumerate((c for c in cards if c.get("type") == "paper"), 1):
        authors = card.get("authors", [])
        if isinstance(authors, list):
            authors = ", ".join(authors)
        paper_lines.append(
            f"[论文{index}] {card.get('title', '')} ({card.get('year') or '未知'}) "
            f"(证据ID: {card.get('source_id', '')}; 质量: {card.get('quality', 'high')})\n"
            f"作者: {authors} | URL: {card.get('url', '')}\n"
            f"证据摘录: {card.get('excerpt', '')}"
        )
    return "\n\n".join(web_lines) or "（无网页证据）", "\n\n".join(paper_lines) or "（无论文证据）"


def summarize_evidence_coverage(cards: list[dict[str, Any]]) -> str:
    """Create a tiny coverage summary for the overall coherence critic."""
    web = [card for card in cards if card.get("type") == "web"]
    papers = [card for card in cards if card.get("type") == "paper"]
    domains = sorted({card.get("domain", "") for card in cards if card.get("domain")})
    by_question: Counter[str] = Counter()
    for card in cards:
        ids = card.get("sub_question_ids") or [card.get("sub_question_id", "")]
        for sq_id in ids:
            if sq_id:
                by_question[str(sq_id)] += 1
    coverage = ", ".join(f"{sq_id}={count}" for sq_id, count in sorted(by_question.items()))
    return (
        f"网页证据 {len(web)} 条，论文证据 {len(papers)} 条，独立域名 {len(domains)} 个。\n"
        f"各子问题证据数: {coverage or '无'}。\n"
        f"来源域名: {', '.join(domains[:20]) or '无'}"
    )


def _source_id(url: str, title: str) -> str:
    digest = hashlib.sha1(f"{url}|{title}".encode("utf-8")).hexdigest()[:10]
    return f"src_{digest}"


def normalize_claims(
    raw_claims: Any,
    cards: list[dict[str, Any]],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Validate the Researcher's compact claim list without another LLM call."""
    if not isinstance(raw_claims, list):
        return []

    valid_ids = {str(card.get("source_id", "")) for card in cards}
    normalized: list[dict[str, Any]] = []
    for item in raw_claims[:limit]:
        if not isinstance(item, dict):
            continue
        statement = str(item.get("claim") or item.get("statement") or "").strip()
        if not statement:
            continue
        raw_ids = item.get("source_ids", [])
        if not isinstance(raw_ids, list):
            raw_ids = []
        source_ids = [
            str(source_id) for source_id in raw_ids
            if str(source_id) in valid_ids
        ]
        # Preserve order while removing repeated ids.
        source_ids = list(dict.fromkeys(source_ids))
        confidence = str(item.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        if not source_ids:
            confidence = "low"
        normalized.append(
            {
                "claim": statement,
                "source_ids": source_ids,
                "confidence": confidence,
                "scope": str(item.get("scope", "")).strip(),
            }
        )
    return normalized


def format_claims(
    claims: list[dict[str, Any]],
    cards: list[dict[str, Any]],
) -> str:
    """Render validated claims using the catalog's final citation numbering."""
    citation_map: dict[str, str] = {}
    web_index = 0
    paper_index = 0
    for card in cards:
        source_id = str(card.get("source_id", ""))
        if card.get("type") == "paper":
            paper_index += 1
            citation_map[source_id] = f"[论文{paper_index}]"
        else:
            web_index += 1
            citation_map[source_id] = f"[网页{web_index}]"

    lines: list[str] = []
    for index, claim in enumerate(claims, 1):
        citations = "".join(
            citation_map[source_id]
            for source_id in claim.get("source_ids", [])
            if source_id in citation_map
        )
        confidence = claim.get("confidence", "low")
        scope = str(claim.get("scope", "")).strip()
        suffix = f"；适用范围：{scope}" if scope else ""
        verification = citations or "（无直接来源，需进一步验证）"
        lines.append(
            f"{index}. {claim.get('claim', '')}{suffix} {verification} "
            f"[证据置信度: {confidence}]"
        )
    return "\n".join(lines)


def _compact(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    return normalized[:limit] + ("…" if len(normalized) > limit else "")
