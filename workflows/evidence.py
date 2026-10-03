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
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    return set(latin + chinese)


def _authority_score(domain: str) -> float:
    if domain.endswith((".gov", ".gov.cn", ".edu", ".edu.cn")):
        return 1.0
    if domain.endswith(".org") or "arxiv.org" in domain:
        return 0.7
    return 0.35


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
        return float(item.score or 0) * 0.55 + overlap * 0.3 + _authority_score(_domain(item.url)) * 0.15

    selected: list[SearchResult] = []
    domains: Counter[str] = Counter()
    for result in sorted(results, key=rank, reverse=True):
        domain = _domain(result.url) or result.source or "unknown"
        if domains[domain] >= per_domain_limit:
            continue
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
        cards.append(
            {
                "source_id": _source_id(result.url, result.title),
                "sub_question_id": sub_question_id,
                "type": "web",
                "title": result.title,
                "url": result.url,
                "domain": _domain(result.url),
                "authority": _authority_score(_domain(result.url)),
                "excerpt": _compact(result.content, excerpt_chars),
            }
        )
    for paper in paper_results:
        cards.append(
            {
                "source_id": _source_id(paper.url, paper.title),
                "sub_question_id": sub_question_id,
                "type": "paper",
                "title": paper.title,
                "url": paper.url,
                "domain": _domain(paper.url),
                "authority": 0.9,
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
            f"[网页{index}] {card.get('title', '')}\n"
            f"URL: {card.get('url', '')}\n"
            f"证据摘录: {card.get('excerpt', '')}"
        )
    for index, card in enumerate((c for c in cards if c.get("type") == "paper"), 1):
        authors = card.get("authors", [])
        if isinstance(authors, list):
            authors = ", ".join(authors)
        paper_lines.append(
            f"[论文{index}] {card.get('title', '')} ({card.get('year') or '未知'})\n"
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


def _compact(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    return normalized[:limit] + ("…" if len(normalized) > limit else "")
