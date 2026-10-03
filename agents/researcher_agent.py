"""
Researcher Agent —— 研究员

负责根据研究计划执行搜索，整合网页搜索和论文搜索结果，
对信息进行聚合和可信度排序。

质量保障：
- 内容质量过滤：过滤过短的碎片内容，截断过长的原始内容
- 论文时间过滤：优先保留近年论文
- 论文引用排序：按引用次数降序排列，高引用论文优先
- 语义压缩：格式化文本过长时用 LLM 语义压缩，而非简单截断
"""

import json
from typing import Any

from agents.base_agent import BaseAgent
from llm.base_llm import BaseLLM
from search.base_search import BaseSearch, SearchResult
from search.search_factory import SearchFactory
from academic.base_academic import BaseAcademic, PaperResult
from academic.academic_factory import AcademicFactory
from config import settings
from memory.context_manager import ContextManager
from workflows.evidence import (
    build_evidence_cards,
    select_paper_evidence,
    select_web_evidence,
)
from prompts.researcher_prompt import (
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCHER_USER_PROMPT,
    RESEARCHER_CRITIQUE_SECTION,
    RESEARCHER_CACHED_SECTION,
)


class ResearcherAgent(BaseAgent):
    """
    Researcher Agent

    工作流程：
    1. 读取 Planner 输出的子问题和搜索关键词
    2. 对每个子问题执行网页搜索和论文搜索
    3. 将搜索结果格式化为文本
    4. 如果格式化文本过长，使用 ContextManager 语义压缩
    5. 调用 LLM 整合研究发现
    6. 将结果写入工作流状态
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        search_engine: BaseSearch | None = None,
        academic_search: BaseAcademic | None = None,
        context_manager: ContextManager | None = None,
    ):
        """
        初始化 Researcher Agent

        Args:
            llm: LLM 实例
            search_engine: 搜索引擎实例，如果为 None 则从工厂创建
            academic_search: 学术搜索引擎实例，如果为 None 则从工厂创建
            context_manager: 上下文管理器，用于语义压缩
        """
        super().__init__(name="Researcher", llm=llm, context_manager=context_manager)
        # 搜索引擎和学术搜索引擎可以外部注入，也可以自动创建
        self.search_engine = search_engine or SearchFactory.create()
        self.academic_search = academic_search or AcademicFactory.create()

    async def _execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        执行研究搜索

        Args:
            state: 工作流状态字典

        Returns:
            更新后的状态字典，包含 search_results、paper_results、findings
        """
        query = state.get("query", "")
        sub_questions = state.get("sub_questions", [])
        search_keywords = state.get("search_keywords", [])
        research_plan = state.get("research_plan", "")
        critique_feedback = state.get("critique_feedback_for_researcher", "")

        self.log(f"开始研究搜索，共 {len(sub_questions)} 个子问题")

        # 如果有上一轮的审查反馈，打印出来
        if critique_feedback:
            self.log(f"📋 上一轮审查反馈: {critique_feedback[:100]}...")

        # 执行搜索
        all_search_results: list[SearchResult] = []
        all_paper_results: list[PaperResult] = []

        # 如果有搜索关键词，按关键词搜索
        if search_keywords:
            for kw_group in search_keywords:
                # 合并中英文关键词进行搜索
                keywords_zh = kw_group.get("keywords_zh", [])
                keywords_en = kw_group.get("keywords_en", [])

                # 用中文关键词搜索网页
                for kw in keywords_zh:
                    self.log(f"搜索网页（中文）: {kw}")
                    try:
                        results = await self.search_engine.search(kw)
                        all_search_results.extend(results)
                        self.log(f"  找到 {len(results)} 条结果")
                    except Exception as e:
                        self.log(f"  搜索失败: {e}")

                # 用英文关键词搜索网页和论文
                for kw in keywords_en:
                    self.log(f"搜索网页（英文）: {kw}")
                    try:
                        results = await self.search_engine.search(kw)
                        all_search_results.extend(results)
                        self.log(f"  找到 {len(results)} 条结果")
                    except Exception as e:
                        self.log(f"  搜索失败: {e}")

                    self.log(f"搜索论文（英文）: {kw}")
                    try:
                        papers = await self.academic_search.search_papers(kw)
                        all_paper_results.extend(papers)
                        self.log(f"  找到 {len(papers)} 篇论文")
                    except Exception as e:
                        self.log(f"  论文搜索失败: {e}")
        else:
            # 没有搜索关键词，直接用原始问题搜索
            self.log(f"使用原始问题搜索: {query}")
            try:
                results = await self.search_engine.search(query)
                all_search_results.extend(results)
            except Exception as e:
                self.log(f"网页搜索失败: {e}")

            try:
                papers = await self.academic_search.search_papers(query)
                all_paper_results.extend(papers)
            except Exception as e:
                self.log(f"论文搜索失败: {e}")

        # 去重搜索结果（按 URL 去重）
        seen_urls = set()
        unique_search_results = []
        for result in all_search_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_search_results.append(result)

        # 去重论文结果（按 URL 去重）
        seen_paper_urls = set()
        unique_paper_results = []
        for paper in all_paper_results:
            if paper.url not in seen_paper_urls:
                seen_paper_urls.add(paper.url)
                unique_paper_results.append(paper)

        self.log(f"去重后：{len(unique_search_results)} 条网页结果，{len(unique_paper_results)} 篇论文")

        # ==================== 内容质量过滤 ====================
        # 1. 过滤内容过短的碎片结果（如只有标题没有摘要的）
        # 2. 截断内容过长的结果（避免原始网页内容淹没有效信息）
        filtered_search_results = self._filter_search_results(unique_search_results)
        filtered_search_results = select_web_evidence(
            filtered_search_results,
            f"{query} {research_plan}",
            settings.max_web_sources_per_sub_question,
            settings.max_sources_per_domain,
        )
        self.log(f"内容质量过滤：{len(unique_search_results)} → {len(filtered_search_results)} 条网页结果")

        # ==================== 论文时间过滤 + 引用排序 ====================
        # 1. 过滤年份过早的论文（默认只保留 2020 年及之后）
        # 2. 按引用次数降序排列（高引用论文更权威，优先展示）
        filtered_paper_results = self._filter_and_sort_papers(unique_paper_results)
        filtered_paper_results = select_paper_evidence(
            filtered_paper_results,
            f"{query} {research_plan}",
            settings.max_paper_sources_per_sub_question,
        )
        self.log(f"论文质量过滤：{len(unique_paper_results)} → {len(filtered_paper_results)} 篇论文")

        # 将搜索结果存入状态
        state["search_results"] = [r.model_dump() for r in filtered_search_results]
        state["paper_results"] = [p.model_dump() for p in filtered_paper_results]

        # 收集所有参考文献来源（供 Writer 自动拼接，无需 LLM 生成）
        sources = []
        for r in filtered_search_results:
            sources.append({
                "title": r.title,
                "url": r.url,
                "type": "web",
            })
        for p in filtered_paper_results:
            sources.append({
                "title": p.title,
                "url": p.url,
                "type": "paper",
                "year": p.year,
                "authors": ", ".join(p.authors[:3]) + ("等" if len(p.authors) > 3 else ""),
            })
        state["sources"] = sources
        state["evidence_cards"] = build_evidence_cards(
            filtered_search_results,
            filtered_paper_results,
            "serial",
            settings.evidence_excerpt_chars,
        )
        self.log(f"收集了 {len(sources)} 条参考文献来源")

        # 格式化搜索结果为文本，供 LLM 整合
        search_text = self._format_search_results(filtered_search_results)
        paper_text = self._format_paper_results(filtered_paper_results)

        # ==================== 语义压缩 ====================
        # 如果格式化后的搜索结果 + 论文结果过长，进行语义压缩
        # 而非简单截断，保留关键信息
        if self.context_manager:
            combined_text = search_text + "\n\n" + paper_text
            combined_tokens = self.context_manager.estimate_text_tokens(combined_text)

            if combined_tokens > self.context_manager.max_tokens:
                self.log(f"搜索结果+论文结果过长 ({combined_tokens} tokens)，进行语义压缩...")

                # 分别压缩搜索结果和论文结果
                search_tokens = self.context_manager.estimate_text_tokens(search_text)
                if search_tokens > self.context_manager.get_compress_threshold("search"):
                    self.log(f"  压缩网页搜索结果 ({search_tokens} tokens)...")
                    search_text = await self.context_manager.compress_search_results(search_text)
                    self.log(f"  网页搜索结果压缩完成: {search_tokens} → "
                             f"{self.context_manager.estimate_text_tokens(search_text)} tokens")

                paper_tokens = self.context_manager.estimate_text_tokens(paper_text)
                if paper_tokens > self.context_manager.get_compress_threshold("paper"):
                    self.log(f"  压缩论文搜索结果 ({paper_tokens} tokens)...")
                    paper_text = await self.context_manager.compress_paper_results(paper_text)
                    self.log(f"  论文搜索结果压缩完成: {paper_tokens} → "
                             f"{self.context_manager.estimate_text_tokens(paper_text)} tokens")

        # 调用 LLM 整合研究发现
        self.log("正在调用 LLM 整合研究发现...")

        # 构造审查反馈部分（如果有）
        critique_section = ""
        if critique_feedback:
            critique_section = RESEARCHER_CRITIQUE_SECTION.format(
                critique_feedback=critique_feedback
            )

        # 构造缓存上下文部分（如果有）
        cached_section = ""
        cached_search = state.get("cached_search_context", "")
        cached_papers = state.get("cached_paper_context", "")
        if cached_search or cached_papers:
            cached_parts = []
            if cached_search:
                cached_parts.append(f"【缓存网页结果】\n{cached_search}")
            if cached_papers:
                cached_parts.append(f"【缓存论文结果】\n{cached_papers}")
            cached_section = RESEARCHER_CACHED_SECTION.format(
                cached_context="\n\n".join(cached_parts)
            )

        user_prompt = RESEARCHER_USER_PROMPT.format(
            query=query,
            plan=research_plan,
            critique_section=critique_section,
            cached_section=cached_section,
            search_results=search_text,
            paper_results=paper_text,
        )

        response = await self.generate(
            prompt=user_prompt,
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            max_tokens=settings.researcher_max_tokens,
        )

        # 解析 LLM 返回的 JSON
        findings_data = self._parse_json_response(response)

        if findings_data:
            state["findings"] = findings_data.get("findings", "")
            self.log(f"研究发现: {state['findings'][:100]}...")
        else:
            # JSON 解析失败，直接使用原始回复
            self.log("⚠️ JSON 解析失败，使用原始回复作为研究发现")
            state["findings"] = response

        # 更新当前步骤
        state["current_step"] = "critic"
        state["use_orchestrator"] = False
        return state

    def _format_search_results(self, results: list[SearchResult]) -> str:
        """
        将搜索结果格式化为文本

        Args:
            results: SearchResult 列表

        Returns:
            格式化后的文本
        """
        if not results:
            return "（无网页搜索结果）"

        formatted = []
        for i, r in enumerate(results, 1):
            text = f"[网页{i}] 标题: {r.title}\n"
            text += f"       链接: {r.url}\n"
            text += f"       内容: {r.content}\n"
            text += f"       来源: {r.source} | 评分: {r.score:.2f}"
            formatted.append(text)

        return "\n\n".join(formatted)

    def _format_paper_results(self, papers: list[PaperResult]) -> str:
        """
        将论文搜索结果格式化为文本

        Args:
            papers: PaperResult 列表

        Returns:
            格式化后的文本
        """
        if not papers:
            return "（无论文搜索结果）"

        formatted = []
        for i, p in enumerate(papers, 1):
            text = f"[论文{i}] 标题: {p.title}\n"
            text += f"       作者: {', '.join(p.authors[:3])}{'等' if len(p.authors) > 3 else ''}\n"
            text += f"       年份: {p.year or '未知'}\n"
            text += f"       链接: {p.url}\n"
            text += f"       摘要: {p.abstract[:200]}{'...' if len(p.abstract) > 200 else ''}"
            if p.citation_count is not None:
                text += f"\n       引用: {p.citation_count} 次"
            formatted.append(text)

        return "\n\n".join(formatted)

    def _parse_json_response(self, response: str) -> dict | None:
        """
        解析 LLM 返回的 JSON 响应

        与 PlannerAgent 中的方法相同，处理 markdown 代码块包裹的情况。
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
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    def _filter_search_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        网页搜索结果内容质量过滤

        过滤规则：
        1. 内容长度 < search_content_min_length 的视为碎片内容，过滤掉
           （如只有标题没有摘要、只有一句话的搜索结果）
        2. 内容长度 > search_content_max_length 的截断到最大长度
           （避免原始网页内容过长，淹没有效信息）

        Args:
            results: 原始搜索结果列表

        Returns:
            过滤后的搜索结果列表
        """
        min_len = settings.search_content_min_length
        max_len = settings.search_content_max_length
        filtered = []

        for r in results:
            content_len = len(r.content)

            # 过滤内容过短的碎片结果
            if content_len < min_len:
                self.log(f"  过滤碎片内容: '{r.title[:30]}...' (内容仅{content_len}字符)")
                continue

            # 截断内容过长的结果
            if content_len > max_len:
                self.log(f"  截断过长内容: '{r.title[:30]}...' ({content_len} → {max_len}字符)")
                r = SearchResult(
                    title=r.title,
                    url=r.url,
                    content=r.content[:max_len] + "...",
                    source=r.source,
                    score=r.score,
                )

            filtered.append(r)

        return filtered

    def _filter_and_sort_papers(self, papers: list[PaperResult]) -> list[PaperResult]:
        """
        论文搜索结果质量过滤与排序

        过滤规则：
        1. 年份过滤：只保留 paper_min_year 及之后的论文
           （默认 2020 年，0 表示不过滤）
           - 年份为 None 的论文保留（无法判断年份，可能是预印本）

        排序规则：
        1. 按引用次数降序排列（高引用论文更权威，优先展示）
           - 有引用次数的排在前面
           - 引用次数为 None 的排在后面
           - paper_sort_by_citation=False 时跳过排序

        Args:
            papers: 原始论文结果列表

        Returns:
            过滤并排序后的论文结果列表
        """
        min_year = settings.paper_min_year
        filtered = []

        for p in papers:
            # 年份过滤：年份为 None 时保留（无法判断年份）
            if min_year > 0 and p.year is not None and p.year < min_year:
                self.log(f"  过滤早期论文: '{p.title[:30]}...' (年份: {p.year})")
                continue

            filtered.append(p)

        # 按引用次数降序排列
        if settings.paper_sort_by_citation:
            # 排序策略：有引用次数的按次数降序，无引用次数的排在后面
            def sort_key(paper: PaperResult) -> float:
                # citation_count 为 None 时返回 -1，排在最后
                # 否则返回引用次数（降序用负数）
                if paper.citation_count is None:
                    return -1.0
                return float(paper.citation_count)

            filtered.sort(key=sort_key, reverse=True)

            # 打印排序后的前几篇论文信息
            if filtered:
                self.log("  论文排序（按引用次数降序）:")
                for i, p in enumerate(filtered[:5], 1):
                    citation = f"{p.citation_count}次引用" if p.citation_count is not None else "无引用数据"
                    year_str = str(p.year) if p.year else "未知年份"
                    self.log(f"    {i}. [{citation}] ({year_str}) {p.title[:40]}...")

        return filtered
