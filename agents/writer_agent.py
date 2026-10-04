"""
Writer Agent —— 研究报告撰写者

负责将研究结果整理成结构化的 Markdown 研究报告，
自动组织章节、插入引用来源。
参考文献由程序自动拼接，无需 LLM 生成，节省 token。

支持两轮交互式写作：
- 第一轮：生成大纲+核心论点，展示给用户审核
- 用户可修改大纲或提出建议
- 第二轮：根据大纲和用户建议生成完整报告
"""

import json
from datetime import datetime
from typing import Any

from json_repair import loads as repair_json_loads

from agents.base_agent import BaseAgent
from config import settings
from llm.base_llm import BaseLLM
from memory.context_manager import ContextManager
from workflows.evidence import format_evidence_catalog
from prompts.writer_prompt import (
    WRITER_SYSTEM_PROMPT,
    WRITER_USER_PROMPT,
    WRITER_OUTLINE_SYSTEM_PROMPT,
    WRITER_OUTLINE_USER_PROMPT,
    WRITER_WITH_OUTLINE_SYSTEM_PROMPT,
    WRITER_WITH_OUTLINE_USER_PROMPT,
)


class WriterAgent(BaseAgent):
    """
    Writer Agent

    工作流程：
    1. 读取研究发现、审查反馈、搜索结果
    2. 如果输入文本过长，使用 ContextManager 语义压缩
    3. 如果启用交互式写作：
       a. 第一轮：调用 LLM 生成大纲（JSON），展示给用户
       b. 用户审核大纲，可输入修改建议
       c. 第二轮：根据大纲+用户建议生成完整报告
    4. 如果未启用交互式写作：直接生成报告
    5. 程序自动在报告末尾拼接参考文献
    6. 保存报告到文件
    7. 将结果写入工作流状态
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        context_manager: ContextManager | None = None,
        interaction_handler: Any = None,
    ):
        super().__init__(name="Writer", llm=llm, context_manager=context_manager)
        self.interactive = settings.writer_interactive
        self.max_tokens = settings.writer_max_tokens
        self.interaction_handler = interaction_handler

    async def _execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        生成研究报告

        Args:
            state: 工作流状态字典

        Returns:
            更新后的状态字典，包含 report
        """
        query = state.get("query", "")
        findings = state.get("findings", "")
        critique = state.get("critique_feedback", "")
        writer_suggestions = state.get("writer_revision_suggestions", [])
        pending_suggestions = [
            item for item in writer_suggestions if item and item not in critique
        ]
        if pending_suggestions:
            suggestion_text = "\n".join(f"- {item}" for item in pending_suggestions)
            critique = (
                f"{critique}\n\n写作阶段必须处理的修订建议:\n{suggestion_text}"
            ).strip()
        search_results = state.get("search_results", [])
        paper_results = state.get("paper_results", [])
        sources = state.get("sources", [])
        evidence_cards = state.get("evidence_cards", [])

        self.log("开始撰写研究报告...")

        # 格式化搜索结果和论文结果，供 LLM 引用
        if evidence_cards:
            search_text, paper_text = format_evidence_catalog(evidence_cards)
            self.log(f"使用 {len(evidence_cards)} 张紧凑证据卡片撰写报告")
        else:
            search_text = self._format_sources(search_results, "网页")
            paper_text = self._format_sources(paper_results, "论文")

        # ==================== 语义压缩 ====================
        findings, search_text, paper_text = await self._compress_if_needed(
            findings, search_text, paper_text
        )

        # ==================== 两轮交互式写作 ====================
        if self.interactive:
            report = await self._interactive_write(
                state, query, findings, critique, search_text, paper_text
            )
        else:
            report = await self._direct_write(
                query, findings, critique, search_text, paper_text
            )

        # 添加报告元信息
        report = self._add_metadata(report, query)

        # 自动拼接参考文献（无需 LLM 生成，节省 token）
        if sources:
            report = self._append_references(report, sources)
            self.log(f"自动拼接了 {len(sources)} 条参考文献")

        # 存入状态
        state["report"] = report
        state["current_step"] = "completed"

        self.log(f"研究报告生成完成，长度: {len(report)} 字符")

        return state

    async def _compress_if_needed(
        self, findings: str, search_text: str, paper_text: str
    ) -> tuple[str, str, str]:
        """
        如果输入文本过长，进行语义压缩

        Returns:
            压缩后的 (findings, search_text, paper_text)
        """
        if not self.context_manager:
            return findings, search_text, paper_text

        # 压缩研究发现
        findings_tokens = self.context_manager.estimate_text_tokens(findings)
        if findings_tokens > self.context_manager.get_compress_threshold("findings"):
            self.log(f"研究发现过长 ({findings_tokens} tokens)，进行语义压缩...")
            findings = await self.context_manager.compress_findings(findings)
            self.log(f"研究发现压缩完成: {findings_tokens} → "
                     f"{self.context_manager.estimate_text_tokens(findings)} tokens")

        # 压缩来源信息
        combined_sources = search_text + "\n\n" + paper_text
        sources_tokens = self.context_manager.estimate_text_tokens(combined_sources)
        if sources_tokens > self.context_manager.get_compress_threshold("sources"):
            self.log(f"来源信息过长 ({sources_tokens} tokens)，进行语义压缩...")

            search_tokens = self.context_manager.estimate_text_tokens(search_text)
            if search_tokens > self.context_manager.get_compress_threshold("search"):
                search_text = await self.context_manager.compress_search_results(search_text)
                self.log(f"  网页来源压缩完成: {search_tokens} → "
                         f"{self.context_manager.estimate_text_tokens(search_text)} tokens")

            paper_tokens = self.context_manager.estimate_text_tokens(paper_text)
            if paper_tokens > self.context_manager.get_compress_threshold("paper"):
                paper_text = await self.context_manager.compress_paper_results(
                    paper_text,
                )
                self.log(f"  论文来源压缩完成: {paper_tokens} → "
                         f"{self.context_manager.estimate_text_tokens(paper_text)} tokens")

        return findings, search_text, paper_text

    async def _interactive_write(
        self,
        state: dict[str, Any],
        query: str,
        findings: str,
        critique: str,
        search_text: str,
        paper_text: str,
    ) -> str:
        """
        两轮交互式写作

        第一轮：生成大纲，展示给用户审核
        第二轮：根据大纲+用户建议生成完整报告
        """
        # ==================== 第一轮：生成大纲 ====================
        self.log("第一轮：生成报告大纲...")

        # 注入上游关键背景（意图澄清 + 研究计划），确保大纲不偏离用户真实意图
        context_prefix = self.context_prompt_prefix()
        if context_prefix:
            self.log("已注入上游关键背景（意图澄清/研究计划）")

        saved_outline = state.get("report_outline", "")
        if saved_outline:
            outline_text = str(saved_outline)
            outline_data = self._parse_json_response(outline_text)
            outline_display = str(
                state.get("report_outline_display", "")
                or (
                    self._format_outline_for_display(outline_data)
                    if outline_data
                    else outline_text
                )
            )
            outline_payload = state.get("report_outline_payload") or outline_data or outline_text
            self.log("从检查点恢复已生成的报告大纲")
        else:
            outline_prompt = context_prefix + WRITER_OUTLINE_USER_PROMPT.format(
                query=query,
                findings=findings,
                critique=critique if critique else "无审查反馈",
            )

            outline_response = await self.generate(
                prompt=outline_prompt,
                system_prompt=WRITER_OUTLINE_SYSTEM_PROMPT,
                max_tokens=settings.writer_outline_max_tokens,
            )

            outline_data = self._parse_json_response(outline_response)

            if outline_data:
                outline_display = self._format_outline_for_display(outline_data)
                outline_text = json.dumps(outline_data, ensure_ascii=False, indent=2)
                outline_payload = outline_data
                self.log("大纲生成完成")
            else:
                self.log("⚠️ 大纲 JSON 解析失败，使用原始文本")
                outline_display = outline_response
                outline_text = outline_response
                outline_payload = outline_response

        # 保存大纲到状态
        state["report_outline"] = outline_text
        state["report_outline_display"] = outline_display
        state["report_outline_payload"] = outline_payload

        # ==================== 用户交互：审核大纲 ====================
        user_feedback = await self._interact_with_user_on_outline(
            outline_display, outline_payload
        )

        # 保存用户反馈到状态
        state["user_outline_feedback"] = user_feedback

        # ==================== 第二轮：基于大纲生成完整报告 ====================
        self.log("第二轮：基于大纲生成完整报告...")

        user_prompt = context_prefix + WRITER_WITH_OUTLINE_USER_PROMPT.format(
            query=query,
            outline=outline_text,
            user_feedback=user_feedback if user_feedback else "无修改建议，按原大纲撰写",
            findings=findings,
            critique=critique if critique else "无审查反馈",
            search_results=search_text,
            paper_results=paper_text,
        )

        report = await self.generate(
            prompt=user_prompt,
            system_prompt=WRITER_WITH_OUTLINE_SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
        )

        return report

    async def _direct_write(
        self,
        query: str,
        findings: str,
        critique: str,
        search_text: str,
        paper_text: str,
    ) -> str:
        """
        直接生成报告（非交互模式，原有行为）
        """
        user_prompt = self.context_prompt_prefix() + WRITER_USER_PROMPT.format(
            query=query,
            findings=findings,
            critique=critique if critique else "无审查反馈",
            search_results=search_text,
            paper_results=paper_text,
        )

        self.log("正在调用 LLM 生成研究报告...")
        report = await self.generate(
            prompt=user_prompt,
            system_prompt=WRITER_SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
        )

        return report

    def _format_outline_for_display(self, outline_data: dict) -> str:
        """
        将大纲 JSON 格式化为用户可读的文本

        Args:
            outline_data: 大纲字典

        Returns:
            格式化后的大纲文本
        """
        lines = []

        title = outline_data.get("title", "未命名报告")
        lines.append(f"标题: {title}")

        summary_points = outline_data.get("summary_points", [])
        if summary_points:
            lines.append("")
            lines.append("摘要要点:")
            for i, point in enumerate(summary_points, 1):
                lines.append(f"  {i}. {point}")

        sections = outline_data.get("sections", [])
        if sections:
            lines.append("")
            lines.append("章节结构:")
            for i, section in enumerate(sections, 1):
                heading = section.get("heading", "")
                lines.append(f"  {i}. {heading}")
                key_arguments = section.get("key_arguments", [])
                for arg in key_arguments:
                    lines.append(f"     - {arg}")

        conclusion_points = outline_data.get("conclusion_points", [])
        if conclusion_points:
            lines.append("")
            lines.append("结论要点:")
            for i, point in enumerate(conclusion_points, 1):
                lines.append(f"  {i}. {point}")

        return "\n".join(lines)

    async def _interact_with_user_on_outline(
        self, outline_display: str, outline_json: dict[str, Any] | str
    ) -> str:
        """
        展示大纲给用户，获取修改建议

        Args:
            outline_display: 格式化后的大纲文本

        Returns:
            用户的修改建议，空字符串表示无修改
        """
        if self.interaction_handler:
            response = await self.interaction_handler.request(
                kind="outline_review",
                payload={
                    "outline": outline_display,
                    "outline_json": outline_json,
                },
            )
            if isinstance(response, dict):
                return str(response.get("feedback", "")).strip()
            return ""

        return self._interact_with_user_on_outline_cli(outline_display)

    def _interact_with_user_on_outline_cli(self, outline_display: str) -> str:
        """CLI 兼容路径：保留原有终端大纲审核。"""
        print("\n📝 报告大纲已生成，请查看：")
        print("━" * 50)
        print(outline_display)
        print("━" * 50)
        print("请输入修改建议（直接回车确认大纲，输入 'skip' 跳过交互直接生成）:")

        try:
            feedback = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            self.log("用户跳过大纲审核")
            return ""

        if feedback.lower() == "skip":
            self.log("用户跳过大纲审核")
            return ""

        if feedback:
            self.log(f"用户对大纲提出了修改建议: {feedback[:80]}...")
        else:
            self.log("用户确认大纲，无需修改")

        return feedback

    def _format_sources(self, sources: list[dict], source_type: str) -> str:
        """
        格式化来源信息，供 LLM 在报告中引用

        Args:
            sources: 来源字典列表
            source_type: 来源类型（"网页" 或 "论文"）

        Returns:
            格式化后的文本
        """
        if not sources:
            return f"（无{source_type}来源）"

        formatted = []
        for i, s in enumerate(sources, 1):
            if source_type == "网页":
                title = s.get("title", "")
                url = s.get("url", "")
                content = s.get("content", "")
                text = f"[网页{i}] {title}\n    URL: {url}\n    摘要: {content}"
            else:
                title = s.get("title", "")
                authors = ", ".join(s.get("authors", [])[:3])
                url = s.get("url", "")
                year = s.get("year", "未知")
                text = f"[论文{i}] {title}\n    作者: {authors}\n    年份: {year}\n    URL: {url}"
            formatted.append(text)

        return "\n\n".join(formatted)

    def _add_metadata(self, report: str, query: str) -> str:
        """
        在报告头部添加元信息

        Args:
            report: LLM 生成的报告内容
            query: 原始研究问题

        Returns:
            添加了元信息的完整报告
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = "> 📋 **DeepResearch Agent 研究报告**\n"
        metadata += f"> 研究问题：{query}\n"
        metadata += f"> 生成时间：{timestamp}\n"
        metadata += "> ---\n\n"

        return metadata + report

    def _append_references(self, report: str, sources: list[dict]) -> str:
        """
        在报告末尾自动拼接参考文献

        按网页来源和论文来源分组排列，
        无需 LLM 生成，节省 token。

        Args:
            report: 已生成的报告内容
            sources: 参考文献来源列表

        Returns:
            拼接了参考文献的完整报告
        """
        lines = [report, ""]

        # 分组：网页来源和论文来源
        web_sources = [s for s in sources if s.get("type") == "web"]
        paper_sources = [s for s in sources if s.get("type") == "paper"]

        lines.append("## 参考文献")
        lines.append("")

        # 网页来源（从1开始编号）
        if web_sources:
            lines.append("### 网页来源")
            lines.append("")
            for i, src in enumerate(web_sources, 1):
                title = src.get("title", "未知标题")
                url = src.get("url", "")
                if url:
                    lines.append(f"[网页{i}] {title} 链接: {url}")
                    lines.append("")

        # 论文来源（从1开始编号）
        if paper_sources:
            lines.append("### 论文来源")
            lines.append("")
            for i, src in enumerate(paper_sources, 1):
                title = src.get("title", "未知标题")
                url = src.get("url", "")
                authors = src.get("authors", "")
                year = src.get("year")
                if url:
                    lines.append(f"[论文{i}] {title} - {authors} ({year}) 链接: {url}")
                    lines.append("")

        return "\n".join(lines)

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
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    pass
            try:
                repaired = repair_json_loads(text)
                return repaired if isinstance(repaired, dict) else None
            except (ValueError, TypeError, json.JSONDecodeError):
                return None
