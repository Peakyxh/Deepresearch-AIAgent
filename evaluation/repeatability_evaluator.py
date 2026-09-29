"""
重复性评估器

对研究报告进行段落间重复性评估，通过随机抽取段落对，
使用 LLM 判断每对段落的重复程度，最终计算 avg_repeat_score。
参考 DeepResearch-Eval-main/Atools.py 中的 judge_repeatability_pair 逻辑。
"""

import logging
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import Optional

import json_repair

from llm.base_llm import BaseLLM
from evaluation.prompts import REPEATABILITY_SYSTEM_PROMPT, REPEATABILITY_USER_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class PairRepeatResult:
    """单对段落的重复性评估结果"""

    score: Optional[float] = None
    explanation: Optional[str] = None
    repetitions_found: Optional[list] = None
    confidence: Optional[str] = None
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepeatabilityResult:
    """整篇报告的重复性评估结果"""

    avg_repeat_score: Optional[float] = None
    pair_results: Optional[list] = None
    total_pairs: int = 0
    success_pairs: int = 0
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class RepeatabilityEvaluator:
    """
    重复性评估器

    工作流程：
    1. 按 ## 标题将报告分割为章节
    2. 过滤掉含 URL 和过短的段落
    3. 随机抽取指定数量的段落对
    4. 使用 LLM 评估每对段落的重复程度（0-4分）
    5. 计算 avg_repeat_score（所有成功评估的段落对的平均分）
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_attempts: int = 3,
        retry_delay: float = 2.0,
        pair_nums: int = 30,
        min_paragraph_length: int = 200,
    ):
        """
        Args:
            llm: LLM 实例
            max_attempts: 每对段落评估的最大重试次数
            retry_delay: 重试间隔（秒）
            pair_nums: 随机抽取的段落对数量
            min_paragraph_length: 段落最小长度（字符数），低于此值的段落被过滤
        """
        self.llm = llm
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.pair_nums = pair_nums
        self.min_paragraph_length = min_paragraph_length

    async def evaluate(self, report: str) -> RepeatabilityResult:
        """
        评估整篇报告的重复性

        Args:
            report: 报告内容（Markdown 格式）

        Returns:
            RepeatabilityResult 包含 avg_repeat_score 和每对段落的详细结果
        """
        # 1. 按 ## 标题分割章节
        sections_with_headings = self._extract_sections(report)
        if len(sections_with_headings) < 2:
            return RepeatabilityResult(
                success=False,
                error="报告段落数不足，无法进行重复性评估",
            )

        # 2. 过滤段落
        sections_with_headings = [
            s for s in sections_with_headings
            if "https" not in s and len(s) >= self.min_paragraph_length
        ]
        if len(sections_with_headings) < 2:
            return RepeatabilityResult(
                success=False,
                error="过滤后有效段落数不足，无法进行重复性评估",
            )

        # 3. 生成随机段落对
        pairs = self._generate_random_pairs(sections_with_headings, self.pair_nums)
        logger.info(f"生成 {len(pairs)} 对段落用于重复性评估")

        # 4. 逐对评估
        pair_results = []
        total_score = 0.0
        success_count = 0

        for i, (idx_a, idx_b) in enumerate(pairs):
            passage1 = sections_with_headings[idx_a]
            passage2 = sections_with_headings[idx_b]

            result = await self._evaluate_pair(passage1, passage2)
            pair_results.append({
                "pair_index": i + 1,
                "section_index_a": idx_a,
                "section_index_b": idx_b,
                "result": result.to_dict(),
            })

            if result.success:
                total_score += result.score
                success_count += 1
                logger.debug(f"段落对 {i + 1} 重复性评分: {result.score}")
            else:
                logger.warning(f"段落对 {i + 1} 评估失败: {result.error}")

        # 5. 计算平均分
        if success_count == 0:
            return RepeatabilityResult(
                pair_results=pair_results,
                total_pairs=len(pairs),
                success_pairs=0,
                success=False,
                error="所有段落对评估均失败",
            )

        avg_score = total_score / success_count
        logger.info(f"重复性评估完成: avg_repeat_score={avg_score:.2f} ({success_count}/{len(pairs)} 对成功)")

        return RepeatabilityResult(
            avg_repeat_score=round(avg_score, 2),
            pair_results=pair_results,
            total_pairs=len(pairs),
            success_pairs=success_count,
            success=True,
        )

    async def _evaluate_pair(self, passage1: str, passage2: str) -> PairRepeatResult:
        """
        评估一对段落的重复程度

        Args:
            passage1: 段落1
            passage2: 段落2

        Returns:
            PairRepeatResult
        """
        user_prompt = REPEATABILITY_USER_PROMPT.format(para1=passage1, para2=passage2)

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=REPEATABILITY_SYSTEM_PROMPT,
                )
                result = self._parse_response(response)
                if result.success:
                    return result
                else:
                    logger.debug(f"重复性评估解析失败（第 {attempt} 次）: {result.error}")
            except Exception as e:
                logger.debug(f"重复性评估调用失败（第 {attempt} 次）: {e}")

            if attempt < self.max_attempts:
                time.sleep(self.retry_delay)

        return PairRepeatResult(
            success=False,
            error=f"重复性评估在 {self.max_attempts} 次尝试后仍失败",
        )

    def _parse_response(self, response: str) -> PairRepeatResult:
        """解析 LLM 返回的 JSON 重复性评估结果"""
        try:
            parsed = json_repair.loads(response)
            if isinstance(parsed, list):
                parsed = parsed[-1]

            score = self._extract_score(parsed, "score")
            if score is None:
                return PairRepeatResult(
                    success=False,
                    error=f"无法提取 score 字段: {response[:200]}",
                )

            return PairRepeatResult(
                score=score,
                explanation=parsed.get("explanation", ""),
                repetitions_found=parsed.get("repetitions_found", []),
                confidence=parsed.get("confidence", ""),
                success=True,
            )
        except Exception as e:
            return PairRepeatResult(
                success=False,
                error=f"JSON 解析失败: {e}, 原始响应: {response[:200]}",
            )

    @staticmethod
    def _extract_score(data: dict, key: str) -> Optional[float]:
        """从解析结果中提取分数"""
        value = data.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_sections(markdown_content: str, pattern: str = r"^## .*$") -> list[str]:
        """
        按 ## 标题将报告分割为章节（带标题）

        Args:
            markdown_content: Markdown 报告内容
            pattern: 匹配标题的正则表达式

        Returns:
            章节列表，每个元素包含标题和对应内容
        """
        headings = re.findall(pattern, markdown_content, re.MULTILINE)
        if not headings:
            # 没有标题则按空行分段
            paragraphs = re.split(r"\n\s*\n", markdown_content.strip())
            return [p.strip() for p in paragraphs if p.strip()]

        sections_with_headings = []
        current_pos = 0

        # 第一个标题之前的内容
        first_heading_pos = markdown_content.find(headings[0])
        if first_heading_pos > 0:
            section0 = markdown_content[:first_heading_pos].strip()
            if section0:
                sections_with_headings.append(section0)

        for i, heading in enumerate(headings):
            heading_pos = markdown_content.find(heading, current_pos)
            if heading_pos != -1:
                if current_pos > 0:
                    section_content = markdown_content[current_pos:heading_pos].strip()
                    sections_with_headings.append(heading + "\n" + section_content)
                current_pos = heading_pos + len(heading)

        # 最后一个标题之后的内容
        if current_pos < len(markdown_content):
            last_section = markdown_content[current_pos:].strip()
            if last_section:
                sections_with_headings.append(headings[-1] + "\n" + last_section)

        return sections_with_headings

    @staticmethod
    def _generate_random_pairs(sections: list, pair_nums: int) -> list[tuple[int, int]]:
        """
        随机生成段落对索引

        Args:
            sections: 段落列表
            pair_nums: 需要生成的对数

        Returns:
            索引对列表 [(idx_a, idx_b), ...]
        """
        length = len(sections)
        max_pairs = (length * (length - 1)) // 2
        pair_nums = min(pair_nums, max_pairs)

        all_possible = [
            (a, b)
            for a in range(length - 1)
            for b in range(a + 1, length)
        ]

        if len(all_possible) > pair_nums:
            return random.sample(all_possible, pair_nums)
        return all_possible
