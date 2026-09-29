"""
质量评估器

对研究报告进行全面性、连贯性、清晰度、洞察力和总体评分的五维度质量评估。
参考 DeepResearch-Eval-main/Atools.py 中的 judge_quality 逻辑，
适配当前项目的 LLM 工厂体系。
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import json_repair

from llm.base_llm import BaseLLM
from evaluation.prompts import QUALITY_SYSTEM_PROMPT, QUALITY_USER_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """质量评估结果"""

    comprehensiveness_score: Optional[float] = None
    coherence_score: Optional[float] = None
    clarity_score: Optional[float] = None
    insightfulness_score: Optional[float] = None
    overall_score: Optional[float] = None
    reason: Optional[str] = None
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def average_score(self) -> Optional[float]:
        """计算四个维度（不含 overall）的平均分"""
        scores = [
            self.comprehensiveness_score,
            self.coherence_score,
            self.clarity_score,
            self.insightfulness_score,
        ]
        valid = [s for s in scores if s is not None]
        return sum(valid) / len(valid) if valid else None


class QualityEvaluator:
    """
    质量评估器

    使用 LLM 对研究报告进行五维度质量评分：
    - Comprehensiveness（全面性）
    - Coherence（连贯性）
    - Clarity（清晰度）
    - Insightfulness（洞察力）
    - Overall（总体）
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_attempts: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Args:
            llm: LLM 实例（通过 LLMFactory 创建）
            max_attempts: 最大重试次数
            retry_delay: 重试间隔（秒）
        """
        self.llm = llm
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay

    async def evaluate(self, topic: str, report: str) -> QualityResult:
        """
        评估单篇报告的质量

        Args:
            topic: 研究问题/主题
            report: 报告内容（Markdown 格式）

        Returns:
            QualityResult 包含五维度评分和评估理由
        """
        user_prompt = QUALITY_USER_PROMPT.format(question=topic, paragraph=report)

        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.debug(f"质量评估第 {attempt}/{self.max_attempts} 次尝试")
                response = await self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=QUALITY_SYSTEM_PROMPT,
                )
                result = self._parse_response(response)
                if result.success:
                    logger.info(
                        f"质量评估成功: overall={result.overall_score}, "
                        f"avg={result.average_score:.2f}"
                    )
                    return result
                else:
                    logger.warning(f"质量评估解析失败（第 {attempt} 次）: {result.error}")

            except Exception as e:
                logger.warning(f"质量评估调用失败（第 {attempt} 次）: {e}")

            if attempt < self.max_attempts:
                time.sleep(self.retry_delay)

        return QualityResult(
            success=False,
            error=f"质量评估在 {self.max_attempts} 次尝试后仍失败",
        )

    def _parse_response(self, response: str) -> QualityResult:
        """
        解析 LLM 返回的 JSON 评分结果

        Args:
            response: LLM 原始返回文本

        Returns:
            QualityResult
        """
        try:
            parsed = json_repair.loads(response)
            # json_repair 有时返回 list，取最后一个元素
            if isinstance(parsed, list):
                parsed = parsed[-1]

            return QualityResult(
                comprehensiveness_score=self._extract_score(parsed, "Comprehensiveness_Score"),
                coherence_score=self._extract_score(parsed, "Coherence_Score"),
                clarity_score=self._extract_score(parsed, "Clarity_Score"),
                insightfulness_score=self._extract_score(parsed, "Insightfulness_Score"),
                overall_score=self._extract_score(parsed, "Overall_Score"),
                reason=parsed.get("Reason", ""),
                success=True,
            )
        except Exception as e:
            return QualityResult(
                success=False,
                error=f"JSON 解析失败: {e}, 原始响应: {response[:200]}",
            )

    @staticmethod
    def _extract_score(data: dict, key: str) -> Optional[float]:
        """从解析结果中提取分数，兼容字符串和数值类型"""
        value = data.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
