"""
评估入口

整合所有评估器（Quality / Repeatability / Fact 等），
提供统一的评估接口，支持单篇和批量评估。
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from config import settings
from llm.llm_factory import LLMFactory
from evaluation.quality_evaluator import QualityEvaluator, QualityResult
from evaluation.repeatability_evaluator import RepeatabilityEvaluator, RepeatabilityResult
from evaluation.fact_evaluator import FactEvaluator, FactEvalResult

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """单篇报告的完整评估结果"""

    file_id: str = ""
    topic: str = ""
    quality: Optional[dict] = None
    repeatability: Optional[dict] = None
    avg_repeat_score: Optional[float] = None
    fact: Optional[dict] = None
    fact_score: Optional[float] = None
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class Evaluator:
    """
    评估器入口

    整合各维度评估器，提供：
    - evaluate_one(): 评估单篇报告
    - evaluate_batch(): 批量评估（支持断点续传）
    - evaluate_file(): 评估单个文件
    """

    def __init__(
        self,
        max_attempts: int = 3,
        retry_delay: float = 2.0,
        repeat_pair_nums: int = 30,
        enable_fact_check: bool = True,
        jina_api_key: Optional[str] = None,
        max_fact_check_sentences: int = 5,
    ):
        """
        Args:
            max_attempts: 每次评估的最大重试次数
            retry_delay: 重试间隔（秒）
            repeat_pair_nums: 重复性评估的段落对数量
            enable_fact_check: 是否启用事实核查
            jina_api_key: Jina API 密钥，为 None 则从 settings.jina_api_key 读取
            max_fact_check_sentences: 事实核查最大句子数，超出时随机采样（0 不限制）
        """
        # 评估使用专用 LLM 提供者（默认 qwen），留空则回退到全局 llm_provider
        eval_provider = settings.evaluator_llm_provider or settings.llm_provider
        llm = LLMFactory.create(provider=eval_provider)
        logger.info(f"评估器使用 LLM 提供者: {eval_provider}")

        self.quality_evaluator = QualityEvaluator(
            llm=llm,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
        )
        self.repeatability_evaluator = RepeatabilityEvaluator(
            llm=llm,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            pair_nums=repeat_pair_nums,
        )
        self.enable_fact_check = enable_fact_check
        # Jina API 密钥优先级：参数传入 > settings 配置 > 环境变量
        effective_jina_key = jina_api_key or settings.jina_api_key
        if enable_fact_check:
            try:
                self.fact_evaluator = FactEvaluator(
                    llm=llm,
                    jina_api_key=effective_jina_key,
                    max_attempts=max_attempts,
                    retry_delay=retry_delay,
                    max_check_sentences=max_fact_check_sentences,
                )
            except ValueError as e:
                logger.warning(f"事实核查初始化失败（已禁用）: {e}")
                self.enable_fact_check = False
                self.fact_evaluator = None
        else:
            self.fact_evaluator = None
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay

    async def evaluate_one(self, topic: str, report: str, record_id: str = "") -> EvalResult:
        """
        评估单篇报告

        Args:
            topic: 研究问题/主题
            report: 报告内容（Markdown 格式）
            record_id: 记录 ID，用于命名输出文件（如 "1" -> "1.json"）

        Returns:
            EvalResult 包含质量评估和重复性评估结果
        """
        file_id = record_id if record_id else hashlib.md5(topic.encode("utf-8")).hexdigest()
        logger.info(f"开始评估: id={file_id}, topic={topic[:50]}")

        # Quality Evaluation
        quality_result = await self.quality_evaluator.evaluate(topic, report)

        if not quality_result.success:
            return EvalResult(
                file_id=file_id,
                topic=topic,
                quality=None,
                success=False,
                error=f"质量评估失败: {quality_result.error}",
            )

        # Repeatability Evaluation
        repeat_result = await self.repeatability_evaluator.evaluate(report)

        avg_repeat_score = repeat_result.avg_repeat_score if repeat_result.success else None

        # Fact Check Evaluation
        fact_result = None
        fact_score = None
        if self.enable_fact_check and self.fact_evaluator:
            fact_result = await self.fact_evaluator.evaluate(report)
            if fact_result.success:
                fact_score = fact_result.fact_score
                logger.info(
                    f"事实核查完成: 支持 {fact_result.supported}/{fact_result.success_checks}, "
                    f"fact_score={fact_score}"
                )
            else:
                logger.warning(f"事实核查失败: {fact_result.error}")

        return EvalResult(
            file_id=file_id,
            topic=topic,
            quality=quality_result.to_dict(),
            repeatability=repeat_result.to_dict() if repeat_result.success else None,
            avg_repeat_score=avg_repeat_score,
            fact=fact_result.to_dict() if fact_result and fact_result.success else None,
            fact_score=fact_score,
            success=True,
        )

    async def evaluate_batch(
        self,
        input_path: str,
        output_dir: str,
        resume: bool = False,
    ) -> list[EvalResult]:
        """
        批量评估 JSONL 文件中的报告

        输入格式：每行一个 JSON 对象，包含 id、topic 和 report 字段

        Args:
            input_path: 输入 JSONL 文件路径
            output_dir: 输出目录
            resume: 是否从断点恢复

        Returns:
            评估结果列表
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 读取输入数据
        all_data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_data.append(json.loads(line))

        # 断点续传：加载已处理的文件 ID
        processed_ids = set()
        if resume:
            processed_ids = self._load_processed_ids(output_dir)
            logger.info(f"断点续传: 已完成 {len(processed_ids)}/{len(all_data)} 篇")

        results = []
        for i, data in enumerate(all_data):
            topic = data.get("topic", "")
            report = data.get("report", "")
            record_id = str(data.get("id", ""))

            if record_id in processed_ids:
                logger.debug(f"跳过已处理: id={record_id}")
                continue

            logger.info(f"处理 {i + 1}/{len(all_data)}: id={record_id}")

            result = await self.evaluate_one(topic, report, record_id=record_id)
            results.append(result)

            # 每评估一篇就保存结果
            if result.success:
                self._save_result(result, output_dir)

        logger.info(f"批量评估完成: {len(results)} 篇")
        return results

    async def evaluate_file(self, file_path: str) -> EvalResult:
        """
        评估单个 Markdown 报告文件

        从文件名或文件内容推断 topic，评估报告质量。

        Args:
            file_path: 报告文件路径（Markdown 格式）

        Returns:
            EvalResult
        """
        file_path = Path(file_path)
        report = file_path.read_text(encoding="utf-8")

        # 优先从报告头部提取"研究问题："作为 topic
        topic = ""
        research_question_match = re.search(r"研究问题[：:]\s*(.+)", report)
        if research_question_match:
            topic = research_question_match.group(1).strip()
        # 回退：从第一个 # 标题提取
        if not topic:
            lines = report.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("# "):
                    topic = line[2:].strip()
                    break
        # 最终回退：使用文件名
        if not topic:
            topic = file_path.stem

        return await self.evaluate_one(topic, report)

    @staticmethod
    def _save_result(result: EvalResult, output_dir: Path) -> None:
        """保存单篇评估结果"""
        output_file = output_dir / f"{result.file_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.debug(f"结果已保存: {output_file}")

    @staticmethod
    def _load_processed_ids(output_dir: Path) -> set[str]:
        """从输出目录中加载已处理的文件 ID"""
        processed = set()
        for json_file in output_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("success") and data.get("file_id"):
                        processed.add(data["file_id"])
            except Exception:
                # 文件损坏则跳过，不当作已处理
                continue
        return processed
