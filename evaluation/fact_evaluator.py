"""
事实核查评估器

对研究报告中的引用句子进行事实核查，通过 Jina API 抓取引用网页内容，
使用 LLM 判断每个句子是否被网页内容支持。
参考 DeepResearch-Eval-main/Atools.py 中的 check_factual 逻辑。

输入格式 (JSONL): {"https://example.com/page": {"contexts": ["句子A", "句子B"]}}
输出格式 (JSONL): {"url": "https://example.com/page", "context": "句子A", "label": {"is_factual": 1, "sentence_support": "..."}}
评分: is_factual 取值 -1（不支持）/ 0（不确定）/ 1（支持）
"""

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import json_repair
import requests

from llm.base_llm import BaseLLM
from evaluation.prompts import FACT_CHECK_SYSTEM_PROMPT, FACT_CHECK_USER_PROMPT

logger = logging.getLogger(__name__)


# ==================== Jina 网页抓取工具 ====================


class JinaScraper:
    """使用 Jina Reader API 抓取网页内容"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY 未设置！请设置 JINA_API_KEY 环境变量或在初始化时传入 api_key"
            )

    def scrape(self, url: str) -> Optional[str]:
        """
        使用 Jina Reader API 抓取网页的 Markdown 内容

        Args:
            url: 目标网页 URL

        Returns:
            网页的 Markdown 内容，抓取失败时返回 None
        """
        try:
            jina_url = f"https://r.jina.ai/{url}"
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "X-Timeout": "60000",
                "X-With-Generated-Alt": "true",
            }
            response = requests.get(jina_url, headers=headers, timeout=60)

            if response.status_code != 200:
                logger.warning(f"Jina 抓取失败 {url}: HTTP {response.status_code}")
                return None

            response_dict = response.json()
            content = response_dict.get("data", {}).get("content", "")
            if not content:
                logger.warning(f"Jina 抓取内容为空: {url}")
                return None
            return content

        except requests.Timeout:
            logger.warning(f"Jina 抓取超时: {url}")
            return None
        except Exception as e:
            logger.warning(f"Jina 抓取异常 {url}: {e}")
            return None


# ==================== 数据类 ====================


@dataclass
class FactCheckLabel:
    """单个句子的事实核查标签"""

    is_factual: int = 0  # -1: 不支持, 0: 不确定, 1: 支持
    sentence_support: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FactCheckItem:
    """单个句子的事实核查结果"""

    url: str = ""
    context: str = ""
    label: Optional[dict] = None
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FactEvalResult:
    """整篇报告的事实核查评估结果"""

    total_checks: int = 0
    success_checks: int = 0
    supported: int = 0  # is_factual == 1
    uncertain: int = 0  # is_factual == 0
    not_supported: int = 0  # is_factual == -1
    fact_score: Optional[float] = None  # 支持率 = supported / success_checks
    items: Optional[list] = None
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ==================== 事实核查评估器 ====================


class FactEvaluator:
    """
    事实核查评估器

    工作流程：
    1. 从报告或 JSONL 输入中提取引用 URL 和对应的句子（contexts）
    2. 使用 Jina API 抓取每个 URL 的网页内容
    3. 使用 LLM 判断每个句子是否被网页内容支持
    4. 输出事实核查结果（is_factual: -1/0/1）

    优化策略：
    - 区分正文和参考文献区域，只从正文提取引用句子
    - 同一句子只核查一次（合并其引用的所有网页内容）
    - 支持最大核查句子数限制（超出时随机采样）
    """

    def __init__(
        self,
        llm: BaseLLM,
        jina_api_key: Optional[str] = None,
        max_attempts: int = 3,
        retry_delay: float = 2.0,
        scrape_timeout: int = 60,
        max_check_sentences: int = 5,
    ):
        """
        Args:
            llm: LLM 实例（通过 LLMFactory 创建）
            jina_api_key: Jina API 密钥，为 None 则从环境变量 JINA_API_KEY 读取
            max_attempts: 每次核查的最大重试次数
            retry_delay: 重试间隔（秒）
            scrape_timeout: 网页抓取超时时间（秒）
            max_check_sentences: 最大核查句子数，超出时随机采样（0 表示不限制）
        """
        self.llm = llm
        self.scraper = JinaScraper(api_key=jina_api_key)
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.scrape_timeout = scrape_timeout
        self.max_check_sentences = max_check_sentences

    # ==================== 核心方法：从报告提取并核查 ====================

    async def evaluate(self, report: str) -> FactEvalResult:
        """
        评估整篇报告的事实性

        从报告中提取引用 URL 和对应的句子，逐条进行事实核查。
        同一句子只核查一次（合并其引用的所有网页内容）。

        Args:
            report: 报告内容（Markdown 格式）

        Returns:
            FactEvalResult 包含事实核查的汇总结果
        """
        # 1. 从报告中提取句子 -> URL 列表映射
        sentence_urls = self._extract_citations_from_report(report)
        if not sentence_urls:
            return FactEvalResult(
                success=False,
                error="报告中未找到可核查的引用句子",
            )

        logger.info(
            f"提取到 {len(sentence_urls)} 个待核查句子，"
            f"涉及 {len(set(u for urls in sentence_urls.values() for u in urls))} 个 URL"
        )

        # 2. 采样限制
        if self.max_check_sentences > 0 and len(sentence_urls) > self.max_check_sentences:
            logger.info(
                f"待核查句子数 ({len(sentence_urls)}) 超过限制 ({self.max_check_sentences})，"
                f"随机采样"
            )
            sampled_keys = random.sample(list(sentence_urls.keys()), self.max_check_sentences)
            sentence_urls = {k: sentence_urls[k] for k in sampled_keys}

        # 3. 按句子核查
        return await self._evaluate_sentences(sentence_urls)

    # ==================== JSONL 批量处理 ====================

    async def evaluate_jsonl(self, input_path: str, output_path: str) -> list[FactCheckItem]:
        """
        从 JSONL 文件读取输入，进行事实核查，输出结果到 JSONL 文件

        输入格式 (JSONL): {"https://example.com/page": {"contexts": ["句子A", "句子B"]}}
        输出格式 (JSONL): {"url": "...", "context": "句子A", "label": {"is_factual": 1, "sentence_support": "..."}}

        Args:
            input_path: 输入 JSONL 文件路径
            output_path: 输出 JSONL 文件路径

        Returns:
            所有核查结果列表
        """
        all_items = []

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line_no, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {line_no} 行 JSON 解析失败: {e}")
                    continue

                # 解析为 {url: [context1, context2, ...]}
                url_contexts = self._parse_jsonl_record(obj)
                if not url_contexts:
                    logger.warning(f"第 {line_no} 行格式不正确，跳过")
                    continue

                # 逐 URL 处理
                for url, contexts in url_contexts.items():
                    url = self._normalize_url(url)

                    # 抓取网页
                    page_content = self.scraper.scrape(url)
                    if page_content is None:
                        page_content = "__SCRAPE_ERROR__: 无法抓取网页内容"

                    # 逐句核查
                    for context in contexts:
                        if not isinstance(context, str):
                            continue

                        item = await self._check_factual(context, page_content)
                        item.url = url
                        item.context = context

                        all_items.append(item)

                        # 写入输出文件
                        output_record = {
                            "url": url,
                            "context": context,
                            "label": item.label if item.success else {"is_factual": 0, "sentence_support": f"核查失败: {item.error}"},
                        }
                        fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")

        logger.info(f"事实核查完成: 共 {len(all_items)} 条结果，保存至 {output_path}")
        return all_items

    # ==================== 内部方法 ====================

    async def _evaluate_sentences(
        self, sentence_urls: dict[str, list[str]]
    ) -> FactEvalResult:
        """
        按句子进行事实核查，同一句子合并所有引用网页内容后只调用一次 LLM

        Args:
            sentence_urls: {sentence: [url1, url2, ...]} 每个句子引用的 URL 列表

        Returns:
            FactEvalResult
        """
        # 1. 收集所有需要抓取的 URL，去重
        all_urls = set()
        for urls in sentence_urls.values():
            all_urls.update(urls)

        # 2. 批量抓取网页内容
        url_contents: dict[str, Optional[str]] = {}
        for url in all_urls:
            logger.info(f"正在抓取: {url}")
            content = self.scraper.scrape(url)
            url_contents[url] = content
            if content is None:
                logger.warning(f"网页抓取失败: {url}")

        # 3. 逐句核查
        items = []
        supported = 0
        uncertain = 0
        not_supported = 0
        success_checks = 0

        for sentence, urls in sentence_urls.items():
            # 合并该句子引用的所有网页内容
            combined_content_parts = []
            for url in urls:
                content = url_contents.get(url)
                if content:
                    combined_content_parts.append(f"--- 来源: {url} ---\n{content}")
                else:
                    combined_content_parts.append(f"--- 来源: {url} ---\n__SCRAPE_ERROR__: 无法抓取网页内容")

            combined_content = "\n\n".join(combined_content_parts)

            item = await self._check_factual(sentence, combined_content)
            # 记录该句子引用的所有 URL
            item.url = ", ".join(urls)
            item.context = sentence
            items.append(item.to_dict())

            if item.success and item.label:
                success_checks += 1
                is_factual = item.label.get("is_factual", 0)
                if is_factual == 1:
                    supported += 1
                elif is_factual == 0:
                    uncertain += 1
                else:
                    not_supported += 1

        total = len(items)
        fact_score = round(supported / success_checks, 4) if success_checks > 0 else None

        return FactEvalResult(
            total_checks=total,
            success_checks=success_checks,
            supported=supported,
            uncertain=uncertain,
            not_supported=not_supported,
            fact_score=fact_score,
            items=items,
            success=True,
        )

    async def _check_factual(self, sentence: str, page_content: str) -> FactCheckItem:
        """
        使用 LLM 判断句子是否被网页内容支持

        Args:
            sentence: 报告中的待核查句子
            page_content: 网页内容（可能是多个来源合并的内容）

        Returns:
            FactCheckItem
        """
        user_prompt = FACT_CHECK_USER_PROMPT.format(
            url_markdown=page_content,
            input=sentence,
        )

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=FACT_CHECK_SYSTEM_PROMPT,
                )
                result = self._parse_response(response)
                if result.success:
                    return result
                else:
                    logger.debug(f"事实核查解析失败（第 {attempt} 次）: {result.error}")
            except Exception as e:
                logger.debug(f"事实核查调用失败（第 {attempt} 次）: {e}")

            if attempt < self.max_attempts:
                time.sleep(self.retry_delay)

        return FactCheckItem(
            success=False,
            error=f"事实核查在 {self.max_attempts} 次尝试后仍失败",
        )

    def _parse_response(self, response: str) -> FactCheckItem:
        """解析 LLM 返回的 JSON 事实核查结果"""
        try:
            parsed = json_repair.loads(response)
            if isinstance(parsed, list):
                parsed = parsed[-1]

            is_factual = parsed.get("is_factual")
            if is_factual is None:
                return FactCheckItem(
                    success=False,
                    error=f"无法提取 is_factual 字段: {response[:200]}",
                )

            # 确保 is_factual 为整数且在有效范围内
            try:
                is_factual = int(is_factual)
            except (ValueError, TypeError):
                is_factual = 0

            if is_factual not in (-1, 0, 1):
                is_factual = 0

            label = {
                "is_factual": is_factual,
                "sentence_support": parsed.get("sentence_support", ""),
            }

            return FactCheckItem(
                label=label,
                success=True,
            )
        except Exception as e:
            return FactCheckItem(
                success=False,
                error=f"JSON 解析失败: {e}, 原始响应: {response[:200]}",
            )

    @staticmethod
    def _normalize_url(raw_url: str) -> str:
        """
        规范化 URL，处理 Markdown 链接格式

        输入可能是 "https://a.com/x](https://a.com/x)" 或包含 Markdown 片段，
        规范化为可用的 http(s) URL。
        策略：优先取最后一个以 http(s):// 开头的 URL 片段。
        """
        candidates = re.findall(r"https?://[^\s)\]]+", raw_url)
        if not candidates:
            return raw_url.strip()
        return candidates[-1]

    @staticmethod
    def _split_report_body_and_refs(report: str) -> tuple[str, str]:
        """
        将报告分割为正文和参考文献两部分

        以"参考文献"标题为界，之前为正文，之后为参考文献。
        支持多种标题格式：## 参考文献、# 参考文献、**参考文献** 等。

        Returns:
            (body, references) 正文内容和参考文献内容
        """
        # 匹配参考文献标题：## 参考文献、# 参考文献、**参考文献** 等
        ref_section_pattern = r"(?:^|\n)(?:#{1,3}\s*参考文献|参考文献\s*[:：]?\s*\n|\*\*参考文献\*\*)"
        match = re.search(ref_section_pattern, report)
        if match:
            body = report[:match.start()]
            references = report[match.start():]
            return body, references
        return report, ""

    @staticmethod
    def _extract_citations_from_report(report: str) -> dict[str, list[str]]:
        """
        从报告中提取引用 URL 和对应的句子

        返回格式改为：{sentence: [url1, url2, ...]}
        同一句子只出现一次，关联其引用的所有 URL。

        支持两种引用格式：
        1. 新格式：正文中 [网页N]、[论文N]、[网页N-M]、[网页N, M, K]
           参考文献列表：[网页8] 标题 链接: `https://...`
        2. 旧格式：正文中 [[N]]、[[N,M]]
           参考文献列表：[N] https://...
        """
        # 优先尝试新格式
        result = FactEvaluator._extract_citations_new_format(report)
        if result:
            return result

        # 回退到旧格式
        return FactEvaluator._extract_citations_old_format(report)

    @staticmethod
    def _extract_citations_new_format(report: str) -> dict[str, list[str]]:
        """
        新格式提取：[网页N]、[论文N] 等

        正文引用格式：[网页11]、[论文8]、[网页26-27]、[网页1, 5, 16, 20]
        参考文献格式：[网页8] 什么是NLP？ 链接: `https://...`

        Returns:
            {sentence: [url1, url2, ...]} 同一句子只出现一次
        """
        # 1. 分割正文和参考文献
        body, references = FactEvaluator._split_report_body_and_refs(report)

        # 2. 从参考文献区域提取 URL 映射：{ref_key: url}
        ref_pattern = r"\[(网页|论文)(\d+)\].*?链接:\s*`?(https?://[^`\s]+)`?"
        ref_matches = re.findall(ref_pattern, references)

        # 如果参考文献区域没找到，尝试从整篇报告提取
        if not ref_matches:
            ref_matches = re.findall(ref_pattern, report)

        if not ref_matches:
            return {}

        ref_map: dict[str, str] = {}
        for ref_type, ref_num, url in ref_matches:
            key = f"{ref_type}{ref_num}"
            ref_map[key] = url

        # 3. 只从正文区域提取带引用标记的句子
        citation_in_text = r"\[(?:网页|论文)\d+(?:\s*[-,]\s*\d+)*\]"
        sentence_pattern = rf"[^。.!?]*{citation_in_text}[^。.!?]*[。.!?]"
        sentences = re.findall(sentence_pattern, body)

        if not sentences:
            logger.debug("报告正文中未找到新格式引用标记的句子")
            return {}

        # 4. 建立 sentence -> [url1, url2, ...] 映射（去重）
        sentence_urls: dict[str, list[str]] = {}
        for sentence in sentences:
            citation_marks = re.findall(
                r"\[(网页|论文)((?:\d+(?:\s*[-,]\s*\d+)*))\]", sentence
            )
            urls_for_sentence = []
            for ref_type, nums_str in citation_marks:
                nums = FactEvaluator._parse_citation_nums(nums_str)
                for num in nums:
                    key = f"{ref_type}{num}"
                    url = ref_map.get(key)
                    if url:
                        url = FactEvaluator._normalize_url(url)
                        if url not in urls_for_sentence:
                            urls_for_sentence.append(url)

            if urls_for_sentence:
                # 去重：如果两个句子内容相同，合并 URL
                if sentence not in sentence_urls:
                    sentence_urls[sentence] = urls_for_sentence
                else:
                    for url in urls_for_sentence:
                        if url not in sentence_urls[sentence]:
                            sentence_urls[sentence].append(url)

        return sentence_urls

    @staticmethod
    def _parse_citation_nums(nums_str: str) -> list[str]:
        """
        解析引用编号字符串

        支持格式：
        - "1" -> ["1"]
        - "1, 5, 16, 20" -> ["1", "5", "16", "20"]
        - "26-27" -> ["26", "27"]
        - "1, 3-5" -> ["1", "3", "4", "5"]
        """
        result = []
        parts = re.split(r"\s*,\s*", nums_str)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            range_match = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                for i in range(start, end + 1):
                    result.append(str(i))
            else:
                num_match = re.match(r"^(\d+)$", part)
                if num_match:
                    result.append(num_match.group(1))
        return result

    @staticmethod
    def _extract_citations_old_format(report: str) -> dict[str, list[str]]:
        """
        旧格式提取：[[N]]、[[N,M]]

        正文引用格式：[[1]]、[[2,3]]
        参考文献格式：[1] https://example.com/page1

        Returns:
            {sentence: [url1, url2, ...]} 同一句子只出现一次
        """
        # 1. 分割正文和参考文献
        body, references = FactEvaluator._split_report_body_and_refs(report)

        # 2. 从参考文献区域提取 URL 映射
        ref_pattern = r"\[(\d+)\]\s*(https?://\S+)"
        ref_matches = re.findall(ref_pattern, references)

        # 如果参考文献区域没找到，尝试从整篇报告提取
        if not ref_matches:
            ref_matches = re.findall(ref_pattern, report)

        ref_map = {num: url for num, url in ref_matches}

        if not ref_map:
            logger.debug("报告中未找到参考来源引用")
            return {}

        # 3. 只从正文区域提取带引用编号的句子
        sentence_pattern = r"[^。.!?]*\[\[\d+(?:,\d+)*\]\][^。.!?]*[。.!?]"
        sentences = re.findall(sentence_pattern, body)

        if not sentences:
            logger.debug("报告正文中未找到旧格式引用标记的句子")
            return {}

        # 4. 建立 sentence -> [url1, url2, ...] 映射（去重）
        sentence_urls: dict[str, list[str]] = {}
        for sentence in sentences:
            citation_nums = re.findall(r"\[\[(\d+(?:,\d+)*)\]\]", sentence)
            urls_for_sentence = []
            for num_str in citation_nums:
                for num in num_str.split(","):
                    num = num.strip()
                    url = ref_map.get(num)
                    if url:
                        url = FactEvaluator._normalize_url(url)
                        if url not in urls_for_sentence:
                            urls_for_sentence.append(url)

            if urls_for_sentence:
                if sentence not in sentence_urls:
                    sentence_urls[sentence] = urls_for_sentence
                else:
                    for url in urls_for_sentence:
                        if url not in sentence_urls[sentence]:
                            sentence_urls[sentence].append(url)

        return sentence_urls

    @staticmethod
    def _parse_jsonl_record(obj: dict) -> dict[str, list[str]]:
        """
        解析 JSONL 输入记录

        输入格式: {"https://example.com/page": {"contexts": ["句子A", "句子B"]}}
        返回: {"https://example.com/page": ["句子A", "句子B"]}
        """
        if not isinstance(obj, dict):
            return {}

        result = {}
        for url, payload in obj.items():
            if isinstance(payload, dict):
                contexts = payload.get("contexts", [])
            elif isinstance(payload, list):
                contexts = payload
            else:
                continue

            if isinstance(contexts, list) and all(isinstance(c, str) for c in contexts):
                result[url] = contexts

        return result
