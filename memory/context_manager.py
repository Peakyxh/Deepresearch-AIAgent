"""
上下文管理器

管理 LLM 的上下文窗口，实现语义压缩和关键信息保留。
当文本超过最大 token 限制时，使用 LLM 进行语义压缩而非简单截断。

核心能力：
1. 文本语义压缩：用 LLM 将长文本压缩为保留关键信息的短文本
2. 关键信息保护：标记为关键的内容享有最高保留优先级（不丢弃、不摘要）
3. 多场景支持：搜索结果压缩、研究发现压缩、审查反馈传递
4. 渐进式压缩：根据上下文使用率分级别触发不同压缩策略
5. 动态 token 限制：根据 LLM 模型自动调整上下文窗口大小

关于 is_key 的语义（重要）：
- is_key=True 表示"小而不可再生的关键事实"（如用户意图澄清、研究计划），
  它在轻量压缩时不去重、不修剪，在深度/紧急压缩时不丢弃、不摘要，
  并且会通过 get_key_info() 注入到下游 Agent（Critic / Writer）的 prompt 中。
- is_key 是**保留优先级**，不是"绝对不可压缩"。体积大、可由 state 重新获取或
  可以由 LLM 重新摘要的内容（如研究发现）不应标记为关键信息，否则压缩将无事可做。
- 兜底保证：当分级压缩后仍然超出 max_tokens 时，_force_shrink() 会把关键信息
  一并降级为摘要或截断，确保上下文永不超出模型窗口。
"""

import hashlib
import logging
import re
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3


class ContextManager:
    """
    上下文管理器

    负责：
    - 管理研究过程中的上下文信息
    - 当文本过长时进行语义压缩（用 LLM 生成摘要，而非简单截断）
    - 保留关键信息，压缩冗余内容
    - 在 Agent 之间传递压缩后的上下文
    - 渐进式压缩：轻量压缩 → 深度压缩 → 紧急压缩

    使用场景：
    1. Researcher：搜索结果过长时压缩后再传给 LLM 整合
    2. Critic 回退：将审查反馈压缩后传给下一轮 Researcher
    3. Writer：搜索结果 + 论文结果 + 研究发现过长时压缩
    """

    def __init__(self, max_tokens: int | None = None, llm: Any = None):
        self.max_tokens = max_tokens or settings.effective_context_max_tokens
        self._llm = llm
        self.context: list[dict[str, Any]] = []
        self._key_info: list[str] = []
        self.token_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

        self._compress_threshold_search = settings.context_compress_threshold_search
        self._compress_threshold_paper = settings.context_compress_threshold_paper
        self._compress_threshold_findings = settings.context_compress_threshold_findings
        self._compress_threshold_prior = settings.context_compress_threshold_prior
        self._compress_threshold_sources = settings.context_compress_threshold_sources
        self._warning_ratio = settings.context_warning_ratio
        self._dialog_compress_ratio = settings.context_dialog_compress_ratio
        self._recent_message_count = settings.context_recent_message_count

        logger.info(f"ContextManager 初始化，最大 token: {self.max_tokens}")

    def set_llm(self, llm: Any) -> None:
        self._llm = llm

    async def _generate(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        """Generate compression text and retain an approximate usage total."""
        response = await self._llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        self.token_usage["calls"] += 1
        self.token_usage["input_tokens"] += max(1, len(f"{system_prompt}\n{prompt}") // 3)
        self.token_usage["output_tokens"] += max(1, len(response or "") // 3)
        return response

    def add_context(self, role: str, content: str, is_key: bool = False, query: str = "") -> None:
        """
        添加上下文信息

        Args:
            role: 角色。跨阶段注入的事实性内容（意图澄清、研究计划、研究发现）
                  应使用 "system"，避免下游消费时被误认为模型自己先前的断言。
            content: 内容
            is_key: 是否为关键信息。关键信息享有最高保留优先级：
                    轻量压缩时不去重、不修剪；深度/紧急压缩时不丢弃、不摘要；
                    并会通过 get_key_info() 注入下游 Agent 的 prompt。
                    请只把"小而不可再生"的事实标记为关键信息，大块可重新摘取的内容
                    （如研究发现）应保持 is_key=False，否则压缩将无事可做。
            query: 关联的研究问题（交互模式下区分不同问题的上下文）

        注意：关键信息不会再额外复制到 _key_info，get_key_info() 直接从
        self.context 派生，避免同一份内容在内存与 prompt 中存在两份。
        """
        self.context.append({
            "role": role,
            "content": content,
            "is_key": is_key,
            "query": query,
        })

    def add_key_info(self, info: str) -> None:
        """添加一条额外的关键事实（不经由 context 列表，直接注入下游 prompt）"""
        self._key_info.append(info)
        logger.info(f"添加关键信息: {info[:50]}...")

    def get_context(self) -> list[dict[str, Any]]:
        return [
            {"role": item["role"], "content": item["content"]}
            for item in self.context
        ]

    def get_key_info(self) -> str:
        """
        获取关键信息文本，供下游 Agent 注入 prompt

        由两部分组成：
        1. add_key_info() 显式添加的短事实
        2. self.context 中 is_key=True 的条目（意图澄清、研究计划等）
        """
        parts = list(self._key_info)
        parts.extend(
            item["content"] for item in self.context if item.get("is_key", False)
        )
        return "\n".join(parts)

    @staticmethod
    def estimate_text_tokens(text: str) -> int:
        return len(text) // CHARS_PER_TOKEN

    @staticmethod
    def _truncate_text_to_token_limit(text: str, limit: int, marker: str) -> str:
        """按当前的字符/token 估算将文本严格限制在 limit 内。"""
        max_chars = max(0, limit * CHARS_PER_TOKEN)
        if len(text) <= max_chars:
            return text
        if max_chars <= len(marker):
            return marker[:max_chars]
        return text[: max_chars - len(marker)] + marker

    async def get_rounds_findings(self) -> str:
        """
        提取 self.context 中累积的各轮研究发现，供重试轮次的 Researcher 使用

        这是 self.context 中非关键条目（研究发现）的真正消费者：
        Critic 审查未通过回退重试时，重试的 sub-agent 只有审查反馈而看不到
        上一轮已有的发现，容易从零重做并丢失仍然有效的结论。本方法把
        上下文中历轮"研究发现: "条目取出，超预算时压缩为摘要，注入重试 prompt。

        预算为 get_compress_threshold("prior")（默认 25% max_tokens），
        因为该文本会注入每个重试 sub-agent 的 prompt，需为搜索结果留出空间。

        注意：若深度/紧急压缩已把研究发现合并进"之前对话的摘要"，
        将提取不到独立条目，此时返回空字符串（退化为不注入，不影响现有行为）。

        Returns:
            可直接注入 prompt 的历轮研究发现文本；无研究发现时返回空字符串
        """
        findings_items = [
            item["content"]
            for item in self.context
            if item.get("role") == "system" and item["content"].startswith("研究发现")
        ]
        if not findings_items:
            return ""

        text = "\n\n".join(findings_items)
        budget = self.get_compress_threshold("prior")
        if self.estimate_text_tokens(text) <= budget:
            return text

        logger.info(
            f"历轮研究发现超预算 ({self.estimate_text_tokens(text)}/{budget} tokens)，压缩后注入重试"
        )
        compressed = await self.compress_findings(text, threshold_tokens=budget)
        if not compressed:
            return ""
        return f"（以下为压缩后的历轮研究发现摘要）\n{compressed}"

    def estimate_tokens(self) -> int:
        total_chars = sum(len(item["content"]) for item in self.context)
        return total_chars // CHARS_PER_TOKEN

    def get_compress_threshold(self, text_type: str) -> int:
        ratios = {
            "search": self._compress_threshold_search,
            "paper": self._compress_threshold_paper,
            "findings": self._compress_threshold_findings,
            "prior": self._compress_threshold_prior,
            "sources": self._compress_threshold_sources,
        }
        ratio = ratios.get(text_type, 0.5)
        return int(self.max_tokens * ratio)

    # ==================== 渐进式压缩 ====================

    async def maybe_compress(self) -> list[dict[str, Any]]:
        """
        检查上下文是否需要压缩，根据使用率分级别触发不同策略

        压缩级别：
        - ≤ warning_ratio (60%)：正常运行，无需压缩
        - ≤ dialog_compress_ratio (80%)：轻量压缩（去重+修剪，不调用 LLM）
        - ≤ 100%：深度压缩（LLM 语义压缩旧对话）
        - > 100%：紧急压缩（激进策略，保留最少信息）

        无论走到哪一级，返回前都会经过 _force_shrink() 兜底，
        保证返回的上下文一定不超过 max_tokens（即使所有条目都被标记为 is_key）。

        Returns:
            压缩后的上下文列表
        """
        current_tokens = self.estimate_tokens()
        if current_tokens == 0:
            return self.get_context()

        ratio = current_tokens / self.max_tokens

        if ratio <= self._warning_ratio:
            return self.get_context()

        if ratio <= self._dialog_compress_ratio:
            logger.info(f"上下文 {ratio:.0%}，执行轻量压缩（去重+修剪）")
            self._lightweight_compress()
        elif ratio <= 1.0:
            logger.info(f"上下文 {ratio:.0%}，执行深度压缩（LLM语义压缩）")
            await self.compress_context()
        else:
            logger.warning(f"上下文 {ratio:.0%}，执行紧急压缩！")
            await self._emergency_compress()

        # 最终保障：is_key 只是保留优先级，不能成为超出模型窗口的借口
        await self._force_shrink()
        return self.get_context()

    # ---- 抽取式摘要的打分规则权重与匹配规则 ----
    _EXTRACT_BONUS_STRUCTURE = 3      # 标题/列表/编号等结构行（骨架信息）
    _EXTRACT_BONUS_CONCLUSION = 2     # 含结论标志词
    _EXTRACT_BONUS_NUMERIC = 2        # 含数据（数字/百分比/年份）
    _EXTRACT_BONUS_FIRST = 2          # 首句（条目标识信息）
    _EXTRACT_BONUS_LAST = 1           # 末句（总结常在尾部）
    _EXTRACT_PENALTY_LENGTH = 1       # 过短或过长降权
    _EXTRACT_MARKER_RESERVE = 40      # 省略标记的字符预留
    _STRUCTURE_PREFIXES = ("#", "-", "*", "+", "•")
    _RE_NUMBERED_LINE = re.compile(r"^\d+[.、)）]")
    _RE_NUMBER = re.compile(r"\d+(?:\.\d+)?%?|\d{4}年")
    _CONCLUSION_MARKERS = (
        "因此", "综上", "结论", "发现", "表明", "结果显示", "研究表明",
        "总之", "核心", "关键", "总结", "意味着", "证明",
        "conclusion", "summary", "therefore", "finding", "indicates",
        "suggests", "in short", "key",
    )

    def _lightweight_compress(self) -> None:
        """
        轻量压缩：去重和抽取式摘要，不调用 LLM

        策略：
        1. 关键信息（is_key=True）原样保留：不参与去重、不做长度修剪
        2. 非关键消息按完整内容 hash 去重（保留首次出现）——避免前缀碰撞误删，
           例如重试轮的研究发现与上一轮开头相同但含修正内容，前缀去重会
           错误保留旧条目、丢弃修正后的新条目
        3. 超长的非关键消息用抽取式摘要压缩（切句打分选句，保留原文措辞，
           不会像硬截断那样丢失尾部结论）
        4. 优先保留当前 query 的上下文（非当前 query 的目标长度更小）

        注意：关键信息不做长度修剪，因此大块内容不应标记为 is_key，
        否则本条策略对它完全无效（越界部分由 _force_shrink() 兜底）。
        """
        current_query = self._get_current_query()

        seen_contents: set[str] = set()
        deduplicated: list[dict[str, Any]] = []
        max_content_chars = self.max_tokens * CHARS_PER_TOKEN // 2
        max_content_chars_other = self.max_tokens * CHARS_PER_TOKEN // 4

        for item in self.context:
            if item.get("is_key", False):
                deduplicated.append(item)
                continue

            content_key = hashlib.md5(item["content"].encode("utf-8")).hexdigest()
            if content_key in seen_contents:
                continue

            seen_contents.add(content_key)

            item_query = item.get("query", "")
            is_current_query = current_query and item_query == current_query
            item_max_chars = max_content_chars if is_current_query else max_content_chars_other

            if len(item["content"]) > item_max_chars:
                summarized = self._extractive_summarize(item["content"], item_max_chars)
                logger.info(
                    f"轻量压缩：条目 {len(item['content'])} → {len(summarized)} 字符（抽取式摘要）"
                )
                deduplicated.append(
                    {
                        "role": item["role"],
                        "content": summarized,
                        "is_key": item.get("is_key", False),
                        "query": item_query,
                    }
                )
            else:
                deduplicated.append(item)

        removed = len(self.context) - len(deduplicated)
        if removed > 0:
            logger.info(f"轻量压缩：去重 {removed} 条重复消息")
        self.context = deduplicated

    def _extractive_summarize(self, text: str, target_chars: int) -> str:
        """
        抽取式摘要（不调用 LLM）：把长文本切句、按重要性打分选句、按原顺序拼回。

        相比硬截断的优势：结论句、数据句、结构行优先保留，且保留原文措辞
        （无幻觉、无改写），信息密度显著高于按位置截断。

        Args:
            text: 待摘要文本
            target_chars: 目标字符数（含省略标记）

        Returns:
            摘要文本；无法切出句子时退化为头尾保留截断
        """
        units = self._split_sentences(text)
        if not units:
            return self._head_tail_trim(text, target_chars)

        budget = max(target_chars - self._EXTRACT_MARKER_RESERVE, 50)

        # 打分并按分数降序选句（稳定排序：同分保持原文顺序）
        scores = [self._score_sentence(u, i, len(units)) for i, u in enumerate(units)]
        order = sorted(range(len(units)), key=lambda i: -scores[i])

        kept: list[int] = []
        used = 0
        for i in order:
            if used + len(units[i]) <= budget:
                kept.append(i)
                used += len(units[i])

        if not kept:
            return self._head_tail_trim(text, target_chars)

        # 按原文顺序重组，缺口处插入省略标记
        kept_sorted = sorted(kept)
        parts: list[str] = []
        prev = -1
        for i in kept_sorted:
            if prev >= 0 and i > prev + 1:
                parts.append("[...中间部分已省略...]")
            parts.append(units[i])
            prev = i
        if kept_sorted[0] > 0:
            parts.insert(0, "[...开头部分已省略...]")
        if kept_sorted[-1] < len(units) - 1:
            parts.append("[...后续部分已省略...]")

        return "\n".join(parts)

    def _score_sentence(self, text: str, index: int, total: int) -> int:
        """单句重要性打分：结构行、结论词、数据、首尾位置加分，过短过长降权"""
        score = 0
        lowered = text.lower()
        if text.startswith(self._STRUCTURE_PREFIXES) or self._RE_NUMBERED_LINE.match(text):
            score += self._EXTRACT_BONUS_STRUCTURE
        if any(marker in lowered for marker in self._CONCLUSION_MARKERS):
            score += self._EXTRACT_BONUS_CONCLUSION
        if self._RE_NUMBER.search(text):
            score += self._EXTRACT_BONUS_NUMERIC
        if index == 0:
            score += self._EXTRACT_BONUS_FIRST
        if index == total - 1:
            score += self._EXTRACT_BONUS_LAST
        if len(text) < 10 or len(text) > 300:
            score -= self._EXTRACT_PENALTY_LENGTH
        return score

    @classmethod
    def _split_sentences(cls, text: str) -> list[str]:
        """
        把文本切分为句子/原子单元。

        markdown 结构行（标题、列表、编号行）作为原子单元不拆分；
        普通文本按中英文句号/问号/叹号/分号切分，保留句尾标点。
        """
        units: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(cls._STRUCTURE_PREFIXES) or cls._RE_NUMBERED_LINE.match(stripped):
                units.append(stripped)
                continue
            # 按句尾标点切分（lookbehind 保留标点）
            for part in re.split(r"(?<=[。！？!?；;])", stripped):
                part = part.strip()
                if part:
                    units.append(part)
        return units

    @staticmethod
    def _head_tail_trim(text: str, target_chars: int) -> str:
        """头尾保留截断：抽取式摘要的保底策略（无法切句时使用）"""
        if len(text) <= target_chars:
            return text
        marker = "\n[...中间内容过长已省略...]\n"
        head = (target_chars - len(marker)) * 2 // 3
        tail = target_chars - len(marker) - head
        if head <= 0 or tail <= 0:
            return text[:target_chars]
        return text[:head] + marker + text[-tail:]

    @staticmethod
    def _classify_messages(
        context: list[dict[str, Any]], recent_count: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """
        按保留优先级把上下文分为三类

        分类规则（is_key 是唯一的保护开关，role 仅作描述性元数据）：
        - protected：is_key=True 的关键信息，不丢弃、不摘要
        - recent：最近 recent_count 条消息，不丢弃、不摘要
        - old：其余消息，可被摘要或丢弃

        Args:
            context: 上下文列表
            recent_count: 需要保留的最近消息条数

        Returns:
            (protected, recent, old) 三元组
        """
        protected: list[dict[str, Any]] = []
        recent: list[dict[str, Any]] = []
        old: list[dict[str, Any]] = []

        total = len(context)
        for i, item in enumerate(context):
            if item.get("is_key", False):
                protected.append(item)
            elif i >= total - recent_count:
                recent.append(item)
            else:
                old.append(item)

        return protected, recent, old

    async def _emergency_compress(self) -> list[dict[str, Any]]:
        """
        紧急压缩：比深度压缩更激进的策略

        保留：关键信息 + 最近若干条消息
        其余全部生成紧急摘要
        """
        recent_count = max(2, self._recent_message_count // 2)
        protected_messages, recent_messages, old_messages = self._classify_messages(
            self.context, recent_count
        )

        summary = ""
        if old_messages:
            if self._llm:
                summary = await self._summarize(old_messages)
            else:
                logger.warning("未设置 LLM，无法进行语义压缩，丢弃早期对话")

        if summary:
            compressed = [
                *protected_messages,
                {"role": "system", "content": f"之前对话的紧急摘要：{summary}", "is_key": False, "query": ""},
                *recent_messages,
            ]
        else:
            compressed = [*protected_messages, *recent_messages]

        old_tokens = self.estimate_tokens()
        self.context = compressed
        new_tokens = self.estimate_tokens()
        logger.warning(f"紧急压缩完成: {old_tokens} → {new_tokens} tokens")
        return self.get_context()

    # ==================== 语义压缩核心方法 ====================

    async def compress_text(
        self, text: str, instruction: str = "", threshold_tokens: int | None = None
    ) -> str:
        """
        语义压缩文本

        Args:
            text: 待压缩文本
            instruction: 压缩指令（为空时使用默认指令）
            threshold_tokens: 触发压缩的 token 阈值。默认 None 时按 max_tokens 判断；
                            调用方已按更小的阈值（如 get_compress_threshold）预判时，
                            应显式传入该阈值，否则阈值与 max_tokens 不匹配会导致压缩被短路。
        """
        limit = threshold_tokens if threshold_tokens is not None else self.max_tokens
        text_tokens = self.estimate_text_tokens(text)

        if text_tokens <= limit:
            logger.info(f"文本未超限 ({text_tokens} tokens)，无需压缩")
            return text

        if not self._llm:
            logger.warning("未设置 LLM，无法进行语义压缩，回退到截断")
            return self._truncate_text_to_token_limit(
                text, limit, "\n\n[...内容过长，已截断...]"
            )

        logger.info(f"文本超限 ({text_tokens}/{limit} tokens)，开始语义压缩...")

        target_ratio = 0.7
        target_tokens = int(limit * target_ratio)

        default_instruction = (
            "请将以下内容压缩为更短的版本，要求：\n"
            "1. 保留所有关键信息、数据、结论\n"
            "2. 保留具体的数字、年份、名称等事实信息\n"
            "3. 去除重复内容和冗余描述\n"
            "4. 保持逻辑连贯性\n"
            f"5. 压缩后大约 {target_tokens} tokens\n"
            "6. 用中文输出压缩结果"
        )
        compress_instruction = instruction or default_instruction

        try:
            compressed = await self._generate(
                prompt=f"{compress_instruction}\n\n---\n\n{text}",
                system_prompt="你是一个文本压缩助手，你的任务是在保留所有关键信息的前提下，将长文本压缩为更短的版本。不要丢失任何重要的事实、数据或结论。",
                max_tokens=min(target_tokens * 2, settings.llm_max_tokens),
            )

            compressed_tokens = self.estimate_text_tokens(compressed)
            logger.info(f"语义压缩完成: {text_tokens} → {compressed_tokens} tokens "
                        f"(压缩率: {(1 - compressed_tokens / text_tokens) * 100:.1f}%)")

            if compressed_tokens > limit and self._llm:
                logger.warning(f"压缩后仍超限 ({compressed_tokens}/{limit})，进行二次压缩")
                compressed = await self._generate(
                    prompt=(
                        f"以下文本仍然过长（{compressed_tokens} tokens），请进一步压缩到约 {target_tokens} tokens，"
                        f"保留最核心的信息：\n\n{compressed}"
                    ),
                    system_prompt="你是一个文本压缩助手，请极致压缩但保留核心信息。",
                    max_tokens=min(target_tokens, settings.llm_max_tokens),
                )
                compressed_tokens = self.estimate_text_tokens(compressed)
                logger.info(f"二次压缩完成: {compressed_tokens} tokens")

            # LLM 输出长度不是硬保证；二次压缩后仍超限时必须有界回退。
            if self.estimate_text_tokens(compressed) > limit:
                logger.warning("二次压缩后仍超限，按本次业务阈值截断")
                compressed = self._truncate_text_to_token_limit(
                    compressed,
                    limit,
                    "\n\n[...压缩结果仍过长，已截断...]",
                )

            return compressed

        except Exception as e:
            logger.error(f"语义压缩失败: {e}，回退到截断")
            return self._truncate_text_to_token_limit(
                text, limit, "\n\n[...语义压缩失败，已截断...]"
            )

    async def compress_search_results(self, results_text: str) -> str:
        instruction = (
            "请将以下搜索结果压缩为更短的版本，要求：\n"
            "1. 保留每条搜索结果的标题和核心发现\n"
            "2. 保留具体的数据、数字、年份\n"
            "3. 去除重复信息（多条结果提到相同内容时合并）\n"
            "4. 去除冗余的描述性文字\n"
            "5. 原样保留搜索结果编号、证据ID和来源信息，不得改写或合并证据ID\n"
            "6. 用中文输出"
        )
        return await self.compress_text(
            results_text,
            instruction,
            threshold_tokens=self.get_compress_threshold("search"),
        )

    async def compress_paper_results(self, results_text: str) -> str:
        instruction = (
            "请将以下论文搜索结果压缩为更短的版本，要求：\n"
            "1. 保留每篇论文的标题、核心发现和关键数据\n"
            "2. 保留引用次数和年份信息\n"
            "3. 去除冗余的摘要描述\n"
            "4. 原样保留论文编号、证据ID和来源信息，不得改写或合并证据ID\n"
            "5. 用中文输出"
        )
        return await self.compress_text(
            results_text,
            instruction,
            threshold_tokens=self.get_compress_threshold("paper"),
        )

    async def compress_findings(
        self, findings_text: str, threshold_tokens: int | None = None
    ) -> str:
        instruction = (
            "请将以下研究发现压缩为更短的版本，要求：\n"
            "1. 保留所有核心发现和关键论点\n"
            "2. 保留具体的数据、数字、结论\n"
            "3. 去除冗余的论证过程和重复描述\n"
            "4. 原样保留引用编号、证据置信度和适用范围\n"
            "5. 保持逻辑结构清晰\n"
            "6. 用中文输出"
        )
        limit = (
            threshold_tokens
            if threshold_tokens is not None
            else self.get_compress_threshold("findings")
        )
        return await self.compress_text(
            findings_text, instruction, threshold_tokens=limit
        )

    async def compress_context(self) -> list[dict[str, Any]]:
        """
        压缩上下文对话历史

        当对话历史超过 token 限制时：
        1. 保留关键信息（is_key=True，不压缩）
        2. 优先压缩非当前 query 的旧对话
        3. 对早期对话进行语义压缩（生成摘要）
        4. 保留最近几轮对话（不压缩，数量可配置）

        注意：role="system" 不再自动免疫压缩，唯一保护开关是 is_key。
        这样由工作流注入的事实性 system 内容（研究发现等）才能真正参与压缩。
        """
        current_tokens = self.estimate_tokens()

        if current_tokens <= self.max_tokens:
            logger.info(f"上下文未超限 ({current_tokens} tokens)，无需压缩")
            return self.get_context()

        logger.info(f"上下文超限 ({current_tokens}/{self.max_tokens} tokens)，开始压缩...")

        current_query = self._get_current_query()

        protected_messages, recent_messages, old_messages = self._classify_messages(
            self.context, self._recent_message_count
        )

        # 非当前 query 的旧对话优先压缩
        if current_query and old_messages:
            current_query_old = [m for m in old_messages if m.get("query", "") == current_query]
            other_query_old = [m for m in old_messages if m.get("query", "") != current_query]
            if other_query_old:
                logger.info(f"优先压缩非当前 query 的旧对话: {len(other_query_old)} 条 "
                            f"(当前 query 旧对话: {len(current_query_old)} 条)")
                old_messages = other_query_old + current_query_old

        summary = ""
        if old_messages:
            if self._llm:
                summary = await self._summarize(old_messages)
            else:
                logger.warning("未设置 LLM，无法进行语义压缩，丢弃早期对话")
        elif protected_messages or recent_messages:
            logger.info("没有可压缩的旧对话（全部被 is_key 或最近消息保护），"
                        "交由 _force_shrink() 兜底")

        if summary:
            compressed = [
                *protected_messages,
                {"role": "system", "content": f"之前对话的摘要：{summary}", "is_key": False, "query": ""},
                *recent_messages,
            ]
        else:
            compressed = [*protected_messages, *recent_messages]

        self.context = compressed
        new_tokens = self.estimate_tokens()
        logger.info(f"上下文压缩完成: {current_tokens} → {new_tokens} tokens")
        return self.get_context()

    async def _force_shrink(self) -> None:
        """
        最终保障：确保上下文一定不超过 max_tokens

        is_key 只是"最高保留优先级"，不是"绝对不可压缩"。当分级压缩之后仍然超限
        （例如所有条目都被标记为关键信息），关键信息也必须降级，否则上下文会无限
        增长并最终溢出模型窗口（表现为反复进入紧急压缩分支却毫无效果）：

        1. 有 LLM 时：把所有消息（含关键信息）合并为一条强制摘要
        2. 摘要失败或仍然超限时：硬截断到预算字符数（_truncate_to_limit）
        """
        current_tokens = self.estimate_tokens()
        if current_tokens <= self.max_tokens:
            return

        logger.warning(
            f"压缩后仍超限 ({current_tokens}/{self.max_tokens} tokens)，"
            f"触发强制收缩（关键信息降级）"
        )

        if self._llm and len(self.context) > 1:
            summary = await self._summarize(self.context)
            if summary:
                self.context = [{
                    "role": "system",
                    "content": f"之前上下文的强制摘要：{summary}",
                    "is_key": False,
                    "query": self._get_current_query(),
                }]
                if self.estimate_tokens() <= self.max_tokens:
                    logger.info(f"强制收缩完成: {current_tokens} → {self.estimate_tokens()} tokens")
                    return

        self._truncate_to_limit()
        logger.info(f"硬截断完成: {current_tokens} → {self.estimate_tokens()} tokens")

    def _truncate_to_limit(self) -> None:
        """
        硬截断兜底：按顺序保留消息，超出预算的部分在字符层面截断

        预算 = max_tokens * CHARS_PER_TOKEN，截断标记本身也计入预算，
        因此返回后 estimate_tokens() 一定 ≤ max_tokens。
        """
        marker = "\n[...已强制截断...]"
        budget = max(0, self.max_tokens * CHARS_PER_TOKEN)
        kept: list[dict[str, Any]] = []
        used = 0

        for item in self.context:
            if used >= budget:
                break

            remaining = budget - used
            content = item["content"]

            if len(content) > remaining:
                if remaining > len(marker):
                    content = content[: remaining - len(marker)] + marker
                else:
                    content = content[:remaining]
                kept.append({**item, "content": content})
                break

            kept.append(item)
            used += len(content)

        if not kept and self.context:
            last = self.context[-1]
            keep = max(0, budget - len(marker))
            kept = [{**last, "content": last["content"][:keep] + marker}]

        self.context = kept

    async def _summarize(self, messages: list[dict[str, Any]]) -> str:
        """
        用 LLM 把一组消息压缩为一段摘要

        Returns:
            摘要文本；失败时返回空字符串，调用方据此保留原始内容而不是用
            失败占位符覆盖掉真实信息。
        """
        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text_parts.append(f"[{role}]: {content}")

        text = "\n".join(text_parts)

        try:
            summary = await self._generate(
                prompt=(
                    "请将以下对话内容压缩为一段简洁的摘要，要求：\n"
                    "1. 保留所有关键信息、数据和结论\n"
                    "2. 去除冗余和重复内容\n"
                    "3. 保持逻辑连贯\n"
                    f"4. 用中文输出\n\n{text}"
                ),
                system_prompt="你是一个文本摘要助手，请在保留关键信息的前提下简洁地总结内容。",
                max_tokens=2048,
            )
            return summary or ""
        except Exception as e:
            logger.error(f"语义摘要生成失败: {e}")
            return ""

    def clear(self) -> None:
        self.context = []
        self._key_info = []

    def _get_current_query(self) -> str:
        """
        获取当前研究问题的 query

        从上下文列表中找到最后一条带有 query 字段的消息，
        返回其 query 值。用于压缩时区分不同问题的上下文。
        """
        for item in reversed(self.context):
            if item.get("query", ""):
                return item["query"]
        return ""
