"""
DeepResearch Agent 配置管理模块

使用 pydantic-settings 从环境变量和 .env 文件加载配置，
所有配置项都有类型注解和默认值，确保类型安全。
"""

from pathlib import Path
from typing import Literal
import uuid

from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "deepseek-chat": 64000,
    "deepseek-reasoner": 64000,
    "deepseek-v3": 64000,
    "deepseek-v4-flash": 64000,
    "meta-llama/Llama-3-8B": 8192,
    "qwen3.6-flash": 131072,
}


class Settings(BaseSettings):
    """
    全局配置类

    通过 pydantic-settings 自动从 .env 文件和环境变量加载配置。
    所有字段都有类型注解，pydantic 会自动进行类型校验。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==================== LLM 配置 ====================

    # 当前使用的 LLM 提供者：openai / deepseek / vllm / qwen
    llm_provider: Literal["openai", "deepseek", "vllm", "qwen"] = "deepseek"

    # OpenAI 配置
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"

    # DeepSeek 配置
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # vLLM 配置（本地部署，OpenAI 兼容接口）
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "meta-llama/Llama-3-8B"
    vllm_api_key: str = "EMPTY"

    # Qwen 配置（通义千问，DashScope OpenAI 兼容接口）
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen3.6-flash"

    # Critic Agent 专用 LLM 提供者（避免同模型自审自导致偏执，留空则使用全局 llm_provider）
    critic_llm_provider: str = "qwen"

    # ==================== Search 配置 ====================

    # 当前使用的搜索引擎：tavily / serpapi / bing / bocha
    search_provider: Literal["tavily", "serpapi", "bing", "bocha"] = "tavily"

    # Tavily 配置
    tavily_api_key: str = ""

    # SerpAPI 配置
    serpapi_api_key: str = ""

    # Bing 配置
    bing_api_key: str = ""

    # Bocha 配置
    bocha_api_key: str = ""

    # ==================== Academic Search 配置 ====================

    # 当前使用的学术搜索：arxiv / semantic_scholar / openalex
    academic_provider: Literal["arxiv", "semantic_scholar", "openalex"] = "arxiv"

    # Semantic Scholar 配置
    semantic_scholar_api_key: str = ""

    # ==================== Memory 配置 ====================

    # PostgreSQL API state persistence. Leave empty for the legacy in-memory mode.
    database_url: str = ""
    database_echo: bool = False
    database_auto_create: bool = True
    database_recover_interrupted_runs: bool = True

    # ==================== Worker / Redis 配置 ====================

    task_execution_mode: Literal["inline", "worker"] = "inline"
    redis_url: str = ""
    redis_stream_name: str = "deepresearch:runs"
    redis_consumer_group: str = "research-workers"
    worker_lease_seconds: int = 120

    # ChromaDB 持久化目录
    chroma_persist_dir: str = "./chroma_data"

    # 会话 ID：用于记忆隔离，不同会话的记忆互不可见
    # 留空则自动生成 UUID（每次运行都是新会话）
    # 可手动设置为固定值（如 "test"）方便调试时复用历史
    session_id: str = ""

    # ==================== Clarifier 配置 ====================

    # 是否启用 Clarifier（关闭后退化为原有行为，用户问题直接传入 Planner）
    clarifier_enabled: bool = True

    # Clarifier 最大交互轮数
    clarifier_max_rounds: int = 1

    # ==================== 搜索质量过滤配置 ====================

    # 网页搜索结果内容最小长度（字符数），低于此值视为碎片内容，过滤掉
    search_content_min_length: int = 50

    # 网页搜索结果内容最大长度（字符数），超过此值的内容截断
    search_content_max_length: int = 2000

    # 每个子问题进入 LLM 上下文的全局证据预算（不是每个关键词的预算）
    max_web_sources_per_sub_question: int = 8
    max_paper_sources_per_sub_question: int = 4
    max_sources_per_domain: int = 2
    evidence_excerpt_chars: int = 600

    # 论文年份过滤：只保留此年份之后的论文（0 表示不过滤）
    paper_min_year: int = 0

    # 论文引用次数排序：是否按引用次数降序排列（需要搜索引擎支持 citation_count）
    paper_sort_by_citation: bool = True

    # ==================== Critic 审查评分配置 ====================

    # 审查通过阈值（总分 ≥ 此值则通过，满分100）
    critique_pass_threshold: int = 70

    # 审查有条件通过阈值（总分在此值和通过阈值之间则有条件通过）
    # 有条件通过仍进入 Writer，但附上审查意见
    critique_conditional_threshold: int = 50

    # 逐子问题评审通过阈值（单个子问题总分 ≥ 此值则通过，满分100）
    sub_question_critique_pass_threshold: int = 60

    # 整体一致性直接通过阈值（满分100）。达到该分数后，非阻断问题交给 Writer 修正。
    overall_coherence_pass_threshold: int = 80

    # 整体一致性有条件通过阈值。处于该区间时，仅阻断性问题触发补充研究。
    overall_coherence_conditional_threshold: int = 70

    # Critic 最多触发几轮补充研究，达到上限后携带剩余意见进入 Writer。
    critique_max_retries: int = 1

    # ==================== Orchestrator 调度配置 ====================

    # 是否启用 Orchestrator 模式（False 则始终使用串行 Researcher）
    orchestrator_enabled: bool = True

    # 最大并行子 Agent 数量（控制搜索 API 并发请求）
    orchestrator_max_concurrent: int = 3

    # 是否启用动态调整（根据中间结果判断是否追加子问题）
    orchestrator_dynamic_adjust: bool = True

    # 动态调整最大追加次数（防止无限追加子问题，每次追加可包含多个子问题）
    orchestrator_dynamic_adjust_max_rounds: int = 1

    # 每次动态调整最多追加的子问题数量
    orchestrator_dynamic_adjust_max_new_per_round: int = 1

    # 动态调整总共最多追加的子问题数量
    orchestrator_dynamic_adjust_max_total_appends: int = 1

    # 子问题重叠判定阈值（关键词/文本 Jaccard 相似度 ≥ 此值判定为重叠，0-1，0 表示禁用去重）
    sub_question_overlap_threshold: float = 0.6

    # ==================== Evaluation 评估配置 ====================

    # 评估专用 LLM 提供者（留空则使用全局 llm_provider）
    evaluator_llm_provider: str = "qwen"

    # Jina API 密钥（用于事实核查时抓取网页内容）
    jina_api_key: str = ""

    # 评估最大重试次数
    eval_max_attempts: int = 2

    # 评估重试间隔（秒）
    eval_retry_delay: float = 2.0

    # 评估结果输出目录
    eval_output_dir: str = "./eval_results"

    # ==================== Context Compression 配置 ====================

    # 上下文最大 token 数（0 表示根据模型上下文窗口自动计算）
    context_max_tokens: int = 0

    # 上下文占模型窗口的比例（当 context_max_tokens 为 0 时生效）
    context_window_usage_ratio: float = 0.5

    # 文本级压缩阈值比例（超过 max_tokens * ratio 时触发压缩）
    context_compress_threshold_search: float = 0.5
    context_compress_threshold_paper: float = 0.5
    context_compress_threshold_findings: float = 0.5
    context_compress_threshold_prior: float = 0.25
    context_compress_threshold_sources: float = 0.33

    # 对话级渐进压缩触发比例
    context_warning_ratio: float = 0.6
    context_dialog_compress_ratio: float = 0.8

    # 对话级压缩保留最近消息数
    context_recent_message_count: int = 8

    # ==================== Writer 配置 ====================

    # 是否启用两轮交互式写作（True: 先生成大纲供用户审核，再生成完整报告；False: 直接生成报告）
    writer_interactive: bool = True

    # Writer 最大输出 token 数
    writer_max_tokens: int = 12000
    writer_outline_max_tokens: int = 2000

    # 各阶段独立输出预算，避免所有 Agent 共用过大的通用上限
    clarifier_max_tokens: int = 1200
    planner_max_tokens: int = 3000
    researcher_max_tokens: int = 5000
    critic_max_tokens: int = 2500
    critic_findings_chars_per_sub_question: int = 2500
    orchestrator_max_tokens: int = 1500

    # ==================== 通用配置 ====================

    # 搜索结果最大数量
    max_search_results: int = 3

    # 论文搜索结果最大数量
    max_paper_results: int = 3

    # LLM 温度参数（0=确定性输出，1=创造性输出）
    llm_temperature: float = 0.3

    # LLM 最大输出 token 数（Writer Agent 会单独使用更大的值）
    llm_max_tokens: int = 8192

    # 报告输出目录
    output_dir: str = "./outputs"

    @property
    def effective_context_max_tokens(self) -> int:
        if self.context_max_tokens > 0:
            return self.context_max_tokens
        model = ""
        if self.llm_provider == "openai":
            model = self.openai_model
        elif self.llm_provider == "deepseek":
            model = self.deepseek_model
        elif self.llm_provider == "vllm":
            model = self.vllm_model
        window = MODEL_CONTEXT_WINDOWS.get(model, 16000)
        return int(window * self.context_window_usage_ratio)

    @property
    def output_path(self) -> Path:
        """获取输出目录的 Path 对象，自动创建目录"""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def effective_session_id(self) -> str:
        """
        获取有效的会话 ID

        如果 session_id 为空，自动生成一个 UUID；
        否则使用配置的固定值。
        自动生成的 ID 在程序重启后会变化，实现会话隔离。
        """
        if not self.session_id:
            # 每次启动生成新 UUID，确保会话隔离
            self.session_id = uuid.uuid4().hex[:8]
        return self.session_id


# 全局单例配置对象
# 在项目任何地方通过 from config import settings 获取配置
settings = Settings()
