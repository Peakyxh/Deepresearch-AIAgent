"""
研究工作流状态定义

使用 Pydantic 定义工作流的状态对象，
确保状态在各个 Agent 之间传递时类型安全。
"""

from pydantic import BaseModel, ConfigDict, Field


class SubQuestion(BaseModel):
    """
    带依赖关系的子问题

    Planner 输出的子问题不再只是字符串，
    而是包含 id、依赖关系、优先级和搜索关键词的结构化对象。
    """

    id: str = Field(default="", description="子问题唯一标识，如 sq_1、sq_2")
    question: str = Field(default="", description="子问题内容")
    depends_on: list[str] = Field(
        default_factory=list,
        description="依赖的子问题 id 列表，如 ['sq_1']，空列表表示无依赖",
    )
    priority: int = Field(
        default=0,
        description="优先级，0=最高，数值越大优先级越低",
    )
    keywords_zh: list[str] = Field(
        default_factory=list, description="中文搜索关键词"
    )
    keywords_en: list[str] = Field(
        default_factory=list, description="英文搜索关键词"
    )


class SubQuestionResult(BaseModel):
    """
    子问题研究结果

    每个 Researcher Sub-Agent 处理一个子问题后返回的结果，
    包含搜索结果、研究发现和参考文献。
    """

    sub_question_id: str = Field(default="", description="对应的子问题 id")
    sub_question: str = Field(default="", description="子问题内容")
    search_results: list[dict] = Field(
        default_factory=list, description="网页搜索结果"
    )
    paper_results: list[dict] = Field(
        default_factory=list, description="论文搜索结果"
    )
    findings: str = Field(default="", description="本子问题的研究发现")
    key_insights: list[str] = Field(
        default_factory=list, description="关键洞察"
    )
    information_gaps: list[str] = Field(
        default_factory=list, description="信息缺口"
    )
    sources: list[dict] = Field(
        default_factory=list, description="本子问题的参考文献来源"
    )
    evidence_cards: list[dict] = Field(
        default_factory=list, description="供下游复用的紧凑结构化证据卡片"
    )


class ResearchState(BaseModel):
    """
    研究工作流状态

    这个状态对象在整个工作流中传递，
    每个 Agent 读取和更新其中的字段。

    数据流：
    用户问题 → Planner → Orchestrator → Researcher Sub-Agents → Critic → Writer
    """

    # ==================== 输入 ====================
    query: str = Field(default="", description="用户原始研究问题")

    # ==================== Clarifier 输出 ====================
    clarified_intent: str = Field(default="", description="结构化意图描述")
    clarification_qa: list[dict] = Field(
        default_factory=list, description="澄清问答对，每项包含 question/answer"
    )
    clarification_round: int = Field(default=0, description="已完成的澄清轮次")
    pending_clarification_questions: list[dict] = Field(
        default_factory=list, description="等待用户回答的澄清问题"
    )

    # ==================== 会话与历史 ====================
    session_id: str = Field(default="", description="会话ID，用于记忆隔离")
    history_context: str = Field(default="", description="历史研究参考上下文")

    # ==================== Planner 输出 ====================
    sub_questions: list[str] = Field(
        default_factory=list, description="拆解后的子问题列表（兼容旧格式）"
    )
    structured_sub_questions: list[SubQuestion] = Field(
        default_factory=list,
        description="带依赖关系的结构化子问题列表（新格式）",
    )
    search_keywords: list[dict] = Field(
        default_factory=list,
        description="搜索关键词列表，每项包含 sub_question/keywords_zh/keywords_en",
    )
    research_plan: str = Field(default="", description="研究计划")

    # ==================== Orchestrator 输出 ====================
    orchestrator_plan: dict = Field(
        default_factory=dict,
        description="Orchestrator 调度计划，包含执行层级和并行分组",
    )
    sub_question_results: list[SubQuestionResult] = Field(
        default_factory=list, description="各子问题的研究结果"
    )

    # ==================== Researcher 输出 ====================
    search_results: list[dict] = Field(
        default_factory=list, description="网页搜索结果"
    )
    paper_results: list[dict] = Field(
        default_factory=list, description="论文搜索结果"
    )
    findings: str = Field(default="", description="研究发现汇总")
    sources: list[dict] = Field(
        default_factory=list,
        description="参考文献来源列表，每项包含 title/url/type/year/authors",
    )
    evidence_cards: list[dict] = Field(
        default_factory=list, description="跨子问题去重后的证据卡片"
    )
    token_usage: dict = Field(
        default_factory=dict, description="按 Agent 汇总的估算输入/输出 token"
    )

    # ==================== Critic 输出 ====================
    critique_passed: bool = Field(default=False, description="审查是否通过")
    critique_feedback: str = Field(default="", description="审查反馈意见")
    critique_feedback_for_researcher: str = Field(
        default="", description="压缩后的审查反馈，供 Researcher 参考"
    )
    prior_round_findings: str = Field(
        default="",
        description="历轮研究发现（来自 ContextManager），供重试轮次的 Researcher 在上一轮基础上修正补充",
    )
    revision_suggestions: list[str] = Field(
        default_factory=list, description="修正建议列表"
    )
    critique_scores: dict = Field(
        default_factory=dict,
        description="多维度评分：factual_accuracy/logic_consistency/completeness/source_sufficiency/timeliness",
    )
    critique_score_details: dict = Field(
        default_factory=dict, description="各维度评分理由"
    )
    critique_total_score: int = Field(default=0, description="审查总分（满分100）")

    # ==================== Critic 输出（逐子问题评审） ====================
    per_sub_question_critiques: dict = Field(
        default_factory=dict,
        description="逐子问题评审结果，key 为子问题 id，value 包含 scores/passed/feedback 等",
    )
    failed_sub_question_ids: list[str] = Field(
        default_factory=list,
        description="评审未通过的子问题 id 列表，供 Orchestrator 选择性重试",
    )
    overall_coherence_score: int = Field(
        default=0, description="整体一致性评分（满分100）"
    )
    overall_coherence_feedback: str = Field(
        default="", description="整体一致性审查反馈"
    )
    overall_coherence_passed: bool = Field(
        default=False, description="整体一致性是否通过"
    )
    critique_outcome: str = Field(
        default="",
        description="审查路由结果：passed/conditional_pass/research_retry/forced_writer",
    )
    writer_revision_suggestions: list[str] = Field(
        default_factory=list,
        description="无需重新搜索、由 Writer 在成稿阶段处理的修改建议",
    )
    missing_research_topics: list[dict] = Field(
        default_factory=list,
        description="整体评审识别出的阻断性缺失研究主题",
    )

    # ==================== Writer 输出 ====================
    report: str = Field(default="", description="最终研究报告")
    report_outline: str = Field(default="", description="报告大纲（两轮写作第一轮生成）")
    report_outline_display: str = Field(default="", description="供界面展示的格式化报告大纲")
    report_outline_payload: dict | str = Field(
        default_factory=dict, description="结构化报告大纲或原始文本"
    )
    user_outline_feedback: str = Field(default="", description="用户对大纲的修改建议")

    # ==================== 流程控制 ====================
    current_step: str = Field(default="planner", description="当前执行阶段")
    retry_count: int = Field(default=0, description="重试次数")
    max_retries: int = Field(default=1, description="最大补充研究次数")
    use_orchestrator: bool = Field(
        default=True,
        description="是否使用 Orchestrator 模式（False 则回退到串行 Researcher）",
    )
    error: str = Field(default="", description="错误信息")

    model_config = ConfigDict(arbitrary_types_allowed=True)
