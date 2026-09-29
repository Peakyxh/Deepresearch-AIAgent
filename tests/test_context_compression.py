"""
上下文压缩回归测试（离线，不调用任何外部 API）

运行方式：
    python tests/test_context_compression.py

覆盖的核心约定：
1. is_key=True 只表示"最高保留优先级"，不是"绝对不可压缩"：
   分级压缩后仍超限时，关键信息也会被摘要/截断，上下文永不超出模型窗口。
2. 非关键的大块内容（研究发现）必须真正参与压缩，否则上下文只增不减。
3. 关键信息不再被重复存储，get_key_info() 直接从 context 派生，
   并且只包含 is_key=True 的内容（研究发现不重复注入下游 prompt）。
4. 下游 Agent（Critic / Writer）能通过 BaseAgent.context_prompt_prefix()
   拿到上游关键背景。
5. 工作流写入上下文的 is_key 标记正确（意图澄清/研究计划为关键，
   研究发现为非关键）。

注意：openai / chromadb 等运行期依赖不参与本测试，因此这里用和其他测试
脚本相同的手法，把 memory / agents / llm 包替换成轻量命名空间包。
"""

import asyncio
import logging
import sys
import types
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# ---- 轻量命名空间包：避免触发 chromadb / openai 等重依赖的 __init__ ----
for pkg_name in ("memory", "agents", "llm", "workflows", "search", "academic"):
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(BASE_DIR / pkg_name)]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg

# llm_factory 会级联导入 openai/deepseek 等运行期依赖，这里替换为桩
_factory_module = types.ModuleType("llm.llm_factory")


class _StubLLMFactory:
    @staticmethod
    def create(*args, **kwargs):
        raise RuntimeError("离线测试不应创建真实 LLM")


_factory_module.LLMFactory = _StubLLMFactory
sys.modules["llm.llm_factory"] = _factory_module

for factory_module_name, factory_class_name in (
    ("search.search_factory", "SearchFactory"),
    ("academic.academic_factory", "AcademicFactory"),
):
    factory_module = types.ModuleType(factory_module_name)

    class _StubFactory:
        @staticmethod
        def create(*args, **kwargs):
            raise RuntimeError("离线测试应显式注入依赖")

    setattr(factory_module, factory_class_name, _StubFactory)
    sys.modules[factory_module_name] = factory_module

from memory.context_manager import ContextManager  # noqa: E402
from tests.mocks import MockLLM  # noqa: E402


logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")


# ==================== 辅助 ====================

def filler(size: int, char: str = "X") -> str:
    return char * size


def context_chars(cm: ContextManager) -> int:
    return sum(len(item["content"]) for item in cm.context)


# ==================== 用例 ====================

async def check_findings_are_compressible() -> str:
    """
    研究发现标记为非关键信息后，必须能被紧急压缩摘要掉，
    同时关键信息（意图澄清/研究计划）原文保留。
    """
    cm = ContextManager(max_tokens=1000, llm=MockLLM())

    cm.add_context("system", "意图澄清: " + filler(100, "I"), is_key=True, query="q1")
    cm.add_context("system", "研究计划: " + filler(100, "P"), is_key=True, query="q1")
    cm.add_context("system", "研究发现: " + filler(12000, "F"), is_key=False, query="q1")
    for i in range(6):
        cm.add_context("system", f"补充资料{i}: " + filler(200), is_key=False, query="q1")

    before = cm.estimate_tokens()
    assert before > cm.max_tokens, "构造的上下文应当超限"

    await cm.maybe_compress()

    after = cm.estimate_tokens()
    assert after <= cm.max_tokens, f"压缩后仍超限: {after}/{cm.max_tokens}"

    max_entry_chars = max(len(item["content"]) for item in cm.context)
    assert max_entry_chars < 1000, (
        f"非关键的研究发现没有被压缩掉，最长条目仍有 {max_entry_chars} 字符"
    )

    key_info = cm.get_key_info()
    assert "意图澄清: " in key_info, "关键信息（意图澄清）被丢失"
    assert "研究计划: " in key_info, "关键信息（研究计划）被丢失"
    assert "研究发现" not in key_info, "非关键的研究发现不应进入关键信息"

    summary_entries = [
        item for item in cm.context if item["content"].startswith("之前对话的紧急摘要")
    ]
    assert summary_entries, "没有生成紧急摘要"

    return f"{before} → {after} tokens，研究发现被摘要，关键信息保留"


async def check_lightweight_keeps_key_but_trims_non_key() -> str:
    """60%~80% 水位：非关键内容被修剪，关键内容原样保留。"""
    cm = ContextManager(max_tokens=1000, llm=MockLLM())

    key_content = "意图澄清: " + filler(100, "I")
    findings_content = "研究发现: " + filler(2000, "F")
    cm.add_context("system", key_content, is_key=True, query="q1")
    cm.add_context("system", findings_content, is_key=False, query="q1")

    ratio = cm.estimate_tokens() / cm.max_tokens
    assert 0.6 < ratio <= 0.8, f"用例未落在轻量压缩水位: {ratio:.2f}"

    await cm.maybe_compress()

    assert cm.context[0]["content"] == key_content, "关键信息被轻量压缩改动了"
    assert len(cm.context[1]["content"]) < len(findings_content), "非关键内容没有被修剪"
    assert cm.estimate_tokens() <= cm.max_tokens

    return f"关键信息原文保留，研究发现 {len(findings_content)} → {len(cm.context[1]['content'])} 字符"


async def check_extractive_summarize_keeps_key_sentences() -> str:
    """
    轻量压缩的修剪使用抽取式摘要（切句打分选句，不调 LLM）：
    1. 结论句/数据句优先保留（硬截断会丢失尾部结论）
    2. 普通叙述句被省略并插入省略标记
    3. 无句边界的文本退化为头尾保留截断（仍有输出，带标记）
    """
    cm = ContextManager(max_tokens=1000, llm=MockLLM())

    conclusion = "因此可以得出结论：该方法在所有测试场景中均优于基线。"
    data = "实验数据显示准确率提升了23.5%，延迟降低了40%。"
    filler_sentence = "这是一句普通的叙述性描述，不涉及任何测量指标，仅用于填充篇幅。"
    long_text = (
        "研究发现的背景介绍。"
        + filler_sentence * 55
        + conclusion
        + "更多的普通叙述句子用于占据中间位置。" * 10
        + data
        + "结尾的总结句。"
    )
    # 触发轻量压缩：ratio ∈ (60%, 80%]，且超过单条修剪上限（max_tokens*3//2 = 1500 字符）
    ratio = cm.estimate_text_tokens(long_text) / cm.max_tokens
    assert 0.6 < ratio <= 0.8, f"用例未落在轻量压缩水位: {ratio:.2f}"
    assert len(long_text) > 1500, "用例应触发单条修剪"

    cm.add_context("system", long_text, is_key=False, query="q1")
    await cm.maybe_compress()

    summarized = cm.context[0]["content"]
    assert len(summarized) < len(long_text), "长文本没有被摘要"
    assert conclusion in summarized, "结论句被丢弃（应优先保留）"
    assert data in summarized, "数据句被丢弃（应优先保留）"
    assert "已省略" in summarized, "被省略的部分应插入省略标记"

    # 无句边界（无标点无换行）→ 头尾保留兜底仍有输出
    cm2 = ContextManager(max_tokens=1000, llm=MockLLM())
    no_boundary = filler(2000)
    cm2.add_context("system", no_boundary, is_key=False, query="q1")
    await cm2.maybe_compress()

    trimmed = cm2.context[0]["content"]
    assert len(trimmed) < len(no_boundary), "无句边界文本没有被修剪"
    assert "已省略" in trimmed, "头尾保留截断应带省略标记"

    return "结论/数据句优先保留，普通句被省略标记；无句边界退化头尾截断"


async def check_hash_dedup_keeps_retry_findings() -> str:
    """
    去重使用完整内容 hash（而非前 200 字符前缀）：
    同前缀、不同内容的重试轮研究发现不会被误删。
    （前缀去重会把重试轮修正后的新发现误判为重复，导致旧错误版本覆盖新结论）
    """
    cm = ContextManager(max_tokens=1000, llm=MockLLM())

    # 共同前缀（>200 字符）+ 不同的修正内容，模拟"重试轮与上一轮开头相同"
    common_prefix = "研究发现: " + "子问题A的有效结论。" + "该子问题上一轮已得到部分结论，本轮针对审查反馈修正。" * 10
    old_findings = common_prefix + filler(800, "F")
    new_findings = common_prefix + filler(800, "G")
    assert old_findings[:200] == new_findings[:200], "用例应构造出前缀碰撞"
    assert old_findings != new_findings, "两条内容应不同"

    cm.add_context("system", old_findings, is_key=False, query="q1")
    cm.add_context("system", new_findings, is_key=False, query="q1")

    ratio = cm.estimate_tokens() / cm.max_tokens
    assert 0.6 < ratio <= 0.8, f"用例未落在轻量压缩水位: {ratio:.2f}"

    await cm.maybe_compress()

    contents = [item["content"] for item in cm.context]
    assert any("F" * 100 in c for c in contents), "旧发现不应被误删"
    assert any("G" * 100 in c for c in contents), (
        "同前缀异内容的新发现被前缀去重误删（hash 去重应保留两者）"
    )

    return "同前缀异内容条目均保留（hash 去重），重试轮新发现不再被旧版本覆盖"


async def check_all_key_context_still_converges() -> str:
    """
    退化场景（曾经的线上问题）：所有条目都是 is_key=True。

    此时"保留优先"的分级压缩无事可做，但 _force_shrink() 必须兜底，
    保证上下文仍然收敛到 max_tokens 以内，且重复调用不再增长。
    """
    cm = ContextManager(max_tokens=1000, llm=MockLLM())

    for i in range(3):
        cm.add_context("system", f"关键事实{i}: " + filler(5000, "K"), is_key=True, query="q1")

    assert cm.estimate_tokens() > cm.max_tokens

    await cm.maybe_compress()
    first_pass = cm.estimate_tokens()
    assert first_pass <= cm.max_tokens, f"全部关键信息时压缩失效: {first_pass}/{cm.max_tokens}"

    await cm.maybe_compress()
    second_pass = cm.estimate_tokens()
    assert second_pass <= cm.max_tokens, "重复压缩后仍然超限"
    assert second_pass <= first_pass, "重复压缩导致上下文增长"

    assert cm.context, "强制收缩后上下文不应为空"
    assert cm.context[0]["content"].startswith("之前上下文的强制摘要"), (
        "强制收缩应当留下一条摘要（关键信息降级为摘要）"
    )

    return f"全部 is_key=True 时仍收敛: → {first_pass} tokens，二次压缩 {second_pass} tokens"


async def check_no_llm_still_bounded() -> str:
    """没有 LLM 时必须走硬截断兜底，绝不允许超限。"""
    cm = ContextManager(max_tokens=100, llm=None)

    for i in range(3):
        cm.add_context("system", f"大块内容{i}: " + filler(5000), is_key=False, query="q1")

    await cm.maybe_compress()

    assert cm.estimate_tokens() <= cm.max_tokens, (
        f"无 LLM 时未收敛: {cm.estimate_tokens()}/{cm.max_tokens}"
    )
    assert context_chars(cm) <= cm.max_tokens * 3, "硬截断没有遵守字符预算"
    assert any("已强制截断" in item["content"] for item in cm.context), "没有留下截断标记"

    return f"无 LLM 时硬截断成功: {context_chars(cm)} 字符 / 预算 {cm.max_tokens * 3}"


def check_key_info_not_duplicated() -> str:
    """关键信息不再重复存储，且只包含 is_key=True 的内容。"""
    cm = ContextManager(max_tokens=100000, llm=None)

    cm.add_context("system", "意图澄清: A", is_key=True, query="q")
    cm.add_context("system", "研究发现: B", is_key=False, query="q")

    assert cm.context[0]["role"] == "system", "工作流注入的事实应使用 system 角色"
    assert cm._key_info == [], "add_context 不应再向 _key_info 复制内容"

    key_info = cm.get_key_info()
    assert key_info == "意图澄清: A", f"关键信息内容不正确: {key_info!r}"
    assert key_info.count("意图澄清: A") == 1, "关键信息被重复拼装"

    cm.add_key_info("额外事实: C")
    assert "额外事实: C" in cm.get_key_info(), "add_key_info 的短事实应保留"

    cm.clear()
    assert cm.context == [] and cm._key_info == [], "clear() 未清空上下文"
    assert cm.get_key_info() == "", "clear() 后关键信息应为空"

    return "关键信息只存一份，非关键内容不进入 get_key_info()，clear() 可隔离会话"


async def check_agent_prompt_injection() -> str:
    """下游 Agent 通过 context_prompt_prefix() 拿到关键背景，且不重复注入研究发现。"""
    import agents.base_agent as base_agent  # noqa: E402

    class _ProbeAgent(base_agent.BaseAgent):
        async def _execute(self, state):
            return state

    cm = ContextManager(max_tokens=100000, llm=None)
    cm.add_context("system", "意图澄清: 关注 2024 年数据", is_key=True, query="q")
    cm.add_context("system", "研究计划: 先查财政再查土地", is_key=True, query="q")
    cm.add_context("system", "研究发现: " + filler(500), is_key=False, query="q")

    agent = _ProbeAgent(name="Probe", llm=MockLLM(), context_manager=cm)
    prefix = agent.context_prompt_prefix()

    assert prefix.startswith("## 研究背景"), "注入前缀缺少小节标题"
    assert "意图澄清: 关注 2024 年数据" in prefix
    assert "研究计划: 先查财政再查土地" in prefix
    assert "研究发现" not in prefix, "大块研究发现不应重复注入 prompt"

    no_manager = _ProbeAgent(name="Probe", llm=MockLLM(), context_manager=None)
    assert no_manager.context_prompt_prefix() == "", "没有上下文管理器时应返回空串"

    cm_non_key = ContextManager(max_tokens=100000, llm=None)
    cm_non_key.add_context("system", "研究发现: 只有非关键内容", is_key=False, query="q")
    only_non_key = _ProbeAgent(name="Probe", llm=MockLLM(), context_manager=cm_non_key)
    assert only_non_key.context_prompt_prefix() == "", "只有非关键内容时不应注入前缀"

    return "Critic/Writer 可取得意图澄清与研究计划，且不重复注入研究发现"


async def check_writer_prompt_injection_end_to_end() -> str:
    """Writer 真实 prompt 中必须包含上游关键背景，且不重复注入研究发现。"""
    from agents.writer_agent import WriterAgent  # noqa: E402

    cm = ContextManager(max_tokens=100000, llm=None)
    cm.add_context("system", "意图澄清: 只关注 2024 年数据", is_key=True, query="q")
    cm.add_context("system", "研究计划: 先查财政再查土地", is_key=True, query="q")
    cm.add_context("system", "研究发现: " + filler(400), is_key=False, query="q")

    llm = MockLLM(default_response="## 报告\n正文")
    agent = WriterAgent(llm=llm, context_manager=cm)
    agent.interactive = False

    await agent._direct_write(
        query="地方财政收入",
        findings="研究发现正文",
        critique="",
        search_text="网页来源",
        paper_text="论文来源",
    )

    prompt = llm.call_log[-1]["prompt"]
    assert "意图澄清: 只关注 2024 年数据" in prompt, "Writer prompt 未注入意图澄清"
    assert "研究计划: 先查财政再查土地" in prompt, "Writer prompt 未注入研究计划"
    assert "研究发现: " + filler(400) not in prompt, "非关键的研究发现被重复注入 prompt"

    return "Writer 调用 LLM 的真实 prompt 中包含意图澄清与研究计划"


async def check_critic_prompt_injection_end_to_end() -> str:
    """Critic 逐子问题评审的 prompt 中必须包含上游关键背景。"""
    from agents.critic_agent import CriticAgent  # noqa: E402

    cm = ContextManager(max_tokens=100000, llm=None)
    cm.add_context("system", "意图澄清: 只关注 2024 年数据", is_key=True, query="q")
    cm.add_context("system", "研究计划: 先查财政再查土地", is_key=True, query="q")

    llm = MockLLM(
        default_response=(
            '{"scores": {"factual_accuracy": 25}, "score_details": {}, '
            '"feedback": "ok", "issues": [], "revision_suggestions": []}'
        )
    )
    agent = CriticAgent(llm=llm, context_manager=cm)

    await agent._critique_single_sub_question(
        query="地方财政收入",
        sq_result={
            "sub_question_id": "sq_1",
            "sub_question": "财政收入的构成是什么？",
            "findings": "研究发现内容",
            "search_results": [],
            "paper_results": [],
        },
    )

    prompt = llm.call_log[-1]["prompt"]
    assert "意图澄清: 只关注 2024 年数据" in prompt, "Critic prompt 未注入意图澄清"
    assert "研究计划: 先查财政再查土地" in prompt, "Critic prompt 未注入研究计划"

    return "Critic 调用 LLM 的真实 prompt 中包含意图澄清与研究计划"


def check_workflow_is_key_wiring() -> str:
    """
    钉住工作流写入上下文时的 is_key 标记（这是最初报告的缺陷）：
    意图澄清/研究计划 = 关键信息，研究发现 = 非关键信息。
    """
    source = (BASE_DIR / "workflows" / "research_workflow.py").read_text(encoding="utf-8")

    def block_around(anchor: str, before: int = 220, after: int = 260) -> str:
        idx = source.find(anchor)
        assert idx != -1, f"未找到锚点: {anchor}"
        start = max(0, idx - before)
        return source[start : idx + after]

    intent_block = block_around('content=f"意图澄清: {clarified_intent}"')
    assert "is_key=True" in intent_block, "意图澄清应标记为关键信息"
    assert 'role="system"' in intent_block, "意图澄清应使用 system 角色"

    plan_block = block_around("content=f\"研究计划: {state.get('research_plan', '')}\"")
    assert "is_key=True" in plan_block, "研究计划应标记为关键信息"
    assert 'role="system"' in plan_block, "研究计划应使用 system 角色"

    findings_block = block_around('content=f"研究发现: {findings_text}"')
    assert "is_key=False" in findings_block, (
        "研究发现体积最大且 state 中已有同一份内容，必须标记为非关键信息，"
        "否则上下文压缩会退化为空操作"
    )
    assert 'role="system"' in findings_block, "研究发现应使用 system 角色"

    assert "self.context_manager.clear()" in source, "run() 开始时缺少会话隔离 clear()"

    return "意图澄清/研究计划 is_key=True，研究发现 is_key=False，run() 会 clear()"


async def check_rounds_findings_consumption() -> str:
    """
    get_rounds_findings() 是 self.context 中研究发现条目的真正消费者：
    1. 无研究发现时返回空串（不注入）
    2. 未超预算时原样拼接返回，不调用 LLM
    3. 超预算时触发压缩并标注摘要前缀
    4. compress_text 的 threshold_tokens 让调用方阈值真正生效
    """
    # 1. 无研究发现
    cm = ContextManager(max_tokens=1000, llm=MockLLM())
    cm.add_context("system", "意图澄清: 关注 2024 年", is_key=True, query="q")
    assert await cm.get_rounds_findings() == "", "无研究发现时应返回空串"

    # 2. 未超预算：原样返回，不调 LLM
    cm = ContextManager(max_tokens=1000, llm=MockLLM())
    cm.add_context("system", "研究发现: 第一轮发现A", is_key=False, query="q")
    cm.add_context("system", "研究发现: 第一轮发现B", is_key=False, query="q")
    llm = cm._llm
    text = await cm.get_rounds_findings()
    assert "第一轮发现A" in text and "第一轮发现B" in text, "应拼接各轮研究发现"
    assert llm.call_log == [], "未超预算时不应调用 LLM 压缩"

    # 3. 超预算（prior 阈值 = 25% * 1000 = 250 tokens = 750 字符）：触发压缩
    big = "研究发现: " + filler(2000, "F")
    cm.add_context("system", big, is_key=False, query="q")
    text = await cm.get_rounds_findings()
    assert llm.call_log, "超预算时应调用 LLM 压缩"
    assert text.startswith("（以下为压缩后的历轮研究发现摘要）"), "压缩结果应带摘要前缀"
    assert '{"result": "mock"}' in text, "压缩结果应为 LLM 返回内容"

    # 4. threshold_tokens：文本未超 max_tokens 但超调用方阈值时，压缩必须真正发生
    cm2 = ContextManager(max_tokens=1000, llm=MockLLM())
    llm2 = cm2._llm
    body = filler(900)  # 300 tokens：未超 max_tokens(1000)，超过 threshold(250)
    out = await cm2.compress_text(body, threshold_tokens=250)
    assert llm2.call_log, "threshold_tokens 触发的压缩不应被 max_tokens 短路"
    assert out == '{"result": "mock"}', "应返回压缩结果而非原文"
    # 不传 threshold_tokens 时保持旧行为：未超 max_tokens 不压缩
    out2 = await cm2.compress_text(body)
    assert out2 == body and len(llm2.call_log) == 1, "默认行为应保持不变"

    return "历轮研究发现可被提取注入重试，threshold_tokens 让调用方阈值真正生效"


async def check_named_compression_thresholds() -> str:
    """各类文本的业务阈值必须传递到 compress_text，不得被 max_tokens 短路。"""
    body = filler(1800)  # 600 tokens：低于 max_tokens=1000，高于 50% 业务阈值

    for method_name in (
        "compress_search_results",
        "compress_paper_results",
        "compress_findings",
    ):
        llm = MockLLM()
        cm = ContextManager(max_tokens=1000, llm=llm)
        output = await getattr(cm, method_name)(body)
        assert llm.call_log, f"{method_name} 未在业务阈值处触发压缩"
        assert output != body, f"{method_name} 被 max_tokens 错误短路"

    # 无 LLM 时的截断标记也必须计入阈值预算。
    cm_no_llm = ContextManager(max_tokens=1000, llm=None)
    truncated = await cm_no_llm.compress_search_results(body)
    limit = cm_no_llm.get_compress_threshold("search")
    assert cm_no_llm.estimate_text_tokens(truncated) <= limit
    assert "已截断" in truncated

    return "search/paper/findings 的业务阈值均生效，无 LLM 回退也严格有界"


async def check_serial_researcher_small_input() -> str:
    """串行 Researcher 的输入未超限分支不得读取未赋值的 paper_tokens。"""
    from agents.researcher_agent import ResearcherAgent  # noqa: E402
    from tests.mocks import MockAcademic, MockLLM, MockSearch  # noqa: E402

    llm = MockLLM(default_response='{"findings": "离线研究结论"}')
    cm = ContextManager(max_tokens=100000, llm=llm)
    agent = ResearcherAgent(
        llm=llm,
        search_engine=MockSearch(),
        academic_search=MockAcademic(),
        context_manager=cm,
    )
    state = {
        "query": "RAG 的最新进展",
        "sub_questions": ["RAG 的最新进展是什么？"],
        "search_keywords": [{"keywords_zh": [], "keywords_en": ["RAG advances"]}],
        "research_plan": "搜索并汇总",
    }

    result = await agent.run(state)
    assert result["findings"] == "离线研究结论"
    assert result["current_step"] == "critic"
    return "串行 Researcher 未超限路径可完整执行"


def check_untrusted_content_policy() -> str:
    """所有消费搜索或研究产出的 Agent 都必须带统一的不可信内容策略。"""
    from prompts.critic_prompt import (
        CRITIC_SYSTEM_PROMPT,
        OVERALL_COHERENCE_CRITIC_SYSTEM_PROMPT,
        SUB_QUESTION_CRITIC_SYSTEM_PROMPT,
    )
    from prompts.researcher_prompt import RESEARCHER_SYSTEM_PROMPT, RESEARCHER_USER_PROMPT
    from prompts.writer_prompt import (
        WRITER_OUTLINE_SYSTEM_PROMPT,
        WRITER_SYSTEM_PROMPT,
        WRITER_WITH_OUTLINE_SYSTEM_PROMPT,
    )

    protected_prompts = (
        RESEARCHER_SYSTEM_PROMPT,
        CRITIC_SYSTEM_PROMPT,
        SUB_QUESTION_CRITIC_SYSTEM_PROMPT,
        OVERALL_COHERENCE_CRITIC_SYSTEM_PROMPT,
        WRITER_SYSTEM_PROMPT,
        WRITER_OUTLINE_SYSTEM_PROMPT,
        WRITER_WITH_OUTLINE_SYSTEM_PROMPT,
    )
    for prompt in protected_prompts:
        assert "不可信数据" in prompt
        assert "绝不执行" in prompt
        assert "API Key" in prompt

    assert "<untrusted_web_sources>" in RESEARCHER_USER_PROMPT
    assert "<untrusted_paper_sources>" in RESEARCHER_USER_PROMPT
    return "Researcher/Critic/Writer 统一防护，搜索与论文内容有显式边界"


# ==================== 运行器 ====================

CHECKS = [
    check_findings_are_compressible,
    check_lightweight_keeps_key_but_trims_non_key,
    check_extractive_summarize_keeps_key_sentences,
    check_hash_dedup_keeps_retry_findings,
    check_all_key_context_still_converges,
    check_no_llm_still_bounded,
    check_agent_prompt_injection,
    check_writer_prompt_injection_end_to_end,
    check_critic_prompt_injection_end_to_end,
    check_rounds_findings_consumption,
    check_named_compression_thresholds,
    check_serial_researcher_small_input,
]


async def main() -> int:
    failures: list[tuple[str, str]] = []

    print("=" * 70)
    print("上下文压缩回归测试（离线）")
    print("=" * 70)

    for check in CHECKS:
        name = check.__name__
        try:
            detail = await check()
            print(f"[PASS] {name}\n   {detail}")
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"[FAIL] {name}\n   {e}")
        except Exception as e:  # noqa: BLE001
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"[ERROR] {name}\n   {type(e).__name__}: {e}")

    for check in (
        check_key_info_not_duplicated,
        check_workflow_is_key_wiring,
        check_untrusted_content_policy,
    ):
        name = check.__name__
        try:
            detail = check()
            print(f"[PASS] {name}\n   {detail}")
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"[FAIL] {name}\n   {e}")
        except Exception as e:  # noqa: BLE001
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"[ERROR] {name}\n   {type(e).__name__}: {e}")

    print("=" * 70)
    if failures:
        print(f"失败 {len(failures)} 项：")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
        return 1

    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
