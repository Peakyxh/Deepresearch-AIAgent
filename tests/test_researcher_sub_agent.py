import asyncio
import json
import logging
import sys
import types
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

agents_pkg = types.ModuleType('agents')
agents_pkg.__path__ = [str(BASE_DIR / 'agents')]
agents_pkg.__package__ = 'agents'
sys.modules['agents'] = agents_pkg

workflows_pkg = types.ModuleType('workflows')
workflows_pkg.__path__ = [str(BASE_DIR / 'workflows')]
workflows_pkg.__package__ = 'workflows'
sys.modules['workflows'] = workflows_pkg

from dotenv import load_dotenv
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

from llm.llm_factory import LLMFactory
from agents.researcher_sub_agent import ResearcherSubAgent
from workflows.state import SubQuestion, SubQuestionResult
from search.search_factory import SearchFactory
from academic.academic_factory import AcademicFactory
from memory.context_manager import ContextManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_state_from_txt(txt_path: Path) -> dict:
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    exec_ns = {
        "SubQuestion": SubQuestion,
        "SubQuestionResult": SubQuestionResult,
    }
    local_ns: dict = {}
    exec(content, exec_ns, local_ns)
    return local_ns["state"]


def save_result(result: SubQuestionResult, output_path: Path):
    """将 SubQuestionResult 保存为 JSON 文件"""
    data = {
        "sub_question_id": result.sub_question_id,
        "sub_question": result.sub_question,
        "findings": result.findings,
        "key_insights": result.key_insights,
        "information_gaps": result.information_gaps,
        "search_results_count": len(result.search_results),
        "paper_results_count": len(result.paper_results),
        "sources_count": len(result.sources),
        "sources": result.sources,
        "search_results": result.search_results,
        "paper_results": result.paper_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def main():
    # 创建依赖
    llm = LLMFactory.create()
    search_engine = SearchFactory.create()
    academic_search = AcademicFactory.create()
    context_manager = ContextManager(llm=llm)

    sub_agent = ResearcherSubAgent(
        llm=llm,
        search_engine=search_engine,
        academic_search=academic_search,
        context_manager=context_manager,
    )

    # 加载 planner 输出的 state
    planner_output_path = Path(__file__).parent / "output_planner_state.txt"
    if not planner_output_path.exists():
        print(f"错误：找不到 {planner_output_path}，请先运行 test_planner.py")
        return

    planner_state = load_state_from_txt(planner_output_path)
    print(f"已从 {planner_output_path.name} 加载 state")

    # 从 state 中获取第一个子问题
    structured_sqs = planner_state.get("structured_sub_questions", [])
    if not structured_sqs:
        print("错误：state 中没有 structured_sub_questions")
        return

    sub_question = structured_sqs[0]
    print(f"\n测试子问题: [{sub_question.id}] {sub_question.question}")
    print(f"中文关键词: {sub_question.keywords_zh}")
    print(f"英文关键词: {sub_question.keywords_en}")
    print()

    # 执行子问题研究
    result = await sub_agent.run(
        sub_question=sub_question,
        query=planner_state.get("query", ""),
        research_plan=planner_state.get("research_plan", ""),
    )

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).parent / f"output_sub_agent_{timestamp}.json"
    save_result(result, output_path)

    # 同时保存一份最新的（方便后续测试引用）
    latest_path = Path(__file__).parent / "output_sub_agent_latest.json"
    save_result(result, latest_path)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"子问题研究完成")
    print(f"{'='*60}")
    print(f"子问题 ID: {result.sub_question_id}")
    print(f"网页搜索结果: {len(result.search_results)} 条")
    print(f"论文搜索结果: {len(result.paper_results)} 篇")
    print(f"参考文献: {len(result.sources)} 条")
    print(f"关键洞察: {len(result.key_insights)} 个")
    print(f"信息缺口: {len(result.information_gaps)} 个")
    print(f"\n研究发现预览:")
    print(f"  {result.findings[:200]}...")
    print(f"\n结果已保存到:")
    print(f"  {output_path}")
    print(f"  {latest_path}")


if __name__ == "__main__":
    asyncio.run(main())
