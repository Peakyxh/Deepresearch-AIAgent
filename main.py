"""
DeepResearch Agent 主入口

启动 DeepResearch Agent，执行深度研究工作流。
"""

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# 加载 .env 环境变量
# 必须在 import config 之前加载，否则配置读不到 .env 中的值
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from config import settings
from llm.llm_factory import LLMFactory
from search.search_factory import SearchFactory
from academic.academic_factory import AcademicFactory


def setup_logging() -> None:
    """
    配置日志系统

    设置日志格式和级别，确保所有模块的日志都能正确输出。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def verify_connections() -> bool:
    """
    验证各模块连接是否正常

    检查 LLM、Search、Academic 三个核心模块是否能正常初始化。
    这一步不会发起实际 API 调用，只验证配置是否正确。

    Returns:
        True 表示所有模块初始化正常
    """
    logger = logging.getLogger("verify")
    all_ok = True

    # 验证 LLM
    try:
        llm = LLMFactory.create()
        logger.info(f"✅ LLM 初始化成功: {llm}")
    except Exception as e:
        logger.error(f"❌ LLM 初始化失败: {e}")
        all_ok = False

    # 验证 Search
    try:
        search = SearchFactory.create()
        logger.info(f"✅ Search 初始化成功: {search}")
    except Exception as e:
        logger.error(f"❌ Search 初始化失败: {e}")
        all_ok = False

    # 验证 Academic
    try:
        academic = AcademicFactory.create()
        logger.info(f"✅ Academic 初始化成功: {academic}")
    except Exception as e:
        logger.error(f"❌ Academic 初始化失败: {e}")
        all_ok = False

    return all_ok


async def quick_test() -> None:
    """
    快速测试 —— 验证 LLM 和 Search 能否正常工作

    发起一次简单的 LLM 调用和搜索调用，
    确保整个链路畅通。
    """
    logger = logging.getLogger("quick_test")

    # 测试 LLM
    try:
        llm = LLMFactory.create()
        logger.info("🔄 测试 LLM 生成能力...")
        response = await llm.generate("请用一句话介绍什么是深度学习。")
        logger.info(f"✅ LLM 回复: {response[:100]}...")
    except Exception as e:
        logger.error(f"❌ LLM 测试失败: {e}")

    # 测试 Search
    try:
        search = SearchFactory.create()
        logger.info("🔄 测试搜索能力...")
        results = await search.search("LangGraph tutorial")
        logger.info(f"✅ 搜索返回 {len(results)} 条结果")
        if results:
            logger.info(f"   第一条: {results[0].title}")
    except Exception as e:
        logger.error(f"❌ Search 测试失败: {e}")

    # 测试 Academic Search
    try:
        academic = AcademicFactory.create()
        logger.info("🔄 测试学术搜索能力...")
        papers = await academic.search_papers("large language model")
        logger.info(f"✅ 学术搜索返回 {len(papers)} 篇论文")
        if papers:
            logger.info(f"   第一篇: {papers[0].title}")
    except Exception as e:
        logger.error(f"❌ Academic Search 测试失败: {e}")


async def run_research(query: str) -> None:
    """
    执行深度研究

    Args:
        query: 用户研究问题
    """
    from workflows.research_workflow import ResearchWorkflow

    logger = logging.getLogger("main")
    logger.info(f"🔬 开始深度研究: {query}")

    workflow = ResearchWorkflow()
    state = await workflow.run(query)

    if state.report:
        # 保存报告到文件
        output_path = settings.output_path / f"report_{hash(query) % 10000}.md"
        output_path.write_text(state.report, encoding="utf-8")
        logger.info(f"📄 研究报告已保存: {output_path}")
        print("\n" + "=" * 60)
        print(state.report)
        print("=" * 60)
    else:
        logger.warning("⚠️ 未能生成研究报告")


def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger("main")

    # 打印启动信息
    print("=" * 60)
    print("  🔬 DeepResearch Agent MVP")
    print("=" * 60)
    print(f"  LLM Provider:     {settings.llm_provider}")
    print(f"  Search Provider:  {settings.search_provider}")
    print(f"  Academic Provider: {settings.academic_provider}")
    print(f"  Critic LLM Provider: {settings.critic_llm_provider}")
    print(f"  Session ID:       {settings.effective_session_id}")
    print("=" * 60)

    # 验证连接
    logger.info("正在验证模块连接...")
    ok = asyncio.run(verify_connections())

    if not ok:
        logger.error("模块初始化失败，请检查 .env 配置")
        return

    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # 快速测试模式
            asyncio.run(quick_test())
        else:
            # 研究模式：将命令行参数作为研究问题
            query = " ".join(sys.argv[1:])
            asyncio.run(run_research(query))
    else:
        # 交互模式
        print("\n输入研究问题（输入 q 退出）:")
        while True:
            try:
                query = input("\n🔍 > ").strip()
                if query.lower() == "q":
                    print("再见！")
                    break
                if query:
                    asyncio.run(run_research(query))
            except (KeyboardInterrupt, EOFError):
                print("\n再见！")
                break


if __name__ == "__main__":
    main()
