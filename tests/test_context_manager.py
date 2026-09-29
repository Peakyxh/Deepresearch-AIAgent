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
from memory.context_manager import ContextManager
from search.search_factory import SearchFactory
from search.base_search import BaseSearch, SearchResult


def format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "（无网页搜索结果）"

    formatted = []
    for i, r in enumerate(results, 1):
        text = f"[网页{i}] 标题: {r.title}\n"
        text += f"       链接: {r.url}\n"
        text += f"       内容: {r.content}\n"
        text += f"       来源: {r.source} | 评分: {r.score:.2f}"
        formatted.append(text)

    return "\n\n".join(formatted)


async def main():
    # 创建依赖
    llm = LLMFactory.create()
    context_manager = ContextManager(llm=llm)
    search_engine = SearchFactory.create()

    # 搜索
    search_results: list[SearchResult] = []
    for kw in ['地方财政收入', '房地产税收构成', '土地出让金占比']:
        results = await search_engine.search(kw)
        search_results.extend(results)

    # 格式化搜索结果
    original_text = format_search_results(search_results)

    # 保存压缩前的内容
    output_dir = Path(__file__).parent / "output_compress"
    output_dir.mkdir(exist_ok=True)

    original_path = output_dir / "before_compress.txt"
    with open(original_path, "w", encoding="utf-8") as f:
        f.write(original_text)

    original_tokens = context_manager.estimate_text_tokens(original_text)
    original_chars = len(original_text)
    print(f"压缩前: {original_chars} 字符, 约 {original_tokens} tokens")
    print(f"已保存到: {original_path}")

    # 压缩
    compressed_text = await context_manager.compress_search_results(original_text)

    # 保存压缩后的内容
    compressed_path = output_dir / "after_compress.txt"
    with open(compressed_path, "w", encoding="utf-8") as f:
        f.write(compressed_text)

    compressed_tokens = context_manager.estimate_text_tokens(compressed_text)
    compressed_chars = len(compressed_text)
    ratio = compressed_tokens / original_tokens * 100 if original_tokens > 0 else 0
    print(f"压缩后: {compressed_chars} 字符, 约 {compressed_tokens} tokens")
    print(f"压缩率: {ratio:.1f}%")
    print(f"已保存到: {compressed_path}")

    # 保存对比摘要
    summary_path = output_dir / "compress_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "search_keywords": ['地方财政收入', '房地产税收构成', '土地出让金占比'],
        "search_results_count": len(search_results),
        "original": {
            "chars": original_chars,
            "tokens": original_tokens,
        },
        "compressed": {
            "chars": compressed_chars,
            "tokens": compressed_tokens,
        },
        "ratio": f"{ratio:.1f}%",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"摘要已保存到: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
