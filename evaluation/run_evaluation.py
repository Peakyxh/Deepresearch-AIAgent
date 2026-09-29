"""
评估系统使用入口

支持三种评估模式：
  1. 评估单篇报告文件（--mode file）
  2. 批量评估 JSONL 文件（--mode batch）
  3. 交互式输入评估（--mode interactive）

用法示例：
  python evaluation/run_evaluation.py --mode file --input ./outputs/report_1234.md
  python evaluation/run_evaluation.py --mode batch --input evaluation/data/reports.jsonl --output evaluation/eval_results --resume
  python evaluation/run_evaluation.py --mode interactive
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保能找到 config 等模块
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 加载 .env 环境变量（必须在 import config 之前）
env_path = Path(_project_root) / ".env"
from dotenv import load_dotenv
load_dotenv(dotenv_path=env_path)

from config import settings
from evaluation.evaluator import Evaluator


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def evaluate_file(input_path: str):
    """评估单个 Markdown 报告文件"""
    evaluator = Evaluator(
        max_attempts=settings.eval_max_attempts,
        retry_delay=settings.eval_retry_delay,
    )
    result = await evaluator.evaluate_file(input_path)

    if result.success:
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        print(f"  主题:           {result.topic}")
        print(f"  全面性:         {result.quality['comprehensiveness_score']}/4")
        print(f"  连贯性:         {result.quality['coherence_score']}/4")
        print(f"  清晰度:         {result.quality['clarity_score']}/4")
        print(f"  洞察力:         {result.quality['insightfulness_score']}/4")
        print(f"  总体评分:       {result.quality['overall_score']}/4")
        if result.avg_repeat_score is not None:
            print(f"  重复性评分:     {result.avg_repeat_score}/4")
        if result.fact is not None:
            print(f"  事实核查:       支持 {result.fact['supported']}/{result.fact['success_checks']}, "
                  f"不确定 {result.fact['uncertain']}, 不支持 {result.fact['not_supported']}")
            if result.fact_score is not None:
                print(f"  事实支持率:     {result.fact_score:.2%}")
        print(f"  评估理由:       {result.quality['reason'][:200]}...")
        print("=" * 60)

        # 保存结果
        output_dir = Path(settings.eval_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{result.file_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_file}")
    else:
        print(f"\n评估失败: {result.error}")


async def evaluate_batch(input_path: str, output_dir: str, resume: bool):
    """批量评估 JSONL 文件"""
    evaluator = Evaluator(
        max_attempts=settings.eval_max_attempts,
        retry_delay=settings.eval_retry_delay,
    )
    results = await evaluator.evaluate_batch(input_path, output_dir, resume=resume)

    # 统计结果
    success_count = sum(1 for r in results if r.success)
    fail_count = len(results) - success_count

    print("\n" + "=" * 60)
    print("批量评估完成")
    print("=" * 60)
    print(f"  总数: {len(results)}")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")

    if success_count > 0:
        avg_scores = {
            "comprehensiveness": 0,
            "coherence": 0,
            "clarity": 0,
            "insightfulness": 0,
            "overall": 0,
            "repeat": 0,
            "fact_score": 0,
        }
        repeat_count = 0
        fact_count = 0
        for r in results:
            if r.success and r.quality:
                avg_scores["comprehensiveness"] += r.quality["comprehensiveness_score"] or 0
                avg_scores["coherence"] += r.quality["coherence_score"] or 0
                avg_scores["clarity"] += r.quality["clarity_score"] or 0
                avg_scores["insightfulness"] += r.quality["insightfulness_score"] or 0
                avg_scores["overall"] += r.quality["overall_score"] or 0
                if r.avg_repeat_score is not None:
                    avg_scores["repeat"] += r.avg_repeat_score
                    repeat_count += 1
                if r.fact_score is not None:
                    avg_scores["fact_score"] += r.fact_score
                    fact_count += 1

        for k in avg_scores:
            if k == "repeat":
                avg_scores[k] = round(avg_scores[k] / repeat_count, 2) if repeat_count > 0 else None
            elif k == "fact_score":
                avg_scores[k] = round(avg_scores[k] / fact_count, 4) if fact_count > 0 else None
            else:
                avg_scores[k] = round(avg_scores[k] / success_count, 2)

        print(f"\n  平均分数:")
        print(f"    全面性:     {avg_scores['comprehensiveness']}/4")
        print(f"    连贯性:     {avg_scores['coherence']}/4")
        print(f"    清晰度:     {avg_scores['clarity']}/4")
        print(f"    洞察力:     {avg_scores['insightfulness']}/4")
        print(f"    总体:       {avg_scores['overall']}/4")
        if avg_scores["repeat"] is not None:
            print(f"    重复性:     {avg_scores['repeat']}/4")
        if avg_scores["fact_score"] is not None:
            print(f"    事实支持率: {avg_scores['fact_score']:.2%}")

    print(f"\n结果保存目录: {output_dir}")
    print("=" * 60)


async def evaluate_interactive():
    """交互式评估"""
    evaluator = Evaluator(
        max_attempts=settings.eval_max_attempts,
        retry_delay=settings.eval_retry_delay,
    )

    print("\n交互式报告评估（输入 q 退出）")
    print("请先输入研究主题，再输入报告内容（以空行结束）:")

    while True:
        try:
            topic = input("\n研究主题 > ").strip()
            if topic.lower() == "q":
                print("再见！")
                break
            if not topic:
                continue

            print("报告内容（输入空行结束）:")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            report = "\n".join(lines)

            if not report.strip():
                print("报告内容为空，请重新输入")
                continue

            result = await evaluator.evaluate_one(topic, report)

            if result.success:
                print("\n--- 评估结果 ---")
                print(f"  全面性:     {result.quality['comprehensiveness_score']}/4")
                print(f"  连贯性:     {result.quality['coherence_score']}/4")
                print(f"  清晰度:     {result.quality['clarity_score']}/4")
                print(f"  洞察力:     {result.quality['insightfulness_score']}/4")
                print(f"  总体:       {result.quality['overall_score']}/4")
                if result.avg_repeat_score is not None:
                    print(f"  重复性:     {result.avg_repeat_score}/4")
                if result.fact is not None:
                    print(f"  事实核查:   支持 {result.fact['supported']}/{result.fact['success_checks']}, "
                          f"不确定 {result.fact['uncertain']}, 不支持 {result.fact['not_supported']}")
                    if result.fact_score is not None:
                        print(f"  事实支持率: {result.fact_score:.2%}")
                print(f"  理由:       {result.quality['reason'][:300]}")
            else:
                print(f"\n评估失败: {result.error}")

        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="DeepResearch 报告评估系统")
    parser.add_argument(
        "--mode",
        choices=["file", "batch", "interactive"],
        default="interactive",
        help="评估模式: file=单文件, batch=批量, interactive=交互式",
    )
    parser.add_argument("--input", type=str, help="输入文件路径")
    parser.add_argument("--output", type=str, help="输出目录（batch 模式）")
    parser.add_argument("--resume", action="store_true", help="断点续传（batch 模式）")

    args = parser.parse_args()

    print("=" * 60)
    print("  DeepResearch 评估系统")
    print("=" * 60)
    print(f"  评估模式: {args.mode}")
    print(f"  LLM:      {settings.evaluator_llm_provider}")
    print("=" * 60)

    if args.mode == "file":
        if not args.input:
            print("错误: file 模式需要 --input 参数")
            sys.exit(1)
        asyncio.run(evaluate_file(args.input))

    elif args.mode == "batch":
        if not args.input:
            print("错误: batch 模式需要 --input 参数")
            sys.exit(1)
        output_dir = args.output or settings.eval_output_dir
        asyncio.run(evaluate_batch(args.input, output_dir, args.resume))

    elif args.mode == "interactive":
        asyncio.run(evaluate_interactive())


if __name__ == "__main__":
    main()
