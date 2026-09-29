"""
报告转 JSONL 工具

将 outputs 目录下的研究报告（Markdown 格式）转换为评估用的 JSONL 文件。
JSONL 格式：{"id": 1, "topic": "研究问题", "report": "#研究报告原文"}

用法：
  python evaluation/convert_to_jsonl.py
  python evaluation/convert_to_jsonl.py --input ./outputs --output ./evaluation/data/reports.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path


def extract_topic(content: str) -> str:
    """
    从报告内容中提取研究问题

    支持以下格式：
    - > 研究问题：xxx
    - > 研究问题: xxx
    - # Research Report: xxx  （作为备选）
    """
    # 优先匹配 "研究问题：xxx" 或 "研究问题: xxx"
    match = re.search(r"研究问题[：:]\s*(.+)", content)
    if match:
        return match.group(1).strip()

    # 备选：匹配一级标题 "Research Report: xxx" 或 "研究报告：xxx"
    match = re.search(r"^#\s*(?:Research Report|研究报告)[：:]\s*(.+)", content, re.MULTILINE)
    if match:
        return match.group(1).strip()

    return ""


def extract_report_body(content: str) -> str:
    """
    提取报告正文，从第一个 # 标题开始，去掉开头的元信息块

    元信息块格式：
    > 📋 **DeepResearch Agent 研究报告**
    > 研究问题：xxx
    > 生成时间：xxx
    > ---
    """
    match = re.search(r"^#", content, re.MULTILINE)
    if match:
        return content[match.start():].strip()
    return content.strip()


def convert_reports(input_dir: str, output_path: str) -> int:
    """
    将目录下的所有 Markdown 报告转换为 JSONL 文件

    Args:
        input_dir: 报告目录路径
        output_path: 输出 JSONL 文件路径

    Returns:
        转换的报告数量
    """
    input_path = Path(input_dir)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"错误: 输入目录不存在 - {input_path}")
        return 0

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_files = sorted(input_path.glob("*.md"))
    if not md_files:
        print(f"警告: 在 {input_path} 下未找到 Markdown 文件")
        return 0

    records = []
    skipped = 0

    for idx, md_file in enumerate(md_files, start=1):
        content = md_file.read_text(encoding="utf-8")
        topic = extract_topic(content)

        if not topic:
            print(f"  跳过 {md_file.name}: 未找到研究问题")
            skipped += 1
            continue

        records.append({
            "id": idx,
            "topic": topic,
            "report": extract_report_body(content),
        })
        print(f"  [{idx}] {md_file.name} -> topic: {topic[:60]}...")

    # 写入 JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n转换完成: {len(records)} 篇报告, 跳过 {skipped} 篇")
    print(f"输出文件: {output_path}")

    return len(records)


def main():
    parser = argparse.ArgumentParser(description="将研究报告转换为评估用 JSONL 文件")
    parser.add_argument(
        "--input", type=str, default="../outputs",
        help="报告目录路径（默认: ../outputs）",
    )
    parser.add_argument(
        "--output", type=str, default="./data/reports.jsonl",
        help="输出 JSONL 文件路径（默认: ./data/reports.jsonl）",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  报告转 JSONL 工具")
    print("=" * 50)
    print(f"  输入目录: {args.input}")
    print(f"  输出文件: {args.output}")
    print()

    count = convert_reports(args.input, args.output)
    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
