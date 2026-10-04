"""Tests for robust report outline parsing and presentation."""

from agents.writer_agent import WriterAgent


def _writer() -> WriterAgent:
    return WriterAgent.__new__(WriterAgent)


def test_parse_outline_repairs_truncated_json() -> None:
    response = """```json
    {
      "title": "Agent 框架研究",
      "summary_points": ["框架各有侧重"],
      "sections": [
        {"heading": "架构对比", "key_arguments": ["状态管理方式不同"]}
      ]
    ```"""

    parsed = _writer()._parse_json_response(response)

    assert parsed is not None
    assert parsed["title"] == "Agent 框架研究"
    assert parsed["sections"][0]["key_arguments"] == ["状态管理方式不同"]


def test_format_outline_for_display_contains_actual_arguments() -> None:
    display = _writer()._format_outline_for_display(
        {
            "title": "Agent 框架研究",
            "summary_points": ["框架各有侧重"],
            "sections": [
                {"heading": "架构对比", "key_arguments": ["状态管理方式不同"]}
            ],
            "conclusion_points": ["按场景选择框架"],
        }
    )

    assert "标题: Agent 框架研究" in display
    assert "架构对比" in display
    assert "状态管理方式不同" in display
    assert "按场景选择框架" in display
