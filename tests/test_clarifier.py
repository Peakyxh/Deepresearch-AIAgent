import asyncio
import json
import logging
import sys
import types
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
from agents.clarifier_agent import ClarifierAgent
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(message)s",
)


async def main():
    llm = LLMFactory.create()

    clarifier = ClarifierAgent(llm=llm)

    initial_state = {
        "query": "在当前中国房地产市场低迷的情况下，政府税收减少，这会多大程度上影响地方政府的财政收入",
        "session_id": "test123",
        "clarified_intent": "",
        "clarification_qa": [],
        "history_context": "",
        "sub_questions": [],
        "structured_sub_questions": [],
        "search_keywords": [],
        "research_plan": "",
        "orchestrator_plan": {},
        "sub_question_results": [],
        "search_results": [],
        "paper_results": [],
        "findings": "",
        "sources": [],
        "critique_passed": False,
        "critique_feedback": "",
        "critique_feedback_for_researcher": "",
        "revision_suggestions": [],
        "critique_scores": {},
        "critique_score_details": {},
        "critique_total_score": 0,
        "per_sub_question_critiques": {},
        "failed_sub_question_ids": [],
        "overall_coherence_score": 0,
        "overall_coherence_feedback": "",
        "overall_coherence_passed": False,
        "report": "",
        "current_step": "planner",
        "retry_count": 0,
        "max_retries": 2,
        "use_orchestrator": settings.orchestrator_enabled,
        "error": "",
    }

    state = await clarifier.run(initial_state)

    print("\n" + "=" * 60)
    print("Clarifier 测试结果")
    print("=" * 60)
    print(f"current_step: {state.get('current_step')}")
    print(f"clarified_intent: {state.get('clarified_intent')}")
    print(f"clarification_qa: {json.dumps(state.get('clarification_qa', []), ensure_ascii=False, indent=2)}")
    print(f"error: {state.get('error', '')}")

    output_path = Path(__file__).parent / "output_clarifier_state.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("state = {\n")
        for key, value in state.items():
            value_str = repr(value)
            f.write(f'    "{key}": {value_str},\n')
        f.write("}\n")

    print(f"\nstate 已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
