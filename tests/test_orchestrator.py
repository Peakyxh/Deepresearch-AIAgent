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
from agents.orchestrator_agent import OrchestratorAgent
from workflows.state import SubQuestion, SubQuestionResult
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(message)s",
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


async def main():
    llm = LLMFactory.create()

    orchestrator = OrchestratorAgent(llm=llm)

    planner_output_path = Path(__file__).parent / "output_planner_state.txt"
    if not planner_output_path.exists():
        print(f"错误：找不到 {planner_output_path}，请先运行 test_planner.py")
        return

    planner_state = load_state_from_txt(planner_output_path)
    print(f"已从 {planner_output_path.name} 加载 state")

    state = await orchestrator.run(planner_state)

    output_path = Path(__file__).parent / "output_orchestrator_state.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("state = {\n")
        for key, value in state.items():
            value_str = repr(value)
            f.write(f'    "{key}": {value_str},\n')
        f.write("}\n")

    print(f"\nstate 已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
