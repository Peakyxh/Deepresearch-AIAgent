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
from agents.planner_agent import PlannerAgent
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(message)s",
)


def load_state_from_txt(txt_path: Path) -> dict:
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    local_ns: dict = {}
    exec(content, {}, local_ns)
    return local_ns["state"]


async def main():
    llm = LLMFactory.create()

    planner = PlannerAgent(llm=llm)

    clarifier_output_path = Path(__file__).parent / "output_clarifier_state.txt"
    if not clarifier_output_path.exists():
        print(f"错误：找不到 {clarifier_output_path}，请先运行 test_clarifier.py")
        return

    clarifier_state = load_state_from_txt(clarifier_output_path)
    print(f"已从 {clarifier_output_path.name} 加载 state")

    state = await planner.run(clarifier_state)

    output_path = Path(__file__).parent / "output_planner_state.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("state = {\n")
        for key, value in state.items():
            value_str = repr(value)
            f.write(f'    "{key}": {value_str},\n')
        f.write("}\n")

    print(f"\nstate 已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
