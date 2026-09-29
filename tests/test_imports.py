import sys
sys.path.insert(0, '.')

print("Testing imports...")

try:
    from agents.planner_agent import PlannerAgent
    print("[OK] PlannerAgent imported")
except ImportError as e:
    if "circular" in str(e).lower() or "partially initialized" in str(e).lower():
        print(f"[FAIL] PlannerAgent CIRCULAR IMPORT: {e}")
    else:
        print(f"[SKIP] PlannerAgent (missing dep): {e}")
except Exception as e:
    print(f"[FAIL] PlannerAgent: {e}")

try:
    from agents.orchestrator_agent import OrchestratorAgent
    print("[OK] OrchestratorAgent imported")
except ImportError as e:
    if "circular" in str(e).lower() or "partially initialized" in str(e).lower():
        print(f"[FAIL] OrchestratorAgent CIRCULAR IMPORT: {e}")
    else:
        print(f"[SKIP] OrchestratorAgent (missing dep): {e}")
except Exception as e:
    print(f"[FAIL] OrchestratorAgent: {e}")

try:
    from agents.researcher_sub_agent import ResearcherSubAgent
    print("[OK] ResearcherSubAgent imported")
except ImportError as e:
    if "circular" in str(e).lower() or "partially initialized" in str(e).lower():
        print(f"[FAIL] ResearcherSubAgent CIRCULAR IMPORT: {e}")
    else:
        print(f"[SKIP] ResearcherSubAgent (missing dep): {e}")
except Exception as e:
    print(f"[FAIL] ResearcherSubAgent: {e}")

try:
    from workflows.research_workflow import ResearchWorkflow
    print("[OK] ResearchWorkflow imported")
except ImportError as e:
    if "circular" in str(e).lower() or "partially initialized" in str(e).lower():
        print(f"[FAIL] ResearchWorkflow CIRCULAR IMPORT: {e}")
    else:
        print(f"[SKIP] ResearchWorkflow (missing dep): {e}")
except Exception as e:
    print(f"[FAIL] ResearchWorkflow: {e}")

try:
    from agents.clarifier_agent import ClarifierAgent
    print("[OK] ClarifierAgent imported")
except ImportError as e:
    if "circular" in str(e).lower() or "partially initialized" in str(e).lower():
        print(f"[FAIL] ClarifierAgent CIRCULAR IMPORT: {e}")
    else:
        print(f"[SKIP] ClarifierAgent (missing dep): {e}")
except Exception as e:
    print(f"[FAIL] ClarifierAgent: {e}")

print("\nAll import tests done!")
