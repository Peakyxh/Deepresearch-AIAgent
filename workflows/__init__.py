from workflows.dag_engine import END, Workflow
from workflows.state import ResearchState, SubQuestion, SubQuestionResult

__all__ = ["END", "Workflow", "ResearchState", "SubQuestion", "SubQuestionResult", "ResearchWorkflow"]


def __getattr__(name: str):
    """Lazy-load the composed workflow to avoid agents/workflows import cycles."""
    if name == "ResearchWorkflow":
        from workflows.research_workflow import ResearchWorkflow

        return ResearchWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
