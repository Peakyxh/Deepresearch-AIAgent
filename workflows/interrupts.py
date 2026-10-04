"""Control-flow signals used by durable workflow execution."""


class WorkflowPaused(Exception):
    """Raised after a durable human interaction has been created."""

    def __init__(self, interaction_id: str, kind: str):
        super().__init__(f"workflow paused for {kind}: {interaction_id}")
        self.interaction_id = interaction_id
        self.kind = kind
