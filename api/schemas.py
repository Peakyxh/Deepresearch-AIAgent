from typing import Any, Literal

from pydantic import BaseModel, Field


RunStatus = Literal[
    "queued",
    "running",
    "waiting_input",
    "completed",
    "failed",
    "cancelled",
]


class CreateRunRequest(BaseModel):
    query: str = Field(min_length=2, max_length=4000)
    session_id: str | None = Field(default=None, max_length=80)


class InteractionResponse(BaseModel):
    interaction_id: str
    answers: list[dict[str, str]] = Field(default_factory=list)
    feedback: str = Field(default="", max_length=12000)


class RunSummary(BaseModel):
    id: str
    query: str
    status: RunStatus
    current_phase: str
    created_at: str
    updated_at: str
    error: str = ""
    source_count: int = 0
    critique_score: int = 0


class RunDetail(RunSummary):
    state: dict[str, Any] = Field(default_factory=dict)
    pending_interaction: dict[str, Any] | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)

