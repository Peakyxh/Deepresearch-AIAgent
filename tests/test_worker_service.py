"""Offline tests for independent worker checkpoint behavior."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import worker.service as service_module
from worker.queue import QueuedRun
from worker.service import DurableInteractionGateway, ResearchWorker
from workflows.interrupts import WorkflowPaused


class FakeQueue:
    def __init__(self) -> None:
        self.touched: list[str] = []

    async def touch(
        self, item: QueuedRun, consumer: str, lease_seconds: int = 120
    ) -> None:
        self.touched.append(f"{item.run_id}:{consumer}")


class FakeRepository:
    def __init__(self) -> None:
        self.checkpoints: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.completed_state: dict[str, Any] | None = None
        self.failed_error = ""
        self.released = False
        self.resolved: list[tuple[str, dict[str, Any]]] = []
        self.consumed: list[str] = []
        self.created_interactions: list[str] = []

    async def claim_run(self, run_id: str, worker_id: str, lease_seconds: int) -> bool:
        return True

    async def load_execution(self, run_id: str) -> dict[str, Any]:
        return {
            "id": run_id,
            "query": "test query",
            "session_id": "session",
            "status": "running",
            "state": {},
            "next_phase": "clarifier",
            "checkpoint_version": 0,
            "cancel_requested": False,
        }

    async def append_worker_event(self, run_id: str, payload: dict[str, Any]) -> dict:
        self.events.append(payload)
        return payload

    async def is_cancel_requested(self, run_id: str) -> bool:
        return False

    async def save_checkpoint(
        self,
        run_id: str,
        state: dict[str, Any],
        completed_phase: str | None,
        next_phase: str | None,
        status: str,
    ) -> int:
        self.checkpoints.append(
            {
                "state": state,
                "completed_phase": completed_phase,
                "next_phase": next_phase,
                "status": status,
            }
        )
        return len(self.checkpoints)

    async def complete_run(self, run_id: str, state: dict[str, Any]) -> None:
        self.completed_state = state

    async def fail_run(self, run_id: str, error: str) -> None:
        self.failed_error = error

    async def renew_lease(self, run_id: str, worker_id: str, lease_seconds: int) -> bool:
        return True

    async def release_claim(self, run_id: str, worker_id: str) -> None:
        self.released = True

    async def cancel_running_run(self, run_id: str) -> None:
        return None

    async def get_resolved_interaction(
        self, run_id: str, kind: str, exclude_ids: list[str]
    ) -> tuple[str, dict[str, Any]] | None:
        for item in self.resolved:
            if item[0] not in exclude_ids:
                return item
        return None

    async def mark_interactions_consumed(self, interaction_ids: list[str]) -> None:
        self.consumed.extend(interaction_ids)

    async def create_interaction_and_pause(
        self, run_id: str, interaction_id: str, kind: str, payload: dict[str, Any]
    ) -> None:
        self.created_interactions.append(interaction_id)


class CompletingWorkflow:
    def __init__(self, *, event_callback, **_: Any) -> None:
        self.event_callback = event_callback

    @staticmethod
    def initial_state(query: str, session_id: str) -> dict[str, Any]:
        return {"query": query, "session_id": session_id}

    async def execute_phase(
        self, state: dict[str, Any], phase: str
    ) -> tuple[dict[str, Any], None]:
        state["clarified_intent"] = "clear intent"
        self.event_callback({"type": "phase_completed", "phase": phase})
        return state, None


def test_worker_saves_checkpoint_and_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        repository = FakeRepository()
        queue = FakeQueue()
        monkeypatch.setattr(service_module, "ResearchWorkflow", CompletingWorkflow)
        worker = ResearchWorker(repository, queue, worker_id="worker-test")

        handled = await worker.process(QueuedRun("1-0", "run_test", "create"))

        assert handled is True
        assert repository.checkpoints[-1]["completed_phase"] == "clarifier"
        assert repository.checkpoints[-1]["status"] == "completed"
        assert repository.completed_state is not None
        assert repository.completed_state["clarified_intent"] == "clear intent"
        assert repository.released is True

    asyncio.run(scenario())


def test_durable_interaction_reuses_response_until_checkpoint() -> None:
    async def scenario() -> None:
        repository = FakeRepository()
        repository.resolved = [("interaction-1", {"feedback": "continue"})]
        gateway = DurableInteractionGateway(repository, "run_test")

        response = await gateway.request("outline_review", {"outline": "test"})
        assert response == {"feedback": "continue"}
        assert repository.consumed == []

        await gateway.commit_consumed()
        assert repository.consumed == ["interaction-1"]

        repository.resolved = []
        with pytest.raises(WorkflowPaused):
            await gateway.request("clarification", {"questions": []})
        assert len(repository.created_interactions) == 1

    asyncio.run(scenario())
