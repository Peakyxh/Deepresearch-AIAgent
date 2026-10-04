"""Offline tests for task events and human-in-the-loop resume semantics."""

import asyncio
from copy import deepcopy

from fastapi.testclient import TestClient

from api.main import app
from api.run_manager import InteractionGateway, RunManager, RunRecord


class RecordingRepository:
    def __init__(self, loaded_runs: list[dict] | None = None) -> None:
        self.loaded_runs = loaded_runs or []
        self.events: list[dict] = []
        self.interactions: list[dict] = []
        self.saved_runs: list[dict] = []
        self.initialized = False
        self.closed = False

    async def initialize(self) -> None:
        self.initialized = True

    async def close(self) -> None:
        self.closed = True

    async def recover_interrupted_runs(self) -> None:
        return None

    async def load_runs(self) -> list[dict]:
        return deepcopy(self.loaded_runs)

    async def create_run(self, record: dict) -> None:
        self.saved_runs.append(deepcopy(record))

    async def save_run(self, record: dict) -> None:
        self.saved_runs.append(deepcopy(record))

    async def append_event(
        self, record: dict, event: dict, pending_interaction: dict | None = None
    ) -> None:
        self.saved_runs.append(deepcopy(record))
        self.events.append(deepcopy(event))
        if pending_interaction:
            self.interactions.append(deepcopy(pending_interaction))

    async def resolve_interaction(
        self, run_id: str, interaction_id: str, response: dict
    ) -> None:
        self.interactions.append(
            {
                "run_id": run_id,
                "id": interaction_id,
                "response": deepcopy(response),
                "status": "resolved",
            }
        )


class RecordingQueue:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []

    async def enqueue(self, run_id: str, reason: str) -> bool:
        self.enqueued.append((run_id, reason))
        return True


def test_health_endpoint() -> None:
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_interaction_pauses_and_resumes() -> None:
    async def scenario() -> None:
        manager = RunManager()
        record = RunRecord(id="run_test", query="test", session_id="test")
        manager._runs[record.id] = record
        gateway = InteractionGateway(manager, record.id)

        request_task = asyncio.create_task(
            gateway.request(
                kind="clarification",
                payload={"questions": [{"question": "研究时间范围？"}]},
            )
        )
        await asyncio.sleep(0)

        assert record.status == "waiting_input"
        assert record.pending_interaction is not None
        interaction_id = record.pending_interaction["id"]

        await manager.respond(
            record.id,
            interaction_id,
            {"answers": [{"question": "研究时间范围？", "answer": "近三年"}]},
        )
        result = await request_task

        assert result["answers"][0]["answer"] == "近三年"
        assert record.status == "running"
        assert record.pending_interaction is None
        assert [event["type"] for event in record.events] == [
            "interaction_required",
            "interaction_resolved",
        ]

    asyncio.run(scenario())


def test_phase_events_update_read_model() -> None:
    manager = RunManager()
    record = RunRecord(id="run_test", query="test", session_id="test")
    manager._runs[record.id] = record

    manager.publish_nowait(
        record.id,
        {
            "type": "phase_completed",
            "phase": "planner",
            "research_plan": "plan",
            "sub_questions": [{"id": "sq_1", "question": "question"}],
        },
    )
    manager.publish_nowait(
        record.id,
        {
            "type": "phase_completed",
            "phase": "research",
            "source_count": 3,
            "sub_question_results": [],
        },
    )

    detail = manager.detail(record)
    assert detail["current_phase"] == "research"
    assert detail["state"]["research_plan"] == "plan"
    assert detail["state"]["structured_sub_questions"][0]["id"] == "sq_1"
    assert detail["source_count"] == 3


def test_token_events_update_live_usage_summary() -> None:
    manager = RunManager()
    record = RunRecord(id="run_test", query="test", session_id="test")
    manager._runs[record.id] = record

    manager.publish_nowait(
        record.id,
        {
            "type": "token_usage",
            "agent": "Critic",
            "input_tokens": 120,
            "output_tokens": 30,
        },
    )

    usage = record.state["token_usage"]
    assert usage["total_calls"] == 1
    assert usage["input_tokens"] == 120
    assert usage["output_tokens"] == 30
    assert usage["by_agent"]["Critic"]["calls"] == 1


def test_manager_loads_persisted_runs_on_startup() -> None:
    async def scenario() -> None:
        repository = RecordingRepository(
            [
                {
                    "id": "run_saved",
                    "query": "saved query",
                    "session_id": "saved-session",
                    "status": "completed",
                    "current_phase": "completed",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:01:00+00:00",
                    "state": {"report": "saved report"},
                    "error": "",
                    "events": [
                        {
                            "id": 1,
                            "type": "run_completed",
                            "created_at": "2026-01-01T00:01:00+00:00",
                            "data": {},
                        }
                    ],
                    "pending_interaction": None,
                }
            ]
        )
        manager = RunManager(repository)

        await manager.initialize()

        assert repository.initialized is True
        assert manager.require("run_saved").state["report"] == "saved report"
        assert manager.require("run_saved").events[0]["type"] == "run_completed"
        await manager.close()
        assert repository.closed is True

    asyncio.run(scenario())


def test_interaction_is_persisted_before_and_after_response() -> None:
    async def scenario() -> None:
        repository = RecordingRepository()
        manager = RunManager(repository)
        record = RunRecord(id="run_persisted", query="test", session_id="test")
        manager._runs[record.id] = record
        gateway = InteractionGateway(manager, record.id)

        request_task = asyncio.create_task(
            gateway.request("clarification", {"questions": [{"question": "范围？"}]})
        )
        await asyncio.sleep(0)
        await manager.flush_persistence()

        assert repository.events[0]["type"] == "interaction_required"
        assert repository.interactions[0]["kind"] == "clarification"
        interaction_id = record.pending_interaction["id"]

        await manager.respond(
            record.id,
            interaction_id,
            {"answers": [{"question": "范围？", "answer": "近三年"}]},
        )
        await request_task

        assert repository.events[-1]["type"] == "interaction_resolved"
        assert repository.interactions[-1]["status"] == "resolved"
        assert repository.interactions[-1]["response"]["answers"][0]["answer"] == "近三年"

    asyncio.run(scenario())


def test_worker_mode_enqueues_without_starting_in_process_task() -> None:
    async def scenario() -> None:
        repository = RecordingRepository()
        queue = RecordingQueue()
        manager = RunManager(
            repository,
            queue,
            execution_mode="worker",
            recover_interrupted=False,
        )

        record = await manager.create_run("durable research")

        assert record.task is None
        assert record.status == "queued"
        assert record.next_phase == "clarifier"
        assert record.state["query"] == "durable research"
        assert queue.enqueued == [(record.id, "create")]
        assert repository.events[0]["type"] == "run_queued"

    asyncio.run(scenario())
