"""Offline tests for task events and human-in-the-loop resume semantics."""

import asyncio

from fastapi.testclient import TestClient

from api.main import app
from api.run_manager import InteractionGateway, RunManager, RunRecord


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
