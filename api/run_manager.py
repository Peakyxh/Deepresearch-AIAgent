import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import uuid4

from workflows.research_workflow import ResearchWorkflow


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunRecord:
    id: str
    query: str
    session_id: str
    status: str = "queued"
    current_phase: str = "queued"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    state: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    pending_interaction: dict[str, Any] | None = None
    interaction_future: asyncio.Future | None = None
    task: asyncio.Task | None = None


class InteractionGateway:
    """Bridges blocking workflow decisions to resumable HTTP interactions."""

    def __init__(self, manager: "RunManager", run_id: str):
        self.manager = manager
        self.run_id = run_id

    async def request(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = self.manager.require(self.run_id)
        interaction_id = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        record.interaction_future = future
        record.pending_interaction = {
            "id": interaction_id,
            "kind": kind,
            "payload": payload,
        }
        record.status = "waiting_input"
        record.updated_at = utc_now()
        self.manager.publish_nowait(
            self.run_id,
            {
                "type": "interaction_required",
                "interaction_id": interaction_id,
                "kind": kind,
                "payload": payload,
            },
        )
        try:
            return await future
        finally:
            if (
                record.pending_interaction
                and record.pending_interaction.get("id") == interaction_id
            ):
                record.pending_interaction = None
            record.interaction_future = None


class RunManager:
    """In-memory task manager with replayable event streams.

    It deliberately keeps persistence out of the first slice. The event contract is
    stable so Redis/PostgreSQL-backed storage can replace this implementation later.
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def require(self, run_id: str) -> RunRecord:
        record = self._runs.get(run_id)
        if not record:
            raise KeyError(run_id)
        return record

    async def create_run(self, query: str, session_id: str | None = None) -> RunRecord:
        run_id = f"run_{uuid4().hex[:12]}"
        record = RunRecord(
            id=run_id,
            query=query.strip(),
            session_id=session_id or run_id,
        )
        self._runs[run_id] = record
        self.publish_nowait(run_id, {"type": "run_queued", "query": record.query})
        record.task = asyncio.create_task(self._execute(record), name=run_id)
        return record

    async def _execute(self, record: RunRecord) -> None:
        record.status = "running"
        record.current_phase = "initializing"
        record.updated_at = utc_now()
        self.publish_nowait(record.id, {"type": "run_started"})

        gateway = InteractionGateway(self, record.id)

        def on_workflow_event(event: dict[str, Any]) -> None:
            self.publish_nowait(record.id, event)

        try:
            workflow = ResearchWorkflow(
                session_id=record.session_id,
                interaction_handler=gateway,
                event_callback=on_workflow_event,
            )
            result = await workflow.run(record.query)
            record.state = result.model_dump(mode="json")
            record.error = result.error
            if result.error:
                record.status = "failed"
                record.current_phase = "failed"
                self.publish_nowait(
                    record.id, {"type": "run_failed", "error": result.error}
                )
            else:
                record.status = "completed"
                record.current_phase = "completed"
                self.publish_nowait(
                    record.id,
                    {
                        "type": "run_completed",
                        "report": result.report,
                        "source_count": len(result.sources),
                    },
                )
        except asyncio.CancelledError:
            record.status = "cancelled"
            record.current_phase = "cancelled"
            self.publish_nowait(record.id, {"type": "run_cancelled"})
        except Exception as exc:  # API boundary: convert failures into observable run state
            record.status = "failed"
            record.current_phase = "failed"
            record.error = str(exc)
            self.publish_nowait(
                record.id, {"type": "run_failed", "error": str(exc)}
            )
        finally:
            record.updated_at = utc_now()

    def publish_nowait(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = self.require(run_id)
        event_type = str(payload.get("type", "event"))
        data = {key: value for key, value in payload.items() if key != "type"}

        if event_type == "phase_started":
            record.current_phase = str(data.get("phase", record.current_phase))
            if record.status != "waiting_input":
                record.status = "running"
        elif event_type == "phase_completed":
            phase = str(data.get("phase", ""))
            record.current_phase = phase or record.current_phase
            if phase == "planner":
                record.state["research_plan"] = data.get("research_plan", "")
                record.state["structured_sub_questions"] = data.get(
                    "sub_questions", []
                )
            elif phase == "research":
                record.state["sub_question_results"] = data.get(
                    "sub_question_results", []
                )
                record.state["source_count"] = data.get("source_count", 0)
            elif phase == "critic":
                record.state["critique_total_score"] = data.get("score", 0)
                record.state["critique_passed"] = data.get("passed", False)
                record.state["critique_feedback"] = data.get("feedback", "")
                record.state["critique_outcome"] = data.get("outcome", "")
                record.state["writer_revision_suggestions"] = data.get(
                    "writer_revision_suggestions", []
                )
            elif phase == "writer":
                record.state["report"] = data.get("report", "")
        elif event_type == "interaction_resolved":
            record.status = "running"
        elif event_type == "token_usage":
            usage = record.state.setdefault(
                "token_usage",
                {
                    "estimated": True,
                    "by_agent": {},
                    "total_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            )
            agent = str(data.get("agent", "unknown"))
            bucket = usage["by_agent"].setdefault(
                agent,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0},
            )
            input_tokens = int(data.get("input_tokens", 0) or 0)
            output_tokens = int(data.get("output_tokens", 0) or 0)
            bucket["calls"] += 1
            bucket["input_tokens"] += input_tokens
            bucket["output_tokens"] += output_tokens
            usage["total_calls"] += 1
            usage["input_tokens"] += input_tokens
            usage["output_tokens"] += output_tokens

        event = {
            "id": len(record.events) + 1,
            "type": event_type,
            "created_at": utc_now(),
            "data": data,
        }
        record.events.append(event)
        record.updated_at = event["created_at"]
        for subscriber in tuple(record.subscribers):
            try:
                subscriber.put_nowait(event)
            except asyncio.QueueFull:
                record.subscribers.discard(subscriber)
        return event

    async def respond(self, run_id: str, interaction_id: str, response: dict[str, Any]) -> None:
        record = self.require(run_id)
        pending = record.pending_interaction
        if not pending or pending.get("id") != interaction_id:
            raise ValueError("interaction is no longer pending")
        future = record.interaction_future
        if not future or future.done():
            raise ValueError("interaction cannot be resumed")

        record.pending_interaction = None
        record.status = "running"
        self.publish_nowait(
            run_id,
            {
                "type": "interaction_resolved",
                "interaction_id": interaction_id,
                "kind": pending.get("kind"),
            },
        )
        future.set_result(response)

    async def cancel(self, run_id: str) -> None:
        record = self.require(run_id)
        if record.status in TERMINAL_STATUSES:
            return
        if record.task and not record.task.done():
            record.task.cancel()

    def list_runs(self) -> list[RunRecord]:
        return sorted(self._runs.values(), key=lambda item: item.updated_at, reverse=True)

    async def stream_events(
        self, run_id: str, after_id: int = 0
    ) -> AsyncIterator[dict[str, Any] | None]:
        record = self.require(run_id)
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        record.subscribers.add(queue)
        try:
            for event in record.events:
                if event["id"] > after_id:
                    yield event
            if record.status in TERMINAL_STATUSES:
                return

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield event
                    if event["type"] in {
                        "run_completed",
                        "run_failed",
                        "run_cancelled",
                    }:
                        return
                except asyncio.TimeoutError:
                    yield None
        finally:
            record.subscribers.discard(queue)

    @staticmethod
    def summary(record: RunRecord) -> dict[str, Any]:
        state = record.state
        return {
            "id": record.id,
            "query": record.query,
            "status": record.status,
            "current_phase": record.current_phase,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "error": record.error,
            "source_count": len(state.get("sources", []))
            or int(state.get("source_count", 0) or 0),
            "critique_score": int(state.get("critique_total_score", 0) or 0),
        }

    @classmethod
    def detail(cls, record: RunRecord) -> dict[str, Any]:
        return {
            **cls.summary(record),
            "state": record.state,
            "pending_interaction": record.pending_interaction,
            "events": record.events[-300:],
        }


run_manager = RunManager()

