import asyncio
import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import uuid4

from config import settings
from workflows.research_workflow import ResearchWorkflow


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
logger = logging.getLogger(__name__)


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
    next_phase: str | None = "clarifier"
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
            "created_at": utc_now(),
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
        await self.manager.flush_persistence()
        try:
            return await future
        finally:
            if (
                record.pending_interaction
                and record.pending_interaction.get("id") == interaction_id
            ):
                record.pending_interaction = None
                self.manager.persist_run_nowait(record)
            record.interaction_future = None


class RunManager:
    """Task manager with a live cache and optional PostgreSQL persistence."""

    def __init__(
        self,
        repository: Any | None = None,
        queue: Any | None = None,
        *,
        execution_mode: str = "inline",
        recover_interrupted: bool = True,
    ) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._repository = repository
        self._queue = queue
        self._execution_mode = execution_mode
        self._recover_interrupted = recover_interrupted
        self._persistence_tasks: set[asyncio.Task] = set()
        self._persistence_tail: asyncio.Task | None = None
        self._persistence_errors: list[BaseException] = []
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized or self._repository is None:
            self._initialized = True
            return
        await self._repository.initialize()
        if self._queue is not None:
            await self._queue.initialize()
        if self._recover_interrupted and self._execution_mode == "inline":
            await self._repository.recover_interrupted_runs()
        for item in await self._repository.load_runs():
            record = self._record_from_item(item)
            self._runs[record.id] = record
        if self._execution_mode == "worker" and self._queue is not None:
            for run_id in await self._repository.list_dispatchable_run_ids():
                await self._queue.enqueue(run_id, "startup_recovery")
        self._initialized = True

    async def close(self) -> None:
        active_tasks = [] if self._execution_mode == "worker" else [
            record.task for record in self._runs.values()
            if record.task is not None and not record.task.done()
        ]
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        await self.flush_persistence()
        if self._queue is not None:
            await self._queue.close()
        if self._repository is not None:
            await self._repository.close()
        self._initialized = False

    @staticmethod
    def _persistence_record(record: RunRecord) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "id": record.id,
                "query": record.query,
                "session_id": record.session_id,
                "status": record.status,
                "current_phase": record.current_phase,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "state": record.state,
                "error": record.error,
                "next_phase": record.next_phase,
            }
        )

    @staticmethod
    def _record_from_item(item: dict[str, Any]) -> RunRecord:
        return RunRecord(
            id=item["id"],
            query=item["query"],
            session_id=item["session_id"],
            status=item["status"],
            current_phase=item["current_phase"],
            created_at=item["created_at"],
            updated_at=item["updated_at"],
            state=item.get("state", {}),
            error=item.get("error", ""),
            next_phase=item.get("next_phase"),
            events=item.get("events", []),
            pending_interaction=item.get("pending_interaction"),
        )

    async def refresh_run(self, run_id: str) -> RunRecord:
        if self._repository is None:
            return self.require(run_id)
        item = await self._repository.load_run(run_id)
        if item is None:
            raise KeyError(run_id)
        record = self._record_from_item(item)
        existing = self._runs.get(run_id)
        if existing is not None:
            record.subscribers = existing.subscribers
            record.task = existing.task
            record.interaction_future = existing.interaction_future
        self._runs[run_id] = record
        return record

    async def get_run(self, run_id: str) -> RunRecord:
        if self._execution_mode == "worker":
            return await self.refresh_run(run_id)
        return self.require(run_id)

    async def list_runs_async(self) -> list[RunRecord]:
        if self._execution_mode == "worker" and self._repository is not None:
            records = [
                self._record_from_item(item)
                for item in await self._repository.load_runs()
            ]
            self._runs = {record.id: record for record in records}
        return self.list_runs()

    def _schedule_persistence(self, operation: Any) -> None:
        if self._repository is None:
            return
        previous = self._persistence_tail

        async def ordered_operation() -> Any:
            if previous is not None:
                try:
                    await previous
                except Exception:
                    # The previous failure is reported by flush_persistence; later
                    # writes should still be attempted in sequence.
                    pass
            return await operation

        task = asyncio.get_running_loop().create_task(ordered_operation())
        self._persistence_tail = task
        self._persistence_tasks.add(task)

        def completed(done: asyncio.Task) -> None:
            self._persistence_tasks.discard(done)
            if not done.cancelled() and (error := done.exception()) is not None:
                self._persistence_errors.append(error)

        task.add_done_callback(completed)

    async def flush_persistence(self) -> None:
        while self._persistence_tasks:
            tasks = tuple(self._persistence_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._persistence_tail is not None and self._persistence_tail.done():
            self._persistence_tail = None
        if self._persistence_errors:
            error = self._persistence_errors.pop(0)
            self._persistence_errors.clear()
            raise error

    def persist_run_nowait(self, record: RunRecord) -> None:
        if self._repository is not None:
            self._schedule_persistence(
                self._repository.save_run(self._persistence_record(record))
            )

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
        if self._execution_mode == "worker":
            record.state = ResearchWorkflow.initial_state(record.query, record.session_id)
            record.next_phase = "clarifier"
        self._runs[run_id] = record
        try:
            if self._repository is not None:
                await self._repository.create_run(self._persistence_record(record))
            self.publish_nowait(run_id, {"type": "run_queued", "query": record.query})
            await self.flush_persistence()
        except Exception:
            self._runs.pop(run_id, None)
            raise
        if self._execution_mode == "worker":
            if self._queue is None:
                raise RuntimeError("worker execution mode requires Redis")
            await self._queue.enqueue(run_id, "create")
        else:
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
        except Exception as exc:
            record.status = "failed"
            record.current_phase = "failed"
            record.error = str(exc)
            self.publish_nowait(
                record.id, {"type": "run_failed", "error": str(exc)}
            )
        finally:
            record.updated_at = utc_now()
            self.persist_run_nowait(record)
            try:
                await self.flush_persistence()
            except Exception:
                logger.exception("Failed to persist terminal run state for %s", record.id)

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
                record.state["plan_coverage"] = data.get("plan_coverage", [])
                record.state["plan_assumptions"] = data.get("plan_assumptions", [])
                record.state["planner_self_check"] = data.get(
                    "planner_self_check", {}
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
        if self._repository is not None:
            self._schedule_persistence(
                self._repository.append_event(
                    self._persistence_record(record),
                    copy.deepcopy(event),
                    copy.deepcopy(record.pending_interaction),
                )
            )
        for subscriber in tuple(record.subscribers):
            try:
                subscriber.put_nowait(event)
            except asyncio.QueueFull:
                record.subscribers.discard(subscriber)
        return event

    async def respond(
        self, run_id: str, interaction_id: str, response: dict[str, Any]
    ) -> None:
        if self._execution_mode == "worker":
            if self._repository is None or self._queue is None:
                raise RuntimeError("worker execution mode is not initialized")
            await self._repository.resolve_interaction_for_resume(
                run_id, interaction_id, response
            )
            await self._queue.enqueue(run_id, "interaction_resolved")
            await self.refresh_run(run_id)
            return
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
        if self._repository is not None:
            self._schedule_persistence(
                self._repository.resolve_interaction(run_id, interaction_id, response)
            )
            await self.flush_persistence()
        future.set_result(response)

    async def cancel(self, run_id: str) -> None:
        record = await self.get_run(run_id)
        if record.status in TERMINAL_STATUSES:
            return
        if self._execution_mode == "worker":
            if self._repository is None:
                raise RuntimeError("worker execution mode is not initialized")
            await self._repository.request_cancel(run_id)
            await self.refresh_run(run_id)
            return
        if record.task and not record.task.done():
            record.task.cancel()
            await asyncio.sleep(0)
            await self.flush_persistence()

    def list_runs(self) -> list[RunRecord]:
        return sorted(self._runs.values(), key=lambda item: item.updated_at, reverse=True)

    async def stream_events(
        self, run_id: str, after_id: int = 0
    ) -> AsyncIterator[dict[str, Any] | None]:
        if self._execution_mode == "worker" and self._repository is not None:
            cursor = after_id
            idle_ticks = 0
            while True:
                events, status = await self._repository.load_events_after(run_id, cursor)
                for event in events:
                    cursor = max(cursor, int(event["id"]))
                    yield event
                if status in TERMINAL_STATUSES:
                    return
                if not events:
                    idle_ticks += 1
                    if idle_ticks >= 15:
                        idle_ticks = 0
                        yield None
                else:
                    idle_ticks = 0
                await asyncio.sleep(1)
            return
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


def _build_repository() -> Any | None:
    if not settings.database_url:
        return None
    from api.persistence import PostgresRunRepository

    return PostgresRunRepository(
        settings.database_url,
        echo=settings.database_echo,
        auto_create=settings.database_auto_create,
    )


def _build_queue() -> Any | None:
    if settings.task_execution_mode != "worker":
        return None
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required when TASK_EXECUTION_MODE=worker")
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required when TASK_EXECUTION_MODE=worker")
    from worker.queue import RedisTaskQueue

    return RedisTaskQueue(
        settings.redis_url,
        stream=settings.redis_stream_name,
        group=settings.redis_consumer_group,
    )


run_manager = RunManager(
    _build_repository(),
    _build_queue(),
    execution_mode=settings.task_execution_mode,
    recover_interrupted=settings.database_recover_interrupted_runs,
)
