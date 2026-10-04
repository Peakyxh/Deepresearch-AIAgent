"""Durable research worker service."""

from __future__ import annotations

import asyncio
import logging
import socket
from contextlib import suppress
from typing import Any
from uuid import uuid4

from api.persistence import PostgresRunRepository
from workflows.interrupts import WorkflowPaused
from workflows.research_workflow import ResearchWorkflow
from workflows.state import ResearchState
from worker.queue import QueuedRun, RedisTaskQueue


logger = logging.getLogger(__name__)


def hydrate_state(state: dict[str, Any]) -> dict[str, Any]:
    """Restore nested Pydantic objects expected by existing agents."""
    return dict(ResearchState(**state).__dict__)


def serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert a live workflow state into a JSON-safe checkpoint."""
    return ResearchState(**state).model_dump(mode="json")


class WorkerEventPublisher:
    """Persist synchronous workflow callbacks without blocking agent code."""

    def __init__(self, repository: PostgresRunRepository, run_id: str) -> None:
        self.repository = repository
        self.run_id = run_id
        self._tail: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()

    def publish(self, payload: dict[str, Any]) -> None:
        previous = self._tail

        async def write() -> None:
            if previous is not None:
                await previous
            await self.repository.append_worker_event(self.run_id, payload)

        task = asyncio.create_task(write())
        self._tail = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def flush(self) -> None:
        if self._tail is not None:
            await self._tail
        self._tail = None


class DurableInteractionGateway:
    def __init__(self, repository: PostgresRunRepository, run_id: str) -> None:
        self.repository = repository
        self.run_id = run_id
        self.consumed_interaction_ids: list[str] = []

    async def request(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        resolved = await self.repository.get_resolved_interaction(
            self.run_id, kind, self.consumed_interaction_ids
        )
        if resolved is not None:
            interaction_id, response = resolved
            self.consumed_interaction_ids.append(interaction_id)
            return response
        interaction_id = uuid4().hex
        await self.repository.create_interaction_and_pause(
            self.run_id, interaction_id, kind, payload
        )
        raise WorkflowPaused(interaction_id, kind)

    async def commit_consumed(self) -> None:
        await self.repository.mark_interactions_consumed(
            self.consumed_interaction_ids
        )
        self.consumed_interaction_ids.clear()


class ResearchWorker:
    def __init__(
        self,
        repository: PostgresRunRepository,
        queue: RedisTaskQueue,
        *,
        worker_id: str | None = None,
        lease_seconds: int = 120,
    ) -> None:
        self.repository = repository
        self.queue = queue
        self.worker_id = worker_id or f"{socket.gethostname()}-{uuid4().hex[:8]}"
        self.lease_seconds = lease_seconds
        self._stopping = False

    async def initialize(self) -> None:
        await self.repository.initialize()
        await self.queue.initialize()

    async def close(self) -> None:
        self._stopping = True
        await self.queue.close()
        await self.repository.close()

    async def run_forever(self) -> None:
        logger.info("Research worker %s started", self.worker_id)
        while not self._stopping:
            try:
                item = await self.queue.reserve(
                    self.worker_id,
                    reclaim_idle_ms=self.lease_seconds * 1000,
                )
                if item is None:
                    # PostgreSQL is the source of truth. This sweep repairs the
                    # narrow window where a DB commit succeeded but Redis enqueue
                    # failed during create/resume.
                    for run_id in await self.repository.list_dispatchable_run_ids():
                        await self.queue.enqueue(run_id, "worker_recovery")
                    continue
                handled = await self.process(item)
                if handled:
                    await self.queue.acknowledge(item)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Worker loop failed; retrying")
                await asyncio.sleep(2)

    async def _heartbeat(self, item: QueuedRun) -> None:
        interval = max(5, self.lease_seconds // 3)
        while True:
            await asyncio.sleep(interval)
            renewed = await self.repository.renew_lease(
                item.run_id, self.worker_id, self.lease_seconds
            )
            if not renewed:
                return
            await self.queue.touch(item, self.worker_id, self.lease_seconds)

    async def process(self, item: QueuedRun) -> bool:
        claimed = await self.repository.claim_run(
            item.run_id, self.worker_id, self.lease_seconds
        )
        if not claimed:
            return True

        heartbeat = asyncio.create_task(self._heartbeat(item))
        publisher = WorkerEventPublisher(self.repository, item.run_id)
        try:
            execution = await self.repository.load_execution(item.run_id)
            raw_state = execution["state"] or ResearchWorkflow.initial_state(
                execution["query"], execution["session_id"]
            )
            state = hydrate_state(raw_state)
            phase = execution["next_phase"] or "clarifier"
            gateway = DurableInteractionGateway(self.repository, item.run_id)
            workflow = ResearchWorkflow(
                session_id=execution["session_id"],
                interaction_handler=gateway,
                event_callback=publisher.publish,
            )
            if execution["checkpoint_version"] == 0:
                publisher.publish({"type": "run_started"})

            while phase:
                if await self.repository.is_cancel_requested(item.run_id):
                    await self.repository.cancel_running_run(item.run_id)
                    await publisher.flush()
                    return True
                current_phase = phase
                state, phase = await workflow.execute_phase(state, current_phase)
                await publisher.flush()
                checkpoint_state = serialize_state(state)
                if phase is None:
                    await self.repository.save_checkpoint(
                        item.run_id, checkpoint_state, current_phase, None, "completed"
                    )
                    await gateway.commit_consumed()
                    await self.repository.complete_run(item.run_id, checkpoint_state)
                    return True
                await self.repository.save_checkpoint(
                    item.run_id, checkpoint_state, current_phase, phase, "running"
                )
                await gateway.commit_consumed()
                state = hydrate_state(checkpoint_state)
                await self.repository.renew_lease(
                    item.run_id, self.worker_id, self.lease_seconds
                )
            return True
        except WorkflowPaused as pause:
            await publisher.flush()
            execution = await self.repository.load_execution(item.run_id)
            await self.repository.save_checkpoint(
                item.run_id,
                serialize_state(state),
                None,
                phase,
                "waiting_input",
            )
            await gateway.commit_consumed()
            logger.info(
                "Run %s paused for %s (%s)", item.run_id, pause.kind, pause.interaction_id
            )
            return True
        except Exception as exc:
            logger.exception("Run %s failed in worker", item.run_id)
            try:
                await publisher.flush()
                await self.repository.fail_run(item.run_id, str(exc))
                return True
            except Exception:
                logger.exception("Could not persist failure for run %s", item.run_id)
                return False
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            await self.repository.release_claim(item.run_id, self.worker_id)
