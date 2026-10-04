"""PostgreSQL persistence for API runs, events, and interactions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
    select,
    update,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


JsonType = JSON().with_variant(JSONB, "postgresql")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def normalize_database_url(url: str) -> str:
    """Normalize common PostgreSQL URLs to SQLAlchemy's asyncpg dialect."""
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    return url


class Base(DeclarativeBase):
    pass


class ResearchRunRow(Base):
    __tablename__ = "research_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    session_id: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    current_phase: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[dict[str, Any]] = mapped_column(JsonType, nullable=False, default=dict)
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")
    next_phase: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checkpoint_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    event_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cancel_requested: Mapped[bool] = mapped_column(nullable=False, default=False)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


class RunEventRow(Base):
    __tablename__ = "run_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_run_events_run_sequence"),
        Index("ix_run_events_run_sequence", "run_id", "sequence"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(JsonType, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RunInteractionRow(Base):
    __tablename__ = "run_interactions"
    __table_args__ = (
        Index("ix_run_interactions_run_status", "run_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JsonType, nullable=False, default=dict)
    response: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RunCheckpointRow(Base):
    __tablename__ = "run_checkpoints"
    __table_args__ = (
        UniqueConstraint("run_id", "version", name="uq_run_checkpoints_run_version"),
        Index("ix_run_checkpoints_run_version", "run_id", "version"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_phase: Mapped[str | None] = mapped_column(String(64), nullable=True)
    next_phase: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    state: Mapped[dict[str, Any]] = mapped_column(JsonType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PostgresRunRepository:
    """Async SQLAlchemy repository backed by PostgreSQL."""

    def __init__(self, database_url: str, *, echo: bool = False, auto_create: bool = True):
        self.engine: AsyncEngine = create_async_engine(
            normalize_database_url(database_url),
            echo=echo,
            pool_pre_ping=True,
        )
        self.sessions: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, expire_on_commit=False
        )
        self.auto_create = auto_create

    async def initialize(self) -> None:
        if self.auto_create:
            async with self.engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
                # create_all does not add columns to installations created by the
                # previous schema version, so keep this development migration
                # idempotent. Production should execute migrations/002.
                for statement in (
                    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS next_phase VARCHAR(64)",
                    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS checkpoint_version INTEGER NOT NULL DEFAULT 0",
                    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS event_sequence INTEGER NOT NULL DEFAULT 0",
                    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS cancel_requested BOOLEAN NOT NULL DEFAULT FALSE",
                    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS worker_id VARCHAR(128)",
                    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ",
                ):
                    await connection.execute(text(statement))
                await connection.execute(
                    text(
                        "UPDATE research_runs SET event_sequence = counts.max_sequence "
                        "FROM (SELECT run_id, COALESCE(MAX(sequence), 0) AS max_sequence "
                        "FROM run_events GROUP BY run_id) AS counts "
                        "WHERE research_runs.id = counts.run_id "
                        "AND research_runs.event_sequence < counts.max_sequence"
                    )
                )

    async def close(self) -> None:
        await self.engine.dispose()

    @staticmethod
    def _apply_run(row: ResearchRunRow, record: dict[str, Any]) -> None:
        row.query = record["query"]
        row.session_id = record["session_id"]
        row.status = record["status"]
        row.current_phase = record["current_phase"]
        row.state = record.get("state", {})
        row.error = record.get("error", "")
        if "next_phase" in record:
            row.next_phase = record.get("next_phase")
        row.created_at = _parse_datetime(record["created_at"])
        row.updated_at = _parse_datetime(record["updated_at"])

    async def create_run(self, record: dict[str, Any]) -> None:
        row = ResearchRunRow(id=record["id"])
        self._apply_run(row, record)
        async with self.sessions.begin() as session:
            session.add(row)

    async def save_run(self, record: dict[str, Any]) -> None:
        async with self.sessions.begin() as session:
            row = await session.get(ResearchRunRow, record["id"])
            if row is None:
                row = ResearchRunRow(id=record["id"])
                session.add(row)
            self._apply_run(row, record)

    async def append_event(
        self,
        record: dict[str, Any],
        event: dict[str, Any],
        pending_interaction: dict[str, Any] | None = None,
    ) -> None:
        async with self.sessions.begin() as session:
            run = await session.get(ResearchRunRow, record["id"])
            if run is None:
                run = ResearchRunRow(id=record["id"])
                session.add(run)
            self._apply_run(run, record)
            run.event_sequence = max(run.event_sequence or 0, int(event["id"]))

            existing_event = await session.scalar(
                select(RunEventRow.id).where(
                    RunEventRow.run_id == record["id"],
                    RunEventRow.sequence == event["id"],
                )
            )
            if existing_event is None:
                session.add(
                    RunEventRow(
                        run_id=record["id"],
                        sequence=event["id"],
                        type=event["type"],
                        data=event.get("data", {}),
                        created_at=_parse_datetime(event["created_at"]),
                    )
                )

            if pending_interaction is not None:
                interaction = await session.get(
                    RunInteractionRow, pending_interaction["id"]
                )
                if interaction is None:
                    session.add(
                        RunInteractionRow(
                            id=pending_interaction["id"],
                            run_id=record["id"],
                            kind=pending_interaction["kind"],
                            payload=pending_interaction.get("payload", {}),
                            response=None,
                            status="pending",
                            created_at=_parse_datetime(
                                pending_interaction.get("created_at", event["created_at"])
                            ),
                        )
                    )

            if record["status"] in {"completed", "failed", "cancelled"}:
                await session.execute(
                    update(RunInteractionRow)
                    .where(
                        RunInteractionRow.run_id == record["id"],
                        RunInteractionRow.status == "pending",
                    )
                    .values(status="abandoned", resolved_at=_utc_now())
                )

    async def resolve_interaction(
        self, run_id: str, interaction_id: str, response: dict[str, Any]
    ) -> None:
        async with self.sessions.begin() as session:
            result = await session.execute(
                update(RunInteractionRow)
                .where(
                    RunInteractionRow.id == interaction_id,
                    RunInteractionRow.run_id == run_id,
                    RunInteractionRow.status == "pending",
                )
                .values(
                    status="resolved", response=response, resolved_at=_utc_now()
                )
            )
            if result.rowcount != 1:
                raise ValueError("interaction is no longer pending")

    @staticmethod
    def _append_locked_event(
        session: AsyncSession,
        run: ResearchRunRow,
        event_type: str,
        data: dict[str, Any],
        *,
        created_at: datetime | None = None,
    ) -> dict[str, Any]:
        now = created_at or _utc_now()
        run.event_sequence = int(run.event_sequence or 0) + 1
        run.updated_at = now
        session.add(
            RunEventRow(
                run_id=run.id,
                sequence=run.event_sequence,
                type=event_type,
                data=data,
                created_at=now,
            )
        )
        return {
            "id": run.event_sequence,
            "type": event_type,
            "created_at": _iso(now),
            "data": data,
        }

    async def append_worker_event(
        self, run_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        event_type = str(payload.get("type", "event"))
        data = {key: value for key, value in payload.items() if key != "type"}
        async with self.sessions.begin() as session:
            run = await session.scalar(
                select(ResearchRunRow)
                .where(ResearchRunRow.id == run_id)
                .with_for_update()
            )
            if run is None:
                raise KeyError(run_id)
            if event_type == "run_started":
                run.status = "running"
                run.current_phase = "initializing"
            elif event_type == "phase_started":
                run.status = "running"
                run.current_phase = str(data.get("phase", run.current_phase))
            return self._append_locked_event(session, run, event_type, data)

    async def claim_run(
        self, run_id: str, worker_id: str, lease_seconds: int
    ) -> bool:
        async with self.sessions.begin() as session:
            run = await session.scalar(
                select(ResearchRunRow)
                .where(ResearchRunRow.id == run_id)
                .with_for_update()
            )
            if run is None or run.status not in {"queued", "running"}:
                return False
            now = _utc_now()
            if run.cancel_requested:
                return False
            if run.worker_id and run.lease_expires_at and run.lease_expires_at > now:
                return False
            run.worker_id = worker_id
            run.lease_expires_at = now + timedelta(seconds=lease_seconds)
            run.status = "running"
            run.updated_at = now
            return True

    async def renew_lease(
        self, run_id: str, worker_id: str, lease_seconds: int
    ) -> bool:
        async with self.sessions.begin() as session:
            result = await session.execute(
                update(ResearchRunRow)
                .where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.worker_id == worker_id,
                    ResearchRunRow.status == "running",
                )
                .values(lease_expires_at=_utc_now() + timedelta(seconds=lease_seconds))
            )
            return result.rowcount == 1

    async def release_claim(self, run_id: str, worker_id: str) -> None:
        async with self.sessions.begin() as session:
            await session.execute(
                update(ResearchRunRow)
                .where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.worker_id == worker_id,
                )
                .values(worker_id=None, lease_expires_at=None)
            )

    async def load_execution(self, run_id: str) -> dict[str, Any]:
        async with self.sessions() as session:
            run = await session.get(ResearchRunRow, run_id)
            if run is None:
                raise KeyError(run_id)
            return {
                "id": run.id,
                "query": run.query,
                "session_id": run.session_id,
                "status": run.status,
                "state": run.state or {},
                "next_phase": run.next_phase or "clarifier",
                "checkpoint_version": int(run.checkpoint_version or 0),
                "cancel_requested": bool(run.cancel_requested),
            }

    async def list_dispatchable_run_ids(self) -> list[str]:
        async with self.sessions() as session:
            return list(
                (
                    await session.scalars(
                        select(ResearchRunRow.id).where(
                            ResearchRunRow.status == "queued",
                            ResearchRunRow.cancel_requested.is_(False),
                        )
                    )
                ).all()
            )

    async def save_checkpoint(
        self,
        run_id: str,
        state: dict[str, Any],
        completed_phase: str | None,
        next_phase: str | None,
        status: str,
    ) -> int:
        async with self.sessions.begin() as session:
            run = await session.scalar(
                select(ResearchRunRow)
                .where(ResearchRunRow.id == run_id)
                .with_for_update()
            )
            if run is None:
                raise KeyError(run_id)
            version = int(run.checkpoint_version or 0) + 1
            now = _utc_now()
            session.add(
                RunCheckpointRow(
                    run_id=run_id,
                    version=version,
                    completed_phase=completed_phase,
                    next_phase=next_phase,
                    status=status,
                    state=state,
                    created_at=now,
                )
            )
            run.state = state
            run.checkpoint_version = version
            run.next_phase = next_phase
            run.status = status
            phase_name = "research" if next_phase == "orchestrator" else next_phase
            run.current_phase = phase_name or ("completed" if status == "completed" else run.current_phase)
            run.updated_at = now
            return version

    async def create_interaction_and_pause(
        self,
        run_id: str,
        interaction_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        async with self.sessions.begin() as session:
            run = await session.scalar(
                select(ResearchRunRow)
                .where(ResearchRunRow.id == run_id)
                .with_for_update()
            )
            if run is None:
                raise KeyError(run_id)
            now = _utc_now()
            session.add(
                RunInteractionRow(
                    id=interaction_id,
                    run_id=run_id,
                    kind=kind,
                    payload=payload,
                    response=None,
                    status="pending",
                    created_at=now,
                )
            )
            run.status = "waiting_input"
            run.worker_id = None
            run.lease_expires_at = None
            self._append_locked_event(
                session,
                run,
                "interaction_required",
                {
                    "interaction_id": interaction_id,
                    "kind": kind,
                    "payload": payload,
                },
                created_at=now,
            )

    async def get_resolved_interaction(
        self, run_id: str, kind: str, exclude_ids: list[str] | None = None
    ) -> tuple[str, dict[str, Any]] | None:
        async with self.sessions() as session:
            conditions = [
                RunInteractionRow.run_id == run_id,
                RunInteractionRow.kind == kind,
                RunInteractionRow.status == "resolved",
            ]
            if exclude_ids:
                conditions.append(RunInteractionRow.id.not_in(exclude_ids))
            interaction = await session.scalar(
                select(RunInteractionRow)
                .where(*conditions)
                .order_by(RunInteractionRow.created_at)
                .limit(1)
            )
            if interaction is None:
                return None
            return interaction.id, interaction.response or {}

    async def mark_interactions_consumed(self, interaction_ids: list[str]) -> None:
        if not interaction_ids:
            return
        async with self.sessions.begin() as session:
            await session.execute(
                update(RunInteractionRow)
                .where(
                    RunInteractionRow.id.in_(interaction_ids),
                    RunInteractionRow.status == "resolved",
                )
                .values(status="consumed")
            )

    async def resolve_interaction_for_resume(
        self, run_id: str, interaction_id: str, response: dict[str, Any]
    ) -> None:
        async with self.sessions.begin() as session:
            run = await session.scalar(
                select(ResearchRunRow)
                .where(ResearchRunRow.id == run_id)
                .with_for_update()
            )
            interaction = await session.get(RunInteractionRow, interaction_id)
            if (
                run is None
                or interaction is None
                or interaction.run_id != run_id
                or interaction.status != "pending"
            ):
                raise ValueError("interaction is no longer pending")
            now = _utc_now()
            interaction.status = "resolved"
            interaction.response = response
            interaction.resolved_at = now
            run.status = "queued"
            run.updated_at = now
            self._append_locked_event(
                session,
                run,
                "interaction_resolved",
                {"interaction_id": interaction_id, "kind": interaction.kind},
                created_at=now,
            )

    async def request_cancel(self, run_id: str) -> None:
        async with self.sessions.begin() as session:
            run = await session.scalar(
                select(ResearchRunRow)
                .where(ResearchRunRow.id == run_id)
                .with_for_update()
            )
            if run is None:
                raise KeyError(run_id)
            if run.status in {"completed", "failed", "cancelled"}:
                return
            run.cancel_requested = True
            if run.status in {"queued", "waiting_input"}:
                run.status = "cancelled"
                run.current_phase = "cancelled"
                run.worker_id = None
                run.lease_expires_at = None
                self._append_locked_event(session, run, "run_cancelled", {})

    async def is_cancel_requested(self, run_id: str) -> bool:
        async with self.sessions() as session:
            value = await session.scalar(
                select(ResearchRunRow.cancel_requested).where(
                    ResearchRunRow.id == run_id
                )
            )
            return bool(value)

    async def complete_run(self, run_id: str, state: dict[str, Any]) -> None:
        async with self.sessions.begin() as session:
            run = await session.scalar(
                select(ResearchRunRow)
                .where(ResearchRunRow.id == run_id)
                .with_for_update()
            )
            if run is None:
                raise KeyError(run_id)
            now = _utc_now()
            run.state = state
            run.status = "completed"
            run.current_phase = "completed"
            run.next_phase = None
            run.error = ""
            run.worker_id = None
            run.lease_expires_at = None
            self._append_locked_event(
                session,
                run,
                "run_completed",
                {
                    "report": state.get("report", ""),
                    "source_count": len(state.get("sources", [])),
                },
                created_at=now,
            )

    async def fail_run(self, run_id: str, error: str) -> None:
        async with self.sessions.begin() as session:
            run = await session.scalar(
                select(ResearchRunRow)
                .where(ResearchRunRow.id == run_id)
                .with_for_update()
            )
            if run is None:
                raise KeyError(run_id)
            run.status = "failed"
            run.current_phase = "failed"
            run.error = error
            run.worker_id = None
            run.lease_expires_at = None
            self._append_locked_event(session, run, "run_failed", {"error": error})

    async def cancel_running_run(self, run_id: str) -> None:
        async with self.sessions.begin() as session:
            run = await session.scalar(
                select(ResearchRunRow)
                .where(ResearchRunRow.id == run_id)
                .with_for_update()
            )
            if run is None:
                raise KeyError(run_id)
            if run.status == "cancelled":
                return
            run.status = "cancelled"
            run.current_phase = "cancelled"
            run.worker_id = None
            run.lease_expires_at = None
            self._append_locked_event(session, run, "run_cancelled", {})

    async def recover_interrupted_runs(self) -> None:
        """Mark process-bound tasks as failed after an API restart."""
        active_statuses = ("queued", "running", "waiting_input")
        async with self.sessions.begin() as session:
            runs = list(
                (
                    await session.scalars(
                        select(ResearchRunRow).where(
                            ResearchRunRow.status.in_(active_statuses)
                        )
                    )
                ).all()
            )
            for run in runs:
                sequence = int(
                    await session.scalar(
                        select(func.coalesce(func.max(RunEventRow.sequence), 0)).where(
                            RunEventRow.run_id == run.id
                        )
                    )
                    or 0
                ) + 1
                now = _utc_now()
                message = "API restarted before the in-process workflow completed"
                run.status = "failed"
                run.current_phase = "failed"
                run.error = message
                run.updated_at = now
                session.add(
                    RunEventRow(
                        run_id=run.id,
                        sequence=sequence,
                        type="run_failed",
                        data={"error": message, "reason": "process_restart"},
                        created_at=now,
                    )
                )
                await session.execute(
                    update(RunInteractionRow)
                    .where(
                        RunInteractionRow.run_id == run.id,
                        RunInteractionRow.status == "pending",
                    )
                    .values(status="abandoned", resolved_at=now)
                )

    async def load_runs(self) -> list[dict[str, Any]]:
        async with self.sessions() as session:
            runs = list(
                (
                    await session.scalars(
                        select(ResearchRunRow).order_by(ResearchRunRow.updated_at.desc())
                    )
                ).all()
            )
            if not runs:
                return []

            run_ids = [row.id for row in runs]
            events = list(
                (
                    await session.scalars(
                        select(RunEventRow)
                        .where(RunEventRow.run_id.in_(run_ids))
                        .order_by(RunEventRow.run_id, RunEventRow.sequence)
                    )
                ).all()
            )
            pending = list(
                (
                    await session.scalars(
                        select(RunInteractionRow).where(
                            RunInteractionRow.run_id.in_(run_ids),
                            RunInteractionRow.status == "pending",
                        )
                    )
                ).all()
            )

        events_by_run: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            events_by_run.setdefault(event.run_id, []).append(
                {
                    "id": event.sequence,
                    "type": event.type,
                    "created_at": _iso(event.created_at),
                    "data": event.data or {},
                }
            )
        pending_by_run = {
            interaction.run_id: {
                "id": interaction.id,
                "kind": interaction.kind,
                "payload": interaction.payload or {},
                "created_at": _iso(interaction.created_at),
            }
            for interaction in pending
        }
        return [
            {
                "id": run.id,
                "query": run.query,
                "session_id": run.session_id,
                "status": run.status,
                "current_phase": run.current_phase,
                "created_at": _iso(run.created_at),
                "updated_at": _iso(run.updated_at),
                "state": run.state or {},
                "error": run.error or "",
                "next_phase": run.next_phase,
                "events": events_by_run.get(run.id, []),
                "pending_interaction": pending_by_run.get(run.id),
            }
            for run in runs
        ]

    async def load_run(self, run_id: str) -> dict[str, Any] | None:
        async with self.sessions() as session:
            run = await session.get(ResearchRunRow, run_id)
            if run is None:
                return None
            events = list(
                (
                    await session.scalars(
                        select(RunEventRow)
                        .where(RunEventRow.run_id == run_id)
                        .order_by(RunEventRow.sequence)
                    )
                ).all()
            )
            interaction = await session.scalar(
                select(RunInteractionRow)
                .where(
                    RunInteractionRow.run_id == run_id,
                    RunInteractionRow.status == "pending",
                )
                .order_by(RunInteractionRow.created_at.desc())
                .limit(1)
            )
        return {
            "id": run.id,
            "query": run.query,
            "session_id": run.session_id,
            "status": run.status,
            "current_phase": run.current_phase,
            "created_at": _iso(run.created_at),
            "updated_at": _iso(run.updated_at),
            "state": run.state or {},
            "error": run.error or "",
            "next_phase": run.next_phase,
            "events": [
                {
                    "id": event.sequence,
                    "type": event.type,
                    "created_at": _iso(event.created_at),
                    "data": event.data or {},
                }
                for event in events
            ],
            "pending_interaction": (
                {
                    "id": interaction.id,
                    "kind": interaction.kind,
                    "payload": interaction.payload or {},
                    "created_at": _iso(interaction.created_at),
                }
                if interaction
                else None
            ),
        }

    async def load_events_after(
        self, run_id: str, after_id: int
    ) -> tuple[list[dict[str, Any]], str]:
        async with self.sessions() as session:
            status = await session.scalar(
                select(ResearchRunRow.status).where(ResearchRunRow.id == run_id)
            )
            if status is None:
                raise KeyError(run_id)
            events = list(
                (
                    await session.scalars(
                        select(RunEventRow)
                        .where(
                            RunEventRow.run_id == run_id,
                            RunEventRow.sequence > after_id,
                        )
                        .order_by(RunEventRow.sequence)
                    )
                ).all()
            )
        return (
            [
                {
                    "id": event.sequence,
                    "type": event.type,
                    "created_at": _iso(event.created_at),
                    "data": event.data or {},
                }
                for event in events
            ],
            status,
        )

