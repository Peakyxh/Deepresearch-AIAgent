"""Redis Streams based durable task queue."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import ResponseError


@dataclass(frozen=True)
class QueuedRun:
    message_id: str
    run_id: str
    reason: str
    raw_message: str = ""
    transport: str = "streams"


class RedisTaskQueue:
    def __init__(
        self,
        redis_url: str,
        *,
        stream: str = "deepresearch:runs",
        group: str = "research-workers",
    ) -> None:
        # RESP2 keeps compatibility with the legacy Windows Redis 3.x port.
        self.redis = Redis.from_url(redis_url, decode_responses=True, protocol=2)
        self.stream = stream
        self.group = group
        self.dedup_prefix = f"{stream}:enqueued:"
        self.list_queue = f"{stream}:list"
        self.processing_list = f"{stream}:processing"
        self.lease_prefix = f"{stream}:lease:"
        self.transport = "streams"

    async def initialize(self) -> None:
        await self.redis.ping()
        info = await self.redis.info("server")
        version_text = str(info.get("redis_version", "0.0"))
        version_parts = version_text.split(".")
        version = tuple(
            int(part) if part.isdigit() else 0
            for part in (version_parts + ["0", "0"])[:3]
        )
        # XAUTOCLAIM, required for safe crashed-consumer recovery, arrived in 6.2.
        if version < (6, 2, 0):
            self.transport = "lists"
            return
        try:
            await self.redis.xgroup_create(
                self.stream, self.group, id="0-0", mkstream=True
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def close(self) -> None:
        await self.redis.aclose()

    async def enqueue(self, run_id: str, reason: str = "execute") -> bool:
        dedup_key = f"{self.dedup_prefix}{run_id}"
        inserted = await self.redis.set(dedup_key, "1", ex=86400, nx=True)
        if not inserted:
            return False
        try:
            if self.transport == "streams":
                await self.redis.xadd(
                    self.stream,
                    {"run_id": run_id, "reason": reason},
                    maxlen=10000,
                    approximate=True,
                )
            else:
                message = json.dumps(
                    {
                        "message_id": uuid4().hex,
                        "run_id": run_id,
                        "reason": reason,
                    },
                    separators=(",", ":"),
                )
                await self.redis.lpush(self.list_queue, message)
        except Exception:
            await self.redis.delete(dedup_key)
            raise
        return True

    @staticmethod
    def _decode(message_id: str, fields: dict[str, Any]) -> QueuedRun:
        return QueuedRun(
            message_id=message_id,
            run_id=str(fields["run_id"]),
            reason=str(fields.get("reason", "execute")),
        )

    async def reserve(
        self,
        consumer: str,
        *,
        block_ms: int = 5000,
        reclaim_idle_ms: int = 120000,
    ) -> QueuedRun | None:
        if self.transport == "lists":
            await self._recover_abandoned_list_message(reclaim_idle_ms)
            raw_message = await self.redis.brpoplpush(
                self.list_queue,
                self.processing_list,
                timeout=max(1, math.ceil(block_ms / 1000)),
            )
            if not raw_message:
                return None
            data = json.loads(raw_message)
            item = QueuedRun(
                message_id=str(data["message_id"]),
                run_id=str(data["run_id"]),
                reason=str(data.get("reason", "execute")),
                raw_message=raw_message,
                transport="lists",
            )
            await self.redis.set(
                f"{self.lease_prefix}{item.message_id}",
                consumer,
                px=reclaim_idle_ms,
            )
            return item

        claimed = await self.redis.xautoclaim(
            self.stream,
            self.group,
            consumer,
            min_idle_time=reclaim_idle_ms,
            start_id="0-0",
            count=1,
        )
        claimed_messages = claimed[1] if len(claimed) > 1 else []
        if claimed_messages:
            message_id, fields = claimed_messages[0]
            return self._decode(message_id, fields)

        response = await self.redis.xreadgroup(
            self.group,
            consumer,
            {self.stream: ">"},
            count=1,
            block=block_ms,
        )
        if not response:
            return None
        _, messages = response[0]
        if not messages:
            return None
        message_id, fields = messages[0]
        return self._decode(message_id, fields)

    async def acknowledge(self, item: QueuedRun) -> None:
        if item.transport == "lists":
            await self.redis.lrem(self.processing_list, 1, item.raw_message)
            await self.redis.delete(f"{self.lease_prefix}{item.message_id}")
        else:
            await self.redis.xack(self.stream, self.group, item.message_id)
            await self.redis.xdel(self.stream, item.message_id)
        await self.redis.delete(f"{self.dedup_prefix}{item.run_id}")

    async def touch(
        self, item: QueuedRun, consumer: str, lease_seconds: int = 120
    ) -> None:
        """Reset pending idle time while a worker is actively processing a run."""
        if item.transport == "lists":
            await self.redis.set(
                f"{self.lease_prefix}{item.message_id}",
                consumer,
                ex=lease_seconds,
            )
            return
        await self.redis.xclaim(
            self.stream,
            self.group,
            consumer,
            min_idle_time=0,
            message_ids=[item.message_id],
            justid=True,
        )

    async def _recover_abandoned_list_message(self, reclaim_idle_ms: int) -> None:
        messages = await self.redis.lrange(self.processing_list, 0, 99)
        script = """
        if redis.call('EXISTS', KEYS[2]) == 1 then return 0 end
        local removed = redis.call('LREM', KEYS[1], 1, ARGV[1])
        if removed == 1 then
            redis.call('RPUSH', KEYS[3], ARGV[1])
        end
        return removed
        """
        for raw_message in messages:
            try:
                message_id = str(json.loads(raw_message)["message_id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            lease_key = f"{self.lease_prefix}{message_id}"
            await self.redis.eval(
                script,
                3,
                self.processing_list,
                lease_key,
                self.list_queue,
                raw_message,
            )
