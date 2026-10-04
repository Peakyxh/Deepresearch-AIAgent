"""Command-line entry point for the independent research worker."""

import asyncio
import logging

from api.persistence import PostgresRunRepository
from config import settings
from worker.queue import RedisTaskQueue
from worker.service import ResearchWorker


async def main() -> None:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required for worker mode")
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required for worker mode")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    repository = PostgresRunRepository(
        settings.database_url,
        echo=settings.database_echo,
        auto_create=settings.database_auto_create,
    )
    queue = RedisTaskQueue(
        settings.redis_url,
        stream=settings.redis_stream_name,
        group=settings.redis_consumer_group,
    )
    worker = ResearchWorker(
        repository,
        queue,
        lease_seconds=settings.worker_lease_seconds,
    )
    await worker.initialize()
    try:
        await worker.run_forever()
    finally:
        await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
