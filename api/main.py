import json
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.run_manager import run_manager
from api.schemas import (
    CreateRunRequest,
    InteractionResponse,
    RunDetail,
    RunSummary,
)


def _allowed_origins() -> list[str]:
    configured = os.getenv("DEEPRESEARCH_CORS_ORIGINS", "")
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]
    return ["http://localhost:3000", "http://127.0.0.1:3000"]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await run_manager.initialize()
    try:
        yield
    finally:
        await run_manager.close()


app = FastAPI(
    title="DeepResearch Agent API",
    version="0.2.0",
    description="Task, event streaming, and human-in-the-loop API for DeepResearch Agent.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs", response_model=RunDetail, status_code=status.HTTP_202_ACCEPTED)
async def create_run(request: CreateRunRequest) -> dict:
    record = await run_manager.create_run(request.query, request.session_id)
    return run_manager.detail(record)


@app.get("/api/runs", response_model=list[RunSummary])
async def list_runs() -> list[dict]:
    return [run_manager.summary(record) for record in await run_manager.list_runs_async()]


@app.get("/api/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: str) -> dict:
    try:
        return run_manager.detail(await run_manager.get_run(run_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


@app.get("/api/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    after: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    try:
        await run_manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc

    cursor = after
    if last_event_id and last_event_id.isdigit():
        cursor = max(cursor, int(last_event_id))

    async def event_stream() -> AsyncIterator[str]:
        async for event in run_manager.stream_events(run_id, cursor):
            if event is None:
                yield ": keep-alive\n\n"
                continue
            encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            # Keep events on the default SSE channel so browser EventSource.onmessage
            # receives every event. The domain event name remains in the JSON body.
            yield f"id: {event['id']}\ndata: {encoded}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/runs/{run_id}/responses", status_code=status.HTTP_204_NO_CONTENT)
async def respond_to_interaction(run_id: str, request: InteractionResponse) -> None:
    try:
        await run_manager.respond(
            run_id,
            request.interaction_id,
            {"answers": request.answers, "feedback": request.feedback},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_run(run_id: str) -> dict[str, str]:
    try:
        await run_manager.cancel(run_id)
        return {"status": "cancelling"}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc

