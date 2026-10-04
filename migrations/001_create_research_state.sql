BEGIN;

CREATE TABLE IF NOT EXISTS research_runs (
    id VARCHAR(64) PRIMARY KEY,
    query TEXT NOT NULL,
    session_id VARCHAR(80) NOT NULL,
    status VARCHAR(32) NOT NULL,
    current_phase VARCHAR(64) NOT NULL,
    state JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT NOT NULL DEFAULT '',
    next_phase VARCHAR(64),
    checkpoint_version INTEGER NOT NULL DEFAULT 0,
    event_sequence INTEGER NOT NULL DEFAULT 0,
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    worker_id VARCHAR(128),
    lease_expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_research_runs_status ON research_runs (status);
CREATE INDEX IF NOT EXISTS ix_research_runs_updated_at ON research_runs (updated_at DESC);

CREATE TABLE IF NOT EXISTS run_events (
    id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    type VARCHAR(64) NOT NULL,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_run_events_run_sequence UNIQUE (run_id, sequence)
);

CREATE INDEX IF NOT EXISTS ix_run_events_run_sequence ON run_events (run_id, sequence);

CREATE TABLE IF NOT EXISTS run_interactions (
    id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    kind VARCHAR(64) NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    response JSONB,
    status VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_run_interactions_status ON run_interactions (status);
CREATE INDEX IF NOT EXISTS ix_run_interactions_run_status ON run_interactions (run_id, status);

CREATE TABLE IF NOT EXISTS run_checkpoints (
    id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    completed_phase VARCHAR(64),
    next_phase VARCHAR(64),
    status VARCHAR(32) NOT NULL,
    state JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_run_checkpoints_run_version UNIQUE (run_id, version)
);

CREATE INDEX IF NOT EXISTS ix_run_checkpoints_run_version ON run_checkpoints (run_id, version);

COMMIT;
