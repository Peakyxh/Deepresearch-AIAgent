BEGIN;

ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS next_phase VARCHAR(64);
ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS checkpoint_version INTEGER NOT NULL DEFAULT 0;
ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS event_sequence INTEGER NOT NULL DEFAULT 0;
ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS cancel_requested BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS worker_id VARCHAR(128);
ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ;

UPDATE research_runs
SET event_sequence = counts.max_sequence
FROM (
    SELECT run_id, COALESCE(MAX(sequence), 0) AS max_sequence
    FROM run_events
    GROUP BY run_id
) AS counts
WHERE research_runs.id = counts.run_id
  AND research_runs.event_sequence < counts.max_sequence;

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

CREATE INDEX IF NOT EXISTS ix_run_checkpoints_run_version
    ON run_checkpoints (run_id, version);

COMMIT;
