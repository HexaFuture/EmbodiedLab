CREATE TABLE IF NOT EXISTS embodiedlab_operation_events (
    operation_id text NOT NULL,
    sequence bigint NOT NULL,
    event_id text NOT NULL UNIQUE,
    data jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (operation_id, sequence)
);

CREATE INDEX IF NOT EXISTS embodiedlab_operation_events_operation_created
ON embodiedlab_operation_events (operation_id, created_at);
