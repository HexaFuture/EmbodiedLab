CREATE TABLE IF NOT EXISTS embodiedlab_operations (
    id text PRIMARY KEY,
    data jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS embodiedlab_workers (
    id text PRIMARY KEY,
    data jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS embodiedlab_resources (
    kind text NOT NULL,
    id text NOT NULL,
    data jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (kind, id)
);

CREATE TABLE IF NOT EXISTS embodiedlab_operation_logs (
    sequence bigserial PRIMARY KEY,
    operation_id text NOT NULL,
    stream text NOT NULL,
    message text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS embodiedlab_operation_logs_operation_sequence
ON embodiedlab_operation_logs (operation_id, sequence);
