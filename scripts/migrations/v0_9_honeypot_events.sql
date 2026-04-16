-- v0.9 Migration 2: GradientHoneypot events table
-- Stores records of honeypot traps being triggered.
-- Fully additive — new table, no existing schema touched.
-- Run once at first v0.9 startup (or manually before starting the server).

CREATE TABLE IF NOT EXISTS honeypot_events (
    id              SERIAL PRIMARY KEY,
    trap_name       VARCHAR(255) NOT NULL,
    triggered_by    VARCHAR(255),            -- tool name or query string that triggered the trap
    triggered_at    TIMESTAMP DEFAULT NOW(),
    agent_namespace VARCHAR(255),
    query_text      TEXT
);

CREATE INDEX IF NOT EXISTS idx_honeypot_events_namespace
  ON honeypot_events (agent_namespace, triggered_at DESC);
