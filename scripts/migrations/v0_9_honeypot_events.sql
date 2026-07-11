-- v0.9 Migration 2 (corrected): GradientHoneypot alerts table
-- Matches spec §5 schema exactly.
-- Fully additive — new table, no existing schema touched.
-- Run once at first v0.9 startup.

CREATE TABLE IF NOT EXISTS honeypot_alerts (
    id                  BIGSERIAL    PRIMARY KEY,
    agent_namespace     VARCHAR(128) NOT NULL,
    tool_called         VARCHAR(64)  NOT NULL,   -- memory_search | knowledge_search | memory_get_entity
    query_text          TEXT         NOT NULL,    -- raw query string that triggered the alert
    matched_pattern     VARCHAR(256) NOT NULL,    -- which pattern fired
    pattern_source      VARCHAR(32)  NOT NULL,    -- 'default' | 'custom'
    suppressed          BOOLEAN      NOT NULL DEFAULT TRUE,
    lockdown_triggered  BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_honeypot_alerts_namespace
  ON honeypot_alerts (agent_namespace);

CREATE INDEX IF NOT EXISTS idx_honeypot_alerts_created_at
  ON honeypot_alerts (created_at DESC);
