-- v0.9 Migration 1: Hash-chained write provenance on episode_log
-- Adds two nullable columns and a traversal index.
-- Fully additive — no existing columns modified.
-- Run once at first v0.9 startup (or manually before starting the server).

ALTER TABLE episode_log
  ADD COLUMN IF NOT EXISTS provenance_hash VARCHAR(64),   -- SHA-256 of this write's receipt
  ADD COLUMN IF NOT EXISTS previous_hash   VARCHAR(64);   -- SHA-256 of previous write in chain (NULL = genesis)

-- Index for chain traversal and audit queries
CREATE INDEX IF NOT EXISTS idx_episode_log_provenance
  ON episode_log (agent_namespace, provenance_hash);
