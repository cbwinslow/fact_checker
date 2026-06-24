-- fact_checker PostgreSQL schema
-- Run: psql -U fact_checker -d fact_checker -f migrations/schema.sql

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- -------------------------------------------------------
-- video_jobs
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS video_jobs (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    url             TEXT,
    local_path      TEXT,
    status          TEXT        NOT NULL DEFAULT 'pending',
    ingest_source   TEXT,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_video_jobs_status ON video_jobs(status);
CREATE INDEX IF NOT EXISTS idx_video_jobs_created ON video_jobs(created_at DESC);

-- -------------------------------------------------------
-- transcript_segments
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS transcript_segments (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id      UUID        NOT NULL REFERENCES video_jobs(id) ON DELETE CASCADE,
    start_sec   NUMERIC(10,3) NOT NULL,
    end_sec     NUMERIC(10,3) NOT NULL,
    text        TEXT        NOT NULL,
    speaker     TEXT
);

CREATE INDEX IF NOT EXISTS idx_segments_job_id ON transcript_segments(job_id);

-- -------------------------------------------------------
-- claims
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS claims (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID        NOT NULL REFERENCES video_jobs(id) ON DELETE CASCADE,
    segment_id      UUID        REFERENCES transcript_segments(id),
    text            TEXT        NOT NULL,
    is_checkable    BOOLEAN     NOT NULL DEFAULT TRUE,
    confidence      NUMERIC(4,3) NOT NULL DEFAULT 1.0,
    context         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_claims_job_id ON claims(job_id);
CREATE INDEX IF NOT EXISTS idx_claims_checkable ON claims(is_checkable);

-- -------------------------------------------------------
-- evidence_items
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS evidence_items (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id            UUID        NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    source_url          TEXT        NOT NULL,
    title               TEXT,
    snippet             TEXT        NOT NULL,
    relevance_score     NUMERIC(4,3) NOT NULL DEFAULT 0.0,
    is_factcheck_source BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evidence_claim_id ON evidence_items(claim_id);
CREATE INDEX IF NOT EXISTS idx_evidence_factcheck ON evidence_items(is_factcheck_source);

-- -------------------------------------------------------
-- verdict_results
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS verdict_results (
    id                      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id                UUID        NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    verdict                 TEXT        NOT NULL,
    explanation             TEXT        NOT NULL,
    confidence              NUMERIC(4,3) NOT NULL,
    requires_human_review   BOOLEAN     NOT NULL DEFAULT FALSE,
    reviewed_by             TEXT,
    reviewed_at             TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_verdicts_claim_id ON verdict_results(claim_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_review ON verdict_results(requires_human_review);
CREATE INDEX IF NOT EXISTS idx_verdicts_verdict ON verdict_results(verdict);

-- -------------------------------------------------------
-- verdict_evidence junction
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS verdict_evidence (
    verdict_id  UUID NOT NULL REFERENCES verdict_results(id) ON DELETE CASCADE,
    evidence_id UUID NOT NULL REFERENCES evidence_items(id) ON DELETE CASCADE,
    PRIMARY KEY (verdict_id, evidence_id)
);
