"""Initial schema - create all fact_checker tables.

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # Enable pgcrypto for gen_random_uuid()
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # video_jobs
    op.create_table(
        "video_jobs",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("local_path", sa.Text(), nullable=True),
        sa.Column("ingest_source", sa.String(50), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_video_jobs_status", "video_jobs", ["status"])

    # transcript_segments
    op.create_table(
        "transcript_segments",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("video_jobs.job_id", ondelete="CASCADE"), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=True),
        sa.Column("end_time", sa.Float(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("speaker", sa.String(100), nullable=True),
    )
    op.create_index("ix_transcript_segments_job_id", "transcript_segments", ["job_id"])

    # claims
    op.create_table(
        "claims",
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("video_jobs.job_id", ondelete="CASCADE"), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("speaker", sa.String(100), nullable=True),
        sa.Column("timestamp_start", sa.Float(), nullable=True),
        sa.Column("timestamp_end", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_claims_job_id", "claims", ["job_id"])

    # evidence_items
    op.create_table(
        "evidence_items",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("claims.claim_id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("snippet", sa.Text(), nullable=True),
        sa.Column("relevance_score", sa.Float(), nullable=True),
    )
    op.create_index("ix_evidence_items_claim_id", "evidence_items", ["claim_id"])

    # verdicts
    op.create_table(
        "verdicts",
        sa.Column("verdict_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("claims.claim_id", ondelete="CASCADE"), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("video_jobs.job_id", ondelete="CASCADE"), nullable=False),
        sa.Column("label", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_verdicts_job_id", "verdicts", ["job_id"])
    op.create_index("ix_verdicts_claim_id", "verdicts", ["claim_id"])


def downgrade() -> None:
    op.drop_table("verdicts")
    op.drop_table("evidence_items")
    op.drop_table("claims")
    op.drop_table("transcript_segments")
    op.drop_table("video_jobs")
