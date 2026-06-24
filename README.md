# fact_checker

A lightweight Python-first skeleton for fact-checking claims from YouTube and news videos.

## Stack
- Python 3.11+
- LangChain for LLM plumbing
- OpenRouter for model access
- FFmpeg/ffprobe for media processing
- PostgreSQL for persistence

## Initial flow
1. Ingest a URL or video file
2. Normalize media and transcript
3. Extract factual claims
4. Retrieve evidence
5. Draft verdicts
6. Route low-confidence items to review

## Planned agents
- IntakeAgent
- TranscriptAgent
- ClaimExtractionAgent
- EvidenceRetrievalAgent
- VerdictDraftAgent
- ReviewAgent
