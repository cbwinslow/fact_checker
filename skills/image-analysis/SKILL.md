---
name: image-analysis
version: 1
summary: Analyze images and extracted video frames for OCR text, visible claims, objects, context, and manipulation risk.
module: src/factchecker/agents/imageanalyst.py
owners:
  - cbwinslow
inputs:
  - local image files
  - extracted video frames
outputs:
  - ImageAnalysis records
  - OCR text
  - visible claims
  - manipulation risk assessments
models:
  - multimodal
prompts:
  - src/factchecker/prompts/imageanalysis.md
---

# Image Analysis

## Purpose
Interpret visual evidence so that image-only or frame-heavy submissions can still enter the fact-check pipeline.

## Responsibilities
- Read image metadata when available.
- Extract OCR text.
- Identify visible claims.
- Flag likely manipulation or ambiguity.
- Pass visible claims downstream as candidate factual claims.

## Handoffs
Writes `ctx.images`; visible claims should be eligible for promotion into `ctx.claims`.
