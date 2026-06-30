---
title: fact_checker repo-ready documentation patch plan
version: 2
status: proposed
repo: fact_checker
---

# Repo-ready patch plan

## Goal
Integrate the new agent and skill documentation into the existing repository without changing runtime behavior.

## Recommended final placement

```text
fact_checker/
├── AGENTS.md
├── README.md
├── FEATURES.md
├── SRS.md
├── docs/
│   └── REPO_PATCH_PLAN.md
├── mcp/
│   ├── factcheckermcpserver.py
│   ├── SKILL.md
│   └── servers.md
├── skills/
│   ├── ingest/
│   │   └── SKILL.md
│   ├── image-analysis/
│   │   └── SKILL.md
│   ├── claim-extraction/
│   │   └── SKILL.md
│   ├── evidence-retrieval/
│   │   └── SKILL.md
│   ├── deep-research/
│   │   └── SKILL.md
│   └── verdict/
│       └── SKILL.md
└── src/
    └── factchecker/
        ├── agents/
        ├── prompts/
        ├── services/
        └── skills/
```

## Placement rationale
- Keep `AGENTS.md` at repo root because the project already has a top-level architecture document.
- Add human-readable `skills/` docs at repo root to avoid mixing markdown contracts with Python implementation modules under `src/factchecker/skills/`.
- Keep MCP docs beside the existing `mcp/factcheckermcpserver.py` entrypoint.
- Keep the patch plan under `docs/` so future cleanup decisions have a stable place to live.

## Merge instructions
1. Replace or merge the existing root `AGENTS.md` with the new canonical version.
2. Add the root `skills/` directory and place each capability `SKILL.md` in its folder.
3. Add `mcp/SKILL.md` and `mcp/servers.md` next to `mcp/factcheckermcpserver.py`.
4. Update `README.md` with a documentation index linking AGENTS, skills, and MCP setup.
5. Add a short note in `FEATURES.md` that contract docs now live under root `skills/` and `mcp/`.

## Exact file mapping
- `AGENTS.md` → repo root
- `skills/ingest/SKILL.md` → new repo-root skills folder
- `skills/image-analysis/SKILL.md` → new repo-root skills folder
- `skills/claim-extraction/SKILL.md` → new repo-root skills folder
- `skills/evidence-retrieval/SKILL.md` → new repo-root skills folder
- `skills/deep-research/SKILL.md` → new repo-root skills folder
- `skills/verdict/SKILL.md` → new repo-root skills folder
- `mcp/SKILL.md` → existing `mcp/` folder
- `mcp/servers.md` → existing `mcp/` folder

## Suggested README additions
Add a new section named `Documentation map` with links to:
- `AGENTS.md`
- `skills/ingest/SKILL.md`
- `skills/image-analysis/SKILL.md`
- `skills/claim-extraction/SKILL.md`
- `skills/evidence-retrieval/SKILL.md`
- `skills/deep-research/SKILL.md`
- `skills/verdict/SKILL.md`
- `mcp/SKILL.md`
- `mcp/servers.md`

## Cleanup decisions to make after merge
- Decide whether the root-level duplicate modules are legacy wrappers or should be removed.
- Decide whether `config.py` or `settings.py` is the long-term configuration source of truth.
- Decide whether frame-to-transcript correlation should be documented as planned or promoted to required behavior.

## Notes
This patch is documentation-only and should not change runtime execution paths.
