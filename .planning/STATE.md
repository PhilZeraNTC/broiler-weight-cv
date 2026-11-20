---
gsd_state_version: "1.0"
current_phase: 1
current_phase_name: Forensic Data Curation & Anti-Leakage Split
status: executing
stopped_at: Phase 1 context gathered
last_updated: "2026-11-20T21:32:41.320Z"
last_activity: 2026-11-20
last_activity_desc: Ingested docs/ (`proposta_pdi.md.pdf` + dataset audit), resolved 4 competing variants into planning artifacts
state_head: 28252bc00b7a83a90cb055607179ec62177b000c
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 3
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-11-20)

**Core value:** Estimar o peso vivo individual de frangos de corte de forma precisa, auditada e 100% interpretável usando apenas PDI clássico calibrado por óptica EXIF — sem deep learning e com zero vazamento do visor LCD — em uma demo web local (`< 500 ms`).
**Current focus:** Phase 1 — Forensic Data Curation & Anti-Leakage Split

## Current Position

Phase: 1 (Forensic Data Curation & Anti-Leakage Split) — READY TO EXECUTE
Plan: 0 of 0 in current phase
Status: Ready to execute
Last activity: 2026-11-20 — Ingested docs/ (`proposta_pdi.md.pdf` + dataset audit), resolved 4 competing variants into planning artifacts

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: -
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: none yet
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: Pin Python `3.12` via `uv` (`--only-binary :all:`) to isolate from host Python `3.14.10`; frontend pinned to React `19.3` + TS `7.0` + Vite `8.3` + Tailwind `v4.3` + Recharts `3.11` (`react-is@19.3.1`) + Biome `2.6`.
- [Init]: Single-source classical PDI engine (`backend/app/pdi/`) shared verbatim between offline extraction (`n_jobs=4`, `768–960 px` aspect-preserving) and FastAPI runtime (`async def` + `run_in_threadpool`, `cv2.setNumThreads(2)`), locked via SHA-256 `PDIConfig` hash.
- [Init]: Ban raw `cv2.imread`/`cv2.imdecode` repo-wide in favor of one Pillow EXIF-first loader (`ImageOps.exif_transpose()` + Xiaomi tag `39321`); enforce outside-mask pixel-perturbation invariance tests (`PROIBIÇÃO 2`).
- [Init]: Dual-bundle `models/weight_regressor.joblib` (`production_pipeline` + 5 `fold_pipelines`) guarantees honest out-of-fold scoring on live pitch presets with zero `.jpg` files committed to Git.
- [Ingest]: Merged `docs/` (`proposta_pdi.md.pdf` + dataset audit): v1 stays image-only (EXIF dual-lens, audited `116–2,093 g` / `~7–29 d` with Mendeley "45 days" disclosure in `WEB-05`), while Watershed (`PDI-05`/`WEB-04`) adds per-instance feature extraction + illustrative per-bird weight & scene mean with explicit group-sum disclaimer.

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 2]: Empirical PDI & optical spike required before finalizing batch extraction — verify SF-400 LCD/platter exclusion, white-on-white `>1,000 g` bucket segmentation (`< 2%` fallback), scale-adaptive Watershed counts on `417`/`757`/`823`/`1520`, and paired dual-lens optical normalization (`area_frac / zoom_multiple**2` vs. `A_px / (f_mm * zoom)**2`) across the `114` mixed-lens sessions to avoid `focal × zoom` double-counting.
- [Phase 3–4]: Code freeze `2026-11-28` for `2026-11-30` pitch (8 calendar days); preserve `< 400 ms` p95 latency margin on native `4080×1840` JPEGs and transparent reporting of the `801–1,216 g` dataset gap.

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| *(none)* | | | | |

## Session Continuity

Last session: 2026-11-20T21:00:21.489Z
Stopped at: Phase 1 context gathered
Resume file: .planning/phases/01-forensic-data-curation-anti-leakage-split/01-CONTEXT.md
