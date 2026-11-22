---
gsd_state_version: "1.0"
current_phase: 2
current_phase_name: Classical PDI Engine, Optical Calibration & Watershed
status: executing
stopped_at: Completed 02-03
last_updated: "2026-11-22T15:08:37.005Z"
last_activity: 2026-11-22
last_activity_desc: Phase 2 execution started
state_head: a236c3417b1e7ed1f1d64d5973e8d0ea5954acf2
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 7
  completed_plans: 6
  percent: 25
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-11-20)

**Core value:** Estimar o peso vivo individual de frangos de corte de forma precisa, auditada e 100% interpretável usando apenas PDI clássico calibrado por óptica EXIF — sem deep learning e com zero vazamento do visor LCD — em uma demo web local (`< 500 ms`).
**Current focus:** Phase 2 — Classical PDI Engine, Optical Calibration & Watershed

## Current Position

Phase: 2 (Classical PDI Engine, Optical Calibration & Watershed) — EXECUTING
Plan: 1 of 4
Status: Executing Phase 2
Last activity: 2026-11-22 — Phase 2 execution started

Progress: [███░░░░░░░] 25%

## Performance Metrics

**Velocity:**

- Total plans completed: 3
- Average duration: -
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | - | - |

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

Last session: 2026-11-22T15:08:36.951Z
Stopped at: Completed 02-03
Resume file: .planning/phases/02-classical-pdi-engine-optical-calibration-watershed/02-03-SUMMARY.md
