---
gsd_state_version: "1.0"
current_phase: 03
current_phase_name: Tabular Regression, Optical Ablation & Production Bundling
status: executing
stopped_at: Phase 3 context gathered
last_updated: "2026-11-23T23:30:11.779Z"
last_activity: 2026-11-23
last_activity_desc: Phase 2 complete, transitioned to Phase 3
state_head: 2efb6e304757e27393c8bdb7d310569dd15e85ed
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 10
  completed_plans: 7
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-11-23)

**Core value:** Estimar o peso vivo individual de frangos de corte de forma precisa, auditada e 100% interpretável usando apenas PDI clássico calibrado por óptica EXIF — sem deep learning e com zero vazamento do visor LCD — em uma demo web local (`< 500 ms`).
**Current focus:** Phase 3 — Tabular Regression, Optical Ablation & Production Bundling

## Current Position

Phase: 03 (Tabular Regression, Optical Ablation & Production Bundling) — READY TO EXECUTE
Plan: Not started
Status: Ready to execute
Last activity: 2026-11-23 — Phase 2 complete, transitioned to Phase 3

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**

- Total plans completed: 7
- Average duration: -
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | - | - |
| 2 | 4 | - | - |

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
- [Phase 2]: Frozen 49-key `FEATURE_COLUMNS` inside `bird_mask == 255` (`pdi_config_hash = 43a3cfaa3208019e`, `working_long_edge = 768`, zero raw resolution/bbox leakage) materialized to `data/processed/features_clean.parquet` (`1,738 × 60` in `46.53 s` at `--workers 4`), with `0.23%` fallback (`0/377` bucket), `0.0`-diff outside-mask invariance, and `p95 = 88.3 ms` single-bird X-Ray latency.
- [Phase 2]: Empirical dual-lens audit across `127` mixed-lens sessions (`175` same-scene pairs, `models/pdi_calibration_audit.json`) proved `area_optical_norm = area_px / (zoom_multiple ** 2)` wins unanimously (`W/UW = 0.9638` on bucket pairs vs. `0.145` double-counting under `(f_mm * zoom)**2`); retained `area_frac_optical_norm` as secondary ablation feature for Phase 3.
- [Phase 2]: Post-review hardening (`bc2217c`) resolved `CR-01`/`CR-02` & `WR-02..07` (Hu moments `binaryImage=True` dropping `55 → 0` saturated, `[0.5, 200.0] mm` EXIF `FocalLength` bounds, 8-connectivity + SF-400 LCD suppression in Watershed with `9/13` exact / `12/13` ±1 / `MAE 0.38`) and verified `136/136` tests passing.

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3–4]: Code freeze `2026-11-28` for `2026-11-30` pitch (8 calendar days); preserve `< 400 ms` p95 latency margin on native `4080×1840` JPEGs and transparent reporting of the `801–1,216 g` dataset gap.

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| *(none)* | | | | |

## Session Continuity

Last session: 2026-11-23T22:39:23.315Z
Stopped at: Phase 3 context gathered
Resume file: .planning/phases/03-tabular-regression-optical-ablation-production-bundling/03-CONTEXT.md
