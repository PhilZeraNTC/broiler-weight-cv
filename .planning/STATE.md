---
gsd_state_version: "1.0"
current_phase: 4
current_phase_name: FastAPI Local Backend & Interactive Pitch Web Studio
status: planning
stopped_at: Phase 4 UI-SPEC approved
last_updated: "2026-11-24T08:42:46.999Z"
last_activity: 2026-11-23
last_activity_desc: Phase 3 complete, transitioned to Phase 4
state_head: e45a98d5763aca4704f0708c114d7176c750c80d
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 10
  completed_plans: 10
  percent: 75
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-11-23)

**Core value:** Estimar o peso vivo individual de frangos de corte de forma precisa, auditada e 100% interpretável usando apenas PDI clássico calibrado por óptica EXIF — sem deep learning e com zero vazamento do visor LCD — em uma demo web local (`< 500 ms`).
**Current focus:** Phase 4 — FastAPI Local Backend & Interactive Pitch Web Studio

## Current Position

Phase: 4 — FastAPI Local Backend & Interactive Pitch Web Studio
Plan: Not started
Status: Ready to plan
Last activity: 2026-11-23 — Phase 3 complete, transitioned to Phase 4

Progress: [████████░░] 75%

## Performance Metrics

**Velocity:**

- Total plans completed: 10
- Average duration: -
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | - | - |
| 2 | 4 | - | - |
| 3 | 3 | - | - |

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
- [Phase 3]: Deterministic winner `LightGBMLog` on `C2_physics_normalized_strict` (`39` strictly mask-gated features, `area_px / zoom²`, `session_band_capped_5x` weights, `duan_smearing=False`) achieved `MAE = 158.11 g`, `RMSE = 264.02 g`, `BalMAPE = 35.11%`, `R² = 0.5582` (`session_averaged_r2 = 0.6905`), unanimously beating `C1_uncalibrated_pdi` with `0.0%` dual-mode counterfactual shift while disqualifying `C3`/`C4` lens-ID shortcuts.
- [Phase 3]: Published immutable dual-bundle `models/weight_regressor.joblib` (git-ignored, `3.18 MiB`, `production_pipeline` + 5 `fold_pipelines` `{0..4}`, `75.2 ms` cold load, `2.4 ms` single-row inference), pure-stdlib `age_stage.py`, byte-deterministic `models/evaluation_report.json` (`b8d21b2d...`), and opaque-`image_id` `models/presets_catalog.json` (`3a0c7e50...`, 6 OOF single-bird presets spanning folds `0..4` incl. same-session `541__s0` Wide `383.22 g` vs. UW `384.16 g` + 4 Watershed presets).
- [Phase 3]: Post-review hardening (`1fc10c4`) resolved `CR-01` (preserving tuple `clahe_tile_grid = (8, 8)` + `PDIConfig(**pdi_config)` reconstruction & hash check inside `validate_model_bundle`), `WR-01..06`, and `IN-01..04`; verified `205/205` pytest passing and `03-VERIFICATION.md` `16/16` (`passed`).

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 4]: Code freeze `2026-11-28` for `2026-11-30` pitch; preserve `< 400 ms` p95 latency margin on native `4080×1840` JPEGs (`p95 = 88.3 ms` PDI + `2.4 ms` ML leaves `> 300 ms` headroom), resolve opaque `image_id` presets strictly via `manifest_clean.parquet`/`manifest_quarantine.parquet` inside `RAW_DATASET_DIR.resolve()`, and transparently report the `801–1,216 g` dataset gap & Mendeley `"45 days"` reconciliation in `WEB-05`.

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| *(none)* | | | | |

## Session Continuity

Last session: 2026-11-24T08:42:46.951Z
Stopped at: Phase 4 UI-SPEC approved
Resume file: .planning/phases/04-fastapi-local-backend-interactive-pitch-web-studio/04-UI-SPEC.md
