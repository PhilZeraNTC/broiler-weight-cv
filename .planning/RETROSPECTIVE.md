# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — MVP

**Shipped:** 2026-11-24  
**Phases:** 4 | **Plans:** 14 | **Tasks:** 36 | **Timeline:** 4 days (`2026-11-20` → `2026-11-24`, 4 days ahead of `2026-11-28` code freeze)

### What Was Built

- **Phase 1 — Forensic Data Curation & Anti-Leakage Split (`DATA-01`, `DATA-02`):** Isolated binary-only CPython `3.12.14` (`uv`), hardened Pillow EXIF-first loader (`ImageOps.exif_transpose()` + Xiaomi tag `39321` `zoomMultiple`, zero `cv2.imread`/`cv2.imdecode` repo-wide), verbatim Audit §9 forensic ingestor resolving level-2 nested folders, `1340_`, `367 → 267 g`, CP866 mojibake, and Eastern-Arabic digits into `1,738` clean / `15` quarantined typed PyArrow Parquets, and reproducible zero-overlap 5-fold `StratifiedGroupKFold` by `session_id` (`models/dataset_audit_summary.json`).
- **Phase 2 — Classical PDI Engine, Optical Calibration & Watershed (`PDI-01`–`PDI-05`):** Frozen `PDIConfig` (`working_long_edge = 768 px`, SHA-256 `43a3cfaa3208019e`), rotation-invariant SF-400 LCD/housing suppressor (`786` activations, `0` bucket false positives), 3-scenario classical single-bird segmenter (`0.23%` primary fallback), `49` strictly mask-gated features (`0.0`-diff outside-mask invariance), empirically proven `area_px / zoom_multiple²` calibration (`W/UW = 0.9638` on bucket pairs across `127` mixed-lens sessions), 4-stage X-Ray renderer (`p95 = 88.3 ms`), capped 4-worker batch CLI (`features_clean.parquet`, `46.53 s`), and isolated Marker-based Watershed engine (`9/13` exact, `12/13` ±1, `MAE = 0.38` birds).
- **Phase 3 — Tabular Regression, Optical Ablation & Production Bundling (`MOD-01`–`MOD-03`):** Leak-free 5-fold OOF benchmark across 4 classical families × 4 optical ablation conditions (`AllometricLogTransformer`, `LogDuanRegressor`, `5×`-capped session×band fold weights), deterministic win of `LightGBMLog / C2_physics_normalized_strict` (`39` features, `MAE = 158.11 g`, `RMSE = 264.02 g`, `BalMAPE = 35.11%`, `R² = 0.5582`, `session_averaged_r2 = 0.6905`, exact `0.0%` dual-mode counterfactual shift while disqualifying `C3`/`C4` lens-ID shortcuts), dual-bundle `models/weight_regressor.joblib` (`production_pipeline` + 5 `fold_pipelines`, `2.4 ms` single-row CPU inference), `age_stage.py`, `models/evaluation_report.json`, and opaque-ID `models/presets_catalog.json`.
- **Phase 4 — FastAPI Local Backend & Interactive Pitch Web Studio (`WEB-01`–`WEB-05`):** Modular FastAPI API (`6` endpoints, `lifespan` caching, `cv2.setNumThreads(2)`, `run_in_threadpool`, ASVS L1 upload guards, honest `fold_pipelines[fold]` OOF scoring, `56.1–79.7 ms` live `/api/predict`, `0.94 g` live Dual-Lens delta, `_NonApiSpaRoute` single-origin serving) paired with a React 19 + TypeScript 7 + Vite 8 + Tailwind CSS v4 + Recharts 3 4-tab pitch studio (`Estúdio Raio-X`, `Separação Watershed` with permanent `group_sum_only` banner and reconciled ASVS V5 sliders, `Auditoria & Métricas` with `1,738→188` scatter and Mendeley forensic funnel, and 100% static `Visão Futura (v2)`).

### What Worked

- **Empirical data spikes before locking contracts:** Running a live dual-lens ratio audit on `127` mixed-lens sessions in Phase 2 caught the `(f_mm * zoom)²` double-counting bug (`W/UW = 0.145`) before Phase 3 started and proved `area_px / zoom²` (`W/UW = 0.9638`), directly unlocking the `0.94 g` live Dual-Lens pitch demo.
- **Dual-mode counterfactual audit (`focal_only` + `physical_lens_swap`):** Proved that tree models in `C3`/`C4` silently reconstructed lens identity via `area_frac` even under a focal-only flip, automatically disqualifying confounded conditions while certifying `C2_physics_normalized_strict` at `0.0%` shift.
- **Adversarial multi-stage review loop (Plan Checker → Code Reviewer → Independent Verifier → Cross-Phase Integration Checker):** Caught real high-impact defects before they ever compounded — including solution-style `tsconfig.json` checking zero files, `dorsal_bucket` vs. `bucket_dorsal` enum drift, `CR-01` `clahe_tile_grid` list-vs-tuple serialization, React 19 `StrictMode` ref-latch blank screens on `#metricas`/`#watershed`, and non-thread-safe `warnings.catch_warnings()` in `upload_guard.py`.
- **Single-source `PDIConfig` hash lock (`43a3cfaa3208019e`) + opaque `image_id` catalog:** Eliminated offline-vs-online feature skew (`0.0000 g` OOF drift across all 1,738 rows and 6 live presets) while maintaining zero `.jpg` or path leakage in Git (`PROIBIÇÃO 4`).

### What Was Inefficient

- **Stale draft parameter ranges in early context docs:** `04-CONTEXT.md` originally drafted Watershed slider bounds that differed from Phase 2's frozen `backend/app/pdi/config.py` ASVS V5 ranges, requiring explicit reconciliation during Phase 4 planning and Wave 4 implementation.
- **Shared files (`REQUIREMENTS.md`, `backend/app/config.py`) in per-phase `covered_files`:** Including cross-phase evolving files inside Phases 1–3 verification fingerprints caused `init.manager` to report `stale` at milestone close even when each phase's own modules had zero regressions, requiring a closeout regression run and fingerprint refresh.
- **Minor bookkeeping omissions caught at milestone audit:** `04-04-SUMMARY.md` initially omitted `requirements-completed: [WEB-04, WEB-05]` frontmatter after a transient Wave 4 quota interruption forced orchestrator completion, and Tab 2's missing-EXIF banner sat inside a collapsed accordion until pre-close micro-polish.

### Patterns Established

- **Single Pillow EXIF-First Loader + Zero `cv2.imread`/`cv2.imdecode` AST Gate:** Every runtime and test image path decodes via Pillow `ImageOps.exif_transpose()` before OpenCV processing, guarded by automated AST repository checks.
- **Strict Mask-Gated Feature Invariance (`PROIBIÇÃO 2`):** Automated outside-mask pixel-perturbation tests asserting exact `0.0` feature delta prove zero SF-400 LCD or background leakage mathematically on every test run.
- **Dual-Bundle `production_pipeline` + `fold_pipelines[0..4]`:** Any curated pitch or demo preset drawn from training data routes strictly through its held-out fold estimator (`oof_evaluation = true`), keeping live demonstrations scientifically honest.
- **TypeScript 7 + Dual-Project `typecheck` + Biome v2 + Recharts v3 Discipline:** Always run `tsc -p tsconfig.app.json --noEmit && tsc -p tsconfig.node.json --noEmit` (never bare `tsc --noEmit` on solution tsconfigs), avoid deprecated Recharts `<Cell />`, set `isAnimationActive={false}` on dense scatters, and guard FastAPI SPA fallbacks (`_NonApiSpaRoute`) so `/api/*` retains JSON `404`/`405`.

### Key Lessons

1. **Verify physical camera metadata empirically before writing formulas:** Smartphone EXIF `FocalLength` and proprietary `zoomMultiple` (`Xiaomi 39321`) already encode lens switching differently from textbook pinhole assumptions; always validate scaling laws directly on paired same-session captures.
2. **Test counterfactuals with physical co-variation, not single-column flips:** Single-column ablation flips miss secondary proxy columns (`area_frac` vs. `area_px`) that GBDTs exploit to recover camera metadata.
3. **Keep per-phase verification `covered_files` scoped to phase-owned modules:** Excluding monotonically evolving milestone-level files (`.planning/REQUIREMENTS.md`) prevents false staleness during later phases while preserving strict regression coverage via full-suite tests.
4. **In React 19 `StrictMode`, never latch a one-shot `useEffect` fetch ref without resetting it in cleanup:** StrictMode's mount→unmount→remount double-invocation aborts the first fetch and leaves unreset `hasBootedRef` latches permanently blank.

### Cost Observations

- **Execution profile:** High-throughput parallel subagent waves (researchers, planners, plan-checkers, executors, code reviewers, verifiers, integration checker) completed all 4 phases (`14` plans, `26,989` LOC, `298` automated tests) in `4` calendar days (`89` commits) with zero unrecovered rework.
- **Notable efficiency win:** Freezing `PDIConfig` and `models/*.json` contracts early in Phases 1–3 allowed Phase 4's entire FastAPI backend and 4-tab React frontend to ship with `0.0000 g` OOF drift and `56–80 ms` live latency on the very first E2E run.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Days | Phases | Plans | Key Change |
|-----------|------|--------|-------|------------|
| v1.0 | 4 | 4 | 14 | Established empirical data spikes, adversarial Plan-Checker + Code-Reviewer + Verifier + Integration-Checker gates, and `PDIConfig` hash locks |

### Cumulative Quality

| Milestone | Automated Tests | Static Checks | Audit Result |
|-----------|-----------------|---------------|--------------|
| v1.0 | `298` (`272` pytest + `26` Vitest) | `ruff`, dual `tsc`, Biome v2, single-origin E2E | `passed` (`15/15` reqs, `24/24` wiring, `6/6` E2E, `5/5` `PROIBIÇÕES`) |

### Top Lessons (Verified Across Milestones)

1. Empirically validate optical/domain assumptions on paired real samples before locking downstream ML and API contracts.
2. Pair automated anti-leakage tests (`0.0`-diff mask invariance, zero `session_id` fold overlap, dual-mode counterfactual checks) with adversarial post-phase code review to keep scientific claims indisputable.
