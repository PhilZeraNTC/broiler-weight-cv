---
phase: 04-fastapi-local-backend-interactive-pitch-web-studio
plan: 01
subsystem: api
tags: [fastapi, pydantic-v2, uvicorn, pillow, opencv, lightgbm, watershed, asvs-l1, pytest]

requires:
  - phase: 01-forensic-data-curation-anti-leakage-split
    provides: "Typed manifest Parquets (manifest_clean.parquet, manifest_quarantine.parquet), dataset_audit_summary.json, and weight-band boundaries"
  - phase: 02-classical-pdi-engine-optical-calibration-watershed
    provides: "Single-source classical PDI engine (run_single_bird_pdi, run_watershed_pdi, load_working_rgb, PDIConfig hash 43a3cfaa3208019e, authoritative Watershed bounds)"
  - phase: 03-tabular-regression-optical-ablation-production-bundling
    provides: "Dual-bundle models/weight_regressor.joblib (production_pipeline + fold_pipelines 0..4), evaluation_report.json, presets_catalog.json, predict_from_feature_dict, and map_weight_to_age_stage"
provides:
  - "Modular local FastAPI REST API under backend/app/api/ serving /api/health, /api/presets, /api/presets/{preset_id}/thumbnail, /api/predict, /api/watershed-demo, and /api/metrics"
  - "Fail-fast lifespan startup locking PDIConfig SHA-256 hash, cv2.setNumThreads(2), ASVS V4 non-symlink preset path resolution, and 10 pre-warmed 320 px JPEG thumbnails in RAM"
  - "Dual-bundle honest OOF preset scoring (fold_pipelines[fold]) vs. upload scoring (production_pipeline) via async def + starlette.concurrency.run_in_threadpool"
  - "ASVS L1 upload guard (15 MiB cap, Pillow magic JPEG/PNG/WEBP allowlist, 40 MP decompression-bomb rejection), localhost-only CORS, sanitized pt-BR HTTP 500 handler, and non-clobbering frontend/dist SPA fallback"
  - "64 isolated FastAPI contract, ASVS L1 security, PROIBIÇÃO 1/2/3/4/5, OOF parity, and < 500 ms SLA pytest tests"
affects: [04-02, 04-03, 04-04, frontend-api-client, single-origin-pitch-serving]

actuals:
  tokens: 27627
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "FastAPI lifespan AppState frozen slotted container injected via Annotated[AppState, Depends(get_app_state)]"
    - "Async bounded UploadFile intake + run_in_threadpool CPU offload for Pillow/OpenCV/scikit-image/LightGBM"
    - "Authoritative single-source Watershed bounds imported verbatim from backend.app.pdi.config"
    - "Non-clobbering optional frontend/dist SPA mount guarding api and api/* to preserve JSON 404"
    - "API-only pytest isolation via create_app(frontend_dist_dir=tmp_path_factory.mktemp('no_dist'))"

key-files:
  created:
    - backend/app/api/__init__.py
    - backend/app/api/schemas.py
    - backend/app/api/dependencies.py
    - backend/app/api/main.py
    - backend/app/api/services/__init__.py
    - backend/app/api/services/upload_guard.py
    - backend/app/api/services/preset_service.py
    - backend/app/api/services/inference_service.py
    - backend/app/api/services/watershed_service.py
    - backend/app/api/services/metrics_service.py
    - backend/app/api/routers/__init__.py
    - backend/app/api/routers/health.py
    - backend/app/api/routers/presets.py
    - backend/app/api/routers/predict.py
    - backend/app/api/routers/watershed.py
    - backend/app/api/routers/metrics.py
    - backend/tests/test_api_contract.py
    - backend/tests/test_api_security.py
    - backend/tests/test_api_latency_and_parity.py
  modified:
    - backend/app/config.py

key-decisions:
  - "Reconciled and locked Watershed slider bounds in schemas.py and watershed_service.py verbatim from backend.app.pdi.config (h_rel [0.05, 0.80] default 0.20, min_distance_rel [0.20, 2.50] default 0.85, min_area_ratio [0.01, 0.45] default 0.10), superseding narrower draft bounds in 04-CONTEXT.md and matching approved 04-UI-SPEC.md"
  - "Used Annotated[..., Depends/File/Form] parameters across all FastAPI routers to satisfy Ruff B008 cleanly without inline noqa suppressions"
  - "Added Pillow im.verify() inside upload_guard.validate_image_bytes() alongside decode-time fallback mapping in inference/watershed services so truncated or structurally corrupt uploads consistently return sanitized pt-BR HTTP 400"
  - "Guarded serve_spa with full_path.lstrip('/') check for 'api' and 'api/' prefixes so unknown /api/* paths and URL-encoded traversal attempts always return JSON 404 even when frontend/dist/index.html is mounted"

patterns-established:
  - "Strict _StrictSchema(BaseModel) with ConfigDict(extra='forbid') and FiniteFloat (allow_inf_nan=False) on all API models"
  - "RFC 2397 Data URI assembly server-side (JPEG for photographic/overlay stages, 1-channel PNG for X-Ray stage3_mask and Watershed stage2_union_mask)"
  - "Fixture isolation from frontend/dist/ via create_app(frontend_dist_dir=no_dist) plus dedicated tmp_path SPA fallback test clients"

requirements-completed: [WEB-01]

coverage:
  - id: D1
    description: "Modular FastAPI package under backend/app/api/ with lifespan startup lock (pdi_config_hash 43a3cfaa3208019e, cv2.setNumThreads(2)), health, presets, pre-warmed 320 px JPEG thumbnails, and startup-cached metrics endpoints"
    requirement: "WEB-01"
    verification:
      - kind: integration
        ref: "backend/tests/test_api_contract.py#test_health_endpoint_contract"
        status: pass
      - kind: integration
        ref: "backend/tests/test_api_contract.py#test_presets_endpoint_contract_and_zero_rel_path"
        status: pass
      - kind: integration
        ref: "backend/tests/test_api_contract.py#test_all_ten_preset_thumbnails_contract"
        status: pass
      - kind: integration
        ref: "backend/tests/test_api_contract.py#test_metrics_endpoint_contract"
        status: pass
    human_judgment: false
  - id: D2
    description: "Single-bird POST /api/predict enforcing file XOR preset_id, honest fold_pipelines[fold] OOF parity (<= 0.01 g across all 6 presets, Dual-Lens delta 0.94 g), EXIF/stripped/RGBA/WEBP upload scoring, manual optical override, RFC 2397 X-Ray Data URIs, and warm p95 83.0 ms (< 500 ms SLA)"
    requirement: "WEB-01"
    verification:
      - kind: integration
        ref: "backend/tests/test_api_contract.py#test_predict_upload_formats_exif_stripped_rgba_webp"
        status: pass
      - kind: integration
        ref: "backend/tests/test_api_contract.py#test_predict_manual_optical_override"
        status: pass
      - kind: integration
        ref: "backend/tests/test_api_latency_and_parity.py#test_all_six_single_bird_presets_oof_parity_dual_lens_and_latency"
        status: pass
      - kind: integration
        ref: "backend/tests/test_api_latency_and_parity.py#test_concurrent_requests_non_blocking_threadpool"
        status: pass
    human_judgment: false
  - id: D3
    description: "Multi-bird POST /api/watershed-demo with authoritative Phase 2 ASVS V5 slider bounds, per-instance production_pipeline scoring, Stage 4 hex swatches, 0-bird null-mean handling, exact default count parity (2, 3, 4, 2) in 142.5-192.2 ms, and permanent group_sum_only pt-BR disclaimer"
    requirement: "WEB-01"
    verification:
      - kind: integration
        ref: "backend/tests/test_api_contract.py#test_watershed_presets_contract_and_hex_swatches"
        status: pass
      - kind: integration
        ref: "backend/tests/test_api_contract.py#test_watershed_custom_upload_and_slider_overrides"
        status: pass
      - kind: integration
        ref: "backend/tests/test_api_contract.py#test_watershed_zero_bird_uniform_dark_image"
        status: pass
      - kind: integration
        ref: "backend/tests/test_api_latency_and_parity.py#test_all_four_watershed_presets_count_parity_sum_consistency_and_latency"
        status: pass
    human_judgment: false
  - id: D4
    description: "ASVS L1 security (15 MiB 413, disallowed format 415, corrupt/40 MP bomb 400, parameter 422, non-symlink containment, URL-encoded & form-body traversal 404 JSON under both no_dist and active dist/index.html, localhost-only CORS, sanitized HTTP 500) and PROIBIÇÃO 1/2/3/4/5 AST + response leak audit"
    requirement: "WEB-01"
    verification:
      - kind: integration
        ref: "backend/tests/test_api_security.py#test_upload_guard_rejections"
        status: pass
      - kind: integration
        ref: "backend/tests/test_api_security.py#test_traversal_and_nonexistent_api_404_with_active_spa_dist"
        status: pass
      - kind: integration
        ref: "backend/tests/test_api_security.py#test_proibicao4_zero_leak_scan_across_all_responses"
        status: pass
      - kind: unit
        ref: "backend/tests/test_api_security.py#test_proibicoes_1_2_3_5_ast_and_route_audit"
        status: pass
      - kind: integration
        ref: "backend/tests/test_api_contract.py#test_spa_fallback_serves_html_and_preserves_api_404"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-11-24
status: complete
---

# Phase 4 Plan 01: FastAPI Local Backend Summary

**Modular `< 500 ms` local FastAPI serving backend (`backend/app/api/`) exposing EXIF-calibrated classical PDI X-Ray inference (`p95 = 83.0 ms`), honest out-of-fold curated presets (`0.0000 g` catalog parity, `0.94 g` Dual-Lens invariance), Marker-based Watershed separation (`p95 = 185.9 ms`), cached `320 px` JPEG thumbnails, startup-cached 5-fold audit metrics, ASVS L1 security controls, and non-clobbering single-origin SPA mounting.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-11-24T09:42:17Z
- **Completed:** 2026-11-24T09:57:31Z
- **Tasks:** 3 / 3
- **Files modified:** 20 (19 created, 1 modified)
- **Measured Warm CPU Latency (`cv2.setNumThreads(2)`):**
  - `POST /api/predict` (all 6 curated single-bird OOF presets): `min = 59.7 ms`, `p50 = 63.2 ms`, **`p95 = 83.0 ms`**, `max = 83.9 ms` (`< 500 ms` SLA, `< 400 ms` warm p95 target)
  - `POST /api/watershed-demo` (all 4 curated multi-bird presets): `min = 142.5 ms`, `p50 = 148.8 ms`, **`p95 = 185.9 ms`**, `max = 192.2 ms` (`< 500 ms` SLA)
  - Live Dual-Lens proof (`sb_dual_lens_wide` vs. `sb_dual_lens_ultrawide`, Fold `0`): `383.2180 g` vs. `384.1577 g` (**`0.9397 g` delta**, `< 2.0 g` invariant)

## Accomplishments

- Extended `backend/app/config.py` with `FRONTEND_DIST_DIR` anchored to `REPO_ROOT / "frontend" / "dist"` and built the complete modular `backend/app/api/` package (`schemas.py`, `dependencies.py`, `main.py`, 5 service modules, and 5 REST routers).
- Implemented fail-fast `lifespan` startup setting `cv2.setNumThreads(2)`, hash-validating `models/weight_regressor.joblib` (`pdi_config_hash = 43a3cfaa3208019e`), verifying all 4 audit JSONs against `PROIBIÇÃO 4`, joining opaque preset IDs against `manifest_clean.parquet` and `manifest_quarantine.parquet` with ASVS V4 non-symlink path containment, and pre-warming all 10 EXIF-transposed `320 px` JPEG thumbnails in RAM.
- Enforced strict `file` XOR `preset_id` validation (`422`), bounded `15 MiB` async upload reads (`413`), real Pillow magic allowlisting for JPEG/PNG/WEBP (`415`), `40 MP` decompression-bomb & structural verification guards (`400`), ASVS V5 finite optical and Watershed slider bounds (`422`), strict localhost-only CORS, sanitized pt-BR HTTP `500` handling, and guarded optional `frontend/dist/` SPA fallback preserving JSON `404` on `/api/*` and URL-encoded traversal attempts.
- Delivered 64 new API tests across `test_api_contract.py` (13), `test_api_security.py` (48), and `test_api_latency_and_parity.py` (3); full repository suite passes **269 / 269** (`uv run pytest`) with `uv run ruff check backend/` exiting `0`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Core FastAPI backend vertical tracer — config, Pydantic v2 schemas, AppState, upload guard, preset/thumbnail allowlist, and honest OOF + upload inference service** - `4a7ecae` (feat)
2. **Task 2: Build Watershed & Metrics services, all 5 FastAPI routers, and `main.py` (lifespan, localhost CORS, sanitized 500 handler, guarded SPA fallback)** - `e8b3ca2` (feat)
3. **Task 3: Build isolated contract, ASVS L1 security & PROIBIÇÃO 1/2/4/5, and OOF parity + < 500 ms latency pytest suites** - `bae1955` (test)

## Files Created/Modified

- `backend/app/config.py` — Added `FRONTEND_DIST_DIR` constant anchored to `REPO_ROOT`.
- `backend/app/api/__init__.py` & `backend/app/api/services/__init__.py` — Lightweight package markers.
- `backend/app/api/schemas.py` — Strict Pydantic v2 response/payload schemas (`extra="forbid"`, `allow_inf_nan=False`, zero `rel_path`, importing authoritative bounds from `backend.app.pdi.config` and `backend.app.pdi.io_exif` plus `SceneType` from `backend.app.pdi.segmentation`).
- `backend/app/api/dependencies.py` — Frozen slotted `AppState` container and `get_app_state` FastAPI dependency.
- `backend/app/api/services/upload_guard.py` — Bounded async `15 MiB` upload reader (`413`) and Pillow magic/40 MP decompression-bomb validator (`415`/`400`).
- `backend/app/api/services/preset_service.py` — Startup manifest join, ASVS V4 non-symlink path containment via `_resolve_safe_image_path`, and pre-warmed `320 px` JPEG thumbnail cache.
- `backend/app/api/services/inference_service.py` — Single-bird classical PDI + dual-bundle honest OOF (`fold_pipelines[fold]`) vs. upload (`production_pipeline`) scoring, Cobb Avian 48 age-stage mapping, band error intervals, and RFC 2397 Data URI assembly.
- `backend/app/api/services/watershed_service.py` — Isolated Marker-based Watershed service, reconciled Phase 2 ASVS V5 parameter validator, Stage 4 BGR-to-hex swatch palette (`BASIN_PALETTE_HEX`), per-instance scoring, `0`-bird null-mean handling, and permanent `group_sum_only` disclaimer.
- `backend/app/api/services/metrics_service.py` — Startup audit JSON loader with `pdi_config_hash` parity, `MetricsResponse` validation, and `PROIBIÇÃO 4` leak scanning.
- `backend/app/api/routers/{__init__,health,presets,predict,watershed,metrics}.py` — All 5 FastAPI routers using `Annotated` dependencies and `run_in_threadpool` CPU offloading.
- `backend/app/api/main.py` — `create_app()`, `lifespan`, strict `LOCALHOST_CORS_ORIGINS` middleware, sanitized pt-BR HTTP `500` exception handler, and guarded optional `frontend/dist/` SPA fallback.
- `backend/tests/test_api_contract.py` — 13 endpoint contract & SPA fallback tests (`422` lines).
- `backend/tests/test_api_security.py` — 48 ASVS L1 upload/parameter/traversal/CORS/500 & `PROIBIÇÃO 1/2/3/4/5` AST/response leak tests (`448` lines).
- `backend/tests/test_api_latency_and_parity.py` — 3 OOF parity, Watershed count parity, `< 500 ms` SLA, and concurrent threadpool tests (`163` lines).

## Decisions Made

- Imported `WATERSHED_MIN_*` / `WATERSHED_MAX_*` directly from `backend.app.pdi.config` so backend validation and response `parameters.bounds` match Phase 2 and approved `04-UI-SPEC.md` verbatim (`h_rel [0.05, 0.80]`, `min_distance_rel [0.20, 2.50]`, `min_area_ratio [0.01, 0.45]`).
- Used FastAPI's `app.openapi()["paths"]` combined with recursive route traversal in `test_proibicoes_1_2_3_5_ast_and_route_audit` because FastAPI `0.165.0` stores included routers as `_IncludedRouter` wrappers inside `app.routes`.
- Filtered Starlette `1.10.0`'s upstream `StarletteDeprecationWarning` (`UserWarning` subclass recommending `httpx2`) narrowly around `TestClient` import inside test helpers while keeping locked `pyproject.toml` untouched (`PROIBIÇÃO 1`).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] FastAPI 0.165.0 `_IncludedRouter` wrapper during route audit test**
- **Found during:** Task 3 (`test_api_security.py`)
- **Issue:** Inspecting `app.routes` directly returned `_IncludedRouter` wrapper objects without top-level `.path`, causing `api_paths` in the `PROIBIÇÃO 5` route audit test to evaluate empty.
- **Fix:** Audited both `set(app.openapi()["paths"].keys())` and recursively traversed nested router `.routes` collections to verify the exact 6 `/api/*` endpoints and zero banned v2 route terms.
- **Files modified:** `backend/tests/test_api_security.py`
- **Verification:** `uv run pytest backend/tests/test_api_security.py -v` passed `48/48`.
- **Committed in:** `bae1955` (Task 3 commit)

**2. [Rule 2 - Missing Critical] Leading-slash normalization in SPA catch-all guard**
- **Found during:** Task 2 (`backend/app/api/main.py`)
- **Issue:** Raw double-slash paths (`//api/...`) could pass a leading `/` into `full_path` if not stripped before checking `api` / `api/`.
- **Fix:** Normalized `normalized = full_path.lstrip("/")` before raising JSON `404` for `normalized == "api" or normalized.startswith("api/")`.
- **Files modified:** `backend/app/api/main.py`
- **Verification:** Verified via Task 2 SPA traversal probe and `test_api_security.py` SPA-enabled traversal suite.
- **Committed in:** `e8b3ca2` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug in test route introspection, 1 defensive SPA guard hardening)
**Impact on plan:** Both auto-fixes strengthened verification accuracy and security defense-in-depth with zero scope creep.

## Issues Encountered

- Task 2's inline `<verify>` snippet in `04-01-PLAN.md` contained a minor typo (`sc.headers['content-type']` on the `TestClient` instance rather than the response object); verified the exact same assertions against the returned response objects (`r_root.headers['content-type']`, `r_est.headers['content-type']`) and covered them permanently in `backend/tests/test_api_contract.py` and `backend/tests/test_api_security.py`.
- Upstream `scikit-image` `grayreconstruct.py:216` emits NumPy `2.5` `DeprecationWarning` during Watershed `h_maxima` execution (same upstream library notice seen in Phase 2/3 tests; zero warnings originate from AvianScale CV code).

## User Setup Required

None — no external service configuration required. Launch locally via:

```bash
uv run uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

## Next Phase Readiness

- All 6 FastAPI endpoints (`/api/health`, `/api/presets`, `/api/presets/{preset_id}/thumbnail`, `/api/predict`, `/api/watershed-demo`, `/api/metrics`) and optional single-origin `frontend/dist/` static SPA serving are complete, verified, and ready for Wave 1–4 frontend integration (`04-02`, `04-03`, `04-04`).
- No blockers or regressions across Phases 1–4 (`269/269` pytest passing, `ruff check backend/` clean).

## Self-Check: PASSED

- Verified all 19 created files and 1 modified file exist on disk and are tracked cleanly in git (`4a7ecae`, `e8b3ca2`, `bae1955`).
- Verified `uv run pytest backend/tests/test_api_contract.py backend/tests/test_api_security.py backend/tests/test_api_latency_and_parity.py -v` (`64 passed`), full `uv run pytest` (`269 passed`), and `uv run ruff check backend/` (`All checks passed!`).
- Verified zero raw dataset filenames or absolute user home paths appear in `04-01-SUMMARY.md` or any API response.

---
*Phase: 04-fastapi-local-backend-interactive-pitch-web-studio*
*Completed: 2026-11-24*
