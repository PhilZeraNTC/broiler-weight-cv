---
phase: 04-fastapi-local-backend-interactive-pitch-web-studio
plan: "04"
subsystem: frontend-watershed-metrics-e2e
tags:
  - watershed
  - recharts-v3
  - metrics-dashboard
  - single-origin-e2e
  - pt-br-ui
dependency_graph:
  requires:
    - "04-01"
    - "04-02"
    - "04-03"
  provides:
    - "Tab 2 Separação Watershed (WEB-04 / D-04 / D-10)"
    - "Tab 3 Auditoria & Métricas (WEB-05 / D-11)"
    - "Complete 4-tab hash-synced AvianScale CV root SPA"
    - "Verified single-origin FastAPI + compiled frontend/dist/ E2E build"
  affects:
    - "frontend/src/App.tsx"
    - "frontend/src/components/watershed/*"
    - "frontend/src/components/metrics/*"
    - "frontend/src/lib/scatter.ts"
tech_stack:
  added: []
  patterns:
    - "Reconciled backend/app/pdi/config.py ASVS V5 slider bounds mirrored in WatershedSlidersCard"
    - "Filter-before-aggregate 1,738-image to 188-session scatter transformation in pure scatter.ts"
    - "Recharts v3 without deprecated <Cell />, isAnimationActive={false}, min-w-0 explicit heights, actual_g axis"
    - "Single-origin FastAPI serving compiled frontend/dist/ with non-clobbering /api/* JSON 404"
key_files:
  created:
    - frontend/src/components/watershed/WatershedPresetStrip.tsx
    - frontend/src/components/watershed/WatershedStageGrid.tsx
    - frontend/src/components/watershed/WatershedSlidersCard.tsx
    - frontend/src/components/watershed/WatershedInstanceTable.tsx
    - frontend/src/components/watershed/WatershedStudioTab.tsx
    - frontend/src/lib/scatter.ts
    - frontend/src/lib/scatter.test.ts
    - frontend/src/components/metrics/ActualVsPredictedScatter.tsx
    - frontend/src/components/metrics/OpticalAblationChart.tsx
    - frontend/src/components/metrics/FeatureImportanceChart.tsx
    - frontend/src/components/metrics/StratifiedErrorsPanel.tsx
    - frontend/src/components/metrics/ModelComparisonTable.tsx
    - frontend/src/components/metrics/ForensicAuditCard.tsx
    - frontend/src/components/metrics/MetricsAuditTab.tsx
  modified:
    - frontend/src/App.tsx
key_decisions:
  - "Watershed sliders locked verbatim to Phase 2 backend/app/pdi/config.py bounds (h_rel 0.05-0.80 default 0.20, min_distance_rel 0.20-2.50 default 0.85, min_area_ratio 0.01-0.45 default 0.10) with AbortController cancellation."
  - "Scatter dashboard filters 1,738 OOF rows before aggregating into 188 sessions using actual_g and tags sessions as wide/ultrawide/mixed so mixed-lens sessions stay scientifically accurate."
  - "Verified full single-origin production build served by FastAPI TestClient(create_app()) while preserving Phases 1-4 pytest isolation and PROIBIÇÃO 4 git hygiene."
metrics:
  duration: "~15m (resumed after transient Wave 4 quota interruption)"
  completed: "2026-11-24"
  tasks_completed: 3
  tasks_total: 3
  frontend_tests_passed: 26
  backend_tests_passed: 269
---

# Phase 04 Plan 04: Separação Watershed, Auditoria & Métricas, and Full-Stack E2E Summary

Completed Tab 2 (`Separação Watershed`), Tab 3 (`Auditoria & Métricas`), final 4-tab `App.tsx` orchestration, compiled `frontend/dist/` production build, and full-stack single-origin FastAPI E2E + Phases 1–4 regression verification.

## Delivered Scope

- **Task 1 (`d52f346`) — Tab 2 (`Separação Watershed`):** Built `WatershedPresetStrip.tsx` (4 curated presets + `"Enviar Foto de Grupo"` upload dropzone), `WatershedStageGrid.tsx` (2×2 `#0F172A` `object-contain` frames + `StageLightbox` triggers), `WatershedSlidersCard.tsx` (reconciled ASVS V5 bounds from `backend/app/pdi/config.py`: `h_rel [0.05, 0.80]` default `0.20`, `min_distance_rel [0.20, 2.50]` default `0.85`, `min_area_ratio [0.01, 0.45]` default `0.10`), `WatershedInstanceTable.tsx` (Stage 4 color-matched swatches + `0`-bird recovery empty state), and `WatershedStudioTab.tsx` (lazy `ws_417` boot, permanent `group_sum_only` Amber disclaimer, 3 KPI cards, `AbortController`).
- **Task 2 (`d86ac70`) — Scatter Math & Recharts v3 Charts:** Implemented pure filter-before-aggregate `scatter.ts` (`filterScatterPoints`, `aggregateScatterBySession`, `computeR2`) verified by `scatter.test.ts` (`8/8` Vitest), plus `ActualVsPredictedScatter.tsx` (`actual_g`, `domain={[0, 2200]}`, dashed `y = x` `<ReferenceLine>`, `isAnimationActive={false}`, `h-[360px] w-full min-w-0`), `OpticalAblationChart.tsx` (`C1`–`C4` without `<Cell />`), and `FeatureImportanceChart.tsx` (Top-12 horizontal `BarChart` without `<Cell />`).
- **Task 3 — Audit Tables, Forensic Reconciliation, Final 4-Tab `App.tsx` & Single-Origin E2E:** Built `StratifiedErrorsPanel.tsx` (5 Executive Audit KPIs, handheld vs. fixed ceiling-camera `R² > 0,90` benchmark context, tabbed weight-band/lens/scene error tables), `ModelComparisonTable.tsx` (16-row comparison highlighting `LightGBMLog C2` winner and disqualified `C3`/`C4`), `ForensicAuditCard.tsx` (`1.753 → 1.738` curation funnel, Mendeley `"1.772 / 45 dias"` vs. disk audit reconciliation, `801–1.216 g` sparse gap, `0,0`-diff invariance proof), `MetricsAuditTab.tsx`, and complete 4-tab hash-synced `App.tsx`.

## Verification Results

- **Frontend TypeScript (`npm run typecheck`):** `0` errors across `tsconfig.app.json` and `tsconfig.node.json`.
- **Frontend Lint/Format (`biome check src`):** `37` files checked, `0` errors, `0` warnings.
- **Frontend Unit Tests (`vitest run`):** **`26/26` passed** (`scatter.test.ts`: `8`, `studioCore.test.ts`: `18`).
- **Production Bundle (`vite build`):** Emitted `frontend/dist/index.html` (`0.43 kB`) + `dist/assets/*.css` (`27.14 kB`) + `dist/assets/*.js` (`864.61 kB`) in `1.05 s`.
- **Single-Origin FastAPI E2E (`TestClient(create_app())` over real `frontend/dist/`):** `PHASE4_FULLSTACK_E2E_OK GIT_HYGIENE_OK` — `GET /` & `/metricas` returned `200 text/html`; both `/assets/*` bundles returned `200`; `/api/does-not-exist` and URL-encoded `/api/presets/..%2F..%2Fetc%2Fpasswd/thumbnail` returned `404 application/json`; live Dual-Lens delta = **`0.94 g`** (`< 2.0 g`, `< 500 ms`); Watershed `ws_417` returned `bird_count == 2` and `group_sum_only` (`< 500 ms`); `/api/metrics` returned `1,738` `actual_g` points.
- **Full Backend Regression (`uv run pytest -v`):** **`269/269` passed** across Phases 1–4 with `frontend/dist/` present on disk; `uv run ruff check backend/` passed cleanly (`All checks passed!`).
- **Git Hygiene (`PROIBIÇÃO 4`):** Zero tracked or unstaged artifacts in `frontend/dist/`, `node_modules/`, `data/raw/`, `data/processed/`, `models/*.joblib`, or image files.

## Deviations from Plan

- Transient quota exhaustion interrupted the Task 3 subagent immediately after writing the 5 Task 3 source files; resumed in the orchestrator session to run Biome formatting, execute the full Task 3 verification & E2E command, and create the atomic Task 3 and summary commits with zero design or contract deviations.

## Self-Check: PASSED
