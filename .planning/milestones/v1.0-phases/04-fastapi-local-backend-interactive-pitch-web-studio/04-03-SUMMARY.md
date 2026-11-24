---
phase: 04-fastapi-local-backend-interactive-pitch-web-studio
plan: 03
subsystem: ui
tags: [react-19, typescript-7, tailwind-v4, xray-pdi, dual-lens-proof, exif-calibration, flock-simulator, future-vision-v2, pt-br]

requires:
  - phase: 04-fastapi-local-backend-interactive-pitch-web-studio
    provides: "FastAPI /api/predict and /api/presets/{preset_id}/thumbnail routes (04-01); Vite 8 / React 19 / TS 7 / Tailwind v4 scaffold, strict api.ts contracts, studio hooks, and AppShell/StageLightbox/InlineErrorBanner primitives (04-02)"
provides:
  - "Complete Tab 1 Estúdio Raio-X workspace (9 modular React components under frontend/src/components/studio/) with zero-empty-state sb_intermediate_grower auto-boot, 6 curated OOF presets, 1-click Dual-Lens 541 g optical invariance comparator (resolving 04-RESEARCH.md Open Question 1), drag-and-drop/click 15 MiB image uploader, 2x2 dark #0F172A X-Ray viewer + StageLightbox, 32px weight/OOF Fold K audit card, bounded EXIF override + missing-EXIF Amber disclosure, 39-feature telemetry drawer, and client-side Flock Simulator with inline 2-step clear confirmation at >= 2 birds"
  - "100% static conceptual Tab 4 Visão Futura (v2) workspace (frontend/src/components/vision/FutureVisionTab.tsx) enforcing PROIBIÇÃO 5 with permanent Amber governance banner, 5-axis MVP v1 vs. v2 architectural matrix, and BARN-01..03 roadmap cards"
  - "Live hash-synced frontend/src/App.tsx wiring Tab 1 (#estudio, kept mounted via CSS hidden state across tab switches) and Tab 4 (#visao-futura) with live latency propagation and clean Wave 4 mount slots for #watershed and #metricas"
affects: [04-04, frontend-studio, frontend-vision, pitch-demo]

actuals:
  tokens: 26765
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Auto-boot live pitch landing state (sb_intermediate_grower) executed through useAbortableRequest with AbortController cancellation on superseded preset/upload/override calls"
    - "Parallel cached Dual-Lens OOF proof (Promise.all over sb_dual_lens_wide + sb_dual_lens_ultrawide) contrasting raw area_px swing against stabilized area_px / zoom² and 0,94 g live delta"
    - "Tab-preservation mounting pattern in App.tsx (CSS hidden + ARIA hidden on #panel-estudio) keeping active X-Ray prediction, Dual-Lens cache, and simulated flock intact across tab navigation"
    - "Strictly static declarative documentation boundary (PROIBIÇÃO 5) on FutureVisionTab.tsx with zero API imports, zero timers, and zero mock telemetry"

key-files:
  created:
    - frontend/src/components/studio/PresetsStrip.tsx
    - frontend/src/components/studio/DualLensComparison.tsx
    - frontend/src/components/studio/UploadDropzone.tsx
    - frontend/src/components/studio/XRayQuadViewer.tsx
    - frontend/src/components/studio/PredictionSummaryCard.tsx
    - frontend/src/components/studio/OpticalCalibrationCard.tsx
    - frontend/src/components/studio/TelemetryPanel.tsx
    - frontend/src/components/studio/FlockBatchCard.tsx
    - frontend/src/components/studio/XRayStudioTab.tsx
    - frontend/src/components/vision/FutureVisionTab.tsx
  modified:
    - frontend/src/App.tsx

key-decisions:
  - "Resolved 04-RESEARCH.md Open Question 1 via DualLensComparison.tsx: 1-click 'Comparar Par Dual-Lens' button in PresetsStrip launches parallel predictSingleBird calls for sb_dual_lens_wide and sb_dual_lens_ultrawide, caches results in React state, displays side-by-side dark #0F172A Stage 4 overlays, and highlights the live OOF delta in Emerald #047857"
  - "Kept XRayStudioTab mounted inside a CSS-hidden tabpanel in App.tsx so switching to Visão Futura (v2) or Wave 4 tabs preserves the presenter's active X-Ray analysis, manual EXIF override state, and simulated flock batch"
  - "Added role='img' to the chromatic plumage ratio bar in TelemetryPanel.tsx and aria-hidden='true' to decorative selection dots in PresetsStrip.tsx to satisfy Biome v2 strict ARIA rules without losing screen-reader context"

patterns-established:
  - "Strict 4-size (12px/14px/20px/32px), 2-weight (400/600), tabular-nums clinical-light visual hierarchy with dark #0F172A object-contain viewports exclusively behind CV frames"
  - "Closed 7-element Accent #047857 allowlist enforced across primary CTAs, active nav tab, selected preset ring/dot, focus ring, and verified SLA/invariance badges"
  - "Zero rel_path, zero raw dataset .jpg filenames, zero dangerouslySetInnerHTML, and safe truncated React text rendering (truncate max-w-[240px]) for uploaded filenames"

requirements-completed: [WEB-02, WEB-03, WEB-04]

coverage:
  - id: D1
    description: "Tab 1 presets gallery (6 curated OOF presets with opaque /api/presets/{preset_id}/thumbnail thumbnails + fallback), highlighted 541 g Dual-Lens tandem with live side-by-side comparator (DualLensComparison), drag-and-drop 15 MiB UploadDropzone, 2x2 dark #0F172A XRayQuadViewer with 44x44 lightbox triggers, and 32px Hero PredictionSummaryCard with OOF Fold K audit"
    requirement: "WEB-03"
    verification:
      - kind: other
        ref: "cd frontend && npm run typecheck && npm run lint && npm test && npm run build"
        status: pass
    human_judgment: false
  - id: D2
    description: "Tab 1 EXIF OpticalCalibrationCard (missing-EXIF Amber disclosure, Wide/Ultrawide quick presets, bounded [0.5, 200] mm / [0.1, 20]x override validation), mask-gated TelemetryPanel (morphometrics, plumage chroma bar, expandable 39-feature font-mono table), client-side FlockBatchCard (0/1/N states, inline 2-step clear confirmation at >= 2 birds), and auto-booting XRayStudioTab orchestrator"
    requirement: "WEB-02"
    verification:
      - kind: other
        ref: "cd frontend && npm run typecheck && npm run lint && npm test && npm run build"
        status: pass
    human_judgment: false
  - id: D3
    description: "100% static conceptual Tab 4 FutureVisionTab (PROIBIÇÃO 5 Amber governance banner, 5-axis MVP v1 vs. v2 comparison matrix, BARN-01..03 cards, zero API/timer imports) wired alongside live Tab 1 inside hash-synced App.tsx producing frontend/dist/index.html"
    requirement: "WEB-04"
    verification:
      - kind: other
        ref: "cd frontend && npm run typecheck && npm run lint && npm test && npm run build && test -f dist/index.html"
        status: pass
    human_judgment: false

duration: 11min
completed: 2026-11-24
status: complete
---

# Phase 4 Plan 03: Tab 1 (`Estúdio Raio-X`) & Static Tab 4 (`Visão Futura (v2)`) Summary

**Live classical PDI X-Ray Studio with 6 honest OOF presets, 1-click Dual-Lens optical invariance proof, EXIF override, 39-feature telemetry, simulated flock uniformity tracker, and 100% static v2 barn roadmap wired into `App.tsx`**

## Performance

- **Duration:** 11 min
- **Started:** 2026-11-24T10:18:34Z
- **Completed:** 2026-11-24T10:30:00Z
- **Tasks:** 3
- **Files modified:** 10 created, 1 modified (11 files total)

## Accomplishments

- Built the complete **Estúdio Raio-X** (`WEB-02`, `WEB-03`) across 9 modular components under `frontend/src/components/studio/`:
  - `PresetsStrip.tsx` renders all 6 curated OOF presets with opaque `/api/presets/{preset_id}/thumbnail` dark frames (with resilient icon fallback on `onError`), Ground Truth + Fold badges, `2px` `#047857` active selection ring/dot, and the highlighted **Par Dual-Lens — Mesmo Frango (541 g · Fold 0)** tandem container with a 1-click **"Comparar Par Dual-Lens"** CTA.
  - `DualLensComparison.tsx` (resolving `04-RESEARCH.md` Open Question 1) runs parallel `predictSingleBird` calls for `sb_dual_lens_wide` and `sb_dual_lens_ultrawide`, caches the pair in React state, renders side-by-side dark `#0F172A` Stage 4 overlays, contrasts raw `area_px` swing vs. stabilized `area_optical_norm = area_px / zoom²`, and highlights the live OOF delta (`0,94 g` / `0,24%`) in Emerald `#047857`.
  - `UploadDropzone.tsx`, `XRayQuadViewer.tsx`, and `PredictionSummaryCard.tsx` deliver client-guarded `JPG/PNG/WEBP <= 15 MB` upload with safe filename truncation (`truncate max-w-[240px]`), 2×2 dark `#0F172A` `object-contain` X-Ray frames with `44×44` `pt-BR` lightbox triggers and non-blocking loading overlay, and the `32px` predicted weight card with `±MAE`/`p68`, Cobb Avian 48 age stage, `< 500 ms` SLA indicator, conditional Amber `out_of_training_range` alert, and held-out **Auditoria OOF (Fold K)** audit strip.
  - `OpticalCalibrationCard.tsx`, `TelemetryPanel.tsx`, `FlockBatchCard.tsx`, and `XRayStudioTab.tsx` deliver EXIF detection + missing-EXIF Amber fallback disclosure + bounded manual override controls (`[0.5, 200.0] mm`, `[0.1, 20.0]×`), mask-gated morphometric/chromatic telemetry + expandable 39-feature `font-mono` table, the client-side Flock Simulator (`0` empty / `1` `"—"` `"Adicione ≥ 2 aves"` / `>= 2` inline 2-step `"Confirmar Limpeza do Lote"` `#B91C1C` & `"Manter Lote Atual"`), and automatic zero-empty-state boot on `sb_intermediate_grower` via `useAbortableRequest`.
- Built **Visão Futura (v2)** (`frontend/src/components/vision/FutureVisionTab.tsx`, `WEB-04` / `D-12` / `PROIBIÇÃO 5`) as 100% static declarative documentation with zero API imports, zero timers, and zero fake telemetry, rendering the verbatim Amber `PROIBIÇÃO 5` governance banner, the 5-axis MVP v1 vs. Arquitetura v2 comparison matrix inside `overflow-x-auto`, and the 3 `Conceitual v2` roadmap cards (`BARN-01`, `BARN-02`, `BARN-03`).
- Wired live `XRayStudioTab` (`#estudio`, preserved via CSS hidden state across tab navigation) and `FutureVisionTab` (`#visao-futura`) into `frontend/src/App.tsx` with latency reporting to `AppShell` and clean Wave 4 mount slots for `#watershed` and `#metricas`, passing dual-project `npm run typecheck`, Biome v2 `npm run lint`, Vitest (`18/18`), and Vite 8 `npm run build` (`frontend/dist/index.html` built in `509 ms`).

## Task Commits

Each task was committed atomically:

1. **Task 1: Build Tab 1 input, presets gallery, Dual-Lens optical proof, 2x2 X-Ray viewport, and Hero Prediction/OOF card** - `69ca8ea` (feat)
2. **Task 2: Build Tab 1 EXIF Calibration Override card, 39-feature Telemetry panel, Simulated Flock card, and `XRayStudioTab.tsx` orchestrator** - `c31fd66` (feat)
3. **Task 3: Build static Tab 4 (`Visão Futura (v2)` — WEB-04 / D-12 / PROIBIÇÃO 5), wire Tabs 1 & 4 into `App.tsx`, and verify production Wave 3 build** - `b55edaf` (feat)

## Files Created/Modified

- `frontend/src/components/studio/PresetsStrip.tsx` - 6 curated OOF preset cards with opaque thumbnail URLs, graceful broken-image fallback, and highlighted 541 g Dual-Lens tandem group with `"Comparar Par Dual-Lens"` CTA.
- `frontend/src/components/studio/DualLensComparison.tsx` - Parallel cached `sb_dual_lens_wide` vs. `sb_dual_lens_ultrawide` side-by-side optical invariance proof panel.
- `frontend/src/components/studio/UploadDropzone.tsx` - Drag-and-drop/click uploader enforcing `validateClientUploadFile` (`15 MiB` + `{JPG, PNG, WEBP}`) with safe truncated filename display.
- `frontend/src/components/studio/XRayQuadViewer.tsx` - 2×2 dark `#0F172A` `object-contain` X-Ray viewer with exact stage labels, `44×44` lightbox triggers, non-blocking loading overlay, and `buildXRayLightboxStages` helper.
- `frontend/src/components/studio/PredictionSummaryCard.tsx` - `32px` predicted weight hero card (`g`/`kg`), `±MAE`/`p68`, Cobb Avian 48 age stage, `< 500 ms` latency badge, conditional Amber range warning, and honest **Auditoria OOF (Fold K)** comparison grid.
- `frontend/src/components/studio/OpticalCalibrationCard.tsx` - EXIF focal/zoom/lens readout, accessible `44×44` `Info` disclosure, missing-EXIF Amber fallback notice, Wide/Ultrawide quick presets, and bounded manual override form.
- `frontend/src/components/studio/TelemetryPanel.tsx` - Mask-gated morphometric descriptors, chromatic yellow-down vs. white-plumage visual bar, scene & LCD exclusion readout, `segmentation_fallback` notice, and expandable 39-feature `font-mono` table.
- `frontend/src/components/studio/FlockBatchCard.tsx` - Client-side simulated flock uniformity accumulator (`0`/`1`/`N` states, pt-BR pluralization, scrollable `max-h-72` list, `44×44` remove triggers, and inline 2-step clear confirmation at `>= 2` birds).
- `frontend/src/components/studio/XRayStudioTab.tsx` - Complete Tab 1 orchestrator wiring auto-boot `sb_intermediate_grower`, abort-safe `/api/predict`, Dual-Lens comparator, lightbox, EXIF overrides, telemetry, and simulated flock.
- `frontend/src/components/vision/FutureVisionTab.tsx` - 100% static conceptual v2 barn roadmap tab (`PROIBIÇÃO 5` Amber banner, 5-axis v1 vs. v2 matrix, and `BARN-01`..`BARN-03` cards).
- `frontend/src/App.tsx` - Updated 4-tab hash-synced shell mounting live `XRayStudioTab` (preserved across tab switches) and `FutureVisionTab` alongside Wave 4 slots for `#watershed` and `#metricas`.

## Decisions Made

- Kept `XRayStudioTab` mounted inside a CSS-hidden tab panel in `App.tsx` so switching tabs during a live pitch preserves the active X-Ray prediction, Dual-Lens comparison cache, and simulated flock state without triggering redundant `/api/predict` calls.
- Kept the Dual-Lens trigger CTA text strictly `"Comparar Par Dual-Lens"` with `aria-pressed` / `aria-expanded` toggling Accent `#047857` fill strictly when active, matching `04-UI-SPEC.md` Accent Allowlist item #1 verbatim.
- Added `role="img"` to the chromatic plumage ratio bar in `TelemetryPanel.tsx` and `aria-hidden="true"` to the decorative selected-preset dot in `PresetsStrip.tsx` to maintain 100% clean Biome v2 ARIA compliance.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Wave 4 (`04-04-PLAN.md` — Tab 2 `Separação Watershed` & Tab 3 `Auditoria & Métricas` + single-origin end-to-end verification) can now drop `WatershedStudioTab` and `MetricsAuditTab` directly into the prepared `#panel-watershed` and `#panel-metricas` slots in `frontend/src/App.tsx`.

## Self-Check: PASSED

- All 11 created/modified files verified on disk and committed atomically across `69ca8ea`, `c31fd66`, and `b55edaf`.
- `npm run typecheck` (`tsc -p tsconfig.app.json --noEmit && tsc -p tsconfig.node.json --noEmit`), `npm run lint` (`biome check src`, 23 files, 0 diagnostics), `npm test` (`18/18` passed), and `npm run build` (`frontend/dist/index.html` emitted in `509 ms`) all exited `0`.
- Automated compliance audit verified 42 verbatim `04-UI-SPEC.md` copy/ARIA/roadmap strings, strictly 4 font sizes (`12px`/`14px`/`20px`/`32px`) & 2 font weights (`400`/`600`), zero `dangerouslySetInnerHTML`, zero `rel_path`, zero raw dataset `.jpg` filenames, and zero `fetch`/hooks/timers in `FutureVisionTab.tsx` (`PROIBIÇÃO 4 & 5`).

---
*Phase: 04-fastapi-local-backend-interactive-pitch-web-studio*
*Completed: 2026-11-24*
