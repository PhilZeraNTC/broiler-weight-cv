---
phase: 04-fastapi-local-backend-interactive-pitch-web-studio
plan: 02
subsystem: ui
tags: [react-19, typescript-7, vite-8, tailwind-v4, biome-v2, vitest, fetch-api, accessibility, pt-br]

requires:
  - phase: 04-fastapi-local-backend-interactive-pitch-web-studio
    provides: "Strict Pydantic v2 REST API contracts in backend/app/api/schemas.py and live /api/{health,presets,predict,watershed-demo,metrics} routes (04-01)"
provides:
  - "Pinned greenfield frontend workspace under frontend/ (React 19.3.1, TypeScript 7.0.2, Vite 8.3.8, Tailwind CSS 4.3.5, Recharts 3.11.1 + react-is 19.3.1, Lucide 1.79.0, Biome 2.6.8, Vitest 5.1.1) with /api proxy to 127.0.0.1:8000"
  - "Dual-project npm run typecheck (tsconfig.app.json && tsconfig.node.json), Biome v2 CSS tailwindDirectives + organizeImports config, offline-safe pt-BR index.html, and 04-UI-SPEC.md @theme tokens"
  - "Strict TypeScript API contracts in frontend/src/types/api.ts mirroring backend/app/api/schemas.py and models/*.json (LensClass + OpticalLensClass, OofScatterPoint.actual_g, WatershedParamBounds tuples, zero rel_path)"
  - "Unit-tested pure utilities (cn, pt-BR formatters with '—' nullable guard, ddof=1 client flock math, 15 MiB/MIME upload guard, FormData optional-field omission, pt-BR ApiError mapping, AbortError detection) verified by 18 Vitest tests"
  - "Composable React studio hooks (useAbortableRequest, useHealth, usePresets, useMetrics, useFlockBatch) and clinical-light layout primitives (AppShell, StageLightbox, InlineErrorBanner) wired into an initial compiling 4-tab hash-synced App.tsx"
affects: [04-03, 04-04, frontend-studio, frontend-watershed, frontend-metrics]

actuals:
  tokens: 18576
  tasks: 3
  commits: 3

tech-stack:
  added:
    - react@19.3.1
    - react-dom@19.3.1
    - react-is@19.3.1
    - recharts@3.11.1
    - lucide-react@1.79.0
    - clsx@2.1.1
    - tailwind-merge@3.7.1
    - vite@8.3.8
    - "@vitejs/plugin-react@6.3.0"
    - tailwindcss@4.3.5
    - "@tailwindcss/vite@4.3.5"
    - typescript@7.0.2
    - "@biomejs/biome@2.6.8"
    - "@types/react@19.3.0"
    - "@types/react-dom@19.3.0"
    - "@types/node@26.6.6"
    - vitest@5.1.1
  patterns:
    - "Tailwind CSS v4 CSS-first @theme design tokens in frontend/src/index.css with zero tailwind.config.js or PostCSS boilerplate"
    - "Dual-project TypeScript 7 verification: tsc -p tsconfig.app.json --noEmit && tsc -p tsconfig.node.json --noEmit"
    - "AbortController request sequence guard (useAbortableRequest) canceling superseded calls and ignoring AbortError without flashing error state"
    - "Strict client-side 15 MiB + {image/jpeg, image/png, image/webp} upload guard and sanitized pt-BR ApiError status mapping (zero stack traces or paths)"
    - "Hash-whitelisted 4-tab AppShell (#estudio, #watershed, #metricas, #visao-futura) with 44x44 accessible icon controls and focus-restoring StageLightbox"

key-files:
  created:
    - frontend/package.json
    - frontend/package-lock.json
    - frontend/vite.config.ts
    - frontend/tsconfig.json
    - frontend/tsconfig.app.json
    - frontend/tsconfig.node.json
    - frontend/biome.json
    - frontend/index.html
    - frontend/src/index.css
    - frontend/src/types/api.ts
    - frontend/src/lib/cn.ts
    - frontend/src/lib/format.ts
    - frontend/src/lib/flock.ts
    - frontend/src/lib/api.ts
    - frontend/src/lib/studioCore.test.ts
    - frontend/src/hooks/useStudioHooks.ts
    - frontend/src/components/layout/AppShell.tsx
    - frontend/src/components/layout/StageLightbox.tsx
    - frontend/src/components/layout/InlineErrorBanner.tsx
    - frontend/src/main.tsx
    - frontend/src/App.tsx
  modified: []

key-decisions:
  - "Migrated Biome 2.6.8 linter rules config to preset: 'recommended' via biome migrate so biome check runs with zero deprecation warnings alongside css.parser.tailwindDirectives: true and assist.actions.source.organizeImports: 'on'"
  - "Kept Accent #047857 strictly confined to the closed 7-element 04-UI-SPEC.md allowlist (using high-contrast white-on-slate for active StageLightbox step switcher buttons inside the dark #0F172A modal)"
  - "Added full strictly-typed EvaluationReport, DatasetAuditSummary, and PdiCalibrationAudit interfaces to frontend/src/types/api.ts matching published models/*.json verbatim so Wave 4 metrics charts consume typed payloads without any casts"
  - "Formatted formatKg(kg) directly over values already in kilograms (matching PredictResponse.predicted_weight_kg) and pluralizePtBr with pt-BR integer grouping ('1 ave', '3 aves', '1.738 imagens')"

patterns-established:
  - "Closed 4-size (12px/14px/20px/32px) and 2-weight (400/600) clinical-light typography with tabular-nums on every telemetry value"
  - "Zero dangerouslySetInnerHTML and zero rel_path substrings across frontend/src/"
  - "Relative /api/* fetch paths routed through Vite dev proxy (127.0.0.1:8000) and FastAPI single-origin dist mount"

requirements-completed: [WEB-02]

coverage:
  - id: D1
    description: "Pinned Vite 8 / React 19 / TypeScript 7 / Tailwind CSS v4 / Biome v2 / Vitest frontend workspace with dual-project npm run typecheck, /api proxy, offline-safe index.html, and 04-UI-SPEC.md @theme tokens"
    requirement: "WEB-02"
    verification:
      - kind: other
        ref: "cd frontend && npm run typecheck && npm run lint && npm run build && test -f dist/index.html"
        status: pass
    human_judgment: false
  - id: D2
    description: "Strict TypeScript API contract mirror (LensClass + OpticalLensClass, OofScatterPoint.actual_g, WatershedParamBounds tuples, zero rel_path), pt-BR formatters, ddof=1 flock math, 15 MiB/MIME upload guard, FormData optional-field omission, pt-BR ApiError mapping, and AbortError handling"
    requirement: "WEB-02"
    verification:
      - kind: unit
        ref: "frontend/src/lib/studioCore.test.ts (18/18 Vitest tests passed)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Shared React hooks (useAbortableRequest, useHealth, usePresets, useMetrics, useFlockBatch), clinical-light AppShell (64px sticky header, 4 hash-synced role='tablist' tabs in overflow-x-auto, live health pill), keyboard-accessible StageLightbox (Escape focus restore, ArrowLeft/ArrowRight, 44x44 pt-BR aria-label controls), InlineErrorBanner, main.tsx, and compiling 4-tab App.tsx"
    requirement: "WEB-02"
    verification:
      - kind: other
        ref: "cd frontend && npm run typecheck && npm run lint && npm test && npm run build"
        status: pass
    human_judgment: false

duration: 16min
completed: 2026-11-24
status: complete
---

# Phase 4 Plan 02: Frontend Foundation, API Layer & Clinical-Light Shell Summary

**Pinned Vite 8 + React 19 + TypeScript 7 + Tailwind CSS v4 + Biome v2 + Vitest workspace with exact Pydantic v2 API contracts, 18 passing core unit tests, abort-safe studio hooks, and a compiling 4-tab clinical-light AppShell**

## Performance

- **Duration:** 16 min
- **Started:** 2026-11-24T10:00:43Z
- **Completed:** 2026-11-24T10:16:30Z
- **Tasks:** 3
- **Files modified:** 21 created, 0 modified

## Accomplishments

- Scaffolded the greenfield `frontend/` toolchain under Node `v24.21.0` / npm `11.19.0` with exact verified versions for all 17 direct packages (`0` vulnerabilities, zero ESLint/`typescript-eslint`), dual-project `"typecheck": "tsc -p tsconfig.app.json --noEmit && tsc -p tsconfig.node.json --noEmit"`, Vite `/api` proxy to `http://127.0.0.1:8000` (`D-05`), Biome v2 (`tailwindDirectives: true`, `assist.actions.source.organizeImports: "on"`), offline-safe `pt-BR` `index.html`, and Tailwind CSS v4 `@theme` design tokens locked verbatim to `04-UI-SPEC.md` (`D-07`).
- Implemented strict TypeScript API types (`frontend/src/types/api.ts`) matching `backend/app/api/schemas.py` and `models/*.json` verbatim (exporting both `LensClass = "wide" | "ultrawide"` and `OpticalLensClass = LensClass | "unknown"`, `OofScatterPoint.actual_g`, `WatershedParamBounds` `[number, number]` tuples, `ground_truth_scope: "group_sum_only"`, and zero `rel_path` properties), pure utilities (`cn`, `format`, `flock`, `api`), and 18 TDD Vitest unit tests (`studioCore.test.ts`) passing in ~211 ms.
- Built the shared React hook layer (`useAbortableRequest`, `useHealth`, `usePresets`, `useMetrics`, `useFlockBatch`), clinical-light `AppShell` (`64px` sticky header, 4 hash-synced `role="tablist"` tabs inside `overflow-x-auto`, live health + latency pill), focus-restoring keyboard `StageLightbox` (`Escape`, `ArrowLeft`/`ArrowRight`, `44×44` `pt-BR` icon controls), `InlineErrorBanner` (`role="alert"`), `main.tsx`, and an initial self-consistent 4-tab `App.tsx` that builds cleanly into git-ignored `frontend/dist/index.html` (`535 ms`).

## Task Commits

Each task was committed atomically:

1. **Task 1: Scaffold Frontend toolchain (Vite 8 / React 19 / TS 7 / Tailwind v4 / Biome v2 / Vitest), dual-project `typecheck` script, and `04-UI-SPEC.md` CSS tokens** - `505221c` (chore)
2. **Task 2: Implement strict TypeScript API contracts (`LensClass` + `OpticalLensClass`), pure utilities (`cn`, `format`, `flock`, `api`), and Vitest core unit test suite** - `38b225e` (feat)
3. **Task 3: Create shared hooks (`useStudioHooks.ts`), global layout shell (`AppShell`, `StageLightbox`, `InlineErrorBanner`), `main.tsx`, and initial self-consistent 4-tab `App.tsx`** - `b79060f` (feat)

## Files Created/Modified

- `frontend/package.json` - Pinned React 19.3.1 / TS 7.0.2 / Vite 8.3.8 / Tailwind v4.3.5 / Recharts 3.11.1 / Biome 2.6.8 / Vitest 5.1.1 manifest with dual-project `typecheck` script.
- `frontend/package-lock.json` - Deterministic npm lockfile verified free of unexpected Linux lifecycle scripts.
- `frontend/vite.config.ts` - Vite 8 + `@vitejs/plugin-react` + `@tailwindcss/vite` config with `/api` proxy to `http://127.0.0.1:8000`.
- `frontend/tsconfig.json`, `frontend/tsconfig.app.json`, `frontend/tsconfig.node.json` - Strict TypeScript 7 project-reference configuration for `src/` and `vite.config.ts`.
- `frontend/biome.json` - Biome v2.6.8 config with `css.parser.tailwindDirectives: true`, `assist.actions.source.organizeImports: "on"`, and `linter.rules.preset: "recommended"`.
- `frontend/index.html` - Offline-safe `lang="pt-BR"` SPA document with zero external CDN font links.
- `frontend/src/index.css` - Tailwind CSS v4 `@import "tailwindcss";`, `@theme` tokens locked to `04-UI-SPEC.md`, base `14px` body rules, and `:focus-visible` `2px solid #047857` outline.
- `frontend/src/types/api.ts` - Complete strict TypeScript mirrors for all REST responses and audit JSON payloads (`LensClass`, `OpticalLensClass`, `SceneType`, `OofScatterPoint.actual_g`, zero `rel_path`).
- `frontend/src/lib/cn.ts` - `clsx` + `tailwind-merge` class combiner.
- `frontend/src/lib/format.ts` - `pt-BR` numeric, gram, kilogram, percent, nullable `"—"`, and singular/plural formatters.
- `frontend/src/lib/flock.ts` - Pure client-side simulated flock accumulator math (`0`/`1`/`>=2` entries, unbiased `ddof=1` sample SD, `CV%`).
- `frontend/src/lib/api.ts` - Native `fetch` + `AbortSignal` client (`/api/*`), `15 MiB`/MIME upload guard, finite optional-field `FormData` serializer, sanitized `pt-BR` `ApiError` status mapper, and `isAbortError`.
- `frontend/src/lib/studioCore.test.ts` - 18 Vitest unit tests covering `cn`, `format`, `flock`, upload guards, `FormData` omission, `ApiError` mapping, `AbortError` pass-through, and type contracts.
- `frontend/src/hooks/useStudioHooks.ts` - `useAbortableRequest`, `useHealth`, `usePresets`, `useMetrics` (lazy cached), and `useFlockBatch` (`sessionStorage`-backed).
- `frontend/src/components/layout/AppShell.tsx` - Sticky `64px` clinical-light header, 4 hash-synced `role="tablist"` tabs inside `overflow-x-auto`, and live API health + latency badge.
- `frontend/src/components/layout/StageLightbox.tsx` - Dark `#0F172A` fullscreen inspection modal with `Escape` focus restore, `ArrowLeft`/`ArrowRight` navigation, visible `"Fechar Ampliação"`, and `44×44` `pt-BR` icon controls.
- `frontend/src/components/layout/InlineErrorBanner.tsx` - Accessible `#FEF2F2`/`#B91C1C` `role="alert"` recovery banner with optional Verb+Noun CTA and dismiss button.
- `frontend/src/main.tsx` - React 19 `createRoot` mount.
- `frontend/src/App.tsx` - Initial self-consistent 4-tab hash-whitelisted shell wiring `AppShell`, `useHealth`, and `usePresets` ready for Wave 3 and Wave 4 workspaces.

## Decisions Made

- Migrated `frontend/biome.json` linter rules from deprecated `"recommended": true` to `"preset": "recommended"` using `npx biome migrate --write` so `npm run lint` runs cleanly with zero deprecation warnings under Biome `2.6.8`.
- Styled active stage switcher pills inside `StageLightbox` with neutral white-on-slate (`border-white bg-white text-viewport-dark`) rather than `#047857`, keeping Accent `#047857` strictly confined to the closed 7-element `04-UI-SPEC.md` allowlist.
- Fully typed `EvaluationReport`, `DatasetAuditSummary`, and `PdiCalibrationAudit` inside `frontend/src/types/api.ts` directly against published `models/*.json` artifacts so Wave 4 Tab 3 components can consume `MetricsResponse` with full compile-time safety.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Migrated Biome v2.6.8 `linter.rules.recommended` deprecation**
- **Found during:** Task 1 (Biome verification)
- **Issue:** Biome `2.6.8` emitted a deprecation diagnostic on `"linter": { "rules": { "recommended": true } }` advising migration to `"preset": "recommended"`, and normalized CSS hex tokens in `src/index.css` to lowercase.
- **Fix:** Ran `npx biome migrate --write` to update `biome.json` to `"preset": "recommended"` and formatted `src/index.css` cleanly.
- **Files modified:** `frontend/biome.json`, `frontend/src/index.css`
- **Verification:** `npx biome check vite.config.ts src/index.css` exited `0` with zero warnings.
- **Committed in:** `505221c` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking tooling migration)
**Impact on plan:** Eliminated Biome 2.6.8 deprecation warning with zero scope creep.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Wave 3 (`04-03-PLAN.md` — Tab 1 `Estúdio Raio-X` & Tab 4 `Visão Futura (v2)`) and Wave 4 (`04-04-PLAN.md` — Tab 2 `Separação Watershed` & Tab 3 `Auditoria & Métricas`) can now mount directly on top of verified `frontend/src/types/api.ts`, `frontend/src/lib/*`, `frontend/src/hooks/useStudioHooks.ts`, and `frontend/src/components/layout/*`.
- `.gitignore` keeps `frontend/node_modules/` and `frontend/dist/` untracked while `npm run build` emits `frontend/dist/index.html` cleanly in `535 ms`.

## Self-Check: PASSED

- All 21 planned frontend files exist on disk and match task commits `505221c`, `38b225e`, and `b79060f`.
- `npm run typecheck` (`tsc -p tsconfig.app.json --noEmit && tsc -p tsconfig.node.json --noEmit`), `npm run lint` (`biome check src`), `npm test` (`18/18` passed), and `npm run build` (`dist/index.html` emitted) all exited `0`.
- Zero `dangerouslySetInnerHTML` and zero `rel_path` occurrences in `frontend/src/`; `frontend/node_modules/` and `frontend/dist/` verified git-ignored via `git status --ignored=no --porcelain`.

---
*Phase: 04-fastapi-local-backend-interactive-pitch-web-studio*
*Completed: 2026-11-24*
