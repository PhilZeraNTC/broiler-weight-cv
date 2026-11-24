# Phase 4: FastAPI Local Backend & Interactive Pitch Web Studio - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-11-24
**Phase:** 4-fastapi-local-backend-interactive-pitch-web-studio
**Areas discussed:** API Contract & Security Guardrails, Pitch Navigation & Visual Theme, Interactive X-Ray Studio & Optical Override UX, Watershed Controls / Metrics Interactivity / Local Run Mode

---

## API Contract, Preset Routing & Security Guardrails (`WEB-01`)

| Option | Description | Selected |
|--------|-------------|----------|
| Modular `backend/app/api/` + lifespan bundle/manifest resolver + `run_in_threadpool` + cached `/api/presets/{id}/thumbnail` | Clean router/service split; opaque `preset_id -> Path` allowlist validated against `RAW_DATASET_DIR.resolve()`; honest `fold_pipelines[fold]` for single-bird presets vs `production_pipeline` for uploads; 15 MB upload cap; non-blocking threadpool CPU offload | ✓ |
| Monolithic single `main.py` with inline base64 thumbnails in `/api/presets` | Faster initial scaffold but bloats `/api/presets` response and mixes routing, validation, and PDI/ML orchestration | |
| Client-supplied relative file paths for presets | Violates ASVS V4 path traversal guardrails and `PROIBIÇÃO 4` raw filename hygiene | |

**User's choice:** Modular `backend/app/api/` + lifespan bundle/manifest resolver + `run_in_threadpool` + cached `/api/presets/{id}/thumbnail` (auto-selected recommended default via `--auto`)
**Notes:** Directly aligns with `.planning/research/ARCHITECTURE.md` Patterns 4–5 and Phase 3 `models/presets_catalog.json` opaque-ID contract (`03-CONTEXT.md` D-11).

---

## Pitch Studio Navigation, Language & Visual Direction (`WEB-02`–`WEB-05`)

| Option | Description | Selected |
|--------|-------------|----------|
| 4 top-level `pt-BR` tabs (`Estúdio Raio-X`, `Separação Watershed`, `Auditoria & Métricas`, `Visão Futura (v2)`) + Clinical Agritech Light Theme with dark technical image frames | Instant 1-click navigation for live pitch, zero ambiguity separating v2 conceptual scope (`WEB-04`), high projector contrast with dark frames behind X-Ray/Watershed outputs | ✓ |
| 3 tabs with `Visão Futura` embedded at bottom of Dashboard | Buries `WEB-04` below long scroll and blurs validated v1 metrics with unimplemented v2 vision | |
| Full dark-mode dashboard | Lower legibility on bright classroom/pitch projectors for dense scientific tables and Recharts axes | |

**User's choice:** 4 top-level `pt-BR` tabs + Clinical Agritech Light Theme with dark technical image frames (auto-selected recommended default via `--auto`)
**Notes:** Exact visual tokens, spacing, typography, and copywriting contract will be verified by `gsd-ui-researcher` + `gsd-ui-checker` in `04-UI-SPEC.md`.

---

## Interactive X-Ray Studio & Optical Override UX (`WEB-02`)

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-load default curated preset on boot + 6-preset thumbnail gallery with `541__s0` Dual-Lens Pair highlight + 2×2/expanded 4-stage viewer + live manual lens/zoom override | Zero empty screen on pitch open, 2-click optical invariance proof (`Wide` vs `Ultrawide`), explicit amber notice & manual calibration path for EXIF-stripped uploads | ✓ |
| Empty dropzone landing page until user uploads or clicks a preset | Requires extra click before audience sees the X-Ray PDI pipeline during pitch | |
| Read-only EXIF metadata display without manual override inputs | Fails `WEB-02` requirement for manual lens/zoom override on EXIF-stripped uploads | |

**User's choice:** Auto-load default curated preset on boot + 6-preset thumbnail gallery with Dual-Lens Pair highlight + 2×2/expanded 4-stage viewer + live manual lens/zoom override (auto-selected recommended default via `--auto`)
**Notes:** Preserves honest OOF fold-model comparison banner whenever a curated preset is active (`oof_evaluation=true`).

---

## Watershed Tuning, Metrics Dashboard & Local Pitch Run Mode (`WEB-03`, `WEB-05`)

| Option | Description | Selected |
|--------|-------------|----------|
| Bounded sliders with commit-triggered re-run + `AbortController` + filterable Recharts `ScatterChart` (1,738 images / 188 sessions toggle) + Vite `/api` proxy & optional FastAPI `frontend/dist/` single-command static serving | Responsive slider ergonomics without CPU queue pile-up, fast SVG scatter rendering (`isAnimationActive={false}`), seamless dev HMR and 1-command pitch startup | ✓ |
| Fire `/api/watershed-demo` on every slider `onChange` tick without aborting | Queues dozens of CPU requests on local hardware and degrades perceived responsiveness | |
| Separate CORS-enabled servers required on pitch day | Extra failure mode during live presentation compared to single-origin `uvicorn` serving built `frontend/dist/` | |

**User's choice:** Bounded sliders with commit-triggered re-run + `AbortController` + filterable Recharts `ScatterChart` + Vite `/api` proxy & optional FastAPI `frontend/dist/` static serving (auto-selected recommended default via `--auto`)
**Notes:** Satisfies `PRD.md` `< 500 ms` SLA and `STACK.md` single-origin pitch-day ergonomics.

---

## the agent's Discretion
- React component/hook structure under `frontend/src/`.
- Exact Tailwind CSS v4 `@theme` design tokens (locked in `04-UI-SPEC.md`).
- In-memory caching strategy for the 10 preset thumbnails.
- FastAPI error response helpers and Biome/TypeScript 7 configuration.

## Deferred Ideas
- Multi-frame barn camera tracking (`BARN-01`) — conceptual roadmap only in `WEB-04`.
- Spatial flock density heatmaps (`BARN-02`) — conceptual roadmap only in `WEB-04`.
- Stationary-bird sanitary alerts (`BARN-03`) — conceptual roadmap only in `WEB-04`.
- Authentication, persistence database, batch CSV export, or cloud deployment.
