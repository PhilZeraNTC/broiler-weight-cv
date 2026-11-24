---
phase: 04-fastapi-local-backend-interactive-pitch-web-studio
reviewed: 2026-11-24T14:51:01Z
depth: deep
files_reviewed: 64
files_reviewed_list:
  - backend/app/config.py
  - backend/app/api/__init__.py
  - backend/app/api/dependencies.py
  - backend/app/api/main.py
  - backend/app/api/schemas.py
  - backend/app/api/routers/__init__.py
  - backend/app/api/routers/health.py
  - backend/app/api/routers/metrics.py
  - backend/app/api/routers/predict.py
  - backend/app/api/routers/presets.py
  - backend/app/api/routers/watershed.py
  - backend/app/api/services/__init__.py
  - backend/app/api/services/inference_service.py
  - backend/app/api/services/metrics_service.py
  - backend/app/api/services/preset_service.py
  - backend/app/api/services/upload_guard.py
  - backend/app/api/services/watershed_service.py
  - backend/tests/test_api_contract.py
  - backend/tests/test_api_latency_and_parity.py
  - backend/tests/test_api_security.py
  - frontend/biome.json
  - frontend/index.html
  - frontend/package.json
  - frontend/tsconfig.app.json
  - frontend/tsconfig.json
  - frontend/tsconfig.node.json
  - frontend/vite.config.ts
  - frontend/src/App.tsx
  - frontend/src/index.css
  - frontend/src/main.tsx
  - frontend/src/types/api.ts
  - frontend/src/hooks/useStudioHooks.ts
  - frontend/src/lib/api.ts
  - frontend/src/lib/cn.ts
  - frontend/src/lib/flock.ts
  - frontend/src/lib/format.ts
  - frontend/src/lib/scatter.ts
  - frontend/src/lib/scatter.test.ts
  - frontend/src/lib/studioCore.test.ts
  - frontend/src/components/layout/AppShell.tsx
  - frontend/src/components/layout/InlineErrorBanner.tsx
  - frontend/src/components/layout/StageLightbox.tsx
  - frontend/src/components/metrics/ActualVsPredictedScatter.tsx
  - frontend/src/components/metrics/FeatureImportanceChart.tsx
  - frontend/src/components/metrics/ForensicAuditCard.tsx
  - frontend/src/components/metrics/MetricsAuditTab.tsx
  - frontend/src/components/metrics/ModelComparisonTable.tsx
  - frontend/src/components/metrics/OpticalAblationChart.tsx
  - frontend/src/components/metrics/StratifiedErrorsPanel.tsx
  - frontend/src/components/studio/DualLensComparison.tsx
  - frontend/src/components/studio/FlockBatchCard.tsx
  - frontend/src/components/studio/OpticalCalibrationCard.tsx
  - frontend/src/components/studio/PredictionSummaryCard.tsx
  - frontend/src/components/studio/PresetsStrip.tsx
  - frontend/src/components/studio/TelemetryPanel.tsx
  - frontend/src/components/studio/UploadDropzone.tsx
  - frontend/src/components/studio/XRayQuadViewer.tsx
  - frontend/src/components/studio/XRayStudioTab.tsx
  - frontend/src/components/vision/FutureVisionTab.tsx
  - frontend/src/components/watershed/WatershedInstanceTable.tsx
  - frontend/src/components/watershed/WatershedPresetStrip.tsx
  - frontend/src/components/watershed/WatershedSlidersCard.tsx
  - frontend/src/components/watershed/WatershedStageGrid.tsx
  - frontend/src/components/watershed/WatershedStudioTab.tsx
findings:
  critical: 2
  warning: 5
  info: 4
  total: 11
status: issues_found
---

# Phase 04: Code Review Report

**Reviewed:** 2026-11-24T14:51:01Z
**Depth:** deep
**Files Reviewed:** 64
**Status:** issues_found

## Summary

Deep cross-file code and security review of all 64 Phase 4 backend and frontend deliverables (`backend/app/config.py`, `backend/app/api/**`, `backend/tests/test_api_*.py`, and `frontend/**`, excluding `frontend/package-lock.json`).

**What passed verification cleanly:**
- **PROIBIÇÕES 1–5 & ASVS L1 Security:** Zero deep learning or OCR imports; zero `cv2.imread`/`cv2.imdecode` calls (all decoding routed via Pillow EXIF-first `load_working_rgb` / Phase 2 engines); honest dual-bundle OOF preset routing (`fold_pipelines[preset.fold]`) vs. `production_pipeline` for uploads; strict localhost-only CORS (`allow_credentials=False`, no wildcards); ASVS V4 per-segment non-symlink path containment + opaque `preset_id` allowlist; zero `rel_path`, raw dataset filenames, or stack traces leaked in responses/headers; `FutureVisionTab.tsx` is 100% static with zero v2 endpoints (`PROIBIÇÃO 5`); zero tracked ignored/build artifacts (`dist/`, `node_modules/`, `data/`, `models/*.joblib`).
- **Schema Parity, Recharts v3 & `04-UI-SPEC.md` Typography/A11y:** Exact field parity between `backend/app/api/schemas.py`, `models/*.json`, and `frontend/src/types/api.ts` (`actual_g`, `bucket_dorsal`, `WatershedParamBounds`, `band_error_interval`, `oof_audit`, `ground_truth_scope`); Recharts v3 charts use zero `<Cell />`, `isAnimationActive={false}` on all series, explicit container heights + `min-w-0`, numeric `y = x` `<ReferenceLine segment={...} />`, and null-safe custom tooltips; strictly 4 font sizes (`12px`, `14px`, `20px`, `32px`), 2 weights (`font-normal`, `font-semibold`), `tabular-nums`, `44×44` icon hit targets, and verbatim `pt-BR` `aria-label`s and scientific disclosures.
- **Automated Checks:** `pytest backend/tests/test_api_*.py` (`64 passed`), `ruff check` (`All checks passed`), `tsc --noEmit`, `biome check src` (`37 files`, 0 errors), and `vitest run` (`26 passed`) all pass.

**Defects found requiring attention:**
- **2 Critical (`CR-01`, `CR-02`):** Both `useMetrics` and `WatershedStudioTab` latch a `useRef` guard before their initial fetch completes and never reset it when React 19 `<StrictMode>` (`frontend/src/main.tsx:12`) triggers `useAbortableRequest`'s unmount abort cleanup — permanently stranding direct `#metricas` or `#watershed` URL loads/refreshes on blank/empty screens in development.
- **5 Warnings (`WR-01`–`WR-05`):** WAI-ARIA tablist keyboard focus trap in `AppShell.tsx`; stale-prediction / mismatched-source metadata and missing upload recovery action in `XRayStudioTab.tsx`; no-op `"Enviar Foto de Grupo"` error CTA and stale source in `WatershedStudioTab.tsx`; stale render-closure slider commits in `WatershedSlidersCard.tsx`; synchronous CPU `validate_image_bytes` on the asyncio event loop paired with thread-unsafe `warnings.catch_warnings()` in `upload_guard.py`.
- **4 Info (`IN-01`–`IN-04`):** Minor robustness and maintainability cleanups in `DualLensComparison.tsx`, decode-stage `ValueError` handling, SPA `405` on unknown non-GET `/api/*`, and minor hardcoded summary literals in metrics components.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: `useMetrics` ref-latch permanently strands `#metricas` blank under React 19 `StrictMode`

**File:** `frontend/src/hooks/useStudioHooks.ts:182-203` (manifesting in `frontend/src/components/metrics/MetricsAuditTab.tsx:44-46`)
**Issue:** `App.tsx:22-30` reads `window.location.hash` on boot, so refreshing or deep-linking directly to `/#metricas` mounts `<MetricsAuditTab isActive={true} />` immediately under `<StrictMode>` (`frontend/src/main.tsx:12`). On Mount 1, `useMetrics` enters its `useEffect`, sets `fetchedRef.current = true` synchronously inside `load()`, and starts `fetchMetrics(signal)` via `useAbortableRequest`. React 19 `StrictMode` immediately executes the simulated unmount cleanup in `useAbortableRequest` (`useStudioHooks.ts:47-51`), calling `controllerRef.current?.abort()`, which aborts the in-flight request and resets `loading = false` while leaving `data = null` and `error = null`. On the simulated remount, React preserves `fetchedRef.current === true`, so `if (enabled && !fetchedRef.current && data === null)` evaluates to `false` and `load()` is never called again. `MetricsAuditTab.tsx:44-46` then hits `if (data === null) return null;`, rendering a permanently empty page with no spinner, no error banner, and no retry button.
**Fix:** Track only whether a request is currently in flight (resetting in `finally` and in `useEffect` cleanup) instead of latching a permanent `fetchedRef` before data succeeds:
```ts
export function useMetrics(enabled: boolean): UseMetricsResult {
  const { data, loading, error, run } = useAbortableRequest<MetricsResponse>();
  const inFlightRef = useRef<boolean>(false);

  const load = useCallback(() => {
    inFlightRef.current = true;
    void run((signal) => fetchMetrics(signal)).finally(() => {
      inFlightRef.current = false;
    });
  }, [run]);

  useEffect(() => {
    if (enabled && data === null && error === null && !inFlightRef.current) {
      load();
    }
    return () => {
      inFlightRef.current = false;
    };
  }, [enabled, data, error, load]);

  return { data, loading, error, retry: load };
}
```

### CR-02: `WatershedStudioTab` `hasBootedRef` latch permanently strands `#watershed` empty under React 19 `StrictMode`

**File:** `frontend/src/components/watershed/WatershedStudioTab.tsx:74,109-125`
**Issue:** Identical React 19 `<StrictMode>` ref-latch failure when loading or refreshing directly on `/#watershed` (`isActive === true` on initial mount). Mount 1 sets `hasBootedRef.current = true` (line 113) and dispatches `executeWatershed(ws_417)`. StrictMode's immediate simulated unmount invokes `useAbortableRequest`'s cleanup (`useStudioHooks.ts:47-51`), aborting the `ws_417` request and leaving `data = null`, `loading = false`, `error = null`. On StrictMode remount, `hasBootedRef.current` remains `true`, causing line 110 (`if (!isActive || hasBootedRef.current) return;`) to bail out permanently. Tab 2 stays stuck showing `"—"` in all 3 KPI cards and `"Aguardando etapa 1..4"` until the user manually clicks a preset.
**Fix:** Guard auto-boot by actual state (`data === null && error === null`) plus a cleanup-resettable `inFlightRef`, or only set `hasBootedRef.current = true` once `executeWatershed` resolves with a non-null response:
```tsx
const hasBootedRef = useRef<boolean>(false);
const bootingRef = useRef<boolean>(false);

useEffect(() => {
  if (!isActive || hasBootedRef.current || bootingRef.current || data !== null || error !== null) {
    return;
  }
  bootingRef.current = true;
  const bootSource: ActiveWatershedSource = {
    sourceType: "preset",
    presetId: DEFAULT_WATERSHED_PRESET_ID,
    file: null,
    fileName: null,
  };
  const defaultSliders: WatershedSliderValues = { ...WATERSHED_SLIDER_DEFAULTS };
  setSource(bootSource);
  setSliders(defaultSliders);
  setOpticalOverride(null);
  void executeWatershed(bootSource, defaultSliders, null)
    .then((ok) => {
      if (ok) {
        hasBootedRef.current = true;
      }
    })
    .finally(() => {
      bootingRef.current = false;
    });

  return () => {
    bootingRef.current = false;
  };
}, [isActive, data, error, executeWatershed]);
```
(Have `executeWatershed` return `Promise<boolean>` indicating `response !== null`.)

## Warnings

### WR-01: WAI-ARIA `role="tablist"` keyboard navigation traps focus and blocks reaching Tabs 3 and 4

**File:** `frontend/src/components/layout/AppShell.tsx:48-64,91-101`
**Issue:** `handleTabKeyDown` calls `onSelectTab(...)` on `ArrowRight`, `ArrowLeft`, `Home`, and `End` without moving DOM focus to the newly activated tab button. Because each tab uses roving `tabIndex={isActive ? 0 : -1}` and captures its own `idx` in `onKeyDown={(event) => handleTabKeyDown(event, idx)}`, pressing `ArrowRight` from Tab 0 (`idx = 0`) activates Tab 1 while leaving DOM focus on Tab 0. Pressing `ArrowRight` again re-fires `handleTabKeyDown(event, 0)` on Tab 0, re-selecting Tab 1 indefinitely and preventing keyboard users from navigating across all 4 tabs.
**Fix:** Maintain a ref map of tab buttons (or focus by deterministic `id`) whenever selecting via keyboard:
```tsx
const focusAndSelectTab = (nextId: TabId) => {
  onSelectTab(nextId);
  document.getElementById(`tab-${nextId}`)?.focus();
};

const handleTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
  if (event.key === "ArrowRight") {
    event.preventDefault();
    focusAndSelectTab(TAB_ITEMS[(index + 1) % TAB_ITEMS.length].id);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    focusAndSelectTab(TAB_ITEMS[(index - 1 + TAB_ITEMS.length) % TAB_ITEMS.length].id);
  } else if (event.key === "Home") {
    event.preventDefault();
    focusAndSelectTab(TAB_ITEMS[0].id);
  } else if (event.key === "End") {
    event.preventDefault();
    focusAndSelectTab(TAB_ITEMS[TAB_ITEMS.length - 1].id);
  }
};
```

### WR-02: Failed upload in `XRayStudioTab` leaves mismatched `activeSource` vs. stale `prediction`, corrupting Flock Batch labels and omitting `400` recovery CTA

**File:** `frontend/src/components/studio/XRayStudioTab.tsx:88-99,127-145,152-160`
**Issue:** `handleSelectFile` (lines 88–99) eagerly sets `activeSource` to the newly chosen upload file before `/api/predict` succeeds, while `useAbortableRequest` retains the previous `prediction` when the server returns HTTP `400` (e.g., corrupt image passing client MIME/size checks). This produces three visible defects:
1. `UploadDropzone` displays the failed file's name as the active `"Upload Externo"` while the X-Ray 4-stage grid and `PredictionSummaryCard` continue showing the previous image's result.
2. Clicking `"Adicionar Ave ao Lote"` (`lines 127-145`, still enabled because `prediction !== null && !loading`) reads `activeSource.fileName` (the failed upload) and pairs it with the previous upload's `prediction.predicted_weight_g` (or falls back to raw `prediction.preset_id` instead of the preset's human label when the previous result was a preset).
3. For `400`/`413`/`415` errors, `errorActionLabel` (`lines 153-160`) is `undefined`, leaving the `InlineErrorBanner` without the `"Carregar Nova Foto"` button explicitly referenced in the `400` copy (`04-UI-SPEC.md:237,270`).
**Fix:** Track the source that actually produced `prediction` (updating `committedSource` inside `.then((result) => { if (result) setCommittedSource(source); })`) for `handleAddCurrentToFlock` and display status, and pass a file-picker trigger callback from `UploadDropzone` so `InlineErrorBanner` wires `"Carregar Nova Foto"` directly to the file input on `400`/`413`/`415` errors:
```tsx
const [committedSource, setCommittedSource] = useState<ActiveInferenceSource>({
  type: "preset",
  presetId: DEFAULT_BOOT_PRESET_ID,
});
// Inside executePredict .then(result):
if (result) {
  setCommittedSource(source);
  onLatencyRef.current(result.timing.total_ms);
}
// Use committedSource (not pending activeSource) inside handleAddCurrentToFlock
// and wire errorActionLabel === "Carregar Nova Foto" -> fileInputRef.current?.click() on 400/413/415.
```

### WR-03: No-op `"Enviar Foto de Grupo"` error CTA and eager source mutation in `WatershedStudioTab`

**File:** `frontend/src/components/watershed/WatershedStudioTab.tsx:144-159,192-210`
**Issue:**
1. When an upload fails with HTTP `400`, `413`, or `415`, `WatershedStudioTab` renders `InlineErrorBanner` with `actionLabel="Enviar Foto de Grupo"` (line 196), but `handleErrorAction` (lines 204–208) only calls `setClientError(null); clearError();` — it dismisses the alert without opening the file picker, duplicating the adjacent `X` dismiss button and violating the CTA's promised action.
2. `handleSelectFile` (lines 144–159) eagerly overwrites `source` with the failed upload while `data` continues displaying the prior scene (`ws_417`), so subsequently nudging a Watershed slider or clicking `"Reprocessar Watershed"` re-submits the broken upload against visuals that belong to `ws_417`.
**Fix:** Expose a file-picker trigger ref/callback from `WatershedPresetStrip` so clicking `"Enviar Foto de Grupo"` in the error banner opens the file dialog, and only commit `source` (or roll back to the last committed source) when `executeWatershed` succeeds:
```tsx
// In WatershedStudioTab:
const uploadInputRef = useRef<HTMLInputElement | null>(null);
// In handleErrorAction for 400/413/415:
if (activeError?.status === 400 || activeError?.status === 413 || activeError?.status === 415) {
  setClientError(null);
  clearError();
  uploadInputRef.current?.click();
  return;
}
```

### WR-04: Stale render-closure slider commits on `onPointerUp` / `onKeyUp` in `WatershedSlidersCard`

**File:** `frontend/src/components/watershed/WatershedSlidersCard.tsx:202-206,255-260,287-292,319-324`
**Issue:** All three `<input type="range">` controls invoke `onPointerUp={() => commitSliders(values)}` and `onKeyUp={handleRangeKeyUp}` reading `values` from the React render closure rather than `Number(event.currentTarget.value)` from the DOM event. On single track-clicks or rapid keyboard steps (`Home`, `End`, `PageUp`, `Arrow*`), if `pointerup`/`keyup` fires before React flushes the state update queued by `onChangeDraft`, `commitSliders` submits the previous slider value to `/api/watershed-demo`. In addition, clicking a slider thumb without moving it triggers a redundant server request.
**Fix:** Read the exact DOM value on pointer/key commit and skip redundant calls when unchanged from the last committed values:
```tsx
const commitFieldFromEvent = (
  field: keyof WatershedSliderValues,
  rawValue: string,
) => {
  const numeric = Number(rawValue);
  const nextValues: WatershedSliderValues = {
    ...values,
    [field]: Number.isFinite(numeric) ? numeric : values[field],
  };
  onChangeDraft(nextValues);
  commitSliders(nextValues);
};

// On each range input:
onPointerUp={(event) => commitFieldFromEvent("h_rel", event.currentTarget.value)}
onKeyUp={(event) => {
  if (SLIDER_COMMIT_KEYS.has(event.key)) {
    commitFieldFromEvent("h_rel", event.currentTarget.value);
  }
}}
```

### WR-05: `validate_image_bytes` blocks the asyncio event loop and uses thread-unsafe `warnings.catch_warnings()`

**File:** `backend/app/api/services/upload_guard.py:54,64-65`
**Issue:**
1. `async def read_and_validate_upload` calls `validate_image_bytes(raw)` synchronously on Uvicorn's asyncio event loop (line 54). For large PNG uploads up to `15 MiB`, Pillow's `im.verify()` decompresses and CRC-checks all chunks synchronously in CPU on the event loop, blocking concurrent `/api/health`, `/api/metrics`, and thumbnail requests — contradicting `04-CONTEXT.md` `D-03`.
2. `validate_image_bytes` wraps `Image.open()` in `with warnings.catch_warnings(): warnings.simplefilter("error", Image.DecompressionBombWarning)` (lines 64–65). Python's `warnings.catch_warnings()` mutates the process-global `warnings.filters` list without locking and is explicitly documented as non-thread-safe once offloaded to `run_in_threadpool`. Because line 71 already enforces `(w * h) > MAX_IMAGE_PIXELS_API` (`40_000_000`) immediately after header-only `Image.open()` (prior to `im.verify()` and pixel decoding) and line 77 already catches `Image.DecompressionBombError`, per-request mutation of `warnings.filters` is unnecessary.
**Fix:** Remove the per-request `warnings.catch_warnings()` wrapper (optionally registering `warnings.filterwarnings("error", category=Image.DecompressionBombWarning)` once at startup in `lifespan`) and offload `validate_image_bytes` via `run_in_threadpool`:
```python
from starlette.concurrency import run_in_threadpool

async def read_and_validate_upload(upload: UploadFile) -> bytes:
    try:
        raw = await upload.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await upload.close()

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail=_ERR_EMPTY_UPLOAD)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=_ERR_OVERSIZE_UPLOAD)

    await run_in_threadpool(validate_image_bytes, raw)
    return raw


def validate_image_bytes(raw: bytes) -> str:
    if not raw:
        raise HTTPException(status_code=400, detail=_ERR_EMPTY_UPLOAD)
    try:
        with Image.open(io.BytesIO(raw)) as im:
            fmt = (im.format or "").upper()
            if fmt not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(status_code=415, detail=_ERR_UNSUPPORTED_FORMAT)
            w, h = im.size
            if w <= 0 or h <= 0 or (w * h) > MAX_IMAGE_PIXELS_API:
                raise HTTPException(status_code=400, detail=_ERR_DECOMPRESSION_BOMB)
            im.verify()
            return fmt
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=400, detail=_ERR_DECOMPRESSION_BOMB) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=_ERR_CORRUPT_IMAGE) from exc
```

## Info

### IN-01: `DualLensComparison` duplicates abort logic and leaves `loading` stuck `true` if aborted externally

**File:** `frontend/src/components/studio/DualLensComparison.tsx:36-74`
**Issue:** Rather than reusing `useAbortableRequest<DualLensPairResult>()` from `frontend/src/hooks/useStudioHooks.ts:40`, `DualLensComparison` manages its own `controllerRef` and exits the `.catch` handler on abort (`lines 56-58`) without resetting `setLoading(false)`. While `isOpen` currently defaults to `false` in `XRayStudioTab.tsx:43`, mounting this component with `isOpen={true}` under `StrictMode` would wedge `loading === true` permanently.
**Fix:** Replace the manual `controllerRef`/`loading`/`error` state with `useAbortableRequest<DualLensPairResult>()` and the same `inFlightRef` cleanup pattern used in `CR-01`.

### IN-02: JPEG/WEBP pixel-decode `ValueError` in upload services bypasses HTTP `400` mapping

**File:** `backend/app/api/services/inference_service.py:152-161` and `backend/app/api/services/watershed_service.py:142-151`
**Issue:** Pillow's `Image.verify()` is a no-op for JPEG and WEBP scan payloads, so malformed pixel streams are decoded later inside `load_working_rgb`. While `upload_guard.py:79` catches `(UnidentifiedImageError, OSError, SyntaxError, ValueError)`, `inference_service.py:152-158` and `watershed_service.py:142-148` omit `ValueError` (e.g., Pillow `"tile cannot extend outside image"` or malformed ICC/codec `ValueError`), which causes such corrupt uploads to fall through to HTTP `500` instead of HTTP `400`.
**Fix:** Narrowly catch decode-stage `ValueError` (or include `ValueError` when `source_type == "upload"` after optical/Watershed parameter validation has already passed) and map it to `HTTPException(status_code=400, detail=_ERR_DECODE_UPLOAD)`.

### IN-03: Active SPA catch-all returns HTTP `405` instead of `404` on unknown non-GET `/api/*` requests

**File:** `backend/app/api/main.py:123-131`
**Issue:** When `frontend/dist/index.html` exists, `@application.get("/{full_path:path}")` matches path prefixes for non-GET methods as well, so requests like `POST /api/nonexistent` return Starlette's HTTP `405 Method Not Allowed` JSON rather than HTTP `404 Not Found` JSON (which is returned when `frontend/dist` is absent). Sanitized JSON is still returned and no HTML/path leak occurs.
**Fix:** Acceptable for local pitch serving; if strict `404` parity across HTTP verbs is desired, add a catch-all `/api/{unknown_path:path}` route for `["GET", "POST", "PUT", "PATCH", "DELETE"]` registered after the real API routers.

### IN-04: Minor hardcoded audit figures alongside dynamic JSON-driven metrics

**File:** `frontend/src/components/metrics/StratifiedErrorsPanel.tsx:74-77,104-107,154` and `frontend/src/components/metrics/ForensicAuditCard.tsx:40`
**Issue:** `StratifiedErrorsPanel.tsx:154` hardcodes `(188 sessões)` in prose instead of `formatNumberPtBr(ds.n_sessions, 0)`, and lines 76 & 106 render static `"0 Sobreposição Entre Folds"` / `"Vencedor C2 · 0,0% Shift"` badges unconditionally without checking `ds.cross_fold_session_overlap === 0` / `maxShift === 0`. Similarly, `ForensicAuditCard.tsx:40` hardcodes `(1.753 → 1.738)` in the section heading instead of interpolating `funnel.raw_images` and `funnel.clean_images`.
**Fix:** Interpolate `ds.n_sessions`, `funnel.raw_images`, and `funnel.clean_images` dynamically and gate the verified badges on the audited JSON values so copy can never drift if artifacts are regenerated.

---

_Reviewed: 2026-11-24T14:51:01Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
