# Phase 4: FastAPI Local Backend & Interactive Pitch Web Studio - Pattern Map

**Mapped:** 2026-11-24
**Files analyzed:** 46 (20 backend source/test files + 26 greenfield frontend files/component groups)
**Analogs found:** 15 / 46 (15 / 20 backend files matched to existing tracked analogs; 5 first-in-repo FastAPI routers + 26 greenfield `frontend/` entries mapped to `04-RESEARCH.md` & `04-UI-SPEC.md`)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `backend/app/config.py` *(modified)* | config | file-I/O | `backend/app/config.py` | exact |
| `backend/app/api/__init__.py` | config | request-response | `backend/app/modeling/__init__.py` | exact |
| `backend/app/api/main.py` | middleware / route | request-response | `backend/app/modeling/artifacts.py` + `backend/app/config.py` | partial |
| `backend/app/api/dependencies.py` | provider | request-response | `backend/app/pdi/pipeline.py` | role-match |
| `backend/app/api/schemas.py` | model | transform | `backend/app/pdi/config.py` + `backend/app/modeling/age_stage.py` | role-match |
| `backend/app/api/services/__init__.py` | config | transform | `backend/app/modeling/__init__.py` | exact |
| `backend/app/api/services/upload_guard.py` | service | file-I/O | `backend/app/pdi/io_exif.py` | role-match |
| `backend/app/api/services/preset_service.py` | service | file-I/O | `backend/app/modeling/artifacts.py` + `backend/app/pdi/io_exif.py` | exact |
| `backend/app/api/services/inference_service.py` | service | transform | `backend/app/pdi/pipeline.py` + `backend/app/modeling/models.py` | exact |
| `backend/app/api/services/watershed_service.py` | service | transform | `backend/app/pdi/watershed.py` + `backend/app/modeling/models.py` | exact |
| `backend/app/api/services/metrics_service.py` | service | file-I/O | `backend/app/modeling/artifacts.py` | role-match |
| `backend/app/api/routers/__init__.py` | config | request-response | `backend/app/modeling/__init__.py` | exact |
| `backend/app/api/routers/health.py` | route | request-response | none *(first FastAPI router)* | none |
| `backend/app/api/routers/presets.py` | route | request-response | none *(first FastAPI router)* | none |
| `backend/app/api/routers/predict.py` | route | request-response | none *(first FastAPI router)* | none |
| `backend/app/api/routers/watershed.py` | route | request-response | none *(first FastAPI router)* | none |
| `backend/app/api/routers/metrics.py` | route | request-response | none *(first FastAPI router)* | none |
| `backend/tests/test_api_health_presets_metrics.py` | test | request-response | `backend/tests/test_train_and_evaluate_integration.py` | exact |
| `backend/tests/test_api_predict.py` | test | request-response | `backend/tests/test_pdi_pipeline_and_xray.py` | exact |
| `backend/tests/test_api_watershed_and_spa.py` | test | request-response | `backend/tests/test_pdi_watershed.py` | exact |
| `frontend/package.json` | config | file-I/O | none *(greenfield)* | none |
| `frontend/vite.config.ts` | config | request-response | none *(greenfield)* | none |
| `frontend/tsconfig.json` | config | file-I/O | none *(greenfield)* | none |
| `frontend/tsconfig.app.json` | config | file-I/O | none *(greenfield)* | none |
| `frontend/tsconfig.node.json` | config | file-I/O | none *(greenfield)* | none |
| `frontend/biome.json` | config | file-I/O | none *(greenfield)* | none |
| `frontend/index.html` | config | file-I/O | none *(greenfield)* | none |
| `frontend/src/main.tsx` | component | event-driven | none *(greenfield)* | none |
| `frontend/src/index.css` | config | transform | none *(greenfield)* | none |
| `frontend/src/App.tsx` | component | event-driven | none *(greenfield)* | none |
| `frontend/src/api/types.ts` | model | transform | none *(greenfield; mirrors Pydantic & `models/*.json`)* | none |
| `frontend/src/api/client.ts` | service | request-response | none *(greenfield)* | none |
| `frontend/src/utils/cn.ts` | utility | transform | none *(greenfield)* | none |
| `frontend/src/utils/format.ts` | utility | transform | none *(greenfield)* | none |
| `frontend/src/utils/flock.ts` | utility | transform | none *(greenfield)* | none |
| `frontend/src/utils/scatter.ts` | utility | transform | none *(greenfield)* | none |
| `frontend/src/hooks/useHealth.ts` | hook | request-response | none *(greenfield)* | none |
| `frontend/src/hooks/usePresets.ts` | hook | request-response | none *(greenfield)* | none |
| `frontend/src/hooks/useMetrics.ts` | hook | request-response | none *(greenfield)* | none |
| `frontend/src/hooks/useFlockBatch.ts` | hook | event-driven | none *(greenfield)* | none |
| `frontend/src/utils/__tests__/*.test.ts` | test | transform | none *(greenfield Vitest)* | none |
| `frontend/src/components/layout/*.tsx` | component | event-driven | none *(greenfield)* | none |
| `frontend/src/components/studio/*.tsx` | component | request-response | none *(greenfield)* | none |
| `frontend/src/components/watershed/*.tsx` | component | request-response | none *(greenfield)* | none |
| `frontend/src/components/metrics/*.tsx` | component | transform | none *(greenfield)* | none |
| `frontend/src/components/vision/FutureVisionTab.tsx` | component | transform | none *(greenfield)* | none |

---

## Pattern Assignments

### `backend/app/config.py` *(modified — config, file-I/O)*

**Analog:** `backend/app/config.py` (extend in place with `FRONTEND_DIST_DIR`)

**Canonical anchored-path and explicit `__all__` pattern** (`backend/app/config.py`, lines 8-40):
```python
from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATASET_DIR: Path = DATA_DIR / "raw" / "Broiler weights of Cobb Avian"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = REPO_ROOT / "models"
CLEAN_MANIFEST_PATH: Path = PROCESSED_DIR / "manifest_clean.parquet"
QUARANTINE_MANIFEST_PATH: Path = PROCESSED_DIR / "manifest_quarantine.parquet"
AUDIT_SUMMARY_PATH: Path = MODELS_DIR / "dataset_audit_summary.json"
FEATURES_CLEAN_PATH: Path = PROCESSED_DIR / "features_clean.parquet"
PDI_CALIBRATION_AUDIT_PATH: Path = MODELS_DIR / "pdi_calibration_audit.json"
MODEL_BUNDLE_PATH: Path = MODELS_DIR / "weight_regressor.joblib"
EVALUATION_REPORT_PATH: Path = MODELS_DIR / "evaluation_report.json"
PRESETS_CATALOG_PATH: Path = MODELS_DIR / "presets_catalog.json"
# Phase 4 addition:
# FRONTEND_DIST_DIR: Path = REPO_ROOT / "frontend" / "dist"
```
Keep `FRONTEND_DIST_DIR` anchored to `REPO_ROOT` and appended to `__all__` so single-origin SPA mounting works identically regardless of Uvicorn's working directory.

---

### `backend/app/api/__init__.py`, `backend/app/api/services/__init__.py`, `backend/app/api/routers/__init__.py` *(config, package barrels)*

**Analog:** `backend/app/modeling/__init__.py` (and `backend/app/pdi/__init__.py`)

**Barrel re-export & sorted `__all__` pattern** (`backend/app/modeling/__init__.py`, lines 1-50):
```python
"""Public package barrel for AvianScale CV classical tabular modeling and bundling."""

from __future__ import annotations

from backend.app.modeling.age_stage import AgeStageResult, map_weight_to_age_stage
from backend.app.modeling.artifacts import (
    EVALUATION_REPORT_SCHEMA_VERSION,
    MODEL_BUNDLE_SCHEMA_VERSION,
    PRESETS_CATALOG_SCHEMA_VERSION,
    load_model_bundle,
)
...
__all__ = [
    "ABLATION_CONDITION_ORDER",
    "AgeStageResult",
    ...
]
```
Every package under `backend/app/api/` MUST start with `from __future__ import annotations`, use absolute `backend.app.*` imports (matching `known-first-party = ["backend"]` in `pyproject.toml`), and declare a clean, sorted `__all__` list (verified by unit tests like `test_pdi_package_barrel_exports_complete_sorted_public_api` in `backend/tests/test_pdi_watershed.py`, lines 481-545).

---

### `backend/app/api/main.py` *(middleware / route — ASGI factory, lifespan, CORS, global exception handler, optional SPA mount)*

**Analog:** `backend/app/modeling/artifacts.py` (fail-fast bundle validation) + `backend/app/config.py` (canonical paths) + `04-RESEARCH.md` Pattern 3 (non-clobbering SPA fallback)

**Startup model-bundle lock pattern** (`backend/app/modeling/artifacts.py`, lines 509-520):
```python
def load_model_bundle(
    path: Path,
    *,
    expected_pdi_config_hash: str | None = None,
) -> dict[str, Any]:
    """Load and strictly validate a serialized D-07 dual-bundle joblib artifact."""
    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
    bundle = joblib.load(bundle_path)
    validate_model_bundle(bundle, expected_pdi_config_hash=expected_pdi_config_hash)
    return bundle
```
In `create_app()` + `@asynccontextmanager async def lifespan(app: FastAPI)`:
1. Call `cv2.setNumThreads(2)` immediately at startup to protect the 7.1 GiB host (`D-01`).
2. Load `models/weight_regressor.joblib` via `load_model_bundle(MODEL_BUNDLE_PATH, expected_pdi_config_hash=PDIConfig().config_hash())` and reconstruct `bundle_pdi_config = PDIConfig(**bundle["pdi_config"])`.
3. Load the 4 JSON audit files (`PRESETS_CATALOG_PATH`, `EVALUATION_REPORT_PATH`, `AUDIT_SUMMARY_PATH`, `PDI_CALIBRATION_AUDIT_PATH`) and warm the 10 opaque preset paths + JPEG thumbnails in `app.state`.
4. Register strict localhost-only `CORSMiddleware` (`http://127.0.0.1:5173`, `http://localhost:5173`, `http://127.0.0.1:8000`, `http://localhost:8000` — never `["*"]`).
5. Register a global `Exception` handler returning sanitized pt-BR JSON `500` with zero stack traces or filesystem paths (`PROIBIÇÃO 4`).
6. Mount all `/api/*` routers **before** optional `frontend/dist/assets` + SPA catch-all, guarding `if full_path == "api" or full_path.startswith("api/"): raise HTTPException(status_code=404, ...)` (`04-RESEARCH.md` lines 666-680).

---

### `backend/app/api/dependencies.py` *(provider, request-response)*

**Analog:** `backend/app/pdi/pipeline.py` (frozen slotted state dataclass)

**Immutable state container pattern** (`backend/app/pdi/pipeline.py`, lines 28-37):
```python
@dataclass(frozen=True, slots=True)
class PDIResult:
    """Complete single-bird classical PDI output contract."""

    features: dict[str, float]
    optical: OpticalContext
    segmentation: SingleBirdSegmentation
    pdi_config_hash: str
    latency_ms: float
    xray: XRayArtifacts | None = None
```
Apply the exact `@dataclass(frozen=True, slots=True)` convention to `AppState` (holding `bundle`, `pdi_config`, `pdi_config_hash`, `preset_catalog`, `preset_paths`, `thumbnail_cache`, `metrics_payload`) attached to `app.state.context`, exposed to routers via typed `def get_app_state(request: Request) -> AppState:` dependency providers.

---

### `backend/app/api/schemas.py` *(model, transform)*

**Analog:** `backend/app/pdi/config.py` (ASVS V5 bounds & finite float checks) + `backend/app/modeling/age_stage.py` (`AgeStageResult`) + `backend/app/pdi/segmentation.py` (`SingleBirdSegmentation`) + `models/presets_catalog.json`

**Authoritative Watershed & Optical bounds imports** (`backend/app/pdi/config.py`, lines 18-23; `backend/app/pdi/io_exif.py`, lines 29-35):
```python
from backend.app.pdi.config import (
    WATERSHED_MAX_AREA_RATIO,
    WATERSHED_MAX_DISTANCE_REL,
    WATERSHED_MAX_H_REL,
    WATERSHED_MIN_AREA_RATIO,
    WATERSHED_MIN_DISTANCE_REL,
    WATERSHED_MIN_H_REL,
)
from backend.app.pdi.io_exif import (
    MAX_VALID_FOCAL_LENGTH_MM,
    MAX_VALID_ZOOM_MULTIPLE,
    MIN_VALID_FOCAL_LENGTH_MM,
    MIN_VALID_ZOOM_MULTIPLE,
)
```
**Domain contract fields to mirror directly** (`backend/app/modeling/age_stage.py`, lines 27-37 & `backend/app/pdi/segmentation.py`, lines 37-50):
```python
# AgeStageResult (backend/app/modeling/age_stage.py:27-37)
class AgeStageResult:
    weight_g: float
    stage_code: str
    stage_label: str
    estimated_age_days_range: str
    out_of_training_range: bool
    valid_training_range_g: tuple[int, int]
    valid_age_days_range: str

# SingleBirdSegmentation audit subset (backend/app/pdi/segmentation.py:44-50)
# ellipse_from_min_area_rect, scene_type, scene_is_bucket,
# lcd_excluded_px, lcd_core_count, segmentation_fallback, mask_area_frac
```
**Validation & hygiene rules for Pydantic v2 `BaseModel` schemas:**
- Set `model_config = ConfigDict(extra="forbid")` on strict response/input models (use controlled `dict[str, Any]` only for nested read-only audit report JSON blocks in `MetricsResponse`).
- Use `Field(..., allow_inf_nan=False, ge=..., le=...)` for all optical and Watershed float fields so `NaN`/`Inf` are rejected at HTTP `422`.
- Mirror `models/presets_catalog.json` keys verbatim (`preset_id`, `label`, `category`, `pair_group_id`, `image_id`, `session_id`, `fold`, `ground_truth_weight_g`, `weight_band`, `lens_class`, `scene_type`, `oof_predicted_g`, `oof_abs_err_g`, `oof_pct_err` + added `thumbnail_url`; and Watershed preset fields including `ground_truth_scope`, `disclaimer`, `expected_bird_count`, `group_sum_weight_g`, `mean_bird_weight_reference_g`, `detected_bird_count_default` + `thumbnail_url`). Never include `rel_path` in any Pydantic schema (`PROIBIÇÃO 4`).

---

### `backend/app/api/services/upload_guard.py` *(service, file-I/O)*

**Analog:** `backend/app/pdi/io_exif.py` (Pillow-only decoding, finite validation, zero OpenCV image decoders)

**Pillow-only image opening & error-safe parsing pattern** (`backend/app/pdi/io_exif.py`, lines 194-197, 217-221):
```python
def _open_image_source(source: Path | bytes) -> Image.Image:
    """Open a Path or raw image bytes using Pillow without altering MAX_IMAGE_PIXELS."""
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    return Image.open(source)
```
Combine this Pillow-only invariant with `04-RESEARCH.md` Pattern 2 (lines 605-659):
- `MAX_UPLOAD_BYTES = 15 * 1024 * 1024` (bounded async read `await upload.read(MAX_UPLOAD_BYTES + 1)` → HTTP `413` on oversize, HTTP `400` on empty).
- `ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})` → HTTP `415` on disallowed format.
- Escalate `Image.DecompressionBombWarning` to error inside `warnings.catch_warnings()` AND enforce `w * h <= 40_000_000` (`MAX_IMAGE_PIXELS_API`) before pixel decode → HTTP `400` with sanitized pt-BR `detail` strings matching `04-UI-SPEC.md` Error States.

---

### `backend/app/api/services/preset_service.py` *(service, file-I/O)*

**Analog:** `backend/app/modeling/artifacts.py` (`_is_contained_non_symlink`, `_resolve_safe_image_path`) + `backend/app/pdi/io_exif.py` (`load_working_rgb`) + `backend/app/pdi/xray.py` (`_encode_jpeg_b64`)

**ASVS V4 non-symlink path containment pattern — reuse directly** (`backend/app/modeling/artifacts.py`, lines 253-295):
```python
def _is_contained_non_symlink(path: Path, canonical_root: Path) -> bool:
    """Return True iff ``path`` is not a symlink and resolves strictly inside ``canonical_root``."""
    if path.is_symlink():
        return False
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved.is_relative_to(canonical_root)


def _resolve_safe_image_path(rel_path: Any, canonical_root: Path) -> Path:
    """Resolve a manifest ``rel_path`` under ``canonical_root`` enforcing ASVS V4 containment."""
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError("ASVS V4 path containment violation: empty or non-string rel_path.")
    if "\\" in rel_path or "\x00" in rel_path:
        raise ValueError("ASVS V4 path containment violation: illegal path characters.")

    rel = Path(rel_path)
    if rel.is_absolute():
        raise ValueError("ASVS V4 path containment violation: absolute path rejected.")
    if len(rel.parts) == 0 or ".." in rel.parts:
        raise ValueError("ASVS V4 path containment violation: parent traversal segment rejected.")

    cur = canonical_root
    for part in rel.parts:
        if part in ("", ".", ".."):
            raise ValueError("ASVS V4 path containment violation: invalid path segment.")
        cur = cur / part
        if not _is_contained_non_symlink(cur, canonical_root):
            raise ValueError(
                "ASVS V4 path containment violation: symlink or out-of-root target rejected."
            )

    resolved = cur.resolve()
    if not resolved.is_relative_to(canonical_root):
        raise ValueError("ASVS V4 path containment violation: resolved path escapes root.")
    if not resolved.is_file():
        raise FileNotFoundError(
            "Manifest references an image file that does not exist inside dataset_root."
        )
    return resolved
```
**EXIF-transposed thumbnail generation pattern** (`backend/app/pdi/io_exif.py`, lines 317-376):
Call `load_working_rgb(image_path, working_long_edge=320)` (runs Pillow EXIF-first `draft()` + `ImageOps.exif_transpose()` + `INTER_AREA` downscale), convert RGB→BGR, and encode JPEG (`quality=78`) into a pre-warmed `dict[str, bytes]` keyed strictly by opaque `preset_id` (`10` presets, `~103 KiB` RAM total). Join `single_bird_presets` by `image_id` against `CLEAN_MANIFEST_PATH` and `watershed_presets` by `image_id` against `QUARANTINE_MANIFEST_PATH` strictly at startup, discarding `rel_path` immediately after building the server-side `dict[str, Path]` allowlist.

---

### `backend/app/api/services/inference_service.py` *(service, transform)*

**Analog:** `backend/app/pdi/pipeline.py` + `backend/app/modeling/models.py` + `backend/app/modeling/age_stage.py` + `backend/app/ingestion/decoding.py`

**Single-source PDI orchestration & timing pattern** (`backend/app/pdi/pipeline.py`, lines 40-86):
```python
def run_single_bird_pdi(
    source: Path | bytes,
    *,
    config: PDIConfig | None = None,
    optical_override: OpticalOverride | None = None,
    return_xray: bool = False,
) -> PDIResult:
    """Run the unified single-bird classical PDI pipeline identically on ``Path`` or ``bytes``."""
    t0 = time.perf_counter()
    cfg = config if config is not None else PDIConfig()
    ...
    latency_ms = (time.perf_counter() - t0) * 1000.0
```
**Zero-warning single-row scoring pattern** (`backend/app/modeling/models.py`, lines 201-238):
```python
def predict_from_feature_dict(
    pipeline: Pipeline,
    feature_names: Sequence[str],
    features: Mapping[str, float],
) -> float:
    """Predict a single bird weight (g) enforcing exact ``feature_names`` column order."""
    ...
    row_df = pd.DataFrame(ordered_values, columns=ordered_names)
    pred = np.asarray(pipeline.predict(row_df), dtype=np.float64)
    ...
    return pred_g
```
**Weight-band & Cobb Avian 48 age-stage classification** (`backend/app/ingestion/decoding.py`, lines 28-29, 111-117; `backend/app/modeling/age_stage.py`, lines 40-77):
```python
BAND_LIGHT_MAX_EXCLUSIVE_G: int = 500
BAND_MEDIUM_MAX_INCLUSIVE_G: int = 1000

def assign_weight_band(weight_g: int) -> str:
    if weight_g < BAND_LIGHT_MAX_EXCLUSIVE_G:
        return "<500g"
    if weight_g <= BAND_MEDIUM_MAX_INCLUSIVE_G:
        return "500-1000g"
    return ">1000g"
```
*(Note: use float comparison `< float(BAND_LIGHT_MAX_EXCLUSIVE_G)` / `<= float(BAND_MEDIUM_MAX_INCLUSIVE_G)` on float `predicted_weight_g` so `499.9` maps to `"<500g"` and `1000.0` maps to `"500-1000g"` consistently with `bundle["band_error_intervals"]`.)*

**Dual-bundle routing & MIME assembly contract:**
- Curated preset (`preset_id`): run `run_single_bird_pdi(preset_path, config=state.pdi_config, optical_override=override, return_xray=True)`, score strictly with `state.bundle["fold_pipelines"][preset.fold]` (`oof_evaluation=True`), compute `oof_audit` (`signed_err_g`, `abs_err_g`, `pct_err`).
- User upload (`bytes`): run `run_single_bird_pdi(raw_bytes, config=state.pdi_config, optical_override=override, return_xray=True)`, score with `state.bundle["production_pipeline"]` (`oof_evaluation=False`, `oof_audit=None`).
- Prefix Data URIs explicitly: `stage1_original`, `stage2_clahe`, `stage4_overlay` → `f"data:image/jpeg;base64,{...}"`; `stage3_mask` → `f"data:image/png;base64,{...}"` (matching `backend/app/pdi/xray.py`, lines 175-178).
- Preserve exact ordered `39`-feature dict `{col: float(pdi_res.features[col]) for col in state.bundle["feature_names"]}` for `model_features`, plus curated 14-key `features_summary`.

---

### `backend/app/api/services/watershed_service.py` *(service, transform)*

**Analog:** `backend/app/pdi/watershed.py` + `backend/app/modeling/models.py` + `backend/app/modeling/artifacts.py`

**Isolated Watershed invocation, BGR palette & disclaimer constants** (`backend/app/pdi/watershed.py`, lines 76-85, 435-445; `backend/app/modeling/artifacts.py`, lines 132-136):
```python
# backend/app/pdi/watershed.py:76-85 — BGR tuples; convert (b, g, r) -> f"#{r:02X}{g:02X}{b:02X}"
_BASIN_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (210, 185, 45),   # #2DB9D2 Teal / cyan
    (55, 175, 245),   # #F5AF37 Warm amber
    (235, 130, 85),   # #5582EB Sky blue
    (130, 95, 240),   # #F05F82 Coral rose
    (105, 225, 120),  # #78E169 Emerald green
    (225, 115, 200),  # #C873E1 Orchid violet
    (70, 220, 235),   # #EBDC46 Golden yellow
    (180, 155, 255),  # #FF9BB4 Soft salmon
)

def run_watershed_pdi(
    source: Path | bytes,
    *,
    config: PDIConfig | None = None,
    optical_override: OpticalOverride | None = None,
    h_rel: float | None = None,
    min_distance_rel: float | None = None,
    min_area_ratio: float | None = None,
    return_visuals: bool = True,
) -> WatershedResult: ...
```
```python
# backend/app/modeling/artifacts.py:132-136
_WATERSHED_GROUP_SUM_DISCLAIMER: str = (
    "Pesagem de grupo na base pública: apenas o peso total conjunto das aves foi "
    "registrado; estimativas individuais por ave obtidas via Watershed são ilustrativas "
    "e não possuem gabarito individual."
)
```
**Service scoring & Data URI rules:**
- Score every `inst.features` via `predict_from_feature_dict(state.bundle["production_pipeline"], state.bundle["feature_names"], inst.features)` inside a timed inference block; handle `bird_count == 0` cleanly (`estimated_total_weight_g = 0.0`, `estimated_mean_weight_g = None`, `instances = []`).
- Map Stage 1/3/4 visuals to `data:image/jpeg;base64,...` and `stage2_union_mask` to `data:image/png;base64,...` (matching `backend/app/pdi/watershed.py`, lines 421-424).
- Always attach `ground_truth_scope = "group_sum_only"` and the exact pt-BR disclaimer string; populate `preset_reference` (`folder_id`, `expected_bird_count`, `group_sum_weight_g`, `mean_bird_weight_reference_g`, `sum_diff_g`, `sum_pct_err`) strictly for curated presets.

---

### `backend/app/api/services/metrics_service.py` *(service, file-I/O)*

**Analog:** `backend/app/modeling/artifacts.py` (`_assert_no_proibicao4_leaks`, `EVALUATION_REPORT_SCHEMA_VERSION`)

**Startup JSON loading & leak-check pattern** (`backend/app/modeling/artifacts.py`, lines 180-187):
```python
def _assert_no_proibicao4_leaks(serialized_text: str, *, context_label: str) -> None:
    """Reject forbidden raw-filename or filesystem-path tokens before touching disk."""
    lowered = serialized_text.lower()
    for token in _FORBIDDEN_LEAK_SUBSTRINGS:
        if token in lowered:
            raise ValueError(
                f"PROIBIÇÃO 4 violation: forbidden token {token!r} detected in {context_label}."
            )
```
Read `EVALUATION_REPORT_PATH`, `AUDIT_SUMMARY_PATH`, and `PDI_CALIBRATION_AUDIT_PATH` once during `lifespan`, verify `pdi_config_hash` parity across the loaded bundle, `evaluation_report`, and `pdi_calibration_audit`, and cache the assembled dict in `AppState.metrics_payload` for zero-I/O `< 2 ms` responses on `GET /api/metrics`.

---

### `backend/tests/test_api_health_presets_metrics.py` *(test, request-response)*

**Analog:** `backend/tests/test_train_and_evaluate_integration.py`

**`PROIBIÇÃO 4` leak scan & AST compliance pattern** (`backend/tests/test_train_and_evaluate_integration.py`, lines 790-842, 916-950):
```python
# Import canonical forbidden markers directly from artifacts.py (avoid duplicating path strings)
from backend.app.modeling.artifacts import _FORBIDDEN_LEAK_SUBSTRINGS

# AST scan over every .py file in backend/app/api/:
banned_dl_ocr = {
    "torch", "torchvision", "torchaudio", "tensorflow", "keras",
    "ultralytics", "timm", "onnxruntime", "paddleocr", "easyocr",
}
banned_cv_loaders = {"imread", "imdecode"}
```
**ASVS V4 path-containment parametrized test pattern** (`backend/tests/test_train_and_evaluate_integration.py`, lines 483-534):
Test `GET /api/health`, `GET /api/presets`, `GET /api/presets/{preset_id}/thumbnail` (all 10 valid presets returning JPEG `<= 320 px` + `Cache-Control`, plus traversal/unknown IDs like `../../etc/passwd` returning clean JSON `404`), `GET /api/metrics` (verifying `1,738` `oof_scatter_points` with `actual_g`, `16` model rows, `dataset_audit_summary`, `pdi_calibration_audit`), full JSON response scan against `_FORBIDDEN_LEAK_SUBSTRINGS`, and AST scan over `backend/app/api/**/*.py` confirming zero DL/OCR imports, zero `cv2.imread`/`cv2.imdecode`, and zero v2 barn-tracking/heatmap routes (`PROIBIÇÃO 1`, `2`, `4`, `5`). Use `with TestClient(create_app()) as client:` so FastAPI `lifespan` runs deterministically.

---

### `backend/tests/test_api_predict.py` *(test, request-response)*

**Analog:** `backend/tests/test_pdi_pipeline_and_xray.py` + `backend/tests/conftest.py`

**Synthetic EXIF/stripped-JPEG/RGBA-PNG fixtures & Data URI verification pattern** (`backend/tests/conftest.py`, lines 21-77; `backend/tests/test_pdi_pipeline_and_xray.py`, lines 260-300, 332-364):
```python
from backend.tests.conftest import make_synthetic_jpeg_bytes, make_synthetic_png_bytes

# Reuse synthetic scene builders + Pillow WEBP encoder:
# - All 6 single_bird_presets via POST /api/predict (preset_id=...)
#   -> verify oof_evaluation is True, exact fold_pipelines[fold] parity <= 0.01 g vs catalog,
#      sb_dual_lens_wide vs sb_dual_lens_ultrawide live delta < 2.0 g,
#      stage1/2/4 start with "data:image/jpeg;base64," and stage3 starts with "data:image/png;base64,",
#      timing.total_ms < 500.0
# - Uploaded bytes: EXIF JPEG, stripped JPEG (exif_optical_present=False, defaults 4.25/1.0),
#   RGBA PNG, WEBP -> oof_evaluation is False, oof_audit is None
# - Manual optical override (focal_length_mm=1.65, zoom_multiple=0.60) -> override_applied=True
# - Validation/security rejections:
#   both file+preset_id -> 422; neither -> 422; blank preset_id -> 422; unknown preset_id -> 404;
#   out-of-range/NaN focal or zoom -> 422; >15 MiB -> 413; BMP/GIF -> 415; corrupt/empty bytes -> 400
```

---

### `backend/tests/test_api_watershed_and_spa.py` *(test, request-response)*

**Analog:** `backend/tests/test_pdi_watershed.py`

**Parametrized ASVS V5 slider-bounds & 4-stage visual contract pattern** (`backend/tests/test_pdi_watershed.py`, lines 150-200, 450-469):
```python
from backend.app.pdi.config import (
    WATERSHED_MAX_AREA_RATIO,
    WATERSHED_MAX_DISTANCE_REL,
    WATERSHED_MAX_H_REL,
    WATERSHED_MIN_AREA_RATIO,
    WATERSHED_MIN_DISTANCE_REL,
    WATERSHED_MIN_H_REL,
)

# Parametrized HTTP 422 tests for out-of-bounds / NaN / Inf slider params:
# h_rel in (0.04, 0.81, "nan", "inf"), min_distance_rel in (0.19, 2.51),
# min_area_ratio in (0.005, 0.46)
# Preset tests across ws_417, ws_757, ws_823, ws_1520:
# - ground_truth_scope == "group_sum_only" + exact pt-BR disclaimer
# - stage2_union_mask starts with "data:image/png;base64,", stages 1/3/4 with "data:image/jpeg;base64,"
# - each instance color_hex matches #RRGGBB from _BASIN_PALETTE_BGR
# - custom synthetic dark image -> bird_count == 0, estimated_mean_weight_g is None
# - Single-origin SPA fallback test (tmp_path fake dist/index.html + dist/assets/app.js):
#   GET / and GET /estudio return 200 HTML; GET /api/unknown still returns 404 JSON!
```

---

## Shared Patterns

### 1. Python Module Conventions & Ruff Quality Gate
**Source:** `pyproject.toml` (lines 35-49) & `backend/app/config.py` (lines 1-10)
**Apply to:** All files in `backend/app/api/` and `backend/tests/test_api_*.py`
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["backend/tests"]
addopts = ["-ra", "-q"]

[tool.ruff]
target-version = "py312"
line-length = 100
src = ["."]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.ruff.lint.isort]
known-first-party = ["backend"]
```
Every Python file starts with a module docstring + `from __future__ import annotations`, uses Python 3.12 PEP 604 unions (`X | None`), enforces `line-length <= 100`, uses `@dataclass(frozen=True, slots=True)` for internal immutable value objects, and defines explicit `__all__`.

### 2. ASVS V4 Path Containment (Zero Arbitrary Reads / Zero Symlinks)
**Source:** `backend/app/modeling/artifacts.py` (lines 253-295)
**Apply to:** `backend/app/api/services/preset_service.py` and `backend/tests/test_api_health_presets_metrics.py`
Never sanitize paths with string `replace("..", "")`. Reuse `_resolve_safe_image_path(rel_path, RAW_DATASET_DIR.resolve())` (canonical root check + per-segment non-symlink verification + `is_relative_to` + `is_file`).

### 3. ASVS V5 Authoritative Parameter Bounds (Single Source of Truth)
**Source:** `backend/app/pdi/config.py` (lines 18-23, 104-106) & `backend/app/pdi/io_exif.py` (lines 29-35)
**Apply to:** `backend/app/api/schemas.py`, `backend/app/api/routers/predict.py`, `backend/app/api/routers/watershed.py`, and `frontend/src/components/watershed/WatershedSlidersCard.tsx`
- Optical bounds: `focal_length_mm ∈ [0.5, 200.0]` (default `4.25`), `zoom_multiple ∈ [0.1, 20.0]` (default `1.0`).
- Watershed bounds (reconciled against Phase 2 lock & approved `04-UI-SPEC.md`):
  - `h_rel ∈ [0.05, 0.80]` (default `0.20`, UI step `0.01`)
  - `min_distance_rel ∈ [0.20, 2.50]` (default `0.85`, UI step `0.05`)
  - `min_area_ratio ∈ [0.01, 0.45]` (default `0.10`, UI step `0.01`)
Import these constants directly in Python and expose `parameters.defaults` + `parameters.bounds` in `WatershedResponse`.

### 4. Non-Blocking FastAPI CPU Offload (`D-03`)
**Source:** `04-CONTEXT.md` `D-03` & `04-RESEARCH.md` Pattern 1–2
**Apply to:** `backend/app/api/routers/predict.py`, `backend/app/api/routers/watershed.py`, `backend/app/api/routers/presets.py`
Define upload/inference routes as `async def` so `await file.read(MAX_UPLOAD_BYTES + 1)` executes asynchronously, then offload all synchronous Pillow/OpenCV/scikit-image/LightGBM work via `await run_in_threadpool(...)` (`from starlette.concurrency import run_in_threadpool`).

### 5. Single-Source EXIF-First Ingestion & Exact Base64 Data URI MIME Types
**Source:** `backend/app/pdi/io_exif.py` (lines 217-238, 317-376), `backend/app/pdi/xray.py` (lines 175-178), `backend/app/pdi/watershed.py` (lines 421-424)
**Apply to:** All `backend/app/api/services/*.py` and `frontend/src/components/{studio,watershed}/`
- Never call `cv2.imread()` or `cv2.imdecode()` anywhere in `backend/app/api/`.
- Assemble full RFC 2397 Data URIs server-side:
  - X-Ray Stages 1, 2, 4 & Watershed Stages 1, 3, 4 → `data:image/jpeg;base64,<raw_b64>`
  - X-Ray Stage 3 (`stage3_mask`) & Watershed Stage 2 (`stage2_union_mask`) → `data:image/png;base64,<raw_b64>`

### 6. `PROIBIÇÃO 4` Zero Raw Filename / Filesystem Path Leakage
**Source:** `backend/app/modeling/artifacts.py` (lines 66-73, 180-187)
**Apply to:** Every API response, HTTP error handler, log line, and React UI component
Import `_FORBIDDEN_LEAK_SUBSTRINGS` in API tests and assert zero forbidden tokens across `/api/health`, `/api/presets`, `/api/metrics`, `/api/predict`, `/api/watershed-demo`, and `400`/`404`/`413`/`415`/`422`/`500` error bodies. Never expose `rel_path`, raw dataset filenames, or absolute paths.

### 7. Greenfield Frontend Architecture, UI Tokens & Recharts v3 Rules
**Source:** `04-UI-SPEC.md` (approved) & `04-RESEARCH.md` Patterns 4–5
**Apply to:** All `frontend/` files
- **Toolchain (`frontend/`):** React `19.3.1`, TypeScript `7.0.2` (`tsc --noEmit`), Vite `8.3.8`, Tailwind CSS `4.3.5` (`@tailwindcss/vite`, `@theme` tokens in `src/index.css`, zero `tailwind.config.js`), Biome `2.6.8` (`assist.actions.source.organizeImports = "on"`, `css.parser.tailwindDirectives = true`, zero ESLint/`typescript-eslint`), Recharts `3.11.1` + matching `react-is@19.3.1`, `lucide-react@1.79.0`, `clsx` + `tailwind-merge`, Vitest `5.1.1`.
- **Visual & Accessibility Contract (`04-UI-SPEC.md`):**
  - Strictly 4 font sizes (`12px`, `14px`, `20px`, `32px`), 2 weights (`400`, `600`), `tabular-nums` on every numeric readout, `font-mono` only for hashes/IDs/formulas/39-feature keys, offline-safe system font stack (zero blocking CDN font links).
  - Surface split: `#F8FAFC` canvas, `#FFFFFF`/`#F1F5F9` cards, `#0F172A` strictly inside 4-stage X-Ray & Watershed image frames (`object-contain` for mixed `288×640` portrait / `640×288` landscape frames), `#047857` accent restricted to the 7 approved elements, `#B45309` amber strictly for scientific warnings, `#B91C1C` strictly for errors / confirmed clear-batch step.
  - Every icon-only button `min-h-[44px] min-w-[44px]` with `type="button"` and exact pt-BR `aria-label`.
  - Every CTA follows pt-BR `Verb + Noun` copywriting locked in `04-UI-SPEC.md`.
- **Request Cancellation & `FormData` Hygiene:** Omit empty optional numeric fields from `FormData`; cancel superseded `/api/predict` and `/api/watershed-demo` requests via `AbortController` and ignore `AbortError`.
- **Recharts v3 (`3.11.1`) Rules:** Never use deprecated `<Cell />` (pass `fill` on data items); wrap `<ResponsiveContainer>` in `<div className="h-[360px] w-full min-w-0">`; set `isAnimationActive={false}` on `1,738`-point `<Scatter>`; use exact field name `actual_g` (not `weight_g`); declare numeric `type="number"` + `domain={[0, 2200]}` on both axes for dashed `y = x` `<ReferenceLine segment={[{ x: 0, y: 0 }, { x: 2200, y: 2200 }]} />`.
- **Conceptual Scope Guard (`PROIBIÇÃO 5`):** `FutureVisionTab.tsx` is 100% static documentation with zero API calls, zero backend routes, and zero fake telemetry.

---

## No Analog Found

Files with no existing analog in the repository (planner MUST reference `04-RESEARCH.md` & approved `04-UI-SPEC.md` directly):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `backend/app/api/routers/health.py` | route | request-response | First FastAPI `APIRouter` module in repo (see `04-RESEARCH.md` Pattern 1) |
| `backend/app/api/routers/presets.py` | route | request-response | First FastAPI `APIRouter` + binary JPEG `Response` cache endpoint in repo |
| `backend/app/api/routers/predict.py` | route | request-response | First FastAPI multipart `UploadFile` XOR `Form` + `run_in_threadpool` route |
| `backend/app/api/routers/watershed.py` | route | request-response | First FastAPI multipart Watershed route with bounded `Form` sliders |
| `backend/app/api/routers/metrics.py` | route | request-response | First FastAPI cached audit JSON route |
| `frontend/package.json` | config | file-I/O | `frontend/` is currently empty (greenfield React 19 / Vite 8 / TS 7 workspace) |
| `frontend/vite.config.ts` | config | request-response | Greenfield Vite 8 + `@tailwindcss/vite` + `/api` proxy (`04-RESEARCH.md` Pattern 4) |
| `frontend/tsconfig.json` | config | file-I/O | Greenfield TypeScript 7 project-references root |
| `frontend/tsconfig.app.json` | config | file-I/O | Greenfield TS 7 browser compiler config (`04-RESEARCH.md` Pattern 4) |
| `frontend/tsconfig.node.json` | config | file-I/O | Greenfield TS 7 Node/Vite config |
| `frontend/biome.json` | config | file-I/O | Greenfield Biome v2.6.8 config (`tailwindDirectives: true`, `04-RESEARCH.md` Pattern 4) |
| `frontend/index.html` | config | file-I/O | Greenfield pt-BR SPA HTML shell (offline-safe fonts) |
| `frontend/src/main.tsx` | component | event-driven | Greenfield React 19 `createRoot` entry |
| `frontend/src/index.css` | config | transform | Greenfield Tailwind CSS v4 `@import "tailwindcss";` + `@theme` tokens (`04-UI-SPEC.md`) |
| `frontend/src/App.tsx` | component | event-driven | Greenfield 4-tab hash-synced SPA root (`#estudio`, `#watershed`, `#metricas`, `#visao-futura`) |
| `frontend/src/api/types.ts` | model | transform | Greenfield TS types mirroring `backend/app/api/schemas.py` & `models/*.json` |
| `frontend/src/api/client.ts` | service | request-response | Greenfield native `fetch` + `AbortSignal` + pt-BR `ApiError` client |
| `frontend/src/utils/cn.ts` | utility | transform | Greenfield `clsx` + `tailwind-merge` helper |
| `frontend/src/utils/format.ts` | utility | transform | Greenfield pt-BR numeric/weight/percent/nullable `"—"` formatters |
| `frontend/src/utils/flock.ts` | utility | transform | Greenfield client-side flock mean/sample-SD/`CV%` math |
| `frontend/src/utils/scatter.ts` | utility | transform | Greenfield filter-before-aggregate `1,738 → 188` session scatter helper |
| `frontend/src/hooks/useHealth.ts` | hook | request-response | Greenfield React hook for `GET /api/health` |
| `frontend/src/hooks/usePresets.ts` | hook | request-response | Greenfield React hook for `GET /api/presets` |
| `frontend/src/hooks/useMetrics.ts` | hook | request-response | Greenfield React hook for lazy cached `GET /api/metrics` |
| `frontend/src/hooks/useFlockBatch.ts` | hook | event-driven | Greenfield React hook for simulated flock state & 2-step clear guard |
| `frontend/src/utils/__tests__/*.test.ts` | test | transform | Greenfield Vitest 5.1.1 pure-logic unit tests |
| `frontend/src/components/layout/*.tsx` | component | event-driven | Greenfield shell, `44×44` icon controls, keyboard lightbox, error banner (`04-UI-SPEC.md`) |
| `frontend/src/components/studio/*.tsx` | component | request-response | Greenfield Tab 1 Estúdio Raio-X, 6 presets, Dual-Lens comparison, EXIF override, flock tracker |
| `frontend/src/components/watershed/*.tsx` | component | request-response | Greenfield Tab 2 Watershed studio, 4 presets, 3 ASVS V5 sliders, instance table |
| `frontend/src/components/metrics/*.tsx` | component | transform | Greenfield Tab 3 Recharts v3 dashboard (`ScatterChart`, ablation `BarChart`, tables, Mendeley funnel) |
| `frontend/src/components/vision/FutureVisionTab.tsx` | component | transform | Greenfield Tab 4 static conceptual v2 roadmap (`PROIBIÇÃO 5`) |

---

## Metadata

**Analog search scope:** `backend/app/`, `backend/tests/`, `models/*.json`, `frontend/` (verified empty), `pyproject.toml`, `.gitignore`
**Files scanned:** 17 tracked files read in full or targeted non-overlapping sections; all 21 cited analog/config paths verified git-tracked via `git ls-files -- <path>`
**Pattern extraction date:** 2026-11-24
