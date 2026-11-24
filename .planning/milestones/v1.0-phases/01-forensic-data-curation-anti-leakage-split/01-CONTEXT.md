# Phase 1: Forensic Data Curation & Anti-Leakage Split - Context

**Gathered:** 2026-11-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a reproducible `uv`-managed Python 3.12 backend environment (`--only-binary :all:`), a shared Pillow EXIF-first metadata/orientation module (`backend/app/pdi/io_exif.py`), a deterministic Audit §9 forensic dataset ingestor crawling `data/raw/Broiler weights of Cobb Avian/`, immutable typed Parquet manifests (`data/processed/manifest_clean.parquet` with 1,738 clean images / 160 unique weights / 188 sessions and `data/processed/manifest_quarantine.parquet` with 15 quarantined images), a git-tracked audit summary (`models/dataset_audit_summary.json`), and reproducible zero-leakage 5-fold `StratifiedGroupKFold` assignments grouped strictly by `session_id` — backed by automated exact-count and anti-leakage tests (`DATA-01`, `DATA-02`).

Classical image preprocessing/segmentation/feature extraction (`PDI-01`..`PDI-05`), regression modeling (`MOD-01`..`MOD-03`), and FastAPI/React serving (`WEB-01`..`WEB-05`) belong strictly to Phases 2–4.

</domain>

<decisions>
## Implementation Decisions

### Python Workspace & Package Layout
- **D-01:** Place `.python-version` (`3.12`), `pyproject.toml`, and `uv.lock` at the repository root (resolving strictly against pre-installed `cpython-3.12.14` via `uv`, decoupled from host `/usr/bin/python3.14`), organizing backend source under `backend/app/` (`backend/app/ingestion/`, `backend/app/pdi/io_exif.py`), CLI entrypoints under `backend/scripts/` (`backend/scripts/build_manifest.py`), and tests under `backend/tests/` with `[tool.pytest.ini_options] pythonpath = ["."]` — **Reversibility:** costly — changing package root or import prefix (`backend.app.*`) later touches every module and test across Phases 2–4.
- **D-02:** Declare the verified core backend/PDI/ML/API dependency pins from `.planning/research/STACK.md` in root `pyproject.toml` (`opencv-python-headless>=4.14.0,<5.0`, `scikit-image>=0.26.0,<0.27`, `scipy>=1.18.0,<2.0`, `pillow>=13.0.0,<14.0`, `numpy>=2.5.0,<3.0`, `pandas>=3.1.0,<4.0`, `pyarrow>=26.0.0,<27.0`, `scikit-learn>=1.9.0,<2.0`, `lightgbm>=4.7.0,<5.0`, `joblib>=1.6.0,<2.0`, `fastapi>=0.165.0,<1.0`, `uvicorn[standard]>=0.54.0,<1.0`, `pydantic>=2.14.0,<3.0`, `python-multipart>=0.0.34`; dev deps `pytest`, `httpx`, `ruff`) and resolve exclusively with binary wheels (`--only-binary :all:`), explicitly omitting all Deep Learning / OCR / GUI OpenCV packages (`PROIBIÇÃO 1 & 2`).

### Parquet Manifest Schema & Derived Forensic Columns
- **D-03:** Persist all 15 Audit §9 base columns (`rel_path`, `folder_chain`, `raw_weight_g`, `weight_g`, `capture_dt`, `capture_date`, `oriented_w`, `oriented_h`, `exif_orientation`, `focal_length_mm`, `zoom_multiple`, `is_multi_bird`, `is_severe_mislabel`, `is_single_bird_clean`, `session_id`) plus 9 canonical derived columns (`image_id`, `filename_pattern`, `was_cp866_recovered`, `was_nested_folder`, `was_lcd_corrected`, `lens_class`, `weight_band`, `quarantine_reason`, `fold`) using POSIX forward-slash paths and Pandas 3 Copy-on-Write safe mutations (`.loc[mask, col] = val`) — **Reversibility:** one-way — Phases 2 (`extract_features.py`), 3 (`train_and_evaluate.py`), and 4 (`presets` & `audit` services) consume this exact Parquet column contract.
- **D-04:** Enforce Audit §9 curation rules verbatim without ever modifying files inside `data/raw/`: recursive level-2 leaf traversal (`171/173`, `190/191`, `190/193`), numeric extraction regex `r"\d+"` sanitizing `1340_` → `1340`, `LCD_CORRECTIONS = {"367": 267}`, CP866 mojibake reversal (`name.encode("cp866").decode("utf-8")` when Arabic-Indic digits `U+0660–U+0669` appear) + Arabic-to-ASCII translation, `capture_dt = fn_dt or (mtime_dt - timedelta(seconds=14401))`, quarantine sets `MULTI_BIRD_FOLDERS = {"417", "757", "823", "1520"}` (13 imgs) and `SEVERE_MISLABEL_FOLDERS = {"1351", "1465"}` (2 imgs), and `session_id = f"{folder_chain}__s{sid}"` splitting on sequential gap `> 600 s` inside each `folder_chain`.

### StratifiedGroupKFold Stratification & Determinism
- **D-05:** Sort clean records deterministically by `["folder_chain", "capture_dt", "rel_path"]` before splitting and assign `fold` (`int8`, `0..4`) using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly by `session_id` and stratified on the 3 coarse PRD weight bands (`<500g`: `weight_g < 500`, `500-1000g`: `500 <= weight_g <= 1000`, `>1000g`: `weight_g > 1000`), while setting `fold = -1` in `manifest_quarantine.parquet` — **Reversibility:** one-way — changing fold assignments alters every downstream 5-fold CV metric and out-of-fold preset selection in Phases 3–4.
- **D-06:** Gate Phase 1 completion with automated pytest assertions verifying: exact row/weight/session totals (`1,753` total = `1,738` clean + `15` quarantined; `160` unique clean weights; `188` unique clean sessions), `0` dates in year `2024` (proving CP866 `547`/`562` timestamp recovery), `367` corrected to `267 g`, `1340_` sanitized to `1340 g`, `171/173`, `190/191`, and `190/193` captured with leaf weights, `0` `session_id` overlap across folds, `0` sklearn warnings during split generation, `≥ 4` heavy (`>1000g`) sessions per fold, and byte-deterministic fold reproducibility.

### Shared Pillow EXIF Loader & Audit Summary JSON
- **D-07:** Build `backend/app/pdi/io_exif.py` in Phase 1 as the single shared Pillow EXIF-first module for the entire repository (banning `cv2.imread`/`cv2.imdecode` repo-wide), exposing `read_exif_metadata(source: Path | bytes)` (fast header-only extraction of `Orientation` `274`, `FocalLength` `37386`, Xiaomi IFD `0x8769` JSON tag `39321` `zoomMultiple`, and pre/post-transpose dimensions) and `load_oriented_rgb(source: Path | bytes)` (extracting EXIF **before** `ImageOps.exif_transpose(im).convert("RGB")`, gracefully flagging missing EXIF for future uploads).
- **D-08:** Generate git-tracked `models/dataset_audit_summary.json` (containing aggregate statistics ONLY — zero raw file paths or image dumps per `PROIBIÇÃO 4`) reporting: Mendeley published claims (`1,772` imgs / `"daily for 45 days"`) vs. disk audit (`1,753` raw / `9` dates / `23` calendar days / `~7–29 days` / `116–2,093 g`), curation funnel counts (`13` multi-bird, `2` severe mislabels, `6` LCD corrections `367→267 g`, `1,738` clean, `160` unique weights, `188` sessions), encoding/EXIF recovery metrics (`Orientation != 1`, CP866, Arabic-Indic, Windows-renamed, nested level-2 folders), weight-band and lens distributions (explicitly quantifying the `801–1,216 g` sparse gap), and per-fold balance (`n_images`, `n_sessions`, `n_heavy_sessions`) ready for Phase 4's `WEB-05` Audit Dashboard.

### the agent's Discretion
- Exact module decomposition inside `backend/app/ingestion/` (e.g., `decoding.py`, `manifest.py`, `splits.py`, `audit_summary.py`) and CLI flag ergonomics (`--dataset-root`, `--output-dir`, `--summary-path`, `--dry-run`) in `backend/scripts/build_manifest.py`.
- Exact `ruff` lint/format configuration rules in `pyproject.toml` (keeping `target-version = "py312"`).
- Using synthetic unit fixtures (in-memory JPEGs with crafted EXIF tags, CP866 strings, nested temp dirs) alongside read-only integration assertions against the generated Parquet manifests so pytest runs fast and never dumps `data/raw/` filenames.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Scope, Requirements & prohibitions
- `PRD.md` — Closed-scope product requirements (`REQ-01`, `REQ-02`, `PROIBIÇÃO 1–5`) and data contract (§2).
- `.planning/PROJECT.md` — Core value, constraints, and locked ingestion/reconciliation decisions.
- `.planning/REQUIREMENTS.md` — Formal v1 requirements `DATA-01`, `DATA-02`, and downstream consumers `PDI-01`, `WEB-05`.
- `.planning/ROADMAP.md` — Phase 1 goal and 4 observable Success Criteria.

### Forensic Dataset Audit (Single Source of Truth)
- `docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md` §2, §4, §5, §6, §7, §8.1–8.2, §9 — Exact file counts, encoding rules, Xiaomi tag `39321` schema, quarantine lists, `367→267 g` correction, `MTIME_OFFSET_SEC = 14401`, and reference Python ingestion pipeline.
- `.planning/intel/context.md` — Consolidated forensic dataset facts extracted by `/gsd-ingest-docs`.
- `.planning/intel/SYNTHESIS.md` — Document ingestion synthesis and resolved proposal reconciliations.

### Research & Architecture Contracts
- `.planning/research/STACK.md` — Verified Python 3.12 (`uv`) binary-wheel pins, Pandas 3.1 Copy-on-Write rules, and PyArrow Parquet notes.
- `.planning/research/ARCHITECTURE.md` — `backend/app/{ingestion,pdi}` module layout, shared `io_exif.py` contract, and artifact boundary schemas.
- `.planning/research/PITFALLS.md` — Pitfalls #6 (stratification & empty strata), #7 (CP866 `mtime`, `cv2.imdecode` EXIF desync), and #8 (host Python 3.14 vs `uv` Python 3.12).
- `.planning/research/SUMMARY.md` — Reconciled engineering standards across all 4 research tracks.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `.gitignore`: Already excludes `data/raw/`, `data/processed/`, `models/*.joblib`, `models/*.pkl`, `*.jpg`, `*.jpeg`, `*.png`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `node_modules/`, `frontend/dist/`. Verify `models/dataset_audit_summary.json` is NOT ignored (it is `.json`, so git-trackable as required).
- `docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md` §9: Contains tested reference Python logic for `parse_timestamp_from_name()`, `build_clean_manifest()`, and `load_oriented_rgb()`.

### Established Patterns
- Greenfield repository: `backend/`, `frontend/`, `models/`, and `data/processed/` are currently empty directories ready for clean scaffolding.
- Local dataset verified present on disk at `data/raw/Broiler weights of Cobb Avian/` (`165` direct subdirectories).
- Pre-installed standalone `cpython-3.12.14` available to `uv` at `/home/philzerantc/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12`.

### Integration Points
- `backend/app/pdi/io_exif.py` created in Phase 1 becomes the mandatory image/EXIF entrypoint imported by Phase 2 (`backend/app/pdi/pipeline.py`, `watershed.py`) and Phase 4 (`backend/app/api/routers/predict.py`).
- `data/processed/manifest_clean.parquet` & `manifest_quarantine.parquet` feed directly into Phase 2 (`extract_features.py` and Watershed validation on `417`/`757`/`823`/`1520`) and Phase 3 (`train_and_evaluate.py`).
- `models/dataset_audit_summary.json` feeds directly into Phase 4 (`GET /api/metrics` and Tab 3 `WEB-05` Forensic Audit Funnel).

</code_context>

<specifics>
## Specific Ideas

- Never recursively list `data/raw/` filenames in stdout logs, test failure dumps, or JSON summary artifacts (`PROIBIÇÃO 4`); report only aggregate counts, folder chains of quarantined/anomalous groups, and summary tables.
- Record `was_mtime_fallback` (or deduce from `filename_pattern == "weight_renamed"`) so the ingestor can assert that `0` CP866 files (`547/`, `562/`) fell back to their 2024 repackaging `mtime`.
- Keep `build_manifest.py` idempotent: re-running it produces identical Parquet row ordering, `session_id`s, `fold` integers, and `models/dataset_audit_summary.json` metrics.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed strictly within Phase 1 scope. (Offline LCD OCR audit `HARD-01` remains deferred to v1.x per `REQUIREMENTS.md`.)

</deferred>

---

*Phase: 01-forensic-data-curation-anti-leakage-split*
*Context gathered: 2026-11-20*
