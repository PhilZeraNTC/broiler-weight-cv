# Walking Skeleton — AvianScale CV

**Phase:** 1
**Generated:** 2026-11-20

## Capability Proven End-to-End

A researcher can execute one reproducible Python 3.12 (`uv`) CLI command (`uv run python -m backend.scripts.build_manifest`) over the read-only raw Cobb Avian broiler dataset to read EXIF optical metadata through the repository's single shared Pillow EXIF-first loader and materialize immutable, audited, zero-leakage 5-fold Parquet manifests (`1,738` clean / `15` quarantined) plus a git-tracked, zero-path-leakage aggregate JSON summary ready for downstream classical PDI extraction, tabular regression, and pitch serving.

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Python Runtime & Dependency Manager | Root `.python-version` (`3.12`), `pyproject.toml`, and `uv.lock` pinned to standalone `cpython-3.12.14` (`[tool.uv] package = false`, `no-build = true`, `python-preference = "only-managed"`) (`D-01`, `D-02`) | Permanently decouples AvianScale CV from host rolling `/usr/bin/python3.14`, disables arbitrary sdist code execution via binary-only wheels, and guarantees uniform `backend.app.*` imports across CLI scripts, pytest, and Phase 4 Uvicorn. |
| Dependency Allowlist & Prohibitions | Locked 17-package backend/PDI/ML/API allowlist from `STACK.md` (`opencv-python-headless`, `scikit-image`, `scipy`, `pillow`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `lightgbm`, `joblib`, `fastapi`, `uvicorn[standard]`, `pydantic`, `python-multipart`; dev `pytest`, `httpx`, `ruff`) (`D-02`) | Strictly honors `PROIBIÇÃO 1 & 2` (zero Deep Learning, zero OCR, zero GUI OpenCV) while locking the full backend dependency graph once in Phase 1. |
| Shared Image & EXIF Entrypoint | Single module `backend/app/pdi/io_exif.py` (`read_exif_metadata`, `load_oriented_rgb`) accepting `Path \| bytes`, extracting EXIF/IFD `0x8769`/Xiaomi tag `39321` BEFORE `ImageOps.exif_transpose()` (`D-07`) | Eliminates orientation desync and EXIF loss across Phase 1 ingestion, Phase 2 offline PDI extraction, and Phase 4 `/api/predict` uploads; bans raw OpenCV image loaders repo-wide. |
| Curated Data Layer | Git-ignored typed PyArrow Parquet manifests (`data/processed/manifest_clean.parquet`, `data/processed/manifest_quarantine.parquet`) following the exact 24-column contract (`15` Audit §9 base + `9` derived columns) (`D-03`, `D-04`) | Preserves `timestamp[us]`, `date32[day]`, nullable Arrow `string`, `int8`, `int32`, `float64`, `bool`, and CP866/Arabic-Indic Unicode losslessly under Pandas 3.1 Copy-on-Write while keeping raw images strictly read-only. |
| Cross-Validation Split Contract | Deterministic 5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly by `session_id` (`gap > 600.0 s` within `folder_chain`) and stratified by weight band (`<500g`, `500-1000g`, `>1000g`), pre-sorted by `["folder_chain", "capture_dt", "rel_path"]` (`D-05`, `D-06`) | Enforces `PROIBIÇÃO 3` (zero burst-capture split leakage across folds), balances heavy `>1000g` sessions (`≥ 4` per fold) with `0` sklearn warnings, and freezes folds once for all downstream Phase 3–4 metrics and OOF presets. |
| Public Audit Artifact | Git-tracked aggregate `models/dataset_audit_summary.json` containing counters, funnel reconciliations, and fold summaries ONLY — zero raw file paths or wall-clock timestamps (`D-08`) | Satisfies `PROIBIÇÃO 4` data hygiene while providing Phase 4's `WEB-05` Audit Dashboard a deterministic, lightweight JSON contract. |
| Directory Layout | `backend/app/{config.py,pdi/,ingestion/}`, `backend/scripts/build_manifest.py`, `backend/tests/` with `[tool.pytest.ini_options] pythonpath = ["."]` (`D-01`) | Clean separation between shared core (`pdi/io_exif.py`), offline curation (`ingestion/`), CLI entrypoints (`scripts/`), and fast unit + integration tests (`tests/`). |
| Auth / External Database / Cloud Deployment | Explicitly None (local single-user CLI + future local FastAPI/Vite MVP) | Zero operational overhead required for a local academic/product pitch MVP targeting code freeze `2026-11-28`. |

## Stack Touched in Phase 1

- [x] Project scaffold — root `uv` Python 3.12 workspace (`.python-version`, `pyproject.toml`, `uv.lock`), `ruff` linter/formatter, and `pytest` runner (`backend/tests/`)
- [x] Shared I/O boundary — `backend/app/pdi/io_exif.py` Pillow EXIF-first reader & orientation transposer (`Path | bytes` parity, ASVS L1 bounds on Xiaomi tag `39321`)
- [x] Data layer — read-only crawl of `data/raw/Broiler weights of Cobb Avian/` AND typed PyArrow Parquet writes (`data/processed/manifest_clean.parquet`, `data/processed/manifest_quarantine.parquet`) plus aggregate JSON write (`models/dataset_audit_summary.json`)
- [x] Reproducible local execution — single idempotent command `uv run python -m backend.scripts.build_manifest` verified by automated `uv run pytest` and `uv run ruff check backend/`
- [ ] HTTP API & React UI — intentionally excluded from Phase 1 (scheduled for Phase 4; no fake UI or throwaway DB added to Phase 1)

## Out of Scope (Deferred to Later Slices)

- Classical PDI preprocessing, SF-400 LCD exclusion, single-bird segmentation, mask-gated morphometric/chromatic/optical feature extraction, 4-stage "X-Ray" rendering, and Marker-based Watershed (`PDI-01`..`PDI-05` → Phase 2)
- Interpretable tabular regression training, 4-way WITH vs. WITHOUT EXIF optical ablation, lens-flip audit, and dual-bundle `.joblib` / OOF presets publishing (`MOD-01`..`MOD-03` → Phase 3)
- Local `< 500 ms` FastAPI REST API (`uvicorn`) and 3-tab React 19 + TypeScript 7 + Vite 8 + Tailwind v4 pitch web studio (`WEB-01`..`WEB-05` → Phase 4)
- Post-pitch hardening (`HARD-01`..`HARD-05`: offline LCD OCR audit, bucket rim mm/px calibration, blur gate, LODO CV, batch export) and v2+ fixed overhead ceiling-camera barn deployment (`BARN-01`..`BARN-03`)
- Permanently prohibited (`PROIBIÇÃO 1–5`): Deep Learning / CNNs / YOLO / ViTs / OCR, outside-mask / LCD pixel features, random `train_test_split` / ungrouped K-Fold, committing `data/raw/` / `data/processed/` / `.joblib` / `.jpg`, and multi-frame video tracking in v1 code

## Subsequent Slice Plan

Each later phase adds one vertical capability slice directly on top of Phase 1's locked Python 3.12 workspace, `io_exif.py` loader, 24-column Parquet schema, and frozen 5-fold `session_id` splits without altering Phase 1's architectural decisions:

- **Phase 2 (`PDI-01`..`PDI-05`):** Build the single-source classical PDI engine atop `backend/app/pdi/io_exif.py` and `manifest_clean.parquet` to isolate silhouettes without LCD leakage, render 4-stage "X-Ray" outputs, materialize `data/processed/features_clean.parquet` (`n_jobs=4`), and validate Marker-based Watershed on the `13` quarantined multi-bird images in `manifest_quarantine.parquet`.
- **Phase 3 (`MOD-01`..`MOD-03`):** Train and benchmark log-target Ridge/Huber, Random Forest, and LightGBM on `features_clean.parquet` using Phase 1's frozen `fold` (`0..4`) splits, publish `models/evaluation_report.json`, `models/weight_regressor.joblib` (production + 5 OOF pipelines), and `models/presets_catalog.json`.
- **Phase 4 (`WEB-01`..`WEB-05`):** Expose `/api/health`, `/api/predict`, `/api/watershed-demo`, `/api/presets`, and `/api/metrics` (serving `models/dataset_audit_summary.json` + `models/evaluation_report.json`) via FastAPI and wire the 3-tab React + Tailwind CSS interactive pitch studio.
