# Phase 1: Forensic Data Curation & Anti-Leakage Split - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-11-20
**Phase:** 01-forensic-data-curation-anti-leakage-split
**Mode:** `--auto` (autonomous recommended-default selection across all gray areas)
**Areas discussed:** Python Workspace & Package Layout, Parquet Manifest Schema & Derived Forensic Columns, StratifiedGroupKFold Stratification & Determinism, Shared Pillow EXIF Loader & Audit Summary JSON

---

## Python Workspace & Package Layout

| Option | Description | Selected |
|--------|-------------|----------|
| Root `pyproject.toml` + `.python-version = 3.12` (`uv`) | Single `uv.lock` at repo root with `backend/app/`, `backend/scripts/`, `backend/tests/` and `pythonpath = ["."]` | ✓ |
| Nested `backend/pyproject.toml` | Isolate Python config inside `backend/`, requiring directory changes or `--project backend` flags | |
| System Python `3.14.10` venv | Use host `/usr/bin/python3.14` directly without `uv` pinning | |

**Selection (`--auto`):** Root `pyproject.toml` + `.python-version = 3.12` (`uv`)
**Notes:** Grounded in `.planning/research/STACK.md` and `PITFALLS.md` #8 — pre-installed standalone `cpython-3.12.14` isolates `.venv` from rolling OS Python 3.14 updates while keeping imports clean across `backend/scripts/` and `backend/app/`.

---

## Parquet Manifest Schema & Derived Forensic Columns

| Option | Description | Selected |
|--------|-------------|----------|
| Audit §9 base columns + 9 canonical derived columns (`image_id`, `filename_pattern`, `was_cp866_recovered`, `was_nested_folder`, `was_lcd_corrected`, `lens_class`, `weight_band`, `quarantine_reason`, `fold`) | Typed Arrow-backed Parquet manifests (`manifest_clean.parquet` & `manifest_quarantine.parquet`) avoiding recomputation in Phases 2–4 | ✓ |
| Minimal Audit §9 columns + `fold` only | Recompute `weight_band`, `lens_class`, and audit flags separately in Phases 2, 3, and 4 | |
| CSV + Parquet dual export | Write redundant CSV files alongside Parquet | |

**Selection (`--auto`):** Audit §9 base columns + 9 canonical derived columns (Parquet-only)
**Notes:** Matches `PRD.md` REQ-01, `STACK.md` (Pandas 3.1 Copy-on-Write + PyArrow), and `.planning/INGEST-CONFLICTS.md` INFO resolution (Parquet-only, no CSV).

---

## StratifiedGroupKFold Stratification & Determinism

| Option | Description | Selected |
|--------|-------------|----------|
| 5-fold `StratifiedGroupKFold(shuffle=True, random_state=42)` on 3 coarse PRD bands (`<500g`, `500-1000g`, `>1000g`) | Deterministic sort prior to split; guarantees `0` session overlap, `0` sklearn warnings, and `≥ 4` heavy (`>1000g`) sessions per fold | ✓ |
| Fine weight deciles (10 bins) for stratification | Risks empty-stratum warnings because the `801–1,216 g` desert contains only 1 valid bird (`1009 g`) | |
| Unshuffled `StratifiedGroupKFold(shuffle=False)` | Deterministic but can clump contiguous folder weights across fold assignment order | |

**Selection (`--auto`):** 5-fold `StratifiedGroupKFold(shuffle=True, random_state=42)` on 3 coarse PRD bands
**Notes:** Directly addresses `PITFALLS.md` #6a and `DATA-02` (`PRD REQ-02`) while protecting the sparse `>1,000 g` tail.

---

## Shared Pillow EXIF Loader & Audit Summary JSON

| Option | Description | Selected |
|--------|-------------|----------|
| Build shared `backend/app/pdi/io_exif.py` (`Path \| bytes`) in Phase 1 + emit git-tracked `models/dataset_audit_summary.json` | Zero training/serving EXIF skew from day one (bans `cv2.imread`/`cv2.imdecode`) and pre-computes `WEB-05` audit funnel + Mendeley "45 days" reconciliation | ✓ |
| Keep EXIF parsing inline inside `build_manifest.py` and re-implement in Phase 2 | Duplicates Xiaomi `39321` / `Orientation` logic and risks `cv2.imdecode` orientation desync | |

**Selection (`--auto`):** Shared `backend/app/pdi/io_exif.py` in Phase 1 + git-tracked `models/dataset_audit_summary.json`
**Notes:** Directly satisfies `ARCHITECTURE.md` single-source contract, `PITFALLS.md` #7b, and the `ingest-docs` resolution for `WEB-05` Mendeley `"45 days / 1,772 images"` vs. disk audit disclosure.

---

## the agent's Discretion

- Internal submodule structure of `backend/app/ingestion/` (`decoding.py`, `manifest.py`, `splits.py`, `audit_summary.py`)
- CLI options and progress logging format of `backend/scripts/build_manifest.py`
- `ruff` rule set configuration in `pyproject.toml`
- Synthetic fixture design in `backend/tests/`

## Deferred Ideas

None — discussion stayed strictly within Phase 1 scope.
