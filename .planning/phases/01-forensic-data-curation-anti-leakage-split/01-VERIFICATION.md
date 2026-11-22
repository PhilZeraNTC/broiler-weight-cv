---
phase: 01-forensic-data-curation-anti-leakage-split
verified: 2026-11-22T12:46:00Z
status: passed
score: "11/11"
covered_files:
  - ".planning/REQUIREMENTS.md"
  - ".planning/phases/01-forensic-data-curation-anti-leakage-split/01-01-PLAN.md"
  - ".planning/phases/01-forensic-data-curation-anti-leakage-split/01-01-SUMMARY.md"
  - ".planning/phases/01-forensic-data-curation-anti-leakage-split/01-02-PLAN.md"
  - ".planning/phases/01-forensic-data-curation-anti-leakage-split/01-02-SUMMARY.md"
  - ".planning/phases/01-forensic-data-curation-anti-leakage-split/01-03-PLAN.md"
  - ".planning/phases/01-forensic-data-curation-anti-leakage-split/01-03-SUMMARY.md"
  - ".python-version"
  - "backend/app/config.py"
  - "backend/app/ingestion/__init__.py"
  - "backend/app/ingestion/audit_summary.py"
  - "backend/app/ingestion/decoding.py"
  - "backend/app/ingestion/manifest.py"
  - "backend/app/ingestion/splits.py"
  - "backend/app/pdi/io_exif.py"
  - "backend/scripts/build_manifest.py"
  - "backend/tests/conftest.py"
  - "backend/tests/test_decoding.py"
  - "backend/tests/test_io_exif.py"
  - "backend/tests/test_manifest_integration.py"
  - "backend/tests/test_splits.py"
  - "models/dataset_audit_summary.json"
  - "pyproject.toml"
  - "uv.lock"
covered_digest: "v1:sha256:aaec244ee4e938f1dd6e063d0890740066faa97d5ccd4404e6f3eace0f89b80a"
behavior_unverified: 0
overrides_applied: 0
overrides: []
gaps: []
---

# Phase 1: Forensic Data Curation & Anti-Leakage Split — Verification Report

**Phase Goal:** Produce a reproducible Python 3.12 workspace and an immutable, Audit §9-exact forensic dataset manifest (`1,738` clean / `15` quarantined images) partitioned into leak-free 5-fold `session_id` splits.
**Verified:** 2026-11-22T12:46:00Z
**Status:** `passed`
**Re-verification:** No — initial adversarial verification

## Executive Summary

Adversarial goal-backward verification against the actual workspace, materialized Parquet manifests, git-tracked audit summary JSON, dependency lockfile, git index, and live test/lint/CLI execution confirms **11/11 observable truths VERIFIED**, **0 gaps**, **0 stub artifacts**, **0 debt markers**, and **39/39 automated tests passing** (`uv run pytest -v`, `2.51s`, plus `uv run ruff check backend/` clean). Independent re-execution of `backend.scripts.build_manifest` against read-only `data/raw/Broiler weights of Cobb Avian/` into an isolated temporary directory reproduced the exact SHA-256 digest of `models/dataset_audit_summary.json` (`b795f969b31c6db90a13a2b4e8626ac6101ab8bdbf58efc2c351a20d28f9363e`) and frame-identical clean/quarantine Parquets (`pd.testing.assert_frame_equal`).

---

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria + Merged PLAN Must-Haves)

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | **ROADMAP SC-1 / 01-03 T1:** Running the deterministic ingestor over `data/raw/` produces `data/processed/manifest_clean.parquet` (`1,738` images, `160` unique weights, `188` sessions, `24` cols), `data/processed/manifest_quarantine.parquet` (`15` images: `13` multi-bird across `417`/`757`/`823`/`1520` + `2` severe mislabels `1351`/`1465`, `fold = -1`), and git-tracked `models/dataset_audit_summary.json`, byte-idempotently. | ✓ VERIFIED | Direct Parquet inspection confirmed `clean=(1738, 24)` (`160` unique `weight_g`, `188` unique `session_id`, `162` unique `folder_chain`), `quarantine=(15, 24)` (`fold == -1`). Fresh CLI re-run into temp dir produced frame-identical Parquets and matching SHA-256 `b795f969b31c6db90a13a2b4e8626ac6101ab8bdbf58efc2c351a20d28f9363e`. Covered by `test_real_parquets_exact_totals_24_column_schema_and_path_hygiene` (PASS). |
| 2 | **ROADMAP SC-2 / 01-02 T1 / 01-03 T2:** Every Audit §9 anomaly is deterministically resolved in `manifest_clean.parquet`: level-2 nested folders (`171/173` [8], `190/191` [9], `190/193` [22]), `1340_` → `1340 g` (6 rows), `367` → `267 g` (6 rows), CP866 mojibake (`547/` [11], `562/` [10] → `21` recovered, `0` mtime fallback, `0` year-2024 captures), Eastern-Arabic (`٠-٩`) + `mtime - 14401 s` (`97` renamed) yielding `9` dates (`2022-03-24..2022-04-15`), and EXIF `Orientation` (`1,435` rotated), `FocalLength` (`1,753/1,753`), and Xiaomi tag `39321` `zoomMultiple` (`1,753/1,753`, `67` variable zoom) via shared Pillow loader. | ✓ VERIFIED | Independent Parquet audit verified `year counts={2022: 1753}`, `year 2024=0`, `focal_length_mm` non-null `1753/1753` (`[1.65, 4.25]`), `zoom_multiple` non-null `1753/1753`, `exif_orientation != 1` count `1435` (`81.86%`), and exact counts matching Audit §1, §2, §9, §10. Covered by `test_audit_section_9_anomalies_and_band_totals_verified_in_real_manifests` and `test_scan_dataset_records_synthetic_tree_enforces_audit_section_9` (PASS). |
| 3 | **ROADMAP SC-3 / 01-02 T8 / 01-03 T3:** Every clean image carries a reproducible `fold` (`0..4`, `int8`) assigned via 5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly by `session_id` (`<= 600.0 s` gap within `folder_chain`) and stratified by weight band (`<500g`, `500-1000g`, `>1000g`), with `0` cross-fold `session_id` overlap, `0` sklearn warnings, balanced `>1000g` sessions (`>= 4`/fold), and permutation/repeat determinism. | ✓ VERIFIED | Independent audit verified all 10 pairwise fold intersections equal `0`, session-gap invariant `<= 600.0 s` intra-session / `> 600.0 s` inter-session across all `1,753` rows, and heavy (`>1000g`) sessions per fold `[4, 5, 5, 5, 5]`. Covered by `test_real_five_fold_splits_zero_leakage_and_permutation_determinism`, `test_audit_section_10_shaped_dataset_produces_balanced_zero_leakage_splits`, and `test_assign_session_folds_is_permutation_and_repeat_deterministic` (PASS). |
| 4 | **ROADMAP SC-4 / 01-01 T1 / 01-03 T5:** Backend resolves exclusively from binary wheels under `uv`-pinned CPython `3.12.14` (`uv.lock`), declares all 17 locked `D-02` packages with zero DL/OCR/GUI-OpenCV packages, and keeps `data/raw/` and `data/processed/` completely untracked by Git while tracking `models/dataset_audit_summary.json`. | ✓ VERIFIED | `uv sync --frozen --no-build --managed-python --check` returned `Checked 49 packages ... Would make no changes`; `uv run python --version` returned `Python 3.12.14`; `uv.lock` audit confirmed `50/50` external packages have wheels (`0` sdist-only) and `0` banned packages; `git ls-files data/raw data/processed` empty; `git ls-files models/` returned `models/dataset_audit_summary.json`; `git check-ignore models/dataset_audit_summary.json` exited `1`. Covered by `test_pyproject_and_uv_lock_comply_with_d02_and_prohibitions` and `test_git_hygiene_excludes_raw_and_processed_while_tracking_audit_summary` (PASS). |
| 5 | **01-01 T2 & T3 (EXIF Loader Parity & CLI Pipeline):** `read_exif_metadata(source)` and `load_oriented_rgb(source)` return identical `ExifMetadata` and bit-identical transposed RGB `uint8` arrays for `Path` and `bytes` across orientations `1, 3, 5, 6, 7, 8` (extracting EXIF before `ImageOps.exif_transpose()`), and `build_manifest` CLI executes end-to-end with zero side effects under `--dry-run`. | ✓ VERIFIED | Inspected `backend/app/pdi/io_exif.py` lines 106–156 confirming `_extract_exif_from_open_image(im)` runs strictly before `ImageOps.exif_transpose(im).convert("RGB")`. Verified by `test_exif_metadata_and_oriented_rgb_path_bytes_parity` (6 parametrized orientations with red/green pixel probe checks), `test_orientation_6_extracts_exif_before_transpose`, and `test_build_manifest_cli_dry_run_and_full_run_on_synthetic_tree` (PASS). |
| 6 | **01-01 T4 (EXIF Security Bounds & Missing-EXIF Fallback):** Stripped-EXIF JPEGs and RGBA PNGs return `has_exif_optical=False` and 3-channel RGB without raising; malformed/non-dict/boolean/out-of-range/oversized (`> 16,384` bytes) Xiaomi tag `39321` payloads and zero-denominator `IFDRational` degrade safely to `None`; `PIL.Image.MAX_IMAGE_PIXELS` remains enabled. | ✓ VERIFIED | Inspected `_parse_focal_length` and `_parse_xiaomi_zoom` in `backend/app/pdi/io_exif.py` lines 54–104 confirming `MAX_XIAOMI_TAG_BYTES = 16_384`, strict UTF-8 decode, `json.loads` only, `dict` check, `bool`/`NaN`/`Inf` rejection, `[0.1, 20.0]` clamp, `ZeroDivisionError` guard, and untouched `Image.MAX_IMAGE_PIXELS`. Verified by 4 dedicated tests in `backend/tests/test_io_exif.py` (PASS). |
| 7 | **01-01 T5 (CWD-Independent Paths & Zero Raw OpenCV):** `backend/app/config.py` anchors all 8 canonical paths to `Path(__file__).resolve().parents[2]`, and zero raw OpenCV image loading calls (`cv2.imread`, `cv2.imdecode`) exist under `backend/app/` or `backend/scripts/`. | ✓ VERIFIED | Inspected `backend/app/config.py` lines 12–19; independent AST scan across all 13 Python files in `backend/` confirmed `0` OpenCV image loading calls or imports. Verified by `test_config_paths_are_repo_root_anchored_and_cwd_independent` and `test_ast_scan_confirms_zero_raw_opencv_image_loading_under_backend_app` (PASS). |
| 8 | **01-02 T2 (Immutable 24-Column Parquet Schema & Containment):** Clean and quarantine manifests enforce the exact 24-column PyArrow schema (`15` base + `9` derived), POSIX forward-slash paths, zero `..` segments, symlink escape rejection (`followlinks=False` + `is_relative_to`), and Pandas 3.1 CoW-safe mutations. | ✓ VERIFIED | Inspected `_MANIFEST_ARROW_SCHEMA`, `enforce_manifest_dtypes`, `_is_contained_non_symlink`, and `write_manifest_parquet` in `backend/app/ingestion/manifest.py`. Verified by `test_manifest_24_column_parquet_schema_and_precision_roundtrip`, `test_symlink_file_and_directory_escape_rejection`, and `test_real_parquets_exact_totals_24_column_schema_and_path_hygiene` (PASS). |
| 9 | **01-02 T3–T7 (DATA-02 Edge Contracts — Boundary, Adjacency, Empty/Single, Ordering, Precision):** Weight bands strictly partition `<500g` (`499`), `500-1000g` (`500`, `1000`), `>1000g` (`1001`); session clustering splits strictly at `> 600.0 s` (`600.0 s` same session, `600.001 s`/`601.0 s` new session); `_1` bursts (`0.0 s`) stay in one session/fold; empty dir returns valid 0-row typed 24-col DataFrame, single-image folder gets `__s0`, `< 5` sessions/empty raises `ValueError`; 3-key `mergesort` guarantees invariant ordering; exact `int32`/`float64`/`datetime64[us]`/`int8` precision holds. | ✓ VERIFIED | Verified across `test_boundary_weight_bands_lens_and_image_id`, `test_session_gap_boundary_exact_600s_and_burst_adjacency`, `test_empty_and_single_image_dataset_contracts`, `test_manifest_24_column_parquet_schema_and_precision_roundtrip`, `test_assign_session_folds_is_permutation_and_repeat_deterministic`, and `test_empty_undersized_and_warning_inputs_fail_loudly` (all PASS). |
| 10 | **01-03 T4 (Zero-Path-Leakage Audit Summary JSON for WEB-05):** Git-tracked `models/dataset_audit_summary.json` exposes all 6 canonical blocks with strictly aggregate counts (zero `.jpg`, `rel_path`, `/home/`, `data/raw`, or wall-clock timestamps), reconciling Mendeley (`1,772` / `"daily for 45 days"`) vs. disk (`1,753` / `9` dates / `23` calendar days / `~7-29 days` / `116–2,093 g`), quantifying the `801–1,216 g` sparse gap (`[1009]`, `6` imgs, `1` session) and `8`-day hiatus (`2022-04-03..2022-04-11`), and publishing per-fold stats. | ✓ VERIFIED | Direct read of `models/dataset_audit_summary.json` (315 lines) and independent raw-text scan confirmed `0` forbidden markers (`".jpg"`, `".jpeg"`, `"rel_path"`, `"/home/"`, `"data/raw"`, username) and exact match to Audit §1/§2/§6/§9/§10. Verified by `test_audit_summary_json_contract_zero_path_leakage_and_determinism` (PASS). |
| 11 | **01-01 T6 / 01-03 T6 (Backstop Truth — DATA-01 Unclassified Anomaly Assumption):** Beyond the 6 audited quarantine folders (`417`, `757`, `823`, `1520`, `1351`, `1465`) and `367 → 267 g` LCD correction from Audit §9, no additional unlabeled mislabels are silently discarded in Phase 1 (full-corpus offline LCD OCR audit deferred to `HARD-01` in v1.x). | ✓ VERIFIED | Directly observed `len(clean_df) + len(quarantine_df) = 1738 + 15 = 1753 == file_count(data/raw)` (`0` files silently dropped), quarantine folders strictly equal `{"417","757","823","1520","1351","1465"}`, and `was_lcd_corrected.sum() == 6` strictly in `"367"`, enforced by `test_real_parquets_exact_totals_24_column_schema_and_path_hygiene` and `test_audit_section_9_anomalies_and_band_totals_verified_in_real_manifests` (PASS, exit 0); `HARD-01` explicitly tracked in `.planning/REQUIREMENTS.md` line 43. |

**Score:** `11/11` truths verified (`0` present, behavior-unverified; `0` failed; `0` overrides)

---

## Artifact Verification (Levels 1–4)

Automated `gsd-tools.cjs query verify.artifacts` returned `all_passed: true` on all 3 PLANs (`8/8` on `01-01`, `6/6` on `01-02`, `4/4` on `01-03`), reinforced by direct file read and runtime import/data-flow checks:

| Artifact | Plan(s) | Exists | Substantive | Wired | Data Flows | Status | Details |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `.python-version` | 01-01 | ✓ | ✓ (1 line: `3.12`) | ✓ (`uv`) | N/A | ✓ VERIFIED | Resolves managed CPython `3.12.14` |
| `pyproject.toml` | 01-01 | ✓ | ✓ (49 lines) | ✓ (`uv`, `pytest`, `ruff`) | N/A | ✓ VERIFIED | 14 runtime + 3 dev D-02 deps, `package=false`, `no-build=true`, `python-preference="only-managed"` |
| `uv.lock` | 01-01 | ✓ | ✓ (51 pkgs: 50 external + virtual root) | ✓ (`uv sync --frozen`) | N/A | ✓ VERIFIED | 50/50 external pkgs wheel-backed (`0` sdist-only), `0` banned pkgs |
| `backend/app/config.py` | 01-01 | ✓ | ✓ (30 lines, 8 exports) | ✓ (imported by `build_manifest.py` & tests) | ✓ | ✓ VERIFIED | Anchored via `Path(__file__).resolve().parents[2]` |
| `backend/app/pdi/io_exif.py` | 01-01 | ✓ | ✓ (156 lines, 4 public exports + parsers) | ✓ (imported by `manifest.py` & `test_io_exif.py`) | ✓ | ✓ VERIFIED | Populates `1,753/1,753` real EXIF records; `Path \| bytes` parity verified |
| `backend/app/ingestion/decoding.py` | 01-02 | ✓ | ✓ (122 lines, 14 exports) | ✓ (imported by `manifest.py`, `splits.py`, `audit_summary.py`, `build_manifest.py`, `__init__.py`) | ✓ | ✓ VERIFIED | Exact Audit §9 constants & CP866/Arabic-Indic decoder |
| `backend/app/ingestion/manifest.py` | 01-02 | ✓ | ✓ (368 lines, 7 exports) | ✓ (imported by `build_manifest.py`, `audit_summary.py`, `__init__.py`, tests) | ✓ | ✓ VERIFIED | Explicit 24-col `_MANIFEST_ARROW_SCHEMA`, CoW-safe, symlink containment |
| `backend/app/ingestion/splits.py` | 01-02 | ✓ | ✓ (211 lines, 6 exports) | ✓ (imported by `build_manifest.py`, `audit_summary.py`, `__init__.py`, tests) | ✓ | ✓ VERIFIED | `StratifiedGroupKFold(5, shuffle=True, random_state=42)`, warning escalation, overlap audit |
| `backend/app/ingestion/__init__.py` | 01-02 | ✓ | ✓ (69 lines, 28 `__all__` exports) | ✓ (imported & tested in `test_splits.py`) | ✓ | ✓ VERIFIED | Clean package barrel for downstream Phases 2–4 |
| `backend/app/ingestion/audit_summary.py` | 01-03 | ✓ | ✓ (337 lines, 7 exports) | ✓ (imported by `build_manifest.py` & `test_manifest_integration.py`) | ✓ | ✓ VERIFIED | Deterministic 6-block JSON + runtime PROIBIÇÃO 4 leak guard |
| `backend/scripts/build_manifest.py` | 01-01, 01-03 | ✓ | ✓ (166 lines, `parse_args`, `run_build_manifest`, `main`) | ✓ (CLI entrypoint & tested in `test_io_exif.py` + `test_manifest_integration.py`) | ✓ | ✓ VERIFIED | Orchestrates full curation → split → Parquet + JSON; zero dry-run side effects |
| `models/dataset_audit_summary.json` | 01-03 | ✓ | ✓ (315 lines, 7.1 KiB, valid JSON) | ✓ (git-tracked, tested by `test_manifest_integration.py`, consumed by Phase 4 `WEB-05`) | ✓ | ✓ VERIFIED | Byte-idempotent (`b795f969...`), `0` forbidden path/filename markers |
| `data/processed/manifest_clean.parquet` *(git-ignored runtime)* | 01-03 | ✓ | ✓ (81 KiB, 1,738 × 24 Arrow cols) | ✓ (read by `test_manifest_integration.py`, feeds Phase 2–3) | ✓ | ✓ VERIFIED | Exact totals, `fold` in `{0..4}`, `0` session overlap |
| `data/processed/manifest_quarantine.parquet` *(git-ignored runtime)* | 01-03 | ✓ | ✓ (15 KiB, 15 × 24 Arrow cols) | ✓ (read by `test_manifest_integration.py`, feeds Phase 2 Watershed) | ✓ | ✓ VERIFIED | `13` multi-bird + `2` severe mislabels, `fold == -1` |
| `backend/tests/conftest.py` | 01-01 | ✓ | ✓ (169 lines) | ✓ (fixtures used across 4 test modules) | ✓ | ✓ VERIFIED | Zero `data/raw/` coupling; exercises all Audit §9 anomalies + 3 weight bands |
| `backend/tests/test_io_exif.py` | 01-01 | ✓ | ✓ (372 lines, 17 tests) | ✓ (`pytest`) | ✓ | ✓ VERIFIED | 17/17 PASS |
| `backend/tests/test_decoding.py` | 01-02 | ✓ | ✓ (408 lines, 9 tests) | ✓ (`pytest`) | ✓ | ✓ VERIFIED | 9/9 PASS |
| `backend/tests/test_splits.py` | 01-02 | ✓ | ✓ (310 lines, 7 tests) | ✓ (`pytest`) | ✓ | ✓ VERIFIED | 7/7 PASS |
| `backend/tests/test_manifest_integration.py` | 01-03 | ✓ | ✓ (419 lines, 6 tests) | ✓ (`pytest`) | ✓ | ✓ VERIFIED | 6/6 PASS |

---

## Key Link Verification

| Plan | From | To | Via | Status | Details |
| --- | --- | --- | --- | --- | --- |
| 01-01 | `backend/scripts/build_manifest.py` | `backend/app/pdi/io_exif.py` | Header-only EXIF extraction during dataset crawl | ✓ WIRED *(transitive in 01-03)* | Planned architecture evolution: `01-01-PLAN.md` Task 1 line 160 & `01-03-PLAN.md` Task 1 line 136 intentionally replaced the Plan `01-01` direct tracer import with delegation `build_manifest.py` → `manifest.py` (`scan_dataset_records`) → `io_exif.py` (`read_exif_metadata`, line 33). `gsd-tools verify.key-links` flagged the obsolete direct regex on `01-01-PLAN.md`, while the live transitive chain is 100% wired and populates `1,753/1,753` EXIF rows. |
| 01-01 | `backend/scripts/build_manifest.py` | `backend/app/config.py` | `from backend.app.config import ...` | ✓ WIRED | Verified by `gsd-tools verify.key-links` and line 21 of `build_manifest.py`. |
| 01-01 | `backend/tests/test_io_exif.py` | `backend/app/pdi/io_exif.py` | `from backend.app.pdi.io_exif import ...` | ✓ WIRED | Verified by `gsd-tools verify.key-links` and lines 19–26 of `test_io_exif.py`. |
| 01-02 | `backend/app/ingestion/manifest.py` | `backend/app/pdi/io_exif.py` | `from backend.app.pdi.io_exif import read_exif_metadata` | ✓ WIRED | Verified by `gsd-tools verify.key-links` (`manifest.py` line 33, called at line 289). |
| 01-02 | `backend/app/ingestion/manifest.py` | `backend/app/ingestion/decoding.py` | `from backend.app.ingestion.decoding import ...` | ✓ WIRED | Verified by `gsd-tools verify.key-links` (`manifest.py` lines 21–32). |
| 01-02 | `backend/app/ingestion/splits.py` | `sklearn.model_selection.StratifiedGroupKFold` | `StratifiedGroupKFold` with warning capture & overlap audit | ✓ WIRED | Verified by `gsd-tools verify.key-links` (`splits.py` lines 18, 181–194). |
| 01-03 | `backend/scripts/build_manifest.py` | `backend/app/ingestion/manifest.py` | `scan_dataset_records`, `split_clean_and_quarantine`, `write_manifest_parquet` | ✓ WIRED | Verified by `gsd-tools verify.key-links` (`build_manifest.py` lines 24–28, 109–110, 129–130). |
| 01-03 | `backend/scripts/build_manifest.py` | `backend/app/ingestion/splits.py` | `assign_session_folds`, `verify_fold_integrity` | ✓ WIRED | Verified by `gsd-tools verify.key-links` (`build_manifest.py` lines 29–33, 117–124). |
| 01-03 | `backend/scripts/build_manifest.py` | `backend/app/ingestion/audit_summary.py` | `build_audit_summary`, `write_audit_summary` | ✓ WIRED | Verified by `gsd-tools verify.key-links` (`build_manifest.py` lines 22, 126, 131). |
| 01-03 | `backend/tests/test_manifest_integration.py` | `models/dataset_audit_summary.json` | `AUDIT_SUMMARY_PATH` contract & zero-leakage assertions | ✓ WIRED | Verified by `gsd-tools verify.key-links` (`test_manifest_integration.py` lines 26, 245–327). |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable / Output | Upstream Source | Produces Real Data | Status |
| --- | --- | --- | --- | --- |
| `data/processed/manifest_clean.parquet` | 1,738 rows × 24 Arrow cols | `data/raw/Broiler weights of Cobb Avian/` via `scan_dataset_records` → `read_exif_metadata` → `split_clean_and_quarantine` → `assign_session_folds` | Yes (`160` weights, `188` sessions, `1,738/1,738` real EXIF tags, `fold` in `0..4`) | ✓ FLOWING |
| `data/processed/manifest_quarantine.parquet` | 15 rows × 24 Arrow cols | Same read-only crawl partitioned on `MULTI_BIRD_FOLDERS` & `SEVERE_MISLABEL_FOLDERS` | Yes (`13` multi-bird + `2` severe mislabels, `fold = -1`) | ✓ FLOWING |
| `models/dataset_audit_summary.json` | 6 canonical `WEB-05` JSON blocks | `build_audit_summary(clean_df, quarantine_df)` over real Parquets | Yes (exact dict equality with fresh `build_audit_summary` & matching SHA-256 on fresh CLI rebuild) | ✓ FLOWING |

---

## Requirements & Locked Decisions Coverage

### Requirements Coverage (`REQUIREMENTS.md` Phase 1 Mapping)

| Requirement | Source Plans | Description | Status | Evidence |
| --- | --- | --- | --- | --- |
| `DATA-01` | `01-01`, `01-02`, `01-03` | Deterministic Python 3.12 (`uv`) Audit §9 ingestor over `data/raw/Broiler weights of Cobb Avian/` persisting `manifest_clean.parquet` (`1,738` imgs, `160` weights, `188` sessions), `manifest_quarantine.parquet` (`15` imgs), and `models/dataset_audit_summary.json`. | ✓ SATISFIED | All 3 plans executed; real Parquets & JSON verified against Audit §1/§2/§9/§10; 39/39 tests pass. |
| `DATA-02` | `01-02`, `01-03` | Reproducible leak-free 5-fold `StratifiedGroupKFold` grouped strictly by `session_id` (`<= 10 min` sequential captures) and stratified by weight band (`<500g`, `500-1000g`, `>1000g`) with zero session overlap and balanced `>1000g` representation. | ✓ SATISFIED | Verified in `manifest_clean.parquet` (`0` overlap, `0` warnings, `[4, 5, 5, 5, 5]` heavy sessions/fold, permutation-invariant) and unit + integration tests. |

*Orphaned Phase 1 requirements in `REQUIREMENTS.md`:* **0** (all mapped requirements accounted for).

### Locked Decisions Coverage (`01-CONTEXT.md`)

| Decision | Status | Direct Evidence |
| --- | --- | --- |
| `D-01` (Root `.python-version`/`pyproject.toml`/`uv.lock`, CPython `3.12.14`, `backend.app.*`, `pythonpath=["."]`) | ✓ VERIFIED | Root files present; `uv run python --version` = `Python 3.12.14`; `pyproject.toml` sets `pythonpath = ["."]`. |
| `D-02` (Exact 17 locked deps, binary wheels only, zero DL/OCR/GUI-OpenCV) | ✓ VERIFIED | `pyproject.toml` matches all 14 runtime + 3 dev deps exactly; `no-build = true`; `50/50` wheel-backed external deps in `uv.lock`; `0` banned packages. |
| `D-03` (24-column Parquet contract: 15 Audit §9 base + 9 derived, POSIX paths, Pandas 3 CoW-safe) | ✓ VERIFIED | `MANIFEST_COLUMNS` & `_MANIFEST_ARROW_SCHEMA` enforce all 24 exact columns/types; `0` backslashes or `..` segments; `.copy()` + `.loc` used throughout. |
| `D-04` (Audit §9 curation rules verbatim, read-only `data/raw/`) | ✓ VERIFIED | `14401`, `600.0`, `{"417","757","823","1520"}`, `{"1351","1465"}`, `{"367": 267}`, CP866/Arabic-Indic, nested level-2 traversal, `1340_` sanitization verified on real data; `data/raw/` latest mtime `2024-03-25T17:32:56`. |
| `D-05` (3-key deterministic sort + `StratifiedGroupKFold(5, shuffle=True, random_state=42)` + `fold=-1` quarantine) | ✓ VERIFIED | `SORT_KEYS = ["folder_chain", "capture_dt", "rel_path"]` (`mergesort`), exact SGKF parameters, `int8` `fold` (`0..4` clean, `-1` quarantine). |
| `D-06` (Automated pytest exact-count, Audit §9 anomaly, zero-overlap, zero-warning, `>=4` heavy/fold, determinism gate) | ✓ VERIFIED | `backend/tests/test_manifest_integration.py` + `test_decoding.py` + `test_splits.py` enforce every D-06 gate and pass `39/39`. |
| `D-07` (Single shared `backend/app/pdi/io_exif.py` accepting `Path \| bytes`, EXIF before `exif_transpose()`, repo-wide OpenCV loader ban) | ✓ VERIFIED | Implemented in `io_exif.py`; AST scan across all `backend/` confirms `0` `cv2.imread`/`cv2.imdecode`; 17/17 tests pass in `test_io_exif.py`. |
| `D-08` (Git-tracked zero-path-leakage `models/dataset_audit_summary.json` for Phase 4 `WEB-05`) | ✓ VERIFIED | Tracked by Git, `0` raw/host path leaks, reconciles Mendeley vs. disk, documents `801–1,216 g` sparse gap & `8`-day hiatus, exposes 5-fold balance. |

---

## Prohibitions & STRIDE Threat Model Verification

### Prohibitions (`PROIBIÇÃO 1–4` & PLAN `must_haves.prohibitions`)

| Prohibition | Automated Enforcement & Direct Evidence | Verdict |
| --- | --- | --- |
| **PROIBIÇÃO 1 (Zero Deep Learning / OCR packages)** | `pyproject.toml` and `uv.lock` contain `0` DL/OCR packages (`torch`, `tensorflow`, `keras`, `ultralytics`, `timm`, `onnxruntime`, `paddleocr`, `easyocr`, `sam2`, `segment-anything`). Enforced by `test_pyproject_and_uv_lock_comply_with_d02_and_prohibitions` + independent `uv.lock` scan. | ✓ VERIFIED (0 violations) |
| **PROIBIÇÃO 2 (Zero GUI OpenCV, zero raw `cv2.imread`/`cv2.imdecode`, zero CSV manifests)** | `opencv-python-headless` locked (`opencv-python` / `opencv-contrib-python` absent). AST scan over all 13 Python files in `backend/` found `0` `cv2.imread`/`cv2.imdecode` calls or `cv2` imports and `0` `.to_csv(` calls. Enforced by `test_ast_scan_confirms_zero_raw_opencv_image_loading_under_backend_app`. | ✓ VERIFIED (0 violations) |
| **PROIBIÇÃO 3 (Zero random `train_test_split` or ungrouped `KFold`)** | AST scan over `backend/app/` and `backend/scripts/` confirmed `0` imports/calls to `train_test_split`, `KFold`, `StratifiedKFold`, `ShuffleSplit`, `StratifiedShuffleSplit`, `GroupShuffleSplit`; only `StratifiedGroupKFold` grouped by `session_id` is used. Enforced by `test_ast_scan_prohibits_random_and_ungrouped_splitters_in_ingestion`. | ✓ VERIFIED (0 violations) |
| **PROIBIÇÃO 4 (Read-only `data/raw/`, zero raw/processed files in Git, zero raw filename/path dumps in stdout or JSON)** | `data/raw/` untouched (`1,753` files, latest mtime `2024-03-25T17:32:56`); `git ls-files data/raw data/processed` empty; `models/dataset_audit_summary.json` raw text has `0` occurrences of `.jpg`, `.jpeg`, `rel_path`, `/home/`, `data/raw`, or host username; CLI stdout outputs strictly aggregate counts (`0` `.jpg`/host paths). Enforced by `write_audit_summary()` runtime guard + `test_audit_summary_json_contract_zero_path_leakage_and_determinism` + `test_git_hygiene_excludes_raw_and_processed_while_tracking_audit_summary` + `test_build_manifest_cli_dry_run_and_full_run_on_synthetic_tree`. | ✓ VERIFIED (0 violations) |

*Non-authoritative prohibition review advisory:* The PLAN frontmatter prohibition entries carried initial `status: unverified` / `flagged: true` tags prior to verification; all four prohibitions are backed above by deterministic automated tests and independent codebase/lockfile/git/JSON audits (0 violations observed; optional human sign-off may be noted during milestone audit if desired).

### ASVS L1 / STRIDE Mitigations (`T-01-SC`, `T-01-01`..`T-01-08`)

| Threat ID | Mitigation Verified | Evidence |
| --- | --- | --- |
| `T-01-SC` (Supply Chain) | ✓ VERIFIED | `[tool.uv] no-build = true`, hash-locked `uv.lock` with `50/50` wheel-backed external deps, strict 17-package `D-02` manifest. |
| `T-01-01` (Malformed/Oversized EXIF & Xiaomi Tag `39321`) | ✓ VERIFIED | `MAX_XIAOMI_TAG_BYTES = 16_384`, strict UTF-8 decode, `json.loads` only, `bool`/`NaN`/`Inf` rejection, `[0.1, 20.0]` zoom clamp, `IFDRational(0, 0)` guard (`test_io_exif.py`). |
| `T-01-02` (Decompression Bomb OOM) | ✓ VERIFIED | `PIL.Image.MAX_IMAGE_PIXELS` remains non-`None` and positive (`test_decompression_bomb_guard_remains_enabled`). |
| `T-01-03` & `T-01-04` (Path Escape / Symlink Traversal) | ✓ VERIFIED | `os.walk(..., followlinks=False)`, `_is_contained_non_symlink()` checking `is_symlink()` and `resolve().is_relative_to(root)`, POSIX path check (`test_symlink_file_and_directory_escape_rejection`). |
| `T-01-05` (Regex ReDoS) | ✓ ACCEPTED / VERIFIED | Anchored linear-time regexes without nested quantifiers in `decoding.py` lines 32–34. |
| `T-01-06` (Silent Split Leakage) | ✓ VERIFIED | 3-key `mergesort`, sklearn warning capture escalated to `RuntimeError`, pairwise overlap audit (`test_splits.py`, `test_manifest_integration.py`). |
| `T-01-07` & `T-01-08` (JSON/Stdout Disclosure & Git Hygiene) | ✓ VERIFIED | Runtime leak guard in `write_audit_summary()`, aggregate-only stdout, `git ls-files` + `git check-ignore` integration tests (`test_manifest_integration.py`). |

---

## Anti-Patterns & Commit Hygiene

- **Debt Markers (`TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, `PLACEHOLDER`):** `0` found across all 13 Python files under `backend/`.
- **Stub / Placeholder Implementations:** `0` found.
- **Linter (`uv run ruff check backend/`):** `All checks passed!` (exit `0`).
- **Atomic Commits (`gsd-tools query verify.commits`):** All 6 task commits (`5c2210e`, `de88c33`, `96a0f7f`, `2b51a2b`, `88b0d84`, `f436d1d`) verified valid in git history (`6/6`).

---

## Commands Run & Observed Output

| Check | Command | Exit | Key Observed Output |
| --- | --- | --- | --- |
| GSD Artifact Verification | `gsd-tools.cjs query verify.artifacts 01-{01,02,03}-PLAN.md` | `0` | `8/8`, `6/6`, `4/4` (`all_passed: true` on all 3 plans) |
| GSD Key Link Verification | `gsd-tools.cjs query verify.key-links 01-{01,02,03}-PLAN.md` | `0` | `01-02`: `3/3`, `01-03`: `4/4`; `01-01`: `2/3` (direct tracer import cleanly superseded by transitive `build_manifest.py → manifest.py → io_exif.py` in `01-03`) |
| GSD Commit Verification | `gsd-tools.cjs query verify.commits 5c2210e de88c33 96a0f7f 2b51a2b 88b0d84 f436d1d` | `0` | `all_valid: true` (`6/6`) |
| Runtime & Lockfile Check | `uv sync --frozen --no-build --managed-python --check && uv run python --version` | `0` | `Checked 49 packages in 0.50ms`, `Would make no changes`, `Python 3.12.14` |
| Full Test Suite | `uv run pytest -v` | `0` | `39 passed in 2.51s` (`test_decoding.py` 9, `test_io_exif.py` 17, `test_manifest_integration.py` 6, `test_splits.py` 7) |
| Static Linter | `uv run ruff check backend/` | `0` | `All checks passed!` |
| Independent Parquet / CLI Idempotency / AST / Wheel / Raw-Immutability Audit | Read-only Python verification script + isolated temp-dir CLI run | `0` | `clean=(1738, 24)` (`160` weights, `188` sessions), `quar=(15, 24)`, `1753/1753` EXIF focal/zoom, `0` year-2024, heavy sessions `[4, 5, 5, 5, 5]`, matching SHA-256 `b795f969...`, frame-identical Parquets, `0` AST/debt violations across 13 files, `50/50` external wheels, `data/raw/` latest mtime `2024-03-25T17:32:56` |
| Git Hygiene | `git ls-files data/raw data/processed`, `git ls-files models/`, `git check-ignore models/dataset_audit_summary.json` | `0`/`0`/`1` | `data/raw` & `data/processed` untracked (`0` files); `models/dataset_audit_summary.json` tracked & not ignored |

---

## Gaps Summary

None — all 4 ROADMAP Success Criteria, both Phase 1 requirements (`DATA-01`, `DATA-02`), all 8 locked decisions (`D-01`..`D-08`), all 11 merged observable truths, all STRIDE mitigations, and all 39 automated tests passed with concrete direct evidence.

---

_Verified: 2026-11-22T12:46:00Z_
_Verifier: gsd-verifier_
