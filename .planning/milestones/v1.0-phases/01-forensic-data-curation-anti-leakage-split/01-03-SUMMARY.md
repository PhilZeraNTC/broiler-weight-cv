---
phase: 01-forensic-data-curation-anti-leakage-split
plan: 03
subsystem: ingestion
tags: [audit-summary, cli, parquet, pyarrow, stratified-group-kfold, anti-leakage, idempotency, git-hygiene]

# Dependency graph
requires:
  - phase: 01-forensic-data-curation-anti-leakage-split
    provides: "Audit §9 forensic decoder (decoding.py), path-contained 24-column PyArrow Parquet manifest engine (manifest.py), deterministic 5-fold StratifiedGroupKFold session splitter (splits.py), and shared Pillow EXIF-first loader (io_exif.py)"
provides:
  - Deterministic, zero-path-leakage aggregate JSON summary builder ready for Phase 4 WEB-05 Forensic Audit Dashboard (backend/app/ingestion/audit_summary.py)
  - Full idempotent production CLI orchestrating Audit §9 ingestion, clean/quarantine splitting, 5-fold StratifiedGroupKFold verification, typed Parquet persistence, and aggregate JSON export (backend/scripts/build_manifest.py)
  - Git-tracked byte-idempotent aggregate dataset audit artifact reconciling Mendeley claims vs. disk audit, curation funnel, encoding/EXIF recovery, 801-1216 g sparse coverage gap, 8-day hiatus, and 5-fold session balance (models/dataset_audit_summary.json)
  - Local git-ignored typed Parquet manifests materialized from read-only data/raw/Broiler weights of Cobb Avian/: data/processed/manifest_clean.parquet (1,738 rows x 24 cols, 160 unique weights, 188 sessions, fold 0..4) and data/processed/manifest_quarantine.parquet (15 rows x 24 cols, fold -1)
  - Full D-06, D-08, ASVS L1, and PROIBIÇÃO 1-4 read-only integration verification test suite (backend/tests/test_manifest_integration.py)
affects: [phase-2-classical-pdi, phase-3-modeling, phase-4-web-mvp]

actuals:
  tokens: 11939
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - Byte-idempotent aggregate JSON serialization (indent=2, sort_keys=True, ensure_ascii=False, trailing newline, zero wall-clock timestamps) with runtime PROIBIÇÃO 4 / T-01-07 leak guard rejecting .jpg, rel_path, /home/, and data/raw tokens
    - Context-aware CLI heavy-session enforcement (--min-heavy-sessions defaults to auto-enforcing >= 4 heavy >1000g sessions/fold on real dataset or >= 20 heavy sessions, while remaining safe for synthetic test trees and explicit --min-heavy-sessions 0 overrides)
    - Zero-side-effect --dry-run CLI execution verified against mtime/existence invariants on default outputs and non-existent target directories

key-files:
  created:
    - backend/app/ingestion/audit_summary.py
    - models/dataset_audit_summary.json
    - backend/tests/test_manifest_integration.py
  modified:
    - backend/scripts/build_manifest.py

key-decisions:
  - "Added a runtime defense-in-depth PROIBIÇÃO 4 guard directly inside write_audit_summary() rejecting any serialized payload containing '.jpg', 'rel_path', '/home/', or 'data/raw' before touching disk, reinforcing the T-01-07 integration tests."
  - "Made --min-heavy-sessions default to None with automatic enforcement of MIN_HEAVY_SESSIONS_PER_FOLD=4 whenever dataset_root resolves to RAW_DATASET_DIR or clean_df contains >= 20 heavy (>1000g) sessions, preserving full backward compatibility with Plan 01-01 synthetic CLI tests while enforcing strict heavy-bird balance on the real corpus."
  - "Omitted any wall-clock generation timestamp from models/dataset_audit_summary.json and sorted every list/dictionary key deterministically so consecutive CLI runs produce byte-identical SHA-256 digests (b795f969b31c6db90a13a2b4e8626ac6101ab8bdbf58efc2c351a20d28f9363e)."

patterns-established:
  - "Pattern 7 (Zero-Path-Leakage Git-Tracked Pitch Artifacts): Only aggregate JSON artifacts (models/dataset_audit_summary.json) enter version control; raw images and generated Parquets remain strictly git-ignored and verified via automated git ls-files / git check-ignore tests."
  - "Pattern 8 (Byte-Idempotent Offline Pipeline Runs): Every offline CLI run over read-only data/raw/ must yield byte-identical JSON digests and identical Parquet row hashes across consecutive runs and arbitrary input permutations."

requirements-completed:
  - DATA-01
  - DATA-02

coverage:
  - id: D1
    description: "Deterministic zero-path-leakage aggregate audit summary builder (backend/app/ingestion/audit_summary.py) and git-tracked models/dataset_audit_summary.json exposing all 6 canonical WEB-05 sections (identity, Mendeley vs. disk reconciliation, curation funnel, encoding/EXIF recovery, distributions & 801-1216 g / 8-day gaps, and 5-fold split audit)"
    requirement: DATA-01
    verification:
      - kind: integration
        ref: "backend/tests/test_manifest_integration.py#test_audit_summary_json_contract_zero_path_leakage_and_determinism"
        status: pass
    human_judgment: false
  - id: D2
    description: "Full idempotent production CLI (backend/scripts/build_manifest.py) materializing data/processed/manifest_clean.parquet (1,738 rows x 24 cols, 160 unique weights, 188 sessions) and data/processed/manifest_quarantine.parquet (15 rows x 24 cols, fold = -1) with zero raw filename leakage, zero dry-run side effects, and verified Audit §9 anomaly resolution"
    requirement: DATA-01
    verification:
      - kind: integration
        ref: "backend/tests/test_manifest_integration.py#test_real_parquets_exact_totals_24_column_schema_and_path_hygiene"
        status: pass
      - kind: integration
        ref: "backend/tests/test_manifest_integration.py#test_audit_section_9_anomalies_and_band_totals_verified_in_real_manifests"
        status: pass
      - kind: unit
        ref: "backend/tests/test_manifest_integration.py#test_build_manifest_cli_dry_run_and_full_run_on_synthetic_tree"
        status: pass
      - kind: integration
        ref: "backend/tests/test_manifest_integration.py#test_git_hygiene_excludes_raw_and_processed_while_tracking_audit_summary"
        status: pass
    human_judgment: false
  - id: D3
    description: "Verified leak-free 5-fold StratifiedGroupKFold assignments persisted in data/processed/manifest_clean.parquet guaranteeing zero cross-fold session_id overlap, zero sklearn warnings, >= 4 heavy (>1000g) sessions per fold ([4, 5, 5, 5, 5]), and permutation/repeat byte-determinism"
    requirement: DATA-02
    verification:
      - kind: integration
        ref: "backend/tests/test_manifest_integration.py#test_real_five_fold_splits_zero_leakage_and_permutation_determinism"
        status: pass
    human_judgment: false

duration: 6min
completed: 2026-11-20
status: complete
---

# Phase 01 Plan 03: Production Manifest CLI, Materialized Parquets & Verified Aggregate Audit Summary

**Full idempotent `build_manifest` CLI materializing verified `1,738`-clean / `15`-quarantined typed Parquet manifests and byte-idempotent, zero-path-leakage `models/dataset_audit_summary.json` passing all 39 Phase 1 unit, Audit §9, anti-leakage, security, and git-hygiene tests**

## Performance

- **Duration:** 6 min
- **Started:** 2026-11-20T21:56:14Z
- **Completed:** 2026-11-20T22:01:48Z
- **Tasks:** 2 (Task 1 TDD + Task 2 full real-dataset materialization & integration gate)
- **Files modified:** 3 created, 1 modified (plus 2 git-ignored Parquet artifacts materialized under `data/processed/`)

## Accomplishments

- Implemented `backend/app/ingestion/audit_summary.py` exporting `AUDIT_SUMMARY_SCHEMA_VERSION` (`"1.0.0"`), `DATASET_IDENTITY` (`doi = "10.17632/zrs8kk9dvr.1"`), `MENDELEY_PUBLISHED_CLAIMS` (`1,772` images / `"daily for 45 days"`), `SPARSE_GAP_LOWER_EXCLUSIVE_G` (`801`), `SPARSE_GAP_UPPER_EXCLUSIVE_G` (`1216`), `build_audit_summary()`, and `write_audit_summary()` with runtime `PROIBIÇÃO 4` leak prevention and zero wall-clock timestamps.
- Upgraded `backend/scripts/build_manifest.py` from its Plan `01-01` tracer into the complete production CLI orchestrating `scan_dataset_records` -> `split_clean_and_quarantine` -> `assign_session_folds` -> `verify_fold_integrity` -> `build_audit_summary` -> `write_manifest_parquet` + `write_audit_summary`, honoring `--dataset-root`, `--output-dir`, `--summary-path`, `--dry-run`, and `--min-heavy-sessions` while logging strictly aggregate counts to stdout.
- Materialized from read-only `data/raw/Broiler weights of Cobb Avian/` and verified byte-identical across consecutive CLI runs (SHA-256 `b795f969b31c6db90a13a2b4e8626ac6101ab8bdbf58efc2c351a20d28f9363e` for `models/dataset_audit_summary.json` and identical row hashes for both Parquets):
  - `data/processed/manifest_clean.parquet` (81 KiB, git-ignored): `1,738` images, `160` unique weights (`116–2,093 g`), `162` leaf folders, `188` sessions, `24` typed Arrow columns, `fold` in `{0, 1, 2, 3, 4}` (`0` session overlap, `0` sklearn warnings, heavy sessions `[4, 5, 5, 5, 5]` across folds).
  - `data/processed/manifest_quarantine.parquet` (15 KiB, git-ignored): `15` images (`13` `multi_bird` across `417`/`757`/`823`/`1520` + `2` `severe_mislabel` across `1351`/`1465`, `fold = -1`).
  - `models/dataset_audit_summary.json` (7.1 KiB, git-tracked): reconciles Mendeley (`1,772` imgs / 45 days) vs. disk (`1,753` raw / `9` dates / `23` calendar days / `~7–29 days`), confirms `0` dates in 2024 (`21` CP866 files recovered with `0` mtime fallback), `1,435` (`81.86%`) rotated EXIF headers, `67` variable digital-zoom images, `367 -> 267 g` (`6` rows), `1340_ -> 1340 g` (`6` rows), `39` nested level-2 images (`171/173`, `190/191`, `190/193`), the `801–1,216 g` sparse gap (`1` clean bird / `6` images at `1009 g`), and the `8`-day hiatus (`2022-04-03` to `2022-04-11`).
- Created `backend/tests/test_manifest_integration.py` (6 comprehensive read-only integration, security, and git-hygiene tests) bringing the complete Phase 1 test suite to `39 passed in 2.49s` with zero `ruff` diagnostics across `backend/`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build deterministic zero-path-leakage audit summary module and upgrade build_manifest.py to full production pipeline** - `88b0d84` (feat)
2. **Task 2: Materialize curated Parquets and models/dataset_audit_summary.json from read-only raw dataset and pass full D-06/D-08 integration gate** - `f436d1d` (feat)

## Files Created/Modified

- `backend/app/ingestion/audit_summary.py` - Deterministic aggregate audit summary builder (`AUDIT_SUMMARY_SCHEMA_VERSION`, `DATASET_IDENTITY`, `MENDELEY_PUBLISHED_CLAIMS`, `SPARSE_GAP_LOWER_EXCLUSIVE_G`, `SPARSE_GAP_UPPER_EXCLUSIVE_G`, `build_audit_summary`, `write_audit_summary`).
- `backend/scripts/build_manifest.py` - Upgraded from Plan `01-01` tracer to full production CLI (`parse_args`, `run_build_manifest`, `main`) with `--min-heavy-sessions` auto-enforcement and zero-side-effect `--dry-run`.
- `models/dataset_audit_summary.json` - Git-tracked, byte-idempotent aggregate forensic audit JSON ready for Phase 4 `WEB-05` Audit Dashboard (zero raw file paths or volatile timestamps).
- `backend/tests/test_manifest_integration.py` - Read-only D-06/D-08 integration verification gate covering 24-column Arrow schema, all Audit §9 anomalies, anti-leakage 5-fold splits & permutation determinism, zero-path-leakage JSON contract, `git ls-files` / `git check-ignore` hygiene, and CLI dry-run/full-run behavior.

## Decisions Made

- Enforced `PROIBIÇÃO 4` (`T-01-07`) both structurally (aggregating strictly counters and audited folder IDs) and at serialization time inside `write_audit_summary()` by rejecting any JSON text containing `.jpg`, `rel_path`, `/home/`, or `data/raw`.
- Kept `--min-heavy-sessions` defaulting to `None` so running over the real corpus (`RAW_DATASET_DIR` or any dataset with `>= 20` heavy sessions) automatically enforces `MIN_HEAVY_SESSIONS_PER_FOLD = 4`, while synthetic 5-heavy-session trees in unit tests run cleanly without flag churn and `--min-heavy-sessions 0` disables the floor explicitly.
- Verified `verify_fold_integrity()` explicitly in `run_build_manifest()` immediately after `assign_session_folds()` as well as inside `build_audit_summary()`, making the anti-leakage audit contract self-verifying on every execution.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **Phase 1 Complete (`01-01`, `01-02`, `01-03`):** Both `DATA-01` and `DATA-02` are fully satisfied and locked behind 39 automated unit, AST, security, git-hygiene, and real-dataset integration tests.
- Ready for **Phase 2 (Classical PDI Feature Extraction & Multi-Bird Watershed Engine)**:
  - `data/processed/manifest_clean.parquet` (1,738 rows × 24 cols) and `data/processed/manifest_quarantine.parquet` (15 rows × 24 cols, including `417`/`757`/`823`/`1520` for Watershed evaluation) are materialized and verified.
  - `backend/app/pdi/io_exif.py` is ready as the single Pillow EXIF-first RGB image loader for Phases 2 and 4.
  - `models/dataset_audit_summary.json` is git-tracked and ready for Phase 4's `GET /api/metrics` and `WEB-05` Audit Dashboard.

## Self-Check: PASSED

- Verified all created/modified tracked files exist on disk: `backend/app/ingestion/audit_summary.py`, `backend/scripts/build_manifest.py`, `models/dataset_audit_summary.json`, `backend/tests/test_manifest_integration.py`.
- Verified git-ignored runtime artifacts exist on disk: `data/processed/manifest_clean.parquet` (81 KiB), `data/processed/manifest_quarantine.parquet` (15 KiB), and `git ls-files data/raw data/processed` returns empty.
- Verified atomic task commits `88b0d84` and `f436d1d` exist in `git log`.
- Verified byte-idempotent SHA-256 (`b795f969b31c6db90a13a2b4e8626ac6101ab8bdbf58efc2c351a20d28f9363e`) across consecutive `uv run python -m backend.scripts.build_manifest` executions, full `uv run pytest -v` (`39 passed in 2.49s`), and `uv run ruff check backend/` (`All checks passed!`).

---
*Phase: 01-forensic-data-curation-anti-leakage-split*
*Completed: 2026-11-20*
