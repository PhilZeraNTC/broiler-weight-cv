---
phase: 01-forensic-data-curation-anti-leakage-split
plan: 02
subsystem: ingestion
tags: [forensic-decoding, cp866, arabic-indic, parquet, pyarrow, pandas3, stratified-group-kfold, anti-leakage]

# Dependency graph
requires:
  - phase: 01-forensic-data-curation-anti-leakage-split
    provides: "uv Python 3.12 workspace, shared Pillow EXIF-first loader (backend/app/pdi/io_exif.py), repo-root config (backend/app/config.py), and synthetic multi-band dataset fixtures (backend/tests/conftest.py)"
provides:
  - Audit §9 forensic constants, CP866/Arabic-Indic filename decoder, folder weight sanitizer, optical lens classifier, coarse weight-band classifier, and deterministic SHA-256 image_id helper (backend/app/ingestion/decoding.py)
  - Path-contained read-only recursive dataset crawler, immutable 24-column PyArrow Parquet schema enforcer, sequential >600.0 s session_id clusterer, and clean/quarantine partitioner (backend/app/ingestion/manifest.py)
  - Deterministic 5-fold StratifiedGroupKFold session splitter with 3-key mergesort ordering, sklearn warning escalation, zero cross-fold session overlap guard, and heavy-bird (>=4 sessions/fold) balance verification (backend/app/ingestion/splits.py)
  - Public ingestion package barrel exporting all 28 contract symbols (backend/app/ingestion/__init__.py)
  - 16 new synthetic unit and AST anti-leakage tests covering all Audit §9 anomalies and all 5 DATA-02 edge contracts (backend/tests/test_decoding.py, backend/tests/test_splits.py)
affects: [01-03, phase-2-classical-pdi, phase-3-modeling, phase-4-web-mvp]

actuals:
  tokens: 14614
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - Verbatim Audit §9 forensic decoding with CP866-to-UTF-8 reversal guarded on Arabic-Indic digits (U+0660..U+0669) and naive local mtime - 14401 s fallback
    - Explicit PyArrow _MANIFEST_ARROW_SCHEMA guaranteeing identical 24-column Parquet types (string, timestamp[us], date32[day], int32, int8, float64, bool) on both empty and non-empty DataFrames
    - Pandas 3.1 Copy-on-Write safe manifest transformations using .copy() and single-step .loc mutations
    - Deterministic 3-key mergesort pre-sort (["folder_chain", "capture_dt", "rel_path"]) prior to session clustering and 5-fold StratifiedGroupKFold(shuffle=True, random_state=42)
    - Zero-warning sklearn capture guard escalating any StratifiedGroupKFold warning into RuntimeError alongside AST enforcement banning random/ungrouped splitters

key-files:
  created:
    - backend/app/ingestion/decoding.py
    - backend/app/ingestion/manifest.py
    - backend/app/ingestion/splits.py
    - backend/app/ingestion/__init__.py
    - backend/tests/test_decoding.py
    - backend/tests/test_splits.py
  modified: []

key-decisions:
  - "Passed an explicit 24-column PyArrow schema (_MANIFEST_ARROW_SCHEMA) to to_parquet() alongside Pandas 3.1 dtype='str' enforcement so both empty DataFrames and all-null quarantine_reason columns serialize identically as Arrow string/date32[day]/timestamp[us] and pass exact pd.testing.assert_frame_equal() round-trips (D-03)."
  - "Filtered non-symlink contained JPEG paths before parsing leaf folder weights in scan_dataset_records() so symlinked files or directories escaping dataset_root are silently ignored without triggering false folder-weight errors (T-01-04)."
  - "Validated per-band session counts and total heavy-session floor upfront in assign_session_folds() when min_heavy_sessions_per_fold is provided, while retaining warnings.catch_warnings escalation to RuntimeError for any unexpected sklearn StratifiedGroupKFold warnings (D-05, D-06, T-01-06)."

patterns-established:
  - "Pattern 4 (Immutable 24-Column Parquet Contract): Always serialize clean and quarantine manifests through write_manifest_parquet() with enforce_manifest_dtypes() and _MANIFEST_ARROW_SCHEMA (never CSV, never float32 optical tags, never timezone-aware capture_dt)."
  - "Pattern 5 (3-Key Deterministic Ordering): Always sort stably with mergesort on ['folder_chain', 'capture_dt', 'rel_path'] before session_id clustering and before StratifiedGroupKFold splitting so same-second _1 bursts remain invariant to filesystem scan order."
  - "Pattern 6 (Zero-Leakage CV Audit): Always assign clean folds exclusively through assign_session_folds() and verify zero cross-fold session_id overlap and heavy-bird representation via verify_fold_integrity()."

requirements-completed:
  - DATA-01
  - DATA-02

coverage:
  - id: D1
    description: "Audit §9 forensic decoder (decoding.py) with verbatim constants, CP866 mojibake reversal, Eastern-Arabic normalization, folder suffix sanitization, weight-band/lens classifiers, and SHA-256 image_id"
    requirement: DATA-01
    verification:
      - kind: unit
        ref: "backend/tests/test_decoding.py#test_audit_section_9_constants_match_verbatim"
        status: pass
      - kind: unit
        ref: "backend/tests/test_decoding.py#test_decode_filename_all_patterns_and_cp866_recovery"
        status: pass
      - kind: unit
        ref: "backend/tests/test_decoding.py#test_parse_folder_weight_g_and_error_handling"
        status: pass
      - kind: unit
        ref: "backend/tests/test_decoding.py#test_boundary_weight_bands_lens_and_image_id"
        status: pass
    human_judgment: false
  - id: D2
    description: "Path-contained read-only dataset crawler, sequential >600.0 s session_id clusterer, clean/quarantine partitioner, and exact 24-column PyArrow Parquet serializer (manifest.py) covering boundary, adjacency, empty, ordering, precision, and symlink containment"
    requirement: DATA-01
    verification:
      - kind: unit
        ref: "backend/tests/test_decoding.py#test_scan_dataset_records_synthetic_tree_enforces_audit_section_9"
        status: pass
      - kind: unit
        ref: "backend/tests/test_decoding.py#test_session_gap_boundary_exact_600s_and_burst_adjacency"
        status: pass
      - kind: unit
        ref: "backend/tests/test_decoding.py#test_empty_and_single_image_dataset_contracts"
        status: pass
      - kind: unit
        ref: "backend/tests/test_decoding.py#test_symlink_file_and_directory_escape_rejection"
        status: pass
      - kind: unit
        ref: "backend/tests/test_decoding.py#test_manifest_24_column_parquet_schema_and_precision_roundtrip"
        status: pass
    human_judgment: false
  - id: D3
    description: "Deterministic 5-fold StratifiedGroupKFold session splitter (splits.py) and ingestion package barrel (__init__.py) enforcing zero cross-fold session overlap, zero sklearn warnings, heavy-session balance (>=4/fold), permutation determinism, and PROIBIÇÃO 3 AST compliance"
    requirement: DATA-02
    verification:
      - kind: unit
        ref: "backend/tests/test_splits.py#test_audit_section_10_shaped_dataset_produces_balanced_zero_leakage_splits"
        status: pass
      - kind: unit
        ref: "backend/tests/test_splits.py#test_assign_session_folds_is_permutation_and_repeat_deterministic"
        status: pass
      - kind: unit
        ref: "backend/tests/test_splits.py#test_assign_session_folds_end_to_end_with_synthetic_dataset_tree"
        status: pass
      - kind: unit
        ref: "backend/tests/test_splits.py#test_empty_undersized_and_warning_inputs_fail_loudly"
        status: pass
      - kind: unit
        ref: "backend/tests/test_splits.py#test_verify_fold_integrity_catches_injected_leakage_and_invalid_folds"
        status: pass
      - kind: unit
        ref: "backend/tests/test_splits.py#test_ast_scan_prohibits_random_and_ungrouped_splitters_in_ingestion"
        status: pass
      - kind: unit
        ref: "backend/tests/test_splits.py#test_ingestion_package_barrel_exports_public_api"
        status: pass
    human_judgment: false

duration: 9min
completed: 2026-11-20
status: complete
---

# Phase 01 Plan 02: Forensic Decoder, 24-Column Parquet Manifest Builder & Leak-Free 5-Fold Session Splitter Summary

**Verbatim Audit §9 forensic decoder (`decoding.py`), path-contained 24-column PyArrow Parquet manifest engine (`manifest.py`), and deterministic zero-leakage 5-fold `StratifiedGroupKFold` session splitter (`splits.py`) with 16 passing synthetic edge-contract and AST anti-leakage tests**

## Performance

- **Duration:** 9 min
- **Started:** 2026-11-20T21:40:33Z
- **Completed:** 2026-11-20T21:49:49Z
- **Tasks:** 2 (both TDD)
- **Files modified:** 6 created, 0 modified

## Accomplishments

- Implemented `backend/app/ingestion/decoding.py` with all verbatim Audit §9 constants (`MTIME_OFFSET_SEC = 14401`, `SESSION_GAP_SECONDS = 600.0`, `MULTI_BIRD_FOLDERS = {"417", "757", "823", "1520"}`, `SEVERE_MISLABEL_FOLDERS = {"1351", "1465"}`, `LCD_CORRECTIONS = {"367": 267}`), anchored ReDoS-free regexes, CP866-to-UTF-8 mojibake recovery guarded on Eastern-Arabic digits (`U+0660..U+0669`), folder weight parsing (`1340_` -> `1340`), lens/weight-band classifiers, and deterministic 16-hex SHA-256 `image_id` hashing.
- Implemented `backend/app/ingestion/manifest.py` enforcing the immutable 24-column Parquet schema (`15` Audit §9 base + `9` derived columns) via explicit PyArrow schema serialization (`timestamp[us]`, `date32[day]`, `string`, `int32`, `int8`, `float64`, `bool`), Pandas 3.1 Copy-on-Write safe operations, ASVS L1 symlink/path-containment pruning (`os.walk(..., followlinks=False)` + `is_relative_to`), shared Pillow EXIF ingestion via `read_exif_metadata`, deterministic 3-key `mergesort` ordering, strict `> 600.0 s` `session_id` clustering, and clean/quarantine partitioning (`fold = -1` for quarantine).
- Implemented `backend/app/ingestion/splits.py` and the 28-symbol package barrel `backend/app/ingestion/__init__.py`, delivering deterministic 5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly by `session_id` and stratified by `weight_band`, escalating any sklearn warning to `RuntimeError`, auditing zero cross-fold session overlap, and enforcing heavy-bird (`>1000g`) session balance (`[4, 5, 5, 5, 5]` on Audit §10's 24 heavy sessions).
- Verified all 33 tests across `test_io_exif.py` (17), `test_decoding.py` (9), and `test_splits.py` (7) — explicitly covering all 5 DATA-02 edge contracts (`boundary`, `adjacency`, `empty`, `ordering`, `precision`), symlink escape rejection, warning escalation, and AST `PROIBIÇÃO 3` enforcement — plus zero `ruff` diagnostics across `backend/`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement Audit §9 forensic decoder and safe 24-column Parquet manifest builder with unit tests** - `96a0f7f` (feat)
2. **Task 2: Implement deterministic 5-fold StratifiedGroupKFold session splitter and package barrel with anti-leakage tests** - `2b51a2b` (feat)

## Files Created/Modified

- `backend/app/ingestion/decoding.py` - Audit §9 constants, `DecodedFilename`, CP866/Arabic-Indic filename decoder, folder weight parser, lens/weight-band classifiers, and SHA-256 `image_id` helper.
- `backend/app/ingestion/manifest.py` - 24-column Parquet schema contract (`RAW_BASE_COLUMNS`, `DERIVED_COLUMNS`, `MANIFEST_COLUMNS`), `enforce_manifest_dtypes`, path-contained `scan_dataset_records`, `split_clean_and_quarantine`, and PyArrow-schema-enforced `write_manifest_parquet`.
- `backend/app/ingestion/splits.py` - `N_SPLITS`, `RANDOM_STATE`, `SORT_KEYS`, `MIN_HEAVY_SESSIONS_PER_FOLD`, `assign_session_folds`, and `verify_fold_integrity` with zero-warning and zero-overlap guarantees.
- `backend/app/ingestion/__init__.py` - Public package barrel re-exporting all 28 ingestion constants, dataclasses, and functions via explicit `__all__`.
- `backend/tests/test_decoding.py` - 9 unit tests covering Audit §9 rules, weight-band/gap boundaries, `_1` burst adjacency, empty/single-image contracts, 3-key ordering, float64/microsecond precision, symlink file/directory escape rejection, and 24-column Parquet round-trips.
- `backend/tests/test_splits.py` - 7 unit/AST tests covering Audit §10-shaped 188-session heavy-bird balance (`[4, 5, 5, 5, 5]`), permutation/repeat determinism, end-to-end synthetic dataset splitting, undersized/warning failure escalation, injected cross-fold leakage detection, AST `PROIBIÇÃO 3` compliance, and package barrel exports.

## Decisions Made

- Used an explicit PyArrow `_MANIFEST_ARROW_SCHEMA` inside `write_manifest_parquet()` paired with Pandas 3.1 `dtype="str"` enforcement so empty manifests and all-null `quarantine_reason` columns write exact `date32[day]` and `string` Arrow fields rather than decaying to Parquet `null` or `large_string`, guaranteeing exact `pd.testing.assert_frame_equal()` round-trip parity.
- Checked `_is_contained_non_symlink()` on candidate files before parsing leaf directory names so a directory holding only a rejected symlink cannot trigger spurious `parse_folder_weight_g` errors.
- Guarded `classify_lens()` against non-finite floats (`NaN`/`Inf`) and `enforce_manifest_dtypes()` against timezone-aware `capture_dt` series to protect downstream Audit §9 naive local timestamp arithmetic and optical calibration.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- In Task 1 TDD verification, the initial count expectation in `test_decoding.py` for the shared `synthetic_dataset_tree` fixture was written as 19 instead of the fixture's actual 18 valid images (16 clean across 15 sessions + 2 quarantined, excluding the external symlink); corrected the test assertion and cleaned up 5 `ruff` line-length / `UP017` / `SIM300` diagnostics before committing.
- In Task 2 TDD verification, confirming sklearn 1.9.1's `StratifiedGroupKFold` warning escalation revealed that sklearn warns when a stratum has fewer than `n_splits=5` total samples (rather than 2 multi-image sessions totaling 13 samples); refined the synthetic warning test fixture to supply 2 single-image heavy samples so the zero-warning `RuntimeError` escalation path is deterministically exercised.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Ready for Plan `01-03`: the complete `backend.app.ingestion` package (`decoding.py`, `manifest.py`, `splits.py`, `__init__.py`) is implemented, verified, and ready to be wired into `audit_summary.py`, the upgraded `backend/scripts/build_manifest.py`, and read-only real-dataset integration tests against `data/raw/Broiler weights of Cobb Avian/`.

## Self-Check: PASSED

- Verified all 6 created files exist on disk (`backend/app/ingestion/decoding.py`, `backend/app/ingestion/manifest.py`, `backend/app/ingestion/splits.py`, `backend/app/ingestion/__init__.py`, `backend/tests/test_decoding.py`, `backend/tests/test_splits.py`).
- Verified atomic task commits `96a0f7f` and `2b51a2b` in `git log`.
- Verified `uv run pytest backend/tests/test_io_exif.py backend/tests/test_decoding.py backend/tests/test_splits.py -v` (`33 passed in 2.03s`), `uv run ruff check backend/` (`All checks passed!`), and `uv run python -c "import backend.app.ingestion as ing; assert len(ing.MANIFEST_COLUMNS) == 24"` (`Barrel import verified: 28 exports`).

---
*Phase: 01-forensic-data-curation-anti-leakage-split*
*Completed: 2026-11-20*
