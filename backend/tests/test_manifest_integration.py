"""Read-only Phase 1 integration, D-06/D-08 audit, security, and git-hygiene tests.

Reads the materialized ``data/processed/manifest_clean.parquet``,
``data/processed/manifest_quarantine.parquet``, and git-tracked
``models/dataset_audit_summary.json`` strictly read-only. Every assertion prints
strictly aggregate metrics — never raw ``rel_path`` lists or ``.jpg`` filenames
(PROIBIÇÃO 4, T-01-07, T-01-08).
"""

from __future__ import annotations

import json
import re
import subprocess
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.app.config import (
    AUDIT_SUMMARY_PATH,
    CLEAN_MANIFEST_PATH,
    QUARANTINE_MANIFEST_PATH,
    REPO_ROOT,
)
from backend.app.ingestion.audit_summary import (
    AUDIT_SUMMARY_SCHEMA_VERSION,
    DATASET_IDENTITY,
    MENDELEY_PUBLISHED_CLAIMS,
    build_audit_summary,
    write_audit_summary,
)
from backend.app.ingestion.manifest import MANIFEST_COLUMNS
from backend.app.ingestion.splits import (
    MIN_HEAVY_SESSIONS_PER_FOLD,
    N_SPLITS,
    RANDOM_STATE,
    assign_session_folds,
    verify_fold_integrity,
)
from backend.scripts import build_manifest

_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
_FORBIDDEN_LEAK_MARKERS: tuple[str, ...] = (".jpg", "rel_path", "/home/", "data/raw")


def _load_real_manifests() -> tuple[pd.DataFrame, pd.DataFrame]:
    assert CLEAN_MANIFEST_PATH.is_file(), "manifest_clean.parquet must be materialized"
    assert QUARANTINE_MANIFEST_PATH.is_file(), "manifest_quarantine.parquet must be materialized"
    clean_df = pd.read_parquet(CLEAN_MANIFEST_PATH)
    quarantine_df = pd.read_parquet(QUARANTINE_MANIFEST_PATH)
    return clean_df, quarantine_df


def test_real_parquets_exact_totals_24_column_schema_and_path_hygiene() -> None:
    clean_df, quarantine_df = _load_real_manifests()

    # Exact 24-column PyArrow Parquet schema contract (D-03)
    expected_arrow_types: dict[str, pa.DataType] = {
        "image_id": pa.string(),
        "rel_path": pa.string(),
        "folder_chain": pa.string(),
        "raw_weight_g": pa.int32(),
        "weight_g": pa.int32(),
        "capture_dt": pa.timestamp("us"),
        "capture_date": pa.date32(),
        "oriented_w": pa.int32(),
        "oriented_h": pa.int32(),
        "exif_orientation": pa.int8(),
        "focal_length_mm": pa.float64(),
        "zoom_multiple": pa.float64(),
        "is_multi_bird": pa.bool_(),
        "is_severe_mislabel": pa.bool_(),
        "is_single_bird_clean": pa.bool_(),
        "session_id": pa.string(),
        "filename_pattern": pa.string(),
        "was_cp866_recovered": pa.bool_(),
        "was_nested_folder": pa.bool_(),
        "was_lcd_corrected": pa.bool_(),
        "lens_class": pa.string(),
        "weight_band": pa.string(),
        "quarantine_reason": pa.string(),
        "fold": pa.int8(),
    }

    for parquet_path in (CLEAN_MANIFEST_PATH, QUARANTINE_MANIFEST_PATH):
        schema = pq.read_schema(parquet_path)
        assert schema.names == list(MANIFEST_COLUMNS)
        assert len(schema.names) == 24
        for col_name, expected_type in expected_arrow_types.items():
            assert schema.field(col_name).type == expected_type, (
                f"Schema mismatch on column {col_name!r}: "
                f"{schema.field(col_name).type} != {expected_type}"
            )

    assert list(clean_df.columns) == list(MANIFEST_COLUMNS)
    assert list(quarantine_df.columns) == list(MANIFEST_COLUMNS)

    # Exact dataset totals (D-06)
    assert len(clean_df) == 1738
    assert len(quarantine_df) == 15
    assert len(clean_df) + len(quarantine_df) == 1753
    assert int(clean_df["weight_g"].nunique()) == 160
    assert int(clean_df["session_id"].nunique()) == 188
    assert int(clean_df["folder_chain"].nunique()) == 162

    # Key dtypes & fold ranges
    for df in (clean_df, quarantine_df):
        assert df["fold"].dtype == np.int8
        assert df["focal_length_mm"].dtype == np.float64
        assert df["zoom_multiple"].dtype == np.float64
        assert df["capture_dt"].dtype == np.dtype("datetime64[us]")

    assert set(clean_df["fold"].unique().tolist()) == {0, 1, 2, 3, 4}
    assert set(quarantine_df["fold"].unique().tolist()) == {-1}

    # Unique 16-char lowercase hex image_id across all 1,753 rows
    all_ids = pd.concat([clean_df["image_id"], quarantine_df["image_id"]], ignore_index=True)
    assert int(all_ids.nunique()) == 1753
    invalid_id_count = sum(1 for val in all_ids.astype(str) if _HEX16_RE.match(val) is None)
    assert invalid_id_count == 0

    # Strict POSIX forward-slash path containment: zero backslashes or parent-traversal segments
    for col in ("rel_path", "folder_chain"):
        all_vals = pd.concat([clean_df[col], quarantine_df[col]], ignore_index=True).astype(str)
        backslash_count = sum(1 for v in all_vals if "\\" in v)
        dotdot_count = sum(1 for v in all_vals if ".." in v.split("/"))
        assert backslash_count == 0, f"Found {backslash_count} backslash path(s) in {col}"
        assert dotdot_count == 0, f"Found {dotdot_count} parent-traversal segment(s) in {col}"


def test_audit_section_9_anomalies_and_band_totals_verified_in_real_manifests() -> None:
    clean_df, quarantine_df = _load_real_manifests()
    full_df = pd.concat([clean_df, quarantine_df], ignore_index=True)

    # Zero 2024 timestamps across both clean and quarantine (CP866 recovery proof)
    assert int((clean_df["capture_dt"].dt.year == 2024).sum()) == 0
    assert int((quarantine_df["capture_dt"].dt.year == 2024).sum()) == 0

    cp866_547 = clean_df.loc[clean_df["folder_chain"] == "547"]
    cp866_562 = clean_df.loc[clean_df["folder_chain"] == "562"]
    assert len(cp866_547) == 11
    assert len(cp866_562) == 10
    cp866_both = pd.concat([cp866_547, cp866_562], ignore_index=True)
    assert bool(cp866_both["was_cp866_recovered"].all())
    assert set(cp866_both["filename_pattern"].unique().tolist()) == {"img_mojibake_cp866"}
    assert int(full_df["was_cp866_recovered"].sum()) == 21

    # 9 distinct capture dates spanning 2022-03-24 .. 2022-04-15
    clean_dates = sorted(clean_df["capture_date"].unique().tolist())
    assert len(clean_dates) == 9
    assert clean_dates[0] == date(2022, 3, 24)
    assert clean_dates[-1] == date(2022, 4, 15)

    # LCD correction: 367 -> 267 g (6 rows)
    lcd_367 = clean_df.loc[clean_df["folder_chain"] == "367"]
    assert len(lcd_367) == 6
    assert set(lcd_367["raw_weight_g"].unique().tolist()) == {367}
    assert set(lcd_367["weight_g"].unique().tolist()) == {267}
    assert bool(lcd_367["was_lcd_corrected"].all())

    # Suffix sanitization: 1340_ -> 1340 g (6 rows)
    sanitized_1340 = clean_df.loc[clean_df["folder_chain"] == "1340_"]
    assert len(sanitized_1340) == 6
    assert set(sanitized_1340["raw_weight_g"].unique().tolist()) == {1340}
    assert set(sanitized_1340["weight_g"].unique().tolist()) == {1340}

    # Level-2 nested folders captured with exact leaf weights
    nested_expectations = {
        "171/173": (8, 173),
        "190/191": (9, 191),
        "190/193": (22, 193),
    }
    for folder_chain, (expected_rows, expected_weight) in nested_expectations.items():
        subset = clean_df.loc[clean_df["folder_chain"] == folder_chain]
        assert len(subset) == expected_rows
        assert set(subset["weight_g"].unique().tolist()) == {expected_weight}
        assert bool(subset["was_nested_folder"].all())
    assert int(clean_df["was_nested_folder"].sum()) == 39

    # Quarantine breakdown: 13 multi-bird + 2 severe mislabels
    multi_bird = quarantine_df.loc[quarantine_df["quarantine_reason"] == "multi_bird"]
    severe_mislabel = quarantine_df.loc[quarantine_df["quarantine_reason"] == "severe_mislabel"]
    assert len(multi_bird) == 13
    assert multi_bird["folder_chain"].value_counts().to_dict() == {
        "417": 4,
        "757": 1,
        "823": 4,
        "1520": 4,
    }
    assert len(severe_mislabel) == 2
    assert severe_mislabel["folder_chain"].value_counts().to_dict() == {
        "1351": 1,
        "1465": 1,
    }

    # Audit §10 clean weight-band totals
    band_expectations = {
        "<500g": (1223, 124),
        "500-1000g": (333, 40),
        ">1000g": (182, 24),
    }
    for band, (expected_imgs, expected_sessions) in band_expectations.items():
        band_df = clean_df.loc[clean_df["weight_band"] == band]
        assert len(band_df) == expected_imgs
        assert int(band_df["session_id"].nunique()) == expected_sessions

    # Total rotated EXIF headers across all 1,753 raw records
    assert int((full_df["exif_orientation"] != 1).sum()) == 1435


def test_real_five_fold_splits_zero_leakage_and_permutation_determinism() -> None:
    clean_df, _ = _load_real_manifests()

    audit = verify_fold_integrity(
        clean_df,
        min_heavy_sessions_per_fold=MIN_HEAVY_SESSIONS_PER_FOLD,
    )
    assert audit["cross_fold_session_overlap"] == 0
    assert len(audit["folds"]) == N_SPLITS
    for fold_entry in audit["folds"]:
        assert fold_entry["n_heavy_sessions"] >= MIN_HEAVY_SESSIONS_PER_FOLD

    # Re-split a randomly shuffled copy and assert 0 warnings and exact parity
    shuffled = clean_df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reassigned = assign_session_folds(
            shuffled,
            min_heavy_sessions_per_fold=MIN_HEAVY_SESSIONS_PER_FOLD,
        )
    assert len(caught) == 0
    assert reassigned["image_id"].tolist() == clean_df["image_id"].tolist()
    assert np.array_equal(reassigned["fold"].to_numpy(), clean_df["fold"].to_numpy())


def test_audit_summary_json_contract_zero_path_leakage_and_determinism(tmp_path: Path) -> None:
    clean_df, quarantine_df = _load_real_manifests()

    assert AUDIT_SUMMARY_PATH.is_file(), "models/dataset_audit_summary.json must exist"
    raw_text = AUDIT_SUMMARY_PATH.read_text(encoding="utf-8")
    lowered_text = raw_text.lower()
    for forbidden in _FORBIDDEN_LEAK_MARKERS:
        assert forbidden not in lowered_text, (
            f"PROIBIÇÃO 4 / T-01-07 violation: forbidden marker {forbidden!r} in audit summary"
        )

    payload = json.loads(raw_text)
    assert payload["schema_version"] == AUDIT_SUMMARY_SCHEMA_VERSION

    # Dataset identity verbatim against Audit §1
    identity = payload["dataset_identity"]
    assert identity == DATASET_IDENTITY
    assert identity["doi"] == "10.17632/zrs8kk9dvr.1"
    assert identity["name"] == "Broiler weights of Cobb Avian"
    assert identity["breed"] == "Cobb Avian 48"
    assert identity["capture_device"] == "Xiaomi Redmi 10"
    assert identity["scale_model"] == "SF-400"

    # Mendeley vs disk reconciliation
    recon = payload["mendeley_vs_disk_reconciliation"]
    assert recon["published_claims"] == MENDELEY_PUBLISHED_CLAIMS
    assert recon["published_claims"]["published_images"] == 1772
    disk = recon["disk_audit"]
    assert disk["raw_images"] == 1753
    assert disk["missing_vs_mendeley_images"] == 19
    assert disk["distinct_capture_dates"] == 9
    assert disk["calendar_span_days"] == 23
    assert disk["clean_weight_min_g"] == 116
    assert disk["clean_weight_max_g"] == 2093

    # Curation funnel
    funnel = payload["curation_funnel"]
    assert funnel["raw_images"] == 1753
    assert funnel["clean_images"] == 1738
    assert funnel["clean_unique_weights"] == 160
    assert funnel["clean_sessions"] == 188
    assert funnel["quarantined_total"] == 15
    assert funnel["quarantined_multi_bird"]["count"] == 13
    assert funnel["quarantined_severe_mislabel"]["count"] == 2
    assert funnel["lcd_corrected_images"]["count"] == 6
    assert funnel["lcd_corrected_images"]["corrections_applied"] == {"367": 267}

    # Encoding & EXIF recovery
    enc = payload["encoding_and_exif_recovery"]
    assert enc["exif_orientation_rotated_count"] == 1435
    assert enc["cp866_recovered_count"] == 21
    assert enc["cp866_mtime_fallback_count"] == 0
    assert enc["year_2024_capture_count"] == 0
    assert enc["nested_level2_folders"] == ["171/173", "190/191", "190/193"]
    assert enc["nested_level2_images_count"] == 39
    assert enc["suffix_sanitized_folders"] == ["1340_"]
    assert enc["variable_digital_zoom_count"] == 67

    # Coverage gaps (801-1216 g sparse interval + 8-day hiatus)
    gaps = payload["distributions"]["coverage_gaps"]
    sparse = gaps["weight_sparse_interval"]
    assert sparse["lower_exclusive_g"] == 801
    assert sparse["upper_exclusive_g"] == 1216
    assert sparse["span_g"] == 415
    assert sparse["clean_images_inside"] == 6
    assert sparse["clean_unique_weights_inside"] == 1
    assert sparse["clean_weights_inside"] == [1009]
    assert sparse["clean_sessions_inside"] == 1
    hiatus = gaps["temporal_hiatus"]
    assert hiatus["max_gap_days"] == 8
    assert hiatus["gap_start_date"] == "2022-04-03"
    assert hiatus["gap_end_date"] == "2022-04-11"

    # Split audit
    split_audit = payload["split_audit"]
    assert split_audit["strategy"] == "StratifiedGroupKFold"
    assert split_audit["n_splits"] == N_SPLITS
    assert split_audit["shuffle"] is True
    assert split_audit["random_state"] == RANDOM_STATE
    assert split_audit["cross_fold_session_overlap"] == 0
    assert len(split_audit["folds"]) == N_SPLITS
    assert all(f["n_heavy_sessions"] >= MIN_HEAVY_SESSIONS_PER_FOLD for f in split_audit["folds"])

    # Deterministic regeneration: zero volatile fields (exact dict equality)
    rebuilt = build_audit_summary(clean_df, quarantine_df)
    assert rebuilt == payload

    # Defense-in-depth: write_audit_summary refuses forbidden path/filename markers (T-01-07)
    with pytest.raises(ValueError, match="PROIBIÇÃO 4"):
        write_audit_summary({"bad": "116/116 (1).jpg"}, tmp_path / "rejected.json")


def test_git_hygiene_excludes_raw_and_processed_while_tracking_audit_summary() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "data/raw", "data/processed"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout.strip() == "", (
        "PROIBIÇÃO 4 / T-01-08 violation: data/raw or data/processed files are tracked by Git."
    )

    ignored_check = subprocess.run(
        ["git", "check-ignore", "models/dataset_audit_summary.json"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ignored_check.returncode != 0, (
        "models/dataset_audit_summary.json must NOT be ignored by .gitignore."
    )


def test_build_manifest_cli_dry_run_and_full_run_on_synthetic_tree(
    synthetic_dataset_tree: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dry_out = tmp_path / "dry_out"
    dry_summary = tmp_path / "dry_sub" / "summary.json"

    rc_dry = build_manifest.main(
        [
            "--dataset-root",
            str(synthetic_dataset_tree),
            "--output-dir",
            str(dry_out),
            "--summary-path",
            str(dry_summary),
            "--dry-run",
            "--min-heavy-sessions",
            "0",
        ]
    )
    dry_captured = capsys.readouterr()
    assert rc_dry == 0
    assert ".jpg" not in dry_captured.out.lower()
    assert not dry_out.exists()
    assert not dry_summary.parent.exists()

    out_dir = tmp_path / "out"
    summary_path = tmp_path / "summary.json"

    rc_full = build_manifest.main(
        [
            "--dataset-root",
            str(synthetic_dataset_tree),
            "--output-dir",
            str(out_dir),
            "--summary-path",
            str(summary_path),
            "--min-heavy-sessions",
            "0",
        ]
    )
    full_captured = capsys.readouterr()
    assert rc_full == 0
    assert ".jpg" not in full_captured.out.lower()

    clean_pq = out_dir / "manifest_clean.parquet"
    quarantine_pq = out_dir / "manifest_quarantine.parquet"
    assert clean_pq.is_file()
    assert quarantine_pq.is_file()
    assert summary_path.is_file()

    summary_raw = summary_path.read_text(encoding="utf-8")
    for forbidden in _FORBIDDEN_LEAK_MARKERS:
        assert forbidden not in summary_raw.lower()

    syn_clean = pd.read_parquet(clean_pq)
    syn_quar = pd.read_parquet(quarantine_pq)
    assert len(syn_clean) == 16
    assert len(syn_quar) == 2
    assert set(syn_clean["fold"].unique().tolist()) == {0, 1, 2, 3, 4}
    assert set(syn_quar["fold"].unique().tolist()) == {-1}
