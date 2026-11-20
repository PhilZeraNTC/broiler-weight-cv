"""Unit tests for Audit §9 forensic decoding and 24-column Parquet manifest building."""

from __future__ import annotations

import math
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.app.ingestion.decoding import (
    ARABIC_TO_ASCII,
    BAND_LIGHT_MAX_EXCLUSIVE_G,
    BAND_MEDIUM_MAX_INCLUSIVE_G,
    LCD_CORRECTIONS,
    MTIME_OFFSET_SEC,
    MULTI_BIRD_FOLDERS,
    SESSION_GAP_SECONDS,
    SEVERE_MISLABEL_FOLDERS,
    WIDE_FOCAL_THRESHOLD_MM,
    DecodedFilename,
    assign_weight_band,
    classify_lens,
    compute_image_id,
    decode_filename,
    parse_folder_weight_g,
)
from backend.app.ingestion.manifest import (
    DERIVED_COLUMNS,
    MANIFEST_COLUMNS,
    RAW_BASE_COLUMNS,
    enforce_manifest_dtypes,
    scan_dataset_records,
    split_clean_and_quarantine,
    write_manifest_parquet,
)
from backend.tests.conftest import make_synthetic_jpeg_bytes


def test_audit_section_9_constants_match_verbatim() -> None:
    assert "٠١٢٣٤٥٦٧٨٩".translate(ARABIC_TO_ASCII) == "0123456789"
    assert MTIME_OFFSET_SEC == 14401
    assert SESSION_GAP_SECONDS == 600.0
    assert frozenset({"417", "757", "823", "1520"}) == MULTI_BIRD_FOLDERS
    assert frozenset({"1351", "1465"}) == SEVERE_MISLABEL_FOLDERS
    assert LCD_CORRECTIONS == {"367": 267}
    assert WIDE_FOCAL_THRESHOLD_MM == 3.0
    assert BAND_LIGHT_MAX_EXCLUSIVE_G == 500
    assert BAND_MEDIUM_MAX_INCLUSIVE_G == 1000
    assert len(RAW_BASE_COLUMNS) == 15
    assert len(DERIVED_COLUMNS) == 9
    assert len(MANIFEST_COLUMNS) == 24
    assert set(MANIFEST_COLUMNS) == set(RAW_BASE_COLUMNS) | set(DERIVED_COLUMNS)


def test_decode_filename_all_patterns_and_cp866_recovery() -> None:
    # 1. CP866 mojibake reversal (folders 547/ and 562/)
    cp866_name = "IMG_٢٠٢٢٠٣٢٨_١٢٠٥٣٠.jpg".encode().decode("cp866")
    res_cp866 = decode_filename(cp866_name)
    assert res_cp866 == DecodedFilename(
        capture_dt_from_name=datetime(2022, 3, 28, 12, 5, 30),
        filename_pattern="img_mojibake_cp866",
        was_cp866_recovered=True,
    )

    # 2. Standard Eastern-Arabic timestamp
    res_arabic = decode_filename("IMG_٢٠٢٢٠٣٢٤_٠١١٠٤١.jpg")
    assert res_arabic == DecodedFilename(
        capture_dt_from_name=datetime(2022, 3, 24, 1, 10, 41),
        filename_pattern="img_arabic",
        was_cp866_recovered=False,
    )

    # 3. Same-second Arabic burst (_1 suffix, case-insensitive extension)
    res_burst = decode_filename("IMG_٢٠٢٢٠٣٢٤_٠١١٠٤١_1.JPG")
    assert res_burst == DecodedFilename(
        capture_dt_from_name=datetime(2022, 3, 24, 1, 10, 41),
        filename_pattern="img_arabic_burst",
        was_cp866_recovered=False,
    )

    # 4. Plain ASCII camera timestamp
    res_ascii = decode_filename("IMG_20220324_011041.jpg")
    assert res_ascii == DecodedFilename(
        capture_dt_from_name=datetime(2022, 3, 24, 1, 10, 41),
        filename_pattern="img_ascii",
        was_cp866_recovered=False,
    )

    # 5. Windows-renamed weight file (requires mtime - 14401 s fallback)
    res_renamed = decode_filename("116 (1).jpg")
    assert res_renamed == DecodedFilename(
        capture_dt_from_name=None,
        filename_pattern="weight_renamed",
        was_cp866_recovered=False,
    )
    assert decode_filename("2093(12).JPG").filename_pattern == "weight_renamed"

    # 6. Unrecognized or invalid calendar date degrades to "other"
    res_other = decode_filename("broiler_sample.jpg")
    assert res_other == DecodedFilename(
        capture_dt_from_name=None,
        filename_pattern="other",
        was_cp866_recovered=False,
    )
    res_bad_date = decode_filename("IMG_20229999_999999.jpg")
    assert res_bad_date == DecodedFilename(
        capture_dt_from_name=None,
        filename_pattern="other",
        was_cp866_recovered=False,
    )


def test_parse_folder_weight_g_and_error_handling() -> None:
    assert parse_folder_weight_g("1340_") == 1340
    assert parse_folder_weight_g("173") == 173
    assert parse_folder_weight_g("weight_267g") == 267

    with pytest.raises(ValueError, match="No numeric weight"):
        parse_folder_weight_g("invalid_folder")


def test_boundary_weight_bands_lens_and_image_id() -> None:
    # DATA-02 boundary contract: <500g, [500, 1000] inclusive, >1000g
    assert assign_weight_band(116) == "<500g"
    assert assign_weight_band(499) == "<500g"
    assert assign_weight_band(500) == "500-1000g"
    assert assign_weight_band(750) == "500-1000g"
    assert assign_weight_band(1000) == "500-1000g"
    assert assign_weight_band(1001) == ">1000g"
    assert assign_weight_band(2093) == ">1000g"

    # Lens classification boundary at 3.0 mm
    assert classify_lens(4.25) == "wide"
    assert classify_lens(3.0) == "wide"
    assert classify_lens(2.99) == "ultrawide"
    assert classify_lens(1.65) == "ultrawide"
    assert classify_lens(None) == "unknown"
    assert classify_lens(math.nan) == "unknown"
    assert classify_lens(math.inf) == "unknown"

    # Deterministic 16-hex SHA-256 prefix
    img_id_a = compute_image_id("171/173/IMG_20220324_020000.jpg")
    img_id_b = compute_image_id("171/173/IMG_20220324_020000.jpg")
    img_id_c = compute_image_id("173/IMG_20220324_020000.jpg")
    assert img_id_a == img_id_b
    assert img_id_a != img_id_c
    assert len(img_id_a) == 16
    int(img_id_a, 16)


def test_scan_dataset_records_synthetic_tree_enforces_audit_section_9(
    synthetic_dataset_tree: Path,
) -> None:
    full_df = scan_dataset_records(synthetic_dataset_tree)
    clean_df, quarantine_df = split_clean_and_quarantine(full_df)

    # 16 clean (15 sessions across 3 weight bands) + 2 quarantined = 18 total; symlink excluded.
    assert len(full_df) == 18
    assert len(clean_df) == 16
    assert clean_df["session_id"].nunique() == 15
    assert len(quarantine_df) == 2
    assert list(full_df.columns) == list(MANIFEST_COLUMNS)
    assert list(clean_df.columns) == list(MANIFEST_COLUMNS)
    assert list(quarantine_df.columns) == list(MANIFEST_COLUMNS)

    # POSIX forward slashes only, zero ".." segments, zero escaped symlink
    for path_val in full_df["rel_path"].tolist():
        assert "\\" not in path_val
        assert ".." not in path_val.split("/")
        assert "escaped_link" not in path_val

    # Level-2 nested folder: 171/173 -> leaf weight 173, was_nested_folder=True
    nested_rows = clean_df.loc[clean_df["folder_chain"] == "171/173"]
    assert len(nested_rows) == 1
    assert bool(nested_rows.iloc[0]["was_nested_folder"]) is True
    assert int(nested_rows.iloc[0]["raw_weight_g"]) == 173
    assert int(nested_rows.iloc[0]["weight_g"]) == 173

    # Audited LCD correction: 367 -> 267 g
    lcd_rows = clean_df.loc[clean_df["folder_chain"] == "367"]
    assert len(lcd_rows) == 1
    assert int(lcd_rows.iloc[0]["raw_weight_g"]) == 367
    assert int(lcd_rows.iloc[0]["weight_g"]) == 267
    assert bool(lcd_rows.iloc[0]["was_lcd_corrected"]) is True
    assert lcd_rows.iloc[0]["weight_band"] == "<500g"

    # Trailing underscore sanitization: 1340_ -> 1340 g
    suffix_rows = clean_df.loc[clean_df["folder_chain"] == "1340_"]
    assert len(suffix_rows) == 1
    assert int(suffix_rows.iloc[0]["raw_weight_g"]) == 1340
    assert int(suffix_rows.iloc[0]["weight_g"]) == 1340
    assert suffix_rows.iloc[0]["weight_band"] == ">1000g"
    assert suffix_rows.iloc[0]["lens_class"] == "ultrawide"

    # CP866 recovery: 547 & 562 recovered to 2022 (never 2024 mtime fallback)
    cp866_rows = clean_df.loc[clean_df["folder_chain"].isin(["547", "562"])]
    assert len(cp866_rows) == 2
    assert cp866_rows["was_cp866_recovered"].all()
    assert (cp866_rows["filename_pattern"] == "img_mojibake_cp866").all()
    assert (cp866_rows["capture_dt"].dt.year == 2022).all()
    assert (full_df["capture_dt"].dt.year != 2024).all()

    # Windows-renamed 116 (1).jpg uses naive local mtime - 14401 s
    renamed_rows = clean_df.loc[clean_df["folder_chain"] == "116"]
    assert len(renamed_rows) == 1
    assert renamed_rows.iloc[0]["filename_pattern"] == "weight_renamed"
    assert renamed_rows.iloc[0]["capture_dt"] == pd.Timestamp("2022-03-24 01:10:41")
    assert renamed_rows.iloc[0]["capture_date"] == date(2022, 3, 24)

    # Burst pair & >600s session split in 250/
    rows_250 = clean_df.loc[clean_df["folder_chain"] == "250"].reset_index(drop=True)
    assert len(rows_250) == 3
    assert rows_250["session_id"].tolist() == ["250__s0", "250__s0", "250__s1"]
    assert rows_250["filename_pattern"].tolist() == [
        "img_arabic",
        "img_arabic_burst",
        "img_arabic",
    ]

    # Quarantine routing with fold == -1 (int8)
    assert clean_df["quarantine_reason"].isna().all()
    assert (quarantine_df["fold"] == -1).all()
    assert quarantine_df["fold"].dtype == np.int8
    q_by_folder = dict(
        zip(quarantine_df["folder_chain"], quarantine_df["quarantine_reason"], strict=True)
    )
    assert q_by_folder == {"417": "multi_bird", "1351": "severe_mislabel"}


def test_session_gap_boundary_exact_600s_and_burst_adjacency(tmp_path: Path) -> None:
    root = tmp_path / "gap_boundary_ds"
    folder = root / "500"
    folder.mkdir(parents=True)
    payload = make_synthetic_jpeg_bytes(
        size=(40, 20), orientation=1, focal_length_mm=4.25, zoom_multiple=1.0
    )

    # Write intentionally out of alphabetical/timestamp order to prove deterministic 3-key sort.
    (folder / "IMG_٢٠٢٢٠٣٢٤_١٠٢٠٠١.jpg").write_bytes(payload)  # t2 (+601.0 s -> s1)
    (folder / "IMG_٢٠٢٢٠٣٢٤_١٠٠٠٠٠_1.jpg").write_bytes(payload)  # t0 burst (0.0 s -> s0)
    (folder / "IMG_٢٠٢٢٠٣٢٤_١٠١٠٠٠.jpg").write_bytes(payload)  # t1 (exact +600.0 s -> s0)
    (folder / "IMG_٢٠٢٢٠٣٢٤_١٠٠٠٠٠.jpg").write_bytes(payload)  # t0 (s0)

    # t3 = 10:30:01.001 via mtime fallback (+600.001 s after t2 -> s2, testing exact float > 600.0)
    fp_t3 = folder / "500 (1).jpg"
    fp_t3.write_bytes(payload)
    target_t3 = datetime(2022, 3, 24, 10, 30, 1, 1000)
    mtime_t3 = (target_t3 + timedelta(seconds=MTIME_OFFSET_SEC)).timestamp()
    os.utime(fp_t3, (mtime_t3, mtime_t3))

    df = scan_dataset_records(root)
    assert len(df) == 5
    assert (df["weight_band"] == "500-1000g").all()

    # Exact deterministic 3-key ordering: ["folder_chain", "capture_dt", "rel_path"]
    assert df["rel_path"].tolist() == [
        "500/IMG_٢٠٢٢٠٣٢٤_١٠٠٠٠٠.jpg",
        "500/IMG_٢٠٢٢٠٣٢٤_١٠٠٠٠٠_1.jpg",
        "500/IMG_٢٠٢٢٠٣٢٤_١٠١٠٠٠.jpg",
        "500/IMG_٢٠٢٢٠٣٢٤_١٠٢٠٠١.jpg",
        "500/500 (1).jpg",
    ]
    assert df["session_id"].tolist() == [
        "500__s0",
        "500__s0",
        "500__s0",
        "500__s1",
        "500__s2",
    ]


def test_empty_and_single_image_dataset_contracts(tmp_path: Path) -> None:
    # Missing dataset root raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        scan_dataset_records(tmp_path / "does_not_exist")

    # Empty directory -> valid typed 0-row 24-column DataFrame
    empty_root = tmp_path / "empty_ds"
    empty_root.mkdir()
    empty_full = scan_dataset_records(empty_root)
    empty_clean, empty_quar = split_clean_and_quarantine(empty_full)

    assert empty_full.empty
    assert empty_clean.empty
    assert empty_quar.empty
    assert list(empty_full.columns) == list(MANIFEST_COLUMNS)
    assert empty_full["weight_g"].dtype == np.int32
    assert empty_full["fold"].dtype == np.int8
    assert empty_full["focal_length_mm"].dtype == np.float64
    assert empty_full["capture_dt"].dtype == np.dtype("datetime64[us]")

    # Empty Parquet write/read round-trip preserves exact PyArrow schema (including date32[day])
    empty_parquet = tmp_path / "empty_clean.parquet"
    write_manifest_parquet(empty_clean, empty_parquet)
    schema = pq.read_schema(empty_parquet)
    assert schema.names == list(MANIFEST_COLUMNS)
    assert schema.field("capture_dt").type == pa.timestamp("us")
    assert schema.field("capture_date").type == pa.date32()
    assert schema.field("quarantine_reason").type == pa.string()
    pd.testing.assert_frame_equal(empty_clean, pd.read_parquet(empty_parquet))

    # Single-image folder gets session_id = "<folder_chain>__s0"
    single_root = tmp_path / "single_ds"
    (single_root / "999").mkdir(parents=True)
    (single_root / "999" / "IMG_20220324_080000.jpg").write_bytes(
        make_synthetic_jpeg_bytes(size=(30, 20), orientation=1, focal_length_mm=4.25)
    )
    single_df = scan_dataset_records(single_root)
    assert len(single_df) == 1
    assert single_df.iloc[0]["session_id"] == "999__s0"
    assert single_df.iloc[0]["weight_band"] == "500-1000g"


def test_symlink_file_and_directory_escape_rejection(tmp_path: Path) -> None:
    external_dir = tmp_path / "outside_dir"
    external_dir.mkdir()
    payload = make_synthetic_jpeg_bytes(size=(30, 20), orientation=1, focal_length_mm=4.25)
    outside_file = external_dir / "IMG_20220324_090000.jpg"
    outside_file.write_bytes(payload)

    dataset_root = tmp_path / "contained_ds"
    valid_dir = dataset_root / "200"
    valid_dir.mkdir(parents=True)
    (valid_dir / "IMG_20220324_080000.jpg").write_bytes(payload)
    # Symlinked file inside valid dir
    (valid_dir / "IMG_20220324_080500.jpg").symlink_to(outside_file)
    # Symlinked directory pointing outside dataset_root
    (dataset_root / "300").symlink_to(external_dir, target_is_directory=True)

    df = scan_dataset_records(dataset_root)
    assert len(df) == 1
    assert df["rel_path"].tolist() == ["200/IMG_20220324_080000.jpg"]


def test_manifest_24_column_parquet_schema_and_precision_roundtrip(
    synthetic_dataset_tree: Path,
    tmp_path: Path,
) -> None:
    full_df = scan_dataset_records(synthetic_dataset_tree)
    clean_df, quarantine_df = split_clean_and_quarantine(full_df)

    clean_path = tmp_path / "nested_out" / "manifest_clean.parquet"
    quar_path = tmp_path / "nested_out" / "manifest_quarantine.parquet"
    write_manifest_parquet(clean_df, clean_path)
    write_manifest_parquet(quarantine_df, quar_path)

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

    for parquet_path, original_df in [(clean_path, clean_df), (quar_path, quarantine_df)]:
        schema = pq.read_schema(parquet_path)
        assert schema.names == list(MANIFEST_COLUMNS)
        for col_name, expected_type in expected_arrow_types.items():
            assert schema.field(col_name).type == expected_type, (
                f"{parquet_path.name}:{col_name} expected {expected_type}, "
                f"got {schema.field(col_name).type}"
            )
        reloaded = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(original_df, reloaded)
        assert reloaded["capture_dt"].dt.tz is None
        assert all(
            isinstance(v, date) and not isinstance(v, datetime) for v in reloaded["capture_date"]
        )

    # Exact float64 optical precision (no float32 drift on 1.65 / 0.6)
    reloaded_clean = pd.read_parquet(clean_path)
    uw_row = reloaded_clean.loc[reloaded_clean["folder_chain"] == "1340_"].iloc[0]
    assert uw_row["focal_length_mm"] == 1.65
    assert uw_row["zoom_multiple"] == 0.6

    # Schema guard rejects missing columns and timezone-aware capture_dt
    with pytest.raises(ValueError, match="Missing required manifest columns"):
        enforce_manifest_dtypes(clean_df.drop(columns=["fold"]))

    tz_bad = clean_df.copy()
    tz_bad["capture_dt"] = tz_bad["capture_dt"].dt.tz_localize(UTC)
    with pytest.raises(ValueError, match="timezone-naive"):
        enforce_manifest_dtypes(tz_bad)
