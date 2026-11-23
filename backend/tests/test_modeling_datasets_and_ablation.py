"""Dataset hygiene, 4-condition ablation contract, leakage rejection, and AST tests (Plan 03-01)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.config import REPO_ROOT
from backend.app.ingestion.decoding import assign_weight_band
from backend.app.modeling.datasets import (
    ABLATION_CONDITION_ORDER,
    DIMENSIONLESS_SHAPE_COLUMNS,
    NAIVE_FXZ_5_COLUMNS,
    NON_FEATURE_AUDIT_COLUMNS,
    OPTICAL_NORM_5_COLUMNS,
    build_ablation_feature_specs,
    load_modeling_dataset,
)
from backend.app.pdi.config import PDIConfig
from backend.app.pdi.features import (
    BANNED_FEATURE_COLUMNS,
    CHROMATIC_TEXTURE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MORPHOMETRIC_FEATURE_COLUMNS,
    OPTICAL_FEATURE_COLUMNS,
)


def _write_synthetic_modeling_parquets(
    tmp_path: Path,
    *,
    sessions_per_band: int = 20,
    imgs_per_session: int = 4,
    seed: int = 42,
    pdi_config_hash: str | None = None,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    """Build reusable 5-fold zero-overlap synthetic manifest & features Parquet fixtures."""
    rng = np.random.default_rng(seed)
    cfg_hash = pdi_config_hash if pdi_config_hash is not None else PDIConfig().config_hash()

    band_specs: list[tuple[str, int, int]] = [
        ("light", 150, 480),
        ("medium", 520, 980),
        ("heavy", 1100, 1950),
    ]

    manifest_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    img_counter = 0

    for band_tag, w_lo, w_hi in band_specs:
        for s_idx in range(sessions_per_band):
            fold = int(s_idx % 5)
            weight_g = int(rng.integers(w_lo, w_hi + 1))
            weight_band = assign_weight_band(weight_g)
            session_id = f"{band_tag}_s{s_idx:03d}"

            # Realistic physical bird area scaling smoothly with weight^(2/3).
            true_phys_area = 280.0 * (float(weight_g) ** (2.0 / 3.0)) * float(
                rng.uniform(0.92, 1.08)
            )
            scene_is_bucket = bool(s_idx % 3 == 0)
            scene_type = "bucket_dorsal" if scene_is_bucket else "scale_or_litter"

            for img_idx in range(imgs_per_session):
                is_uw = bool(img_idx % 2 == 1)
                focal_mm = 1.65 if is_uw else 4.25
                zoom = 0.6 if is_uw else 1.0
                lens_class = "ultrawide" if is_uw else "wide"

                area_px = true_phys_area * (zoom ** 2)
                sqrt_area_px = math.sqrt(area_px)
                perimeter_px = sqrt_area_px * 3.9
                major_axis_px = sqrt_area_px * 1.45
                minor_axis_px = sqrt_area_px * 0.92
                convex_area_px = area_px * 1.08
                sensor_px = (3264.0 * 1472.0) if is_uw else (4080.0 * 1840.0)
                area_frac = area_px / sensor_px

                image_id = f"{img_counter:016x}"
                img_counter += 1

                manifest_rows.append(
                    {
                        "image_id": image_id,
                        "rel_path": f"synthetic_folder/{image_id}",
                        "fold": np.int8(fold),
                        "session_id": session_id,
                        "weight_g": np.int32(weight_g),
                        "weight_band": weight_band,
                        "lens_class": lens_class,
                    }
                )

                feat_row: dict[str, object] = {
                    "image_id": image_id,
                    "fold": np.int8(fold),
                    "session_id": session_id,
                    "weight_g": np.int32(weight_g),
                    "weight_band": weight_band,
                    "lens_class": lens_class,
                    "scene_type": scene_type,
                    "scene_is_bucket": scene_is_bucket,
                    "lcd_excluded_px": np.int32(0 if scene_is_bucket else 420),
                    "segmentation_fallback": False,
                    "pdi_config_hash": cfg_hash,
                    # Morphometrics (14)
                    "area_px": area_px,
                    "perimeter_px": perimeter_px,
                    "major_axis_px": major_axis_px,
                    "minor_axis_px": minor_axis_px,
                    "convex_area_px": convex_area_px,
                    "area_frac": area_frac,
                    "aspect_ratio_ellipse": major_axis_px / minor_axis_px,
                    "eccentricity": 0.72,
                    "solidity": area_px / convex_area_px,
                    "circularity": 0.78,
                    "extent_rotrect": 0.74,
                    "log_hu_1": 0.65,
                    "log_hu_2": 1.85,
                    "log_hu_3": 3.10,
                    # Optical & calibrated (9)
                    "focal_length_mm": focal_mm,
                    "zoom_multiple": zoom,
                    "exif_optical_present": 1.0,
                    "area_optical_norm": area_px / (zoom ** 2),
                    "sqrt_area_optical_norm": math.sqrt(area_px / (zoom ** 2)),
                    "perim_optical_norm": perimeter_px / zoom,
                    "major_axis_optical_norm": major_axis_px / zoom,
                    "minor_axis_optical_norm": minor_axis_px / zoom,
                    "area_frac_optical_norm": area_frac / (zoom ** 2),
                }
                # Chromatic & eroded interior texture (26)
                for col in CHROMATIC_TEXTURE_FEATURE_COLUMNS:
                    if col.startswith("lab_a"):
                        feat_row[col] = 124.0 + 0.01 * float(weight_g) + float(rng.normal(0.0, 0.5))
                    else:
                        feat_row[col] = float(rng.uniform(0.1, 120.0))

                feature_rows.append(feat_row)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df["fold"] = manifest_df["fold"].astype("int8")
    manifest_df["weight_g"] = manifest_df["weight_g"].astype("int32")

    features_cols = [
        "image_id",
        "fold",
        "session_id",
        "weight_g",
        "weight_band",
        "lens_class",
        "scene_type",
        "scene_is_bucket",
        "lcd_excluded_px",
        "segmentation_fallback",
        "pdi_config_hash",
        *FEATURE_COLUMNS,
    ]
    features_df = pd.DataFrame(feature_rows, columns=features_cols)
    features_df["fold"] = features_df["fold"].astype("int8")
    features_df["weight_g"] = features_df["weight_g"].astype("int32")
    features_df["lcd_excluded_px"] = features_df["lcd_excluded_px"].astype("int32")

    manifest_path = tmp_path / "manifest_clean.parquet"
    features_path = tmp_path / "features_clean.parquet"
    manifest_df.to_parquet(manifest_path, engine="pyarrow", index=False)
    features_df.to_parquet(features_path, engine="pyarrow", index=False)
    return manifest_path, features_path, manifest_df, features_df


def test_ablation_feature_specs_counts_disjointness_and_hygiene() -> None:
    specs = build_ablation_feature_specs()
    assert tuple(specs.keys()) == ABLATION_CONDITION_ORDER
    assert len(specs["C1_uncalibrated_pdi"]) == 40
    assert len(specs["C2_physics_normalized_strict"]) == 39
    assert len(specs["C3_physics_plus_exif_meta"]) == 49
    assert len(specs["C4_naive_double_count_fxz"]) == 39

    assert len(DIMENSIONLESS_SHAPE_COLUMNS) == 8
    assert len(OPTICAL_NORM_5_COLUMNS) == 5
    assert len(NAIVE_FXZ_5_COLUMNS) == 5
    assert set(DIMENSIONLESS_SHAPE_COLUMNS).issubset(MORPHOMETRIC_FEATURE_COLUMNS)
    assert set(OPTICAL_NORM_5_COLUMNS).issubset(OPTICAL_FEATURE_COLUMNS)

    prohibited = BANNED_FEATURE_COLUMNS | NON_FEATURE_AUDIT_COLUMNS | {
        "scene_type",
        "scene_is_bucket",
        "lcd_excluded_px",
        "segmentation_fallback",
        "lens_class",
        "fold",
        "session_id",
        "rel_path",
    }

    for cond, cols in specs.items():
        assert len(cols) == len(set(cols)), f"Duplicate feature names in {cond}"
        assert prohibited.isdisjoint(cols), f"Prohibited column leaked into {cond}"

    # C1 excludes all optical columns (baseline WITHOUT EXIF).
    assert set(specs["C1_uncalibrated_pdi"]).isdisjoint(OPTICAL_FEATURE_COLUMNS)

    # C2 excludes all raw EXIF and uncalibrated pixel/area_frac columns (D-04).
    c2_forbidden = {
        "focal_length_mm",
        "zoom_multiple",
        "exif_optical_present",
        "area_px",
        "perimeter_px",
        "major_axis_px",
        "minor_axis_px",
        "convex_area_px",
        "area_frac",
        "area_frac_optical_norm",
    }
    assert set(specs["C2_physics_normalized_strict"]).isdisjoint(c2_forbidden)
    assert set(OPTICAL_NORM_5_COLUMNS).issubset(specs["C2_physics_normalized_strict"])

    # C3 equals the exact 49-col frozen FEATURE_COLUMNS.
    assert specs["C3_physics_plus_exif_meta"] == FEATURE_COLUMNS

    # C4 replaces the 5 optical_norm columns strictly with the 5 offline naive fxz siblings.
    assert set(NAIVE_FXZ_5_COLUMNS).issubset(specs["C4_naive_double_count_fxz"])
    assert set(specs["C4_naive_double_count_fxz"]).isdisjoint(OPTICAL_NORM_5_COLUMNS)
    assert set(specs["C4_naive_double_count_fxz"]).isdisjoint(c2_forbidden)


def test_load_modeling_dataset_synthetic_valid_and_naive_fxz_math(tmp_path: Path) -> None:
    manifest_path, features_path, _, _ = _write_synthetic_modeling_parquets(tmp_path)
    ds = load_modeling_dataset(manifest_path=manifest_path, features_path=features_path)

    assert len(ds.df) == 240
    assert ds.pdi_config_hash == PDIConfig().config_hash()
    assert ds.fold_audit["cross_fold_session_overlap"] == 0
    assert "rel_path" not in ds.df.columns
    assert BANNED_FEATURE_COLUMNS.isdisjoint(ds.df.columns)

    fxz = ds.df["focal_length_mm"].to_numpy(dtype=np.float64) * ds.df["zoom_multiple"].to_numpy(
        dtype=np.float64
    )
    expected_area_naive = ds.df["area_px"].to_numpy(dtype=np.float64) / (fxz ** 2)
    np.testing.assert_allclose(ds.df["area_naive_fxz"].to_numpy(), expected_area_naive, rtol=1e-12)
    np.testing.assert_allclose(
        ds.df["sqrt_area_naive_fxz"].to_numpy(),
        np.sqrt(expected_area_naive),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        ds.df["perim_naive_fxz"].to_numpy(),
        ds.df["perimeter_px"].to_numpy(dtype=np.float64) / fxz,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        ds.df["major_naive_fxz"].to_numpy(),
        ds.df["major_axis_px"].to_numpy(dtype=np.float64) / fxz,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        ds.df["minor_naive_fxz"].to_numpy(),
        ds.df["minor_axis_px"].to_numpy(dtype=np.float64) / fxz,
        rtol=1e-12,
    )


def test_load_modeling_dataset_rejects_mismatched_hash_misalignment_nan_banned_and_leakage(
    tmp_path: Path,
) -> None:
    # 1. Mismatched pdi_config_hash
    bad_hash_dir = tmp_path / "bad_hash"
    bad_hash_dir.mkdir()
    m_path, f_path, _, _ = _write_synthetic_modeling_parquets(
        bad_hash_dir,
        pdi_config_hash="0000000000000000",
    )
    with pytest.raises(ValueError, match="hash"):
        load_modeling_dataset(manifest_path=m_path, features_path=f_path)

    # 2. Misaligned image_id row order
    misaligned_dir = tmp_path / "misaligned"
    misaligned_dir.mkdir()
    m_path, f_path, _, f_df = _write_synthetic_modeling_parquets(misaligned_dir)
    reordered = pd.concat([f_df.iloc[1:2], f_df.iloc[0:1], f_df.iloc[2:]], ignore_index=True)
    reordered.to_parquet(f_path, engine="pyarrow", index=False)
    with pytest.raises((ValueError, RuntimeError), match="alignment"):
        load_modeling_dataset(manifest_path=m_path, features_path=f_path)

    # 3. Injected NaN and Inf features
    for label, bad_val in (("nan", math.nan), ("inf", math.inf)):
        nonfinite_dir = tmp_path / f"bad_{label}"
        nonfinite_dir.mkdir()
        m_path, f_path, _, f_df = _write_synthetic_modeling_parquets(nonfinite_dir)
        tampered = f_df.copy()
        tampered.loc[tampered.index[0], "area_px"] = bad_val
        tampered.to_parquet(f_path, engine="pyarrow", index=False)
        with pytest.raises(ValueError, match="Non-finite"):
            load_modeling_dataset(manifest_path=m_path, features_path=f_path)

    # 4. Injected BANNED_FEATURE_COLUMNS column ('oriented_w')
    banned_dir = tmp_path / "banned_col"
    banned_dir.mkdir()
    m_path, f_path, _, f_df = _write_synthetic_modeling_parquets(banned_dir)
    tampered_banned = f_df.copy()
    tampered_banned.loc[:, "oriented_w"] = 1840
    tampered_banned.to_parquet(f_path, engine="pyarrow", index=False)
    with pytest.raises(ValueError, match="PROIBIÇÃO 2"):
        load_modeling_dataset(manifest_path=m_path, features_path=f_path)

    # 5. Injected cross-fold session_id overlap (PROIBIÇÃO 3)
    leak_dir = tmp_path / "session_leak"
    leak_dir.mkdir()
    m_path, f_path, m_df, f_df = _write_synthetic_modeling_parquets(leak_dir)
    m_leaked = m_df.copy()
    f_leaked = f_df.copy()
    orig_fold = int(m_leaked.loc[m_leaked.index[0], "fold"])
    new_fold = np.int8((orig_fold + 1) % 5)
    m_leaked.loc[m_leaked.index[0], "fold"] = new_fold
    f_leaked.loc[f_leaked.index[0], "fold"] = new_fold
    m_leaked.to_parquet(m_path, engine="pyarrow", index=False)
    f_leaked.to_parquet(f_path, engine="pyarrow", index=False)
    with pytest.raises(RuntimeError, match="Session leakage"):
        load_modeling_dataset(manifest_path=m_path, features_path=f_path)


def test_load_modeling_dataset_materialized_parquets_contract() -> None:
    ds = load_modeling_dataset()

    assert len(ds.df) == 1738
    assert ds.pdi_config_hash == "43a3cfaa3208019e"
    assert ds.fold_audit["cross_fold_session_overlap"] == 0
    assert "rel_path" not in ds.df.columns
    assert BANNED_FEATURE_COLUMNS.isdisjoint(ds.df.columns)
    assert int(ds.df["session_id"].nunique()) == 188
    assert int(ds.df["weight_g"].nunique()) == 160
    assert [len(ds.ablation_columns[c]) for c in ABLATION_CONDITION_ORDER] == [40, 39, 49, 39]

    fxz = ds.df["focal_length_mm"].to_numpy(dtype=np.float64) * ds.df["zoom_multiple"].to_numpy(
        dtype=np.float64
    )
    expected_area_naive = ds.df["area_px"].to_numpy(dtype=np.float64) / (fxz ** 2)
    np.testing.assert_allclose(
        ds.df["area_naive_fxz"].to_numpy(dtype=np.float64),
        expected_area_naive,
        rtol=1e-12,
        atol=0.0,
    )


def test_ast_scan_prohibits_dl_ocr_and_banned_splitters_in_modeling() -> None:
    modeling_dir = REPO_ROOT / "backend" / "app" / "modeling"
    py_files = sorted(modeling_dir.rglob("*.py"))
    assert len(py_files) >= 4, f"Expected modeling modules under {modeling_dir}"

    banned_dl_ocr_modules = {
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "keras",
        "ultralytics",
        "timm",
        "onnxruntime",
        "paddleocr",
        "easyocr",
    }
    banned_splitters = {
        "train_test_split",
        "KFold",
        "StratifiedKFold",
        "ShuffleSplit",
        "StratifiedShuffleSplit",
        "GroupShuffleSplit",
    }

    dl_violations: list[str] = []
    splitter_violations: list[str] = []

    for py_file in py_files:
        rel_file = py_file.relative_to(REPO_ROOT)
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_pkg = alias.name.split(".")[0]
                    if top_pkg in banned_dl_ocr_modules:
                        dl_violations.append(f"{rel_file}:{node.lineno}:import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_pkg = node.module.split(".")[0]
                    if top_pkg in banned_dl_ocr_modules:
                        dl_violations.append(f"{rel_file}:{node.lineno}:from {node.module}")
                for alias in node.names:
                    if alias.name in banned_splitters:
                        splitter_violations.append(f"{rel_file}:{node.lineno}:import {alias.name}")
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in banned_splitters:
                    splitter_violations.append(f"{rel_file}:{node.lineno}:call {func.id}")
                elif isinstance(func, ast.Attribute) and func.attr in banned_splitters:
                    splitter_violations.append(f"{rel_file}:{node.lineno}:call {func.attr}")

    assert not dl_violations, f"PROIBIÇÃO 1 violation — banned DL/OCR imports: {dl_violations}"
    assert not splitter_violations, (
        f"PROIBIÇÃO 3 violation — banned splitters found: {splitter_violations}"
    )
