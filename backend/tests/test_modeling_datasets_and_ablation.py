"""Dataset hygiene, ablation, evaluation edge-cases, counterfactual gate & AST tests (03-01/02)."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

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
from backend.app.modeling.evaluation import (
    COUNTERFACTUAL_FOCAL_MEDIAN_THRESHOLD_PCT,
    COUNTERFACTUAL_PHYS_MEDIAN_THRESHOLD_PCT,
    COUNTERFACTUAL_PHYS_P90_THRESHOLD_PCT,
    LENS_CLASS_ORDER,
    SCENE_TYPE_ORDER,
    WEIGHT_BAND_ORDER,
    CandidateEvaluation,
    compute_regression_metrics,
    compute_session_level_metrics,
    compute_stratified_metrics,
    compute_within_session_dual_lens_consistency,
    evaluate_anti_confounding_gate,
    evaluate_single_candidate,
    select_winning_candidate,
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
    file_names = {p.name for p in py_files}
    assert {"wrappers.py", "datasets.py", "models.py", "age_stage.py", "evaluation.py"}.issubset(
        file_names
    ), f"Expected core modeling modules under {modeling_dir}, found {sorted(file_names)}"

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


def _make_stub_candidate(
    *,
    model_name: str,
    ablation_condition: str,
    balanced_band_mape_pct: float,
    global_mape_pct: float = 35.0,
    global_mae_g: float = 150.0,
    dual_lens_median_abs_diff_g: float | None = 60.0,
    lens_confounding_detected: bool = False,
    eligible_for_production: bool | None = None,
) -> CandidateEvaluation:
    """Create a lightweight CandidateEvaluation stub for deterministic selector testing."""
    is_eligible = (
        not lens_confounding_detected
        if eligible_for_production is None
        else eligible_for_production
    )
    return CandidateEvaluation(
        model_name=model_name,
        ablation_condition=ablation_condition,
        feature_names=("area_optical_norm",),
        weighting_mode="session_band_capped_5x",
        duan_smearing_enabled=False,
        oof_predictions=np.array([500.0], dtype=np.float64),
        fold_pipelines={},
        global_metrics={
            "n": 100,
            "mae_g": global_mae_g,
            "rmse_g": global_mae_g * 1.4,
            "mape_pct": global_mape_pct,
            "r2": 0.55,
            "session_averaged_r2": 0.68,
            "mbe_g": -10.0,
            "median_abs_err_g": 80.0,
            "p68_abs_err_g": 140.0,
            "p90_abs_err_g": 300.0,
        },
        session_level_metrics={
            "n_sessions": 20,
            "session_averaged_r2": 0.68,
            "mae_g": global_mae_g * 0.9,
            "rmse_g": global_mae_g * 1.2,
            "mape_pct": global_mape_pct * 0.9,
            "mbe_g": -10.0,
        },
        balanced_band_mape_pct=balanced_band_mape_pct,
        per_fold=[],
        by_weight_band={},
        by_lens={},
        by_scene={},
        by_band_and_scene={},
        by_band_and_lens={},
        band_error_intervals={},
        dual_lens_consistency={
            "n_mixed_lens_sessions": 10 if dual_lens_median_abs_diff_g is not None else 0,
            "mean_abs_diff_g": dual_lens_median_abs_diff_g,
            "median_abs_diff_g": dual_lens_median_abs_diff_g,
            "mean_rel_diff_pct": 15.0 if dual_lens_median_abs_diff_g is not None else None,
            "median_rel_diff_pct": 15.0 if dual_lens_median_abs_diff_g is not None else None,
            "mean_signed_wide_minus_uw_g": 0.0 if dual_lens_median_abs_diff_g is not None else None,
            "median_signed_wide_minus_uw_g": (
                0.0 if dual_lens_median_abs_diff_g is not None else None
            ),
        },
        counterfactual_lens_flip={
            "n_evaluated": 100,
            "focal_only": {
                "median_shift_pct": 0.0,
                "mean_shift_pct": 0.0,
                "p90_shift_pct": 0.0,
            },
            "physical_lens_swap": {
                "median_shift_pct": 0.0,
                "mean_shift_pct": 0.0,
                "p90_shift_pct": 0.0,
            },
        },
        lens_confounding_detected=lens_confounding_detected,
        confounding_reasons=["synthetic flag"] if lens_confounding_detected else [],
        eligible_for_production=is_eligible,
        is_winner=False,
    )


def _write_synthetic_confounded_parquets(
    tmp_path: Path,
    *,
    seed: int = 123,
) -> tuple[Path, Path]:
    """Build a compact 5-fold lens-confounded dataset (150 rows, 25 sessions).

    Light birds (<500g) are predominantly wide (4.25 mm / 1.0x) and heavy birds (>1000g)
    are predominantly ultrawide (1.65 mm / 0.6x), with mixed-lens sessions across folds
    and realistic handheld distance jitter on physical area so C3/C4 exploit the lens-ID
    shortcut while C2 remains strictly lens-invariant (0.00% counterfactual shift).
    """
    rng = np.random.default_rng(seed)
    cfg_hash = PDIConfig().config_hash()

    # 25 sessions total across 5 folds (5 sessions/fold: 2 light, 1 medium, 2 heavy) x 6 imgs = 150.
    session_plan: list[tuple[str, int, int, list[bool]]] = []
    mostly_wide_a = [False, False, False, False, False, True]
    mostly_wide_b = [False, False, False, False, True, False]
    balanced_lens = [False, False, False, True, True, True]
    mostly_uw_a = [False, True, True, True, True, True]
    mostly_uw_b = [True, True, True, True, True, True]
    for fold in range(5):
        # 2 light sessions per fold: mostly wide (5W/1UW) -> mixed-lens + <500g UW present.
        session_plan.append((f"light_a_f{fold}", fold, int(rng.integers(170, 320)), mostly_wide_a))
        session_plan.append((f"light_b_f{fold}", fold, int(rng.integers(220, 440)), mostly_wide_b))
        # 1 medium session per fold: balanced 3W / 3UW.
        session_plan.append((f"med_f{fold}", fold, int(rng.integers(560, 880)), balanced_lens))
        # 2 heavy sessions per fold: mostly ultrawide (1W/5UW and 0W/6UW).
        session_plan.append((f"heavy_a_f{fold}", fold, int(rng.integers(1250, 1650)), mostly_uw_a))
        session_plan.append((f"heavy_b_f{fold}", fold, int(rng.integers(1500, 1980)), mostly_uw_b))

    manifest_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    img_counter = 0

    for session_id, fold, weight_g, uw_flags in session_plan:
        weight_band = assign_weight_band(weight_g)
        scene_is_bucket = bool(session_id.startswith("heavy_b") or session_id.startswith("light_b"))
        scene_type = "bucket_dorsal" if scene_is_bucket else "scale_or_litter"

        for is_uw in uw_flags:
            focal_mm = 1.65 if is_uw else 4.25
            zoom = 0.6 if is_uw else 1.0
            lens_class = "ultrawide" if is_uw else "wide"

            # Realistic handheld height Z jitter (~35%) makes lens-ID shortcut tempting to C3/C4.
            z_jitter = float(np.exp(rng.normal(0.0, 0.35)))
            phys_area = 260.0 * (float(weight_g) ** (2.0 / 3.0)) * z_jitter
            area_px = phys_area * (zoom ** 2)
            sqrt_area_px = math.sqrt(area_px)
            perimeter_px = sqrt_area_px * 3.85
            major_axis_px = sqrt_area_px * 1.42
            minor_axis_px = sqrt_area_px * 0.91
            convex_area_px = area_px * 1.07
            sensor_px = (3264.0 * 1472.0) if is_uw else (4080.0 * 1840.0)
            area_frac = area_px / sensor_px

            image_id = f"cf{img_counter:014x}"
            img_counter += 1

            manifest_rows.append(
                {
                    "image_id": image_id,
                    "rel_path": f"synthetic_cf/{image_id}",
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
                "lcd_excluded_px": np.int32(0 if scene_is_bucket else 350),
                "segmentation_fallback": False,
                "pdi_config_hash": cfg_hash,
                "area_px": area_px,
                "perimeter_px": perimeter_px,
                "major_axis_px": major_axis_px,
                "minor_axis_px": minor_axis_px,
                "convex_area_px": convex_area_px,
                "area_frac": area_frac,
                "aspect_ratio_ellipse": major_axis_px / minor_axis_px,
                "eccentricity": float(rng.uniform(0.60, 0.82)),
                "solidity": area_px / convex_area_px,
                "circularity": float(rng.uniform(0.68, 0.86)),
                "extent_rotrect": float(rng.uniform(0.65, 0.80)),
                "log_hu_1": float(rng.uniform(0.55, 0.78)),
                "log_hu_2": float(rng.uniform(1.60, 2.20)),
                "log_hu_3": float(rng.uniform(2.80, 3.50)),
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
            for col in CHROMATIC_TEXTURE_FEATURE_COLUMNS:
                if col == "lab_a_mean":
                    feat_row[col] = 124.0 + 0.004 * float(weight_g) + float(rng.normal(0.0, 2.0))
                else:
                    feat_row[col] = float(rng.uniform(5.0, 100.0))

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
    return manifest_path, features_path


def test_compute_regression_metrics_normal_empty_single_constant_and_validation() -> None:
    # Empty slice (MOD-02 empty edge): zero ZeroDivisionError, None metric fields.
    empty = compute_regression_metrics([], [])
    assert empty["n"] == 0
    for key in (
        "mae_g",
        "rmse_g",
        "mape_pct",
        "r2",
        "mbe_g",
        "median_abs_err_g",
        "p68_abs_err_g",
        "p90_abs_err_g",
    ):
        assert empty[key] is None

    # Single-row slice: finite error metrics, r2 is None.
    single = compute_regression_metrics([500.0], [520.0])
    assert single["n"] == 1
    assert single["mae_g"] == 20.0
    assert single["rmse_g"] == 20.0
    assert single["mape_pct"] == 4.0
    assert single["mbe_g"] == 20.0
    assert single["median_abs_err_g"] == 20.0
    assert single["p68_abs_err_g"] == 20.0
    assert single["p90_abs_err_g"] == 20.0
    assert single["r2"] is None

    # Zero-variance y_true: r2 is None, finite error metrics.
    const = compute_regression_metrics([500.0, 500.0], [510.0, 490.0])
    assert const["n"] == 2
    assert const["r2"] is None
    assert const["mae_g"] == 10.0
    assert const["mbe_g"] == 0.0

    # Normal unweighted slice.
    y_true = np.array([200.0, 500.0, 1000.0, 1500.0], dtype=np.float64)
    y_pred = np.array([220.0, 450.0, 1100.0, 1350.0], dtype=np.float64)
    normal = compute_regression_metrics(y_true, y_pred)
    assert normal["n"] == 4
    assert normal["mae_g"] == pytest.approx(80.0, abs=1e-4)
    assert normal["mbe_g"] == pytest.approx(-20.0, abs=1e-4)
    assert normal["mape_pct"] == pytest.approx(10.0, abs=1e-4)
    assert normal["r2"] is not None and normal["r2"] > 0.95

    # Invalid inputs rejected cleanly.
    with pytest.raises(ValueError, match="equal length"):
        compute_regression_metrics([100.0, 200.0], [100.0])
    with pytest.raises(ValueError, match="boolean"):
        compute_regression_metrics([True, False], [1.0, 0.0])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-finite"):
        compute_regression_metrics([500.0, math.nan], [500.0, 500.0])


def test_compute_session_level_metrics_improves_r2_on_frame_noise_and_handles_empty() -> None:
    empty_df = pd.DataFrame(
        {
            "session_id": pd.Series([], dtype="str"),
            "weight_g": pd.Series([], dtype="float64"),
        }
    )
    empty_res = compute_session_level_metrics(empty_df, [])
    assert empty_res["n_sessions"] == 0
    assert empty_res["session_averaged_r2"] is None
    assert empty_res["mae_g"] is None

    # 5 sessions with symmetric per-frame noise averaging out completely at session level.
    rows: list[dict[str, object]] = []
    preds: list[float] = []
    for idx, weight in enumerate((200, 450, 750, 1200, 1800)):
        sid = f"sess_{idx}"
        for noise in (-90.0, 90.0, -60.0, 60.0):
            rows.append({"session_id": sid, "weight_g": weight})
            preds.append(float(weight) + noise)

    # Deliberately use non-RangeIndex to verify positional groupby indexing safety.
    df = pd.DataFrame(rows, index=range(100, 100 + len(rows)))
    frame_metrics = compute_regression_metrics(df["weight_g"].to_numpy(), preds)
    session_metrics = compute_session_level_metrics(df, preds)

    assert session_metrics["n_sessions"] == 5
    assert session_metrics["mae_g"] == 0.0
    assert session_metrics["mbe_g"] == 0.0
    assert session_metrics["session_averaged_r2"] == 1.0
    assert frame_metrics["r2"] is not None
    assert session_metrics["session_averaged_r2"] > frame_metrics["r2"]


def test_compute_stratified_metrics_weight_band_adjacency_and_empty_slices() -> None:
    # Exact Phase 1 assign_weight_band() boundary points: 499 / 500 / 1000 / 1001.
    boundary_weights = [499, 500, 1000, 1001]
    assert [assign_weight_band(w) for w in boundary_weights] == [
        "<500g",
        "500-1000g",
        "500-1000g",
        ">1000g",
    ]
    df = pd.DataFrame(
        {
            "session_id": ["s0", "s1", "s2", "s3"],
            "fold": np.array([0, 1, 2, 3], dtype=np.int8),
            "weight_g": np.array(boundary_weights, dtype=np.int32),
            "lens_class": ["wide", "wide", "ultrawide", "ultrawide"],
            "scene_type": ["bucket_dorsal", "scale_or_litter", "scale_or_litter", "bucket_dorsal"],
        }
    )
    preds = np.array([548.9, 550.0, 900.0, 1101.1], dtype=np.float64)
    strat = compute_stratified_metrics(df, preds)

    assert tuple(strat["by_weight_band"].keys()) == WEIGHT_BAND_ORDER
    assert tuple(strat["by_lens"].keys()) == LENS_CLASS_ORDER
    assert tuple(strat["by_scene"].keys()) == SCENE_TYPE_ORDER
    assert strat["by_weight_band"]["<500g"]["n"] == 1
    assert strat["by_weight_band"]["500-1000g"]["n"] == 2
    assert strat["by_weight_band"][">1000g"]["n"] == 1

    # Fold 4 has 0 samples -> safe empty slice with None metrics, zero ZeroDivisionError.
    assert len(strat["per_fold"]) == 5
    assert strat["per_fold"][4]["n"] == 0
    assert strat["per_fold"][4]["n_sessions"] == 0
    assert strat["per_fold"][4]["mae_g"] is None

    # Empty cross-slice (<500g__ultrawide) returns n=0 and None metrics.
    assert strat["by_band_and_lens"]["<500g__ultrawide"]["n"] == 0
    assert strat["by_band_and_lens"]["<500g__ultrawide"]["mae_g"] is None

    # Exact 10% error on every point -> balanced_band_mape_pct == 10.0%.
    assert strat["balanced_band_mape_pct"] == pytest.approx(10.0, abs=1e-4)

    # D-07 band_error_intervals keys present per weight band.
    expected_interval_keys = {
        "n",
        "mae_g",
        "mape_pct",
        "median_abs_err_g",
        "p68_abs_err_g",
        "p90_abs_err_g",
        "mbe_g",
    }
    for band in WEIGHT_BAND_ORDER:
        assert set(strat["band_error_intervals"][band].keys()) == expected_interval_keys

    # Empty DataFrame safety.
    empty_df = df.iloc[0:0].copy()
    empty_strat = compute_stratified_metrics(empty_df, [])
    assert empty_strat["balanced_band_mape_pct"] is None
    assert all(row["n"] == 0 and row["n_sessions"] == 0 for row in empty_strat["per_fold"])


def test_compute_within_session_dual_lens_consistency_mixed_and_single_lens() -> None:
    # 2 mixed-lens sessions + 2 single-lens sessions (ignored).
    df = pd.DataFrame(
        {
            "session_id": [
                "s_mix1",
                "s_mix1",
                "s_mix1",
                "s_mix2",
                "s_mix2",
                "s_wide_only",
                "s_uw_only",
            ],
            "weight_g": [500, 500, 500, 1000, 1000, 300, 1500],
            "lens_class": ["wide", "wide", "ultrawide", "wide", "ultrawide", "wide", "ultrawide"],
        }
    )
    # s_mix1: mean wide = 520, mean uw = 470 -> signed=+50, abs=50, rel=10%
    # s_mix2: mean wide = 900, mean uw = 1000 -> signed=-100, abs=100, rel=10%
    preds = [510.0, 530.0, 470.0, 900.0, 1000.0, 310.0, 1450.0]
    res = compute_within_session_dual_lens_consistency(df, preds)

    assert res["n_mixed_lens_sessions"] == 2
    assert res["mean_abs_diff_g"] == pytest.approx(75.0, abs=1e-4)
    assert res["median_abs_diff_g"] == pytest.approx(75.0, abs=1e-4)
    assert res["mean_rel_diff_pct"] == pytest.approx(10.0, abs=1e-4)
    assert res["median_rel_diff_pct"] == pytest.approx(10.0, abs=1e-4)
    assert res["mean_signed_wide_minus_uw_g"] == pytest.approx(-25.0, abs=1e-4)
    assert res["median_signed_wide_minus_uw_g"] == pytest.approx(-25.0, abs=1e-4)

    # Zero mixed-lens sessions -> n=0 and all summary metrics None (MOD-02 empty edge).
    single_lens_mask = df["session_id"].isin(["s_wide_only", "s_uw_only"])
    single_lens_df = df.loc[single_lens_mask].reset_index(drop=True)
    none_res = compute_within_session_dual_lens_consistency(single_lens_df, [310.0, 1450.0])
    assert none_res["n_mixed_lens_sessions"] == 0
    for key in (
        "mean_abs_diff_g",
        "median_abs_diff_g",
        "mean_rel_diff_pct",
        "median_rel_diff_pct",
        "mean_signed_wide_minus_uw_g",
        "median_signed_wide_minus_uw_g",
    ):
        assert none_res[key] is None


def test_evaluate_anti_confounding_gate_exact_strict_inequality_boundaries() -> None:
    def _cf(focal_med: float, phys_med: float, phys_p90: float) -> dict[str, Any]:
        return {
            "n_evaluated": 100,
            "focal_only": {
                "median_shift_pct": focal_med,
                "mean_shift_pct": focal_med,
                "p90_shift_pct": focal_med,
            },
            "physical_lens_swap": {
                "median_shift_pct": phys_med,
                "mean_shift_pct": phys_med,
                "p90_shift_pct": phys_p90,
            },
        }

    c2_slices = {"<500g__ultrawide": {"n": 20, "mae_g": 110.0, "mbe_g": 60.0}}

    # C2 passes at exact 0.0 and raises loudly if lens invariance is broken.
    flagged, reasons = evaluate_anti_confounding_gate(
        "C2_physics_normalized_strict", _cf(0.0, 0.0, 0.0), c2_slices, c2_slices
    )
    assert flagged is False and reasons == []
    with pytest.raises(RuntimeError, match="0.0% lens invariance"):
        evaluate_anti_confounding_gate(
            "C2_physics_normalized_strict", _cf(0.0, 0.5, 1.0), c2_slices, c2_slices
        )

    # C1 baseline without EXIF never trips the EXIF confounding gate.
    flagged, reasons = evaluate_anti_confounding_gate(
        "C1_uncalibrated_pdi", _cf(0.0, 12.0, 25.0), c2_slices, c2_slices
    )
    assert flagged is False and reasons == []

    # Exact threshold equality (10.0 / 10.0 / 15.0) must NOT trip strict '>' gate (adjacency).
    flagged, reasons = evaluate_anti_confounding_gate(
        "C3_physics_plus_exif_meta",
        _cf(
            COUNTERFACTUAL_FOCAL_MEDIAN_THRESHOLD_PCT,
            COUNTERFACTUAL_PHYS_MEDIAN_THRESHOLD_PCT,
            COUNTERFACTUAL_PHYS_P90_THRESHOLD_PCT,
        ),
        c2_slices,
        c2_slices,
    )
    assert flagged is False and reasons == []

    # +0.01 above each threshold MUST trip the gate with explicit reason.
    f1, r1 = evaluate_anti_confounding_gate(
        "C3_physics_plus_exif_meta", _cf(10.01, 5.0, 10.0), c2_slices, c2_slices
    )
    assert f1 is True and any("focal_only" in msg for msg in r1)

    f2, r2 = evaluate_anti_confounding_gate(
        "C4_naive_double_count_fxz", _cf(0.0, 10.01, 14.0), c2_slices, c2_slices
    )
    assert f2 is True and any("physical_lens_swap median" in msg for msg in r2)

    f3, r3 = evaluate_anti_confounding_gate(
        "C3_physics_plus_exif_meta", _cf(0.0, 8.0, 15.01), c2_slices, c2_slices
    )
    assert f3 is True and any("physical_lens_swap p90" in msg for msg in r3)

    # <500g ultrawide degradation requires BOTH mae_g > c2_mae_g AND |mbe_g| > |c2_mbe_g|.
    only_mae_worse = {"<500g__ultrawide": {"n": 20, "mae_g": 120.0, "mbe_g": 55.0}}
    f4, _ = evaluate_anti_confounding_gate(
        "C3_physics_plus_exif_meta", _cf(0.0, 5.0, 10.0), only_mae_worse, c2_slices
    )
    assert f4 is False

    only_mbe_worse = {"<500g__ultrawide": {"n": 20, "mae_g": 110.0, "mbe_g": -75.0}}
    f5, _ = evaluate_anti_confounding_gate(
        "C3_physics_plus_exif_meta", _cf(0.0, 5.0, 10.0), only_mbe_worse, c2_slices
    )
    assert f5 is False

    both_worse = {"<500g__ultrawide": {"n": 20, "mae_g": 110.01, "mbe_g": -60.01}}
    f6, r6 = evaluate_anti_confounding_gate(
        "C4_naive_double_count_fxz", _cf(0.0, 5.0, 10.0), both_worse, c2_slices
    )
    assert f6 is True and any("<500g ultrawide degraded" in msg for msg in r6)


def test_select_winning_candidate_deterministic_tie_breaks_and_empty_eligible() -> None:
    # Disqualified candidate with better BalMAPE must be ignored.
    cheater = _make_stub_candidate(
        model_name="LightGBMLog",
        ablation_condition="C4_naive_double_count_fxz",
        balanced_band_mape_pct=20.0,
        lens_confounding_detected=True,
    )
    # Exact tie on balanced_band_mape_pct=35.0; test full tie-breaker cascade.
    c_worse_gmape = _make_stub_candidate(
        model_name="RidgeLog",
        ablation_condition="C2_physics_normalized_strict",
        balanced_band_mape_pct=35.0,
        global_mape_pct=36.0,
        global_mae_g=140.0,
        dual_lens_median_abs_diff_g=40.0,
    )
    c_worse_mae = _make_stub_candidate(
        model_name="HuberLog",
        ablation_condition="C2_physics_normalized_strict",
        balanced_band_mape_pct=35.0,
        global_mape_pct=35.5,
        global_mae_g=155.0,
        dual_lens_median_abs_diff_g=40.0,
    )
    c_none_dl = _make_stub_candidate(
        model_name="RandomForestLog",
        ablation_condition="C2_physics_normalized_strict",
        balanced_band_mape_pct=35.0,
        global_mape_pct=35.5,
        global_mae_g=150.0,
        dual_lens_median_abs_diff_g=None,
    )
    c_lex_loser = _make_stub_candidate(
        model_name="RidgeLog",
        ablation_condition="C2_physics_normalized_strict",
        balanced_band_mape_pct=35.0,
        global_mape_pct=35.5,
        global_mae_g=150.0,
        dual_lens_median_abs_diff_g=50.0,
    )
    c_winner = _make_stub_candidate(
        model_name="LightGBMLog",
        ablation_condition="C2_physics_normalized_strict",
        balanced_band_mape_pct=35.0,
        global_mape_pct=35.5,
        global_mae_g=150.0,
        dual_lens_median_abs_diff_g=50.0,
    )

    base_list = [cheater, c_worse_gmape, c_worse_mae, c_none_dl, c_lex_loser, c_winner]
    # Permutation-invariant deterministic selection (MOD-02 ordering edge).
    for perm in (base_list, list(reversed(base_list)), base_list[3:] + base_list[:3]):
        selected = select_winning_candidate(perm)
        assert selected is c_winner

    # Raises RuntimeError when zero candidates are eligible.
    with pytest.raises(RuntimeError, match="No production-eligible candidates"):
        select_winning_candidate([cheater])
    with pytest.raises(RuntimeError, match="No production-eligible candidates"):
        select_winning_candidate([])


def test_synthetic_lens_confounded_dataset_flags_c3_and_c4_and_selects_c2(
    tmp_path: Path,
) -> None:
    m_path, f_path = _write_synthetic_confounded_parquets(tmp_path)
    ds = load_modeling_dataset(
        manifest_path=m_path,
        features_path=f_path,
        min_heavy_sessions_per_fold=2,
    )

    ridge_c2 = evaluate_single_candidate(ds, "RidgeLog", "C2_physics_normalized_strict", n_jobs=2)
    lgb_c2 = evaluate_single_candidate(ds, "LightGBMLog", "C2_physics_normalized_strict", n_jobs=2)

    ridge_c3 = evaluate_single_candidate(
        ds,
        "RidgeLog",
        "C3_physics_plus_exif_meta",
        n_jobs=2,
        c2_by_band_and_lens=ridge_c2.by_band_and_lens,
    )
    lgb_c3 = evaluate_single_candidate(
        ds,
        "LightGBMLog",
        "C3_physics_plus_exif_meta",
        n_jobs=2,
        c2_by_band_and_lens=lgb_c2.by_band_and_lens,
    )
    ridge_c4 = evaluate_single_candidate(
        ds,
        "RidgeLog",
        "C4_naive_double_count_fxz",
        n_jobs=2,
        c2_by_band_and_lens=ridge_c2.by_band_and_lens,
    )
    lgb_c4 = evaluate_single_candidate(
        ds,
        "LightGBMLog",
        "C4_naive_double_count_fxz",
        n_jobs=2,
        c2_by_band_and_lens=lgb_c2.by_band_and_lens,
    )

    # C2 must prove exact 0.0% counterfactual shift under BOTH modes and pass the gate.
    for c2 in (ridge_c2, lgb_c2):
        assert c2.counterfactual_lens_flip["n_evaluated"] == len(ds.df)
        assert c2.counterfactual_lens_flip["focal_only"]["median_shift_pct"] == 0.0
        assert c2.counterfactual_lens_flip["physical_lens_swap"]["median_shift_pct"] == 0.0
        assert c2.counterfactual_lens_flip["physical_lens_swap"]["p90_shift_pct"] == 0.0
        assert c2.lens_confounding_detected is False
        assert c2.confounding_reasons == []
        assert c2.eligible_for_production is True
        assert c2.global_metrics["session_averaged_r2"] == c2.session_level_metrics[
            "session_averaged_r2"
        ]
        assert c2.dual_lens_consistency["n_mixed_lens_sessions"] > 0

    # C3 RidgeLog caught via focal_only (>10%); C3 LightGBMLog caught via physical_lens_swap/gate.
    assert ridge_c3.counterfactual_lens_flip["focal_only"]["median_shift_pct"] > 10.0
    assert ridge_c3.lens_confounding_detected is True
    assert ridge_c3.confounding_reasons
    assert ridge_c3.eligible_for_production is False

    assert lgb_c3.lens_confounding_detected is True
    assert lgb_c3.confounding_reasons
    assert lgb_c3.eligible_for_production is False

    # C4 naive (f*z) double-counting caught via physical_lens_swap on both linear and tree models.
    for c4 in (ridge_c4, lgb_c4):
        assert c4.lens_confounding_detected is True
        assert c4.confounding_reasons
        assert c4.eligible_for_production is False
        assert c4.counterfactual_lens_flip["physical_lens_swap"]["p90_shift_pct"] > 15.0

    # Winner selection deterministically picks a C2_physics_normalized_strict candidate.
    winner = select_winning_candidate([ridge_c2, ridge_c3, ridge_c4, lgb_c2, lgb_c3, lgb_c4])
    assert winner.ablation_condition == "C2_physics_normalized_strict"
    assert winner.eligible_for_production is True
