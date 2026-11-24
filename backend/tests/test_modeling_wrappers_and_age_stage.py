"""Unit, boundary, and fresh-subprocess joblib pickling tests for Phase 3 Plan 01."""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from backend.app.config import REPO_ROOT
from backend.app.modeling.age_stage import (
    VALID_AGE_DAYS_RANGE,
    VALID_TRAINING_WEIGHT_MAX_G,
    VALID_TRAINING_WEIGHT_MIN_G,
    AgeStageResult,
    map_weight_to_age_stage,
)
from backend.app.modeling.datasets import build_ablation_feature_specs
from backend.app.modeling.models import (
    DEFAULT_WEIGHTING_MODE,
    MAX_N_JOBS_CAP,
    MODEL_FAMILY_NAMES,
    build_regressor_pipeline,
    compute_sample_weights,
    extract_feature_importances,
    predict_from_feature_dict,
)
from backend.app.modeling.wrappers import (
    ALLOMETRIC_LOG_COLUMNS,
    AllometricLogTransformer,
    LogDuanRegressor,
)


def _make_synthetic_c2_regression_data(
    n_rows: int = 120,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Generate a fast synthetic 39-col C2 dataset for unit/pickling tests (zero data/raw I/O)."""
    rng = np.random.default_rng(seed)
    c2_cols = build_ablation_feature_specs()["C2_physics_normalized_strict"]

    area = rng.uniform(5_000.0, 120_000.0, size=n_rows)
    sqrt_area = np.sqrt(area)
    lab_a = rng.uniform(122.0, 145.0, size=n_rows)

    data: dict[str, np.ndarray] = {}
    for idx, col in enumerate(c2_cols):
        if col == "area_optical_norm":
            data[col] = area
        elif col == "sqrt_area_optical_norm":
            data[col] = sqrt_area
        elif col == "perim_optical_norm":
            data[col] = sqrt_area * rng.uniform(3.5, 4.5, size=n_rows)
        elif col == "major_axis_optical_norm":
            data[col] = sqrt_area * rng.uniform(1.2, 1.8, size=n_rows)
        elif col == "minor_axis_optical_norm":
            data[col] = sqrt_area * rng.uniform(0.7, 1.1, size=n_rows)
        elif col == "lab_a_mean":
            data[col] = lab_a
        else:
            data[col] = rng.uniform(0.1, 2.0, size=n_rows) + (idx * 0.01)

    X_df = pd.DataFrame(data, columns=list(c2_cols))
    # Smooth positive log-allometric synthetic target across 150..1850 g.
    log_y = (
        0.55 * np.log1p(area)
        + 0.03 * (lab_a - 128.0)
        + rng.normal(0.0, 0.08, size=n_rows)
        + 0.35
    )
    y_g = np.clip(np.expm1(log_y), 120.0, 2000.0)
    sessions = np.array([f"sess_{i // 6}" for i in range(n_rows)], dtype=str)
    bands = np.where(y_g < 500.0, "<500g", np.where(y_g <= 1000.0, "500-1000g", ">1000g"))
    return X_df, y_g, sessions, bands, c2_cols


def test_age_stage_module_uses_strictly_stdlib_imports() -> None:
    age_stage_path = REPO_ROOT / "backend" / "app" / "modeling" / "age_stage.py"
    tree = ast.parse(age_stage_path.read_text(encoding="utf-8"), filename=str(age_stage_path))

    allowed_stdlib = set(sys.stdlib_module_names) | {"__future__"}
    third_party_found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_pkg = alias.name.split(".")[0]
                if top_pkg not in allowed_stdlib:
                    third_party_found.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_pkg = node.module.split(".")[0]
            if top_pkg not in allowed_stdlib:
                third_party_found.append(node.module)

    assert not third_party_found, (
        f"age_stage.py must remain pure stdlib, found non-stdlib imports: {third_party_found}"
    )


@pytest.mark.parametrize(
    ("weight_g", "expected_code", "expected_label", "expected_age", "expected_out_of_range"),
    [
        (115.99, "inicial_pintinho", "Inicial / Pintinho", "~7-13 dias", True),
        (116.0, "inicial_pintinho", "Inicial / Pintinho", "~7-13 dias", False),
        (349.99, "inicial_pintinho", "Inicial / Pintinho", "~7-13 dias", False),
        (350.0, "crescimento_inicial", "Crescimento Inicial", "~14-19 dias", False),
        (700.0, "crescimento_inicial", "Crescimento Inicial", "~14-19 dias", False),
        (700.01, "crescimento_intermediario", "Crescimento Intermediário", "~20-24 dias", False),
        (1200.0, "crescimento_intermediario", "Crescimento Intermediário", "~20-24 dias", False),
        (1200.01, "terminacao_pre_abate", "Terminação / Pré-Abate", "~25-29 dias", False),
        (2093.0, "terminacao_pre_abate", "Terminação / Pré-Abate", "~25-29 dias", False),
        (2093.01, "terminacao_pre_abate", "Terminação / Pré-Abate", "~25-29 dias", True),
    ],
)
def test_map_weight_to_age_stage_exact_boundaries(
    weight_g: float,
    expected_code: str,
    expected_label: str,
    expected_age: str,
    expected_out_of_range: bool,
) -> None:
    res = map_weight_to_age_stage(weight_g)
    assert isinstance(res, AgeStageResult)
    assert res.weight_g == pytest.approx(weight_g)
    assert res.stage_code == expected_code
    assert res.stage_label == expected_label
    assert res.estimated_age_days_range == expected_age
    assert res.out_of_training_range is expected_out_of_range
    assert res.valid_training_range_g == (
        VALID_TRAINING_WEIGHT_MIN_G,
        VALID_TRAINING_WEIGHT_MAX_G,
    )
    assert res.valid_training_range_g == (116, 2093)
    assert res.valid_age_days_range == VALID_AGE_DAYS_RANGE == "~7-29 days"


@pytest.mark.parametrize(
    "bad_weight",
    [True, False, 0.0, 0, -50.0, math.nan, math.inf, -math.inf, None, "500"],
)
def test_map_weight_to_age_stage_rejects_invalid_inputs(bad_weight: object) -> None:
    with pytest.raises(ValueError):
        map_weight_to_age_stage(bad_weight)  # type: ignore[arg-type]


def test_allometric_log_transformer_targets_geometric_cols_and_clips_negatives() -> None:
    assert len(ALLOMETRIC_LOG_COLUMNS) == 17
    assert "feather_grad_mean" not in ALLOMETRIC_LOG_COLUMNS
    assert "feather_laplacian_var" not in ALLOMETRIC_LOG_COLUMNS
    assert "solidity" not in ALLOMETRIC_LOG_COLUMNS
    assert "lab_a_mean" not in ALLOMETRIC_LOG_COLUMNS

    df = pd.DataFrame(
        {
            "area_optical_norm": [-5.0, 0.0, 100.0, 2500.0],
            "perim_optical_norm": [0.0, 10.0, -2.5, 400.0],
            "solidity": [0.85, 0.91, 0.77, 0.95],
            "lab_a_mean": [128.0, 134.5, 121.25, 140.0],
            "feather_grad_mean": [12.5, 25.0, 40.0, 88.0],
        }
    )
    original_copy = df.copy()

    transformer = AllometricLogTransformer()
    out = transformer.fit_transform(df)

    # Original DataFrame must never be mutated in place.
    pd.testing.assert_frame_equal(df, original_copy)
    assert out.dtype == np.float64
    assert transformer.log_indices_ == (0, 1)

    np.testing.assert_allclose(
        out[:, 0],
        np.log1p(np.array([0.0, 0.0, 100.0, 2500.0])),
    )
    np.testing.assert_allclose(
        out[:, 1],
        np.log1p(np.array([0.0, 10.0, 0.0, 400.0])),
    )
    # Chromatic, texture, and dimensionless shape columns must remain bit-identical.
    assert np.array_equal(out[:, 2], df["solidity"].to_numpy(dtype=np.float64))
    assert np.array_equal(out[:, 3], df["lab_a_mean"].to_numpy(dtype=np.float64))
    assert np.array_equal(out[:, 4], df["feather_grad_mean"].to_numpy(dtype=np.float64))


def test_log_duan_regressor_roundtrip_smearing_weights_and_validation() -> None:
    rng = np.random.default_rng(7)
    X_clean = np.linspace(1.0, 5.0, 40).reshape(-1, 1)
    y_clean = np.expm1(2.0 + 0.8 * X_clean[:, 0])

    reg_exact = LogDuanRegressor(Ridge(alpha=1e-12), use_smearing=False)
    reg_exact.fit(X_clean, y_clean)
    assert reg_exact.smearing_factor_ == 1.0
    np.testing.assert_allclose(reg_exact.predict(X_clean), y_clean, rtol=1e-6)

    # Noisy log-normal residuals must yield Duan's smearing factor > 1.0.
    y_noisy = np.expm1(5.5 + 0.3 * X_clean[:, 0] + rng.normal(0.0, 0.35, size=len(X_clean)))
    reg_no_smear = LogDuanRegressor(Ridge(alpha=1.0), use_smearing=False).fit(X_clean, y_noisy)
    reg_smear = LogDuanRegressor(Ridge(alpha=1.0), use_smearing=True).fit(X_clean, y_noisy)
    assert reg_no_smear.smearing_factor_ == 1.0
    assert reg_smear.smearing_factor_ > 1.01

    # Verify Pipeline forwarding of reg__sample_weight alters weighted Duan smearing factor.
    pipe_unweighted = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", LogDuanRegressor(Ridge(alpha=1.0), use_smearing=True)),
        ]
    )
    pipe_weighted = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", LogDuanRegressor(Ridge(alpha=1.0), use_smearing=True)),
        ]
    )
    skewed_weights = np.where(np.arange(len(X_clean)) < 10, 5.0, 1.0)
    pipe_unweighted.fit(X_clean, y_noisy)
    pipe_weighted.fit(X_clean, y_noisy, reg__sample_weight=skewed_weights)
    assert pipe_weighted.named_steps["reg"].smearing_factor_ > 1.0
    assert pipe_weighted.named_steps["reg"].smearing_factor_ != pytest.approx(
        pipe_unweighted.named_steps["reg"].smearing_factor_,
        abs=1e-5,
    )

    # ASVS V5 invalid input rejection.
    with pytest.raises(ValueError, match="non-None"):
        LogDuanRegressor(estimator=None).fit(X_clean, y_clean)
    for bad_y in (
        np.array([0.0, 100.0]),
        np.array([-5.0, 100.0]),
        np.array([math.nan, 100.0]),
        np.array([math.inf, 100.0]),
        np.array([True, True]),
    ):
        with pytest.raises(ValueError):
            LogDuanRegressor(Ridge(alpha=1.0)).fit(X_clean[:2], bad_y)

    for bad_w in (
        np.array([-1.0, 2.0]),
        np.array([0.0, 0.0]),
        np.array([math.nan, 1.0]),
        np.array([True, True]),
        np.array([1.0]),
    ):
        with pytest.raises(ValueError):
            LogDuanRegressor(Ridge(alpha=1.0)).fit(
                X_clean[:2],
                y_clean[:2],
                sample_weight=bad_w,
            )


def test_compute_sample_weights_cap_normalization_and_validation() -> None:
    assert DEFAULT_WEIGHTING_MODE == "session_band_capped_5x"

    # Highly imbalanced sessions (47 vs 1) x imbalanced bands (<500g vs >1000g).
    sessions = ["s_big"] * 47 + ["s_med_1"] * 5 + ["s_med_2"] * 5 + ["s_rare"]
    bands = ["<500g"] * 47 + ["500-1000g"] * 10 + [">1000g"]

    w1 = compute_sample_weights(sessions, bands, max_weight_ratio=5.0)
    w2 = compute_sample_weights(sessions, bands, max_weight_ratio=5.0)
    assert np.array_equal(w1, w2)
    assert w1.dtype == np.float64
    assert np.all(np.isfinite(w1))
    assert np.all(w1 > 0.0)
    assert float(np.mean(w1)) == pytest.approx(1.0, abs=1e-9)
    assert float(np.max(w1) / np.min(w1)) <= 5.0 + 1e-9
    # Rare heavy session must outweigh the 47-image light session.
    assert w1[-1] > w1[0]

    # Single-element edge case returns [1.0].
    single = compute_sample_weights(["only_session"], ["<500g"])
    assert np.array_equal(single, np.array([1.0], dtype=np.float64))

    # Validation failures.
    with pytest.raises(ValueError):
        compute_sample_weights([], [])
    with pytest.raises(ValueError):
        compute_sample_weights(["s0", "s1"], ["<500g"])
    with pytest.raises(ValueError):
        compute_sample_weights(["s0", None], ["<500g", "<500g"])
    for bad_ratio in (0.5, 0.0, -2.0, True, math.nan, math.inf):
        with pytest.raises(ValueError):
            compute_sample_weights(sessions, bands, max_weight_ratio=bad_ratio)  # type: ignore[arg-type]


def test_all_four_families_fit_predict_ordering_importances_and_validation() -> None:
    assert MODEL_FAMILY_NAMES == ("RidgeLog", "HuberLog", "RandomForestLog", "LightGBMLog")
    X_df, y_g, sessions, bands, c2_cols = _make_synthetic_c2_regression_data(n_rows=120, seed=42)
    sw = compute_sample_weights(sessions, bands)

    sample_row = X_df.iloc[0].to_dict()
    reversed_row = dict(reversed(list(sample_row.items())))

    for model_name in MODEL_FAMILY_NAMES:
        pipe = build_regressor_pipeline(model_name, use_smearing=False, n_jobs=4, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            pipe.fit(X_df, y_g, reg__sample_weight=sw)
            pred_orig = predict_from_feature_dict(pipe, c2_cols, sample_row)
            pred_rev = predict_from_feature_dict(pipe, c2_cols, reversed_row)

        assert math.isfinite(pred_orig) and pred_orig > 0.0
        assert pred_orig == pytest.approx(pred_rev, rel=0.0, abs=1e-9)
        if model_name != "RandomForestLog":
            assert pred_orig == pred_rev

        importances = extract_feature_importances(pipe, c2_cols)
        assert len(importances) == len(c2_cols)
        assert sum(item["importance_pct"] for item in importances) == pytest.approx(100.0, abs=0.05)
        expected_order = sorted(
            importances,
            key=lambda item: (-float(item["importance_pct"]), str(item["feature"])),
        )
        assert importances == expected_order

    # Verify thread cap clamping & invalid arguments on build_regressor_pipeline.
    clamped_rf = build_regressor_pipeline("RandomForestLog", n_jobs=16)
    assert clamped_rf.named_steps["reg"].estimator.n_jobs == MAX_N_JOBS_CAP == 4
    with pytest.raises(ValueError):
        build_regressor_pipeline("UnknownNet")
    for bad_jobs in (-1, 0, True):
        with pytest.raises(ValueError):
            build_regressor_pipeline("LightGBMLog", n_jobs=bad_jobs)  # type: ignore[arg-type]

    # Verify predict_from_feature_dict input validation.
    fitted_ridge = build_regressor_pipeline("RidgeLog").fit(X_df, y_g)
    with pytest.raises(ValueError, match="non-empty"):
        predict_from_feature_dict(fitted_ridge, (), sample_row)
    with pytest.raises(ValueError, match="non-empty"):
        predict_from_feature_dict(fitted_ridge, c2_cols, {})
    incomplete = dict(sample_row)
    incomplete.pop(c2_cols[0])
    with pytest.raises(ValueError, match="Missing required feature"):
        predict_from_feature_dict(fitted_ridge, c2_cols, incomplete)
    for bad_val in (math.nan, math.inf, True, "bad"):
        corrupted = dict(sample_row)
        corrupted[c2_cols[0]] = bad_val  # type: ignore[assignment]
        with pytest.raises(ValueError):
            predict_from_feature_dict(fitted_ridge, c2_cols, corrupted)


def test_extract_feature_importances_deterministic_tie_break_and_zero_sum() -> None:
    cols = ("z_feat", "a_feat", "m_feat")
    X_sym = pd.DataFrame(
        {
            "z_feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "a_feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "m_feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    y_const = np.full(6, 500.0, dtype=np.float64)

    # Constant target yields exact zero coefficients -> all 0.0%, sorted alphabetically.
    pipe_zero = build_regressor_pipeline("RidgeLog", use_allometric_log=False).fit(X_sym, y_const)
    imp_zero = extract_feature_importances(pipe_zero, cols)
    assert [row["feature"] for row in imp_zero] == ["a_feat", "m_feat", "z_feat"]
    assert all(row["importance_pct"] == 0.0 for row in imp_zero)

    # Identical columns with varying target yield identical coef_ -> deterministic alpha tie-break.
    y_linear = np.array([200.0, 300.0, 450.0, 650.0, 900.0, 1300.0], dtype=np.float64)
    pipe_tie = build_regressor_pipeline("RidgeLog", use_allometric_log=False).fit(X_sym, y_linear)
    imp_tie = extract_feature_importances(pipe_tie, cols)
    assert [row["feature"] for row in imp_tie] == ["a_feat", "m_feat", "z_feat"]
    assert all(row["importance_pct"] == pytest.approx(33.3333, abs=1e-3) for row in imp_tie)


def test_all_four_model_families_fresh_subprocess_joblib_roundtrip_and_latency(
    tmp_path: Path,
) -> None:
    X_df, y_g, sessions, bands, c2_cols = _make_synthetic_c2_regression_data(n_rows=120, seed=99)
    sw = compute_sample_weights(sessions, bands)
    row_dict = X_df.iloc[0].to_dict()

    pipelines: dict[str, Pipeline] = {}
    expected_preds: dict[str, float] = {}
    for model_name in MODEL_FAMILY_NAMES:
        pipe = build_regressor_pipeline(model_name, use_smearing=True, n_jobs=4, random_state=42)
        pipe.fit(X_df, y_g, reg__sample_weight=sw)
        pipelines[model_name] = pipe
        expected_preds[model_name] = predict_from_feature_dict(pipe, c2_cols, row_dict)

    bundle_path = tmp_path / "all_families_bundle.joblib"
    joblib.dump({"pipelines": pipelines, "feature_names": c2_cols}, bundle_path, compress=3)

    subprocess_code = "\n".join(
        [
            "import json, sys, time, joblib",
            "from backend.app.modeling.models import predict_from_feature_dict",
            "bundle = joblib.load(sys.argv[1])",
            "row = json.loads(sys.argv[2])",
            "cols = bundle['feature_names']",
            "results = {}",
            "for name, pipe in bundle['pipelines'].items():",
            "    _ = predict_from_feature_dict(pipe, cols, row)",
            "    t0 = time.perf_counter()",
            "    pred = predict_from_feature_dict(pipe, cols, row)",
            "    ms = (time.perf_counter() - t0) * 1000.0",
            "    results[name] = {'pred': pred, 'ms': ms}",
            "print(json.dumps(results))",
        ]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::UserWarning",
            "-c",
            subprocess_code,
            str(bundle_path),
            json.dumps(row_dict),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(proc.stdout.strip())
    for model_name in MODEL_FAMILY_NAMES:
        assert parsed[model_name]["pred"] == pytest.approx(
            expected_preds[model_name],
            rel=0.0,
            abs=1e-9,
        )
    lgb_ms = float(parsed["LightGBMLog"]["ms"])
    assert lgb_ms < 5.0, f"LightGBMLog single-row latency exceeded 5.0 ms budget: {lgb_ms:.3f} ms"


def test_review_hardening_wrappers_complex_and_numpy_age_scalars() -> None:
    """Regression tests for WR-01 (DataFrame column-order check), WR-03 (complex), WR-04 (NumPy)."""
    X_df, y_g, _, _, c2_cols = _make_synthetic_c2_regression_data(n_rows=60, seed=7)

    # WR-01: AllometricLogTransformer rejects swapped DataFrame cols & clears stale ndarray names.
    tf = AllometricLogTransformer().fit(X_df)
    swapped = X_df.loc[:, list(reversed(c2_cols))]
    with pytest.raises(ValueError, match="column order mismatch"):
        tf.transform(swapped)

    ridge_pipe = build_regressor_pipeline("RidgeLog").fit(X_df, y_g)
    with pytest.raises(ValueError, match="column order mismatch"):
        ridge_pipe.predict(swapped)

    tf.fit(X_df.to_numpy(dtype=np.float64))
    assert not hasattr(tf, "feature_names_in_")

    # WR-03: predict_from_feature_dict rejects complex numbers cleanly (no ComplexWarning drop).
    row = X_df.iloc[0].to_dict()
    for bad_complex in (1.0 + 2.0j, np.complex128(3.0 + 4.0j)):
        corrupted = dict(row)
        corrupted[c2_cols[0]] = bad_complex  # type: ignore[assignment]
        with pytest.raises(ValueError, match="finite real numeric"):
            predict_from_feature_dict(ridge_pipe, c2_cols, corrupted)

    # WR-04: map_weight_to_age_stage accepts NumPy real integer/float scalars and rejects complex.
    for np_scalar in (np.int32(180), np.int64(550), np.float32(1500.5), np.float64(2093.0)):
        res = map_weight_to_age_stage(np_scalar)  # type: ignore[arg-type]
        assert res.weight_g == pytest.approx(float(np_scalar), abs=1e-4)
        assert res.out_of_training_range is False
    with pytest.raises(ValueError, match="finite positive real"):
        map_weight_to_age_stage(500.0 + 1.0j)  # type: ignore[arg-type]
