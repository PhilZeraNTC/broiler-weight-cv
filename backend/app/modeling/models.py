"""Classical tabular regressor factory, fold sample weighting & inference helpers (D-01..D-03).

Exposes strictly classical/tabular families compliant with PROIBIÇÃO 1 (zero deep
learning): ``RidgeLog``, ``HuberLog``, ``RandomForestLog``, and ``LightGBMLog``.
All CPU concurrency is capped at ``MAX_N_JOBS_CAP = 4`` (never ``-1``) to protect
the 7.1 GiB RAM local host, and ``SimpleImputer(strategy="median")`` precedes
``LightGBMLog`` so single-row and batch inference emit zero feature-name warnings.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.modeling.wrappers import AllometricLogTransformer, LogDuanRegressor

MODEL_FAMILY_NAMES: tuple[str, ...] = (
    "RidgeLog",
    "HuberLog",
    "RandomForestLog",
    "LightGBMLog",
)
MAX_N_JOBS_CAP: int = 4
DEFAULT_WEIGHTING_MODE: str = "session_band_capped_5x"


def _resolve_n_jobs(n_jobs: int) -> int:
    """Validate positive integer thread count and clamp strictly to ``[1, MAX_N_JOBS_CAP]``."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive integer >= 1, got {n_jobs!r}")
    return min(n_jobs, MAX_N_JOBS_CAP)


def compute_sample_weights(
    train_sessions: Sequence[object] | np.ndarray | pd.Series,
    train_bands: Sequence[object] | np.ndarray | pd.Series,
    *,
    max_weight_ratio: float = 5.0,
) -> np.ndarray:
    """Compute fold-internal capped inverse ``sqrt(session) * sqrt(band)`` weights (D-03)."""
    if isinstance(max_weight_ratio, bool) or not isinstance(max_weight_ratio, (int, float)):
        raise ValueError(
            f"max_weight_ratio must be a finite number >= 1.0, got {max_weight_ratio!r}"
        )
    ratio = float(max_weight_ratio)
    if not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError(f"max_weight_ratio must be finite and >= 1.0, got {max_weight_ratio!r}")

    s_raw = pd.Series(list(train_sessions))
    b_raw = pd.Series(list(train_bands))
    if len(s_raw) != len(b_raw):
        raise ValueError(
            f"train_sessions ({len(s_raw)}) and train_bands ({len(b_raw)}) must have equal length."
        )
    if len(s_raw) == 0:
        raise ValueError("Cannot compute sample weights for empty training fold arrays.")
    if s_raw.isna().any() or b_raw.isna().any():
        raise ValueError("train_sessions and train_bands must not contain missing/NaN values.")

    if len(s_raw) == 1:
        return np.array([1.0], dtype=np.float64)

    s_series = s_raw.astype("str")
    b_series = b_raw.astype("str")
    s_counts = s_series.map(s_series.value_counts()).to_numpy(dtype=np.float64)
    b_counts = b_series.map(b_series.value_counts()).to_numpy(dtype=np.float64)

    raw_w = (1.0 / np.sqrt(s_counts)) * (1.0 / np.sqrt(b_counts))
    med = float(np.median(raw_w))
    half_span = math.sqrt(ratio)
    clipped = np.clip(raw_w, med / half_span, med * half_span)
    normalized = (clipped / float(np.mean(clipped))).astype(np.float64, copy=False)

    if not np.all(np.isfinite(normalized)) or np.any(normalized <= 0.0):
        raise ValueError("Computed non-finite or non-positive sample weights.")
    observed_ratio = float(np.max(normalized) / np.min(normalized))
    if observed_ratio > ratio + 1e-9:
        raise ValueError(
            f"Sample weight ratio {observed_ratio:.6f} exceeded cap {ratio:.6f}."
        )
    return normalized


def build_regressor_pipeline(
    model_name: str,
    *,
    use_smearing: bool = False,
    n_jobs: int = 4,
    random_state: int = 42,
    use_allometric_log: bool = True,
) -> Pipeline:
    """Instantiate one of the 4 locked classical/tabular log-target regressor pipelines."""
    if model_name not in MODEL_FAMILY_NAMES:
        raise ValueError(
            f"Unknown model_name {model_name!r}; expected one of {MODEL_FAMILY_NAMES!r}."
        )
    if not isinstance(use_smearing, bool):
        raise ValueError(f"use_smearing must be a bool, got {use_smearing!r}")
    if not isinstance(use_allometric_log, bool):
        raise ValueError(f"use_allometric_log must be a bool, got {use_allometric_log!r}")
    if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
        raise ValueError(f"random_state must be a non-negative integer, got {random_state!r}")

    eff_jobs = _resolve_n_jobs(n_jobs)

    if model_name == "RidgeLog":
        steps: list[tuple[str, Any]] = []
        if use_allometric_log:
            steps.append(("allometric", AllometricLogTransformer()))
        steps.extend(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("reg", LogDuanRegressor(Ridge(alpha=10.0), use_smearing=use_smearing)),
            ]
        )
        return Pipeline(steps)

    if model_name == "HuberLog":
        steps = []
        if use_allometric_log:
            steps.append(("allometric", AllometricLogTransformer()))
        steps.extend(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "reg",
                    LogDuanRegressor(
                        HuberRegressor(alpha=10.0, epsilon=1.35, max_iter=1000),
                        use_smearing=use_smearing,
                    ),
                ),
            ]
        )
        return Pipeline(steps)

    if model_name == "RandomForestLog":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "reg",
                    LogDuanRegressor(
                        RandomForestRegressor(
                            n_estimators=400,
                            max_depth=14,
                            min_samples_leaf=3,
                            max_features=0.5,
                            random_state=random_state,
                            n_jobs=eff_jobs,
                        ),
                        use_smearing=use_smearing,
                    ),
                ),
            ]
        )

    # LightGBMLog
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "reg",
                LogDuanRegressor(
                    LGBMRegressor(
                        n_estimators=500,
                        learning_rate=0.025,
                        num_leaves=24,
                        max_depth=6,
                        min_child_samples=12,
                        subsample=0.8,
                        subsample_freq=1,
                        colsample_bytree=0.55,
                        reg_alpha=0.3,
                        reg_lambda=2.0,
                        deterministic=True,
                        force_col_wise=True,
                        random_state=random_state,
                        n_jobs=eff_jobs,
                        verbosity=-1,
                    ),
                    use_smearing=use_smearing,
                ),
            ),
        ]
    )


def predict_from_feature_dict(
    pipeline: Pipeline,
    feature_names: Sequence[str],
    features: Mapping[str, float],
) -> float:
    """Predict a single bird weight (g) enforcing exact ``feature_names`` column order."""
    ordered_names = [str(name) for name in feature_names]
    if not ordered_names:
        raise ValueError("feature_names must be non-empty.")
    if len(ordered_names) != len(set(ordered_names)):
        raise ValueError("feature_names must not contain duplicate column names.")
    if not isinstance(features, Mapping) or len(features) == 0:
        raise ValueError("features must be a non-empty mapping of feature names to finite floats.")

    missing = [name for name in ordered_names if name not in features]
    if missing:
        raise ValueError(f"Missing required feature key(s): {missing}")

    ordered_values = np.empty((1, len(ordered_names)), dtype=np.float64)
    for idx, name in enumerate(ordered_names):
        raw_val = features[name]
        if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float, np.number)):
            raise ValueError(
                f"Feature {name!r} must be a finite numeric value, got {raw_val!r}"
            )
        numeric = float(raw_val)
        if not math.isfinite(numeric):
            raise ValueError(f"Feature {name!r} must be finite, got {raw_val!r}")
        ordered_values[0, idx] = numeric

    row_df = pd.DataFrame(ordered_values, columns=ordered_names)
    pred = np.asarray(pipeline.predict(row_df), dtype=np.float64)
    if pred.size != 1:
        raise ValueError(f"Expected single scalar prediction, got shape {pred.shape!r}")
    pred_g = float(pred[0])
    if not math.isfinite(pred_g) or pred_g <= 0.0:
        raise ValueError(f"Model produced non-finite or non-positive prediction: {pred_g!r}")
    return pred_g


def extract_feature_importances(
    pipeline: Pipeline,
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Extract normalized percentage importances with deterministic ``(-pct, feature)`` ordering."""
    ordered_names = [str(name) for name in feature_names]
    if not ordered_names:
        raise ValueError("feature_names must be non-empty.")

    reg = pipeline.named_steps.get("reg")
    if reg is None or not hasattr(reg, "estimator_"):
        raise ValueError("Pipeline must contain a fitted 'reg' LogDuanRegressor step.")
    estimator = reg.estimator_

    if isinstance(estimator, LGBMRegressor) and hasattr(estimator, "booster_"):
        raw = np.asarray(
            estimator.booster_.feature_importance(importance_type="gain"),
            dtype=np.float64,
        )
    elif hasattr(estimator, "feature_importances_"):
        raw = np.asarray(estimator.feature_importances_, dtype=np.float64)
    elif hasattr(estimator, "coef_"):
        raw = np.abs(np.asarray(estimator.coef_, dtype=np.float64).ravel())
    else:
        raise TypeError(
            f"Unsupported fitted estimator type for feature importance extraction: "
            f"{type(estimator).__name__}"
        )

    if raw.shape != (len(ordered_names),):
        raise ValueError(
            f"Feature importance length ({raw.shape[0]}) != len(feature_names) "
            f"({len(ordered_names)})."
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("Non-finite feature importances encountered.")

    raw_nonneg = np.clip(raw, 0.0, None)
    total = float(np.sum(raw_nonneg))
    pcts = (raw_nonneg / total) * 100.0 if total > 0.0 else np.zeros_like(raw_nonneg)

    items: list[dict[str, Any]] = [
        {
            "feature": name,
            "importance_pct": round(float(pct), 4),
            "raw_importance": round(float(raw_val), 6),
        }
        for name, pct, raw_val in zip(ordered_names, pcts, raw_nonneg, strict=True)
    ]
    items.sort(key=lambda item: (-float(item["importance_pct"]), str(item["feature"])))
    return items


__all__ = [
    "MODEL_FAMILY_NAMES",
    "MAX_N_JOBS_CAP",
    "DEFAULT_WEIGHTING_MODE",
    "compute_sample_weights",
    "build_regressor_pipeline",
    "predict_from_feature_dict",
    "extract_feature_importances",
]
