"""Pickling-safe scikit-learn transformers and log-target Duan regressors (D-01, D-02).

All custom estimators live strictly in this importable package module — never
under ``__main__`` or via ``lambda`` closures — so serialized ``joblib`` bundles
unpickle deterministically across CLI scripts, test subprocesses, and Phase 4
FastAPI workers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

# Strictly positive geometric area/linear descriptors across ablation conditions C1-C4.
# Excludes chromatic/texture descriptors (feather_grad_mean, feather_laplacian_var)
# and dimensionless shape descriptors so linear models capture W ~ A^(3/2) allometry.
ALLOMETRIC_LOG_COLUMNS: frozenset[str] = frozenset(
    {
        "area_px",
        "perimeter_px",
        "major_axis_px",
        "minor_axis_px",
        "convex_area_px",
        "area_frac",
        "area_optical_norm",
        "sqrt_area_optical_norm",
        "perim_optical_norm",
        "major_axis_optical_norm",
        "minor_axis_optical_norm",
        "area_frac_optical_norm",
        "area_naive_fxz",
        "sqrt_area_naive_fxz",
        "perim_naive_fxz",
        "major_naive_fxz",
        "minor_naive_fxz",
    }
)


def _contains_bool_values(raw: np.ndarray) -> bool:
    if raw.dtype == np.bool_:
        return True
    if raw.dtype == object:
        return any(isinstance(v, bool) for v in raw.ravel())
    return False


class AllometricLogTransformer(BaseEstimator, TransformerMixin):
    """Apply ``np.log1p(np.clip(x, 0.0, None))`` strictly to geometric area/linear columns."""

    def __init__(self, log_columns: tuple[str, ...] | None = None) -> None:
        self.log_columns = log_columns

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
    ) -> AllometricLogTransformer:
        target = (
            set(self.log_columns) if self.log_columns is not None else set(ALLOMETRIC_LOG_COLUMNS)
        )
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
            self.n_features_in_ = int(X.shape[1])
            self.log_indices_: tuple[int, ...] = tuple(
                i for i, col in enumerate(X.columns) if str(col) in target
            )
        else:
            arr = np.asarray(X)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D input array, got shape {arr.shape!r}")
            self.n_features_in_ = int(arr.shape[1])
            self.log_indices_ = ()
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, "log_indices_")
        arr = (
            X.to_numpy(dtype=np.float64, copy=True)
            if isinstance(X, pd.DataFrame)
            else np.array(X, dtype=np.float64, copy=True)
        )
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {arr.shape!r}")
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} feature columns, got {arr.shape[1]}"
            )
        if self.log_indices_:
            idx = list(self.log_indices_)
            arr[:, idx] = np.log1p(np.clip(arr[:, idx], 0.0, None))
        return arr


class LogDuanRegressor(BaseEstimator, RegressorMixin):
    """Log1p-target regressor with optional fold-internal Duan's smearing correction (D-01)."""

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        *,
        use_smearing: bool = False,
        min_prediction_g: float = 1.0,
    ) -> None:
        self.estimator = estimator
        self.use_smearing = use_smearing
        self.min_prediction_g = min_prediction_g

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> LogDuanRegressor:
        if self.estimator is None:
            raise ValueError("LogDuanRegressor requires a non-None base estimator.")
        if not isinstance(self.use_smearing, bool):
            raise ValueError(f"use_smearing must be a bool, got {self.use_smearing!r}")
        if isinstance(self.min_prediction_g, bool) or not isinstance(
            self.min_prediction_g, (int, float)
        ):
            raise ValueError(
                f"min_prediction_g must be a finite number >= 0, got {self.min_prediction_g!r}"
            )
        min_pred = float(self.min_prediction_g)
        if not math.isfinite(min_pred) or min_pred < 0.0:
            raise ValueError(
                f"min_prediction_g must be finite and >= 0, got {self.min_prediction_g!r}"
            )

        y_raw = np.asarray(y)
        if _contains_bool_values(y_raw):
            raise ValueError("Target weights must be numeric, not boolean.")
        try:
            y_arr = np.asarray(y_raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Target weights must be convertible to float64.") from exc
        if y_arr.ndim != 1 or y_arr.size == 0:
            raise ValueError(f"Target weights must be a non-empty 1D array, got {y_arr.shape!r}")
        if not np.all(np.isfinite(y_arr)) or np.any(y_arr <= 0.0):
            raise ValueError("Target weights must be finite and strictly positive.")

        w: np.ndarray | None = None
        if sample_weight is not None:
            w_raw = np.asarray(sample_weight)
            if _contains_bool_values(w_raw):
                raise ValueError("sample_weight must be numeric, not boolean.")
            try:
                w = np.asarray(w_raw, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("sample_weight must be convertible to float64.") from exc
            if w.ndim != 1 or w.shape[0] != y_arr.shape[0]:
                raise ValueError(
                    f"sample_weight must be 1D with length {y_arr.shape[0]}, got {w.shape!r}"
                )
            if not np.all(np.isfinite(w)) or np.any(w < 0.0) or float(np.sum(w)) <= 0.0:
                raise ValueError(
                    "sample_weight must be finite, non-negative, and have strictly positive sum."
                )

        self.estimator_ = clone(self.estimator)
        y_log = np.log1p(y_arr)

        if w is not None:
            self.estimator_.fit(X, y_log, sample_weight=w)
        else:
            self.estimator_.fit(X, y_log)

        if self.use_smearing:
            pred_log = np.asarray(self.estimator_.predict(X), dtype=np.float64)
            exp_resid = np.exp(np.clip(y_log - pred_log, -20.0, 20.0))
            smearing = (
                float(np.average(exp_resid, weights=w))
                if w is not None
                else float(np.mean(exp_resid))
            )
            if not math.isfinite(smearing) or smearing <= 0.0:
                raise ValueError(f"Computed non-finite Duan smearing factor: {smearing!r}")
            self.smearing_factor_: float = smearing
        else:
            self.smearing_factor_ = 1.0
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("estimator_", "smearing_factor_"))
        pred_log = np.asarray(self.estimator_.predict(X), dtype=np.float64)
        pred_g = np.exp(np.clip(pred_log, -20.0, 20.0)) * self.smearing_factor_ - 1.0
        return np.clip(pred_g, float(self.min_prediction_g), None)


__all__ = [
    "ALLOMETRIC_LOG_COLUMNS",
    "AllometricLogTransformer",
    "LogDuanRegressor",
]
