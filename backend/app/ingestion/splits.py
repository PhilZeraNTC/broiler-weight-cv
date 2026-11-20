"""Deterministic anti-leakage 5-fold StratifiedGroupKFold session splitter.

Implements locked decisions D-05, D-06, and PROIBIÇÃO 3: pre-sorts clean
records stably by ``["folder_chain", "capture_dt", "rel_path"]``, splits
exclusively via ``StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)``
grouped strictly by ``session_id`` and stratified by ``weight_band``, escalates
any sklearn warning to ``RuntimeError``, and verifies zero cross-fold session
overlap and heavy-bird (``>1000g``) session balance.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from backend.app.ingestion.decoding import BAND_MEDIUM_MAX_INCLUSIVE_G

N_SPLITS: int = 5
RANDOM_STATE: int = 42
SORT_KEYS: list[str] = ["folder_chain", "capture_dt", "rel_path"]
MIN_HEAVY_SESSIONS_PER_FOLD: int = 4

_VERIFY_REQUIRED_COLS: frozenset[str] = frozenset(
    {"session_id", "weight_band", "weight_g", "fold", "lens_class"}
)
_ASSIGN_REQUIRED_COLS: frozenset[str] = frozenset(
    set(SORT_KEYS) | {"session_id", "weight_band", "weight_g", "lens_class"}
)


def verify_fold_integrity(
    clean_df: pd.DataFrame,
    *,
    min_heavy_sessions_per_fold: int | None = None,
) -> dict[str, Any]:
    """Audit 5-fold session assignments for zero leakage, completeness, and heavy balance."""
    missing = sorted(_VERIFY_REQUIRED_COLS - set(clean_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for fold verification: {missing}")
    if clean_df.empty:
        raise ValueError("Cannot verify fold integrity on an empty DataFrame.")

    fold_series = clean_df["fold"]
    if fold_series.isna().any():
        raise RuntimeError("Invalid fold assignment detected: missing/NaN fold values present.")
    if not pd.api.types.is_integer_dtype(fold_series):
        raise RuntimeError(
            f"Invalid fold assignment detected: expected integer dtype, got {fold_series.dtype}."
        )

    valid_folds = set(range(N_SPLITS))
    observed_folds = {int(v) for v in fold_series.unique().tolist()}
    invalid_values = observed_folds - valid_folds
    if invalid_values:
        raise RuntimeError(
            f"Invalid fold values detected outside 0..{N_SPLITS - 1}: {sorted(invalid_values)}"
        )
    missing_folds = valid_folds - observed_folds
    if missing_folds:
        raise RuntimeError(f"Missing folds in clean DataFrame: {sorted(missing_folds)}")

    fold_sessions: dict[int, set[str]] = {
        fold_idx: set(clean_df.loc[fold_series == fold_idx, "session_id"].astype(str))
        for fold_idx in range(N_SPLITS)
    }
    for i in range(N_SPLITS):
        for j in range(i + 1, N_SPLITS):
            overlap = fold_sessions[i] & fold_sessions[j]
            if overlap:
                raise RuntimeError(
                    f"Session leakage detected between folds {i} and {j}: "
                    f"{len(overlap)} overlapping session(s)."
                )

    session_fold_nunique = clean_df.groupby("session_id", sort=False)["fold"].nunique()
    if (session_fold_nunique != 1).any():
        leaked_count = int((session_fold_nunique != 1).sum())
        raise RuntimeError(
            f"Session leakage detected: {leaked_count} session(s) span multiple folds."
        )

    fold_summaries: list[dict[str, Any]] = []
    for fold_idx in range(N_SPLITS):
        fold_df = clean_df.loc[fold_series == fold_idx]
        n_images = int(len(fold_df))
        n_sessions = int(fold_df["session_id"].nunique())

        heavy_df = fold_df.loc[fold_df["weight_g"] > BAND_MEDIUM_MAX_INCLUSIVE_G]
        n_heavy_sessions = int(heavy_df["session_id"].nunique())
        n_heavy_images = int(len(heavy_df))

        if (
            min_heavy_sessions_per_fold is not None
            and n_heavy_sessions < min_heavy_sessions_per_fold
        ):
            raise RuntimeError(
                f"Fold {fold_idx} has only {n_heavy_sessions} heavy (>1000g) sessions; "
                f"required >= {min_heavy_sessions_per_fold}."
            )

        band_vc = fold_df["weight_band"].value_counts()
        weight_band_counts: dict[str, int] = {
            "<500g": int(band_vc.get("<500g", 0)),
            "500-1000g": int(band_vc.get("500-1000g", 0)),
            ">1000g": int(band_vc.get(">1000g", 0)),
        }

        lens_vc = fold_df["lens_class"].value_counts()
        lens_counts: dict[str, int] = {
            "wide": int(lens_vc.get("wide", 0)),
            "ultrawide": int(lens_vc.get("ultrawide", 0)),
        }
        unknown_lens = int(lens_vc.get("unknown", 0))
        if unknown_lens > 0:
            lens_counts["unknown"] = unknown_lens

        fold_summaries.append(
            {
                "fold": fold_idx,
                "n_images": n_images,
                "n_sessions": n_sessions,
                "n_heavy_sessions": n_heavy_sessions,
                "n_heavy_images": n_heavy_images,
                "weight_band_counts": weight_band_counts,
                "lens_counts": lens_counts,
            }
        )

    return {
        "cross_fold_session_overlap": 0,
        "folds": fold_summaries,
    }


def assign_session_folds(
    clean_df: pd.DataFrame,
    *,
    min_heavy_sessions_per_fold: int | None = None,
) -> pd.DataFrame:
    """Deterministically assign 5-fold StratifiedGroupKFold IDs grouped strictly by session_id."""
    missing = sorted(_ASSIGN_REQUIRED_COLS - set(clean_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for fold assignment: {missing}")
    if clean_df.empty:
        raise ValueError("Cannot assign session folds to an empty DataFrame.")

    n_sessions = int(clean_df["session_id"].nunique())
    if n_sessions < N_SPLITS:
        raise ValueError(
            f"At least {N_SPLITS} distinct session_id values are required for "
            f"{N_SPLITS}-fold CV; got {n_sessions}."
        )

    if min_heavy_sessions_per_fold is not None:
        for band, band_group in clean_df.groupby("weight_band", sort=True):
            band_sessions = int(band_group["session_id"].nunique())
            if band_sessions < N_SPLITS:
                raise ValueError(
                    f"Weight band {band!r} has only {band_sessions} distinct sessions; "
                    f"at least {N_SPLITS} are required for {N_SPLITS}-fold StratifiedGroupKFold."
                )
        total_heavy_sessions = int(
            clean_df.loc[
                clean_df["weight_g"] > BAND_MEDIUM_MAX_INCLUSIVE_G, "session_id"
            ].nunique()
        )
        required_heavy = N_SPLITS * min_heavy_sessions_per_fold
        if total_heavy_sessions < required_heavy:
            raise ValueError(
                f"Only {total_heavy_sessions} heavy (>1000g) sessions available; "
                f"at least {required_heavy} required for >= {min_heavy_sessions_per_fold} per fold."
            )

    df = clean_df.sort_values(SORT_KEYS, kind="mergesort").reset_index(drop=True).copy()
    df["fold"] = np.int8(-1)

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        splits = list(
            splitter.split(
                df,
                df["weight_band"],
                groups=df["session_id"],
            )
        )

    if caught:
        messages = [f"{w.category.__name__}: {w.message}" for w in caught]
        raise RuntimeError(
            f"Unexpected warnings emitted during StratifiedGroupKFold: {messages}"
        )

    for fold_idx, (_, val_idx) in enumerate(splits):
        df.loc[df.index[val_idx], "fold"] = np.int8(fold_idx)

    df["fold"] = df["fold"].astype("int8")

    verify_fold_integrity(
        df,
        min_heavy_sessions_per_fold=min_heavy_sessions_per_fold,
    )
    return df
