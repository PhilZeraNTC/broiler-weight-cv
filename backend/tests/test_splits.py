"""Unit and anti-leakage invariant tests for 5-fold StratifiedGroupKFold session splits."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backend.app.ingestion as ingestion
from backend.app.config import REPO_ROOT
from backend.app.ingestion.decoding import assign_weight_band, compute_image_id
from backend.app.ingestion.manifest import (
    MANIFEST_COLUMNS,
    scan_dataset_records,
    split_clean_and_quarantine,
)
from backend.app.ingestion.splits import (
    MIN_HEAVY_SESSIONS_PER_FOLD,
    N_SPLITS,
    RANDOM_STATE,
    SORT_KEYS,
    assign_session_folds,
    verify_fold_integrity,
)


def _build_audit_shaped_clean_df(
    n_light_sessions: int = 124,
    n_medium_sessions: int = 40,
    n_heavy_sessions: int = 24,
) -> pd.DataFrame:
    """Synthesize a clean DataFrame matching Audit §10 stratum session counts & _1 bursts."""
    rows: list[dict[str, object]] = []
    base_dt = datetime(2022, 3, 24, 8, 0, 0)

    specs: list[tuple[str, int, int, int]] = [
        ("light", n_light_sessions, 180, 300),
        ("medium", n_medium_sessions, 520, 460),
        ("heavy", n_heavy_sessions, 1050, 1000),
    ]

    session_counter = 0
    for band_tag, n_sessions, base_weight, weight_span in specs:
        for s_idx in range(n_sessions):
            weight_g = base_weight + ((s_idx * 17) % max(1, weight_span))
            if band_tag == "light":
                weight_g = min(weight_g, 499)
            elif band_tag == "medium":
                weight_g = max(500, min(weight_g, 1000))
            else:
                weight_g = max(1001, weight_g)

            folder_chain = f"{weight_g}"
            session_id = f"{folder_chain}__s{s_idx}"
            # Vary session sizes 6..13 (matching Audit §10 heavy session range)
            n_imgs = 6 + (s_idx % 8)
            session_start = base_dt + timedelta(hours=session_counter)
            session_counter += 1

            for img_idx in range(n_imgs):
                # Inject same-second _1 burst capture on img_idx == 1 for first 10 sessions
                if img_idx == 1 and s_idx < 10:
                    capture_dt = session_start
                    fname = f"IMG_{capture_dt.strftime('%Y%m%d_%H%M%S')}_1.jpg"
                    pattern = "img_arabic_burst"
                else:
                    capture_dt = session_start + timedelta(seconds=img_idx * 5)
                    fname = f"IMG_{capture_dt.strftime('%Y%m%d_%H%M%S')}.jpg"
                    pattern = "img_arabic"

                rel_path = f"{folder_chain}/s{s_idx}_{img_idx}_{fname}"
                is_uw = (band_tag == "heavy" and img_idx % 2 == 0) or (
                    band_tag != "heavy" and img_idx % 4 == 0
                )
                rows.append(
                    {
                        "image_id": compute_image_id(rel_path),
                        "rel_path": rel_path,
                        "folder_chain": folder_chain,
                        "raw_weight_g": weight_g,
                        "weight_g": weight_g,
                        "capture_dt": capture_dt,
                        "capture_date": capture_dt.date(),
                        "oriented_w": 1472 if is_uw else 1840,
                        "oriented_h": 3264 if is_uw else 4080,
                        "exif_orientation": 6,
                        "focal_length_mm": 1.65 if is_uw else 4.25,
                        "zoom_multiple": 0.6 if is_uw else 1.0,
                        "is_multi_bird": False,
                        "is_severe_mislabel": False,
                        "is_single_bird_clean": True,
                        "session_id": session_id,
                        "filename_pattern": pattern,
                        "was_cp866_recovered": False,
                        "was_nested_folder": False,
                        "was_lcd_corrected": False,
                        "lens_class": "ultrawide" if is_uw else "wide",
                        "weight_band": assign_weight_band(weight_g),
                        "quarantine_reason": None,
                        "fold": np.int8(-1),
                    }
                )

    df = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    df["capture_dt"] = pd.to_datetime(df["capture_dt"]).astype("datetime64[us]")
    df["raw_weight_g"] = df["raw_weight_g"].astype("int32")
    df["weight_g"] = df["weight_g"].astype("int32")
    df["fold"] = df["fold"].astype("int8")
    return df


def test_audit_section_10_shaped_dataset_produces_balanced_zero_leakage_splits() -> None:
    clean_df = _build_audit_shaped_clean_df(
        n_light_sessions=124,
        n_medium_sessions=40,
        n_heavy_sessions=24,
    )
    assert clean_df["session_id"].nunique() == 188

    assigned = assign_session_folds(
        clean_df,
        min_heavy_sessions_per_fold=MIN_HEAVY_SESSIONS_PER_FOLD,
    )
    assert assigned["fold"].dtype == np.int8
    assert set(assigned["fold"].unique().tolist()) == {0, 1, 2, 3, 4}

    audit = verify_fold_integrity(
        assigned,
        min_heavy_sessions_per_fold=MIN_HEAVY_SESSIONS_PER_FOLD,
    )
    assert audit["cross_fold_session_overlap"] == 0
    assert len(audit["folds"]) == N_SPLITS

    heavy_counts = sorted(f["n_heavy_sessions"] for f in audit["folds"])
    assert heavy_counts == [4, 5, 5, 5, 5]
    assert sum(f["n_sessions"] for f in audit["folds"]) == 188
    assert sum(f["n_images"] for f in audit["folds"]) == len(clean_df)

    # Same-second _1 burst pairs must never be split across folds.
    burst_sessions = assigned.loc[
        assigned["filename_pattern"] == "img_arabic_burst", "session_id"
    ].unique()
    assert len(burst_sessions) > 0
    for sid in burst_sessions:
        assert assigned.loc[assigned["session_id"] == sid, "fold"].nunique() == 1


def test_assign_session_folds_is_permutation_and_repeat_deterministic() -> None:
    clean_df = _build_audit_shaped_clean_df(
        n_light_sessions=20,
        n_medium_sessions=10,
        n_heavy_sessions=24,
    )

    ref_a = assign_session_folds(clean_df, min_heavy_sessions_per_fold=4)
    ref_b = assign_session_folds(clean_df, min_heavy_sessions_per_fold=4)
    reversed_out = assign_session_folds(clean_df.iloc[::-1].copy(), min_heavy_sessions_per_fold=4)

    rng = np.random.default_rng(12345)
    perm_idx = rng.permutation(len(clean_df))
    permuted_out = assign_session_folds(
        clean_df.iloc[perm_idx].reset_index(drop=True),
        min_heavy_sessions_per_fold=4,
    )

    for candidate in (ref_b, reversed_out, permuted_out):
        assert candidate["image_id"].tolist() == ref_a["image_id"].tolist()
        assert candidate["rel_path"].tolist() == ref_a["rel_path"].tolist()
        assert candidate["session_id"].tolist() == ref_a["session_id"].tolist()
        assert np.array_equal(candidate["fold"].to_numpy(), ref_a["fold"].to_numpy())


def test_assign_session_folds_end_to_end_with_synthetic_dataset_tree(
    synthetic_dataset_tree: Path,
) -> None:
    full_df = scan_dataset_records(synthetic_dataset_tree)
    clean_df, quarantine_df = split_clean_and_quarantine(full_df)

    assigned = assign_session_folds(clean_df)
    audit = verify_fold_integrity(assigned)

    assert audit["cross_fold_session_overlap"] == 0
    assert set(assigned["fold"].unique().tolist()) == {0, 1, 2, 3, 4}
    assert assigned.loc[assigned["session_id"] == "250__s0", "fold"].nunique() == 1
    assert (quarantine_df["fold"] == -1).all()


def test_empty_undersized_and_warning_inputs_fail_loudly() -> None:
    empty_df = pd.DataFrame(columns=list(MANIFEST_COLUMNS))
    with pytest.raises(ValueError, match="empty"):
        assign_session_folds(empty_df)

    with pytest.raises(ValueError, match="empty"):
        verify_fold_integrity(empty_df)

    # Fewer than 5 distinct sessions -> ValueError
    tiny_df = _build_audit_shaped_clean_df(
        n_light_sessions=4,
        n_medium_sessions=0,
        n_heavy_sessions=0,
    )
    with pytest.raises(ValueError, match="distinct session_id"):
        assign_session_folds(tiny_df)

    # Fewer than 5 sessions in one band with min_heavy_sessions_per_fold set -> ValueError
    insufficient_heavy = _build_audit_shaped_clean_df(
        n_light_sessions=10,
        n_medium_sessions=10,
        n_heavy_sessions=3,
    )
    with pytest.raises(ValueError, match="sessions"):
        assign_session_folds(insufficient_heavy, min_heavy_sessions_per_fold=4)

    # Undersized minority class (< n_splits=5 total samples) with min_heavy_sessions_per_fold=None
    # triggers sklearn's UserWarning, which must escalate into RuntimeError (D-06 warning gate).
    warning_base = _build_audit_shaped_clean_df(
        n_light_sessions=10,
        n_medium_sessions=10,
        n_heavy_sessions=2,
    )
    non_heavy = warning_base.loc[warning_base["weight_band"] != ">1000g"]
    two_heavy_samples = (
        warning_base.loc[warning_base["weight_band"] == ">1000g"]
        .drop_duplicates(subset=["session_id"])
        .head(2)
    )
    warning_df = pd.concat([non_heavy, two_heavy_samples], ignore_index=True)
    with pytest.raises(RuntimeError, match="Unexpected warnings"):
        assign_session_folds(warning_df)


def test_verify_fold_integrity_catches_injected_leakage_and_invalid_folds() -> None:
    clean_df = _build_audit_shaped_clean_df(
        n_light_sessions=10,
        n_medium_sessions=10,
        n_heavy_sessions=10,
    )
    assigned = assign_session_folds(clean_df, min_heavy_sessions_per_fold=2)

    # Inject cross-fold session overlap by moving one image of a session to another fold
    tampered = assigned.copy()
    target_sid = str(tampered.iloc[0]["session_id"])
    sid_indices = tampered.index[tampered["session_id"] == target_sid].tolist()
    assert len(sid_indices) >= 2
    orig_fold = int(tampered.loc[sid_indices[0], "fold"])
    tampered.loc[sid_indices[0], "fold"] = np.int8((orig_fold + 1) % N_SPLITS)

    with pytest.raises(RuntimeError, match="Session leakage"):
        verify_fold_integrity(tampered)

    # Invalid fold integer (-1 in clean DataFrame)
    bad_fold = assigned.copy()
    bad_fold.loc[bad_fold.index[0], "fold"] = np.int8(-1)
    with pytest.raises(RuntimeError, match="Invalid fold"):
        verify_fold_integrity(bad_fold)

    # Unmet heavy-session floor raises RuntimeError
    with pytest.raises(RuntimeError, match="heavy"):
        verify_fold_integrity(assigned, min_heavy_sessions_per_fold=5)


def test_ast_scan_prohibits_random_and_ungrouped_splitters_in_ingestion() -> None:
    ingestion_dir = REPO_ROOT / "backend" / "app" / "ingestion"
    py_files = sorted(ingestion_dir.rglob("*.py"))
    assert py_files

    banned_splitters = {
        "train_test_split",
        "KFold",
        "StratifiedKFold",
        "ShuffleSplit",
        "StratifiedShuffleSplit",
        "GroupShuffleSplit",
    }
    violations: list[str] = []
    saw_sgkf = False

    for py_file in py_files:
        rel_file = py_file.relative_to(REPO_ROOT)
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in banned_splitters:
                        violations.append(f"{rel_file}:{node.lineno}:import {alias.name}")
                    if alias.name == "StratifiedGroupKFold":
                        saw_sgkf = True
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in banned_splitters:
                    violations.append(f"{rel_file}:{node.lineno}:call {func.id}")
                elif isinstance(func, ast.Attribute) and func.attr in banned_splitters:
                    violations.append(f"{rel_file}:{node.lineno}:call {func.attr}")

    assert not violations, f"PROIBIÇÃO 3 violation — banned splitters found: {violations}"
    assert saw_sgkf, "Expected StratifiedGroupKFold usage in backend/app/ingestion/splits.py"


def test_ingestion_package_barrel_exports_public_api() -> None:
    assert N_SPLITS == 5
    assert RANDOM_STATE == 42
    assert SORT_KEYS == ["folder_chain", "capture_dt", "rel_path"]
    assert MIN_HEAVY_SESSIONS_PER_FOLD == 4
    assert len(ingestion.MANIFEST_COLUMNS) == 24
    for symbol in ingestion.__all__:
        assert hasattr(ingestion, symbol)
