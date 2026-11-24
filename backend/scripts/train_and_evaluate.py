"""Single deterministic CLI for Phase 3 training, optical ablation & artifact publishing.

Orchestrates the full AvianScale CV Phase 3 pipeline (D-01..D-11):
1. Loads and verifies 1:1 alignment between ``manifest_clean.parquet`` and
   ``features_clean.parquet``, locks ``PDIConfig().config_hash()`` (``43a3cfaa3208019e``),
   and asserts zero cross-fold ``session_id`` overlap via ``verify_fold_integrity()``
   (PROIBIÇÃO 2 & 3).
2. Runs leak-free 5-fold OOF cross-validation across all 4 classical tabular regressor
   families x 4 EXIF optical ablation conditions, 2x2 weighting x smearing sensitivity,
   linear allometric-log diagnostics, 127-session dual-lens consistency, and dual-mode
   counterfactual anti-confounding auditing (PROIBIÇÃO 1).
3. Refits the deterministic D-06 winner on 100% clean data and atomically writes
   git-ignored ``models/weight_regressor.joblib`` alongside git-tracked,
   byte-deterministic, zero-path-leakage ``models/evaluation_report.json`` and
   opaque-ID ``models/presets_catalog.json`` (PROIBIÇÃO 4 & ASVS V4).

Custom scikit-learn estimators and transformers live exclusively inside
``backend.app.modeling.wrappers`` — zero classes are defined in this CLI module to
eliminate ``__main__`` pickling hazards across subprocesses and FastAPI workers.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import (
    CLEAN_MANIFEST_PATH,
    EVALUATION_REPORT_PATH,
    FEATURES_CLEAN_PATH,
    MODEL_BUNDLE_PATH,
    PRESETS_CATALOG_PATH,
    QUARANTINE_MANIFEST_PATH,
    RAW_DATASET_DIR,
)
from backend.app.ingestion.splits import MIN_HEAVY_SESSIONS_PER_FOLD
from backend.app.modeling import (
    build_evaluation_report,
    build_model_bundle,
    build_presets_catalog,
    load_modeling_dataset,
    run_full_evaluation,
    save_model_bundle,
    write_evaluation_report,
    write_presets_catalog,
)

MAX_N_JOBS_CAP: int = 4


def _resolve_effective_n_jobs(n_jobs: int) -> int:
    """Validate positive integer thread count and clamp strictly to ``[1, MAX_N_JOBS_CAP]``."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f"--n-jobs must be a positive integer >= 1, got {n_jobs!r}")
    return min(n_jobs, MAX_N_JOBS_CAP)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for deterministic Phase 3 training and evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Train, audit, and atomically publish AvianScale CV classical tabular weight "
            "regression models, optical ablation report, and curated OOF/Watershed presets."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=CLEAN_MANIFEST_PATH,
        help="Path to materialized manifest_clean.parquet.",
    )
    parser.add_argument(
        "--quarantine-manifest-path",
        type=Path,
        default=QUARANTINE_MANIFEST_PATH,
        help="Path to materialized manifest_quarantine.parquet.",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=FEATURES_CLEAN_PATH,
        help="Path to materialized 60-column features_clean.parquet.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=RAW_DATASET_DIR,
        help="Read-only root directory of raw broiler weight images (Watershed preset audit).",
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=MODEL_BUNDLE_PATH,
        help="Destination path for git-ignored dual-bundle weight_regressor.joblib.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=EVALUATION_REPORT_PATH,
        help="Destination path for git-tracked byte-deterministic evaluation_report.json.",
    )
    parser.add_argument(
        "--presets-path",
        type=Path,
        default=PRESETS_CATALOG_PATH,
        help="Destination path for git-tracked opaque-ID presets_catalog.json.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=MAX_N_JOBS_CAP,
        help=f"Bounded CPU thread count for tree regressors (clamped to 1..{MAX_N_JOBS_CAP}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full training, ablation, and preset selection without writing any files.",
    )
    args = parser.parse_args(argv)
    args.n_jobs = _resolve_effective_n_jobs(args.n_jobs)
    return args


def run_train_and_evaluate(
    manifest_path: Path = CLEAN_MANIFEST_PATH,
    quarantine_manifest_path: Path = QUARANTINE_MANIFEST_PATH,
    features_path: Path = FEATURES_CLEAN_PATH,
    dataset_root: Path = RAW_DATASET_DIR,
    bundle_path: Path = MODEL_BUNDLE_PATH,
    report_path: Path = EVALUATION_REPORT_PATH,
    presets_path: Path = PRESETS_CATALOG_PATH,
    *,
    n_jobs: int = MAX_N_JOBS_CAP,
    dry_run: bool = False,
    min_heavy_sessions_per_fold: int | None = MIN_HEAVY_SESSIONS_PER_FOLD,
) -> dict[str, Any]:
    """Execute full Phase 3 CV, optical ablation, dual-bundle build, and atomic publishing."""
    t0 = time.perf_counter()
    eff_jobs = _resolve_effective_n_jobs(n_jobs)

    quarantine_file = Path(quarantine_manifest_path)
    if not quarantine_file.is_file():
        raise FileNotFoundError(f"Quarantine manifest Parquet not found: {quarantine_file}")

    dataset = load_modeling_dataset(
        manifest_path=manifest_path,
        features_path=features_path,
        min_heavy_sessions_per_fold=min_heavy_sessions_per_fold,
    )
    quarantine_df = pd.read_parquet(quarantine_file)

    eval_result = run_full_evaluation(dataset, n_jobs=eff_jobs)
    bundle = build_model_bundle(dataset, eval_result, n_jobs=eff_jobs)
    report = build_evaluation_report(dataset, eval_result)
    catalog = build_presets_catalog(
        dataset,
        eval_result,
        quarantine_df,
        dataset_root,
        config=dataset.pdi_config,
    )

    if not dry_run:
        save_model_bundle(
            bundle,
            bundle_path,
            expected_pdi_config_hash=dataset.pdi_config_hash,
        )
        write_evaluation_report(report, report_path)
        write_presets_catalog(catalog, presets_path)

    wall_clock_sec = time.perf_counter() - t0
    winner = eval_result.winner
    g_metrics = winner.global_metrics

    # PROIBIÇÃO 4: emit strictly aggregate metrics — never raw filenames or host paths.
    # wall_clock_sec is logged ONLY to stdout, never inside git-tracked JSON files.
    print(
        "Training and evaluation complete: "
        f"winning_model={winner.model_name} "
        f"winning_condition={winner.ablation_condition} "
        f"balanced_band_mape_pct={winner.balanced_band_mape_pct:.4f} "
        f"global_mae_g={g_metrics['mae_g']:.4f} "
        f"global_r2={g_metrics['r2']} "
        f"session_averaged_r2={g_metrics['session_averaged_r2']} "
        f"n_jobs={eff_jobs} "
        f"pdi_config_hash={dataset.pdi_config_hash} "
        f"n_single_presets={len(catalog['single_bird_presets'])} "
        f"n_watershed_presets={len(catalog['watershed_presets'])} "
        f"wall_clock_sec={wall_clock_sec:.2f} "
        f"dry_run={dry_run}"
    )

    return {
        "bundle": bundle,
        "report": report,
        "catalog": catalog,
        "wall_clock_sec": wall_clock_sec,
        "dry_run": bool(dry_run),
        "n_jobs": eff_jobs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    run_train_and_evaluate(
        manifest_path=args.manifest_path,
        quarantine_manifest_path=args.quarantine_manifest_path,
        features_path=args.features_path,
        dataset_root=args.dataset_root,
        bundle_path=args.bundle_path,
        report_path=args.report_path,
        presets_path=args.presets_path,
        n_jobs=args.n_jobs,
        dry_run=args.dry_run,
    )
    return 0


__all__ = [
    "MAX_N_JOBS_CAP",
    "parse_args",
    "run_train_and_evaluate",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
