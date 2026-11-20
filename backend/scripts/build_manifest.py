"""Production CLI entrypoint for AvianScale CV forensic manifest generation.

Orchestrates the complete Phase 1 offline data curation pipeline strictly
read-only over ``data/raw/Broiler weights of Cobb Avian/``:
  1. Recursive path-contained dataset crawl & Audit §9 decoding (``scan_dataset_records``)
  2. Clean vs. quarantine partitioning (``split_clean_and_quarantine``)
  3. Deterministic 5-fold ``StratifiedGroupKFold`` session splitting (``assign_session_folds``)
  4. Zero-path-leakage aggregate audit JSON construction (``build_audit_summary``)
  5. Typed PyArrow Parquet + byte-deterministic JSON persistence (skipped on ``--dry-run``)
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import AUDIT_SUMMARY_PATH, PROCESSED_DIR, RAW_DATASET_DIR
from backend.app.ingestion.audit_summary import build_audit_summary, write_audit_summary
from backend.app.ingestion.decoding import BAND_MEDIUM_MAX_INCLUSIVE_G
from backend.app.ingestion.manifest import (
    scan_dataset_records,
    split_clean_and_quarantine,
    write_manifest_parquet,
)
from backend.app.ingestion.splits import (
    MIN_HEAVY_SESSIONS_PER_FOLD,
    assign_session_folds,
    verify_fold_integrity,
)

_AUTO_HEAVY_ENFORCEMENT_MIN_SESSIONS: int = 20


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for forensic manifest building."""
    parser = argparse.ArgumentParser(
        description="Build AvianScale CV forensic Parquet manifests and aggregate audit summary.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=RAW_DATASET_DIR,
        help="Root directory containing raw broiler weight JPEG folders (read-only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Output directory for generated Parquet manifests.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=AUDIT_SUMMARY_PATH,
        help="Destination path for the git-tracked aggregate JSON audit summary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full curation and split verification without writing any files or directories.",
    )
    parser.add_argument(
        "--min-heavy-sessions",
        type=int,
        default=None,
        help=(
            "Minimum heavy (>1000g) sessions required per fold. "
            "Default auto-enforces 4 on the real dataset (>= 20 heavy sessions); "
            "pass 0 to disable."
        ),
    )
    return parser.parse_args(argv)


def _resolve_min_heavy_sessions(
    dataset_root: Path,
    clean_df: pd.DataFrame,
    requested: int | None,
) -> int | None:
    if requested is not None:
        if requested < 0:
            raise ValueError(f"--min-heavy-sessions must be >= 0; got {requested}.")
        return None if requested == 0 else requested

    is_real_dataset = dataset_root.resolve() == RAW_DATASET_DIR.resolve()
    heavy_sessions = int(
        clean_df.loc[
            clean_df["weight_g"] > BAND_MEDIUM_MAX_INCLUSIVE_G, "session_id"
        ].nunique()
    )
    if is_real_dataset or heavy_sessions >= _AUTO_HEAVY_ENFORCEMENT_MIN_SESSIONS:
        return MIN_HEAVY_SESSIONS_PER_FOLD
    return None


def run_build_manifest(
    dataset_root: Path,
    output_dir: Path,
    summary_path: Path,
    *,
    dry_run: bool = False,
    min_heavy_sessions_per_fold: int | None = None,
) -> dict[str, Any]:
    """Execute end-to-end forensic curation, 5-fold splitting, and aggregate summary export."""
    full_df = scan_dataset_records(dataset_root)
    clean_df, quarantine_df = split_clean_and_quarantine(full_df)

    effective_min_heavy = _resolve_min_heavy_sessions(
        dataset_root=dataset_root,
        clean_df=clean_df,
        requested=min_heavy_sessions_per_fold,
    )
    clean_df = assign_session_folds(
        clean_df,
        min_heavy_sessions_per_fold=effective_min_heavy,
    )
    verify_fold_integrity(
        clean_df,
        min_heavy_sessions_per_fold=effective_min_heavy,
    )

    summary = build_audit_summary(clean_df, quarantine_df)

    if not dry_run:
        write_manifest_parquet(clean_df, output_dir / "manifest_clean.parquet")
        write_manifest_parquet(quarantine_df, output_dir / "manifest_quarantine.parquet")
        write_audit_summary(summary, summary_path)

    funnel = summary["curation_funnel"]
    split_audit = summary["split_audit"]
    # PROIBIÇÃO 4: log strictly aggregate metrics — never raw filenames or host paths.
    print(
        "Manifest build complete: "
        f"raw_images={funnel['raw_images']} "
        f"clean_images={funnel['clean_images']} "
        f"quarantined_total={funnel['quarantined_total']} "
        f"clean_unique_weights={funnel['clean_unique_weights']} "
        f"clean_sessions={funnel['clean_sessions']} "
        f"cross_fold_session_overlap={split_audit['cross_fold_session_overlap']} "
        f"dry_run={dry_run}"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    run_build_manifest(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        dry_run=args.dry_run,
        min_heavy_sessions_per_fold=args.min_heavy_sessions,
    )
    return 0


__all__ = ["parse_args", "run_build_manifest", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
