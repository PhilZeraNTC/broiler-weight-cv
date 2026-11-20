"""CLI entrypoint for AvianScale CV forensic manifest generation.

Plan 01-01 tracer slice: crawls a dataset directory read-only, extracts EXIF
metadata via ``backend.app.pdi.io_exif.read_exif_metadata``, and writes a typed
PyArrow Parquet manifest. Plan 01-03 replaces the internal tracer helper with
the full Audit §9 forensic curation, 5-fold StratifiedGroupKFold, and audit
summary pipeline while preserving the CLI interface.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from backend.app.config import AUDIT_SUMMARY_PATH, PROCESSED_DIR, RAW_DATASET_DIR
from backend.app.pdi.io_exif import read_exif_metadata

_TRACER_COLUMNS = [
    "rel_path",
    "oriented_w",
    "oriented_h",
    "exif_orientation",
    "focal_length_mm",
    "zoom_multiple",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for manifest building."""
    parser = argparse.ArgumentParser(
        description="Build AvianScale CV forensic Parquet manifests from raw JPEGs.",
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
        help="Destination path for the aggregate JSON audit summary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect dataset and report aggregate counts without writing output files.",
    )
    return parser.parse_args(argv)


def run_tracer_ingest(
    dataset_root: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> int:
    """Safely crawl JPEGs under ``dataset_root`` and write a typed tracer Parquet."""
    resolved_root = dataset_root.resolve()
    rows: list[dict[str, object]] = []

    for dirpath, dirnames, filenames in os.walk(dataset_root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        dir_p = Path(dirpath)
        for fname in filenames:
            if not fname.lower().endswith(".jpg"):
                continue
            fp = dir_p / fname
            if fp.is_symlink():
                continue
            resolved_fp = fp.resolve()
            if not resolved_fp.is_relative_to(resolved_root):
                continue

            meta = read_exif_metadata(fp)
            rows.append(
                {
                    "rel_path": fp.relative_to(dataset_root).as_posix(),
                    "oriented_w": meta.oriented_w,
                    "oriented_h": meta.oriented_h,
                    "exif_orientation": meta.exif_orientation,
                    "focal_length_mm": meta.focal_length_mm,
                    "zoom_multiple": meta.zoom_multiple,
                }
            )

    if rows:
        df = pd.DataFrame(rows, columns=_TRACER_COLUMNS)
    else:
        df = pd.DataFrame(columns=_TRACER_COLUMNS)

    df["rel_path"] = df["rel_path"].astype("string")
    df["oriented_w"] = df["oriented_w"].astype("int32")
    df["oriented_h"] = df["oriented_h"].astype("int32")
    df["exif_orientation"] = df["exif_orientation"].astype("int8")
    df["focal_length_mm"] = df["focal_length_mm"].astype("float64")
    df["zoom_multiple"] = df["zoom_multiple"].astype("float64")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_dir / "manifest_clean.parquet", engine="pyarrow", index=False)

    print(f"Tracer ingest complete: {len(df)} images processed (dry_run={dry_run}).")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    return run_tracer_ingest(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
