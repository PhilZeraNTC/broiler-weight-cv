"""Startup curated preset allowlist resolver & EXIF-transposed thumbnail cache (D-01..D-03).

Joins opaque ``image_id`` entries in ``models/presets_catalog.json`` strictly against
``manifest_clean.parquet`` (single-bird presets) and ``manifest_quarantine.parquet``
(Watershed presets), validates every ``rel_path`` under ``RAW_DATASET_DIR.resolve()``
via ASVS V4 per-segment non-symlink containment, discards ``rel_path`` immediately,
and pre-warms compact 320 px JPEG thumbnails in RAM (~103 KiB total).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from backend.app.api.schemas import (
    PresetsResponse,
    SingleBirdPresetItem,
    WatershedPresetItem,
)
from backend.app.config import (
    CLEAN_MANIFEST_PATH,
    PRESETS_CATALOG_PATH,
    QUARANTINE_MANIFEST_PATH,
    RAW_DATASET_DIR,
)
from backend.app.modeling.artifacts import (
    _assert_no_proibicao4_leaks,
    _resolve_safe_image_path,
)
from backend.app.pdi.io_exif import load_working_rgb

THUMBNAIL_LONG_EDGE: int = 320
THUMBNAIL_JPEG_QUALITY: int = 78
THUMBNAIL_CACHE_CONTROL: str = "public, max-age=3600"


@dataclass(frozen=True, slots=True)
class PresetStore:
    """Immutable startup allowlist and pre-warmed JPEG thumbnail cache keyed by ``preset_id``."""

    catalog: PresetsResponse
    single_bird_by_id: dict[str, SingleBirdPresetItem]
    watershed_by_id: dict[str, WatershedPresetItem]
    single_bird_paths: dict[str, Path]
    watershed_paths: dict[str, Path]
    thumbnails: dict[str, bytes]


def render_thumbnail_jpeg(
    image_path: Path,
    *,
    long_edge: int = THUMBNAIL_LONG_EDGE,
    quality: int = THUMBNAIL_JPEG_QUALITY,
) -> bytes:
    """Decode EXIF-transposed RGB via Pillow working loader and encode compact JPEG bytes."""
    working = load_working_rgb(image_path, working_long_edge=long_edge)
    bgr = cv2.cvtColor(working.rgb_work, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode preset JPEG thumbnail.")
    return buf.tobytes()


def _load_manifest_id_to_rel_path(manifest_path: Path) -> dict[str, Any]:
    """Load strictly ``image_id -> rel_path`` lookup from a Parquet manifest."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest parquet not found: {path.name}")
    df = pd.read_parquet(path, columns=["image_id", "rel_path"])
    lookup: dict[str, Any] = {}
    for raw_id, rel_path in zip(df["image_id"].tolist(), df["rel_path"].tolist(), strict=True):
        if raw_id is None or pd.isna(raw_id):
            continue
        img_id = str(raw_id).strip()
        if not img_id:
            continue
        if img_id in lookup:
            raise ValueError(f"Duplicate image_id {img_id!r} detected in manifest {path.name}.")
        lookup[img_id] = None if pd.isna(rel_path) else rel_path
    return lookup


def build_preset_store(
    *,
    catalog_path: Path = PRESETS_CATALOG_PATH,
    clean_manifest_path: Path = CLEAN_MANIFEST_PATH,
    quarantine_manifest_path: Path = QUARANTINE_MANIFEST_PATH,
    dataset_root: Path = RAW_DATASET_DIR,
    expected_pdi_config_hash: str | None = None,
) -> PresetStore:
    """Build verified preset allowlist, safe non-symlink paths, and RAM thumbnail cache."""
    cat_path = Path(catalog_path)
    if not cat_path.is_file():
        raise FileNotFoundError(f"Presets catalog not found: {cat_path.name}")

    raw_text = cat_path.read_text(encoding="utf-8")
    _assert_no_proibicao4_leaks(raw_text, context_label="presets_catalog.json")
    raw_catalog = json.loads(raw_text)

    if not isinstance(raw_catalog, dict):
        raise ValueError("presets_catalog.json must be a JSON object.")

    catalog_hash = str(raw_catalog.get("pdi_config_hash", ""))
    if expected_pdi_config_hash is not None and catalog_hash != expected_pdi_config_hash:
        raise ValueError(
            f"presets_catalog.json pdi_config_hash mismatch: {catalog_hash!r} != "
            f"{expected_pdi_config_hash!r}."
        )

    root_path = Path(dataset_root)
    if not root_path.is_dir():
        raise FileNotFoundError("Dataset root directory does not exist.")
    canonical_root = root_path.resolve()

    clean_lookup = _load_manifest_id_to_rel_path(clean_manifest_path)
    quarantine_lookup = _load_manifest_id_to_rel_path(quarantine_manifest_path)

    single_bird_items: list[SingleBirdPresetItem] = []
    watershed_items: list[WatershedPresetItem] = []
    single_bird_by_id: dict[str, SingleBirdPresetItem] = {}
    watershed_by_id: dict[str, WatershedPresetItem] = {}
    single_bird_paths: dict[str, Path] = {}
    watershed_paths: dict[str, Path] = {}
    thumbnails: dict[str, bytes] = {}

    for entry in raw_catalog.get("single_bird_presets", []):
        if not isinstance(entry, dict):
            raise ValueError("Each single_bird_presets entry must be a JSON object.")
        preset_id = str(entry.get("preset_id", "")).strip()
        image_id = str(entry.get("image_id", "")).strip()
        if not preset_id or preset_id in thumbnails:
            raise ValueError(f"Missing or duplicate single-bird preset_id: {preset_id!r}.")
        if image_id not in clean_lookup:
            raise ValueError(
                f"Single-bird preset {preset_id!r} references unknown clean image_id {image_id!r}."
            )
        resolved = _resolve_safe_image_path(clean_lookup[image_id], canonical_root)
        item = SingleBirdPresetItem(
            **entry,
            thumbnail_url=f"/api/presets/{preset_id}/thumbnail",
        )
        single_bird_items.append(item)
        single_bird_by_id[preset_id] = item
        single_bird_paths[preset_id] = resolved
        thumbnails[preset_id] = render_thumbnail_jpeg(resolved)

    for entry in raw_catalog.get("watershed_presets", []):
        if not isinstance(entry, dict):
            raise ValueError("Each watershed_presets entry must be a JSON object.")
        preset_id = str(entry.get("preset_id", "")).strip()
        image_id = str(entry.get("image_id", "")).strip()
        if not preset_id or preset_id in thumbnails:
            raise ValueError(f"Missing or duplicate Watershed preset_id: {preset_id!r}.")
        if image_id not in quarantine_lookup:
            raise ValueError(
                f"Watershed preset {preset_id!r} references unknown quarantine image_id "
                f"{image_id!r}."
            )
        resolved = _resolve_safe_image_path(quarantine_lookup[image_id], canonical_root)
        item = WatershedPresetItem(
            **entry,
            thumbnail_url=f"/api/presets/{preset_id}/thumbnail",
        )
        watershed_items.append(item)
        watershed_by_id[preset_id] = item
        watershed_paths[preset_id] = resolved
        thumbnails[preset_id] = render_thumbnail_jpeg(resolved)

    catalog = PresetsResponse(
        schema_version=str(raw_catalog["schema_version"]),
        pdi_config_hash=catalog_hash,
        winning_model_name=str(raw_catalog["winning_model_name"]),
        winning_ablation_condition=str(raw_catalog["winning_ablation_condition"]),
        single_bird_presets=single_bird_items,
        watershed_presets=watershed_items,
    )
    return PresetStore(
        catalog=catalog,
        single_bird_by_id=single_bird_by_id,
        watershed_by_id=watershed_by_id,
        single_bird_paths=single_bird_paths,
        watershed_paths=watershed_paths,
        thumbnails=thumbnails,
    )


__all__ = [
    "PresetStore",
    "THUMBNAIL_CACHE_CONTROL",
    "THUMBNAIL_JPEG_QUALITY",
    "THUMBNAIL_LONG_EDGE",
    "build_preset_store",
    "render_thumbnail_jpeg",
]
