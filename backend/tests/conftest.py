"""Shared synthetic test fixtures for AvianScale CV Phase 1–4 tests.

All image and dataset fixtures are synthesized in memory or under ``tmp_path``
with zero reads from or mutations to ``data/raw/`` (PROIBIÇÃO 4).
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

_MTIME_OFFSET_SEC = 14401


def make_synthetic_jpeg_bytes(
    size: tuple[int, int] = (40, 20),
    orientation: int | None = 6,
    focal_length_mm: float | None = 4.25,
    zoom_multiple: float | None = 1.0,
    raw_xiaomi_tag: str | bytes | None = None,
    draw_corner_marker: bool = True,
) -> bytes:
    """Create a small asymmetric RGB JPEG with optional EXIF & Xiaomi sub-IFD tags."""
    width, height = size
    im = Image.new("RGB", (width, height), color=(40, 120, 200))
    if draw_corner_marker:
        px = im.load()
        marker_w = min(8, width // 3)
        marker_h = min(6, height // 2)
        # Distinct red top-left block and green top-right block to verify all 8 orientations.
        for x in range(marker_w):
            for y in range(marker_h):
                px[x, y] = (250, 10, 10)
        for x in range(width - marker_w, width):
            for y in range(marker_h):
                px[x, y] = (10, 240, 10)

    exif = Image.Exif()
    has_exif = False
    if orientation is not None:
        exif[274] = orientation
        has_exif = True

    if focal_length_mm is not None or zoom_multiple is not None or raw_xiaomi_tag is not None:
        ifd = exif.get_ifd(0x8769)
        if focal_length_mm is not None:
            ifd[37386] = focal_length_mm
        if raw_xiaomi_tag is not None:
            ifd[39321] = raw_xiaomi_tag
        elif zoom_multiple is not None:
            ifd[39321] = json.dumps({"zoomMultiple": zoom_multiple})
        has_exif = True

    buf = io.BytesIO()
    if has_exif:
        im.save(buf, format="JPEG", quality=95, subsampling=0, exif=exif.tobytes())
    else:
        im.save(buf, format="JPEG", quality=95, subsampling=0)
    return buf.getvalue()


def make_synthetic_png_bytes(
    size: tuple[int, int] = (24, 16),
    mode: str = "RGBA",
) -> bytes:
    """Create an EXIF-less PNG payload (default RGBA) for upload robustness tests."""
    color: tuple[int, ...] = (120, 180, 90, 255) if mode == "RGBA" else (120, 180, 90)
    im = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def synthetic_dataset_tree(tmp_path: Path) -> Path:
    """Build a miniature synthetic broiler dataset tree exercising every Audit §9 anomaly.

    Contains >= 5 clean sessions in every weight band (<500g, 500-1000g, >1000g)
    so the tree is directly compatible with 5-fold StratifiedGroupKFold tests,
    plus quarantined multi-bird (417/) and severe-mislabel (1351/) folders and
    an external symlink escape target to verify path-containment rejection.
    """
    root = tmp_path / "synthetic_broiler_dataset"
    root.mkdir(parents=True)

    wide_o6 = make_synthetic_jpeg_bytes(
        size=(40, 20), orientation=6, focal_length_mm=4.25, zoom_multiple=1.0
    )
    wide_o1 = make_synthetic_jpeg_bytes(
        size=(40, 20), orientation=1, focal_length_mm=4.25, zoom_multiple=1.3
    )
    ultrawide_o8 = make_synthetic_jpeg_bytes(
        size=(32, 16), orientation=8, focal_length_mm=1.65, zoom_multiple=0.6
    )

    def _write_jpg(
        rel_dir: str,
        filename: str,
        payload: bytes,
        mtime_dt: datetime | None = None,
    ) -> Path:
        folder = root / rel_dir
        folder.mkdir(parents=True, exist_ok=True)
        fp = folder / filename
        fp.write_bytes(payload)
        if mtime_dt is not None:
            ts = mtime_dt.timestamp()
            os.utime(fp, (ts, ts))
        return fp

    # <500g clean sessions (5 sessions total):
    # 1. 116/ — Windows-renamed file with calibrated mtime (2022-03-24 01:10:41)
    expected_116_dt = datetime(2022, 3, 24, 1, 10, 41)
    _write_jpg(
        "116",
        "116 (1).jpg",
        wide_o6,
        mtime_dt=expected_116_dt + timedelta(seconds=_MTIME_OFFSET_SEC),
    )

    # 2 & 3. 250/ — Arabic-Indic + same-second _1 burst (s0) + >600s gap (s1)
    _write_jpg("250", "IMG_٢٠٢٢٠٣٢٤_٠١١٠٤١.jpg", wide_o6)
    _write_jpg("250", "IMG_٢٠٢٢٠٣٢٤_٠١١٠٤١_1.jpg", wide_o6)
    _write_jpg("250", "IMG_٢٠٢٢٠٣٢٤_٠١٢٥٠٠.jpg", wide_o6)

    # 4. 367/ — Audited LCD correction folder (367 -> 267 g)
    _write_jpg("367", "IMG_٢٠٢٢٠٣٢٦_١٠٠٠٠٠.jpg", wide_o6)

    # 5. 171/173/ — Level-2 nested folder (leaf weight = 173 g)
    _write_jpg("171/173", "IMG_٢٠٢٢٠٣٢٤_٠٢٠٠٠٠.jpg", wide_o6)

    # 500-1000g clean sessions (5 sessions total):
    # 1 & 2. 547/ and 562/ — CP866 mojibake filenames with 2024 mtime to verify zero mtime fallback
    cp866_547 = "IMG_٢٠٢٢٠٣٢٨_١٢٠٠٠٠.jpg".encode().decode("cp866")
    cp866_562 = "IMG_٢٠٢٢٠٣٢٨_١٢٠٥٠٠.jpg".encode().decode("cp866")
    repack_2024 = datetime(2024, 6, 15, 12, 0, 0)
    _write_jpg("547", cp866_547, wide_o6, mtime_dt=repack_2024)
    _write_jpg("562", cp866_562, wide_o6, mtime_dt=repack_2024)

    # 3, 4, 5. 600/, 700/, 800/
    _write_jpg("600", "IMG_٢٠٢٢٠٣٣٠_٠٩٠٠٠٠.jpg", wide_o6)
    _write_jpg("700", "IMG_٢٠٢٢٠٤٠١_٠٩٠٠٠٠.jpg", ultrawide_o8)
    _write_jpg("800", "IMG_٢٠٢٢٠٤٠٣_٠٩٠٠٠٠.jpg", wide_o6)

    # >1000g clean sessions (5 sessions total):
    # 1. 1340_/ — Trailing underscore suffix sanitization -> 1340 g
    _write_jpg("1340_", "IMG_٢٠٢٢٠٤١١_١٠٠٠٠٠.jpg", ultrawide_o8)
    # 2, 3, 4, 5. 1400/, 1500/, 1600/, 1700/
    _write_jpg("1400", "IMG_٢٠٢٢٠٤١١_١١٠٠٠٠.jpg", ultrawide_o8)
    _write_jpg("1500", "IMG_٢٠٢٢٠٤١٣_١٠٠٠٠٠.jpg", wide_o6)
    _write_jpg("1600", "IMG_٢٠٢٢٠٤١٣_١١٠٠٠٠.jpg", ultrawide_o8)
    _write_jpg("1700", "IMG_٢٠٢٢٠٤١٥_١٠٠٠٠٠.jpg", wide_o1)

    # Quarantined folders: 417/ (multi-bird) and 1351/ (severe mislabel)
    _write_jpg("417", "IMG_٢٠٢٢٠٣٢٦_١٢٠٠٠٠.jpg", wide_o6)
    _write_jpg("1351", "IMG_٢٠٢٢٠٣٢٦_١٢١٠٠٠.jpg", wide_o6)

    # External JPEG linked via symlink inside dataset tree (must be rejected by path containment)
    outside_jpg = tmp_path / "outside_secret.jpg"
    outside_jpg.write_bytes(wide_o6)
    (root / "250" / "escaped_link.jpg").symlink_to(outside_jpg)

    return root
