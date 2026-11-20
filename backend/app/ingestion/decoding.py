"""Audit §9 forensic constants and filename/folder decoding utilities.

Implements verbatim curation rules from ``DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md``
Section 9 (D-04): Eastern-Arabic digit normalization, CP866 mojibake reversal,
numeric folder weight extraction, audited LCD weight corrections, quarantine sets,
lens classification, coarse PRD weight-band partitioning, and deterministic
SHA-256 image identifiers.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime

# Verbatim Audit §9 forensic constants (D-04)
ARABIC_TO_ASCII: dict[int, int | None] = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
MTIME_OFFSET_SEC: int = 14401
SESSION_GAP_SECONDS: float = 600.0
MULTI_BIRD_FOLDERS: frozenset[str] = frozenset({"417", "757", "823", "1520"})
SEVERE_MISLABEL_FOLDERS: frozenset[str] = frozenset({"1351", "1465"})
LCD_CORRECTIONS: dict[str, int] = {"367": 267}

# Optical lens & PRD coarse weight-band boundaries (D-03, D-05)
WIDE_FOCAL_THRESHOLD_MM: float = 3.0
BAND_LIGHT_MAX_EXCLUSIVE_G: int = 500
BAND_MEDIUM_MAX_INCLUSIVE_G: int = 1000

# Anchored linear-time regexes (ReDoS-free; T-01-05)
_IMG_TS_RE = re.compile(r"^IMG_(\d{8})_(\d{6})(?:_(\d+))?\.jpg$", re.IGNORECASE)
_WEIGHT_RENAMED_RE = re.compile(r"^\d+\s*\(\d+\)\.jpg$", re.IGNORECASE)
_FOLDER_WEIGHT_RE = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class DecodedFilename:
    """Result of decoding a raw broiler image filename under Audit §9 rules."""

    capture_dt_from_name: datetime | None
    filename_pattern: str
    was_cp866_recovered: bool


def _contains_arabic_indic_digits(text: str) -> bool:
    return any("\u0660" <= ch <= "\u0669" for ch in text)


def parse_folder_weight_g(folder_name: str) -> int:
    """Extract the integer gram weight from a leaf folder name (e.g. ``'1340_'`` -> ``1340``)."""
    match = _FOLDER_WEIGHT_RE.search(folder_name)
    if match is None:
        raise ValueError(f"No numeric weight found in folder name: {folder_name!r}")
    return int(match.group(0))


def decode_filename(filename: str) -> DecodedFilename:
    """Reverse CP866 mojibake (if present), normalize Arabic-Indic digits, and parse timestamp."""
    fixed = filename
    was_cp866 = False
    try:
        candidate = filename.encode("cp866").decode("utf-8")
        if _contains_arabic_indic_digits(candidate):
            fixed = candidate
            was_cp866 = True
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    has_arabic = _contains_arabic_indic_digits(fixed)
    ascii_name = fixed.translate(ARABIC_TO_ASCII)
    match = _IMG_TS_RE.match(ascii_name)

    capture_dt: datetime | None = None
    if match is not None:
        try:
            capture_dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            capture_dt = None

    if capture_dt is not None:
        if was_cp866:
            pattern = "img_mojibake_cp866"
        elif has_arabic and match is not None and match.group(3) is not None:
            pattern = "img_arabic_burst"
        elif has_arabic:
            pattern = "img_arabic"
        else:
            pattern = "img_ascii"
    elif _WEIGHT_RENAMED_RE.match(ascii_name):
        pattern = "weight_renamed"
    else:
        pattern = "other"

    return DecodedFilename(
        capture_dt_from_name=capture_dt,
        filename_pattern=pattern,
        was_cp866_recovered=(was_cp866 and capture_dt is not None),
    )


def classify_lens(focal_length_mm: float | None) -> str:
    """Classify optical lens family from EXIF focal length (``>= 3.0 mm`` wide vs ultrawide)."""
    if focal_length_mm is None or not math.isfinite(focal_length_mm):
        return "unknown"
    if focal_length_mm >= WIDE_FOCAL_THRESHOLD_MM:
        return "wide"
    return "ultrawide"


def assign_weight_band(weight_g: int) -> str:
    """Map integer gram weight into the 3 coarse PRD weight bands (D-05)."""
    if weight_g < BAND_LIGHT_MAX_EXCLUSIVE_G:
        return "<500g"
    if weight_g <= BAND_MEDIUM_MAX_INCLUSIVE_G:
        return "500-1000g"
    return ">1000g"


def compute_image_id(rel_path: str) -> str:
    """Compute a deterministic 16-hex SHA-256 identifier from a POSIX ``rel_path``."""
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:16]
