"""CLI, security, atomic-write, determinism & real-artifact tests for ``train_and_evaluate.py``.

Verifies Phase 3 Plan 03 contracts (D-07..D-11, MOD-01..MOD-03):
- CLI defaults, ``--n-jobs`` clamping (``1..4``), and ``--dry-run`` zero-file writes with
  strictly aggregate-only stdout (PROIBIÇÃO 4).
- Synthetic two-run byte-identical SHA-256 parity on ``evaluation_report.json`` and
  ``presets_catalog.json``, plus clean-subprocess ``joblib.load()`` with
  ``-W error::UserWarning`` and ``PYTHONPATH=REPO_ROOT`` under ``[tool.uv] package = false``.
- ASVS V4 non-symlink path containment rejecting traversal, absolute paths, symlinks,
  backslashes, null bytes, and non-string ``rel_path`` entries in quarantine manifests.
- Defense-in-depth PROIBIÇÃO 4 leak guards and atomic ``os.replace()`` write preservation.
- ``load_model_bundle()`` corruption rejection and ``build_presets_catalog()`` empty-pool
  rejection.
- AST scan prohibiting DL/OCR imports, banned splitters, raw OpenCV loaders, and custom
  classes inside ``backend/scripts/train_and_evaluate.py``.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from backend.app.config import (
    CLEAN_MANIFEST_PATH,
    EVALUATION_REPORT_PATH,
    FEATURES_CLEAN_PATH,
    MODEL_BUNDLE_PATH,
    PRESETS_CATALOG_PATH,
    QUARANTINE_MANIFEST_PATH,
    RAW_DATASET_DIR,
    REPO_ROOT,
)
from backend.app.ingestion.decoding import assign_weight_band
from backend.app.modeling.artifacts import (
    _FORBIDDEN_LEAK_SUBSTRINGS,
    EVALUATION_REPORT_SCHEMA_VERSION,
    MODEL_BUNDLE_SCHEMA_VERSION,
    PRESETS_CATALOG_SCHEMA_VERSION,
    WATERSHED_FOLDER_SPECS,
    _resolve_safe_image_path,
    build_presets_catalog,
    load_model_bundle,
    save_model_bundle,
    validate_model_bundle,
    write_evaluation_report,
    write_presets_catalog,
)
from backend.app.modeling.datasets import LoadedModelingDataset, load_modeling_dataset
from backend.app.modeling.evaluation import FullEvaluationResult, evaluate_single_candidate
from backend.app.pdi.config import PDIConfig
from backend.app.pdi.features import FEATURE_COLUMNS
from backend.scripts import train_and_evaluate

_FORBIDDEN_LEAK_MARKERS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    "rel_path",
    "/home/",
    "data/raw",
)


def _make_synthetic_multibird_jpeg(
    *,
    n_birds: int,
    width: int = 480,
    height: int = 320,
    focal_length_mm: float = 4.25,
    zoom_multiple: float = 1.0,
) -> bytes:
    """Synthesize a clean multi-bird straw-bedding JPEG for fast Watershed preset tests."""
    rng = np.random.default_rng(1000 + n_birds)
    base = np.full((height, width, 3), (45, 78, 96), dtype=np.int16)
    noise = rng.integers(-16, 17, size=(height, width, 3), dtype=np.int16)
    img_bgr = np.clip(base + noise, 0, 255).astype(np.uint8)

    if n_birds == 4:
        centers = [(150, 110), (330, 110), (150, 220), (330, 220)]
        axes = (44, 34)
    else:
        centers = [(165, 160), (315, 160)]
        axes = (56, 42)

    for cx, cy in centers:
        cv2.ellipse(img_bgr, (cx, cy), axes, 0.0, 0, 360, (232, 236, 240), -1)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb, mode="RGB")

    exif = Image.Exif()
    exif[274] = 1
    ifd = exif.get_ifd(0x8769)
    ifd[37386] = focal_length_mm
    ifd[39321] = json.dumps({"zoomMultiple": zoom_multiple})

    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90, subsampling=0, exif=exif.tobytes())
    return buf.getvalue()


def _create_synthetic_cli_workspace(tmp_path: Path) -> dict[str, Path]:
    """Create a complete self-contained 5-fold clean + 4-folder quarantine synthetic workspace."""
    rng = np.random.default_rng(2026)
    cfg_hash = PDIConfig().config_hash()

    # 20 sessions per weight band (4 heavy sessions/fold == MIN_HEAVY_SESSIONS_PER_FOLD) x 4 = 240.
    band_specs: list[tuple[str, int, int]] = [
        ("light", 145, 235),
        ("medium", 500, 640),
        ("heavy", 1420, 1820),
    ]

    manifest_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    img_counter = 0

    for band_tag, w_lo, w_hi in band_specs:
        for s_idx in range(20):
            fold = int(s_idx % 5)
            weight_g = int(rng.integers(w_lo, w_hi + 1))
            weight_band = assign_weight_band(weight_g)
            session_id = f"{band_tag}_s{s_idx:03d}"

            # Realistic ~18% height-Z noise keeps OOF errors in median tier (>= 3%).
            scene_is_bucket = bool(s_idx % 3 == 0)
            scene_type = "bucket_dorsal" if scene_is_bucket else "scale_or_litter"

            for img_idx in range(4):
                is_uw = bool(img_idx % 2 == 1)
                focal_mm = 1.65 if is_uw else 4.25
                zoom = 0.6 if is_uw else 1.0
                lens_class = "ultrawide" if is_uw else "wide"

                z_jitter = float(np.exp(rng.normal(0.0, 0.18)))
                phys_area = 275.0 * (float(weight_g) ** (2.0 / 3.0)) * z_jitter
                area_px = phys_area * (zoom ** 2)
                sqrt_area_px = math.sqrt(area_px)
                perimeter_px = sqrt_area_px * 3.85
                major_axis_px = sqrt_area_px * 1.44
                minor_axis_px = sqrt_area_px * 0.91
                convex_area_px = area_px * 1.08
                sensor_px = (3264.0 * 1472.0) if is_uw else (4080.0 * 1840.0)
                area_frac = area_px / sensor_px

                image_id = f"aa{img_counter:014x}"
                img_counter += 1

                manifest_rows.append(
                    {
                        "image_id": image_id,
                        "rel_path": f"clean_synth/{image_id}.jpg",
                        "fold": np.int8(fold),
                        "session_id": session_id,
                        "weight_g": np.int32(weight_g),
                        "weight_band": weight_band,
                        "lens_class": lens_class,
                    }
                )

                feat_row: dict[str, object] = {
                    "image_id": image_id,
                    "fold": np.int8(fold),
                    "session_id": session_id,
                    "weight_g": np.int32(weight_g),
                    "weight_band": weight_band,
                    "lens_class": lens_class,
                    "scene_type": scene_type,
                    "scene_is_bucket": scene_is_bucket,
                    "lcd_excluded_px": np.int32(0 if scene_is_bucket else 360),
                    "segmentation_fallback": False,
                    "pdi_config_hash": cfg_hash,
                    "area_px": area_px,
                    "perimeter_px": perimeter_px,
                    "major_axis_px": major_axis_px,
                    "minor_axis_px": minor_axis_px,
                    "convex_area_px": convex_area_px,
                    "area_frac": area_frac,
                    "aspect_ratio_ellipse": major_axis_px / minor_axis_px,
                    "eccentricity": 0.72,
                    "solidity": area_px / convex_area_px,
                    "circularity": 0.78,
                    "extent_rotrect": 0.74,
                    "log_hu_1": 0.65,
                    "log_hu_2": 1.85,
                    "log_hu_3": 3.10,
                    "focal_length_mm": focal_mm,
                    "zoom_multiple": zoom,
                    "exif_optical_present": 1.0,
                    "area_optical_norm": area_px / (zoom ** 2),
                    "sqrt_area_optical_norm": math.sqrt(area_px / (zoom ** 2)),
                    "perim_optical_norm": perimeter_px / zoom,
                    "major_axis_optical_norm": major_axis_px / zoom,
                    "minor_axis_optical_norm": minor_axis_px / zoom,
                    "area_frac_optical_norm": area_frac / (zoom ** 2),
                }
                for col in FEATURE_COLUMNS:
                    if col not in feat_row:
                        if col.startswith("lab_a"):
                            feat_row[col] = 123.0 + 0.008 * float(weight_g) + float(
                                rng.normal(0.0, 1.4)
                            )
                        else:
                            feat_row[col] = float(rng.uniform(0.5, 110.0))

                feature_rows.append(feat_row)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df["fold"] = manifest_df["fold"].astype("int8")
    manifest_df["weight_g"] = manifest_df["weight_g"].astype("int32")

    features_cols = [
        "image_id",
        "fold",
        "session_id",
        "weight_g",
        "weight_band",
        "lens_class",
        "scene_type",
        "scene_is_bucket",
        "lcd_excluded_px",
        "segmentation_fallback",
        "pdi_config_hash",
        *FEATURE_COLUMNS,
    ]
    features_df = pd.DataFrame(feature_rows, columns=features_cols)
    features_df["fold"] = features_df["fold"].astype("int8")
    features_df["weight_g"] = features_df["weight_g"].astype("int32")
    features_df["lcd_excluded_px"] = features_df["lcd_excluded_px"].astype("int32")

    dataset_root = tmp_path / "raw_synth"
    dataset_root.mkdir(parents=True)

    quarantine_rows: list[dict[str, object]] = [
        # Include 1 severe_mislabel row to confirm multi_bird-only filtering.
        {
            "image_id": "ee00000000000001",
            "rel_path": "1351/mislabel_placeholder.jpg",
            "folder_chain": "1351",
            "quarantine_reason": "severe_mislabel",
            "lens_class": "wide",
            "weight_g": np.int32(1351),
        }
    ]
    for idx, (folder_id, expected_count, group_sum) in enumerate(WATERSHED_FOLDER_SPECS):
        folder_dir = dataset_root / folder_id
        folder_dir.mkdir(parents=True, exist_ok=True)
        rel_path = f"{folder_id}/group_{idx}.jpg"
        (dataset_root / rel_path).write_bytes(
            _make_synthetic_multibird_jpeg(n_birds=expected_count)
        )
        quarantine_rows.append(
            {
                "image_id": f"bb0000000000{idx:04x}",
                "rel_path": rel_path,
                "folder_chain": folder_id,
                "quarantine_reason": "multi_bird",
                "lens_class": "wide" if idx % 2 == 0 else "ultrawide",
                "weight_g": np.int32(group_sum),
            }
        )

    manifest_path = tmp_path / "manifest_clean.parquet"
    quarantine_path = tmp_path / "manifest_quarantine.parquet"
    features_path = tmp_path / "features_clean.parquet"
    manifest_df.to_parquet(manifest_path, engine="pyarrow", index=False)
    pd.DataFrame(quarantine_rows).to_parquet(quarantine_path, engine="pyarrow", index=False)
    features_df.to_parquet(features_path, engine="pyarrow", index=False)

    return {
        "manifest_path": manifest_path,
        "quarantine_path": quarantine_path,
        "features_path": features_path,
        "dataset_root": dataset_root,
    }


def test_parse_args_defaults_n_jobs_clamping_and_invalid_rejection() -> None:
    defaults = train_and_evaluate.parse_args([])
    assert defaults.manifest_path == CLEAN_MANIFEST_PATH
    assert defaults.quarantine_manifest_path == QUARANTINE_MANIFEST_PATH
    assert defaults.features_path == FEATURES_CLEAN_PATH
    assert defaults.dataset_root == RAW_DATASET_DIR
    assert defaults.bundle_path == MODEL_BUNDLE_PATH
    assert defaults.report_path == EVALUATION_REPORT_PATH
    assert defaults.presets_path == PRESETS_CATALOG_PATH
    assert defaults.n_jobs == train_and_evaluate.MAX_N_JOBS_CAP == 4
    assert defaults.dry_run is False

    assert train_and_evaluate.parse_args(["--n-jobs", "1"]).n_jobs == 1
    assert train_and_evaluate.parse_args(["--n-jobs", "3"]).n_jobs == 3
    assert train_and_evaluate.parse_args(["--n-jobs", "16"]).n_jobs == 4

    for invalid in ("0", "-1", "-4"):
        with pytest.raises(ValueError, match="n-jobs"):
            train_and_evaluate.parse_args(["--n-jobs", invalid])

    with pytest.raises(ValueError, match="n-jobs"):
        train_and_evaluate._resolve_effective_n_jobs(True)


def test_cli_dry_run_writes_zero_files_and_emits_aggregate_only_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ws = _create_synthetic_cli_workspace(tmp_path)
    out_bundle = tmp_path / "dry_bundle_dir" / "weight_regressor.joblib"
    out_report = tmp_path / "dry_report_dir" / "evaluation_report.json"
    out_presets = tmp_path / "dry_presets_dir" / "presets_catalog.json"

    rc = train_and_evaluate.main(
        [
            "--manifest-path",
            str(ws["manifest_path"]),
            "--quarantine-manifest-path",
            str(ws["quarantine_path"]),
            "--features-path",
            str(ws["features_path"]),
            "--dataset-root",
            str(ws["dataset_root"]),
            "--bundle-path",
            str(out_bundle),
            "--report-path",
            str(out_report),
            "--presets-path",
            str(out_presets),
            "--n-jobs",
            "4",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    for target in (out_bundle, out_report, out_presets):
        assert not target.exists()
        assert not target.parent.exists()

    lowered = captured.out.lower()
    for forbidden in _FORBIDDEN_LEAK_MARKERS:
        assert forbidden not in lowered, f"PROIBIÇÃO 4 leak {forbidden!r} in CLI stdout"

    assert "winning_model=" in captured.out
    assert "winning_condition=" in captured.out
    assert "n_single_presets=6" in captured.out
    assert "n_watershed_presets=4" in captured.out
    assert "dry_run=True" in captured.out


def test_cli_two_run_byte_identical_sha256_and_subprocess_joblib_load(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ws = _create_synthetic_cli_workspace(tmp_path)
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"

    for out_dir, req_jobs in ((run1, "16"), (run2, "4")):
        rc = train_and_evaluate.main(
            [
                "--manifest-path",
                str(ws["manifest_path"]),
                "--quarantine-manifest-path",
                str(ws["quarantine_path"]),
                "--features-path",
                str(ws["features_path"]),
                "--dataset-root",
                str(ws["dataset_root"]),
                "--bundle-path",
                str(out_dir / "weight_regressor.joblib"),
                "--report-path",
                str(out_dir / "evaluation_report.json"),
                "--presets-path",
                str(out_dir / "presets_catalog.json"),
                "--n-jobs",
                req_jobs,
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0
        assert "n_jobs=4" in captured.out
        assert "dry_run=False" in captured.out
        for forbidden in _FORBIDDEN_LEAK_MARKERS:
            assert forbidden not in captured.out.lower()

    report_1 = run1 / "evaluation_report.json"
    report_2 = run2 / "evaluation_report.json"
    presets_1 = run1 / "presets_catalog.json"
    presets_2 = run2 / "presets_catalog.json"
    bundle_1 = run1 / "weight_regressor.joblib"

    # Byte-identical SHA-256 across consecutive runs for BOTH JSON artifacts.
    assert hashlib.sha256(report_1.read_bytes()).hexdigest() == hashlib.sha256(
        report_2.read_bytes()
    ).hexdigest()
    assert hashlib.sha256(presets_1.read_bytes()).hexdigest() == hashlib.sha256(
        presets_2.read_bytes()
    ).hexdigest()

    # PROIBIÇÃO 4 zero-path-leakage verification on raw JSON text.
    raw_report = report_1.read_text(encoding="utf-8")
    raw_presets = presets_1.read_text(encoding="utf-8")
    for forbidden in _FORBIDDEN_LEAK_MARKERS:
        assert forbidden not in raw_report.lower()
        assert forbidden not in raw_presets.lower()

    report_obj = json.loads(raw_report)
    assert report_obj["schema_version"] == EVALUATION_REPORT_SCHEMA_VERSION
    assert report_obj["pdi_config_hash"] == PDIConfig().config_hash()
    assert report_obj["dataset_summary"]["n_clean_images"] == 240
    assert report_obj["dataset_summary"]["cross_fold_session_overlap"] == 0
    assert len(report_obj["model_comparison"]) == 16
    assert len(report_obj["mechanics_ablation"]) == 4
    assert len(report_obj["allometric_log_ablation"]) == 4
    assert len(report_obj["oof_scatter_points"]) == 240
    assert "r2" in report_obj["winning_model"]["global"]
    assert "session_averaged_r2" in report_obj["winning_model"]["global"]
    assert "wall_clock_sec" not in raw_report

    presets_obj = json.loads(raw_presets)
    assert presets_obj["schema_version"] == PRESETS_CATALOG_SCHEMA_VERSION
    assert len(presets_obj["single_bird_presets"]) == 6
    assert len(presets_obj["watershed_presets"]) == 4
    sb_folds = [int(p["fold"]) for p in presets_obj["single_bird_presets"]]
    assert set(sb_folds) == {0, 1, 2, 3, 4}
    assert [p["preset_id"] for p in presets_obj["watershed_presets"]] == [
        "ws_417",
        "ws_757",
        "ws_823",
        "ws_1520",
    ]

    # Clean-subprocess joblib.load() + single-row prediction with -W error::UserWarning.
    features_df = pd.read_parquet(ws["features_path"])
    sample_row = {col: float(features_df.iloc[0][col]) for col in FEATURE_COLUMNS}
    sub_code = (
        "import json, math, time\n"
        "from pathlib import Path\n"
        "from backend.app.modeling import load_model_bundle, predict_from_feature_dict\n"
        "from backend.app.pdi.config import PDIConfig\n"
        f"bundle = load_model_bundle(Path({str(bundle_1)!r}), "
        "expected_pdi_config_hash=PDIConfig().config_hash())\n"
        f"row = json.loads({json.dumps(sample_row)!r})\n"
        "fn = bundle['feature_names']\n"
        "_ = predict_from_feature_dict(bundle['production_pipeline'], fn, row)\n"
        "t0 = time.perf_counter()\n"
        "y_prod = predict_from_feature_dict(bundle['production_pipeline'], fn, row)\n"
        "lat_ms = (time.perf_counter() - t0) * 1000.0\n"
        "y_fold0 = predict_from_feature_dict(bundle['fold_pipelines'][0], fn, row)\n"
        "assert math.isfinite(y_prod) and y_prod > 0.0\n"
        "assert math.isfinite(y_fold0) and y_fold0 > 0.0\n"
        "assert lat_ms < 5.0, f'latency {lat_ms:.3f} ms >= 5.0 ms'\n"
        "print(json.dumps({'y_prod': y_prod, 'y_fold0': y_fold0, 'lat_ms': lat_ms}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", "-c", sub_code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        check=True,
        capture_output=True,
        text=True,
    )
    sub_out = json.loads(proc.stdout.strip())
    assert sub_out["lat_ms"] < 5.0


@pytest.mark.parametrize(
    "case_name",
    [
        "parent_traversal",
        "absolute_path",
        "external_symlink",
        "internal_symlink",
        "symlink_dir",
        "backslash",
        "null_byte",
        "non_string",
    ],
)
def test_asvs_v4_quarantine_path_traversal_and_symlink_rejection(
    tmp_path: Path,
    case_name: str,
) -> None:
    dataset_root = tmp_path / "root"
    (dataset_root / "417").mkdir(parents=True)
    valid_bytes = _make_synthetic_multibird_jpeg(n_birds=2)
    inside_file = dataset_root / "417" / "real.jpg"
    inside_file.write_bytes(valid_bytes)

    outside_file = tmp_path / "secret_outside.jpg"
    outside_file.write_bytes(valid_bytes)

    if case_name == "parent_traversal":
        bad_rel: Any = "../secret_outside.jpg"
    elif case_name == "absolute_path":
        bad_rel = str(outside_file)
    elif case_name == "external_symlink":
        (dataset_root / "417" / "escape.jpg").symlink_to(outside_file)
        bad_rel = "417/escape.jpg"
    elif case_name == "internal_symlink":
        (dataset_root / "417" / "alias.jpg").symlink_to(inside_file)
        bad_rel = "417/alias.jpg"
    elif case_name == "symlink_dir":
        outside_dir = tmp_path / "outside_dir"
        outside_dir.mkdir()
        (outside_dir / "nested.jpg").write_bytes(valid_bytes)
        (dataset_root / "linked_dir").symlink_to(outside_dir)
        bad_rel = "linked_dir/nested.jpg"
    elif case_name == "backslash":
        bad_rel = "417\\real.jpg"
    elif case_name == "null_byte":
        bad_rel = "417/real.jpg\x00"
    else:
        bad_rel = None

    with pytest.raises(ValueError, match="ASVS V4"):
        _resolve_safe_image_path(bad_rel, dataset_root.resolve())


def _make_fast_eval_result(dataset: LoadedModelingDataset) -> FullEvaluationResult:
    """Build a lightweight 5-fold RidgeLog FullEvaluationResult (<0.2s) for preset-gate tests."""
    cand = evaluate_single_candidate(
        dataset,
        "RidgeLog",
        "C2_physics_normalized_strict",
        n_jobs=1,
    )
    return FullEvaluationResult(
        candidates=[cand],
        winner=cand,
        mechanics_ablation=[],
        allometric_log_ablation=[],
        optical_ablation={},
        feature_importances=[],
        oof_scatter_points=[],
    )


@pytest.mark.parametrize("attack_mode", ["parent_traversal", "external_symlink"])
def test_asvs_v4_build_presets_catalog_rejects_malicious_quarantine_manifest_rows(
    tmp_path: Path,
    attack_mode: str,
) -> None:
    ws = _create_synthetic_cli_workspace(tmp_path)
    dataset = load_modeling_dataset(
        manifest_path=ws["manifest_path"],
        features_path=ws["features_path"],
    )
    eval_result = _make_fast_eval_result(dataset)
    quarantine_df = pd.read_parquet(ws["quarantine_path"]).copy()

    outside_file = tmp_path / "outside_escape.jpg"
    outside_file.write_bytes(_make_synthetic_multibird_jpeg(n_birds=2))

    if attack_mode == "parent_traversal":
        malicious_rel = "../outside_escape.jpg"
    else:
        symlink_path = ws["dataset_root"] / "417" / "symlink_escape.jpg"
        symlink_path.symlink_to(outside_file)
        malicious_rel = "417/symlink_escape.jpg"

    mask = quarantine_df["folder_chain"].astype(str) == "417"
    quarantine_df.loc[mask, "rel_path"] = malicious_rel

    with pytest.raises(ValueError, match="ASVS V4"):
        build_presets_catalog(dataset, eval_result, quarantine_df, ws["dataset_root"])


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"leak": "116/116 (1).jpg"},
        {"leak": "116/IMAGE.JPEG"},
        {"leak": "116/mask.png"},
        {"rel_path": "116/opaque"},
        {"host": "/home/user/broiler"},
        {"dir": "data/raw/broiler"},
    ],
)
def test_proibicao4_leak_guards_and_atomic_write_preserves_existing_target(
    tmp_path: Path,
    bad_payload: dict[str, Any],
) -> None:
    assert _FORBIDDEN_LEAK_SUBSTRINGS == _FORBIDDEN_LEAK_MARKERS

    for writer in (write_evaluation_report, write_presets_catalog):
        fresh_target = tmp_path / "nonexistent_dir" / f"{writer.__name__}.json"
        with pytest.raises(ValueError, match="PROIBIÇÃO 4"):
            writer(bad_payload, fresh_target)
        assert not fresh_target.exists()
        assert not fresh_target.parent.exists()

        existing_target = tmp_path / f"existing_{writer.__name__}.json"
        original_content = '{"schema_version": "1.0.0"}\n'
        existing_target.write_text(original_content, encoding="utf-8")
        with pytest.raises(ValueError, match="PROIBIÇÃO 4"):
            writer(bad_payload, existing_target)
        assert existing_target.read_text(encoding="utf-8") == original_content

    # Non-finite NaN rejection preserves existing target and leaves no sibling temp files.
    target_nan = tmp_path / "atomic_nan.json"
    target_nan.write_text('{"ok": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        write_evaluation_report({"bad": math.nan}, target_nan)
    assert target_nan.read_text(encoding="utf-8") == '{"ok": true}\n'
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_cleans_up_temp_file_and_preserves_target_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "evaluation_report.json"
    original = '{"preserved": true}\n'
    target.write_text(original, encoding="utf-8")

    def _boom(src: Any, dst: Any) -> None:
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        write_evaluation_report({"preserved": False}, target)

    assert target.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.tmp"))


def test_load_model_bundle_rejects_corrupt_schema_hash_folds_and_features(tmp_path: Path) -> None:
    class _DummyPipe:
        def predict(self, x: Any) -> np.ndarray:
            return np.ones(len(x), dtype=np.float64)

    def _make_valid_bundle() -> dict[str, Any]:
        return {
            "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
            "model_version": "avianscale-cv-v1.0.0",
            "pdi_config": {"working_long_edge": 768},
            "pdi_config_hash": PDIConfig().config_hash(),
            "winning_model_name": "LightGBMLog",
            "winning_ablation_condition": "C2_physics_normalized_strict",
            "feature_names": ("area_optical_norm",),
            "weighting_mode": "session_band_capped_5x",
            "duan_smearing_enabled": False,
            "production_pipeline": _DummyPipe(),
            "fold_pipelines": {k: _DummyPipe() for k in range(5)},
            "band_error_intervals": {
                band: {
                    "n": 10,
                    "mae_g": 10.0,
                    "mape_pct": 5.0,
                    "median_abs_err_g": 8.0,
                    "p68_abs_err_g": 12.0,
                    "p90_abs_err_g": 20.0,
                    "mbe_g": 0.0,
                }
                for band in ("<500g", "500-1000g", ">1000g")
            },
            "valid_weight_range_g": {"min": 116, "max": 2093},
            "valid_age_days_range": "~7-29 days",
        }

    # Valid bundle passes validation.
    valid = _make_valid_bundle()
    validate_model_bundle(valid, expected_pdi_config_hash=PDIConfig().config_hash())

    # Corrupt serialized non-dict rejected by load_model_bundle.
    non_dict_path = tmp_path / "non_dict.joblib"
    joblib.dump(["not", "a", "dict"], non_dict_path)
    with pytest.raises(ValueError, match="dict"):
        load_model_bundle(non_dict_path)

    # Missing required key.
    missing_key = _make_valid_bundle()
    del missing_key["fold_pipelines"]
    with pytest.raises(ValueError, match="missing required"):
        save_model_bundle(missing_key, tmp_path / "missing.joblib")
    assert not (tmp_path / "missing.joblib").exists()

    # Wrong schema_version.
    bad_schema = _make_valid_bundle()
    bad_schema["schema_version"] = "2.0.0"
    with pytest.raises(ValueError, match="schema_version"):
        validate_model_bundle(bad_schema)

    # Mismatched pdi_config_hash.
    bad_hash = _make_valid_bundle()
    bad_hash["pdi_config_hash"] = "0000000000000000"
    with pytest.raises(ValueError, match="pdi_config_hash mismatch"):
        validate_model_bundle(bad_hash, expected_pdi_config_hash=PDIConfig().config_hash())

    # Empty or non-tuple feature_names.
    empty_fn = _make_valid_bundle()
    empty_fn["feature_names"] = ()
    with pytest.raises(ValueError, match="feature_names"):
        validate_model_bundle(empty_fn)

    list_fn = _make_valid_bundle()
    list_fn["feature_names"] = ["area_optical_norm"]
    with pytest.raises(ValueError, match="feature_names"):
        validate_model_bundle(list_fn)

    # Incomplete fold_pipelines (!= {0..4}).
    incomplete_folds = _make_valid_bundle()
    del incomplete_folds["fold_pipelines"][4]
    with pytest.raises(ValueError, match="fold_pipelines"):
        validate_model_bundle(incomplete_folds)

    # Missing weight band interval.
    bad_bands = _make_valid_bundle()
    del bad_bands["band_error_intervals"][">1000g"]
    with pytest.raises(ValueError, match="band_error_intervals"):
        validate_model_bundle(bad_bands)


def test_build_presets_catalog_rejects_empty_single_bird_and_watershed_pools(
    tmp_path: Path,
) -> None:
    ws = _create_synthetic_cli_workspace(tmp_path)
    dataset = load_modeling_dataset(
        manifest_path=ws["manifest_path"],
        features_path=ws["features_path"],
    )
    eval_result = _make_fast_eval_result(dataset)
    quarantine_df = pd.read_parquet(ws["quarantine_path"])

    # 1. All single-bird rows marked segmentation_fallback=True -> ValueError.
    fb_df = dataset.df.copy()
    fb_df.loc[:, "segmentation_fallback"] = True
    fb_dataset = type(dataset)(
        df=fb_df,
        pdi_config=dataset.pdi_config,
        pdi_config_hash=dataset.pdi_config_hash,
        fold_audit=dataset.fold_audit,
        ablation_columns=dataset.ablation_columns,
    )
    with pytest.raises(ValueError, match="segmentation_fallback"):
        build_presets_catalog(fb_dataset, eval_result, quarantine_df, ws["dataset_root"])

    # 2. Empty >1000g bucket_dorsal pool -> ValueError.
    no_bucket_df = dataset.df.copy()
    no_bucket_df.loc[no_bucket_df["weight_band"] == ">1000g", "scene_type"] = "scale_or_litter"
    no_bucket_ds = type(dataset)(
        df=no_bucket_df,
        pdi_config=dataset.pdi_config,
        pdi_config_hash=dataset.pdi_config_hash,
        fold_audit=dataset.fold_audit,
        ablation_columns=dataset.ablation_columns,
    )
    with pytest.raises(ValueError, match="sb_hard_white_bucket"):
        build_presets_catalog(no_bucket_ds, eval_result, quarantine_df, ws["dataset_root"])

    # 3. Missing one of the 4 multi-bird Watershed folders -> ValueError.
    partial_q = quarantine_df.loc[quarantine_df["folder_chain"] != "823"].reset_index(drop=True)
    with pytest.raises(ValueError, match="823"):
        build_presets_catalog(dataset, eval_result, partial_q, ws["dataset_root"])

    # 4. Empty quarantine manifest -> ValueError.
    empty_q = quarantine_df.iloc[0:0].copy()
    with pytest.raises(ValueError, match="empty quarantine"):
        build_presets_catalog(dataset, eval_result, empty_q, ws["dataset_root"])


def test_ast_scan_train_and_evaluate_prohibitions_and_no_main_classes() -> None:
    cli_path = REPO_ROOT / "backend" / "scripts" / "train_and_evaluate.py"
    artifacts_path = REPO_ROOT / "backend" / "app" / "modeling" / "artifacts.py"

    banned_dl_ocr = {
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "keras",
        "ultralytics",
        "timm",
        "onnxruntime",
        "paddleocr",
        "easyocr",
    }
    banned_splitters = {
        "train_test_split",
        "KFold",
        "StratifiedKFold",
        "ShuffleSplit",
        "StratifiedShuffleSplit",
        "GroupShuffleSplit",
    }
    banned_cv_loaders = {"imread", "imdecode"}

    for py_file in (cli_path, artifacts_path):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if py_file == cli_path:
                assert not isinstance(node, ast.ClassDef), (
                    f"Custom class {node.name!r} in train_and_evaluate.py creates __main__ "
                    "pickling risk."
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned_dl_ocr
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module.split(".")[0] not in banned_dl_ocr
                for alias in node.names:
                    assert alias.name not in banned_splitters
                    if node.module == "cv2":
                        assert alias.name not in banned_cv_loaders
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    assert func.id not in banned_splitters
                    assert func.id not in banned_cv_loaders
                elif isinstance(func, ast.Attribute):
                    assert func.attr not in banned_splitters
                    assert func.attr not in banned_cv_loaders
