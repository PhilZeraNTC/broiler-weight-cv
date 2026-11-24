"""Startup read-only audit JSON loader & PROIBIÇÃO 4 compliance guard (D-01)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.api.schemas import MetricsResponse
from backend.app.config import (
    AUDIT_SUMMARY_PATH,
    EVALUATION_REPORT_PATH,
    PDI_CALIBRATION_AUDIT_PATH,
)
from backend.app.modeling.artifacts import (
    EVALUATION_REPORT_SCHEMA_VERSION,
    _assert_no_proibicao4_leaks,
)

METRICS_CACHE_CONTROL: str = "public, max-age=300"


def _read_checked_json(path: Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Required audit JSON artifact not found: {target.name}")
    raw_text = target.read_text(encoding="utf-8")
    _assert_no_proibicao4_leaks(raw_text, context_label=target.name)
    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Audit JSON artifact {target.name} must be a JSON object.")
    return parsed


def load_metrics_payload(
    *,
    evaluation_report_path: Path = EVALUATION_REPORT_PATH,
    audit_summary_path: Path = AUDIT_SUMMARY_PATH,
    pdi_calibration_audit_path: Path = PDI_CALIBRATION_AUDIT_PATH,
    expected_pdi_config_hash: str,
) -> dict[str, Any]:
    """Load, hash-verify, leak-check, and schema-validate the unified `/api/metrics` payload."""
    evaluation_report = _read_checked_json(evaluation_report_path)
    dataset_audit_summary = _read_checked_json(audit_summary_path)
    pdi_calibration_audit = _read_checked_json(pdi_calibration_audit_path)

    eval_hash = str(evaluation_report.get("pdi_config_hash", ""))
    if eval_hash != expected_pdi_config_hash:
        raise ValueError(
            f"evaluation_report.json pdi_config_hash mismatch: {eval_hash!r} != "
            f"{expected_pdi_config_hash!r}."
        )

    pdi_hash = str(pdi_calibration_audit.get("pdi_config_hash", ""))
    if pdi_hash != expected_pdi_config_hash:
        raise ValueError(
            f"pdi_calibration_audit.json pdi_config_hash mismatch: {pdi_hash!r} != "
            f"{expected_pdi_config_hash!r}."
        )

    payload: dict[str, Any] = {
        "schema_version": str(
            evaluation_report.get("schema_version", EVALUATION_REPORT_SCHEMA_VERSION)
        ),
        "pdi_config_hash": expected_pdi_config_hash,
        "evaluation_report": evaluation_report,
        "dataset_audit_summary": dataset_audit_summary,
        "pdi_calibration_audit": pdi_calibration_audit,
    }
    MetricsResponse.model_validate(payload)
    _assert_no_proibicao4_leaks(
        json.dumps(payload, ensure_ascii=False),
        context_label="metrics_payload",
    )
    return payload


__all__ = [
    "METRICS_CACHE_CONTROL",
    "load_metrics_payload",
]
