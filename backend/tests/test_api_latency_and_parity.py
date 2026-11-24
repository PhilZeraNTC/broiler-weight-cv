"""Honest OOF preset parity, Watershed parity, < 500 ms latency SLA & concurrency tests.

Isolates the FastAPI test fixture via ``create_app(frontend_dist_dir=no_dist)`` and
verifies zero serving skew against ``models/presets_catalog.json`` across all 6 curated
single-bird OOF presets (including Dual-Lens invariance ``< 2.0 g``), all 4 Watershed
presets, warm p95 latency ``< 400 ms`` (every single request ``< 500 ms``), and
non-blocking concurrent execution via ``run_in_threadpool`` (D-02, D-03).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from backend.app.api.main import create_app
from backend.app.config import PRESETS_CATALOG_PATH


def _open_test_client(app: Any, **kwargs: Any) -> Any:
    """Instantiate FastAPI TestClient while filtering Starlette's httpx2 deprecation notice."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*httpx.*starlette\.testclient.*",
            category=UserWarning,
        )
        from fastapi.testclient import TestClient

    return TestClient(app, **kwargs)


@pytest.fixture(scope="module")
def api_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """Module-scoped API-only TestClient isolated from ``frontend/dist/``."""
    no_dist = tmp_path_factory.mktemp("no_dist")
    with _open_test_client(create_app(frontend_dist_dir=no_dist)) as client:
        yield client


def test_all_six_single_bird_presets_oof_parity_dual_lens_and_latency(api_client: Any) -> None:
    catalog = json.loads(PRESETS_CATALOG_PATH.read_text(encoding="utf-8"))
    single_presets = catalog["single_bird_presets"]
    assert len(single_presets) == 6

    # Warm OpenCV/LightGBM threadpool once before measuring warm p95 SLA.
    warm_resp = api_client.post("/api/predict", data={"preset_id": "sb_intermediate_grower"})
    assert warm_resp.status_code == 200

    live_preds: dict[str, float] = {}
    timings_ms: list[float] = []

    for entry in single_presets:
        preset_id = entry["preset_id"]
        resp = api_client.post("/api/predict", data={"preset_id": preset_id})
        assert resp.status_code == 200
        body = resp.json()

        assert body["source_type"] == "preset"
        assert body["preset_id"] == preset_id
        assert body["oof_evaluation"] is True
        assert body["oof_audit"] is not None
        assert body["oof_audit"]["fold"] == entry["fold"]
        assert body["oof_audit"]["session_id"] == entry["session_id"]
        assert body["oof_audit"]["ground_truth_weight_g"] == float(entry["ground_truth_weight_g"])

        pred_g = float(body["predicted_weight_g"])
        catalog_pred_g = float(entry["oof_predicted_g"])
        assert abs(pred_g - catalog_pred_g) <= 0.01, (
            f"OOF parity drift on {preset_id}: live={pred_g:.4f} vs catalog={catalog_pred_g:.4f}"
        )

        total_ms = float(body["timing"]["total_ms"])
        assert total_ms < 500.0, f"SLA breach on {preset_id}: {total_ms:.1f} ms >= 500 ms"
        assert body["timing"]["within_sla"] is True

        live_preds[preset_id] = pred_g
        timings_ms.append(total_ms)

    dual_lens_delta_g = abs(
        live_preds["sb_dual_lens_wide"] - live_preds["sb_dual_lens_ultrawide"]
    )
    assert dual_lens_delta_g < 2.0, (
        f"Dual-Lens live invariance failed: delta={dual_lens_delta_g:.4f} g >= 2.0 g"
    )

    p95_ms = float(np.percentile(timings_ms, 95))
    assert p95_ms < 400.0, f"Warm p95 latency {p95_ms:.1f} ms >= 400.0 ms"


def test_all_four_watershed_presets_count_parity_sum_consistency_and_latency(
    api_client: Any,
) -> None:
    catalog = json.loads(PRESETS_CATALOG_PATH.read_text(encoding="utf-8"))
    ws_presets = catalog["watershed_presets"]
    assert len(ws_presets) == 4

    expected_defaults = {"ws_417": 2, "ws_757": 3, "ws_823": 4, "ws_1520": 2}

    for entry in ws_presets:
        preset_id = entry["preset_id"]
        resp = api_client.post("/api/watershed-demo", data={"preset_id": preset_id})
        assert resp.status_code == 200
        body = resp.json()

        assert body["bird_count"] == entry["detected_bird_count_default"]
        assert body["bird_count"] == expected_defaults[preset_id]
        assert len(body["instances"]) == body["bird_count"]

        inst_sum = sum(float(inst["estimated_weight_g"]) for inst in body["instances"])
        assert abs(inst_sum - float(body["estimated_total_weight_g"])) < 0.05
        expected_mean = float(body["estimated_total_weight_g"]) / float(body["bird_count"])
        assert abs(float(body["estimated_mean_weight_g"]) - expected_mean) < 0.05

        total_ms = float(body["timing"]["total_ms"])
        assert total_ms < 500.0, f"Watershed SLA breach on {preset_id}: {total_ms:.1f} ms"
        assert body["timing"]["within_sla"] is True


def test_concurrent_requests_non_blocking_threadpool(api_client: Any) -> None:
    """Verify parallel /api/predict, /api/health, /api/metrics, and thumbnail calls succeed."""
    tasks = (
        ("predict", "sb_intermediate_grower", 424.0463),
        ("health", None, None),
        ("metrics", None, None),
        ("predict", "sb_young_chick", 249.7296),
        ("thumbnail", "ws_417", None),
        ("health", None, None),
    )

    def _run_one(item: tuple[str, str | None, float | None]) -> tuple[str, int, Any]:
        kind, arg, expected = item
        if kind == "predict":
            resp = api_client.post("/api/predict", data={"preset_id": arg})
            return kind, resp.status_code, (resp.json(), expected)
        if kind == "health":
            resp = api_client.get("/api/health")
            return kind, resp.status_code, resp.json()
        if kind == "metrics":
            resp = api_client.get("/api/metrics")
            return kind, resp.status_code, resp.json()
        resp = api_client.get(f"/api/presets/{arg}/thumbnail")
        return kind, resp.status_code, resp.headers["content-type"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_run_one, tasks))

    for kind, status_code, payload in results:
        assert status_code == 200, f"Concurrent {kind} request failed with {status_code}"
        if kind == "predict":
            body, expected_pred = payload
            assert abs(float(body["predicted_weight_g"]) - float(expected_pred)) <= 0.01
        elif kind == "health":
            assert payload["status"] == "ok"
        elif kind == "metrics":
            assert len(payload["evaluation_report"]["oof_scatter_points"]) == 1738
        elif kind == "thumbnail":
            assert str(payload).startswith("image/jpeg")
