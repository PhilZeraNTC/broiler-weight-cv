"""Public package barrel for AvianScale CV classical tabular modeling and bundling."""

from __future__ import annotations

from backend.app.modeling.age_stage import AgeStageResult, map_weight_to_age_stage
from backend.app.modeling.artifacts import (
    EVALUATION_REPORT_SCHEMA_VERSION,
    MODEL_BUNDLE_SCHEMA_VERSION,
    PRESETS_CATALOG_SCHEMA_VERSION,
    build_evaluation_report,
    build_model_bundle,
    build_presets_catalog,
    load_model_bundle,
    save_model_bundle,
    write_evaluation_report,
    write_presets_catalog,
)
from backend.app.modeling.datasets import ABLATION_CONDITION_ORDER, load_modeling_dataset
from backend.app.modeling.evaluation import run_full_evaluation
from backend.app.modeling.models import (
    MODEL_FAMILY_NAMES,
    build_regressor_pipeline,
    compute_sample_weights,
    predict_from_feature_dict,
)
from backend.app.modeling.wrappers import AllometricLogTransformer, LogDuanRegressor

__all__ = [
    "ABLATION_CONDITION_ORDER",
    "AgeStageResult",
    "AllometricLogTransformer",
    "EVALUATION_REPORT_SCHEMA_VERSION",
    "LogDuanRegressor",
    "MODEL_BUNDLE_SCHEMA_VERSION",
    "MODEL_FAMILY_NAMES",
    "PRESETS_CATALOG_SCHEMA_VERSION",
    "build_evaluation_report",
    "build_model_bundle",
    "build_presets_catalog",
    "build_regressor_pipeline",
    "compute_sample_weights",
    "load_model_bundle",
    "load_modeling_dataset",
    "map_weight_to_age_stage",
    "predict_from_feature_dict",
    "run_full_evaluation",
    "save_model_bundle",
    "write_evaluation_report",
    "write_presets_catalog",
]
