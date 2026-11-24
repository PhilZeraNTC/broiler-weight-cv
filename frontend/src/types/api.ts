/**
 * Strict TypeScript REST API contracts mirroring `backend/app/api/schemas.py`
 * and published `models/*.json` artifacts (PROIBIÇÃO 4: opaque IDs only, zero filesystem paths).
 */

export type WeightBand = "<500g" | "500-1000g" | ">1000g";
export type LensClass = "wide" | "ultrawide";
export type OpticalLensClass = LensClass | "unknown";
export type SceneType = "bucket_dorsal" | "scale_or_litter";
export type SourceType = "preset" | "upload";

export interface HealthResponse {
  status: "ok";
  model_version: string;
  winning_model_name: string;
  winning_ablation_condition: string;
  pdi_config_hash: string;
  n_features: number;
  single_bird_presets_count: number;
  watershed_presets_count: number;
  classical_pdi_only: true;
}

export interface SingleBirdPresetItem {
  preset_id: string;
  label: string;
  category: string;
  pair_group_id: string | null;
  image_id: string;
  session_id: string;
  fold: number;
  ground_truth_weight_g: number;
  weight_band: WeightBand;
  lens_class: LensClass;
  scene_type: SceneType;
  oof_predicted_g: number;
  oof_abs_err_g: number;
  oof_pct_err: number;
  thumbnail_url: string;
}

export interface WatershedPresetItem {
  preset_id: string;
  label: string;
  folder_id: string;
  image_id: string;
  lens_class: LensClass;
  expected_bird_count: number;
  detected_bird_count_default: number;
  group_sum_weight_g: number;
  mean_bird_weight_reference_g: number;
  ground_truth_scope: "group_sum_only";
  disclaimer: string;
  thumbnail_url: string;
}

export interface PresetsResponse {
  schema_version: string;
  pdi_config_hash: string;
  winning_model_name: string;
  winning_ablation_condition: string;
  single_bird_presets: SingleBirdPresetItem[];
  watershed_presets: WatershedPresetItem[];
}

export interface BandErrorInterval {
  n: number;
  mae_g: number;
  mape_pct: number;
  median_abs_err_g: number;
  p68_abs_err_g: number;
  p90_abs_err_g: number;
  mbe_g: number;
}

export interface AgeStagePayload {
  stage_code: string;
  stage_label: string;
  estimated_age_days_range: string;
  out_of_training_range: boolean;
  valid_training_range_g: [number, number];
  valid_age_days_range: string;
}

export interface OofAuditPayload {
  fold: number;
  session_id: string;
  ground_truth_weight_g: number;
  abs_err_g: number;
  pct_err: number;
  signed_err_g: number;
  pair_group_id: string | null;
}

export interface OpticalPayload {
  focal_length_mm: number;
  zoom_multiple: number;
  exif_optical_present: boolean;
  override_applied: boolean;
  lens_class: OpticalLensClass;
}

export interface SegmentationAuditPayload {
  scene_type: SceneType;
  scene_is_bucket: boolean;
  lcd_excluded_px: number;
  lcd_core_count: number;
  segmentation_fallback: boolean;
  mask_area_frac: number;
  ellipse_from_min_area_rect: boolean;
}

export interface XRayPayload {
  stage1_original: string;
  stage2_clahe: string;
  stage3_mask: string;
  stage4_overlay: string;
  display_w: number;
  display_h: number;
  total_b64_bytes: number;
}

export interface TimingPayload {
  pdi_ms: number;
  inference_ms: number;
  total_ms: number;
  sla_target_ms: number;
  within_sla: boolean;
}

export interface PredictResponse {
  source_type: SourceType;
  preset_id: string | null;
  oof_evaluation: boolean;
  model_version: string;
  winning_model_name: string;
  pdi_config_hash: string;
  predicted_weight_g: number;
  predicted_weight_kg: number;
  weight_band: WeightBand;
  band_error_interval: BandErrorInterval;
  age_stage: AgeStagePayload;
  oof_audit: OofAuditPayload | null;
  optical: OpticalPayload;
  segmentation: SegmentationAuditPayload;
  xray: XRayPayload;
  features_summary: Record<string, number>;
  model_features: Record<string, number>;
  timing: TimingPayload;
}

export interface WatershedParamBounds {
  h_rel: [number, number];
  min_distance_rel: [number, number];
  min_area_ratio: [number, number];
}

export interface WatershedParamDefaults {
  h_rel: number;
  min_distance_rel: number;
  min_area_ratio: number;
}

export interface WatershedParamsPayload {
  h_rel: number;
  min_distance_rel: number;
  min_area_ratio: number;
  defaults: WatershedParamDefaults;
  bounds: WatershedParamBounds;
}

export interface WatershedPresetReference {
  folder_id: string;
  expected_bird_count: number;
  group_sum_weight_g: number;
  mean_bird_weight_reference_g: number;
  sum_diff_g: number;
  sum_pct_err: number;
}

export interface WatershedInstancePayload {
  instance_id: number;
  color_hex: string;
  centroid_xy: [number, number];
  area_work_px: number;
  area_px: number;
  area_frac: number;
  area_optical_norm: number;
  estimated_weight_g: number;
  share_of_total_pct: number;
  age_stage: AgeStagePayload;
}

export interface WatershedVisualsPayload {
  stage1_original: string;
  stage2_union_mask: string;
  stage3_distance_peaks: string;
  stage4_color_basins: string;
  total_b64_bytes: number;
}

export interface WatershedResponse {
  source_type: SourceType;
  preset_id: string | null;
  ground_truth_scope: "group_sum_only";
  disclaimer: string;
  bird_count: number;
  estimated_total_weight_g: number;
  estimated_mean_weight_g: number | null;
  preset_reference: WatershedPresetReference | null;
  parameters: WatershedParamsPayload;
  optical: OpticalPayload;
  red_feeder_excluded_px: number;
  pdi_config_hash: string;
  instances: WatershedInstancePayload[];
  visuals: WatershedVisualsPayload;
  timing: TimingPayload;
}

export interface OofScatterPoint {
  actual_g: number;
  predicted_g: number;
  abs_err_g: number;
  pct_err: number;
  fold: number;
  image_id: string;
  session_id: string;
  lens_class: LensClass;
  scene_type: SceneType;
  weight_band: WeightBand;
}

export interface MetricSlice {
  n: number;
  mae_g: number;
  rmse_g: number;
  mape_pct: number;
  median_abs_err_g: number;
  p68_abs_err_g: number;
  p90_abs_err_g: number;
  mbe_g: number;
  r2: number | null;
  session_averaged_r2?: number | null;
}

export interface SessionLevelMetrics {
  n_sessions: number;
  mae_g: number;
  rmse_g: number;
  mape_pct: number;
  median_abs_err_g: number;
  p68_abs_err_g: number;
  p90_abs_err_g: number;
  mbe_g: number;
  session_averaged_r2: number;
}

export interface CounterfactualShiftMode {
  mean_shift_pct: number;
  median_shift_pct: number;
  p90_shift_pct: number;
}

export interface CounterfactualLensFlipSummary {
  n_evaluated: number;
  focal_only: CounterfactualShiftMode;
  physical_lens_swap: CounterfactualShiftMode;
  lens_confounding_detected?: boolean;
  confounding_reasons?: string[];
  lt500_ultrawide_mae_g?: number;
  lt500_ultrawide_mbe_g?: number;
}

export interface DualLensConsistencySummary {
  n_mixed_lens_sessions: number;
  mean_abs_diff_g: number;
  median_abs_diff_g: number;
  mean_rel_diff_pct: number;
  median_rel_diff_pct: number;
  mean_signed_wide_minus_uw_g: number;
  median_signed_wide_minus_uw_g: number;
}

export type AblationConditionKey =
  | "C1_uncalibrated_pdi"
  | "C2_physics_normalized_strict"
  | "C3_physics_plus_exif_meta"
  | "C4_naive_double_count_fxz";

export interface ModelComparisonRow {
  model_name: string;
  ablation_condition: AblationConditionKey;
  n_features: number;
  weighting_mode: string;
  duan_smearing_enabled: boolean;
  balanced_band_mape_pct: number;
  global: MetricSlice;
  session_level: SessionLevelMetrics;
  by_weight_band: Record<WeightBand, MetricSlice>;
  dual_lens_consistency: DualLensConsistencySummary;
  counterfactual_lens_flip: CounterfactualLensFlipSummary;
  lens_confounding_detected: boolean;
  confounding_reasons: string[];
  eligible_for_production: boolean;
  is_winner: boolean;
}

export interface OpticalAblationConditionReport {
  n_features: number;
  balanced_band_mape_pct: number;
  global: MetricSlice;
  session_level: SessionLevelMetrics;
  by_weight_band: Record<WeightBand, MetricSlice>;
  lt500_ultrawide: MetricSlice;
  dual_lens_consistency: DualLensConsistencySummary;
  counterfactual_lens_flip: CounterfactualLensFlipSummary;
  lens_confounding_detected: boolean;
  confounding_reasons: string[];
  eligible_for_production: boolean;
}

export interface OpticalAblationDeltaVsC1 {
  delta_balanced_band_mape_pct: number;
  delta_mae_g: number;
  delta_mape_pct: number;
  delta_r2: number;
  delta_rmse_g: number;
  delta_session_averaged_r2: number;
}

export interface FeatureImportanceItem {
  feature: string;
  importance_pct: number;
  raw_importance: number;
}

export interface FoldMetricRow extends MetricSlice {
  fold: number;
  n_sessions: number;
}

export interface WinningModelReport {
  model_name: string;
  ablation_condition: AblationConditionKey;
  n_features: number;
  feature_names: string[];
  weighting_mode: string;
  duan_smearing_enabled: boolean;
  balanced_band_mape_pct: number;
  global: MetricSlice;
  session_level: SessionLevelMetrics;
  per_fold: FoldMetricRow[];
  by_weight_band: Record<WeightBand, MetricSlice>;
  by_lens: Record<LensClass, MetricSlice>;
  by_scene: Record<SceneType, MetricSlice>;
  by_band_and_scene: Record<string, MetricSlice>;
  by_band_and_lens: Record<string, MetricSlice>;
  band_error_intervals: Record<WeightBand, BandErrorInterval>;
  dual_lens_consistency: DualLensConsistencySummary;
  counterfactual_lens_flip: CounterfactualLensFlipSummary;
  feature_importances: FeatureImportanceItem[];
}

export interface EvaluationReport {
  schema_version: string;
  pdi_config_hash: string;
  dataset_summary: {
    cross_fold_session_overlap: number;
    n_clean_images: number;
    n_folds: number;
    n_mixed_lens_sessions: number;
    n_sessions: number;
    n_unique_weights: number;
    valid_age_days_range: string;
    weight_max_g: number;
    weight_min_g: number;
  };
  literature_benchmark: {
    note: string;
    r2_reference: number;
  };
  selection_criteria: {
    primary_metric: string;
    tie_breakers: string[];
    anti_confounding_gate: {
      focal_only_median_threshold_pct: number;
      lt500_ultrawide_vs_c2_degradation_check: boolean;
      physical_lens_swap_median_threshold_pct: number;
      physical_lens_swap_p90_threshold_pct: number;
      strict_inequality: boolean;
    };
  };
  model_comparison: ModelComparisonRow[];
  optical_ablation: {
    reference_model_name: string;
    conditions: Record<AblationConditionKey, OpticalAblationConditionReport>;
    deltas_vs_c1: Record<
      Exclude<AblationConditionKey, "C1_uncalibrated_pdi">,
      OpticalAblationDeltaVsC1
    >;
    dual_lens_consistency: Record<AblationConditionKey, DualLensConsistencySummary>;
    counterfactual_lens_flip: Record<AblationConditionKey, CounterfactualLensFlipSummary>;
    family_means_by_condition: Record<
      AblationConditionKey,
      {
        mean_balanced_band_mape_pct: number;
        mean_mae_g: number;
        mean_mape_pct: number;
        mean_r2: number;
        mean_rmse_g: number;
        n_models: number;
      }
    >;
  };
  winning_model: WinningModelReport;
  oof_scatter_points: OofScatterPoint[];
}

export interface DatasetAuditSummary {
  schema_version: string;
  dataset_identity: {
    breed: string;
    capture_device: string;
    doi: string;
    name: string;
    scale_model: string;
  };
  curation_funnel: {
    raw_images: number;
    raw_sessions: number;
    quarantined_total: number;
    quarantined_multi_bird: {
      count: number;
      folders: string[];
      per_folder_counts: Record<string, number>;
    };
    quarantined_severe_mislabel: {
      count: number;
      folders: string[];
      per_folder_counts: Record<string, number>;
    };
    lcd_corrected_images: {
      count: number;
      corrections_applied: Record<string, number>;
    };
    clean_images: number;
    clean_leaf_folders: number;
    clean_unique_weights: number;
    clean_sessions: number;
  };
  mendeley_vs_disk_reconciliation: {
    published_claims: {
      published_duration_claim: string;
      published_images: number;
      published_weight_note: string;
    };
    disk_audit: {
      calendar_span_days: number;
      clean_weight_min_g: number;
      clean_weight_max_g: number;
      date_min: string;
      date_max: string;
      distinct_capture_dates: number;
      estimated_developmental_age_days_range: string;
      missing_vs_mendeley_images: number;
      raw_images: number;
      raw_leaf_folders: number;
      raw_sessions: number;
      raw_unique_weights: number;
    };
  };
  distributions: {
    coverage_gaps: {
      temporal_hiatus: {
        gap_start_date: string;
        gap_end_date: string;
        max_gap_days: number;
      };
      weight_sparse_interval: {
        lower_exclusive_g: number;
        upper_exclusive_g: number;
        span_g: number;
        clean_images_inside: number;
        clean_sessions_inside: number;
        clean_unique_weights_inside: number;
        clean_weights_inside: number[];
      };
    };
    lenses: Record<LensClass, { n_images: number; n_sessions: number; pct_images: number }>;
    weight_bands: Record<
      WeightBand,
      {
        n_images: number;
        n_sessions: number;
        n_ultrawide: number;
        n_wide: number;
        pct_images: number;
      }
    >;
    timeline_by_date: Array<{
      date: string;
      n_images: number;
      n_sessions: number;
      weight_min_g: number;
      weight_max_g: number;
    }>;
  };
  encoding_and_exif_recovery: {
    cp866_recovered_count: number;
    cp866_mtime_fallback_count: number;
    exif_orientation_rotated_count: number;
    exif_orientation_rotated_pct: number;
    mtime_fallback_count: number;
    nested_level2_folders: string[];
    nested_level2_images_count: number;
    suffix_sanitized_folders: string[];
    variable_digital_zoom_count: number;
    year_2024_capture_count: number;
  };
  split_audit: {
    cross_fold_session_overlap: number;
    group_column: string;
    n_splits: number;
    random_state: number;
    session_gap_threshold_seconds: number;
    shuffle: boolean;
    strategy: string;
    stratify_column: string;
  };
}

export interface PdiCalibrationAudit {
  schema_version: string;
  pdi_config_hash: string;
  dataset_extraction_summary: {
    n_clean_images: number;
    n_bucket_dorsal: number;
    n_scale_or_litter: number;
    primary_fallback_count: number;
    primary_fallback_pct: number;
    lcd_suppression_triggered_count: number;
    lcd_suppression_bucket_false_positives: number;
    workers: number;
  };
  paired_dual_lens_calibration: {
    winning_formula: string;
    rejected_double_count_formula: string;
    expected_zoom_area_ratio_wide_over_uw: number;
    resolution_area_ratio_wide_over_uw: number;
    scopes: Record<
      string,
      {
        n_pairs: number;
        formulas: Record<
          string,
          {
            iqr_25: number;
            iqr_75: number;
            median_abs_log_ratio: number;
            median_wide_over_uw: number;
          }
        >;
      }
    >;
  };
}

export interface MetricsResponse {
  schema_version: string;
  pdi_config_hash: string;
  evaluation_report: EvaluationReport;
  dataset_audit_summary: DatasetAuditSummary;
  pdi_calibration_audit: PdiCalibrationAudit;
}
