import type { LensClass, OofScatterPoint, SceneType, WeightBand } from "../types/api.ts";

export interface ScatterFilterState {
  weightBand: WeightBand | "all";
  lensClass: LensClass | "all";
  sceneType: SceneType | "all";
}

export type SessionLensGroup = LensClass | "mixed";

export interface SessionScatterPoint {
  session_id: string;
  actual_g: number;
  predicted_g: number;
  abs_err_g: number;
  pct_err: number;
  fold: number;
  weight_band: WeightBand;
  n_images: number;
  lens_group: SessionLensGroup;
  scene_types: SceneType[];
}

export interface RegressionPair {
  actual_g: number;
  predicted_g: number;
}

/**
 * Filter raw out-of-fold scatter points by weight band, lens class, and scene type.
 * Always applied BEFORE session aggregation (`04-RESEARCH.md:385-388`).
 */
export function filterScatterPoints(
  points: readonly OofScatterPoint[],
  filters: ScatterFilterState,
): OofScatterPoint[] {
  return points.filter((point) => {
    if (filters.weightBand !== "all" && point.weight_band !== filters.weightBand) {
      return false;
    }
    if (filters.lensClass !== "all" && point.lens_class !== filters.lensClass) {
      return false;
    }
    if (filters.sceneType !== "all" && point.scene_type !== filters.sceneType) {
      return false;
    }
    return true;
  });
}

/**
 * Aggregate already-filtered OOF scatter points into one point per `session_id`.
 * Preserves session-invariant `actual_g`, `weight_band`, and `fold`, averages `predicted_g`,
 * recomputes `abs_err_g` & `pct_err`, and tags `lens_group` as `"wide"`, `"ultrawide"`, or `"mixed"`.
 */
export function aggregateScatterBySession(
  filteredPoints: readonly OofScatterPoint[],
): SessionScatterPoint[] {
  if (filteredPoints.length === 0) {
    return [];
  }

  interface SessionBucket {
    session_id: string;
    actual_g: number;
    fold: number;
    weight_band: WeightBand;
    predSum: number;
    count: number;
    hasWide: boolean;
    hasUltrawide: boolean;
    scenes: Set<SceneType>;
  }

  const buckets = new Map<string, SessionBucket>();

  for (const point of filteredPoints) {
    let bucket = buckets.get(point.session_id);
    if (!bucket) {
      bucket = {
        session_id: point.session_id,
        actual_g: point.actual_g,
        fold: point.fold,
        weight_band: point.weight_band,
        predSum: 0,
        count: 0,
        hasWide: false,
        hasUltrawide: false,
        scenes: new Set<SceneType>(),
      };
      buckets.set(point.session_id, bucket);
    }

    bucket.predSum += point.predicted_g;
    bucket.count += 1;
    if (point.lens_class === "wide") {
      bucket.hasWide = true;
    } else {
      bucket.hasUltrawide = true;
    }
    bucket.scenes.add(point.scene_type);
  }

  const result: SessionScatterPoint[] = [];
  for (const bucket of buckets.values()) {
    const meanPredictedG = bucket.predSum / bucket.count;
    const absErrG = Math.abs(meanPredictedG - bucket.actual_g);
    const pctErr = bucket.actual_g > 0 ? (absErrG / bucket.actual_g) * 100 : 0;
    const lensGroup: SessionLensGroup =
      bucket.hasWide && bucket.hasUltrawide ? "mixed" : bucket.hasWide ? "wide" : "ultrawide";

    result.push({
      session_id: bucket.session_id,
      actual_g: bucket.actual_g,
      predicted_g: meanPredictedG,
      abs_err_g: absErrG,
      pct_err: pctErr,
      fold: bucket.fold,
      weight_band: bucket.weight_band,
      n_images: bucket.count,
      lens_group: lensGroup,
      scene_types: Array.from(bucket.scenes),
    });
  }

  return result;
}

/**
 * Compute exact coefficient of determination (`R²`) over `actual_g` vs `predicted_g` pairs.
 * Returns `null` when fewer than 2 valid finite pairs exist or target variance (`SS_tot`) is zero.
 */
export function computeR2(pairs: readonly RegressionPair[]): number | null {
  const valid = pairs.filter((p) => Number.isFinite(p.actual_g) && Number.isFinite(p.predicted_g));
  if (valid.length < 2) {
    return null;
  }

  let actualSum = 0;
  for (const pair of valid) {
    actualSum += pair.actual_g;
  }
  const actualMean = actualSum / valid.length;

  let ssTot = 0;
  let ssRes = 0;
  for (const pair of valid) {
    const diffMean = pair.actual_g - actualMean;
    const diffPred = pair.actual_g - pair.predicted_g;
    ssTot += diffMean * diffMean;
    ssRes += diffPred * diffPred;
  }

  if (ssTot <= 0) {
    return null;
  }

  const r2 = 1 - ssRes / ssTot;
  return Number.isFinite(r2) ? r2 : null;
}
