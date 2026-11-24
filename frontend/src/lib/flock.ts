import type { SourceType } from "../types/api.ts";

export interface FlockEntry {
  id: string;
  label: string;
  weightG: number;
  sourceType: SourceType;
  presetId: string | null;
  ageStageLabel: string;
  addedAtIso: string;
}

export interface FlockStats {
  count: number;
  totalG: number | null;
  meanG: number | null;
  stdDevG: number | null;
  cvPct: number | null;
  minG: number | null;
  maxG: number | null;
}

/**
 * Compute pure client-side simulated flock batch statistics.
 * Uses unbiased sample standard deviation (`ddof = 1`) and `CV% = (SD / mean) * 100`
 * strictly when `count >= 2`; returns `null` for dispersion metrics at `0` or `1` entries.
 */
export function computeFlockStats(entries: readonly FlockEntry[]): FlockStats {
  const count = entries.length;
  if (count === 0) {
    return {
      count: 0,
      totalG: null,
      meanG: null,
      stdDevG: null,
      cvPct: null,
      minG: null,
      maxG: null,
    };
  }

  let totalG = 0;
  let minG = Number.POSITIVE_INFINITY;
  let maxG = Number.NEGATIVE_INFINITY;

  for (const entry of entries) {
    const w = entry.weightG;
    totalG += w;
    if (w < minG) minG = w;
    if (w > maxG) maxG = w;
  }

  const meanG = totalG / count;

  if (count === 1) {
    return {
      count: 1,
      totalG,
      meanG,
      stdDevG: null,
      cvPct: null,
      minG,
      maxG,
    };
  }

  let squaredDiffSum = 0;
  for (const entry of entries) {
    const diff = entry.weightG - meanG;
    squaredDiffSum += diff * diff;
  }

  const stdDevG = Math.sqrt(squaredDiffSum / (count - 1));
  const cvPct = meanG > 0 ? (stdDevG / meanG) * 100 : null;

  return {
    count,
    totalG,
    meanG,
    stdDevG,
    cvPct,
    minG,
    maxG,
  };
}
