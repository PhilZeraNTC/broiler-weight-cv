import { describe, expect, it } from "vitest";
import type { OofScatterPoint } from "../types/api.ts";
import {
  aggregateScatterBySession,
  computeR2,
  filterScatterPoints,
  type ScatterFilterState,
} from "./scatter.ts";

const FIXTURE_POINTS: readonly OofScatterPoint[] = [
  {
    actual_g: 541,
    predicted_g: 380,
    abs_err_g: 161,
    pct_err: 29.7597,
    fold: 0,
    image_id: "img_541_w1",
    session_id: "541__s0",
    lens_class: "wide",
    scene_type: "scale_or_litter",
    weight_band: "500-1000g",
  },
  {
    actual_g: 541,
    predicted_g: 420,
    abs_err_g: 121,
    pct_err: 22.366,
    fold: 0,
    image_id: "img_541_uw1",
    session_id: "541__s0",
    lens_class: "ultrawide",
    scene_type: "scale_or_litter",
    weight_band: "500-1000g",
  },
  {
    actual_g: 180,
    predicted_g: 210,
    abs_err_g: 30,
    pct_err: 16.6667,
    fold: 4,
    image_id: "img_180_w1",
    session_id: "180__s0",
    lens_class: "wide",
    scene_type: "bucket_dorsal",
    weight_band: "<500g",
  },
  {
    actual_g: 180,
    predicted_g: 190,
    abs_err_g: 10,
    pct_err: 5.5556,
    fold: 4,
    image_id: "img_180_w2",
    session_id: "180__s0",
    lens_class: "wide",
    scene_type: "scale_or_litter",
    weight_band: "<500g",
  },
  {
    actual_g: 1555,
    predicted_g: 1400,
    abs_err_g: 155,
    pct_err: 9.9678,
    fold: 1,
    image_id: "img_1555_uw1",
    session_id: "1555__s0",
    lens_class: "ultrawide",
    scene_type: "bucket_dorsal",
    weight_band: ">1000g",
  },
];

const ALL_FILTERS: ScatterFilterState = {
  weightBand: "all",
  lensClass: "all",
  sceneType: "all",
};

describe("filterScatterPoints", () => {
  it("returns all points untouched when all filters are 'all'", () => {
    const res = filterScatterPoints(FIXTURE_POINTS, ALL_FILTERS);
    expect(res).toHaveLength(5);
    expect(res[0]?.actual_g).toBe(541);
  });

  it("filters accurately by weightBand, lensClass, and sceneType combinations", () => {
    expect(
      filterScatterPoints(FIXTURE_POINTS, { ...ALL_FILTERS, weightBand: "<500g" }),
    ).toHaveLength(2);

    expect(
      filterScatterPoints(FIXTURE_POINTS, { ...ALL_FILTERS, lensClass: "ultrawide" }).map(
        (p) => p.image_id,
      ),
    ).toEqual(["img_541_uw1", "img_1555_uw1"]);

    expect(
      filterScatterPoints(FIXTURE_POINTS, {
        weightBand: "<500g",
        lensClass: "wide",
        sceneType: "bucket_dorsal",
      }).map((p) => p.image_id),
    ).toEqual(["img_180_w1"]);

    expect(
      filterScatterPoints(FIXTURE_POINTS, {
        weightBand: ">1000g",
        lensClass: "wide",
        sceneType: "all",
      }),
    ).toEqual([]);
  });
});

describe("aggregateScatterBySession (filter-before-aggregate semantics)", () => {
  it("groups filtered points by session_id, preserves actual_g/fold/weight_band, and tags wide/ultrawide/mixed", () => {
    const sessions = aggregateScatterBySession(FIXTURE_POINTS);
    expect(sessions).toHaveLength(3);

    const mixedSession = sessions.find((s) => s.session_id === "541__s0");
    expect(mixedSession).toBeDefined();
    expect(mixedSession?.actual_g).toBe(541);
    expect(mixedSession?.predicted_g).toBeCloseTo(400, 6);
    expect(mixedSession?.abs_err_g).toBeCloseTo(141, 6);
    expect(mixedSession?.pct_err).toBeCloseTo((141 / 541) * 100, 6);
    expect(mixedSession?.n_images).toBe(2);
    expect(mixedSession?.fold).toBe(0);
    expect(mixedSession?.weight_band).toBe("500-1000g");
    expect(mixedSession?.lens_group).toBe("mixed");

    const wideSession = sessions.find((s) => s.session_id === "180__s0");
    expect(wideSession?.lens_group).toBe("wide");
    expect(wideSession?.predicted_g).toBeCloseTo(200, 6);
    expect(wideSession?.abs_err_g).toBeCloseTo(20, 6);
    expect(wideSession?.n_images).toBe(2);

    const uwSession = sessions.find((s) => s.session_id === "1555__s0");
    expect(uwSession?.lens_group).toBe("ultrawide");
    expect(uwSession?.predicted_g).toBeCloseTo(1400, 6);
    expect(uwSession?.n_images).toBe(1);
  });

  it("applies lens filter before session aggregation so mixed sessions collapse cleanly to the filtered lens", () => {
    const filtered = filterScatterPoints(FIXTURE_POINTS, {
      ...ALL_FILTERS,
      lensClass: "wide",
    });
    const sessions = aggregateScatterBySession(filtered);
    expect(sessions).toHaveLength(2);

    const session541 = sessions.find((s) => s.session_id === "541__s0");
    expect(session541?.lens_group).toBe("wide");
    expect(session541?.n_images).toBe(1);
    expect(session541?.predicted_g).toBe(380);
    expect(session541?.abs_err_g).toBe(161);
  });

  it("handles 0-point and 1-point inputs cleanly", () => {
    expect(aggregateScatterBySession([])).toEqual([]);

    const single = aggregateScatterBySession([FIXTURE_POINTS[4]]);
    expect(single).toHaveLength(1);
    expect(single[0]?.actual_g).toBe(1555);
    expect(single[0]?.n_images).toBe(1);
    expect(single[0]?.lens_group).toBe("ultrawide");
  });
});

describe("computeR2", () => {
  it("returns null for 0 points, 1 point, or zero target variance (SS_tot == 0)", () => {
    expect(computeR2([])).toBeNull();
    expect(computeR2([{ actual_g: 500, predicted_g: 490 }])).toBeNull();
    expect(
      computeR2([
        { actual_g: 500, predicted_g: 480 },
        { actual_g: 500, predicted_g: 520 },
      ]),
    ).toBeNull();
  });

  it("computes exact R² = 1.0 on perfect predictions and known fractional R² on noisy predictions", () => {
    expect(
      computeR2([
        { actual_g: 200, predicted_g: 200 },
        { actual_g: 600, predicted_g: 600 },
        { actual_g: 1600, predicted_g: 1600 },
      ]),
    ).toBeCloseTo(1.0, 10);

    // y = [100, 200, 300], mean = 200, SS_tot = 10000 + 0 + 10000 = 20000
    // yhat = [120, 200, 280], SS_res = 400 + 0 + 400 = 800
    // R^2 = 1 - 800 / 20000 = 0.96
    expect(
      computeR2([
        { actual_g: 100, predicted_g: 120 },
        { actual_g: 200, predicted_g: 200 },
        { actual_g: 300, predicted_g: 280 },
      ]),
    ).toBeCloseTo(0.96, 10);
  });

  it("computes consistent session-level R² directly on aggregated SessionScatterPoint rows", () => {
    const sessions = aggregateScatterBySession(FIXTURE_POINTS);
    const r2 = computeR2(sessions);
    expect(r2).not.toBeNull();
    // actuals: [541, 180, 1555], mean = 2276/3 = 758.6666667
    // SS_tot = (541-m)^2 + (180-m)^2 + (1555-m)^2 = 3049142 / 3 = 1016380.6666666666
    // SS_res = 141^2 + 20^2 + 155^2 = 19881 + 400 + 24025 = 44306
    // R2 = 1 - 44306 / (3049142 / 3) = 0.956408065...
    expect(r2 ?? 0).toBeCloseTo(1 - 44306 / (3049142 / 3), 10);
  });
});
