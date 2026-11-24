import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  LensClass,
  OofScatterPoint,
  OpticalLensClass,
  OpticalPayload,
  WatershedParamBounds,
} from "../types/api.ts";
import {
  ALLOWED_UPLOAD_EXTENSIONS,
  ALLOWED_UPLOAD_MIMES,
  ApiError,
  fetchHealth,
  fetchMetrics,
  fetchPresets,
  isAbortError,
  MAX_UPLOAD_BYTES,
  predictSingleBird,
  runWatershedDemo,
  validateClientUploadFile,
} from "./api.ts";
import { cn } from "./cn.ts";
import { computeFlockStats, type FlockEntry } from "./flock.ts";
import {
  formatGrams,
  formatKg,
  formatNullableMetric,
  formatNumberPtBr,
  formatPercent,
  pluralizePtBr,
} from "./format.ts";

function makeFlockEntry(id: string, weightG: number): FlockEntry {
  return {
    id,
    label: `Ave ${id}`,
    weightG,
    sourceType: "preset",
    presetId: "sb_intermediate_grower",
    ageStageLabel: "Crescimento Inicial",
    addedAtIso: "2026-11-24T10:00:00.000Z",
  };
}

function makeMockFile(name: string, type: string, sizeBytes: number): File {
  const file = new File(["x"], name, { type });
  Object.defineProperty(file, "size", { value: sizeBytes });
  return file;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("cn utility", () => {
  it("merges conditional Tailwind classes cleanly without conflicts", () => {
    expect(cn("px-2 py-1", false && "hidden", "px-4")).toBe("py-1 px-4");
  });
});

describe("format utilities (pt-BR & nullable guard)", () => {
  it("formats numbers, grams, kilograms, and percentages in pt-BR", () => {
    expect(formatNumberPtBr(158.11, 2)).toBe("158,11");
    expect(formatNumberPtBr(1738, 0)).toBe("1.738");
    expect(formatNumberPtBr(-0, 1)).toBe("0,0");
    expect(formatGrams(1580.4, 1)).toBe("1.580,4 g");
    expect(formatGrams(424, 0)).toBe("424 g");
    expect(formatKg(1.555, 2)).toBe("1,56 kg");
    expect(formatPercent(35.1129, 2)).toBe("35,11%");
  });

  it("renders em-dash '—' (never NaN or Infinity) for null, undefined, or non-finite inputs", () => {
    for (const invalid of [
      null,
      undefined,
      Number.NaN,
      Number.POSITIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
    ]) {
      expect(formatNumberPtBr(invalid)).toBe("—");
      expect(formatGrams(invalid)).toBe("—");
      expect(formatKg(invalid)).toBe("—");
      expect(formatPercent(invalid)).toBe("—");
      expect(formatNullableMetric(invalid, 2, " g")).toBe("—");
    }
    expect(formatNullableMetric(0.6905, 4)).toBe("0,6905");
    expect(formatNullableMetric(88.3, 1, " ms")).toBe("88,3 ms");
  });

  it("pluralizes pt-BR counts cleanly for 0, 1, and many items", () => {
    expect(pluralizePtBr(0, "ave", "aves")).toBe("0 aves");
    expect(pluralizePtBr(1, "ave", "aves")).toBe("1 ave");
    expect(pluralizePtBr(3, "ave", "aves")).toBe("3 aves");
    expect(pluralizePtBr(1, "imagem", "imagens")).toBe("1 imagem");
    expect(pluralizePtBr(1738, "imagem", "imagens")).toBe("1.738 imagens");
  });
});

describe("computeFlockStats (client-side flock uniformity)", () => {
  it("returns all null metrics for 0 entries", () => {
    expect(computeFlockStats([])).toEqual({
      count: 0,
      totalG: null,
      meanG: null,
      stdDevG: null,
      cvPct: null,
      minG: null,
      maxG: null,
    });
  });

  it("populates mean/min/max and keeps stdDevG and cvPct null for 1 entry", () => {
    expect(computeFlockStats([makeFlockEntry("1", 424.0)])).toEqual({
      count: 1,
      totalG: 424.0,
      meanG: 424.0,
      stdDevG: null,
      cvPct: null,
      minG: 424.0,
      maxG: 424.0,
    });
  });

  it("computes unbiased sample standard deviation (ddof=1) and CV% for >= 2 entries", () => {
    const stats = computeFlockStats([
      makeFlockEntry("1", 400),
      makeFlockEntry("2", 500),
      makeFlockEntry("3", 600),
    ]);
    expect(stats.count).toBe(3);
    expect(stats.totalG).toBeCloseTo(1500, 6);
    expect(stats.meanG).toBeCloseTo(500, 6);
    // Sample SD of [400, 500, 600] with ddof=1 is sqrt((10000 + 0 + 10000) / 2) = 100
    expect(stats.stdDevG).toBeCloseTo(100, 6);
    expect(stats.cvPct).toBeCloseTo(20, 6);
    expect(stats.minG).toBe(400);
    expect(stats.maxG).toBe(600);
  });
});

describe("validateClientUploadFile (15 MiB & MIME allowlist guard)", () => {
  it("exposes authoritative 15 MiB limit and image allowlists", () => {
    expect(MAX_UPLOAD_BYTES).toBe(15 * 1024 * 1024);
    expect(ALLOWED_UPLOAD_MIMES.has("image/jpeg")).toBe(true);
    expect(ALLOWED_UPLOAD_MIMES.has("image/png")).toBe(true);
    expect(ALLOWED_UPLOAD_MIMES.has("image/webp")).toBe(true);
    expect(ALLOWED_UPLOAD_EXTENSIONS.has(".jpg")).toBe(true);
    expect(ALLOWED_UPLOAD_EXTENSIONS.has(".jpeg")).toBe(true);
    expect(ALLOWED_UPLOAD_EXTENSIONS.has(".png")).toBe(true);
    expect(ALLOWED_UPLOAD_EXTENSIONS.has(".webp")).toBe(true);
  });

  it("accepts valid JPEG, PNG, and WEBP uploads within 15 MiB", () => {
    expect(() =>
      validateClientUploadFile(makeMockFile("bird.jpg", "image/jpeg", 1024)),
    ).not.toThrow();
    expect(() =>
      validateClientUploadFile(makeMockFile("bird.png", "image/png", 2048)),
    ).not.toThrow();
    expect(() =>
      validateClientUploadFile(makeMockFile("bird.webp", "image/webp", MAX_UPLOAD_BYTES)),
    ).not.toThrow();
    // Empty MIME falls back to allowed extension check
    expect(() => validateClientUploadFile(makeMockFile("bird.jpeg", "", 1024))).not.toThrow();
  });

  it("rejects files over 15 MiB with ApiError 413 and exact pt-BR copy", () => {
    const oversize = makeMockFile("huge.jpg", "image/jpeg", MAX_UPLOAD_BYTES + 1);
    expect(() => validateClientUploadFile(oversize)).toThrowError(ApiError);
    try {
      validateClientUploadFile(oversize);
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      const apiErr = err as ApiError;
      expect(apiErr.status).toBe(413);
      expect(apiErr.userMessage).toContain("limite local de 15 MB");
    }
  });

  it("rejects unsupported MIME types or extensions with ApiError 415", () => {
    const bmpFile = makeMockFile("frame.bmp", "image/bmp", 1024);
    try {
      validateClientUploadFile(bmpFile);
      expect.unreachable("should have rejected BMP");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      const apiErr = err as ApiError;
      expect(apiErr.status).toBe(415);
      expect(apiErr.userMessage).toContain("Formato de imagem não suportado");
    }

    const unknownExt = makeMockFile("notes.txt", "", 128);
    expect(() => validateClientUploadFile(unknownExt)).toThrowError(ApiError);
  });

  it("rejects empty 0-byte uploads with ApiError 400", () => {
    const emptyFile = makeMockFile("empty.jpg", "image/jpeg", 0);
    try {
      validateClientUploadFile(emptyFile);
      expect.unreachable("should have rejected empty file");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).status).toBe(400);
    }
  });
});

describe("API client FormData serialization, ApiError mapping, and AbortError handling", () => {
  it("omits undefined or non-finite optional fields from predictSingleBird FormData", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ predicted_weight_g: 424.0 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const controller = new AbortController();
    await predictSingleBird(
      {
        presetId: "sb_intermediate_grower",
        focalLengthMm: undefined,
        zoomMultiple: Number.NaN,
      },
      controller.signal,
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/predict");
    expect(init.method).toBe("POST");
    expect(init.signal).toBe(controller.signal);
    const body = init.body as FormData;
    expect(body.get("preset_id")).toBe("sb_intermediate_grower");
    expect(body.has("file")).toBe(false);
    expect(body.has("focal_length_mm")).toBe(false);
    expect(body.has("zoom_multiple")).toBe(false);
  });

  it("omits undefined slider/optical overrides from runWatershedDemo FormData and includes finite ones", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ bird_count: 2 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await runWatershedDemo({
      presetId: "ws_417",
      hRel: 0.25,
      minDistanceRel: undefined,
      minAreaRatio: 0.1,
      focalLengthMm: undefined,
      zoomMultiple: 1.0,
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/watershed-demo");
    const body = init.body as FormData;
    expect(body.get("preset_id")).toBe("ws_417");
    expect(body.get("h_rel")).toBe("0.25");
    expect(body.has("min_distance_rel")).toBe(false);
    expect(body.get("min_area_ratio")).toBe("0.1");
    expect(body.has("focal_length_mm")).toBe(false);
    expect(body.get("zoom_multiple")).toBe("1");
  });

  it("maps HTTP 400, 413, 415, 422, 500 and offline errors to sanitized pt-BR ApiError messages", async () => {
    const statusExpectations: Array<[number, string]> = [
      [400, "Não foi possível decodificar esta imagem"],
      [413, "limite local de 15 MB"],
      [415, "Formato de imagem não suportado"],
      [422, "fora dos limites aceitos"],
      [500, "127.0.0.1:8000"],
    ];

    for (const [status, expectedSnippet] of statusExpectations) {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify({ detail: "Traceback /home/secret/data/raw/a.jpg" }), {
            status,
            headers: { "Content-Type": "application/json" },
          }),
        ),
      );
      try {
        await fetchHealth();
        expect.unreachable(`expected status ${status} to throw`);
      } catch (err) {
        expect(err).toBeInstanceOf(ApiError);
        const apiErr = err as ApiError;
        expect(apiErr.status).toBe(status);
        expect(apiErr.userMessage).toContain(expectedSnippet);
        expect(apiErr.userMessage).not.toContain("Traceback");
        expect(apiErr.userMessage).not.toContain(".jpg");
      }
    }

    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    try {
      await fetchPresets();
      expect.unreachable("expected offline network error to throw");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).status).toBe(0);
      expect((err as ApiError).userMessage).toContain("127.0.0.1:8000");
    }
  });

  it("rethrows AbortError untouched and detects it via isAbortError", async () => {
    const abortErr = new DOMException("The operation was aborted.", "AbortError");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(abortErr));

    expect(isAbortError(abortErr)).toBe(true);
    expect(isAbortError(new ApiError(500, "erro"))).toBe(false);
    expect(isAbortError(null)).toBe(false);

    try {
      await fetchMetrics();
      expect.unreachable("expected abort to reject");
    } catch (err) {
      expect(isAbortError(err)).toBe(true);
      expect(err).not.toBeInstanceOf(ApiError);
    }
  });

  it("blocks invalid upload files client-side before calling fetch", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const invalid = makeMockFile("bad.gif", "image/gif", 1024);
    expect(() => predictSingleBird({ file: invalid })).toThrowError(ApiError);
    expect(() => runWatershedDemo({ file: invalid })).toThrowError(ApiError);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("enforces LensClass vs OpticalLensClass and OofScatterPoint.actual_g type contracts", () => {
    const presetLens: LensClass = "wide";
    const unknownOptical: OpticalLensClass = "unknown";
    const optical: OpticalPayload = {
      focal_length_mm: 4.25,
      zoom_multiple: 1.0,
      exif_optical_present: false,
      override_applied: false,
      lens_class: unknownOptical,
    };
    const scatter: OofScatterPoint = {
      actual_g: 541,
      predicted_g: 383.22,
      abs_err_g: 157.78,
      pct_err: 29.16,
      fold: 0,
      image_id: "c42aa07d27d4e9e8",
      session_id: "541__s0",
      lens_class: presetLens,
      scene_type: "scale_or_litter",
      weight_band: "500-1000g",
    };
    const bounds: WatershedParamBounds = {
      h_rel: [0.05, 0.8],
      min_distance_rel: [0.2, 2.5],
      min_area_ratio: [0.01, 0.45],
    };

    expect(optical.lens_class).toBe("unknown");
    expect(scatter.actual_g).toBe(541);
    expect(bounds.h_rel).toEqual([0.05, 0.8]);
  });
});
