import { AlertTriangle, ChevronDown, ChevronUp, RotateCcw, SlidersHorizontal } from "lucide-react";
import { type FormEvent, type KeyboardEvent, useEffect, useId, useRef, useState } from "react";
import { formatNumberPtBr } from "../../lib/format.ts";
import type { OpticalPayload } from "../../types/api.ts";

/**
 * Authoritative Watershed defaults & ASVS V5 bounds strictly reconciled to
 * `backend/app/pdi/config.py` and approved `04-UI-SPEC.md` (superseding draft D-04/D-10 values).
 */
export const WATERSHED_SLIDER_DEFAULTS = {
  h_rel: 0.2,
  min_distance_rel: 0.85,
  min_area_ratio: 0.1,
} as const;

/** Tuple order: `[min, max, step]`. */
export const WATERSHED_SLIDER_BOUNDS = {
  h_rel: [0.05, 0.8, 0.01],
  min_distance_rel: [0.2, 2.5, 0.05],
  min_area_ratio: [0.01, 0.45, 0.01],
} as const;

const MIN_FOCAL_MM = 0.5;
const MAX_FOCAL_MM = 200.0;
const MIN_ZOOM_MULTIPLE = 0.1;
const MAX_ZOOM_MULTIPLE = 20.0;

const OUT_OF_RANGE_ERROR_COPY =
  "Um ou mais parâmetros estão fora dos limites aceitos (distância focal 0,5–200 mm; zoom 0,1–20×; limites do Watershed). Ajuste o campo destacado ou clique em 'Restaurar EXIF Original' (Aba 1) / 'Restaurar Parâmetros Padrão' (Aba 2).";

export interface WatershedSliderValues {
  h_rel: number;
  min_distance_rel: number;
  min_area_ratio: number;
}

export interface WatershedOpticalOverride {
  focalLengthMm?: number;
  zoomMultiple?: number;
}

export interface WatershedSlidersCardProps {
  values: WatershedSliderValues;
  committedValues?: WatershedSliderValues;
  optical: OpticalPayload | null;
  opticalOverride: WatershedOpticalOverride | null;
  loading: boolean;
  hasActiveSource: boolean;
  onChangeDraft: (next: WatershedSliderValues) => void;
  onCommit: (
    nextSliders: WatershedSliderValues,
    nextOptical: WatershedOpticalOverride | null,
  ) => void;
  onResetDefaults: () => void;
}

function parseLocaleFloat(raw: string): number {
  const normalized = raw.trim().replace(",", ".");
  if (!normalized) {
    return Number.NaN;
  }
  return Number(normalized);
}

function clampSlider(val: number, min: number, max: number): number {
  if (!Number.isFinite(val)) return min;
  return Math.min(max, Math.max(min, val));
}

const SLIDER_COMMIT_KEYS = new Set([
  "ArrowLeft",
  "ArrowRight",
  "ArrowUp",
  "ArrowDown",
  "Home",
  "End",
  "PageUp",
  "PageDown",
  "Enter",
]);

export function WatershedSlidersCard({
  values,
  committedValues,
  optical,
  opticalOverride,
  loading,
  hasActiveSource,
  onChangeDraft,
  onCommit,
  onResetDefaults,
}: WatershedSlidersCardProps) {
  const hRelId = useId();
  const distId = useId();
  const areaId = useId();
  const focalId = useId();
  const zoomId = useId();

  const [opticalOpen, setOpticalOpen] = useState<boolean>(false);
  const [focalInput, setFocalInput] = useState<string>("");
  const [zoomInput, setZoomInput] = useState<string>("");
  const [validationError, setValidationError] = useState<string | null>(null);
  const lastCommittedRef = useRef<WatershedSliderValues>({ ...(committedValues ?? values) });

  useEffect(() => {
    if (committedValues) {
      lastCommittedRef.current = { ...committedValues };
    }
  }, [committedValues]);

  useEffect(() => {
    if (opticalOverride) {
      setFocalInput(
        opticalOverride.focalLengthMm !== undefined
          ? String(Number(opticalOverride.focalLengthMm.toFixed(2)))
          : "",
      );
      setZoomInput(
        opticalOverride.zoomMultiple !== undefined
          ? String(Number(opticalOverride.zoomMultiple.toFixed(2)))
          : "",
      );
    } else if (optical) {
      setFocalInput(String(Number(optical.focal_length_mm.toFixed(2))));
      setZoomInput(String(Number(optical.zoom_multiple.toFixed(2))));
    }
    setValidationError(null);
  }, [optical, opticalOverride]);

  const resolveOpticalOverrideForSubmit = (): {
    valid: boolean;
    override: WatershedOpticalOverride | null;
  } => {
    if (!opticalOpen) {
      return { valid: true, override: opticalOverride };
    }

    const focalTrimmed = focalInput.trim();
    const zoomTrimmed = zoomInput.trim();
    if (!focalTrimmed && !zoomTrimmed) {
      return { valid: true, override: null };
    }

    const parsedFocal = focalTrimmed ? parseLocaleFloat(focalTrimmed) : undefined;
    const parsedZoom = zoomTrimmed ? parseLocaleFloat(zoomTrimmed) : undefined;

    const focalValid =
      parsedFocal === undefined ||
      (Number.isFinite(parsedFocal) && parsedFocal >= MIN_FOCAL_MM && parsedFocal <= MAX_FOCAL_MM);
    const zoomValid =
      parsedZoom === undefined ||
      (Number.isFinite(parsedZoom) &&
        parsedZoom >= MIN_ZOOM_MULTIPLE &&
        parsedZoom <= MAX_ZOOM_MULTIPLE);

    if (!focalValid || !zoomValid) {
      return { valid: false, override: opticalOverride };
    }

    // If user left the original EXIF values untouched, avoid marking override unnecessarily
    if (
      optical &&
      !opticalOverride &&
      parsedFocal !== undefined &&
      parsedZoom !== undefined &&
      Math.abs(parsedFocal - optical.focal_length_mm) < 0.005 &&
      Math.abs(parsedZoom - optical.zoom_multiple) < 0.005
    ) {
      return { valid: true, override: null };
    }

    return {
      valid: true,
      override: {
        focalLengthMm: parsedFocal,
        zoomMultiple: parsedZoom,
      },
    };
  };

  const commitSliders = (
    nextValues: WatershedSliderValues,
    { skipIfUnchanged = false }: { skipIfUnchanged?: boolean } = {},
  ) => {
    if (!hasActiveSource) {
      return;
    }
    const clamped: WatershedSliderValues = {
      h_rel: clampSlider(
        nextValues.h_rel,
        WATERSHED_SLIDER_BOUNDS.h_rel[0],
        WATERSHED_SLIDER_BOUNDS.h_rel[1],
      ),
      min_distance_rel: clampSlider(
        nextValues.min_distance_rel,
        WATERSHED_SLIDER_BOUNDS.min_distance_rel[0],
        WATERSHED_SLIDER_BOUNDS.min_distance_rel[1],
      ),
      min_area_ratio: clampSlider(
        nextValues.min_area_ratio,
        WATERSHED_SLIDER_BOUNDS.min_area_ratio[0],
        WATERSHED_SLIDER_BOUNDS.min_area_ratio[1],
      ),
    };
    if (
      skipIfUnchanged &&
      Math.abs(clamped.h_rel - lastCommittedRef.current.h_rel) < 1e-9 &&
      Math.abs(clamped.min_distance_rel - lastCommittedRef.current.min_distance_rel) < 1e-9 &&
      Math.abs(clamped.min_area_ratio - lastCommittedRef.current.min_area_ratio) < 1e-9
    ) {
      return;
    }
    const opticalCheck = resolveOpticalOverrideForSubmit();
    if (!opticalCheck.valid) {
      setValidationError(OUT_OF_RANGE_ERROR_COPY);
      return;
    }
    setValidationError(null);
    lastCommittedRef.current = clamped;
    onCommit(clamped, opticalCheck.override);
  };

  const commitFieldFromInput = (field: keyof WatershedSliderValues, rawValue: string) => {
    const liveVal = Number(rawValue);
    const nextValues: WatershedSliderValues = Number.isFinite(liveVal)
      ? { ...values, [field]: liveVal }
      : values;
    onChangeDraft(nextValues);
    commitSliders(nextValues, { skipIfUnchanged: true });
  };

  const handleRangeKeyUp = (
    field: keyof WatershedSliderValues,
    event: KeyboardEvent<HTMLInputElement>,
  ) => {
    if (SLIDER_COMMIT_KEYS.has(event.key)) {
      commitFieldFromInput(field, event.currentTarget.value);
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    commitSliders(values);
  };

  const handleReset = () => {
    setValidationError(null);
    lastCommittedRef.current = { ...WATERSHED_SLIDER_DEFAULTS };
    onResetDefaults();
  };

  return (
    <section
      aria-label="Ajuste Clássico do Watershed"
      className="space-y-4 rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      <div className="space-y-1">
        <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
          Limites Validados ASVS V5 · PDIConfig
        </p>
        <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
          Ajuste Clássico do Watershed
        </h2>
        <p className="text-[14px] font-normal leading-[1.5] text-ink-muted">
          Ajuste os 3 parâmetros de supressão de picos e áreas para separar aves encostadas sem
          redes neurais. Solte o controle ou clique em Reprocessar.
        </p>
      </div>

      <form onSubmit={handleSubmit} noValidate className="space-y-4">
        {/* Slider 1: h_rel [0.05, 0.80] step 0.01 default 0.20 */}
        <div className="space-y-1 rounded-lg border border-border bg-canvas p-4">
          <div className="flex items-center justify-between gap-2">
            <label htmlFor={hRelId} className="text-[12px] font-semibold leading-[1.33] text-ink">
              Proeminência de Pico (h_rel)
            </label>
            <span className="rounded bg-surface px-2 py-0.5 font-mono text-[12px] font-semibold leading-[1.33] tabular-nums text-ink">
              {formatNumberPtBr(values.h_rel, 2)}
            </span>
          </div>
          <input
            id={hRelId}
            type="range"
            min={WATERSHED_SLIDER_BOUNDS.h_rel[0]}
            max={WATERSHED_SLIDER_BOUNDS.h_rel[1]}
            step={WATERSHED_SLIDER_BOUNDS.h_rel[2]}
            value={values.h_rel}
            disabled={!hasActiveSource}
            onChange={(event) => {
              onChangeDraft({ ...values, h_rel: Number(event.target.value) });
            }}
            onPointerUp={(event) => commitFieldFromInput("h_rel", event.currentTarget.value)}
            onKeyUp={(event) => handleRangeKeyUp("h_rel", event)}
            className="min-h-[44px] w-full cursor-pointer accent-accent disabled:cursor-not-allowed disabled:opacity-60"
          />
          <div className="flex items-center justify-between text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
            <span>0,05</span>
            <span>Padrão: 0,20</span>
            <span>0,80</span>
          </div>
        </div>

        {/* Slider 2: min_distance_rel [0.20, 2.50] step 0.05 default 0.85 */}
        <div className="space-y-1 rounded-lg border border-border bg-canvas p-4">
          <div className="flex items-center justify-between gap-2">
            <label htmlFor={distId} className="text-[12px] font-semibold leading-[1.33] text-ink">
              Distância Mínima entre Picos (min_distance_rel)
            </label>
            <span className="rounded bg-surface px-2 py-0.5 font-mono text-[12px] font-semibold leading-[1.33] tabular-nums text-ink">
              {formatNumberPtBr(values.min_distance_rel, 2)}
            </span>
          </div>
          <input
            id={distId}
            type="range"
            min={WATERSHED_SLIDER_BOUNDS.min_distance_rel[0]}
            max={WATERSHED_SLIDER_BOUNDS.min_distance_rel[1]}
            step={WATERSHED_SLIDER_BOUNDS.min_distance_rel[2]}
            value={values.min_distance_rel}
            disabled={!hasActiveSource}
            onChange={(event) => {
              onChangeDraft({ ...values, min_distance_rel: Number(event.target.value) });
            }}
            onPointerUp={(event) =>
              commitFieldFromInput("min_distance_rel", event.currentTarget.value)
            }
            onKeyUp={(event) => handleRangeKeyUp("min_distance_rel", event)}
            className="min-h-[44px] w-full cursor-pointer accent-accent disabled:cursor-not-allowed disabled:opacity-60"
          />
          <div className="flex items-center justify-between text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
            <span>0,20</span>
            <span>Padrão: 0,85</span>
            <span>2,50</span>
          </div>
        </div>

        {/* Slider 3: min_area_ratio [0.01, 0.45] step 0.01 default 0.10 */}
        <div className="space-y-1 rounded-lg border border-border bg-canvas p-4">
          <div className="flex items-center justify-between gap-2">
            <label htmlFor={areaId} className="text-[12px] font-semibold leading-[1.33] text-ink">
              Área Mínima Relativa (min_area_ratio)
            </label>
            <span className="rounded bg-surface px-2 py-0.5 font-mono text-[12px] font-semibold leading-[1.33] tabular-nums text-ink">
              {formatNumberPtBr(values.min_area_ratio, 2)}
            </span>
          </div>
          <input
            id={areaId}
            type="range"
            min={WATERSHED_SLIDER_BOUNDS.min_area_ratio[0]}
            max={WATERSHED_SLIDER_BOUNDS.min_area_ratio[1]}
            step={WATERSHED_SLIDER_BOUNDS.min_area_ratio[2]}
            value={values.min_area_ratio}
            disabled={!hasActiveSource}
            onChange={(event) => {
              onChangeDraft({ ...values, min_area_ratio: Number(event.target.value) });
            }}
            onPointerUp={(event) =>
              commitFieldFromInput("min_area_ratio", event.currentTarget.value)
            }
            onKeyUp={(event) => handleRangeKeyUp("min_area_ratio", event)}
            className="min-h-[44px] w-full cursor-pointer accent-accent disabled:cursor-not-allowed disabled:opacity-60"
          />
          <div className="flex items-center justify-between text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
            <span>0,01</span>
            <span>Padrão: 0,10</span>
            <span>0,45</span>
          </div>
        </div>

        {/* Always-visible missing-EXIF fallback disclosure when an upload lacks EXIF optical tags */}
        {optical && !optical.exif_optical_present ? (
          <div
            role="note"
            className="flex items-start gap-2 rounded-lg border border-warning-border bg-warning-tint p-4 text-warning"
          >
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <div className="space-y-2">
              <p className="text-[14px] font-normal leading-[1.5]">
                Metadados ópticos EXIF ausentes nesta imagem — aplicando padrão Wide (4,25 mm · zoom
                1,0×). Se a foto foi capturada em lente Ultrawide, ajuste a calibração óptica abaixo
                e clique em 'Reprocessar Watershed'.
              </p>
              {!opticalOpen ? (
                <button
                  type="button"
                  onClick={() => setOpticalOpen(true)}
                  className="inline-flex min-h-[44px] items-center rounded-lg border border-warning-border bg-surface px-4 text-[12px] font-semibold leading-[1.33] text-warning transition-colors hover:bg-warning-tint"
                >
                  Abrir Calibração Óptica
                </button>
              ) : null}
            </div>
          </div>
        ) : null}

        {/* Optional collapsible optical calibration override section */}
        <div className="rounded-lg border border-border bg-surface-subtle/60">
          <button
            type="button"
            onClick={() => setOpticalOpen((prev) => !prev)}
            aria-expanded={opticalOpen}
            className="flex min-h-[44px] w-full items-center justify-between gap-2 px-4 text-left text-[12px] font-semibold leading-[1.33] text-ink"
          >
            <span>Calibração Óptica EXIF Opcional (Focal / Zoom)</span>
            {opticalOpen ? (
              <ChevronUp className="h-4 w-4 shrink-0 text-ink-muted" aria-hidden="true" />
            ) : (
              <ChevronDown className="h-4 w-4 shrink-0 text-ink-muted" aria-hidden="true" />
            )}
          </button>

          {opticalOpen ? (
            <div className="space-y-4 border-t border-border bg-surface p-4">
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                <button
                  type="button"
                  disabled={loading || !hasActiveSource}
                  onClick={() => {
                    setFocalInput("4.25");
                    setZoomInput("1.0");
                    setValidationError(null);
                  }}
                  className="min-h-[44px] rounded-lg border border-border bg-canvas px-2 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink hover:bg-surface-subtle disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Wide 4,25 mm / 1,0×
                </button>
                <button
                  type="button"
                  disabled={loading || !hasActiveSource}
                  onClick={() => {
                    setFocalInput("1.65");
                    setZoomInput("0.6");
                    setValidationError(null);
                  }}
                  className="min-h-[44px] rounded-lg border border-border bg-canvas px-2 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink hover:bg-surface-subtle disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Ultrawide 1,65 mm / 0,6×
                </button>
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="space-y-1">
                  <label
                    htmlFor={focalId}
                    className="block text-[12px] font-semibold leading-[1.33] text-ink-muted"
                  >
                    Focal (0,5–200 mm)
                  </label>
                  <input
                    id={focalId}
                    type="number"
                    inputMode="decimal"
                    min={MIN_FOCAL_MM}
                    max={MAX_FOCAL_MM}
                    step="0.05"
                    value={focalInput}
                    disabled={loading || !hasActiveSource}
                    onChange={(event) => {
                      setFocalInput(event.target.value);
                      setValidationError(null);
                    }}
                    className="min-h-[44px] w-full rounded-lg border border-border bg-canvas px-4 text-[14px] font-normal leading-[1.5] tabular-nums text-ink disabled:cursor-not-allowed disabled:opacity-60"
                  />
                </div>

                <div className="space-y-1">
                  <label
                    htmlFor={zoomId}
                    className="block text-[12px] font-semibold leading-[1.33] text-ink-muted"
                  >
                    Zoom (0,1–20×)
                  </label>
                  <input
                    id={zoomId}
                    type="number"
                    inputMode="decimal"
                    min={MIN_ZOOM_MULTIPLE}
                    max={MAX_ZOOM_MULTIPLE}
                    step="0.05"
                    value={zoomInput}
                    disabled={loading || !hasActiveSource}
                    onChange={(event) => {
                      setZoomInput(event.target.value);
                      setValidationError(null);
                    }}
                    className="min-h-[44px] w-full rounded-lg border border-border bg-canvas px-4 text-[14px] font-normal leading-[1.5] tabular-nums text-ink disabled:cursor-not-allowed disabled:opacity-60"
                  />
                </div>
              </div>
            </div>
          ) : null}
        </div>

        {validationError ? (
          <p
            role="alert"
            className="rounded-lg border border-danger-border bg-danger-tint p-4 text-[14px] font-normal leading-[1.5] text-danger"
          >
            {validationError}
          </p>
        ) : null}

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="submit"
            disabled={loading || !hasActiveSource}
            className="inline-flex min-h-[44px] flex-1 items-center justify-center gap-2 rounded-lg bg-accent px-4 text-[14px] font-normal leading-[1.5] text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-60"
          >
            <SlidersHorizontal className="h-4 w-4 shrink-0" aria-hidden="true" />
            <span>Reprocessar Watershed</span>
          </button>

          <button
            type="button"
            disabled={loading || !hasActiveSource}
            onClick={handleReset}
            className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-lg border border-border bg-surface px-4 text-[14px] font-normal leading-[1.5] text-ink transition-colors hover:bg-surface-subtle disabled:cursor-not-allowed disabled:opacity-60"
          >
            <RotateCcw className="h-4 w-4 shrink-0" aria-hidden="true" />
            <span>Restaurar Parâmetros Padrão</span>
          </button>
        </div>
      </form>
    </section>
  );
}
