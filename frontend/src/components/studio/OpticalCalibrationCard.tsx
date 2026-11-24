import { AlertTriangle, Info, RotateCcw, SlidersHorizontal } from "lucide-react";
import { type FormEvent, useEffect, useId, useState } from "react";
import { cn } from "../../lib/cn.ts";
import { formatNumberPtBr } from "../../lib/format.ts";
import type { OpticalLensClass, OpticalPayload } from "../../types/api.ts";

const MIN_FOCAL_MM = 0.5;
const MAX_FOCAL_MM = 200.0;
const MIN_ZOOM_MULTIPLE = 0.1;
const MAX_ZOOM_MULTIPLE = 20.0;

const OUT_OF_RANGE_ERROR_COPY =
  "Um ou mais parâmetros estão fora dos limites aceitos (distância focal 0,5–200 mm; zoom 0,1–20×; limites do Watershed). Ajuste o campo destacado ou clique em 'Restaurar EXIF Original' (Aba 1) / 'Restaurar Parâmetros Padrão' (Aba 2).";

export interface OpticalOverrideValues {
  focalLengthMm: number;
  zoomMultiple: number;
}

export interface OpticalCalibrationCardProps {
  optical: OpticalPayload | null;
  loading: boolean;
  hasActiveSource: boolean;
  onApplyOverride: (values: OpticalOverrideValues) => void;
  onResetExif: () => void;
}

function formatOpticalLensLabel(lensClass: OpticalLensClass): string {
  if (lensClass === "wide") return "Wide";
  if (lensClass === "ultrawide") return "Ultrawide";
  return "Desconhecida";
}

function parseLocaleFloat(raw: string): number {
  const normalized = raw.trim().replace(",", ".");
  if (!normalized) {
    return Number.NaN;
  }
  return Number(normalized);
}

export function OpticalCalibrationCard({
  optical,
  loading,
  hasActiveSource,
  onApplyOverride,
  onResetExif,
}: OpticalCalibrationCardProps) {
  const focalId = useId();
  const zoomId = useId();
  const [focalInput, setFocalInput] = useState<string>("4.25");
  const [zoomInput, setZoomInput] = useState<string>("1.00");
  const [infoOpen, setInfoOpen] = useState<boolean>(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  useEffect(() => {
    if (!optical) {
      return;
    }
    setFocalInput(String(Number(optical.focal_length_mm.toFixed(2))));
    setZoomInput(String(Number(optical.zoom_multiple.toFixed(2))));
    setValidationError(null);
  }, [optical]);

  const applyQuickPreset = (focal: number, zoom: number) => {
    setFocalInput(String(focal));
    setZoomInput(String(zoom));
    setValidationError(null);
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const parsedFocal = parseLocaleFloat(focalInput);
    const parsedZoom = parseLocaleFloat(zoomInput);

    const focalValid =
      Number.isFinite(parsedFocal) && parsedFocal >= MIN_FOCAL_MM && parsedFocal <= MAX_FOCAL_MM;
    const zoomValid =
      Number.isFinite(parsedZoom) &&
      parsedZoom >= MIN_ZOOM_MULTIPLE &&
      parsedZoom <= MAX_ZOOM_MULTIPLE;

    if (!focalValid || !zoomValid) {
      setValidationError(OUT_OF_RANGE_ERROR_COPY);
      return;
    }

    setValidationError(null);
    onApplyOverride({
      focalLengthMm: parsedFocal,
      zoomMultiple: parsedZoom,
    });
  };

  const handleReset = () => {
    setValidationError(null);
    onResetExif();
  };

  const parsedFocalPreview = parseLocaleFloat(focalInput);
  const parsedZoomPreview = parseLocaleFloat(zoomInput);
  const isWideDraft =
    Number.isFinite(parsedFocalPreview) &&
    Number.isFinite(parsedZoomPreview) &&
    Math.abs(parsedFocalPreview - 4.25) < 0.01 &&
    Math.abs(parsedZoomPreview - 1.0) < 0.01;
  const isUltrawideDraft =
    Number.isFinite(parsedFocalPreview) &&
    Number.isFinite(parsedZoomPreview) &&
    Math.abs(parsedFocalPreview - 1.65) < 0.01 &&
    Math.abs(parsedZoomPreview - 0.6) < 0.01;

  return (
    <section
      aria-label="Calibração Óptica EXIF e Override Manual"
      className="space-y-4 rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Normalização Física · EXIF &amp; Xiaomi Tag 39321
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            Calibração Óptica EXIF
          </h2>
        </div>

        <button
          type="button"
          onClick={() => setInfoOpen((prev) => !prev)}
          aria-expanded={infoOpen}
          aria-label="Explicação sobre calibração óptica EXIF"
          className="inline-flex min-h-[44px] min-w-[44px] shrink-0 items-center justify-center rounded-lg border border-border bg-surface text-ink-muted transition-colors hover:bg-surface-subtle hover:text-ink"
        >
          <Info className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>

      {infoOpen ? (
        <div
          role="note"
          className="rounded-lg border border-border bg-surface-subtle p-4 text-[14px] font-normal leading-[1.5] text-ink"
        >
          A alternância entre lentes Wide (<span className="tabular-nums">4,25 mm</span>) e
          Ultrawide (<span className="tabular-nums">1,65 mm</span>) altera a área projetada da ave
          em pixels em até <span className="tabular-nums">6,6×</span>. O motor clássico aplica{" "}
          <span className="font-mono tabular-nums">area_optical_norm = area_px / zoom²</span>{" "}
          (condição <span className="font-mono">C2</span>) para neutralizar essa escala óptica sem
          vazar a identidade da lente ao regressor.
        </div>
      ) : null}

      {/* Detected EXIF readout */}
      <div className="grid grid-cols-3 gap-2 text-[12px] font-semibold leading-[1.33] tabular-nums">
        <div className="rounded-lg border border-border bg-canvas p-2">
          <span className="block text-ink-muted">Focal (mm)</span>
          <span className="mt-1 block text-[14px] font-semibold text-ink">
            {optical ? `${formatNumberPtBr(optical.focal_length_mm, 2)} mm` : "—"}
          </span>
        </div>
        <div className="rounded-lg border border-border bg-canvas p-2">
          <span className="block text-ink-muted">Zoom</span>
          <span className="mt-1 block text-[14px] font-semibold text-ink">
            {optical ? `${formatNumberPtBr(optical.zoom_multiple, 2)}×` : "—"}
          </span>
        </div>
        <div className="rounded-lg border border-border bg-canvas p-2">
          <span className="block text-ink-muted">Lente</span>
          <span className="mt-1 block text-[14px] font-semibold text-ink">
            {optical ? formatOpticalLensLabel(optical.lens_class) : "—"}
          </span>
        </div>
      </div>

      {optical ? (
        <div className="flex flex-wrap items-center gap-2 text-[12px] font-semibold leading-[1.33]">
          <span className="rounded-full border border-border bg-surface-subtle px-2 py-1 text-ink-muted">
            {optical.exif_optical_present ? "EXIF Óptico Detectado" : "EXIF Óptico Ausente"}
          </span>
          <span className="rounded-full border border-border bg-surface-subtle px-2 py-1 text-ink-muted">
            {optical.override_applied ? "Override Manual Ativo" : "Metadado Original"}
          </span>
        </div>
      ) : null}

      {/* Mandatory missing-EXIF Amber fallback banner (verbatim 04-UI-SPEC.md:245) */}
      {optical && !optical.exif_optical_present ? (
        <div
          role="note"
          className="flex items-start gap-2 rounded-lg border border-warning-border bg-warning-tint p-4 text-warning"
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          <p className="text-[14px] font-normal leading-[1.5]">
            Metadados ópticos EXIF ausentes nesta imagem — aplicando padrão Wide (4,25 mm · zoom
            1,0×). Se a foto foi capturada em lente Ultrawide, ajuste os controles de calibração e
            clique em 'Recalibrar Óptica'.
          </p>
        </div>
      ) : null}

      {/* Quick lens preset buttons */}
      <div className="space-y-2">
        <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
          Atalhos Rápidos de Lente
        </p>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <button
            type="button"
            disabled={loading || !hasActiveSource}
            onClick={() => applyQuickPreset(4.25, 1.0)}
            className={cn(
              "min-h-[44px] rounded-lg border px-2 text-[12px] font-semibold leading-[1.33] tabular-nums transition-colors disabled:cursor-not-allowed disabled:opacity-60",
              isWideDraft
                ? "border-ink bg-surface-subtle text-ink"
                : "border-border bg-surface text-ink-muted hover:bg-canvas hover:text-ink",
            )}
          >
            Wide 4,25 mm / 1,0×
          </button>
          <button
            type="button"
            disabled={loading || !hasActiveSource}
            onClick={() => applyQuickPreset(1.65, 0.6)}
            className={cn(
              "min-h-[44px] rounded-lg border px-2 text-[12px] font-semibold leading-[1.33] tabular-nums transition-colors disabled:cursor-not-allowed disabled:opacity-60",
              isUltrawideDraft
                ? "border-ink bg-surface-subtle text-ink"
                : "border-border bg-surface text-ink-muted hover:bg-canvas hover:text-ink",
            )}
          >
            Ultrawide 1,65 mm / 0,6×
          </button>
        </div>
      </div>

      {/* Bounded numeric manual override form */}
      <form onSubmit={handleSubmit} noValidate className="space-y-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="space-y-1">
            <label
              htmlFor={focalId}
              className="block text-[12px] font-semibold leading-[1.33] text-ink-muted"
            >
              Distância Focal (0,5–200 mm)
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
              Zoom Óptico/Digital (0,1–20×)
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
            <span>Recalibrar Óptica</span>
          </button>

          <button
            type="button"
            disabled={loading || !hasActiveSource}
            onClick={handleReset}
            className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-lg border border-border bg-surface px-4 text-[14px] font-normal leading-[1.5] text-ink transition-colors hover:bg-surface-subtle disabled:cursor-not-allowed disabled:opacity-60"
          >
            <RotateCcw className="h-4 w-4 shrink-0" aria-hidden="true" />
            <span>Restaurar EXIF Original</span>
          </button>
        </div>
      </form>
    </section>
  );
}
