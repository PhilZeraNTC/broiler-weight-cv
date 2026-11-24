import { CheckCircle2, Eye, Loader2, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, isAbortError, predictSingleBird } from "../../lib/api.ts";
import {
  formatGrams,
  formatNullableMetric,
  formatNumberPtBr,
  formatPercent,
} from "../../lib/format.ts";
import type { PredictResponse } from "../../types/api.ts";
import { InlineErrorBanner } from "../layout/InlineErrorBanner.tsx";

const DUAL_LENS_WIDE_ID = "sb_dual_lens_wide";
const DUAL_LENS_ULTRAWIDE_ID = "sb_dual_lens_ultrawide";

const FALLBACK_ERROR_COPY =
  "Não foi possível conectar à API local FastAPI em 127.0.0.1:8000. Verifique se o servidor Uvicorn está ativo no terminal e clique em 'Tentar Análise Novamente'.";

interface DualLensPairResult {
  wide: PredictResponse;
  ultrawide: PredictResponse;
}

export interface DualLensComparisonProps {
  isOpen: boolean;
  onClose: () => void;
  onInspectPreset: (presetId: string) => void;
}

export function DualLensComparison({ isOpen, onClose, onInspectPreset }: DualLensComparisonProps) {
  const [pair, setPair] = useState<DualLensPairResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<ApiError | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const loadPair = useCallback(() => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setLoading(true);
    setError(null);

    Promise.all([
      predictSingleBird({ presetId: DUAL_LENS_WIDE_ID }, controller.signal),
      predictSingleBird({ presetId: DUAL_LENS_ULTRAWIDE_ID }, controller.signal),
    ])
      .then(([wide, ultrawide]) => {
        if (controller.signal.aborted) {
          return;
        }
        setPair({ wide, ultrawide });
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (isAbortError(err) || controller.signal.aborted) {
          return;
        }
        setError(err instanceof ApiError ? err : new ApiError(0, FALLBACK_ERROR_COPY));
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (isOpen && pair === null && !loading && error === null) {
      loadPair();
    }
  }, [isOpen, pair, loading, error, loadPair]);

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);

  if (!isOpen) {
    return null;
  }

  const wideAreaPx = pair?.wide.features_summary.area_px ?? null;
  const uwAreaPx = pair?.ultrawide.features_summary.area_px ?? null;
  const rawAreaRatio =
    wideAreaPx !== null && uwAreaPx !== null && uwAreaPx > 0 ? wideAreaPx / uwAreaPx : null;

  const wideAreaNorm = pair?.wide.features_summary.area_optical_norm ?? null;
  const uwAreaNorm = pair?.ultrawide.features_summary.area_optical_norm ?? null;
  const normAreaRatio =
    wideAreaNorm !== null && uwAreaNorm !== null && uwAreaNorm > 0
      ? wideAreaNorm / uwAreaNorm
      : null;

  const predDeltaG = pair
    ? Math.abs(pair.wide.predicted_weight_g - pair.ultrawide.predicted_weight_g)
    : null;
  const predMeanG = pair
    ? (pair.wide.predicted_weight_g + pair.ultrawide.predicted_weight_g) / 2
    : null;
  const predDeltaPct =
    predDeltaG !== null && predMeanG !== null && predMeanG > 0
      ? (predDeltaG / predMeanG) * 100
      : null;

  const gtWeightG = pair?.wide.oof_audit?.ground_truth_weight_g ?? 541;
  const sessionId = pair?.wide.oof_audit?.session_id ?? "541__s0";
  const foldNumber = pair?.wide.oof_audit?.fold ?? 0;

  return (
    <section
      aria-label="Comparação lado a lado do Par Dual-Lens"
      className="rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-border pb-4">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Prova Empírica de Invariância Óptica EXIF · Out-of-Fold
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            Par Dual-Lens — Mesmo Frango ({formatGrams(gtWeightG, 0)} · Fold {foldNumber})
          </h2>
          <p className="text-[14px] font-normal leading-[1.5] text-ink-muted">
            Sessão <span className="font-mono tabular-nums text-ink">{sessionId}</span> avaliada
            fora do treino nas duas lentes nativas (Wide 4,25 mm vs. Ultrawide 1,65 mm) com
            normalização física{" "}
            <span className="font-mono tabular-nums text-ink">area_px / zoom²</span>.
          </p>
        </div>

        <button
          type="button"
          onClick={onClose}
          aria-label="Fechar comparação Dual-Lens"
          className="inline-flex min-h-[44px] items-center gap-2 rounded-lg border border-border bg-surface px-4 text-[14px] font-normal leading-[1.5] text-ink transition-colors hover:bg-surface-subtle"
        >
          <X className="h-4 w-4 shrink-0" aria-hidden="true" />
          <span>Fechar Comparação</span>
        </button>
      </div>

      {error ? (
        <div className="mt-4">
          <InlineErrorBanner
            title="Erro ao Carregar Comparação Dual-Lens"
            message={error.userMessage}
            actionLabel="Tentar Análise Novamente"
            onAction={loadPair}
          />
        </div>
      ) : null}

      {loading && pair === null ? (
        <div className="mt-4 space-y-4">
          <div className="flex items-center gap-2 rounded-lg border border-border bg-surface-subtle p-4 text-[14px] font-normal leading-[1.5] text-ink">
            <Loader2 className="h-4 w-4 shrink-0 animate-spin text-ink-muted" aria-hidden="true" />
            <span>Executando pipeline PDI clássico e regressão LightGBM...</span>
          </div>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {[1, 2].map((slot) => (
              <div
                key={slot}
                className="aspect-[4/3] w-full animate-pulse rounded-lg border border-slate-800 bg-viewport-dark"
              />
            ))}
          </div>
        </div>
      ) : pair !== null ? (
        <div className="mt-4 space-y-4">
          {/* Optical calibration invariance summary KPIs */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="rounded-lg border border-border bg-canvas p-4">
              <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
                Oscilação Bruta (<span className="font-mono">area_px</span> Wide / UW)
              </p>
              <p className="mt-1 text-[20px] font-semibold leading-[1.2] tabular-nums text-ink">
                {formatNullableMetric(rawAreaRatio, 2, "×")}
              </p>
              <p className="mt-1 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
                {formatNumberPtBr(wideAreaPx, 0)} px² vs. {formatNumberPtBr(uwAreaPx, 0)} px²
              </p>
            </div>

            <div className="rounded-lg border border-border bg-canvas p-4">
              <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
                Área Normalizada (<span className="font-mono">area_px / zoom²</span>)
              </p>
              <p className="mt-1 text-[20px] font-semibold leading-[1.2] tabular-nums text-ink">
                Razão {formatNullableMetric(normAreaRatio, 2, "×")}
              </p>
              <p className="mt-1 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
                {formatNumberPtBr(wideAreaNorm, 0)} vs. {formatNumberPtBr(uwAreaNorm, 0)}
              </p>
            </div>

            <div className="rounded-lg border border-accent/30 bg-accent-tint p-4">
              <div className="flex items-center gap-1 text-[12px] font-semibold leading-[1.33] text-accent">
                <CheckCircle2 className="h-4 w-4 shrink-0" aria-hidden="true" />
                <span>Diferença na Estimativa OOF (Wide vs. UW)</span>
              </div>
              <p className="mt-1 text-[20px] font-semibold leading-[1.2] tabular-nums text-accent">
                {formatGrams(predDeltaG, 2)} ({formatPercent(predDeltaPct, 2)})
              </p>
              <p className="mt-1 text-[12px] font-semibold leading-[1.33] tabular-nums text-accent">
                {formatGrams(pair.wide.predicted_weight_g, 1)} vs.{" "}
                {formatGrams(pair.ultrawide.predicted_weight_g, 1)}
              </p>
            </div>
          </div>

          {/* Side-by-side dark #0F172A Stage 4 overlays */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {(
              [
                {
                  presetId: DUAL_LENS_WIDE_ID,
                  title: "Lente Wide (4,25 mm)",
                  response: pair.wide,
                },
                {
                  presetId: DUAL_LENS_ULTRAWIDE_ID,
                  title: "Lente Ultrawide (1,65 mm)",
                  response: pair.ultrawide,
                },
              ] as const
            ).map(({ presetId, title, response }) => (
              <div
                key={presetId}
                className="flex flex-col rounded-xl border border-border bg-canvas p-4"
              >
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2">
                  <div>
                    <h3 className="text-[14px] font-semibold leading-[1.5] text-ink">{title}</h3>
                    <p className="text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
                      Focal {formatNumberPtBr(response.optical.focal_length_mm, 2)} mm · Zoom{" "}
                      {formatNumberPtBr(response.optical.zoom_multiple, 2)}× ·{" "}
                      {formatNumberPtBr(response.timing.total_ms, 0)} ms
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={() => onInspectPreset(presetId)}
                    className="inline-flex min-h-[44px] items-center gap-2 rounded-lg border border-border bg-surface px-4 text-[14px] font-normal leading-[1.5] text-ink transition-colors hover:bg-surface-subtle"
                  >
                    <Eye className="h-4 w-4 shrink-0" aria-hidden="true" />
                    <span>Inspecionar no Estúdio</span>
                  </button>
                </div>

                <div className="relative aspect-[4/3] w-full overflow-hidden rounded-lg border border-slate-800 bg-viewport-dark">
                  <img
                    src={response.xray.stage4_overlay}
                    alt={`Etapa 4 — ${title}`}
                    className="h-full w-full object-contain"
                  />
                  <span className="pointer-events-none absolute top-2 left-2 rounded-md bg-viewport-dark/85 px-2 py-1 text-[12px] font-semibold leading-[1.33] text-white">
                    4. Contorno + Elipse + Eixos
                  </span>
                </div>

                <div className="mt-4 grid grid-cols-3 gap-2 text-[12px] font-semibold leading-[1.33] tabular-nums">
                  <div className="rounded-lg border border-border bg-surface p-2">
                    <span className="block text-ink-muted">Estimado OOF</span>
                    <span className="mt-1 block text-[14px] font-semibold text-ink">
                      {formatGrams(response.predicted_weight_g, 1)}
                    </span>
                  </div>
                  <div className="rounded-lg border border-border bg-surface p-2">
                    <span className="block font-mono text-ink-muted">area_px</span>
                    <span className="mt-1 block text-[14px] font-normal text-ink">
                      {formatNumberPtBr(response.features_summary.area_px, 0)}
                    </span>
                  </div>
                  <div className="rounded-lg border border-border bg-surface p-2">
                    <span className="block font-mono text-ink-muted">area_optical_norm</span>
                    <span className="mt-1 block text-[14px] font-normal text-ink">
                      {formatNumberPtBr(response.features_summary.area_optical_norm, 0)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}
