import { AlertTriangle, CheckCircle2, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import { formatNumberPtBr, formatPercent } from "../../lib/format.ts";
import type { PredictResponse } from "../../types/api.ts";

export interface TelemetryPanelProps {
  prediction: PredictResponse | null;
  loading: boolean;
}

function clampRatioPct(value: number | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, value * 100));
}

export function TelemetryPanel({ prediction, loading }: TelemetryPanelProps) {
  const [expanded, setExpanded] = useState<boolean>(false);

  if (!prediction) {
    return (
      <section
        aria-label="Telemetria PDI Interpretável"
        aria-busy={loading}
        className="rounded-xl border border-border bg-surface p-4 lg:p-6"
      >
        <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
          Extração Estritamente em bird_mask == 255
        </p>
        <h2 className="mt-1 text-[20px] font-semibold leading-[1.2] text-ink">
          Telemetria PDI Interpretável
        </h2>
        <p className="mt-4 text-[14px] font-normal leading-[1.5] text-ink-muted">
          {loading
            ? "Executando pipeline PDI clássico e regressão LightGBM..."
            : "Os atributos morfométricos e cromáticos aparecerão aqui após a análise da imagem."}
        </p>
      </section>
    );
  }

  const { features_summary: summary, segmentation, model_features: modelFeatures } = prediction;
  const yellowPct = clampRatioPct(summary.yellow_down_ratio);
  const whitePct = clampRatioPct(summary.white_plumage_ratio);
  const otherPct = Math.max(0, 100 - Math.min(100, yellowPct + whitePct));
  const orderedModelFeatures = Object.entries(modelFeatures);

  const sceneLabel =
    segmentation.scene_type === "bucket_dorsal"
      ? "Balde Dorsal (bucket_dorsal)"
      : "Balança / Cama (scale_or_litter)";

  return (
    <section
      aria-label="Telemetria PDI Interpretável"
      className="space-y-4 rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Estritamente Dentro da Máscara · Zero Vazamento do LCD
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            Telemetria PDI Interpretável
          </h2>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded-full border border-accent/25 bg-accent-tint px-2 py-1 text-[12px] font-semibold leading-[1.33] text-accent">
            <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            <span>Invariância Externa 0,0 diff</span>
          </span>
          <span className="rounded-full border border-border bg-surface-subtle px-2 py-1 font-mono text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
            {prediction.pdi_config_hash}
          </span>
        </div>
      </div>

      {/* Conditional Amber warning when primary segmentation triggered fallback */}
      {segmentation.segmentation_fallback ? (
        <div
          role="note"
          className="flex items-start gap-2 rounded-lg border border-warning-border bg-warning-tint p-4 text-warning"
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          <p className="text-[14px] font-normal leading-[1.5]">
            Segmentação primária acionou fallback clássico nesta imagem devido a baixo contraste
            local. Verifique a Etapa 3 para auditoria da máscara obtida.
          </p>
        </div>
      ) : null}

      {/* Core morphometric descriptors */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        <div className="rounded-lg border border-border bg-canvas p-4">
          <p className="font-mono text-[12px] font-semibold leading-[1.33] text-ink-muted">
            area_optical_norm
          </p>
          <p className="mt-1 text-[14px] font-semibold leading-[1.5] tabular-nums text-ink">
            {formatNumberPtBr(summary.area_optical_norm, 0)}
          </p>
        </div>

        <div className="rounded-lg border border-border bg-canvas p-4">
          <p className="font-mono text-[12px] font-semibold leading-[1.33] text-ink-muted">
            area_px
          </p>
          <p className="mt-1 text-[14px] font-semibold leading-[1.5] tabular-nums text-ink">
            {formatNumberPtBr(summary.area_px, 0)} px²
          </p>
        </div>

        <div className="rounded-lg border border-border bg-canvas p-4">
          <p className="font-mono text-[12px] font-semibold leading-[1.33] text-ink-muted">
            major_axis_optical_norm
          </p>
          <p className="mt-1 text-[14px] font-semibold leading-[1.5] tabular-nums text-ink">
            {formatNumberPtBr(summary.major_axis_optical_norm, 1)}
          </p>
        </div>

        <div className="rounded-lg border border-border bg-canvas p-4">
          <p className="font-mono text-[12px] font-semibold leading-[1.33] text-ink-muted">
            minor_axis_optical_norm
          </p>
          <p className="mt-1 text-[14px] font-semibold leading-[1.5] tabular-nums text-ink">
            {formatNumberPtBr(summary.minor_axis_optical_norm, 1)}
          </p>
        </div>

        <div className="rounded-lg border border-border bg-canvas p-4">
          <p className="font-mono text-[12px] font-semibold leading-[1.33] text-ink-muted">
            solidity
          </p>
          <p className="mt-1 text-[14px] font-semibold leading-[1.5] tabular-nums text-ink">
            {formatNumberPtBr(summary.solidity, 3)}
          </p>
        </div>

        <div className="rounded-lg border border-border bg-canvas p-4">
          <p className="font-mono text-[12px] font-semibold leading-[1.33] text-ink-muted">
            eccentricity
          </p>
          <p className="mt-1 text-[14px] font-semibold leading-[1.5] tabular-nums text-ink">
            {formatNumberPtBr(summary.eccentricity, 3)}
          </p>
        </div>
      </div>

      {/* Plumage chroma proportion bar (yellow down vs white plumage) */}
      <div className="space-y-2 rounded-lg border border-border bg-canvas p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 text-[12px] font-semibold leading-[1.33] tabular-nums">
          <span className="text-ink">Perfil Cromático da Plumagem (Maturação)</span>
          <span className="text-ink-muted">
            Penugem Amarela {formatPercent(yellowPct, 1)} · Plumagem Branca{" "}
            {formatPercent(whitePct, 1)}
          </span>
        </div>

        <div
          role="img"
          aria-label={`Proporção cromática: ${formatPercent(yellowPct, 1)} penugem amarela e ${formatPercent(whitePct, 1)} plumagem branca`}
          className="flex h-3 w-full overflow-hidden rounded-full border border-border bg-surface"
        >
          <div
            style={{ width: `${yellowPct}%` }}
            className="h-full bg-warning-border transition-all"
          />
          <div
            style={{ width: `${Math.min(whitePct, 100 - yellowPct)}%` }}
            className="h-full bg-chart-slate transition-all"
          />
          {otherPct > 0 ? (
            <div style={{ width: `${otherPct}%` }} className="h-full bg-surface-subtle" />
          ) : null}
        </div>

        <div className="flex flex-wrap items-center gap-4 text-[12px] font-semibold leading-[1.33] text-ink-muted">
          <span className="inline-flex items-center gap-1">
            <span className="h-2.5 w-2.5 rounded-full bg-warning-border" aria-hidden="true" />
            <span className="font-mono">yellow_down_ratio</span>
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-2.5 w-2.5 rounded-full bg-chart-slate" aria-hidden="true" />
            <span className="font-mono">white_plumage_ratio</span>
          </span>
        </div>
      </div>

      {/* Scene classification & LCD anti-leakage audit */}
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3 text-[12px] font-semibold leading-[1.33] tabular-nums">
        <div className="rounded-lg border border-border bg-surface-subtle p-2">
          <span className="block text-ink-muted">Cenário Detectado</span>
          <span className="mt-1 block text-[14px] font-normal text-ink">{sceneLabel}</span>
        </div>

        <div className="rounded-lg border border-border bg-surface-subtle p-2">
          <span className="block text-ink-muted">Pixels LCD Excluídos</span>
          <span className="mt-1 block text-[14px] font-semibold text-ink">
            {formatNumberPtBr(segmentation.lcd_excluded_px, 0)} px
          </span>
        </div>

        <div className="rounded-lg border border-border bg-surface-subtle p-2">
          <span className="block text-ink-muted">Fração de Máscara</span>
          <span className="mt-1 block text-[14px] font-semibold text-ink">
            {formatPercent(segmentation.mask_area_frac * 100, 2)}
          </span>
        </div>
      </div>

      {/* Expandable 39-feature technical jury table */}
      <div className="rounded-lg border border-border bg-canvas">
        <div className="flex items-center justify-between gap-2 px-4 py-2">
          <div className="min-w-0">
            <p className="text-[14px] font-semibold leading-[1.5] text-ink">
              Tabela Completa de {orderedModelFeatures.length} Atributos PDI (Modelo C2)
            </p>
            <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
              Vetor tabular exato entregue ao regressor LightGBMLog
            </p>
          </div>

          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            aria-expanded={expanded}
            aria-label={
              expanded
                ? "Recolher tabela completa de 39 atributos PDI"
                : "Expandir tabela completa de 39 atributos PDI"
            }
            className="inline-flex min-h-[44px] min-w-[44px] shrink-0 items-center justify-center rounded-lg border border-border bg-surface text-ink transition-colors hover:bg-surface-subtle"
          >
            {expanded ? (
              <ChevronUp className="h-5 w-5" aria-hidden="true" />
            ) : (
              <ChevronDown className="h-5 w-5" aria-hidden="true" />
            )}
          </button>
        </div>

        {expanded ? (
          <div className="max-h-80 overflow-x-auto overflow-y-auto border-t border-border">
            <table className="w-full border-collapse text-left">
              <thead className="sticky top-0 bg-surface-subtle">
                <tr className="border-b border-border text-[12px] font-semibold leading-[1.33] text-ink-muted">
                  <th scope="col" className="w-12 px-4 py-2 tabular-nums">
                    #
                  </th>
                  <th scope="col" className="px-4 py-2">
                    Atributo PDI
                  </th>
                  <th scope="col" className="px-4 py-2 text-right">
                    Valor
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border bg-surface">
                {orderedModelFeatures.map(([featureKey, value], idx) => (
                  <tr key={featureKey} className="hover:bg-canvas">
                    <td className="px-4 py-2 font-mono text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
                      {idx + 1}
                    </td>
                    <td
                      title={featureKey}
                      className="max-w-[240px] break-all px-4 py-2 font-mono text-[12px] font-semibold leading-[1.33] text-ink"
                    >
                      {featureKey}
                    </td>
                    <td className="whitespace-nowrap px-4 py-2 text-right font-mono text-[14px] font-normal leading-[1.5] tabular-nums text-ink">
                      {formatNumberPtBr(value, 4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </section>
  );
}
