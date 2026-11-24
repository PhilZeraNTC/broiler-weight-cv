import { RotateCcw, SearchX } from "lucide-react";
import { formatGrams, formatNumberPtBr, formatPercent, pluralizePtBr } from "../../lib/format.ts";
import type { WatershedInstancePayload } from "../../types/api.ts";

const VALID_HEX_COLOR_RE = /^#[0-9A-Fa-f]{6}$/;

function sanitizeSwatchHex(hex: string): string {
  return VALID_HEX_COLOR_RE.test(hex) ? hex : "#047857";
}

export interface WatershedInstanceTableProps {
  instances: readonly WatershedInstancePayload[];
  birdCount: number;
  estimatedTotalWeightG: number;
  hasResponse: boolean;
  loading: boolean;
  onResetDefaults: () => void;
}

export function WatershedInstanceTable({
  instances,
  birdCount,
  estimatedTotalWeightG,
  hasResponse,
  loading,
  onResetDefaults,
}: WatershedInstanceTableProps) {
  return (
    <section
      aria-label="Detalhamento individual ilustrativo das bacias Watershed"
      className="rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-2 pb-4">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Atributos &amp; Estimativa Ilustrativa por Bacia (#ID sincronizado com Etapa 4)
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            Instâncias Separadas pelo Watershed
          </h2>
        </div>

        {hasResponse ? (
          <span className="rounded-full border border-border bg-surface-subtle px-2 py-1 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
            {pluralizePtBr(birdCount, "ave detectada", "aves detectadas")}
          </span>
        ) : null}
      </div>

      {hasResponse && birdCount === 0 ? (
        <div className="flex flex-col items-center justify-center gap-4 rounded-xl border border-border bg-canvas p-6 text-center">
          <div className="flex h-11 w-11 items-center justify-center rounded-lg border border-border bg-surface text-ink-muted">
            <SearchX className="h-5 w-5" aria-hidden="true" />
          </div>
          <div className="max-w-lg space-y-1">
            <h3 className="text-[20px] font-semibold leading-[1.2] text-ink">
              Nenhuma ave separada com os parâmetros atuais
            </h3>
            <p className="text-[14px] font-normal leading-[1.5] text-ink-muted">
              Os limiares selecionados filtraram todos os componentes da cena. Reduza 'Área Mínima
              Relativa' ou clique em 'Restaurar Parâmetros Padrão' para recalcular.
            </p>
          </div>
          <button
            type="button"
            disabled={loading}
            onClick={onResetDefaults}
            className="inline-flex min-h-[44px] items-center gap-2 rounded-lg border border-border bg-surface px-4 text-[14px] font-normal leading-[1.5] text-ink transition-colors hover:bg-surface-subtle disabled:cursor-not-allowed disabled:opacity-60"
          >
            <RotateCcw className="h-4 w-4 shrink-0" aria-hidden="true" />
            <span>Restaurar Parâmetros Padrão</span>
          </button>
        </div>
      ) : instances.length === 0 ? (
        <p className="rounded-lg border border-border bg-canvas p-4 text-[14px] font-normal leading-[1.5] text-ink-muted">
          Aguardando execução do módulo Watershed para listar as instâncias separadas.
        </p>
      ) : (
        <div className="space-y-4">
          <div className="overflow-x-auto rounded-lg border border-border">
            <table className="w-full min-w-[680px] border-collapse text-left">
              <thead>
                <tr className="border-b border-border bg-surface-subtle text-[12px] font-semibold leading-[1.33] text-ink-muted">
                  <th scope="col" className="px-4 py-2">
                    #ID
                  </th>
                  <th scope="col" className="px-4 py-2 text-right">
                    Área (px²)
                  </th>
                  <th scope="col" className="px-4 py-2 text-right">
                    Área Óptica Norm.
                  </th>
                  <th scope="col" className="px-4 py-2 text-right">
                    Peso Ilustrativo (g)
                  </th>
                  <th scope="col" className="px-4 py-2 text-right">
                    % da Soma Estimada
                  </th>
                  <th scope="col" className="px-4 py-2">
                    Estágio Estimado
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border text-[14px] font-normal leading-[1.5] text-ink">
                {instances.map((inst) => {
                  const swatchHex = sanitizeSwatchHex(inst.color_hex);
                  return (
                    <tr key={inst.instance_id} className="hover:bg-canvas">
                      <td className="px-4 py-2">
                        <span className="inline-flex items-center gap-2 font-mono text-[12px] font-semibold leading-[1.33] tabular-nums text-ink">
                          <span
                            aria-hidden="true"
                            className="h-3 w-3 shrink-0 rounded-sm border border-viewport-dark/25"
                            style={{ backgroundColor: swatchHex }}
                          />
                          <span>#{inst.instance_id}</span>
                        </span>
                      </td>
                      <td className="px-4 py-2 text-right tabular-nums">
                        {formatNumberPtBr(inst.area_px, 0)}
                      </td>
                      <td className="px-4 py-2 text-right tabular-nums">
                        {formatNumberPtBr(inst.area_optical_norm, 0)}
                      </td>
                      <td className="px-4 py-2 text-right font-semibold tabular-nums text-ink">
                        {formatGrams(inst.estimated_weight_g, 1)}
                      </td>
                      <td className="px-4 py-2 text-right tabular-nums text-ink-muted">
                        {formatPercent(inst.share_of_total_pct, 1)}
                      </td>
                      <td className="px-4 py-2">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-[14px] font-normal leading-[1.5] text-ink">
                            {inst.age_stage.stage_label}
                          </span>
                          <span className="rounded-full border border-border bg-surface-subtle px-2 py-0.5 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
                            {inst.age_stage.estimated_age_days_range}
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
              <tfoot>
                <tr className="border-t border-border bg-surface-subtle/60 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink">
                  <td colSpan={3} className="px-4 py-2">
                    Total ilustrativo (
                    {pluralizePtBr(birdCount, "ave detectada", "aves detectadas")})
                  </td>
                  <td className="px-4 py-2 text-right">{formatGrams(estimatedTotalWeightG, 1)}</td>
                  <td className="px-4 py-2 text-right">100,0%</td>
                  <td className="px-4 py-2 text-ink-muted">Escopo: group_sum_only</td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}
