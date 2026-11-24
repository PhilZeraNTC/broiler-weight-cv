import { AlertTriangle, CheckCircle2 } from "lucide-react";
import { cn } from "../../lib/cn.ts";
import { formatNullableMetric, formatNumberPtBr, formatPercent } from "../../lib/format.ts";
import type { AblationConditionKey, ModelComparisonRow } from "../../types/api.ts";

export interface ModelComparisonTableProps {
  rows: readonly ModelComparisonRow[];
}

const CONDITION_LABELS: Readonly<Record<AblationConditionKey, string>> = {
  C1_uncalibrated_pdi: "C1 · PDI Sem Calibração",
  C2_physics_normalized_strict: "C2 · Física Estrita (area/zoom²)",
  C3_physics_plus_exif_meta: "C3 · Física + EXIF Meta",
  C4_naive_double_count_fxz: "C4 · Dupla Contagem (f×zoom)²",
};

export function ModelComparisonTable({ rows }: ModelComparisonTableProps) {
  return (
    <section
      aria-label="Comparativo completo de 16 combinações modelo versus condição óptica"
      className="space-y-4 rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Matriz Completa 5-Fold OOF · 4 Famílias Tabulares × 4 Condições Ópticas
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            Comparativo Completo de Modelos ({formatNumberPtBr(rows.length, 0)} Combinações)
          </h2>
        </div>

        <span className="inline-flex items-center gap-1 rounded-full border border-accent/30 bg-accent-tint px-2 py-1 text-[12px] font-semibold leading-[1.33] text-accent">
          <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          <span>Vencedor: LightGBMLog C2</span>
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full min-w-[960px] border-collapse text-left">
          <thead>
            <tr className="border-b border-border bg-surface-subtle text-[12px] font-semibold leading-[1.33] text-ink-muted">
              <th scope="col" className="px-4 py-2">
                Modelo
              </th>
              <th scope="col" className="px-4 py-2">
                Condição
              </th>
              <th scope="col" className="px-4 py-2 text-right">
                Atributos
              </th>
              <th scope="col" className="px-4 py-2 text-right">
                MAE (g)
              </th>
              <th scope="col" className="px-4 py-2 text-right">
                RMSE (g)
              </th>
              <th scope="col" className="px-4 py-2 text-right">
                BalMAPE (%)
              </th>
              <th scope="col" className="px-4 py-2 text-right">
                R² Global
              </th>
              <th scope="col" className="px-4 py-2 text-right">
                R² Sessão
              </th>
              <th scope="col" className="px-4 py-2 text-right">
                Shift Dual-Lens (%)
              </th>
              <th scope="col" className="px-4 py-2">
                Status
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border text-[14px] font-normal leading-[1.5] text-ink">
            {rows.map((row) => {
              const isWinner = row.is_winner;
              const isDisqualified = !row.eligible_for_production || row.lens_confounding_detected;
              const swapShift = row.counterfactual_lens_flip.physical_lens_swap.median_shift_pct;
              const focalShift = row.counterfactual_lens_flip.focal_only.median_shift_pct;
              const displayShift = Math.max(swapShift, focalShift);

              return (
                <tr
                  key={`${row.model_name}-${row.ablation_condition}`}
                  className={cn(
                    isWinner
                      ? "bg-accent-tint/70"
                      : isDisqualified
                        ? "bg-warning-tint/40"
                        : "hover:bg-canvas",
                  )}
                >
                  <td className="px-4 py-2 font-semibold text-ink">{row.model_name}</td>
                  <td className="px-4 py-2">
                    <span
                      title={row.ablation_condition}
                      className="block max-w-[240px] truncate text-[14px] font-normal leading-[1.5] text-ink"
                    >
                      {CONDITION_LABELS[row.ablation_condition]}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums text-ink-muted">
                    {formatNumberPtBr(row.n_features, 0)}
                  </td>
                  <td className="px-4 py-2 text-right font-semibold tabular-nums">
                    {formatNumberPtBr(row.global.mae_g, 2)}
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums text-ink-muted">
                    {formatNumberPtBr(row.global.rmse_g, 2)}
                  </td>
                  <td className="px-4 py-2 text-right font-semibold tabular-nums">
                    {formatPercent(row.balanced_band_mape_pct, 2)}
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums text-ink-muted">
                    {formatNullableMetric(row.global.r2, 4)}
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums text-ink-muted">
                    {formatNullableMetric(row.session_level.session_averaged_r2, 4)}
                  </td>
                  <td
                    className={cn(
                      "px-4 py-2 text-right font-semibold tabular-nums",
                      isWinner ? "text-accent" : isDisqualified ? "text-warning" : "text-ink-muted",
                    )}
                  >
                    {formatPercent(displayShift, 1)}
                  </td>
                  <td className="px-4 py-2">
                    {isWinner ? (
                      <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-full border border-accent/30 bg-accent-tint px-2 py-0.5 text-[12px] font-semibold leading-[1.33] text-accent">
                        <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                        <span>Vencedor C2 · 0,0% Shift</span>
                      </span>
                    ) : isDisqualified ? (
                      <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-full border border-warning-border bg-warning-tint px-2 py-0.5 text-[12px] font-semibold leading-[1.33] text-warning">
                        <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                        <span>Desqualificada · Atalho Lente</span>
                      </span>
                    ) : (
                      <span className="inline-flex items-center whitespace-nowrap rounded-full border border-border bg-surface-subtle px-2 py-0.5 text-[12px] font-semibold leading-[1.33] text-ink-muted">
                        Elegível
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
