import { ShieldCheck } from "lucide-react";
import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatNumberPtBr, formatPercent } from "../../lib/format.ts";
import type { FeatureImportanceItem } from "../../types/api.ts";

export interface FeatureImportanceChartProps {
  featureImportances: readonly FeatureImportanceItem[];
  totalModelFeatures?: number;
}

interface ImportanceTooltipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload?: FeatureImportanceItem }>;
}

function ImportanceTooltipContent({ active, payload }: ImportanceTooltipProps) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  const item = payload[0]?.payload;
  if (!item) {
    return null;
  }

  return (
    <div className="rounded-lg border border-border bg-surface p-4 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink shadow-md">
      <p className="font-mono text-[12px] font-semibold text-ink">{item.feature}</p>
      <p className="mt-1 text-[14px] font-semibold text-accent">
        Importância relativa: {formatPercent(item.importance_pct, 2)}
      </p>
      <p className="mt-0.5 font-normal text-ink-muted">
        Ganho bruto LightGBM: {formatNumberPtBr(item.raw_importance, 1)}
      </p>
    </div>
  );
}

export function FeatureImportanceChart({
  featureImportances,
  totalModelFeatures = 39,
}: FeatureImportanceChartProps) {
  const top12 = useMemo(() => {
    return [...featureImportances].sort((a, b) => b.importance_pct - a.importance_pct).slice(0, 12);
  }, [featureImportances]);

  const top12SharePct = useMemo(
    () => top12.reduce((acc, item) => acc + item.importance_pct, 0),
    [top12],
  );

  return (
    <section
      aria-label="Top-12 atributos mais importantes do LightGBMLog C2"
      className="flex flex-col gap-4 rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Interpretabilidade Clássica · {formatNumberPtBr(totalModelFeatures, 0)} Atributos
            Estritamente na Máscara
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            Top-12 Atributos Mais Importantes (LightGBMLog C2)
          </h2>
        </div>

        <span className="rounded-full border border-border bg-surface-subtle px-2 py-1 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
          Top-12 = {formatPercent(top12SharePct, 1)} do ganho
        </span>
      </div>

      {/* Explicit-height Recharts v3 horizontal BarChart (zero deprecated <Cell />) */}
      <div className="h-[380px] w-full min-w-0">
        <ResponsiveContainer width="100%" height={380}>
          <BarChart
            data={top12}
            layout="vertical"
            margin={{ top: 4, right: 24, bottom: 4, left: 8 }}
          >
            <CartesianGrid stroke="#E2E8F0" strokeDasharray="3 3" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fontSize: 12, fill: "#475569" }}
              tickFormatter={(v: number) => `${formatNumberPtBr(v, 1)}%`}
            />
            <YAxis
              type="category"
              dataKey="feature"
              width={186}
              interval={0}
              tick={{
                fontSize: 12,
                fill: "#0F172A",
                fontFamily: "var(--font-mono)",
              }}
            />
            <Tooltip content={<ImportanceTooltipContent />} />
            <Bar
              dataKey="importance_pct"
              name="Importância (%)"
              fill="#047857"
              radius={[0, 4, 4, 0]}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Audit compliance footnote */}
      <div className="flex items-start gap-2 rounded-lg border border-border bg-canvas p-4 text-[14px] font-normal leading-[1.5] text-ink-muted">
        <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-accent" aria-hidden="true" />
        <p>
          Os atributos dominantes combinam cromaticidade de plumagem (
          <span className="font-mono text-[12px] text-ink">lab_a_*</span>,{" "}
          <span className="font-mono text-[12px] text-ink">feather_grad_mean</span>) e morfometria
          normalizada pela óptica (
          <span className="font-mono text-[12px] text-ink">minor_axis_optical_norm</span>,{" "}
          <span className="font-mono text-[12px] text-ink">area_optical_norm</span>). Zero dimensões
          brutas de resolução, zero IDs de lente EXIF e zero pixels fora de{" "}
          <span className="font-mono text-[12px] text-ink">bird_mask == 255</span> (visor LCD)
          entram no modelo.
        </p>
      </div>
    </section>
  );
}
