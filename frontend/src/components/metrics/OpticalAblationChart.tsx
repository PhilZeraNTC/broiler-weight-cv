import { AlertTriangle, CheckCircle2 } from "lucide-react";
import { useMemo } from "react";
import {
  Bar,
  BarChart,
  type BarShapeProps,
  CartesianGrid,
  Rectangle,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { cn } from "../../lib/cn.ts";
import { formatGrams, formatNumberPtBr, formatPercent } from "../../lib/format.ts";
import type { AblationConditionKey, EvaluationReport } from "../../types/api.ts";

export interface OpticalAblationChartProps {
  opticalAblation: EvaluationReport["optical_ablation"];
}

interface AblationDatum {
  key: AblationConditionKey;
  shortLabel: string;
  fullLabel: string;
  mae_g: number;
  bal_mape_pct: number;
  focalShiftPct: number;
  swapShiftPct: number;
  eligible: boolean;
  isWinner: boolean;
  fill: string;
}

const CONDITION_META: ReadonlyArray<{
  key: AblationConditionKey;
  shortLabel: string;
  fullLabel: string;
  barFill: string;
}> = [
  {
    key: "C1_uncalibrated_pdi",
    shortLabel: "C1 · Base",
    fullLabel: "C1 — PDI Sem Calibração Óptica",
    barFill: "#94A3B8",
  },
  {
    key: "C2_physics_normalized_strict",
    shortLabel: "C2 · Física",
    fullLabel: "C2 — Física Estrita (area / zoom²)",
    barFill: "#047857",
  },
  {
    key: "C3_physics_plus_exif_meta",
    shortLabel: "C3 · +EXIF",
    fullLabel: "C3 — Física + Metadados EXIF",
    barFill: "#B45309",
  },
  {
    key: "C4_naive_double_count_fxz",
    shortLabel: "C4 · f×zoom",
    fullLabel: "C4 — Dupla Contagem (f × zoom)²",
    barFill: "#B45309",
  },
];

interface AblationTooltipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload?: AblationDatum }>;
}

function AblationTooltipContent({ active, payload }: AblationTooltipProps) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  const datum = payload[0]?.payload;
  if (!datum) {
    return null;
  }

  return (
    <div className="rounded-lg border border-border bg-surface p-4 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink shadow-md">
      <p className="border-b border-border pb-2 text-[14px] font-semibold leading-[1.5] text-ink">
        {datum.fullLabel}
      </p>
      <dl className="mt-2 space-y-1">
        <div className="flex items-center justify-between gap-4">
          <dt className="text-ink-muted">MAE Global OOF:</dt>
          <dd className="text-ink">{formatGrams(datum.mae_g, 2)}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-ink-muted">BalMAPE (3 faixas):</dt>
          <dd className="text-ink">{formatPercent(datum.bal_mape_pct, 2)}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-ink-muted">Shift Focal-Only:</dt>
          <dd className="text-ink">{formatPercent(datum.focalShiftPct, 2)}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-ink-muted">Shift Troca de Lente:</dt>
          <dd className="text-ink">{formatPercent(datum.swapShiftPct, 2)}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-ink-muted">Elegibilidade:</dt>
          <dd
            className={
              datum.isWinner ? "text-accent" : datum.eligible ? "text-ink" : "text-warning"
            }
          >
            {datum.isWinner
              ? "Vencedor C2 · 0,0% Shift"
              : datum.eligible
                ? "Baseline Elegível"
                : "Desqualificada (Atalho de Lente)"}
          </dd>
        </div>
      </dl>
    </div>
  );
}

function renderBalMapeBarShape(props: BarShapeProps) {
  return <Rectangle {...props} fill="#0369A1" radius={[4, 4, 0, 0]} />;
}

export function OpticalAblationChart({ opticalAblation }: OpticalAblationChartProps) {
  const chartData = useMemo<AblationDatum[]>(() => {
    return CONDITION_META.map((meta) => {
      const cond = opticalAblation.conditions[meta.key];
      const cf = cond.counterfactual_lens_flip;
      return {
        key: meta.key,
        shortLabel: meta.shortLabel,
        fullLabel: meta.fullLabel,
        mae_g: cond.global.mae_g,
        bal_mape_pct: cond.balanced_band_mape_pct,
        focalShiftPct: cf.focal_only.median_shift_pct,
        swapShiftPct: cf.physical_lens_swap.median_shift_pct,
        eligible: cond.eligible_for_production,
        isWinner: meta.key === "C2_physics_normalized_strict",
        // Recharts v3 reads datum.fill directly when <Bar> omits a hardcoded fill prop (zero <Cell />)
        fill: meta.barFill,
      };
    });
  }, [opticalAblation]);

  const deltaC2 = opticalAblation.deltas_vs_c1.C2_physics_normalized_strict;

  return (
    <section
      aria-label="Ablação Óptica EXIF C1 a C4 e Auditoria Contrafactual"
      className="flex flex-col gap-4 rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Ablação 4-Vias ({opticalAblation.reference_model_name}) · Prova Anti-Confounding
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            Ablação Óptica EXIF (C1–C4)
          </h2>
        </div>

        <span className="inline-flex items-center gap-1 rounded-full border border-accent/30 bg-accent-tint px-2 py-1 text-[12px] font-semibold leading-[1.33] tabular-nums text-accent">
          <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          <span>Vencedor C2 · 0,0% Shift</span>
        </span>
      </div>

      {/* Explicit-height Recharts v3 grouped BarChart (zero deprecated <Cell />) */}
      <div className="h-[320px] w-full min-w-0">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData} margin={{ top: 12, right: 12, bottom: 8, left: 0 }} barGap={6}>
            <CartesianGrid stroke="#E2E8F0" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="shortLabel"
              tick={{ fontSize: 12, fill: "#475569", fontWeight: 600 }}
              interval={0}
            />
            <YAxis
              yAxisId="mae"
              orientation="left"
              domain={[140, 175]}
              tick={{ fontSize: 12, fill: "#475569" }}
              tickFormatter={(v: number) => `${formatNumberPtBr(v, 0)} g`}
              width={56}
            />
            <YAxis
              yAxisId="mape"
              orientation="right"
              domain={[30, 40]}
              tick={{ fontSize: 12, fill: "#0369A1" }}
              tickFormatter={(v: number) => `${formatNumberPtBr(v, 0)}%`}
              width={44}
            />
            <Tooltip content={<AblationTooltipContent />} />
            <Bar
              yAxisId="mae"
              dataKey="mae_g"
              name="MAE (g)"
              radius={[4, 4, 0, 0]}
              isAnimationActive={false}
            />
            <Bar
              yAxisId="mape"
              dataKey="bal_mape_pct"
              name="BalMAPE (%)"
              fill="#0369A1"
              shape={renderBalMapeBarShape}
              radius={[4, 4, 0, 0]}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Legend + C2 gain summary */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
        <div className="flex flex-wrap items-center gap-4">
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden="true" className="h-2.5 w-2.5 rounded-xs bg-accent" />
            <span>MAE C2 (Elegível)</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden="true" className="h-2.5 w-2.5 rounded-xs bg-chart-slate" />
            <span>MAE C1 (Base)</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden="true" className="h-2.5 w-2.5 rounded-xs bg-warning" />
            <span>MAE C3/C4 (Desqual.)</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden="true" className="h-2.5 w-2.5 rounded-xs bg-chart-sky" />
            <span>BalMAPE (%)</span>
          </span>
        </div>

        <span className="text-accent">
          C2 vs. C1: {formatNumberPtBr(deltaC2.delta_mae_g, 2)} g MAE ·{" "}
          {formatNumberPtBr(deltaC2.delta_balanced_band_mape_pct, 2)} pp BalMAPE
        </span>
      </div>

      {/* Dual-Mode Counterfactual Shift proof strip */}
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {chartData.map((row) => (
          <div
            key={row.key}
            className={cn(
              "rounded-lg border p-2 text-[12px] font-semibold leading-[1.33] tabular-nums",
              row.isWinner
                ? "border-accent/30 bg-accent-tint text-accent"
                : row.eligible
                  ? "border-border bg-canvas text-ink"
                  : "border-warning-border bg-warning-tint text-warning",
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="truncate">{row.shortLabel}</span>
              {row.isWinner ? (
                <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
              ) : !row.eligible ? (
                <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
              ) : null}
            </div>
            <p className="mt-1 font-normal text-ink">
              Shift troca de lente:{" "}
              <span className="font-semibold">{formatPercent(row.swapShiftPct, 1)}</span> · focal:{" "}
              <span className="font-semibold">{formatPercent(row.focalShiftPct, 1)}</span>
            </p>
            <p className="mt-0.5 text-[12px] font-semibold">
              {row.isWinner
                ? "Elegível · Zero atalho de lente"
                : row.eligible
                  ? "Baseline sem correção óptica"
                  : "Desqualificada (atalho de lente)"}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
