import { CheckCircle2, Info } from "lucide-react";
import { useState } from "react";
import { cn } from "../../lib/cn.ts";
import {
  formatGrams,
  formatNullableMetric,
  formatNumberPtBr,
  formatPercent,
} from "../../lib/format.ts";
import type {
  EvaluationReport,
  MetricSlice,
  PdiCalibrationAudit,
  WinningModelReport,
} from "../../types/api.ts";

export interface ExecutiveAuditKpiStripProps {
  evaluationReport: EvaluationReport;
  pdiAudit: PdiCalibrationAudit;
}

/**
 * Top-line 5-Card Executive Audit KPI strip + uncontrolled handheld smartphone vs.
 * controlled fixed ceiling-camera `R² > 0,90` benchmark context bar (`04-UI-SPEC.md:149-151`).
 */
export function ExecutiveAuditKpiStrip({
  evaluationReport,
  pdiAudit,
}: ExecutiveAuditKpiStripProps) {
  const ds = evaluationReport.dataset_summary;
  const win = evaluationReport.winning_model;
  const ext = pdiAudit.dataset_extraction_summary;
  const swapShift = win.counterfactual_lens_flip.physical_lens_swap.median_shift_pct;
  const focalShift = win.counterfactual_lens_flip.focal_only.median_shift_pct;
  const maxShift = Math.max(swapShift, focalShift);

  return (
    <section aria-label="Resumo executivo da validação cruzada 5-fold" className="space-y-4">
      <div className="space-y-1">
        <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
          Validação Cruzada 5-Fold Agrupada · Auditoria Científica &amp; Ablação Óptica
        </p>
        <h1 className="text-[20px] font-semibold leading-[1.2] text-ink">
          Auditoria &amp; Métricas — AvianScale CV
        </h1>
      </div>

      {/* 5-Card Executive KPI Strip */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {/* KPI 1: Clean dataset counts */}
        <div className="rounded-xl border border-border bg-surface p-4">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Base Limpa Auditada
          </p>
          <p className="mt-2 text-[32px] font-semibold leading-[1.2] tabular-nums text-ink">
            {formatNumberPtBr(ds.n_clean_images, 0)}
          </p>
          <p className="mt-1 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
            {formatNumberPtBr(ds.n_sessions, 0)} sessões ·{" "}
            {formatNumberPtBr(ds.n_unique_weights, 0)} pesos ({formatNumberPtBr(ds.weight_min_g, 0)}
            –{formatNumberPtBr(ds.weight_max_g, 0)} g)
          </p>
        </div>

        {/* KPI 2: Cross-fold session leakage */}
        <div className="rounded-xl border border-border bg-surface p-4">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Split Anti-Leakage
          </p>
          <p className="mt-2 text-[32px] font-semibold leading-[1.2] tabular-nums text-ink">
            {formatNumberPtBr(ds.cross_fold_session_overlap, 0)}
          </p>
          <div className="mt-1">
            {ds.cross_fold_session_overlap === 0 ? (
              <span className="inline-flex items-center gap-1 rounded-full border border-accent/30 bg-accent-tint px-2 py-0.5 text-[12px] font-semibold leading-[1.33] text-accent">
                <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span>0 Sobreposição Entre Folds</span>
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 rounded-full border border-warning-border bg-warning-tint px-2 py-0.5 text-[12px] font-semibold leading-[1.33] text-warning">
                <span>
                  {formatNumberPtBr(ds.cross_fold_session_overlap, 0)} Sobreposição Entre Folds
                </span>
              </span>
            )}
          </div>
        </div>

        {/* KPI 3: Winning model MAE & Session R2 */}
        <div className="rounded-xl border border-border bg-surface p-4">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            {win.model_name} C2 (OOF)
          </p>
          <p className="mt-2 text-[32px] font-semibold leading-[1.2] tabular-nums text-ink">
            {formatGrams(win.global.mae_g, 2)}
          </p>
          <p className="mt-1 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
            R² Sessão {formatNullableMetric(win.session_level.session_averaged_r2, 4)} · BalMAPE{" "}
            {formatPercent(win.balanced_band_mape_pct, 2)}
          </p>
        </div>

        {/* KPI 4: Dual-Lens Counterfactual Shift */}
        <div className="rounded-xl border border-border bg-surface p-4">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Shift Dual-Lens
          </p>
          <p className="mt-2 text-[32px] font-semibold leading-[1.2] tabular-nums text-ink">
            {formatPercent(maxShift, 1)}
          </p>
          <div className="mt-1">
            <span
              className={cn(
                "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[12px] font-semibold leading-[1.33] tabular-nums",
                maxShift < 0.05
                  ? "border-accent/30 bg-accent-tint text-accent"
                  : "border-warning-border bg-warning-tint text-warning",
              )}
            >
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
              <span>Vencedor C2 · {formatPercent(maxShift, 1)} Shift</span>
            </span>
          </div>
        </div>

        {/* KPI 5: PDI Fallback & Bucket LCD Safety */}
        <div className="rounded-xl border border-border bg-surface p-4">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            PDI Fallback Clássico
          </p>
          <p className="mt-2 text-[32px] font-semibold leading-[1.2] tabular-nums text-ink">
            {formatPercent(ext.primary_fallback_pct, 2)}
          </p>
          <p className="mt-1 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
            {formatNumberPtBr(ext.primary_fallback_count, 0)}/
            {formatNumberPtBr(ext.n_clean_images, 0)} ·{" "}
            {formatNumberPtBr(ext.lcd_suppression_bucket_false_positives, 0)}/
            {formatNumberPtBr(ext.n_bucket_dorsal, 0)} LCD balde
          </p>
        </div>
      </div>

      {/* Literature R² > 0,90 Benchmark Context Bar */}
      <div
        role="note"
        className="flex items-start gap-4 rounded-xl border border-border bg-surface-subtle p-4 text-ink"
      >
        <Info className="mt-0.5 h-5 w-5 shrink-0 text-ink-muted" aria-hidden="true" />
        <p className="text-[14px] font-normal leading-[1.5] text-ink">
          <span className="font-semibold">
            Contexto Científico (Smartphone Handheld vs. Literatura R² &gt;{" "}
            {formatNumberPtBr(evaluationReport.literature_benchmark.r2_reference, 2)}):
          </span>{" "}
          Trabalhos da literatura que atingem{" "}
          <span className="font-semibold tabular-nums">
            R² &gt; {formatNumberPtBr(evaluationReport.literature_benchmark.r2_reference, 2)}
          </span>{" "}
          operam com câmera fixa perpendicular no teto a altura constante (
          <span className="font-mono text-[12px]">Z_const</span>). Na base pública auditada (captura
          manual em smartphone com distância câmera-ave variável e inclinação livre), o pipeline
          clássico calibrado por EXIF alcança{" "}
          <span className="font-semibold tabular-nums">
            R² = {formatNullableMetric(win.global.r2, 4)}
          </span>{" "}
          por quadro isolado e{" "}
          <span className="font-semibold tabular-nums">
            R² = {formatNullableMetric(win.session_level.session_averaged_r2, 4)}
          </span>{" "}
          na média por sessão ({formatNumberPtBr(ds.n_sessions, 0)} sessões) — a transição para
          câmera fixa de teto está documentada na aba{" "}
          <span className="font-semibold">Visão Futura (v2)</span>.
        </p>
      </div>
    </section>
  );
}

export interface StratifiedErrorsPanelProps {
  winningModel: WinningModelReport;
}

type StratificationDimension = "band" | "lens" | "scene";

interface StratifiedRow {
  key: string;
  label: string;
  slice: MetricSlice;
}

export function StratifiedErrorsPanel({ winningModel }: StratifiedErrorsPanelProps) {
  const [dimension, setDimension] = useState<StratificationDimension>("band");

  const rows: StratifiedRow[] =
    dimension === "band"
      ? [
          { key: "<500g", label: "Leve (<500 g)", slice: winningModel.by_weight_band["<500g"] },
          {
            key: "500-1000g",
            label: "Intermediário (500–1000 g)",
            slice: winningModel.by_weight_band["500-1000g"],
          },
          {
            key: ">1000g",
            label: "Pesado (>1000 g)",
            slice: winningModel.by_weight_band[">1000g"],
          },
        ]
      : dimension === "lens"
        ? [
            { key: "wide", label: "Lente Wide (4,25 mm)", slice: winningModel.by_lens.wide },
            {
              key: "ultrawide",
              label: "Lente Ultrawide (1,65 mm)",
              slice: winningModel.by_lens.ultrawide,
            },
          ]
        : [
            {
              key: "bucket_dorsal",
              label: "Balde Dorsal (Fundo Branco)",
              slice: winningModel.by_scene.bucket_dorsal,
            },
            {
              key: "scale_or_litter",
              label: "Balança SF-400 / Cama de Palha",
              slice: winningModel.by_scene.scale_or_litter,
            },
          ];

  return (
    <section
      aria-label="Erros estratificados Out-of-Fold por faixa de peso, lente e cenário"
      className="flex flex-col gap-4 rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Diagnóstico por Subgrupo · {winningModel.model_name} C2
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            Erros Estratificados OOF
          </h2>
        </div>

        <fieldset className="inline-flex rounded-lg border border-border bg-surface-subtle p-1">
          <legend className="sr-only">Dimensão de estratificação dos erros OOF</legend>
          {(
            [
              { id: "band", label: "Faixa de Peso" },
              { id: "lens", label: "Lente" },
              { id: "scene", label: "Cenário" },
            ] as const
          ).map((tab) => {
            const active = dimension === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                aria-pressed={active}
                onClick={() => setDimension(tab.id)}
                className={cn(
                  "min-h-[44px] rounded-md px-4 text-[12px] font-semibold leading-[1.33] transition-colors",
                  active ? "bg-surface text-ink shadow-xs" : "text-ink-muted hover:text-ink",
                )}
              >
                {tab.label}
              </button>
            );
          })}
        </fieldset>
      </div>

      {/* Stratified error table inside overflow-x-auto */}
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full min-w-[540px] border-collapse text-left">
          <thead>
            <tr className="border-b border-border bg-surface-subtle text-[12px] font-semibold leading-[1.33] text-ink-muted">
              <th scope="col" className="px-4 py-2">
                Estrato
              </th>
              <th scope="col" className="px-4 py-2 text-right">
                N
              </th>
              <th scope="col" className="px-4 py-2 text-right">
                MAE (g)
              </th>
              <th scope="col" className="px-4 py-2 text-right">
                RMSE (g)
              </th>
              <th scope="col" className="px-4 py-2 text-right">
                MAPE (%)
              </th>
              <th scope="col" className="px-4 py-2 text-right">
                MBE (g)
              </th>
              <th scope="col" className="px-4 py-2 text-right">
                R²
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border text-[14px] font-normal leading-[1.5] text-ink">
            {rows.map(({ key, label, slice }) => (
              <tr key={key} className="hover:bg-canvas">
                <td className="px-4 py-2 font-semibold text-ink">{label}</td>
                <td className="px-4 py-2 text-right tabular-nums">
                  {formatNumberPtBr(slice.n, 0)}
                </td>
                <td className="px-4 py-2 text-right font-semibold tabular-nums">
                  {formatNumberPtBr(slice.mae_g, 1)}
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-ink-muted">
                  {formatNumberPtBr(slice.rmse_g, 1)}
                </td>
                <td className="px-4 py-2 text-right tabular-nums">
                  {formatPercent(slice.mape_pct, 1)}
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-ink-muted">
                  {slice.mbe_g > 0 ? "+" : ""}
                  {formatNumberPtBr(slice.mbe_g, 1)}
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-ink-muted">
                  {formatNullableMetric(slice.r2, 4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-fold consistency mini-summary */}
      <div className="space-y-2 rounded-lg border border-border bg-canvas p-4">
        <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
          Consistência por Fold (5-Fold StratifiedGroupKFold por session_id)
        </p>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[460px] border-collapse text-left">
            <thead>
              <tr className="border-b border-border text-[12px] font-semibold leading-[1.33] text-ink-muted">
                <th scope="col" className="py-1 pr-4">
                  Fold
                </th>
                <th scope="col" className="px-2 py-1 text-right">
                  Imgs / Sess.
                </th>
                <th scope="col" className="px-2 py-1 text-right">
                  MAE (g)
                </th>
                <th scope="col" className="px-2 py-1 text-right">
                  MAPE (%)
                </th>
                <th scope="col" className="py-1 pl-2 text-right">
                  R² Global
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border text-[12px] font-semibold leading-[1.33] tabular-nums text-ink">
              {winningModel.per_fold.map((foldRow) => (
                <tr key={foldRow.fold}>
                  <td className="py-1 pr-4 font-mono">Fold {foldRow.fold}</td>
                  <td className="px-2 py-1 text-right text-ink-muted">
                    {formatNumberPtBr(foldRow.n, 0)} / {formatNumberPtBr(foldRow.n_sessions, 0)}
                  </td>
                  <td className="px-2 py-1 text-right">{formatNumberPtBr(foldRow.mae_g, 1)}</td>
                  <td className="px-2 py-1 text-right">{formatPercent(foldRow.mape_pct, 1)}</td>
                  <td className="py-1 pl-2 text-right text-ink-muted">
                    {formatNullableMetric(foldRow.r2, 4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
