import { AlertTriangle, CheckCircle2, Clock, ShieldCheck } from "lucide-react";
import { cn } from "../../lib/cn.ts";
import { formatGrams, formatKg, formatNumberPtBr, formatPercent } from "../../lib/format.ts";
import type { PredictResponse } from "../../types/api.ts";

export interface PredictionSummaryCardProps {
  prediction: PredictResponse | null;
  loading: boolean;
}

export function PredictionSummaryCard({ prediction, loading }: PredictionSummaryCardProps) {
  if (!prediction) {
    return (
      <section
        aria-label="Estimativa de peso vivo"
        aria-busy={loading}
        className="rounded-xl border border-border bg-surface p-4 lg:p-6"
      >
        <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
          Estimativa de Peso Vivo
        </p>
        <p className="mt-2 text-[32px] font-semibold leading-[1.2] tabular-nums text-ink-muted">
          —
        </p>
        <p className="mt-2 text-[14px] font-normal leading-[1.5] text-ink-muted">
          {loading
            ? "Executando pipeline PDI clássico e regressão LightGBM..."
            : "Selecione um preset auditado ou envie uma foto para estimar o peso vivo."}
        </p>
      </section>
    );
  }

  const {
    band_error_interval: bandErr,
    age_stage: ageStage,
    timing,
    oof_audit: oofAudit,
  } = prediction;
  const signedPrefix = oofAudit && oofAudit.signed_err_g > 0 ? "+" : "";

  return (
    <section
      aria-label="Estimativa de peso vivo e auditoria OOF"
      className="space-y-4 rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      {/* Header + Latency SLA badge */}
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Estimativa de Peso Vivo · {prediction.winning_model_name}
          </p>
          <div className="flex flex-wrap items-baseline gap-2">
            <p className="text-[32px] font-semibold leading-[1.2] tabular-nums text-ink">
              {formatGrams(prediction.predicted_weight_g, 1)}
            </p>
            <span className="text-[14px] font-normal leading-[1.5] tabular-nums text-ink-muted">
              ({formatKg(prediction.predicted_weight_kg, 2)})
            </span>
          </div>
        </div>

        <div className="flex flex-col items-end gap-1">
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[12px] font-semibold leading-[1.33] tabular-nums",
              timing.within_sla
                ? "border-accent/25 bg-accent-tint text-accent"
                : "border-warning-border bg-warning-tint text-warning",
            )}
          >
            {timing.within_sla ? (
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            ) : (
              <Clock className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            )}
            <span>
              {formatNumberPtBr(timing.total_ms, 1)} ms{" "}
              {timing.within_sla ? `· < ${timing.sla_target_ms} ms` : ""}
            </span>
          </span>
          <span className="text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
            PDI {formatNumberPtBr(timing.pdi_ms, 1)} ms + ML{" "}
            {formatNumberPtBr(timing.inference_ms, 1)} ms
          </span>
        </div>
      </div>

      {/* Error interval & Cobb Avian 48 developmental age stage */}
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        <div className="rounded-lg border border-border bg-canvas p-4">
          <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
            Margem Empírica ({prediction.weight_band})
          </p>
          <p className="mt-1 text-[14px] font-semibold leading-[1.5] tabular-nums text-ink">
            ±{formatGrams(bandErr.mae_g, 1)} (MAE)
          </p>
          <p className="text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
            p68: ±{formatGrams(bandErr.p68_abs_err_g, 1)} · n = {formatNumberPtBr(bandErr.n, 0)}
          </p>
        </div>

        <div className="rounded-lg border border-border bg-canvas p-4">
          <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
            Estágio Etário (Cobb Avian 48)
          </p>
          <p className="mt-1 text-[14px] font-semibold leading-[1.5] text-ink">
            {ageStage.stage_label}
          </p>
          <p className="text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
            Idade estimada: {ageStage.estimated_age_days_range}
          </p>
        </div>
      </div>

      {/* Mandatory out-of-training-range Amber scientific disclosure */}
      {ageStage.out_of_training_range ? (
        <div
          role="note"
          className="flex items-start gap-2 rounded-lg border border-warning-border bg-warning-tint p-4 text-warning"
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          <p className="text-[14px] font-normal leading-[1.5]">
            Estimativa fora da faixa treinada (116–2.093 g · ~7–29 dias). O valor exibido é uma
            extrapolação fora do domínio auditado.
          </p>
        </div>
      ) : null}

      {/* Honest Out-of-Fold Audit section (rendered strictly for curated OOF presets) */}
      {prediction.oof_evaluation && oofAudit !== null ? (
        <div className="rounded-lg border border-border bg-surface-subtle p-4">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-2">
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 shrink-0 text-ink" aria-hidden="true" />
              <h3 className="text-[14px] font-semibold leading-[1.5] text-ink">
                Auditoria OOF (Fold {oofAudit.fold})
              </h3>
            </div>
            <span className="rounded bg-surface px-2 py-0.5 font-mono text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
              {oofAudit.session_id}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[12px] font-semibold leading-[1.33] tabular-nums">
            <div className="rounded-lg border border-border bg-surface p-2">
              <span className="block text-ink-muted">Peso Real (g)</span>
              <span className="mt-1 block text-[14px] font-semibold text-ink">
                {formatGrams(oofAudit.ground_truth_weight_g, 0)}
              </span>
            </div>

            <div className="rounded-lg border border-border bg-surface p-2">
              <span className="block text-ink-muted">Peso Estimado OOF (g)</span>
              <span className="mt-1 block text-[14px] font-semibold text-ink">
                {formatGrams(prediction.predicted_weight_g, 1)}
              </span>
            </div>

            <div className="rounded-lg border border-border bg-surface p-2">
              <span className="block text-ink-muted">Erro Absoluto (g)</span>
              <span className="mt-1 block text-[14px] font-normal text-ink">
                {formatGrams(oofAudit.abs_err_g, 1)}{" "}
                <span className="text-[12px] font-semibold text-ink-muted">
                  ({signedPrefix}
                  {formatNumberPtBr(oofAudit.signed_err_g, 1)} g)
                </span>
              </span>
            </div>

            <div className="rounded-lg border border-border bg-surface p-2">
              <span className="block text-ink-muted">Erro Percentual (%)</span>
              <span className="mt-1 block text-[14px] font-normal text-ink">
                {formatPercent(oofAudit.pct_err, 1)}
              </span>
            </div>
          </div>

          <p className="mt-2 text-[12px] font-semibold leading-[1.33] text-ink-muted">
            Avaliação honesta fora do treino usando exclusivamente o pipeline do Fold{" "}
            {oofAudit.fold}.
          </p>
        </div>
      ) : (
        <p className="rounded-lg border border-border bg-canvas px-4 py-2 text-[12px] font-semibold leading-[1.33] text-ink-muted">
          Upload externo avaliado pelo pipeline de produção (sem gabarito de balança).
        </p>
      )}
    </section>
  );
}
