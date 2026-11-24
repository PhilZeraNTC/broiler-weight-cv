import { Loader2 } from "lucide-react";
import { useMetrics } from "../../hooks/useStudioHooks.ts";
import { InlineErrorBanner } from "../layout/InlineErrorBanner.tsx";
import { ActualVsPredictedScatter } from "./ActualVsPredictedScatter.tsx";
import { FeatureImportanceChart } from "./FeatureImportanceChart.tsx";
import { ForensicAuditCard } from "./ForensicAuditCard.tsx";
import { ModelComparisonTable } from "./ModelComparisonTable.tsx";
import { OpticalAblationChart } from "./OpticalAblationChart.tsx";
import { ExecutiveAuditKpiStrip, StratifiedErrorsPanel } from "./StratifiedErrorsPanel.tsx";

export interface MetricsAuditTabProps {
  isActive: boolean;
}

export function MetricsAuditTab({ isActive }: MetricsAuditTabProps) {
  const { data, loading, error, retry } = useMetrics(isActive);

  if (loading && data === null) {
    return (
      <section
        aria-live="polite"
        aria-busy="true"
        className="flex min-h-[360px] flex-col items-center justify-center gap-4 rounded-xl border border-border bg-surface p-6 text-center"
      >
        <Loader2 className="h-6 w-6 shrink-0 animate-spin text-accent" aria-hidden="true" />
        <p className="text-[14px] font-normal leading-[1.5] text-ink">
          Carregando relatório de validação cruzada 5-fold (1.738 imagens)...
        </p>
      </section>
    );
  }

  if (error && data === null) {
    return (
      <InlineErrorBanner
        title="Relatório de Auditoria Indisponível"
        message={error.userMessage}
        actionLabel="Tentar Análise Novamente"
        onAction={retry}
      />
    );
  }

  if (data === null) {
    return null;
  }

  const { evaluation_report, dataset_audit_summary, pdi_calibration_audit } = data;

  return (
    <div className="space-y-8">
      {error ? (
        <InlineErrorBanner
          title="Atualização do Relatório Falhou"
          message={error.userMessage}
          actionLabel="Tentar Análise Novamente"
          onAction={retry}
        />
      ) : null}

      {/* Top Strip: 5 Executive Audit KPIs + Literature R² > 0,90 Benchmark Context */}
      <ExecutiveAuditKpiStrip
        evaluationReport={evaluation_report}
        pdiAudit={pdi_calibration_audit}
      />

      {/* Row 1 Visual Proofs: 7 cols Interactive ScatterChart + 5 cols Optical Ablation BarChart */}
      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-7">
          <ActualVsPredictedScatter
            points={evaluation_report.oof_scatter_points}
            totalSessions={evaluation_report.dataset_summary.n_sessions}
          />
        </div>

        <div className="min-w-0 lg:col-span-5">
          <OpticalAblationChart opticalAblation={evaluation_report.optical_ablation} />
        </div>
      </div>

      {/* Row 2 Stratified Errors & Interpretability: 6 cols Stratified Tables + 6 cols Top-12 Feature Importances */}
      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-6">
          <StratifiedErrorsPanel winningModel={evaluation_report.winning_model} />
        </div>

        <div className="min-w-0 lg:col-span-6">
          <FeatureImportanceChart
            featureImportances={evaluation_report.winning_model.feature_importances}
            totalModelFeatures={evaluation_report.winning_model.n_features}
          />
        </div>
      </div>

      {/* Row 3 Benchmark Matrix & Forensic Data Funnel + Mendeley Reconciliation */}
      <div className="space-y-8">
        <ModelComparisonTable rows={evaluation_report.model_comparison} />
        <ForensicAuditCard datasetAudit={dataset_audit_summary} pdiAudit={pdi_calibration_audit} />
      </div>
    </div>
  );
}
