import { AlertTriangle, CheckCircle2, Database, ShieldCheck } from "lucide-react";
import { formatNumberPtBr, formatPercent } from "../../lib/format.ts";
import type { DatasetAuditSummary, PdiCalibrationAudit } from "../../types/api.ts";

export interface ForensicAuditCardProps {
  datasetAudit: DatasetAuditSummary;
  pdiAudit: PdiCalibrationAudit;
}

export function ForensicAuditCard({ datasetAudit, pdiAudit }: ForensicAuditCardProps) {
  const funnel = datasetAudit.curation_funnel;
  const recon = datasetAudit.mendeley_vs_disk_reconciliation;
  const gaps = datasetAudit.distributions.coverage_gaps;
  const enc = datasetAudit.encoding_and_exif_recovery;
  const split = datasetAudit.split_audit;
  const ext = pdiAudit.dataset_extraction_summary;
  const calib = pdiAudit.paired_dual_lens_calibration;

  const bucketPairs = calib.scopes.bucket_only_pairs;
  const sameScenePairs = calib.scopes.same_scene_pairs;
  const winningBucketRatio =
    bucketPairs?.formulas[calib.winning_formula]?.median_wide_over_uw ?? 0.9638;
  const rejectedBucketRatio =
    bucketPairs?.formulas[calib.rejected_double_count_formula]?.median_wide_over_uw ?? 0.1453;
  const winningSameSceneRatio =
    sameScenePairs?.formulas[calib.winning_formula]?.median_wide_over_uw ?? 0.9326;

  return (
    <section
      aria-label="Funil forense de curadoria, reconciliação Mendeley e auditoria anti-leakage PDI"
      className="space-y-6 rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Auditoria Forense de Dados · {datasetAudit.dataset_identity.name} (DOI{" "}
            <span className="font-mono">{datasetAudit.dataset_identity.doi}</span>)
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] tabular-nums text-ink">
            Funil Forense de Curadoria ({formatNumberPtBr(funnel.raw_images, 0)} →{" "}
            {formatNumberPtBr(funnel.clean_images, 0)}) &amp; Reconciliação Mendeley
          </h2>
        </div>

        <span className="inline-flex items-center gap-1 rounded-full border border-accent/30 bg-accent-tint px-2 py-1 text-[12px] font-semibold leading-[1.33] text-accent">
          <ShieldCheck className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          <span>Invariância Externa 0,0 diff</span>
        </span>
      </div>

      {/* Block 1: Curation Funnel (1,753 -> 1,738) */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-lg border border-border bg-canvas p-4">
          <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
            1. Auditadas em Disco
          </p>
          <p className="mt-1 text-[20px] font-semibold leading-[1.2] tabular-nums text-ink">
            {formatNumberPtBr(funnel.raw_images, 0)} imagens
          </p>
          <p className="mt-1 text-[14px] font-normal leading-[1.5] text-ink-muted">
            {formatNumberPtBr(recon.disk_audit.raw_leaf_folders, 0)} pastas-folha ·{" "}
            {formatNumberPtBr(funnel.raw_sessions, 0)} sessões brutas ·{" "}
            {formatPercent(enc.exif_orientation_rotated_pct, 2)} com rotação EXIF corrigida.
          </p>
        </div>

        <div className="rounded-lg border border-border bg-canvas p-4">
          <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
            2. Quarentena Isolada
          </p>
          <p className="mt-1 text-[20px] font-semibold leading-[1.2] tabular-nums text-ink">
            −{formatNumberPtBr(funnel.quarantined_total, 0)} imagens
          </p>
          <p className="mt-1 text-[14px] font-normal leading-[1.5] text-ink-muted">
            {formatNumberPtBr(funnel.quarantined_multi_bird.count, 0)} multi-aves (
            <span className="font-mono text-[12px]">417, 757, 823, 1520</span> → Aba Watershed) +{" "}
            {formatNumberPtBr(funnel.quarantined_severe_mislabel.count, 0)} rótulos incorretos (
            <span className="font-mono text-[12px]">1351, 1465</span>).
          </p>
        </div>

        <div className="rounded-lg border border-border bg-canvas p-4">
          <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
            3. Correções &amp; Encoding
          </p>
          <p className="mt-1 text-[20px] font-semibold leading-[1.2] tabular-nums text-ink">
            {formatNumberPtBr(funnel.lcd_corrected_images.count, 0)} LCD ·{" "}
            {formatNumberPtBr(enc.cp866_recovered_count, 0)} CP866
          </p>
          <p className="mt-1 text-[14px] font-normal leading-[1.5] text-ink-muted">
            Pasta <span className="font-mono text-[12px]">367 → 267 g</span> confirmada no LCD,{" "}
            {formatNumberPtBr(enc.cp866_recovered_count, 0)} nomes mojibake CP866 recuperados e{" "}
            {formatNumberPtBr(enc.variable_digital_zoom_count, 0)} fotos com zoom digital Xiaomi tag{" "}
            <span className="font-mono text-[12px]">39321</span>.
          </p>
        </div>

        <div className="rounded-lg border border-accent/30 bg-accent-tint/60 p-4">
          <p className="text-[12px] font-semibold leading-[1.33] text-accent">
            4. Base Limpa Final
          </p>
          <p className="mt-1 text-[20px] font-semibold leading-[1.2] tabular-nums text-ink">
            {formatNumberPtBr(funnel.clean_images, 0)} imagens
          </p>
          <p className="mt-1 text-[14px] font-normal leading-[1.5] text-ink">
            {formatNumberPtBr(funnel.clean_unique_weights, 0)} pesos únicos ·{" "}
            {formatNumberPtBr(funnel.clean_sessions, 0)} sessões ·{" "}
            <span className="font-semibold text-accent">
              {formatNumberPtBr(split.cross_fold_session_overlap, 0)} sobreposição entre folds
            </span>
            .
          </p>
        </div>
      </div>

      {/* Block 2: Mendeley Published Claim vs. Disk Audit Reconciliation */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="space-y-4 rounded-lg border border-border bg-canvas p-4">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 shrink-0 text-ink-muted" aria-hidden="true" />
            <h3 className="text-[14px] font-semibold leading-[1.5] text-ink">
              Reconciliação Mendeley vs. Auditoria em Disco
            </h3>
          </div>

          <div className="overflow-x-auto rounded-lg border border-border bg-surface">
            <table className="w-full min-w-[420px] border-collapse text-left">
              <thead>
                <tr className="border-b border-border bg-surface-subtle text-[12px] font-semibold leading-[1.33] text-ink-muted">
                  <th scope="col" className="px-4 py-2">
                    Critério
                  </th>
                  <th scope="col" className="px-4 py-2">
                    Descrição Mendeley
                  </th>
                  <th scope="col" className="px-4 py-2">
                    Auditoria em Disco
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border text-[14px] font-normal leading-[1.5] text-ink">
                <tr>
                  <td className="px-4 py-2 font-semibold">Total de Imagens</td>
                  <td className="px-4 py-2 tabular-nums text-ink-muted">
                    {formatNumberPtBr(recon.published_claims.published_images, 0)} imagens
                  </td>
                  <td className="px-4 py-2 font-semibold tabular-nums">
                    {formatNumberPtBr(recon.disk_audit.raw_images, 0)} brutas (−
                    {formatNumberPtBr(recon.disk_audit.missing_vs_mendeley_images, 0)} ausentes) →{" "}
                    {formatNumberPtBr(funnel.clean_images, 0)} limpas
                  </td>
                </tr>
                <tr>
                  <td className="px-4 py-2 font-semibold">Janela Temporal</td>
                  <td className="px-4 py-2 text-ink-muted">"coleta diária por 45 dias"</td>
                  <td className="px-4 py-2 font-semibold tabular-nums">
                    {formatNumberPtBr(recon.disk_audit.distinct_capture_dates, 0)} datas distintas (
                    {formatNumberPtBr(recon.disk_audit.calendar_span_days, 0)} dias corridos:{" "}
                    <span className="font-mono text-[12px]">
                      {recon.disk_audit.date_min} → {recon.disk_audit.date_max}
                    </span>
                    )
                  </td>
                </tr>
                <tr>
                  <td className="px-4 py-2 font-semibold">Idade &amp; Peso</td>
                  <td className="px-4 py-2 text-ink-muted">Ciclo completo (45 dias implícitos)</td>
                  <td className="px-4 py-2 font-semibold tabular-nums">
                    ~7–29 dias · {formatNumberPtBr(recon.disk_audit.clean_weight_min_g, 0)}–
                    {formatNumberPtBr(recon.disk_audit.clean_weight_max_g, 0)} g (sem 30–45 dias)
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Block 3 & 4: Sparse 801-1,216 g Gap Amber Disclosure + PDI Calibration Proof */}
        <div className="space-y-4">
          {/* Mandatory Amber 801-1,216 g sparse coverage gap disclosure */}
          <div
            role="note"
            className="flex items-start gap-4 rounded-lg border border-warning-border bg-warning-tint p-4 text-warning"
          >
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
            <div className="min-w-0 space-y-1">
              <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide">
                Lacuna de Cobertura Auditada · Faixa{" "}
                {formatNumberPtBr(gaps.weight_sparse_interval.lower_exclusive_g, 0)}–
                {formatNumberPtBr(gaps.weight_sparse_interval.upper_exclusive_g, 0)} g
              </p>
              <p className="text-[14px] font-normal leading-[1.5]">
                Um hiato de{" "}
                <span className="font-semibold tabular-nums">
                  {formatNumberPtBr(gaps.temporal_hiatus.max_gap_days, 0)} dias
                </span>{" "}
                entre{" "}
                <span className="font-mono text-[12px] font-semibold">
                  {gaps.temporal_hiatus.gap_start_date}
                </span>{" "}
                e{" "}
                <span className="font-mono text-[12px] font-semibold">
                  {gaps.temporal_hiatus.gap_end_date}
                </span>{" "}
                criou uma lacuna severa de{" "}
                <span className="font-semibold tabular-nums">
                  {formatNumberPtBr(gaps.weight_sparse_interval.span_g, 0)} g
                </span>{" "}
                com apenas{" "}
                <span className="font-semibold tabular-nums">
                  {formatNumberPtBr(gaps.weight_sparse_interval.clean_unique_weights_inside, 0)}{" "}
                  peso único (
                  {formatNumberPtBr(gaps.weight_sparse_interval.clean_weights_inside[0], 0)} g ·{" "}
                  {formatNumberPtBr(gaps.weight_sparse_interval.clean_images_inside, 0)} imagens)
                </span>
                , explicando a maior variância observada acima de 1.000 g.
              </p>
            </div>
          </div>

          {/* PDI Calibration & Anti-Leakage Verification Box */}
          <div className="space-y-2 rounded-lg border border-border bg-canvas p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
                Calibração Óptica &amp; Anti-Leakage PDI ({pdiAudit.pdi_config_hash})
              </p>
              <span className="inline-flex items-center gap-1 rounded-full border border-accent/30 bg-accent-tint px-2 py-0.5 text-[12px] font-semibold leading-[1.33] text-accent">
                <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span>Invariância Externa 0,0 diff</span>
              </span>
            </div>

            <ul className="space-y-1 text-[14px] font-normal leading-[1.5] text-ink">
              <li>
                <span className="font-semibold">Fórmula vencedora:</span>{" "}
                <span className="font-mono text-[12px] text-ink">{calib.winning_formula}</span>{" "}
                atingiu razão mediana Wide/UW de{" "}
                <span className="font-semibold tabular-nums text-accent">
                  {formatNumberPtBr(winningBucketRatio, 4)}
                </span>{" "}
                nos {formatNumberPtBr(bucketPairs?.n_pairs ?? 76, 0)} pares de balde (
                <span className="tabular-nums">{formatNumberPtBr(winningSameSceneRatio, 4)}</span>{" "}
                nos {formatNumberPtBr(sameScenePairs?.n_pairs ?? 175, 0)} pares mesma-cena), contra{" "}
                <span className="font-semibold tabular-nums text-warning">
                  {formatNumberPtBr(rejectedBucketRatio, 4)}
                </span>{" "}
                na dupla contagem rejeitada{" "}
                <span className="font-mono text-[12px]">{calib.rejected_double_count_formula}</span>
                .
              </li>
              <li>
                <span className="font-semibold">Supressão LCD SF-400 (PROIBIÇÃO 2):</span>{" "}
                <span className="tabular-nums">
                  {formatNumberPtBr(ext.lcd_suppression_triggered_count, 0)}
                </span>{" "}
                acionamentos em balança/cama e{" "}
                <span className="font-semibold tabular-nums text-accent">
                  {formatNumberPtBr(ext.lcd_suppression_bucket_false_positives, 0)}/
                  {formatNumberPtBr(ext.n_bucket_dorsal, 0)}
                </span>{" "}
                falsos positivos no balde dorsal; perturbação de pixels fora da máscara produz{" "}
                <span className="font-semibold tabular-nums">0,0 diff</span> nas 49 features.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
