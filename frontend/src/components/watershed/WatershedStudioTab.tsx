import { AlertTriangle, CheckCircle2, Clock } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAbortableRequest } from "../../hooks/useStudioHooks.ts";
import { type ApiError, runWatershedDemo } from "../../lib/api.ts";
import { cn } from "../../lib/cn.ts";
import { formatGrams, formatNumberPtBr, formatPercent, pluralizePtBr } from "../../lib/format.ts";
import type {
  OpticalLensClass,
  SourceType,
  WatershedPresetItem,
  WatershedResponse,
} from "../../types/api.ts";
import { InlineErrorBanner } from "../layout/InlineErrorBanner.tsx";
import { StageLightbox } from "../layout/StageLightbox.tsx";
import { WatershedInstanceTable } from "./WatershedInstanceTable.tsx";
import { WatershedPresetStrip } from "./WatershedPresetStrip.tsx";
import {
  WATERSHED_SLIDER_DEFAULTS,
  type WatershedOpticalOverride,
  WatershedSlidersCard,
  type WatershedSliderValues,
} from "./WatershedSlidersCard.tsx";
import { buildWatershedLightboxStages, WatershedStageGrid } from "./WatershedStageGrid.tsx";

const DEFAULT_WATERSHED_PRESET_ID = "ws_417";

/** Verbatim permanent scientific disclaimer (`04-UI-SPEC.md:247`, `ground_truth_scope = "group_sum_only"`). */
const WATERSHED_GROUP_SUM_DISCLAIMER =
  "Pesagem de grupo na base pública: apenas o peso total conjunto das aves foi registrado; estimativas individuais por ave obtidas via Watershed são ilustrativas e não possuem gabarito individual.";

export interface WatershedStudioTabProps {
  isActive: boolean;
  presets: readonly WatershedPresetItem[];
  presetsLoading: boolean;
  onLatency?: (totalMs: number) => void;
}

interface ActiveWatershedSource {
  sourceType: SourceType;
  presetId: string | null;
  file: File | null;
  fileName: string | null;
}

function formatLensLabel(lensClass: OpticalLensClass): string {
  if (lensClass === "wide") return "Wide";
  if (lensClass === "ultrawide") return "Ultrawide";
  return "Desconhecida";
}

function formatSignedGrams(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatGrams(value, 1)}`;
}

export function WatershedStudioTab({
  isActive,
  presets,
  presetsLoading,
  onLatency,
}: WatershedStudioTabProps) {
  const { data, loading, error, run, clearError } = useAbortableRequest<WatershedResponse>();
  const [clientError, setClientError] = useState<ApiError | null>(null);
  const [source, setSource] = useState<ActiveWatershedSource>({
    sourceType: "preset",
    presetId: DEFAULT_WATERSHED_PRESET_ID,
    file: null,
    fileName: null,
  });
  const [sliders, setSliders] = useState<WatershedSliderValues>({ ...WATERSHED_SLIDER_DEFAULTS });
  const [opticalOverride, setOpticalOverride] = useState<WatershedOpticalOverride | null>(null);
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  const hasBootedRef = useRef<boolean>(false);

  const executeWatershed = useCallback(
    async (
      targetSource: ActiveWatershedSource,
      targetSliders: WatershedSliderValues,
      targetOptical: WatershedOpticalOverride | null,
    ) => {
      setClientError(null);
      const response = await run((signal) =>
        runWatershedDemo(
          {
            presetId:
              targetSource.sourceType === "preset"
                ? (targetSource.presetId ?? undefined)
                : undefined,
            file:
              targetSource.sourceType === "upload" ? (targetSource.file ?? undefined) : undefined,
            hRel: targetSliders.h_rel,
            minDistanceRel: targetSliders.min_distance_rel,
            minAreaRatio: targetSliders.min_area_ratio,
            focalLengthMm: targetOptical?.focalLengthMm,
            zoomMultiple: targetOptical?.zoomMultiple,
          },
          signal,
        ),
      );
      if (response) {
        onLatency?.(response.timing.total_ms);
      }
    },
    [run, onLatency],
  );

  // Auto-load ws_417 once on first activation of Tab 2 (cached in state thereafter)
  useEffect(() => {
    if (!isActive || hasBootedRef.current) {
      return;
    }
    hasBootedRef.current = true;
    const bootSource: ActiveWatershedSource = {
      sourceType: "preset",
      presetId: DEFAULT_WATERSHED_PRESET_ID,
      file: null,
      fileName: null,
    };
    const defaultSliders: WatershedSliderValues = { ...WATERSHED_SLIDER_DEFAULTS };
    setSource(bootSource);
    setSliders(defaultSliders);
    setOpticalOverride(null);
    void executeWatershed(bootSource, defaultSliders, null);
  }, [isActive, executeWatershed]);

  const handleSelectPreset = useCallback(
    (presetId: string) => {
      const nextSource: ActiveWatershedSource = {
        sourceType: "preset",
        presetId,
        file: null,
        fileName: null,
      };
      const defaultSliders: WatershedSliderValues = { ...WATERSHED_SLIDER_DEFAULTS };
      setSource(nextSource);
      setSliders(defaultSliders);
      setOpticalOverride(null);
      void executeWatershed(nextSource, defaultSliders, null);
    },
    [executeWatershed],
  );

  const handleSelectFile = useCallback(
    (file: File) => {
      const nextSource: ActiveWatershedSource = {
        sourceType: "upload",
        presetId: null,
        file,
        fileName: file.name,
      };
      const defaultSliders: WatershedSliderValues = { ...WATERSHED_SLIDER_DEFAULTS };
      setSource(nextSource);
      setSliders(defaultSliders);
      setOpticalOverride(null);
      void executeWatershed(nextSource, defaultSliders, null);
    },
    [executeWatershed],
  );

  const handleCommitControls = useCallback(
    (nextSliders: WatershedSliderValues, nextOptical: WatershedOpticalOverride | null) => {
      setSliders(nextSliders);
      setOpticalOverride(nextOptical);
      void executeWatershed(source, nextSliders, nextOptical);
    },
    [executeWatershed, source],
  );

  const handleResetDefaults = useCallback(() => {
    const defaultSliders: WatershedSliderValues = { ...WATERSHED_SLIDER_DEFAULTS };
    setSliders(defaultSliders);
    setOpticalOverride(null);
    void executeWatershed(source, defaultSliders, null);
  }, [executeWatershed, source]);

  const handleRetry = useCallback(() => {
    void executeWatershed(source, sliders, opticalOverride);
  }, [executeWatershed, source, sliders, opticalOverride]);

  const lightboxStages = useMemo(
    () => buildWatershedLightboxStages(data?.visuals ?? null),
    [data?.visuals],
  );

  const activeError = clientError ?? error;
  const presetRef = data?.preset_reference ?? null;
  const hasActiveSource =
    (source.sourceType === "preset" && source.presetId !== null) ||
    (source.sourceType === "upload" && source.file !== null);

  const errorActionLabel =
    activeError?.status === 422
      ? "Restaurar Parâmetros Padrão"
      : activeError?.status === 400 || activeError?.status === 413 || activeError?.status === 415
        ? "Enviar Foto de Grupo"
        : "Tentar Análise Novamente";

  const handleErrorAction = () => {
    if (activeError?.status === 422) {
      handleResetDefaults();
      return;
    }
    if (activeError?.status === 400 || activeError?.status === 413 || activeError?.status === 415) {
      setClientError(null);
      clearError();
      return;
    }
    handleRetry();
  };

  return (
    <div className="space-y-6">
      {/* 1. Source strip: 4 curated multi-bird presets + custom group upload */}
      <WatershedPresetStrip
        presets={presets}
        loadingCatalog={presetsLoading}
        activeSourceType={source.sourceType}
        activePresetId={source.presetId}
        activeFileName={source.fileName}
        loading={loading}
        onSelectPreset={handleSelectPreset}
        onSelectFile={handleSelectFile}
        onValidationError={(err) => setClientError(err)}
      />

      {/* 2. Permanent Amber scientific disclaimer banner (ground_truth_scope = group_sum_only) */}
      <div
        role="note"
        aria-label="Aviso científico sobre escopo de gabarito em grupo"
        className="flex items-start gap-4 rounded-xl border border-warning-border bg-warning-tint p-4 text-warning"
      >
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
        <div className="min-w-0 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide">
              Escopo Científico Obrigatório
            </span>
            <span className="rounded-full border border-warning-border bg-surface px-2 py-0.5 font-mono text-[12px] font-semibold leading-[1.33] text-warning">
              ground_truth_scope: group_sum_only
            </span>
          </div>
          <p className="text-[14px] font-normal leading-[1.5]">{WATERSHED_GROUP_SUM_DISCLAIMER}</p>
        </div>
      </div>

      {/* Inline error banner when API/validation fails */}
      {activeError ? (
        <InlineErrorBanner
          title="Falha na Separação Watershed"
          message={activeError.userMessage}
          actionLabel={errorActionLabel}
          onAction={handleErrorAction}
          onDismiss={() => {
            setClientError(null);
            clearError();
          }}
        />
      ) : null}

      {/* 3. Aggregate KPI strip (3 cards, 24px gutter) */}
      <section
        aria-label="Indicadores agregados do Watershed"
        className="grid grid-cols-1 gap-6 md:grid-cols-3"
      >
        {/* Card 1: Aves Detectadas */}
        <div className="rounded-xl border border-border bg-surface p-6">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Aves Detectadas
          </p>
          <p className="mt-2 text-[32px] font-semibold leading-[1.2] tabular-nums text-ink">
            {data ? formatNumberPtBr(data.bird_count, 0) : "—"}
          </p>
          <p className="mt-2 text-[14px] font-normal leading-[1.5] tabular-nums text-ink-muted">
            {presetRef
              ? `Esperadas na cena: ${pluralizePtBr(presetRef.expected_bird_count, "ave", "aves")}`
              : data
                ? "Upload customizado (sem contagem esperada)"
                : "Aguardando processamento"}
          </p>
        </div>

        {/* Card 2: Soma Estimada */}
        <div className="rounded-xl border border-border bg-surface p-6">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Soma Estimada
          </p>
          <p className="mt-2 text-[32px] font-semibold leading-[1.2] tabular-nums text-ink">
            {data ? formatGrams(data.estimated_total_weight_g, 1) : "—"}
          </p>
          <p className="mt-2 text-[14px] font-normal leading-[1.5] tabular-nums text-ink-muted">
            {presetRef
              ? `Soma do Grupo: ${formatGrams(presetRef.group_sum_weight_g, 0)} (${formatSignedGrams(presetRef.sum_diff_g)} · ${formatPercent(presetRef.sum_pct_err, 1)})`
              : data
                ? "Upload customizado (sem peso de grupo registrado)"
                : "Aguardando processamento"}
          </p>
        </div>

        {/* Card 3: Peso Médio Ilustrativo */}
        <div className="rounded-xl border border-border bg-surface p-6">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Peso Médio Ilustrativo
          </p>
          <p className="mt-2 text-[32px] font-semibold leading-[1.2] tabular-nums text-ink">
            {data
              ? data.bird_count === 0
                ? "—"
                : formatGrams(data.estimated_mean_weight_g, 1)
              : "—"}
          </p>
          <p className="mt-2 text-[14px] font-normal leading-[1.5] tabular-nums text-ink-muted">
            {presetRef
              ? `Referência média do grupo: ${formatGrams(presetRef.mean_bird_weight_reference_g, 1)}`
              : data
                ? data.bird_count === 0
                  ? "Nenhuma bacia ativa para cálculo de média"
                  : "Estimativa ilustrativa sem gabarito individual"
                : "Aguardando processamento"}
          </p>
        </div>
      </section>

      {/* Compact runtime telemetry bar (red_feeder_excluded_px, optical context, latency SLA) */}
      {data ? (
        <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-border bg-surface-subtle px-4 py-2 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
          <div className="flex flex-wrap items-center gap-4">
            <span>
              Comedouro Vermelho Excluído:{" "}
              <span className="text-ink">
                {formatNumberPtBr(data.red_feeder_excluded_px, 0)} px
              </span>
            </span>
            <span>
              Óptica EXIF:{" "}
              <span className="text-ink">
                {formatLensLabel(data.optical.lens_class)} ·{" "}
                {formatNumberPtBr(data.optical.focal_length_mm, 2)} mm ·{" "}
                {formatNumberPtBr(data.optical.zoom_multiple, 2)}×
              </span>
            </span>
            <span className="font-mono">PDIConfig: {data.pdi_config_hash}</span>
          </div>

          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5",
              data.timing.within_sla
                ? "border-accent/30 bg-accent-tint text-accent"
                : "border-warning-border bg-warning-tint text-warning",
            )}
          >
            {data.timing.within_sla ? (
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            ) : (
              <Clock className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            )}
            <span>
              {formatNumberPtBr(data.timing.total_ms, 1)} ms (PDI{" "}
              {formatNumberPtBr(data.timing.pdi_ms, 1)} + ML{" "}
              {formatNumberPtBr(data.timing.inference_ms, 1)})
            </span>
          </span>
        </div>
      ) : null}

      {/* 4. Primary Split Workspace: 8 cols Watershed 2x2 viewport + 4 cols sliders panel */}
      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-8">
          <WatershedStageGrid
            visuals={data?.visuals ?? null}
            loading={loading}
            onOpenStage={(idx) => setLightboxIndex(idx)}
          />
        </div>

        <div className="min-w-0 lg:col-span-4">
          <WatershedSlidersCard
            values={sliders}
            optical={data?.optical ?? null}
            opticalOverride={opticalOverride}
            loading={loading}
            hasActiveSource={hasActiveSource}
            onChangeDraft={setSliders}
            onCommit={handleCommitControls}
            onResetDefaults={handleResetDefaults}
          />
        </div>
      </div>

      {/* 5. Per-Bird Instance Table */}
      <WatershedInstanceTable
        instances={data?.instances ?? []}
        birdCount={data?.bird_count ?? 0}
        estimatedTotalWeightG={data?.estimated_total_weight_g ?? 0}
        hasResponse={data !== null}
        loading={loading}
        onResetDefaults={handleResetDefaults}
      />

      {/* Fullscreen dark lightbox over all 4 Watershed stages */}
      <StageLightbox
        stages={lightboxStages}
        activeIndex={lightboxIndex}
        onClose={() => setLightboxIndex(null)}
        onSelectIndex={(idx) => setLightboxIndex(idx)}
      />
    </div>
  );
}
