import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAbortableRequest, useFlockBatch } from "../../hooks/useStudioHooks.ts";
import { type ApiError, predictSingleBird } from "../../lib/api.ts";
import type { PredictResponse, SingleBirdPresetItem } from "../../types/api.ts";
import { InlineErrorBanner } from "../layout/InlineErrorBanner.tsx";
import { StageLightbox } from "../layout/StageLightbox.tsx";
import { DualLensComparison } from "./DualLensComparison.tsx";
import { FlockBatchCard } from "./FlockBatchCard.tsx";
import { OpticalCalibrationCard, type OpticalOverrideValues } from "./OpticalCalibrationCard.tsx";
import { PredictionSummaryCard } from "./PredictionSummaryCard.tsx";
import { PresetsStrip } from "./PresetsStrip.tsx";
import { TelemetryPanel } from "./TelemetryPanel.tsx";
import { UploadDropzone } from "./UploadDropzone.tsx";
import { buildXRayLightboxStages, XRayQuadViewer } from "./XRayQuadViewer.tsx";

const DEFAULT_BOOT_PRESET_ID = "sb_intermediate_grower";

type ActiveInferenceSource =
  | { type: "preset"; presetId: string }
  | { type: "upload"; file: File; fileName: string };

export interface XRayStudioTabProps {
  presets: readonly SingleBirdPresetItem[];
  presetsLoading: boolean;
  onLatency: (latencyMs: number) => void;
}

export function XRayStudioTab({ presets, presetsLoading, onLatency }: XRayStudioTabProps) {
  const {
    data: prediction,
    loading,
    error: requestError,
    run,
    clearError: clearRequestError,
  } = useAbortableRequest<PredictResponse>();
  const flock = useFlockBatch();

  const [activeSource, setActiveSource] = useState<ActiveInferenceSource>({
    type: "preset",
    presetId: DEFAULT_BOOT_PRESET_ID,
  });
  const [clientError, setClientError] = useState<ApiError | null>(null);
  const [dualLensOpen, setDualLensOpen] = useState<boolean>(false);
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  const lastOverrideRef = useRef<OpticalOverrideValues | undefined>(undefined);
  const onLatencyRef = useRef(onLatency);
  onLatencyRef.current = onLatency;

  const executePredict = useCallback(
    (source: ActiveInferenceSource, override?: OpticalOverrideValues) => {
      setClientError(null);
      lastOverrideRef.current = override;

      void run((signal) =>
        predictSingleBird(
          {
            presetId: source.type === "preset" ? source.presetId : undefined,
            file: source.type === "upload" ? source.file : undefined,
            focalLengthMm: override?.focalLengthMm,
            zoomMultiple: override?.zoomMultiple,
          },
          signal,
        ),
      ).then((result) => {
        if (result) {
          onLatencyRef.current(result.timing.total_ms);
        }
      });
    },
    [run],
  );

  // Zero-empty-state pitch boot (D-08): immediately score sb_intermediate_grower on mount
  useEffect(() => {
    executePredict({ type: "preset", presetId: DEFAULT_BOOT_PRESET_ID });
  }, [executePredict]);

  const handleSelectPreset = useCallback(
    (presetId: string) => {
      const nextSource: ActiveInferenceSource = { type: "preset", presetId };
      setActiveSource(nextSource);
      executePredict(nextSource);
    },
    [executePredict],
  );

  const handleSelectFile = useCallback(
    (file: File) => {
      const nextSource: ActiveInferenceSource = {
        type: "upload",
        file,
        fileName: file.name,
      };
      setActiveSource(nextSource);
      executePredict(nextSource);
    },
    [executePredict],
  );

  const handleValidationError = useCallback((error: ApiError) => {
    setClientError(error);
  }, []);

  const handleApplyOverride = useCallback(
    (values: OpticalOverrideValues) => {
      executePredict(activeSource, values);
    },
    [activeSource, executePredict],
  );

  const handleResetExif = useCallback(() => {
    executePredict(activeSource, undefined);
  }, [activeSource, executePredict]);

  const handleRetry = useCallback(() => {
    executePredict(activeSource, lastOverrideRef.current);
  }, [activeSource, executePredict]);

  const activePresetMeta = useMemo(() => {
    if (activeSource.type !== "preset") {
      return null;
    }
    return presets.find((item) => item.preset_id === activeSource.presetId) ?? null;
  }, [activeSource, presets]);

  const handleAddCurrentToFlock = useCallback(() => {
    if (!prediction) {
      return;
    }
    const label =
      prediction.source_type === "preset"
        ? (activePresetMeta?.label ?? prediction.preset_id ?? "Preset Auditado OOF")
        : activeSource.type === "upload"
          ? activeSource.fileName
          : "Upload Externo";

    flock.addEntry({
      label,
      weightG: prediction.predicted_weight_g,
      sourceType: prediction.source_type,
      presetId: prediction.preset_id,
      ageStageLabel: prediction.age_stage.stage_label,
    });
  }, [prediction, activePresetMeta, activeSource, flock]);

  const lightboxStages = useMemo(
    () => buildXRayLightboxStages(prediction?.xray ?? null),
    [prediction],
  );

  const displayedError = clientError ?? requestError;
  const errorActionLabel =
    displayedError?.status === 422
      ? "Restaurar EXIF Original"
      : displayedError && (displayedError.status === 0 || displayedError.status >= 500)
        ? "Tentar Análise Novamente"
        : undefined;
  const errorActionHandler =
    displayedError?.status === 422 ? handleResetExif : errorActionLabel ? handleRetry : undefined;

  return (
    <div className="space-y-6">
      {/* Row 1: 6-preset OOF gallery + Dual-Lens tandem trigger */}
      <PresetsStrip
        presets={presets}
        loadingCatalog={presetsLoading}
        activePresetId={activeSource.type === "preset" ? activeSource.presetId : null}
        dualLensOpen={dualLensOpen}
        onSelectPreset={handleSelectPreset}
        onToggleDualLens={() => setDualLensOpen((prev) => !prev)}
      />

      {/* Collapsible live Dual-Lens 541 g Fold 0 invariance proof */}
      <DualLensComparison
        isOpen={dualLensOpen}
        onClose={() => setDualLensOpen(false)}
        onInspectPreset={(presetId) => {
          handleSelectPreset(presetId);
        }}
      />

      {/* Drag-and-drop / click upload strip */}
      <UploadDropzone
        activeSourceType={activeSource.type}
        activePresetLabel={
          activePresetMeta?.label ?? (activeSource.type === "preset" ? activeSource.presetId : null)
        }
        activeFileName={activeSource.type === "upload" ? activeSource.fileName : null}
        loading={loading}
        onSelectFile={handleSelectFile}
        onValidationError={handleValidationError}
      />

      {/* Recoverable client/server error banner */}
      {displayedError ? (
        <InlineErrorBanner
          title="Falha na Análise da Imagem"
          message={displayedError.userMessage}
          actionLabel={errorActionLabel}
          onAction={errorActionHandler}
          onDismiss={() => {
            setClientError(null);
            clearRequestError();
          }}
        />
      ) : null}

      {/* Primary Split Workspace: 8 cols X-Ray 4-Stage Viewport + 4 cols Prediction & Optical Override rail */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-8">
          <XRayQuadViewer
            xray={prediction?.xray ?? null}
            loading={loading}
            onOpenStage={setLightboxIndex}
          />
        </div>

        <div className="min-w-0 space-y-6 lg:col-span-4">
          <PredictionSummaryCard prediction={prediction} loading={loading} />
          <OpticalCalibrationCard
            optical={prediction?.optical ?? null}
            loading={loading}
            hasActiveSource
            onApplyOverride={handleApplyOverride}
            onResetExif={handleResetExif}
          />
        </div>
      </div>

      {/* Lower Audit & Flock Row (32px major workspace rhythm via pt-2 + space-y-6 = 32px) */}
      <div className="grid grid-cols-1 gap-6 pt-2 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-6">
          <TelemetryPanel prediction={prediction} loading={loading} />
        </div>

        <div className="min-w-0 lg:col-span-6">
          <FlockBatchCard
            entries={flock.entries}
            stats={flock.stats}
            canAddCurrent={prediction !== null && !loading}
            onAddCurrent={handleAddCurrentToFlock}
            onRemoveEntry={flock.removeEntry}
            onClearEntries={flock.clearEntries}
          />
        </div>
      </div>

      {/* Fullscreen keyboard-navigable stage lightbox */}
      <StageLightbox
        stages={lightboxStages}
        activeIndex={lightboxIndex}
        onClose={() => setLightboxIndex(null)}
        onSelectIndex={setLightboxIndex}
      />
    </div>
  );
}
