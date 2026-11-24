import { ImageOff, ImageUp, Upload } from "lucide-react";
import { type ChangeEvent, type DragEvent, type RefObject, useRef, useState } from "react";
import { ApiError, validateClientUploadFile } from "../../lib/api.ts";
import { cn } from "../../lib/cn.ts";
import { formatGrams, pluralizePtBr } from "../../lib/format.ts";
import type { SourceType, WatershedPresetItem } from "../../types/api.ts";

export interface WatershedPresetStripProps {
  presets: readonly WatershedPresetItem[];
  loadingCatalog: boolean;
  activeSourceType: SourceType;
  activePresetId: string | null;
  activeFileName: string | null;
  loading: boolean;
  fileInputRef?: RefObject<HTMLInputElement | null>;
  onSelectPreset: (presetId: string) => void;
  onSelectFile: (file: File) => void;
  onValidationError: (error: ApiError) => void;
}

const UNSUPPORTED_FALLBACK_MESSAGE =
  "Formato de imagem não suportado. Envie um arquivo JPG, PNG ou WEBP ou escolha um dos presets auditados na galeria.";

function formatLensShortLabel(lensClass: WatershedPresetItem["lens_class"]): string {
  return lensClass === "ultrawide" ? "UW · 1,65 mm" : "Wide · 4,25 mm";
}

export function WatershedPresetStrip({
  presets,
  loadingCatalog,
  activeSourceType,
  activePresetId,
  activeFileName,
  loading,
  fileInputRef,
  onSelectPreset,
  onSelectFile,
  onValidationError,
}: WatershedPresetStripProps) {
  const internalInputRef = useRef<HTMLInputElement | null>(null);
  const inputRef = fileInputRef ?? internalInputRef;
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [failedThumbnails, setFailedThumbnails] = useState<ReadonlySet<string>>(() => new Set());

  const handleThumbnailError = (presetId: string) => {
    setFailedThumbnails((prev) => {
      if (prev.has(presetId)) {
        return prev;
      }
      const next = new Set(prev);
      next.add(presetId);
      return next;
    });
  };

  const processCandidateFile = (file: File | null | undefined) => {
    if (!file) {
      return;
    }
    try {
      validateClientUploadFile(file);
      onSelectFile(file);
    } catch (err) {
      if (err instanceof ApiError) {
        onValidationError(err);
      } else {
        onValidationError(new ApiError(415, UNSUPPORTED_FALLBACK_MESSAGE));
      }
    }
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    processCandidateFile(file);
  };

  const handleDragOver = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    if (!loading) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    setIsDragging(false);
    if (loading) {
      return;
    }
    const file = event.dataTransfer.files?.[0];
    processCandidateFile(file);
  };

  const triggerFilePicker = () => {
    if (!loading) {
      inputRef.current?.click();
    }
  };

  const isCustomUpload = activeSourceType === "upload" && activeFileName !== null;

  return (
    <section
      aria-label="Cenas curadas multi-aves e envio customizado para Watershed"
      className="rounded-xl border border-border bg-surface p-4"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2">
        <div>
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Módulo Multi-Aves · Marker-based Watershed Clássico (Quarentena Reaproveitada)
          </p>
          <h1 className="text-[20px] font-semibold leading-[1.2] text-ink">
            Separação Watershed — Cenas com Aves Encostadas
          </h1>
        </div>

        <span className="rounded-full border border-border bg-surface-subtle px-2 py-1 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
          {presets.length > 0
            ? pluralizePtBr(presets.length, "cena multi-aves curada", "cenas multi-aves curadas")
            : "Carregando catálogo"}
        </span>
      </div>

      <div className="mt-2 grid grid-cols-1 gap-4 xl:grid-cols-12">
        {/* 4 curated multi-bird preset cards in horizontal overflow-x-auto strip */}
        <div className="min-w-0 xl:col-span-8">
          {loadingCatalog && presets.length === 0 ? (
            <div className="flex gap-4 overflow-x-auto pb-2" aria-busy="true">
              {[1, 2, 3, 4].map((slot) => (
                <div
                  key={slot}
                  className="h-48 w-[204px] shrink-0 animate-pulse rounded-xl border border-border bg-surface-subtle p-2"
                >
                  <div className="h-24 w-full rounded-lg bg-viewport-dark/80" />
                  <div className="mt-2 h-4 w-3/4 rounded bg-border" />
                  <div className="mt-2 h-3 w-1/2 rounded bg-border" />
                </div>
              ))}
            </div>
          ) : presets.length === 0 ? (
            <p className="rounded-lg border border-border bg-canvas p-4 text-[14px] font-normal leading-[1.5] text-ink-muted">
              Nenhum preset multi-aves disponível no momento. Verifique a API local ou envie uma
              foto de grupo ao lado.
            </p>
          ) : (
            <div className="overflow-x-auto pb-2">
              <div className="flex min-w-max items-stretch gap-4">
                {presets.map((preset) => {
                  const isSelected =
                    activeSourceType === "preset" && activePresetId === preset.preset_id;
                  const thumbFailed = failedThumbnails.has(preset.preset_id);

                  return (
                    <button
                      key={preset.preset_id}
                      type="button"
                      disabled={loading}
                      onClick={() => onSelectPreset(preset.preset_id)}
                      aria-pressed={isSelected}
                      className={cn(
                        "flex w-[204px] shrink-0 flex-col rounded-xl border bg-surface p-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-60",
                        isSelected
                          ? "border-accent ring-2 ring-accent"
                          : "border-border hover:border-ink-muted/60 hover:bg-canvas",
                      )}
                    >
                      {/* Dark #0F172A thumbnail frame strictly using opaque preset_id URL */}
                      <div className="relative h-24 w-full overflow-hidden rounded-lg bg-viewport-dark">
                        {thumbFailed ? (
                          <div className="flex h-full w-full flex-col items-center justify-center gap-1 px-2 text-center text-slate-400">
                            <ImageOff className="h-5 w-5 shrink-0" aria-hidden="true" />
                            <span className="text-[12px] font-semibold leading-[1.33]">
                              Prévia indisponível
                            </span>
                          </div>
                        ) : (
                          <img
                            src={preset.thumbnail_url}
                            alt={preset.label}
                            loading="lazy"
                            onError={() => handleThumbnailError(preset.preset_id)}
                            className="h-full w-full object-contain"
                          />
                        )}

                        <span className="pointer-events-none absolute bottom-1 left-1 rounded bg-viewport-dark/85 px-1.5 py-0.5 text-[12px] font-semibold leading-[1.33] tabular-nums text-white">
                          {formatLensShortLabel(preset.lens_class)}
                        </span>

                        {isSelected ? (
                          <span
                            aria-hidden="true"
                            className="absolute top-2 right-2 h-2.5 w-2.5 rounded-full bg-accent ring-2 ring-white"
                          />
                        ) : null}
                      </div>

                      <div className="mt-2 flex flex-1 flex-col justify-between gap-2">
                        <p className="line-clamp-2 text-[14px] font-normal leading-[1.5] text-ink">
                          {preset.label}
                        </p>

                        <div className="space-y-1 border-t border-border pt-2 text-[12px] font-semibold leading-[1.33] tabular-nums">
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-ink">
                              {pluralizePtBr(
                                preset.expected_bird_count,
                                "ave esperada",
                                "aves esperadas",
                              )}
                            </span>
                            <span className="rounded bg-surface-subtle px-1.5 py-0.5 font-mono text-ink-muted">
                              #{preset.folder_id}
                            </span>
                          </div>
                          <p className="text-ink-muted">
                            Soma grupo: {formatGrams(preset.group_sum_weight_g, 0)}
                          </p>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Adjacent custom multi-bird upload dropzone card */}
        <section
          aria-label="Envio de foto de grupo multi-aves"
          onDragEnter={handleDragOver}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn(
            "flex flex-col justify-between gap-4 rounded-xl border border-dashed bg-canvas p-4 transition-colors xl:col-span-4",
            isDragging ? "border-accent bg-accent-tint/50" : "border-border",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
            onChange={handleInputChange}
            className="sr-only"
            aria-label="Selecionar foto de grupo multi-aves (JPG, PNG ou WEBP até 15 MB)"
          />

          <div className="flex min-w-0 items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-border bg-surface text-ink-muted">
              <ImageUp className="h-5 w-5" aria-hidden="true" />
            </div>

            <div className="min-w-0 space-y-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded-full border border-border bg-surface px-2 py-0.5 text-[12px] font-semibold leading-[1.33] text-ink-muted">
                  {isCustomUpload ? "Upload Externo" : "Upload Customizado"}
                </span>
                {isCustomUpload ? (
                  <span
                    title={activeFileName}
                    className="max-w-[240px] truncate text-[12px] font-semibold leading-[1.33] text-ink"
                  >
                    {activeFileName}
                  </span>
                ) : null}
              </div>

              <p className="text-[14px] font-normal leading-[1.5] text-ink-muted">
                Arraste uma foto JPG, PNG ou WEBP (até 15 MB) contendo múltiplas aves para testar a
                separação clássica por Transformada de Distância (EDT).
              </p>
            </div>
          </div>

          <button
            type="button"
            disabled={loading}
            onClick={triggerFilePicker}
            className="inline-flex min-h-[44px] w-full items-center justify-center gap-2 rounded-lg border border-border bg-surface px-4 text-[14px] font-normal leading-[1.5] text-ink transition-colors hover:bg-surface-subtle disabled:cursor-not-allowed disabled:opacity-60"
          >
            <Upload className="h-4 w-4 shrink-0" aria-hidden="true" />
            <span>Enviar Foto de Grupo</span>
          </button>
        </section>
      </div>
    </section>
  );
}
