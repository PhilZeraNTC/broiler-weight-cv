import { ArrowLeftRight, ImageOff } from "lucide-react";
import { useState } from "react";
import { cn } from "../../lib/cn.ts";
import { formatGrams } from "../../lib/format.ts";
import type { SingleBirdPresetItem } from "../../types/api.ts";

export interface PresetsStripProps {
  presets: readonly SingleBirdPresetItem[];
  loadingCatalog: boolean;
  activePresetId: string | null;
  disabled?: boolean;
  dualLensOpen: boolean;
  onSelectPreset: (presetId: string) => void;
  onToggleDualLens: () => void;
}

function formatLensShortLabel(lensClass: SingleBirdPresetItem["lens_class"]): string {
  return lensClass === "ultrawide" ? "UW · 1,65 mm" : "Wide · 4,25 mm";
}

function formatSceneLabel(sceneType: SingleBirdPresetItem["scene_type"]): string {
  return sceneType === "bucket_dorsal" ? "Balde Dorsal" : "Balança/Cama";
}

interface PresetCardProps {
  preset: SingleBirdPresetItem;
  isSelected: boolean;
  disabled?: boolean;
  thumbnailFailed: boolean;
  onThumbnailError: (presetId: string) => void;
  onSelect: (presetId: string) => void;
  compact?: boolean;
}

function PresetCard({
  preset,
  isSelected,
  disabled,
  thumbnailFailed,
  onThumbnailError,
  onSelect,
  compact = false,
}: PresetCardProps) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => onSelect(preset.preset_id)}
      aria-pressed={isSelected}
      className={cn(
        "group flex shrink-0 flex-col rounded-xl border bg-surface p-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-60",
        compact ? "w-[184px]" : "w-[196px]",
        isSelected
          ? "border-accent ring-2 ring-accent"
          : "border-border hover:border-ink-muted/60 hover:bg-canvas",
      )}
    >
      {/* Dark #0F172A thumbnail frame strictly fed by opaque /api/presets/{preset_id}/thumbnail */}
      <div className="relative h-24 w-full overflow-hidden rounded-lg bg-viewport-dark">
        {thumbnailFailed ? (
          <div className="flex h-full w-full flex-col items-center justify-center gap-1 px-2 text-center text-slate-400">
            <ImageOff className="h-5 w-5 shrink-0" aria-hidden="true" />
            <span className="text-[12px] font-semibold leading-[1.33]">Prévia indisponível</span>
          </div>
        ) : (
          <img
            src={preset.thumbnail_url}
            alt={preset.label}
            loading="lazy"
            onError={() => onThumbnailError(preset.preset_id)}
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

      {/* Metadata block */}
      <div className="mt-2 flex flex-1 flex-col justify-between gap-2">
        <p className="line-clamp-2 text-[14px] font-normal leading-[1.5] text-ink">
          {preset.label}
        </p>

        <div className="space-y-1 border-t border-border pt-2">
          <div className="flex items-center justify-between gap-2 text-[12px] font-semibold leading-[1.33] tabular-nums">
            <span className="text-ink">Real: {formatGrams(preset.ground_truth_weight_g, 0)}</span>
            <span className="rounded bg-surface-subtle px-1.5 py-0.5 font-mono text-ink-muted">
              Fold {preset.fold}
            </span>
          </div>
          <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
            {formatSceneLabel(preset.scene_type)}
          </p>
        </div>
      </div>
    </button>
  );
}

export function PresetsStrip({
  presets,
  loadingCatalog,
  activePresetId,
  disabled,
  dualLensOpen,
  onSelectPreset,
  onToggleDualLens,
}: PresetsStripProps) {
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

  const standalonePresets = presets.filter((item) => item.pair_group_id === null);
  const dualLensPresets = presets.filter((item) => item.pair_group_id !== null);
  const dualLensGt = dualLensPresets[0]?.ground_truth_weight_g ?? 541;
  const dualLensFold = dualLensPresets[0]?.fold ?? 0;

  return (
    <section
      aria-label="Galeria de Presets Auditados OOF"
      className="rounded-xl border border-border bg-surface p-4"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2">
        <div>
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Galeria de Validação Honesta (Out-of-Fold · 5 Folds)
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            Presets Curados para Demonstração
          </h2>
        </div>

        <span className="rounded-full border border-border bg-surface-subtle px-2 py-1 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
          {presets.length > 0 ? `${presets.length} presets auditados` : "Carregando catálogo"}
        </span>
      </div>

      {loadingCatalog && presets.length === 0 ? (
        <div className="mt-2 flex gap-4 overflow-x-auto pb-2" aria-busy="true">
          {[1, 2, 3, 4, 5, 6].map((slot) => (
            <div
              key={slot}
              className="h-44 w-[196px] shrink-0 animate-pulse rounded-xl border border-border bg-surface-subtle p-2"
            >
              <div className="h-24 w-full rounded-lg bg-viewport-dark/80" />
              <div className="mt-2 h-4 w-3/4 rounded bg-border" />
              <div className="mt-2 h-3 w-1/2 rounded bg-border" />
            </div>
          ))}
        </div>
      ) : presets.length === 0 ? (
        <p className="mt-2 rounded-lg border border-border bg-canvas p-4 text-[14px] font-normal leading-[1.5] text-ink-muted">
          Nenhum preset auditado carregado no momento. Verifique a conexão com a API local ou envie
          uma foto manualmente.
        </p>
      ) : (
        <div className="mt-2 overflow-x-auto pb-2">
          <div className="flex min-w-max items-stretch gap-4">
            {/* 4 standalone weight/difficulty OOF presets */}
            {standalonePresets.map((preset) => (
              <PresetCard
                key={preset.preset_id}
                preset={preset}
                isSelected={activePresetId === preset.preset_id}
                disabled={disabled}
                thumbnailFailed={failedThumbnails.has(preset.preset_id)}
                onThumbnailError={handleThumbnailError}
                onSelect={onSelectPreset}
              />
            ))}

            {/* Highlighted Dual-Lens tandem group (541 g · Fold 0) */}
            {dualLensPresets.length > 0 ? (
              <div className="flex shrink-0 flex-col gap-2 rounded-xl border border-border bg-surface-subtle p-2">
                <div className="flex flex-wrap items-center justify-between gap-2 px-1">
                  <span className="text-[12px] font-semibold leading-[1.33] tabular-nums text-ink">
                    Par Dual-Lens — Mesmo Frango ({formatGrams(dualLensGt, 0)} · Fold {dualLensFold}
                    )
                  </span>
                  <button
                    type="button"
                    onClick={onToggleDualLens}
                    aria-pressed={dualLensOpen}
                    aria-expanded={dualLensOpen}
                    className={cn(
                      "inline-flex min-h-[44px] items-center gap-2 rounded-lg px-4 text-[14px] font-normal leading-[1.5] transition-colors",
                      dualLensOpen
                        ? "bg-accent text-white hover:bg-accent-hover"
                        : "border border-border bg-surface text-ink hover:bg-canvas",
                    )}
                  >
                    <ArrowLeftRight className="h-4 w-4 shrink-0" aria-hidden="true" />
                    <span>Comparar Par Dual-Lens</span>
                  </button>
                </div>

                <div className="flex items-stretch gap-2">
                  {dualLensPresets.map((preset) => (
                    <PresetCard
                      key={preset.preset_id}
                      preset={preset}
                      isSelected={activePresetId === preset.preset_id}
                      disabled={disabled}
                      thumbnailFailed={failedThumbnails.has(preset.preset_id)}
                      onThumbnailError={handleThumbnailError}
                      onSelect={onSelectPreset}
                      compact
                    />
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </section>
  );
}
