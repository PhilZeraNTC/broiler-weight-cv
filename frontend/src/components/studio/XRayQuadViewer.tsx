import { Image as ImageIcon, Loader2, Maximize2 } from "lucide-react";
import { formatNumberPtBr } from "../../lib/format.ts";
import type { XRayPayload } from "../../types/api.ts";
import type { LightboxStage } from "../layout/StageLightbox.tsx";

interface XRayStageMeta {
  stepNumber: number;
  title: string;
  badge: string;
  ariaLabel: string;
  caption: string;
  key: keyof Pick<
    XRayPayload,
    "stage1_original" | "stage2_clahe" | "stage3_mask" | "stage4_overlay"
  >;
}

const XRAY_STAGE_DEFINITIONS: readonly XRayStageMeta[] = [
  {
    stepNumber: 1,
    title: "Original Orientada EXIF",
    badge: "1. Original Orientada EXIF",
    ariaLabel: "Ampliar etapa 1: Original Orientada EXIF",
    caption:
      "Entrada decodificada via Pillow com transposição obrigatória de orientação EXIF antes do redimensionamento.",
    key: "stage1_original",
  },
  {
    stepNumber: 2,
    title: "Pré-processamento CLAHE-Lab",
    badge: "2. Pré-processamento CLAHE-Lab",
    ariaLabel: "Ampliar etapa 2: Pré-processamento CLAHE-Lab",
    caption:
      "Equalização adaptativa CLAHE no canal de luminância L (CIE Lab) associada à suavização clássica para preservar bordas.",
    key: "stage2_clahe",
  },
  {
    stepNumber: 3,
    title: "Máscara Binária — LCD Excluído",
    badge: "3. Máscara Binária — LCD Excluído",
    ariaLabel: "Ampliar etapa 3: Máscara Binária — LCD Excluído",
    caption:
      "Segmentação clássica multi-espaço com exclusão geométrica do visor LCD/prato da balança SF-400 antes da extração de atributos.",
    key: "stage3_mask",
  },
  {
    stepNumber: 4,
    title: "Contorno + Elipse + Eixos",
    badge: "4. Contorno + Elipse + Eixos",
    ariaLabel: "Ampliar etapa 4: Contorno + Elipse + Eixos",
    caption:
      "Contorno principal da ave com ajuste de elipse e eixos maior/menor normalizados pela calibração óptica EXIF.",
    key: "stage4_overlay",
  },
] as const;

/** Build the ordered 4-stage descriptor array for fullscreen `StageLightbox` inspection. */
export function buildXRayLightboxStages(xray: XRayPayload | null): LightboxStage[] {
  if (!xray) {
    return [];
  }
  return XRAY_STAGE_DEFINITIONS.map((stage) => ({
    stepNumber: stage.stepNumber,
    title: stage.title,
    imageSrc: xray[stage.key],
    caption: stage.caption,
  }));
}

export interface XRayQuadViewerProps {
  xray: XRayPayload | null;
  loading: boolean;
  onOpenStage: (stageIndex: number) => void;
}

export function XRayQuadViewer({ xray, loading, onOpenStage }: XRayQuadViewerProps) {
  return (
    <section
      aria-label="Visualização Raio-X das 4 etapas do PDI clássico"
      className="rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-2 pb-4">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Visão Raio-X · Pipeline PDI Clássico Interpretável
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            4 Etapas de Processamento Digital de Imagens
          </h2>
        </div>

        {xray ? (
          <span className="rounded-full border border-border bg-surface-subtle px-2 py-1 font-mono text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
            {formatNumberPtBr(xray.display_w, 0)}×{formatNumberPtBr(xray.display_h, 0)} px
          </span>
        ) : null}
      </div>

      {/* Preserve fixed 2x2 dark #0F172A aspect-[4/3] geometry during loading to eliminate layout shift */}
      <div className="relative" aria-busy={loading}>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {XRAY_STAGE_DEFINITIONS.map((stage, index) => {
            const imgSrc = xray ? xray[stage.key] : null;
            return (
              <figure
                key={stage.key}
                className="relative aspect-[4/3] w-full overflow-hidden rounded-lg border border-slate-800 bg-viewport-dark"
              >
                {imgSrc ? (
                  <img src={imgSrc} alt={stage.badge} className="h-full w-full object-contain" />
                ) : (
                  <div className="flex h-full w-full flex-col items-center justify-center gap-2 p-4 text-center text-slate-400">
                    <ImageIcon className="h-6 w-6 shrink-0" aria-hidden="true" />
                    <span className="text-[12px] font-semibold leading-[1.33]">
                      Aguardando etapa {stage.stepNumber}
                    </span>
                  </div>
                )}

                <figcaption className="pointer-events-none absolute top-2 left-2 max-w-[calc(100%-64px)] truncate rounded-md bg-viewport-dark/85 px-2 py-1 text-[12px] font-semibold leading-[1.33] text-white">
                  {stage.badge}
                </figcaption>

                <button
                  type="button"
                  disabled={!imgSrc}
                  onClick={() => onOpenStage(index)}
                  aria-label={stage.ariaLabel}
                  className="absolute top-2 right-2 inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-lg border border-slate-700 bg-viewport-dark/85 text-white transition-colors hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Maximize2 className="h-4 w-4" aria-hidden="true" />
                </button>
              </figure>
            );
          })}
        </div>

        {loading ? (
          <div
            aria-live="polite"
            className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 rounded-lg bg-viewport-dark/80 p-4 text-center text-white"
          >
            <Loader2 className="h-6 w-6 shrink-0 animate-spin text-white" aria-hidden="true" />
            <p className="max-w-md text-[14px] font-normal leading-[1.5] text-white">
              Executando pipeline PDI clássico e regressão LightGBM...
            </p>
          </div>
        ) : null}
      </div>
    </section>
  );
}
