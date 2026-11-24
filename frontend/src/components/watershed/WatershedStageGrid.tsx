import { Image as ImageIcon, Loader2, Maximize2 } from "lucide-react";
import { formatNumberPtBr } from "../../lib/format.ts";
import type { WatershedVisualsPayload } from "../../types/api.ts";
import type { LightboxStage } from "../layout/StageLightbox.tsx";

interface WatershedStageMeta {
  stepNumber: number;
  title: string;
  badge: string;
  ariaLabel: string;
  caption: string;
  key: keyof Pick<
    WatershedVisualsPayload,
    "stage1_original" | "stage2_union_mask" | "stage3_distance_peaks" | "stage4_color_basins"
  >;
}

const WATERSHED_STAGE_DEFINITIONS: readonly WatershedStageMeta[] = [
  {
    stepNumber: 1,
    title: "Original Orientada EXIF",
    badge: "1. Original Orientada EXIF",
    ariaLabel: "Ampliar etapa Watershed 1: Original Orientada EXIF",
    caption:
      "Cena multi-aves decodificada via Pillow com transposição EXIF preservando o enquadramento real sem distorção.",
    key: "stage1_original",
  },
  {
    stepNumber: 2,
    title: "Máscara União Multi-Ave",
    badge: "2. Máscara União Multi-Ave",
    ariaLabel: "Ampliar etapa Watershed 2: Máscara União Multi-Ave",
    caption:
      "União binária das aves encostadas após pré-supressão cromática de comedouro vermelho e exclusão do visor LCD/prato SF-400.",
    key: "stage2_union_mask",
  },
  {
    stepNumber: 3,
    title: "Mapa de Distância EDT + Picos",
    badge: "3. Mapa de Distância EDT + Picos",
    ariaLabel: "Ampliar etapa Watershed 3: Mapa de Distância EDT + Picos",
    caption:
      "Transformada de Distância Euclidiana exata (SciPy EDT) com picos h-maxima filtrados por proeminência (h_rel) e separação mínima.",
    key: "stage3_distance_peaks",
  },
  {
    stepNumber: 4,
    title: "Bacias Coloridas #ID",
    badge: "4. Bacias Coloridas #ID",
    ariaLabel: "Ampliar etapa Watershed 4: Bacias Coloridas #ID",
    caption:
      "Bacias separadas por Marker-based Watershed clássico com código de cor correspondente à tabela de instâncias abaixo.",
    key: "stage4_color_basins",
  },
] as const;

/** Build the ordered 4-stage descriptor array for fullscreen `StageLightbox` inspection. */
export function buildWatershedLightboxStages(
  visuals: WatershedVisualsPayload | null,
): LightboxStage[] {
  if (!visuals) {
    return [];
  }
  return WATERSHED_STAGE_DEFINITIONS.map((stage) => ({
    stepNumber: stage.stepNumber,
    title: stage.title,
    imageSrc: visuals[stage.key],
    caption: stage.caption,
  }));
}

export interface WatershedStageGridProps {
  visuals: WatershedVisualsPayload | null;
  loading: boolean;
  onOpenStage: (stageIndex: number) => void;
}

export function WatershedStageGrid({ visuals, loading, onOpenStage }: WatershedStageGridProps) {
  return (
    <section
      aria-label="Visualização das 4 etapas do Marker-based Watershed clássico"
      className="rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-2 pb-4">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Pipeline Isolado Multi-Aves · SciPy EDT + scikit-image Watershed
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            4 Etapas da Separação Watershed
          </h2>
        </div>

        {visuals ? (
          <span className="rounded-full border border-border bg-surface-subtle px-2 py-1 font-mono text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
            {formatNumberPtBr(visuals.total_b64_bytes / 1024, 1)} KiB
          </span>
        ) : null}
      </div>

      {/* Fixed 2x2 dark #0F172A aspect-[4/3] frames eliminate layout shift during slider re-runs */}
      <div className="relative" aria-busy={loading}>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {WATERSHED_STAGE_DEFINITIONS.map((stage, index) => {
            const imgSrc = visuals ? visuals[stage.key] : null;
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
              Calculando Transformada de Distância (EDT) e bacias Watershed...
            </p>
          </div>
        ) : null}
      </div>
    </section>
  );
}
