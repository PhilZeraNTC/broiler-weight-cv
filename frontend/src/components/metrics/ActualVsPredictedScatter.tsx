import { FilterX, RotateCcw } from "lucide-react";
import { useMemo, useState } from "react";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { cn } from "../../lib/cn.ts";
import {
  formatGrams,
  formatNullableMetric,
  formatNumberPtBr,
  formatPercent,
  pluralizePtBr,
} from "../../lib/format.ts";
import {
  aggregateScatterBySession,
  computeR2,
  filterScatterPoints,
  type ScatterFilterState,
  type SessionScatterPoint,
} from "../../lib/scatter.ts";
import type { LensClass, OofScatterPoint, SceneType, WeightBand } from "../../types/api.ts";

export interface ActualVsPredictedScatterProps {
  points: readonly OofScatterPoint[];
  totalSessions?: number;
}

type ScatterViewMode = "images" | "sessions";

const DEFAULT_FILTERS: ScatterFilterState = {
  weightBand: "all",
  lensClass: "all",
  sceneType: "all",
};

const BAND_OPTIONS: ReadonlyArray<{ value: WeightBand | "all"; label: string }> = [
  { value: "all", label: "Todas" },
  { value: "<500g", label: "<500 g" },
  { value: "500-1000g", label: "500–1000 g" },
  { value: ">1000g", label: ">1000 g" },
];

const LENS_OPTIONS: ReadonlyArray<{ value: LensClass | "all"; label: string }> = [
  { value: "all", label: "Todas" },
  { value: "wide", label: "Wide 4,25 mm" },
  { value: "ultrawide", label: "Ultrawide 1,65 mm" },
];

const SCENE_OPTIONS: ReadonlyArray<{ value: SceneType | "all"; label: string }> = [
  { value: "all", label: "Todos" },
  { value: "bucket_dorsal", label: "Balde Dorsal" },
  { value: "scale_or_litter", label: "Balança/Cama" },
];

const AXIS_TICKS = [0, 500, 1000, 1500, 2000];

function isSessionPoint(
  datum: OofScatterPoint | SessionScatterPoint,
): datum is SessionScatterPoint {
  return "n_images" in datum;
}

function formatSceneLabel(scene: SceneType): string {
  return scene === "bucket_dorsal" ? "Balde Dorsal" : "Balança/Cama";
}

function formatLensGroupLabel(lens: LensClass | "mixed"): string {
  if (lens === "wide") return "Wide (4,25 mm)";
  if (lens === "ultrawide") return "Ultrawide (1,65 mm)";
  return "Misto (Wide + UW)";
}

interface CustomScatterTooltipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload?: OofScatterPoint | SessionScatterPoint }>;
}

function ScatterTooltipContent({ active, payload }: CustomScatterTooltipProps) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  const datum = payload[0]?.payload;
  if (!datum) {
    return null;
  }

  return (
    <div className="rounded-lg border border-border bg-surface p-4 text-[14px] font-normal leading-[1.5] text-ink shadow-md">
      <div className="flex items-center justify-between gap-4 border-b border-border pb-2">
        <span className="font-mono text-[12px] font-semibold leading-[1.33] text-ink">
          Sessão {datum.session_id}
        </span>
        <span className="rounded bg-surface-subtle px-2 py-0.5 font-mono text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
          Fold {datum.fold}
        </span>
      </div>

      <dl className="mt-2 space-y-1 text-[12px] font-semibold leading-[1.33] tabular-nums">
        <div className="flex items-center justify-between gap-4">
          <dt className="text-ink-muted">Peso Real:</dt>
          <dd className="text-ink">{formatGrams(datum.actual_g, 0)}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-ink-muted">Peso Estimado OOF:</dt>
          <dd className="text-ink">{formatGrams(datum.predicted_g, 1)}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-ink-muted">Erro Absoluto / %:</dt>
          <dd className="text-ink">
            {formatGrams(datum.abs_err_g, 1)} ({formatPercent(datum.pct_err, 1)})
          </dd>
        </div>
        {isSessionPoint(datum) ? (
          <>
            <div className="flex items-center justify-between gap-4">
              <dt className="text-ink-muted">Amostras na Sessão:</dt>
              <dd className="text-ink">{pluralizePtBr(datum.n_images, "imagem", "imagens")}</dd>
            </div>
            <div className="flex items-center justify-between gap-4">
              <dt className="text-ink-muted">Lente:</dt>
              <dd className="text-ink">{formatLensGroupLabel(datum.lens_group)}</dd>
            </div>
          </>
        ) : (
          <>
            <div className="flex items-center justify-between gap-4">
              <dt className="text-ink-muted">Lente:</dt>
              <dd className="text-ink">{formatLensGroupLabel(datum.lens_class)}</dd>
            </div>
            <div className="flex items-center justify-between gap-4">
              <dt className="text-ink-muted">Cenário:</dt>
              <dd className="text-ink">{formatSceneLabel(datum.scene_type)}</dd>
            </div>
          </>
        )}
      </dl>
    </div>
  );
}

export function ActualVsPredictedScatter({
  points,
  totalSessions = 188,
}: ActualVsPredictedScatterProps) {
  const [viewMode, setViewMode] = useState<ScatterViewMode>("images");
  const [filters, setFilters] = useState<ScatterFilterState>(DEFAULT_FILTERS);

  const hasActiveFilter =
    filters.weightBand !== "all" || filters.lensClass !== "all" || filters.sceneType !== "all";

  // Strictly filter raw 1,738 OOF points BEFORE aggregating to 188 sessions
  const filteredImages = useMemo(() => filterScatterPoints(points, filters), [points, filters]);

  const sessionPoints = useMemo(() => aggregateScatterBySession(filteredImages), [filteredImages]);

  const imageWideSeries = useMemo(
    () => filteredImages.filter((p) => p.lens_class === "wide"),
    [filteredImages],
  );

  const imageUwSeries = useMemo(
    () => filteredImages.filter((p) => p.lens_class === "ultrawide"),
    [filteredImages],
  );

  const sessionWideSeries = useMemo(
    () => sessionPoints.filter((s) => s.lens_group === "wide"),
    [sessionPoints],
  );

  const sessionUwSeries = useMemo(
    () => sessionPoints.filter((s) => s.lens_group === "ultrawide"),
    [sessionPoints],
  );

  const sessionMixedSeries = useMemo(
    () => sessionPoints.filter((s) => s.lens_group === "mixed"),
    [sessionPoints],
  );

  const activeCount = viewMode === "images" ? filteredImages.length : sessionPoints.length;
  const activeR2 = useMemo(
    () => computeR2(viewMode === "images" ? filteredImages : sessionPoints),
    [viewMode, filteredImages, sessionPoints],
  );

  const handleClearFilters = () => {
    setFilters(DEFAULT_FILTERS);
  };

  return (
    <section
      aria-label="Dispersão Peso Real versus Peso Estimado OOF"
      className="flex flex-col gap-4 rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      {/* Header + View Mode Toggle */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Validação Out-of-Fold · LightGBMLog C2 (Diagonal Ideal y = x)
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            Peso Real vs. Peso Estimado
          </h2>
        </div>

        <fieldset className="inline-flex rounded-lg border border-border bg-surface-subtle p-1">
          <legend className="sr-only">Granularidade da dispersão OOF</legend>
          <button
            type="button"
            onClick={() => setViewMode("images")}
            aria-pressed={viewMode === "images"}
            className={cn(
              "min-h-[44px] rounded-md px-4 text-[12px] font-semibold leading-[1.33] tabular-nums transition-colors",
              viewMode === "images"
                ? "bg-surface text-ink shadow-xs"
                : "text-ink-muted hover:text-ink",
            )}
          >
            Todas as Imagens ({formatNumberPtBr(points.length, 0)})
          </button>
          <button
            type="button"
            onClick={() => setViewMode("sessions")}
            aria-pressed={viewMode === "sessions"}
            className={cn(
              "min-h-[44px] rounded-md px-4 text-[12px] font-semibold leading-[1.33] tabular-nums transition-colors",
              viewMode === "sessions"
                ? "bg-surface text-ink shadow-xs"
                : "text-ink-muted hover:text-ink",
            )}
          >
            Média por Sessão ({formatNumberPtBr(totalSessions, 0)})
          </button>
        </fieldset>
      </div>

      {/* Interactive Filter Pills */}
      <div className="flex flex-wrap items-center gap-4 rounded-lg border border-border bg-canvas p-4">
        {/* Band pills */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[12px] font-semibold leading-[1.33] text-ink-muted">Faixa:</span>
          {BAND_OPTIONS.map((opt) => {
            const active = filters.weightBand === opt.value;
            return (
              <button
                key={opt.value}
                type="button"
                aria-pressed={active}
                onClick={() => setFilters((prev) => ({ ...prev, weightBand: opt.value }))}
                className={cn(
                  "rounded-full border px-2.5 py-1 text-[12px] font-semibold leading-[1.33] tabular-nums transition-colors",
                  active
                    ? "border-ink bg-ink text-white"
                    : "border-border bg-surface text-ink-muted hover:text-ink",
                )}
              >
                {opt.label}
              </button>
            );
          })}
        </div>

        {/* Lens pills */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[12px] font-semibold leading-[1.33] text-ink-muted">Lente:</span>
          {LENS_OPTIONS.map((opt) => {
            const active = filters.lensClass === opt.value;
            return (
              <button
                key={opt.value}
                type="button"
                aria-pressed={active}
                onClick={() => setFilters((prev) => ({ ...prev, lensClass: opt.value }))}
                className={cn(
                  "rounded-full border px-2.5 py-1 text-[12px] font-semibold leading-[1.33] tabular-nums transition-colors",
                  active
                    ? "border-ink bg-ink text-white"
                    : "border-border bg-surface text-ink-muted hover:text-ink",
                )}
              >
                {opt.label}
              </button>
            );
          })}
        </div>

        {/* Scene pills */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[12px] font-semibold leading-[1.33] text-ink-muted">Cenário:</span>
          {SCENE_OPTIONS.map((opt) => {
            const active = filters.sceneType === opt.value;
            return (
              <button
                key={opt.value}
                type="button"
                aria-pressed={active}
                onClick={() => setFilters((prev) => ({ ...prev, sceneType: opt.value }))}
                className={cn(
                  "rounded-full border px-2.5 py-1 text-[12px] font-semibold leading-[1.33] transition-colors",
                  active
                    ? "border-ink bg-ink text-white"
                    : "border-border bg-surface text-ink-muted hover:text-ink",
                )}
              >
                {opt.label}
              </button>
            );
          })}
        </div>

        {hasActiveFilter ? (
          <button
            type="button"
            onClick={handleClearFilters}
            className="ml-auto inline-flex min-h-[44px] items-center gap-2 rounded-lg border border-border bg-surface px-3 text-[12px] font-semibold leading-[1.33] text-ink transition-colors hover:bg-surface-subtle"
          >
            <RotateCcw className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            <span>Limpar Filtros do Gráfico</span>
          </button>
        ) : null}
      </div>

      {/* Live filtered count + filtered R² + legend strip */}
      <div className="flex flex-wrap items-center justify-between gap-4 text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
        <div className="flex flex-wrap items-center gap-4">
          <span>
            Exibindo:{" "}
            <span className="text-ink">
              {viewMode === "images"
                ? pluralizePtBr(activeCount, "imagem", "imagens")
                : pluralizePtBr(activeCount, "sessão", "sessões")}
            </span>
          </span>
          <span>
            R² do subconjunto: <span className="text-ink">{formatNullableMetric(activeR2, 4)}</span>
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden="true" className="h-2.5 w-2.5 rounded-full bg-accent" />
            <span>Wide (4,25 mm)</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden="true" className="h-2.5 w-2.5 rounded-full bg-chart-sky" />
            <span>Ultrawide (1,65 mm)</span>
          </span>
          {viewMode === "sessions" ? (
            <span className="inline-flex items-center gap-1.5">
              <span aria-hidden="true" className="h-2.5 w-2.5 rounded-full bg-ink-muted" />
              <span>Misto (Wide + UW)</span>
            </span>
          ) : null}
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden="true" className="w-4 border-t-2 border-dashed border-ink-muted" />
            <span>Ideal y = x</span>
          </span>
        </div>
      </div>

      {/* Empty state (0 filtered points) OR explicit-height Recharts v3 ScatterChart */}
      {activeCount === 0 ? (
        <div className="flex min-h-[360px] flex-col items-center justify-center gap-4 rounded-xl border border-border bg-canvas p-6 text-center">
          <div className="flex h-11 w-11 items-center justify-center rounded-lg border border-border bg-surface text-ink-muted">
            <FilterX className="h-5 w-5" aria-hidden="true" />
          </div>
          <div className="max-w-lg space-y-1">
            <h3 className="text-[20px] font-semibold leading-[1.2] text-ink">
              Nenhuma observação corresponde aos filtros ativos
            </h3>
            <p className="text-[14px] font-normal leading-[1.5] text-ink-muted">
              A combinação atual de faixa de peso, lente e cenário não possui amostras OOF. Clique
              em 'Limpar Filtros do Gráfico' para restaurar as 1.738 imagens.
            </p>
          </div>
          <button
            type="button"
            onClick={handleClearFilters}
            className="inline-flex min-h-[44px] items-center gap-2 rounded-lg border border-border bg-surface px-4 text-[14px] font-normal leading-[1.5] text-ink transition-colors hover:bg-surface-subtle"
          >
            <RotateCcw className="h-4 w-4 shrink-0" aria-hidden="true" />
            <span>Limpar Filtros do Gráfico</span>
          </button>
        </div>
      ) : (
        <div className="h-[360px] w-full min-w-0">
          <ResponsiveContainer width="100%" height={360}>
            <ScatterChart margin={{ top: 12, right: 16, bottom: 24, left: 8 }}>
              <CartesianGrid stroke="#E2E8F0" strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="actual_g"
                name="Peso Real"
                domain={[0, 2200]}
                ticks={AXIS_TICKS}
                tick={{ fontSize: 12, fill: "#475569" }}
                tickFormatter={(value: number) => `${formatNumberPtBr(value, 0)}`}
                label={{
                  value: "Peso Real na Balança (g)",
                  position: "insideBottom",
                  offset: -14,
                  fontSize: 12,
                  fontWeight: 600,
                  fill: "#475569",
                }}
              />
              <YAxis
                type="number"
                dataKey="predicted_g"
                name="Peso Estimado"
                domain={[0, 2200]}
                ticks={AXIS_TICKS}
                tick={{ fontSize: 12, fill: "#475569" }}
                tickFormatter={(value: number) => `${formatNumberPtBr(value, 0)}`}
                label={{
                  value: "Peso Estimado OOF (g)",
                  angle: -90,
                  position: "insideLeft",
                  offset: 4,
                  fontSize: 12,
                  fontWeight: 600,
                  fill: "#475569",
                  style: { textAnchor: "middle" },
                }}
              />
              <ZAxis range={[22, 22]} />
              <Tooltip
                cursor={{ strokeDasharray: "3 3", stroke: "#94A3B8" }}
                content={<ScatterTooltipContent />}
              />
              <ReferenceLine
                segment={[
                  { x: 0, y: 0 },
                  { x: 2200, y: 2200 },
                ]}
                stroke="#64748B"
                strokeDasharray="4 4"
                ifOverflow="extendDomain"
              />

              {viewMode === "images" ? (
                <>
                  <Scatter
                    name="Wide (4,25 mm)"
                    data={imageWideSeries}
                    fill="#047857"
                    fillOpacity={0.55}
                    isAnimationActive={false}
                  />
                  <Scatter
                    name="Ultrawide (1,65 mm)"
                    data={imageUwSeries}
                    fill="#0369A1"
                    fillOpacity={0.55}
                    isAnimationActive={false}
                  />
                </>
              ) : (
                <>
                  <Scatter
                    name="Wide (4,25 mm)"
                    data={sessionWideSeries}
                    fill="#047857"
                    fillOpacity={0.85}
                    isAnimationActive={false}
                  />
                  <Scatter
                    name="Ultrawide (1,65 mm)"
                    data={sessionUwSeries}
                    fill="#0369A1"
                    fillOpacity={0.85}
                    isAnimationActive={false}
                  />
                  <Scatter
                    name="Misto (Wide + UW)"
                    data={sessionMixedSeries}
                    fill="#475569"
                    fillOpacity={0.85}
                    isAnimationActive={false}
                  />
                </>
              )}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}
