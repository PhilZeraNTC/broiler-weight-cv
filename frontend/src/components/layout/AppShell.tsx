import { Activity, AlertTriangle, CheckCircle2, RefreshCw } from "lucide-react";
import type { KeyboardEvent, ReactNode } from "react";
import type { HealthStatus } from "../../hooks/useStudioHooks.ts";
import { cn } from "../../lib/cn.ts";
import { formatNumberPtBr } from "../../lib/format.ts";
import type { HealthResponse } from "../../types/api.ts";

export type TabId = "estudio" | "watershed" | "metricas" | "visao-futura";

export interface TabItemConfig {
  id: TabId;
  hash: `#${TabId}`;
  label: string;
  badge?: string;
}

export const TAB_ITEMS: readonly TabItemConfig[] = [
  { id: "estudio", hash: "#estudio", label: "Estúdio Raio-X" },
  { id: "watershed", hash: "#watershed", label: "Separação Watershed" },
  { id: "metricas", hash: "#metricas", label: "Auditoria & Métricas" },
  {
    id: "visao-futura",
    hash: "#visao-futura",
    label: "Visão Futura (v2)",
    badge: "Conceitual",
  },
] as const;

export interface AppShellProps {
  activeTab: TabId;
  onSelectTab: (tab: TabId) => void;
  health: HealthResponse | null;
  healthStatus: HealthStatus;
  lastLatencyMs: number | null;
  onRefreshHealth?: () => void;
  children: ReactNode;
}

export function AppShell({
  activeTab,
  onSelectTab,
  health,
  healthStatus,
  lastLatencyMs,
  onRefreshHealth,
  children,
}: AppShellProps) {
  const handleTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      const next = TAB_ITEMS[(index + 1) % TAB_ITEMS.length];
      onSelectTab(next.id);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      const prev = TAB_ITEMS[(index - 1 + TAB_ITEMS.length) % TAB_ITEMS.length];
      onSelectTab(prev.id);
    } else if (event.key === "Home") {
      event.preventDefault();
      onSelectTab(TAB_ITEMS[0].id);
    } else if (event.key === "End") {
      event.preventDefault();
      onSelectTab(TAB_ITEMS[TAB_ITEMS.length - 1].id);
    }
  };

  const latencyWithinSla = lastLatencyMs !== null && lastLatencyMs < 500;

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <header className="sticky top-0 z-30 border-b border-border bg-surface">
        <div className="mx-auto flex min-h-16 max-w-[1440px] flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 lg:h-16 lg:flex-nowrap lg:px-6">
          {/* Left cluster: Brand mark + classical PDI identity pill */}
          <div className="flex min-w-0 items-center gap-2 py-2 lg:py-0">
            <span className="whitespace-nowrap text-[20px] font-semibold leading-[1.2] tracking-tight text-ink">
              AvianScale CV
            </span>
            <span className="hidden whitespace-nowrap rounded-full border border-border bg-surface-subtle px-2 py-1 text-[12px] font-semibold leading-[1.33] text-ink-muted sm:inline-flex">
              PDI Clássico + EXIF · Zero Deep Learning
            </span>
          </div>

          {/* Center cluster: 4-tab primary navigation inside horizontal scroll container */}
          <nav
            aria-label="Áreas de trabalho do AvianScale CV"
            className="order-3 w-full min-w-0 overflow-x-auto lg:order-2 lg:h-full lg:w-auto"
          >
            <div role="tablist" className="flex h-full min-w-max items-stretch gap-2">
              {TAB_ITEMS.map((item, idx) => {
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    id={`tab-${item.id}`}
                    type="button"
                    role="tab"
                    aria-selected={isActive}
                    aria-controls={`panel-${item.id}`}
                    tabIndex={isActive ? 0 : -1}
                    onClick={() => onSelectTab(item.id)}
                    onKeyDown={(event) => handleTabKeyDown(event, idx)}
                    className={cn(
                      "inline-flex min-h-[44px] items-center gap-2 whitespace-nowrap border-b-2 px-3 text-[14px] leading-[1.5] transition-colors",
                      isActive
                        ? "border-accent font-semibold text-accent"
                        : "border-transparent font-normal text-ink-muted hover:text-ink",
                    )}
                  >
                    <span>{item.label}</span>
                    {item.badge ? (
                      <span className="rounded-full border border-warning-border bg-warning-tint px-2 py-0.5 text-[12px] font-semibold leading-[1.33] text-warning">
                        {item.badge}
                      </span>
                    ) : null}
                  </button>
                );
              })}
            </div>
          </nav>

          {/* Right cluster: Live API health status + latency pill */}
          <div className="order-2 flex shrink-0 items-center gap-2 lg:order-3">
            {healthStatus === "online" ? (
              <div
                aria-live="polite"
                className="inline-flex items-center gap-2 rounded-full border border-accent/25 bg-accent-tint px-3 py-1 text-[12px] font-semibold leading-[1.33] text-accent"
              >
                <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span className="whitespace-nowrap">
                  {health
                    ? `API Online · ${health.winning_model_name} C2`
                    : "API Online · < 500 ms"}
                </span>
                {lastLatencyMs !== null ? (
                  <span
                    className={cn(
                      "rounded-full border px-1.5 py-0.5 tabular-nums",
                      latencyWithinSla
                        ? "border-accent/30 bg-surface text-accent"
                        : "border-warning-border bg-warning-tint text-warning",
                    )}
                  >
                    {formatNumberPtBr(lastLatencyMs, 0)} ms
                  </span>
                ) : null}
              </div>
            ) : healthStatus === "checking" ? (
              <div
                aria-live="polite"
                className="inline-flex items-center gap-2 rounded-full border border-border bg-surface-subtle px-3 py-1 text-[12px] font-semibold leading-[1.33] text-ink-muted"
              >
                <Activity className="h-3.5 w-3.5 shrink-0 animate-pulse" aria-hidden="true" />
                <span className="whitespace-nowrap">Verificando API...</span>
              </div>
            ) : (
              <button
                type="button"
                onClick={onRefreshHealth}
                className="inline-flex min-h-[44px] items-center gap-2 rounded-full border border-danger-border bg-danger-tint px-3 py-1 text-[12px] font-semibold leading-[1.33] text-danger transition-colors hover:bg-danger/10"
              >
                <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span className="whitespace-nowrap">API Indisponível</span>
                {onRefreshHealth ? (
                  <RefreshCw className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                ) : null}
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1440px] px-4 py-6 lg:px-6 lg:pb-16">{children}</main>
    </div>
  );
}
