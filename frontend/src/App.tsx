import { useCallback, useEffect, useState } from "react";
import { AppShell, TAB_ITEMS, type TabId } from "./components/layout/AppShell.tsx";
import { InlineErrorBanner } from "./components/layout/InlineErrorBanner.tsx";
import { useHealth, usePresets } from "./hooks/useStudioHooks.ts";
import { pluralizePtBr } from "./lib/format.ts";

const VALID_HASH_TO_TAB: Readonly<Record<string, TabId>> = {
  "#estudio": "estudio",
  "#watershed": "watershed",
  "#metricas": "metricas",
  "#visao-futura": "visao-futura",
};

function parseTabFromHash(hash: string): TabId {
  return VALID_HASH_TO_TAB[hash] ?? "estudio";
}

function getInitialTab(): TabId {
  if (typeof window === "undefined") {
    return "estudio";
  }
  return parseTabFromHash(window.location.hash);
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>(getInitialTab);
  const [lastLatencyMs] = useState<number | null>(null);
  const { health, status: healthStatus, refresh: refreshHealth } = useHealth();
  const {
    presets,
    loading: presetsLoading,
    error: presetsError,
    retry: retryPresets,
  } = usePresets();

  const handleSelectTab = useCallback((nextTab: TabId) => {
    setActiveTab(nextTab);
    if (typeof window !== "undefined") {
      const targetHash = `#${nextTab}`;
      if (window.location.hash !== targetHash) {
        window.location.hash = targetHash;
      }
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const handleHashChange = () => {
      setActiveTab(parseTabFromHash(window.location.hash));
    };
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  const currentTabConfig = TAB_ITEMS.find((item) => item.id === activeTab) ?? TAB_ITEMS[0];

  return (
    <AppShell
      activeTab={activeTab}
      onSelectTab={handleSelectTab}
      health={health}
      healthStatus={healthStatus}
      lastLatencyMs={lastLatencyMs}
      onRefreshHealth={refreshHealth}
    >
      <div className="space-y-6">
        {presetsError ? (
          <InlineErrorBanner
            title="Catálogo de Presets Indisponível"
            message={presetsError.userMessage}
            actionLabel="Tentar Análise Novamente"
            onAction={() => {
              refreshHealth();
              retryPresets();
            }}
          />
        ) : null}

        <section
          role="tabpanel"
          id={`panel-${activeTab}`}
          aria-labelledby={`tab-${activeTab}`}
          className="rounded-xl border border-border bg-surface p-6"
        >
          <div className="flex flex-wrap items-start justify-between gap-4 border-b border-border pb-4">
            <div className="space-y-1">
              <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
                AvianScale CV · Módulo Ativo
              </p>
              <h1 className="text-[20px] font-semibold leading-[1.2] text-ink">
                {currentTabConfig.label}
              </h1>
            </div>
            {health ? (
              <span className="rounded-full border border-border bg-surface-subtle px-3 py-1 font-mono text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
                {health.pdi_config_hash}
              </span>
            ) : null}
          </div>

          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div className="rounded-lg border border-border bg-canvas p-4">
              <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
                Presets de Ave Única (OOF)
              </p>
              <p className="mt-1 text-[32px] font-semibold leading-[1.2] tabular-nums text-ink">
                {presetsLoading
                  ? "—"
                  : pluralizePtBr(presets?.single_bird_presets.length ?? 0, "preset", "presets")}
              </p>
            </div>

            <div className="rounded-lg border border-border bg-canvas p-4">
              <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
                Presets Watershed Multi-Aves
              </p>
              <p className="mt-1 text-[32px] font-semibold leading-[1.2] tabular-nums text-ink">
                {presetsLoading
                  ? "—"
                  : pluralizePtBr(presets?.watershed_presets.length ?? 0, "cena", "cenas")}
              </p>
            </div>

            <div className="rounded-lg border border-border bg-canvas p-4">
              <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
                Atributos PDI Auditados
              </p>
              <p className="mt-1 text-[32px] font-semibold leading-[1.2] tabular-nums text-ink">
                {health ? pluralizePtBr(health.n_features, "atributo", "atributos") : "—"}
              </p>
            </div>
          </div>
        </section>
      </div>
    </AppShell>
  );
}
