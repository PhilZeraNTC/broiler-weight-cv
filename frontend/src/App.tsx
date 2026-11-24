import { useCallback, useEffect, useState } from "react";
import { AppShell, type TabId } from "./components/layout/AppShell.tsx";
import { InlineErrorBanner } from "./components/layout/InlineErrorBanner.tsx";
import { MetricsAuditTab } from "./components/metrics/MetricsAuditTab.tsx";
import { XRayStudioTab } from "./components/studio/XRayStudioTab.tsx";
import { FutureVisionTab } from "./components/vision/FutureVisionTab.tsx";
import { WatershedStudioTab } from "./components/watershed/WatershedStudioTab.tsx";
import { useHealth, usePresets } from "./hooks/useStudioHooks.ts";
import { cn } from "./lib/cn.ts";

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
  const [lastLatencyMs, setLastLatencyMs] = useState<number | null>(null);
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

  const singleBirdPresets = presets?.single_bird_presets ?? [];
  const watershedPresets = presets?.watershed_presets ?? [];

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

        {/* Tab 1: Estúdio Raio-X (kept mounted across tab switches to preserve analysis & simulated flock) */}
        <section
          role="tabpanel"
          id="panel-estudio"
          aria-labelledby="tab-estudio"
          hidden={activeTab !== "estudio"}
          className={cn(activeTab !== "estudio" && "hidden")}
        >
          <XRayStudioTab
            presets={singleBirdPresets}
            presetsLoading={presetsLoading}
            onLatency={setLastLatencyMs}
          />
        </section>

        {/* Tab 2: Separação Watershed (auto-loads ws_417 lazily on first activation, kept mounted thereafter) */}
        <section
          role="tabpanel"
          id="panel-watershed"
          aria-labelledby="tab-watershed"
          hidden={activeTab !== "watershed"}
          className={cn(activeTab !== "watershed" && "hidden")}
        >
          <WatershedStudioTab
            isActive={activeTab === "watershed"}
            presets={watershedPresets}
            presetsLoading={presetsLoading}
            onLatency={setLastLatencyMs}
          />
        </section>

        {/* Tab 3: Auditoria & Métricas (lazily fetches and caches GET /api/metrics via useMetrics) */}
        <section
          role="tabpanel"
          id="panel-metricas"
          aria-labelledby="tab-metricas"
          hidden={activeTab !== "metricas"}
          className={cn(activeTab !== "metricas" && "hidden")}
        >
          <MetricsAuditTab isActive={activeTab === "metricas"} />
        </section>

        {/* Tab 4: Visão Futura (v2) — 100% static conceptual documentation (PROIBIÇÃO 5) */}
        <section
          role="tabpanel"
          id="panel-visao-futura"
          aria-labelledby="tab-visao-futura"
          hidden={activeTab !== "visao-futura"}
          className={cn(activeTab !== "visao-futura" && "hidden")}
        >
          <FutureVisionTab />
        </section>
      </div>
    </AppShell>
  );
}
