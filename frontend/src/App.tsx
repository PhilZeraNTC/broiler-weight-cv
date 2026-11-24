import { useCallback, useEffect, useState } from "react";
import { AppShell, type TabId } from "./components/layout/AppShell.tsx";
import { InlineErrorBanner } from "./components/layout/InlineErrorBanner.tsx";
import { XRayStudioTab } from "./components/studio/XRayStudioTab.tsx";
import { FutureVisionTab } from "./components/vision/FutureVisionTab.tsx";
import { useHealth, usePresets } from "./hooks/useStudioHooks.ts";
import { cn } from "./lib/cn.ts";
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

        {/* Tab 2: Separação Watershed (Wave 4 mount slot) */}
        <section
          role="tabpanel"
          id="panel-watershed"
          aria-labelledby="tab-watershed"
          hidden={activeTab !== "watershed"}
          className={cn(activeTab !== "watershed" && "hidden")}
        >
          <div className="rounded-xl border border-border bg-surface p-6">
            <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
              Módulo Multi-Aves · Marker-based Watershed Clássico
            </p>
            <h1 className="mt-1 text-[20px] font-semibold leading-[1.2] text-ink">
              Separação Watershed
            </h1>
            <p className="mt-2 text-[14px] font-normal leading-[1.5] text-ink-muted">
              {presetsLoading
                ? "Carregando presets multi-aves..."
                : `${pluralizePtBr(watershedPresets.length, "cena curada disponível", "cenas curadas disponíveis")} para separação clássica via Transformada de Distância (EDT).`}
            </p>
          </div>
        </section>

        {/* Tab 3: Auditoria & Métricas (Wave 4 mount slot) */}
        <section
          role="tabpanel"
          id="panel-metricas"
          aria-labelledby="tab-metricas"
          hidden={activeTab !== "metricas"}
          className={cn(activeTab !== "metricas" && "hidden")}
        >
          <div className="rounded-xl border border-border bg-surface p-6">
            <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
              Validação Cruzada 5-Fold · Ablação Óptica C1–C4
            </p>
            <h1 className="mt-1 text-[20px] font-semibold leading-[1.2] text-ink">
              Auditoria &amp; Métricas
            </h1>
            <p className="mt-2 text-[14px] font-normal leading-[1.5] text-ink-muted">
              Painel científico de validação OOF pronto para conexão ao relatório de avaliação
              (1.738 imagens · 188 sessões ·{" "}
              {health ? `${health.n_features} atributos` : "39 atributos"}).
            </p>
          </div>
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
