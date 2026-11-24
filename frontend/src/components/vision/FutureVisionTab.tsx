import { AlertTriangle, Camera, Cpu, Layers } from "lucide-react";

interface ComparisonRow {
  axis: string;
  v1: string;
  v2: string;
}

interface RoadmapCardItem {
  code: string;
  title: string;
  summary: string;
  foundationNote: string;
  icon: "camera" | "layers" | "cpu";
}

const COMPARISON_ROWS: readonly ComparisonRow[] = [
  {
    axis: "Geometria de Captura",
    v1: "Smartphone handheld (Xiaomi Redmi 10) em balde plástico dorsal, balança SF-400 e cama de palha sob ângulos variáveis.",
    v2: "Câmera fixa perpendicular (top-down) montada no teto do aviário a altura constante conhecida (Z_const), eliminando variações de perspectiva.",
  },
  {
    axis: "Calibração Espacial",
    v1: "Calibração óptica EXIF (FocalLength + Xiaomi tag 39321) normalizando área projetada via area_px / zoom² sem usar ID da lente.",
    v2: "Parâmetros intrínsecos da câmera fixa + Z_const convertendo diretamente cada silhueta segmentada para área física real em cm².",
  },
  {
    axis: "Separação & Amostragem",
    v1: "Inferência unitária por foto + módulo demonstrativo Marker-based Watershed em cenas estáticas multi-aves (417, 757, 823, 1520).",
    v2: "Amostragem contínua automatizada combinando Watershed clássico com filtros de postura e agregação estatística por lote ao longo do dia.",
  },
  {
    axis: "Faixa de Peso e Idade",
    v1: "Validado rigorosamente em 116–2.093 g (~7–29 dias, Cobb Avian 48), divulgando de forma transparente a lacuna 801–1.216 g.",
    v2: "Coleta calibrada estendida para cobrir integralmente a fase comercial de terminação (2,1–3,2 kg · 30–45 dias até o abate).",
  },
  {
    axis: "Infraestrutura Edge CPU-Only",
    v1: "API FastAPI local em CPU (< 500 ms; p95 PDI 88,3 ms) + SPA React/Vite para validação científica e demonstração interativa.",
    v2: "Empacotamento embarcado CPU-Only (Raspberry Pi / Smart Camera industrial) executando PDI clássico leve sem GPU diretamente no galpão.",
  },
] as const;

const ROADMAP_CARDS: readonly RoadmapCardItem[] = [
  {
    code: "BARN-01",
    title: "Calibração Intrínseca de Teto & Rastreamento Multi-Frame",
    summary:
      "Integra câmeras superiores fixas com altura calibrada (Z_const) para acompanhar lotes ao longo do ciclo produtivo, reduzindo a variância de captura manual e eliminando o estresse da pesagem por amostragem.",
    foundationNote: "Base v1 reutilizada: Pré-processamento CLAHE-Lab + Watershed Clássico",
    icon: "camera",
  },
  {
    code: "BARN-02",
    title: "Expansão Comercial 30–45 Dias & Mapa de Densidade Espacial",
    summary:
      "Estende a calibração alométrica para a faixa de terminação (2,1–3,2 kg) e agrega posições de centroides sobre a planta do galpão para evidenciar aglomerações associadas a conforto térmico e manejo de comedouros.",
    foundationNote: "Base v1 reutilizada: Segmentação Multi-Espaço + Morfometria Mask-Gated",
    icon: "layers",
  },
  {
    code: "BARN-03",
    title: "Empacotamento Edge CPU-Only & Alerta Sanitário de Aves Estacionárias",
    summary:
      "Capitaliza a leveza computacional do pipeline clássico (sem redes neurais pesadas) para execução contínua em hardware de baixo custo, sinalizando padrões prolongados de imobilidade para inspeção veterinária precoce.",
    foundationNote: "Base v1 reutilizada: Motor OpenCV/scikit-image + LightGBM (< 100 ms CPU)",
    icon: "cpu",
  },
] as const;

function RoadmapIcon({ kind }: { kind: RoadmapCardItem["icon"] }) {
  if (kind === "camera") {
    return <Camera className="h-5 w-5 shrink-0 text-ink" aria-hidden="true" />;
  }
  if (kind === "layers") {
    return <Layers className="h-5 w-5 shrink-0 text-ink" aria-hidden="true" />;
  }
  return <Cpu className="h-5 w-5 shrink-0 text-ink" aria-hidden="true" />;
}

/**
 * Static conceptual roadmap tab (WEB-04 / D-12 / PROIBIÇÃO 5).
 * Strictly zero API requests, zero timers, zero live video, and zero mock telemetry.
 */
export function FutureVisionTab() {
  return (
    <div className="space-y-8">
      {/* Permanent full-width Amber PROIBIÇÃO 5 governance banner (verbatim 04-UI-SPEC.md:248) */}
      <div
        role="note"
        aria-label="Aviso de escopo futuro v2"
        className="flex items-start gap-4 rounded-xl border border-warning-border bg-warning-tint p-6 text-warning"
      >
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide">
            Governança de Escopo · Documentação Conceitual Estática
          </p>
          <p className="text-[14px] font-normal leading-[1.5]">
            Escopo Futuro v2 — Arquitetura Conceitual (Não Implementado no MVP v1). Rastreamento
            multi-frame, mapas de densidade e alertas sanitários não possuem backend ou telemetria
            ativa nesta versão (PROIBIÇÃO 5).
          </p>
        </div>
      </div>

      {/* Section 1: 2-Column MVP v1 vs. Arquitetura v2 Comparison Matrix across 5 engineering axes */}
      <section
        aria-label="Matriz comparativa MVP v1 versus Arquitetura v2"
        className="rounded-xl border border-border bg-surface p-6"
      >
        <div className="flex flex-wrap items-start justify-between gap-4 pb-4">
          <div className="space-y-1">
            <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
              Evolução de Engenharia · Do MVP Auditado ao Aviário Comercial
            </p>
            <h1 className="text-[20px] font-semibold leading-[1.2] text-ink">
              MVP v1 (Smartphone Handheld / Balde) vs. Arquitetura v2 (Câmera Fixa de Teto)
            </h1>
          </div>

          <span className="rounded-full border border-warning-border bg-warning-tint px-2 py-1 text-[12px] font-semibold leading-[1.33] text-warning">
            Roadmap Conceitual v2
          </span>
        </div>

        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full min-w-[680px] border-collapse text-left">
            <thead className="bg-surface-subtle">
              <tr className="border-b border-border text-[12px] font-semibold leading-[1.33] text-ink-muted">
                <th scope="col" className="w-48 px-4 py-4">
                  Eixo de Engenharia
                </th>
                <th scope="col" className="px-4 py-4">
                  MVP v1 — Validado e Operacional (Atual)
                </th>
                <th scope="col" className="px-4 py-4">
                  Arquitetura v2 — Câmera Fixa de Teto (Conceitual)
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border bg-surface">
              {COMPARISON_ROWS.map((row) => (
                <tr key={row.axis} className="align-top hover:bg-canvas">
                  <th
                    scope="row"
                    className="px-4 py-4 text-[14px] font-semibold leading-[1.5] text-ink"
                  >
                    {row.axis}
                  </th>
                  <td className="px-4 py-4 text-[14px] font-normal leading-[1.5] text-ink">
                    {row.v1}
                  </td>
                  <td className="px-4 py-4 text-[14px] font-normal leading-[1.5] text-ink-muted">
                    {row.v2}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Section 2: 3 Conceptual Roadmap Cards (BARN-01, BARN-02, BARN-03) */}
      <section aria-label="Módulos conceituais do roadmap v2" className="space-y-4">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Extensões Pós-MVP · Continuidade sobre PDI Clássico Interpretável
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            Três Pilares da Arquitetura v2 em Aviário Comercial
          </h2>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {ROADMAP_CARDS.map((card) => (
            <article
              key={card.code}
              className="flex flex-col justify-between gap-4 rounded-xl border border-border bg-surface p-6"
            >
              <div className="space-y-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border bg-surface-subtle">
                      <RoadmapIcon kind={card.icon} />
                    </div>
                    <span className="font-mono text-[12px] font-semibold leading-[1.33] tabular-nums text-ink">
                      {card.code}
                    </span>
                  </div>

                  <span className="rounded-full border border-warning-border bg-warning-tint px-2 py-1 text-[12px] font-semibold leading-[1.33] text-warning">
                    Conceitual v2
                  </span>
                </div>

                <div className="space-y-2">
                  <h3 className="text-[20px] font-semibold leading-[1.2] text-ink">{card.title}</h3>
                  <p className="text-[14px] font-normal leading-[1.5] text-ink-muted">
                    {card.summary}
                  </p>
                </div>
              </div>

              <div className="rounded-lg border border-border bg-canvas px-4 py-2 text-[12px] font-semibold leading-[1.33] text-ink-muted">
                {card.foundationNote}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
