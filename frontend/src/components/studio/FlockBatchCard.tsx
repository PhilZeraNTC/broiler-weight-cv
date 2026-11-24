import { Plus, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import type { FlockEntry, FlockStats } from "../../lib/flock.ts";
import { formatGrams, formatPercent, pluralizePtBr } from "../../lib/format.ts";

export interface FlockBatchCardProps {
  entries: readonly FlockEntry[];
  stats: FlockStats;
  canAddCurrent: boolean;
  onAddCurrent: () => void;
  onRemoveEntry: (id: string) => void;
  onClearEntries: () => void;
}

export function FlockBatchCard({
  entries,
  stats,
  canAddCurrent,
  onAddCurrent,
  onRemoveEntry,
  onClearEntries,
}: FlockBatchCardProps) {
  const [confirmingClear, setConfirmingClear] = useState<boolean>(false);

  useEffect(() => {
    if (entries.length < 2 && confirmingClear) {
      setConfirmingClear(false);
    }
  }, [entries.length, confirmingClear]);

  const handleClearRequest = () => {
    if (entries.length >= 2) {
      setConfirmingClear(true);
      return;
    }
    onClearEntries();
  };

  const handleConfirmClear = () => {
    setConfirmingClear(false);
    onClearEntries();
  };

  return (
    <section
      aria-label="Simulador de lote com peso médio e uniformidade"
      className="flex flex-col space-y-4 rounded-xl border border-border bg-surface p-4 lg:p-6"
    >
      {/* Header & actions (including inline 2-step destructive confirmation at >= 2 birds) */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <p className="text-[12px] font-semibold leading-[1.33] uppercase tracking-wide text-ink-muted">
            Acumulador Local da Sessão · Uniformidade de Lote
          </p>
          <h2 className="text-[20px] font-semibold leading-[1.2] text-ink">
            Simulador de Lote (Média e CV%)
          </h2>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={!canAddCurrent}
            onClick={onAddCurrent}
            className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-lg bg-accent px-4 text-[14px] font-normal leading-[1.5] text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-60"
          >
            <Plus className="h-4 w-4 shrink-0" aria-hidden="true" />
            <span>Adicionar Ave ao Lote</span>
          </button>

          {entries.length > 0 ? (
            confirmingClear && entries.length >= 2 ? (
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={handleConfirmClear}
                  className="inline-flex min-h-[44px] items-center justify-center rounded-lg bg-danger px-4 text-[14px] font-normal leading-[1.5] text-white transition-colors hover:bg-danger/90"
                >
                  Confirmar Limpeza do Lote
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmingClear(false)}
                  className="inline-flex min-h-[44px] items-center justify-center rounded-lg border border-border bg-surface px-4 text-[14px] font-normal leading-[1.5] text-ink transition-colors hover:bg-surface-subtle"
                >
                  Manter Lote Atual
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={handleClearRequest}
                className="inline-flex min-h-[44px] items-center justify-center rounded-lg border border-border bg-surface px-4 text-[14px] font-normal leading-[1.5] text-ink transition-colors hover:bg-surface-subtle"
              >
                Limpar Lote Simulado
              </button>
            )
          ) : null}
        </div>
      </div>

      {/* 0-bird Empty State verbatim from 04-UI-SPEC.md:217 */}
      {entries.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center rounded-lg border border-dashed border-border bg-canvas p-6 text-center">
          <p className="text-[14px] font-semibold leading-[1.5] text-ink">
            Nenhuma ave acumulada no lote simulado
          </p>
          <p className="mt-2 max-w-md text-[14px] font-normal leading-[1.5] text-ink-muted">
            Clique em 'Adicionar Ave ao Lote' após executar uma análise para acompanhar o peso médio
            e a uniformidade (CV%) do lote.
          </p>
        </div>
      ) : (
        <>
          {/* Batch summary KPI strip */}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <div className="rounded-lg border border-border bg-canvas p-4">
              <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">Amostra</p>
              <p className="mt-1 text-[20px] font-semibold leading-[1.2] tabular-nums text-ink">
                {pluralizePtBr(stats.count, "ave", "aves")}
              </p>
            </div>

            <div className="rounded-lg border border-border bg-canvas p-4">
              <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
                Peso Médio (g)
              </p>
              <p className="mt-1 text-[20px] font-semibold leading-[1.2] tabular-nums text-ink">
                {formatGrams(stats.meanG, 1)}
              </p>
            </div>

            <div className="rounded-lg border border-border bg-canvas p-4">
              <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
                Desvio Padrão (g)
              </p>
              <p className="mt-1 text-[20px] font-semibold leading-[1.2] tabular-nums text-ink">
                {stats.stdDevG !== null ? formatGrams(stats.stdDevG, 1) : "—"}
              </p>
              {stats.count === 1 ? (
                <p className="mt-1 text-[12px] font-semibold leading-[1.33] text-ink-muted">
                  Adicione ≥ 2 aves
                </p>
              ) : null}
            </div>

            <div className="rounded-lg border border-border bg-canvas p-4">
              <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
                Uniformidade (CV%)
              </p>
              <p className="mt-1 text-[20px] font-semibold leading-[1.2] tabular-nums text-ink">
                {stats.cvPct !== null ? formatPercent(stats.cvPct, 1) : "—"}
              </p>
              {stats.count === 1 ? (
                <p className="mt-1 text-[12px] font-semibold leading-[1.33] text-ink-muted">
                  Adicione ≥ 2 aves
                </p>
              ) : null}
            </div>
          </div>

          {/* Scrollable bird list (overflow-y-auto capped at max-h-72) */}
          <ul className="max-h-72 divide-y divide-border overflow-y-auto rounded-lg border border-border bg-canvas">
            {entries.map((entry, idx) => {
              const displayIndex = idx + 1;
              return (
                <li key={entry.id} className="flex items-center justify-between gap-2 px-4 py-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="shrink-0 rounded bg-surface-subtle px-2 py-0.5 font-mono text-[12px] font-semibold leading-[1.33] tabular-nums text-ink-muted">
                      #{displayIndex}
                    </span>
                    <div className="min-w-0">
                      <p
                        title={entry.label}
                        className="max-w-[240px] truncate text-[14px] font-normal leading-[1.5] text-ink"
                      >
                        {entry.label}
                      </p>
                      <p className="text-[12px] font-semibold leading-[1.33] text-ink-muted">
                        {entry.ageStageLabel} ·{" "}
                        {entry.sourceType === "preset" ? "Preset OOF" : "Upload Externo"}
                      </p>
                    </div>
                  </div>

                  <div className="flex shrink-0 items-center gap-2">
                    <span className="text-[14px] font-semibold leading-[1.5] tabular-nums text-ink">
                      {formatGrams(entry.weightG, 1)}
                    </span>
                    <button
                      type="button"
                      onClick={() => onRemoveEntry(entry.id)}
                      aria-label={`Remover ave #${displayIndex} do lote simulado`}
                      className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-lg text-ink-muted transition-colors hover:bg-danger-tint hover:text-danger"
                    >
                      <Trash2 className="h-4 w-4" aria-hidden="true" />
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </section>
  );
}
