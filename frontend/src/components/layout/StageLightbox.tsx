import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useEffect, useRef } from "react";
import { cn } from "../../lib/cn.ts";

export interface LightboxStage {
  stepNumber: number;
  title: string;
  imageSrc: string;
  caption?: string;
}

export interface StageLightboxProps {
  stages: readonly LightboxStage[];
  activeIndex: number | null;
  onClose: () => void;
  onSelectIndex: (nextIndex: number) => void;
}

export function StageLightbox({ stages, activeIndex, onClose, onSelectIndex }: StageLightboxProps) {
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const isOpen = activeIndex !== null && activeIndex >= 0 && activeIndex < stages.length;

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    previousFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeButtonRef.current?.focus();

    return () => {
      previousFocusRef.current?.focus();
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || activeIndex === null) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      } else if (event.key === "ArrowLeft" && stages.length > 1) {
        event.preventDefault();
        onSelectIndex((activeIndex - 1 + stages.length) % stages.length);
      } else if (event.key === "ArrowRight" && stages.length > 1) {
        event.preventDefault();
        onSelectIndex((activeIndex + 1) % stages.length);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, activeIndex, stages.length, onClose, onSelectIndex]);

  if (!isOpen || activeIndex === null) {
    return null;
  }

  const currentStage = stages[activeIndex];
  const hasMultiple = stages.length > 1;

  const handlePrev = () => {
    onSelectIndex((activeIndex - 1 + stages.length) % stages.length);
  };

  const handleNext = () => {
    onSelectIndex((activeIndex + 1) % stages.length);
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="stage-lightbox-title"
      className="fixed inset-0 z-50 flex flex-col bg-viewport-dark/95 p-4 text-white lg:p-6"
    >
      {/* Top modal bar */}
      <div className="mx-auto flex w-full max-w-[1440px] flex-wrap items-center justify-between gap-4 border-b border-slate-700/80 pb-4">
        <div className="flex min-w-0 items-center gap-3">
          <span className="shrink-0 rounded-full border border-slate-600 bg-slate-800 px-2.5 py-1 text-[12px] font-semibold leading-[1.33] tabular-nums text-slate-200">
            Etapa {currentStage.stepNumber} / {stages.length}
          </span>
          <h2
            id="stage-lightbox-title"
            className="truncate text-[20px] font-semibold leading-[1.2] text-white"
          >
            {currentStage.title}
          </h2>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onClose}
            className="inline-flex min-h-[44px] items-center rounded-lg border border-slate-600 bg-slate-800 px-4 text-[14px] font-normal leading-[1.5] text-white transition-colors hover:bg-slate-700"
          >
            Fechar Ampliação
          </button>
          <button
            ref={closeButtonRef}
            type="button"
            onClick={onClose}
            aria-label="Fechar visualização ampliada"
            className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-lg border border-slate-600 bg-slate-800 text-white transition-colors hover:bg-slate-700"
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* Center dark CV image viewport */}
      <div className="relative mx-auto flex min-h-0 w-full max-w-[1440px] flex-1 items-center justify-center py-4">
        {hasMultiple ? (
          <button
            type="button"
            onClick={handlePrev}
            aria-label="Ver etapa anterior"
            className="z-10 mr-2 inline-flex min-h-[44px] min-w-[44px] shrink-0 items-center justify-center rounded-lg border border-slate-600 bg-slate-800/90 text-white transition-colors hover:bg-slate-700 lg:mr-4"
          >
            <ChevronLeft className="h-5 w-5" aria-hidden="true" />
          </button>
        ) : null}

        <div className="flex h-full max-h-[76vh] min-h-0 flex-1 items-center justify-center overflow-hidden rounded-xl border border-slate-700/80 bg-viewport-dark p-2">
          <img
            src={currentStage.imageSrc}
            alt={currentStage.title}
            className="max-h-full max-w-full object-contain"
          />
        </div>

        {hasMultiple ? (
          <button
            type="button"
            onClick={handleNext}
            aria-label="Ver próxima etapa"
            className="z-10 ml-2 inline-flex min-h-[44px] min-w-[44px] shrink-0 items-center justify-center rounded-lg border border-slate-600 bg-slate-800/90 text-white transition-colors hover:bg-slate-700 lg:ml-4"
          >
            <ChevronRight className="h-5 w-5" aria-hidden="true" />
          </button>
        ) : null}
      </div>

      {/* Caption & quick stage switcher */}
      <div className="mx-auto flex w-full max-w-[1440px] flex-col items-center gap-3 border-t border-slate-700/80 pt-4">
        {currentStage.caption ? (
          <p className="max-w-3xl text-center text-[14px] font-normal leading-[1.5] text-slate-300">
            {currentStage.caption}
          </p>
        ) : null}

        {hasMultiple ? (
          <div className="flex max-w-full items-center gap-2 overflow-x-auto pb-1">
            {stages.map((stage, idx) => {
              const selected = idx === activeIndex;
              return (
                <button
                  key={`${stage.stepNumber}-${stage.title}`}
                  type="button"
                  onClick={() => onSelectIndex(idx)}
                  aria-current={selected ? "step" : undefined}
                  className={cn(
                    "min-h-[44px] whitespace-nowrap rounded-lg border px-3 text-[12px] font-semibold leading-[1.33] tabular-nums transition-colors",
                    selected
                      ? "border-white bg-white text-viewport-dark"
                      : "border-slate-700 bg-slate-800/80 text-slate-200 hover:bg-slate-700",
                  )}
                >
                  {stage.stepNumber}. {stage.title}
                </button>
              );
            })}
          </div>
        ) : null}
      </div>
    </div>
  );
}
