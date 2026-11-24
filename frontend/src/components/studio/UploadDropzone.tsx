import { ImageUp, Upload } from "lucide-react";
import { type ChangeEvent, type DragEvent, type RefObject, useRef, useState } from "react";
import { ApiError, validateClientUploadFile } from "../../lib/api.ts";
import { cn } from "../../lib/cn.ts";
import type { SourceType } from "../../types/api.ts";

export interface UploadDropzoneProps {
  activeSourceType: SourceType;
  activePresetLabel: string | null;
  activeFileName: string | null;
  loading: boolean;
  fileInputRef?: RefObject<HTMLInputElement | null>;
  onSelectFile: (file: File) => void;
  onValidationError: (error: ApiError) => void;
}

const UNSUPPORTED_FALLBACK_MESSAGE =
  "Formato de imagem não suportado. Envie um arquivo JPG, PNG ou WEBP ou escolha um dos presets auditados na galeria.";

export function UploadDropzone({
  activeSourceType,
  activePresetLabel,
  activeFileName,
  loading,
  fileInputRef,
  onSelectFile,
  onValidationError,
}: UploadDropzoneProps) {
  const internalInputRef = useRef<HTMLInputElement | null>(null);
  const inputRef = fileInputRef ?? internalInputRef;
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const processCandidateFile = (file: File | null | undefined) => {
    if (!file) {
      return;
    }
    try {
      validateClientUploadFile(file);
      onSelectFile(file);
    } catch (err) {
      if (err instanceof ApiError) {
        onValidationError(err);
      } else {
        onValidationError(new ApiError(415, UNSUPPORTED_FALLBACK_MESSAGE));
      }
    }
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    processCandidateFile(file);
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (!loading) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    if (loading) {
      return;
    }
    const file = event.dataTransfer.files?.[0];
    processCandidateFile(file);
  };

  const triggerFilePicker = () => {
    if (!loading) {
      inputRef.current?.click();
    }
  };

  const isExternalUpload = activeSourceType === "upload" && activeFileName !== null;

  return (
    <section
      aria-label="Envio de foto local para inferência PDI"
      onDragEnter={handleDragOver}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "flex flex-col justify-between gap-4 rounded-xl border border-dashed bg-surface p-4 transition-colors sm:flex-row sm:items-center",
        isDragging ? "border-accent bg-accent-tint/50" : "border-border",
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
        onChange={handleInputChange}
        className="sr-only"
        aria-label="Selecionar foto de frango de corte (JPG, PNG ou WEBP até 15 MB)"
      />

      <div className="flex min-w-0 items-start gap-4">
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-border bg-surface-subtle text-ink-muted">
          <ImageUp className="h-5 w-5" aria-hidden="true" />
        </div>

        <div className="min-w-0 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-border bg-surface-subtle px-2 py-0.5 text-[12px] font-semibold leading-[1.33] text-ink-muted">
              {isExternalUpload ? "Upload Externo" : "Preset Auditado OOF"}
            </span>

            {isExternalUpload ? (
              <span
                title={activeFileName}
                className="max-w-[240px] truncate text-[12px] font-semibold leading-[1.33] text-ink"
              >
                {activeFileName}
              </span>
            ) : activePresetLabel ? (
              <span
                title={activePresetLabel}
                className="max-w-[240px] truncate text-[12px] font-semibold leading-[1.33] text-ink"
              >
                {activePresetLabel}
              </span>
            ) : null}
          </div>

          <p className="text-[14px] font-semibold leading-[1.5] text-ink">
            Envie uma foto de frango de corte
          </p>
          <p className="text-[14px] font-normal leading-[1.5] text-ink-muted">
            Arraste um arquivo JPG, PNG ou WEBP (até 15 MB) ou selecione um dos 6 presets auditados
            acima para inspecionar os 4 estágios do PDI clássico.
          </p>
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-2">
        <button
          type="button"
          disabled={loading}
          onClick={triggerFilePicker}
          className={cn(
            "inline-flex min-h-[44px] items-center justify-center gap-2 whitespace-nowrap rounded-lg px-4 text-[14px] font-normal leading-[1.5] transition-colors disabled:cursor-not-allowed disabled:opacity-60",
            isExternalUpload
              ? "border border-border bg-surface text-ink hover:bg-surface-subtle"
              : "bg-accent text-white hover:bg-accent-hover",
          )}
        >
          <Upload className="h-4 w-4 shrink-0" aria-hidden="true" />
          <span>{isExternalUpload ? "Carregar Nova Foto" : "Analisar Imagem"}</span>
        </button>
      </div>
    </section>
  );
}
