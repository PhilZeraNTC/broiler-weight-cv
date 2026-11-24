import type {
  HealthResponse,
  MetricsResponse,
  PredictResponse,
  PresetsResponse,
  WatershedResponse,
} from "../types/api.ts";

/** Maximum client-side upload size mirroring backend `MAX_UPLOAD_BYTES` (15 MiB). */
export const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;

export const ALLOWED_UPLOAD_MIMES: ReadonlySet<string> = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);

export const ALLOWED_UPLOAD_EXTENSIONS: ReadonlySet<string> = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
]);

const PT_BR_ERROR_COPY = {
  badImage400:
    "Não foi possível decodificar esta imagem com segurança. Verifique se o arquivo não está corrompido e clique em 'Carregar Nova Foto'.",
  notFound404:
    "Preset auditado ou recurso não encontrado na API local. Selecione um preset disponível na galeria e tente novamente.",
  oversize413:
    "O arquivo ultrapassa o limite local de 15 MB. Reduza a resolução da foto ou selecione um preset da galeria para testar o pipeline.",
  unsupported415:
    "Formato de imagem não suportado. Envie um arquivo JPG, PNG ou WEBP ou escolha um dos presets auditados na galeria.",
  validation422:
    "Um ou mais parâmetros estão fora dos limites aceitos (distância focal 0,5–200 mm; zoom 0,1–20×; limites do Watershed). Ajuste o campo destacado ou clique em 'Restaurar EXIF Original' (Aba 1) / 'Restaurar Parâmetros Padrão' (Aba 2).",
  offlineOrServer:
    "Não foi possível conectar à API local FastAPI em 127.0.0.1:8000. Verifique se o servidor Uvicorn está ativo no terminal e clique em 'Tentar Análise Novamente'.",
} as const;

export class ApiError extends Error {
  readonly status: number;
  readonly userMessage: string;

  constructor(status: number, userMessage: string) {
    super(userMessage);
    this.name = "ApiError";
    this.status = status;
    this.userMessage = userMessage;
  }
}

export function isAbortError(error: unknown): boolean {
  if (error instanceof DOMException && error.name === "AbortError") {
    return true;
  }
  if (error instanceof Error && error.name === "AbortError") {
    return true;
  }
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    (error as { name?: unknown }).name === "AbortError"
  );
}

function mapStatusToPtBrMessage(status: number): string {
  switch (status) {
    case 400:
      return PT_BR_ERROR_COPY.badImage400;
    case 404:
      return PT_BR_ERROR_COPY.notFound404;
    case 413:
      return PT_BR_ERROR_COPY.oversize413;
    case 415:
      return PT_BR_ERROR_COPY.unsupported415;
    case 422:
      return PT_BR_ERROR_COPY.validation422;
    default:
      return PT_BR_ERROR_COPY.offlineOrServer;
  }
}

function extractExtension(filename: string): string {
  const lastDot = filename.lastIndexOf(".");
  if (lastDot < 0) return "";
  return filename.slice(lastDot).toLowerCase();
}

/**
 * Client-side pre-flight guard matching server upload limits (`15 MiB`, JPG/PNG/WEBP).
 * Throws a sanitized `ApiError` (`400`, `413`, or `415`) before touching `FormData`.
 */
export function validateClientUploadFile(file: File): void {
  if (file.size <= 0) {
    throw new ApiError(400, PT_BR_ERROR_COPY.badImage400);
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new ApiError(413, PT_BR_ERROR_COPY.oversize413);
  }

  const normalizedMime = file.type.trim().toLowerCase();
  if (normalizedMime) {
    if (!ALLOWED_UPLOAD_MIMES.has(normalizedMime)) {
      throw new ApiError(415, PT_BR_ERROR_COPY.unsupported415);
    }
    return;
  }

  const ext = extractExtension(file.name);
  if (!ALLOWED_UPLOAD_EXTENSIONS.has(ext)) {
    throw new ApiError(415, PT_BR_ERROR_COPY.unsupported415);
  }
}

function appendFiniteFormNumber(formData: FormData, key: string, value: number | undefined): void {
  if (typeof value === "number" && Number.isFinite(value)) {
    formData.append(key, String(value));
  }
}

async function requestJson<T>(url: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (err) {
    if (isAbortError(err)) {
      throw err;
    }
    throw new ApiError(0, PT_BR_ERROR_COPY.offlineOrServer);
  }

  if (!response.ok) {
    throw new ApiError(response.status, mapStatusToPtBrMessage(response.status));
  }

  try {
    return (await response.json()) as T;
  } catch (err) {
    if (isAbortError(err)) {
      throw err;
    }
    throw new ApiError(500, PT_BR_ERROR_COPY.offlineOrServer);
  }
}

export interface PredictRequestParams {
  presetId?: string;
  file?: File;
  focalLengthMm?: number;
  zoomMultiple?: number;
}

export interface WatershedRequestParams {
  presetId?: string;
  file?: File;
  hRel?: number;
  minDistanceRel?: number;
  minAreaRatio?: number;
  focalLengthMm?: number;
  zoomMultiple?: number;
}

export function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/api/health", {
    method: "GET",
    signal,
  });
}

export function fetchPresets(signal?: AbortSignal): Promise<PresetsResponse> {
  return requestJson<PresetsResponse>("/api/presets", {
    method: "GET",
    signal,
  });
}

export function fetchMetrics(signal?: AbortSignal): Promise<MetricsResponse> {
  return requestJson<MetricsResponse>("/api/metrics", {
    method: "GET",
    signal,
  });
}

export function predictSingleBird(
  params: PredictRequestParams,
  signal?: AbortSignal,
): Promise<PredictResponse> {
  const formData = new FormData();

  if (params.file) {
    validateClientUploadFile(params.file);
    formData.append("file", params.file);
  } else if (params.presetId?.trim()) {
    formData.append("preset_id", params.presetId.trim());
  }

  appendFiniteFormNumber(formData, "focal_length_mm", params.focalLengthMm);
  appendFiniteFormNumber(formData, "zoom_multiple", params.zoomMultiple);

  return requestJson<PredictResponse>("/api/predict", {
    method: "POST",
    body: formData,
    signal,
  });
}

export function runWatershedDemo(
  params: WatershedRequestParams,
  signal?: AbortSignal,
): Promise<WatershedResponse> {
  const formData = new FormData();

  if (params.file) {
    validateClientUploadFile(params.file);
    formData.append("file", params.file);
  } else if (params.presetId?.trim()) {
    formData.append("preset_id", params.presetId.trim());
  }

  appendFiniteFormNumber(formData, "h_rel", params.hRel);
  appendFiniteFormNumber(formData, "min_distance_rel", params.minDistanceRel);
  appendFiniteFormNumber(formData, "min_area_ratio", params.minAreaRatio);
  appendFiniteFormNumber(formData, "focal_length_mm", params.focalLengthMm);
  appendFiniteFormNumber(formData, "zoom_multiple", params.zoomMultiple);

  return requestJson<WatershedResponse>("/api/watershed-demo", {
    method: "POST",
    body: formData,
    signal,
  });
}
