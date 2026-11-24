import {
  type Dispatch,
  type SetStateAction,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ApiError, fetchHealth, fetchMetrics, fetchPresets, isAbortError } from "../lib/api.ts";
import { computeFlockStats, type FlockEntry, type FlockStats } from "../lib/flock.ts";
import type { HealthResponse, MetricsResponse, PresetsResponse } from "../types/api.ts";

const FALLBACK_NETWORK_ERROR_MESSAGE =
  "Não foi possível conectar à API local FastAPI em 127.0.0.1:8000. Verifique se o servidor Uvicorn está ativo no terminal e clique em 'Tentar Análise Novamente'.";

const FLOCK_SESSION_STORAGE_KEY = "avianscale_cv_flock_batch_v1";

function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) {
    return error;
  }
  return new ApiError(0, FALLBACK_NETWORK_ERROR_MESSAGE);
}

export interface AbortableRequestResult<T> {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
  run: (executor: (signal: AbortSignal) => Promise<T>) => Promise<T | null>;
  setData: Dispatch<SetStateAction<T | null>>;
  clearError: () => void;
}

/**
 * Shared abort-safe async request hook.
 * Cancels any prior in-flight `AbortController` before firing a new request,
 * ignores `AbortError` cleanly without flashing error banners, and aborts on unmount.
 */
export function useAbortableRequest<T>(): AbortableRequestResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<ApiError | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const seqRef = useRef<number>(0);

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const run = useCallback(async (executor: (signal: AbortSignal) => Promise<T>) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const seq = ++seqRef.current;

    setLoading(true);
    setError(null);

    try {
      const result = await executor(controller.signal);
      if (seq !== seqRef.current) {
        return null;
      }
      setData(result);
      setLoading(false);
      return result;
    } catch (err) {
      if (isAbortError(err) || seq !== seqRef.current) {
        if (seq === seqRef.current) {
          setLoading(false);
        }
        return null;
      }
      setError(toApiError(err));
      setLoading(false);
      return null;
    }
  }, []);

  return {
    data,
    loading,
    error,
    run,
    setData,
    clearError,
  };
}

export type HealthStatus = "checking" | "online" | "offline";

export interface UseHealthResult {
  health: HealthResponse | null;
  status: HealthStatus;
  refresh: () => void;
}

/** Polls `GET /api/health` on mount and exposes manual refresh for the header status pill. */
export function useHealth(): UseHealthResult {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [status, setStatus] = useState<HealthStatus>("checking");
  const controllerRef = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setStatus((prev) => (prev === "online" ? "online" : "checking"));

    fetchHealth(controller.signal)
      .then((payload) => {
        if (!controller.signal.aborted) {
          setHealth(payload);
          setStatus("online");
        }
      })
      .catch((err) => {
        if (isAbortError(err) || controller.signal.aborted) {
          return;
        }
        setHealth(null);
        setStatus("offline");
      });
  }, []);

  useEffect(() => {
    refresh();
    return () => {
      controllerRef.current?.abort();
    };
  }, [refresh]);

  return { health, status, refresh };
}

export interface UsePresetsResult {
  presets: PresetsResponse | null;
  loading: boolean;
  error: ApiError | null;
  retry: () => void;
}

/** Fetches the curated `GET /api/presets` catalog once on mount with retry support. */
export function usePresets(): UsePresetsResult {
  const { data, loading, error, run } = useAbortableRequest<PresetsResponse>();

  const load = useCallback(() => {
    void run((signal) => fetchPresets(signal));
  }, [run]);

  useEffect(() => {
    load();
  }, [load]);

  return {
    presets: data,
    loading,
    error,
    retry: load,
  };
}

export interface UseMetricsResult {
  data: MetricsResponse | null;
  loading: boolean;
  error: ApiError | null;
  retry: () => void;
}

/**
 * Lazily fetches `GET /api/metrics` the first time `enabled` becomes `true`
 * and caches the response in memory across subsequent tab visits.
 * React 19 `<StrictMode>`-safe: latches only after a non-aborted response resolves.
 */
export function useMetrics(enabled: boolean): UseMetricsResult {
  const { data, loading, error, run } = useAbortableRequest<MetricsResponse>();
  const resolvedRef = useRef<boolean>(false);

  const load = useCallback(() => {
    void run((signal) => fetchMetrics(signal)).then((result) => {
      if (result !== null) {
        resolvedRef.current = true;
      }
    });
  }, [run]);

  useEffect(() => {
    if (enabled && !resolvedRef.current && data === null) {
      load();
    }
  }, [enabled, data, load]);

  return {
    data,
    loading,
    error,
    retry: load,
  };
}

export type NewFlockEntryInput = Omit<FlockEntry, "id" | "addedAtIso"> & {
  id?: string;
  addedAtIso?: string;
};

export interface UseFlockBatchResult {
  entries: FlockEntry[];
  stats: FlockStats;
  addEntry: (entry: NewFlockEntryInput) => void;
  removeEntry: (id: string) => void;
  clearEntries: () => void;
}

function isStoredFlockEntry(candidate: unknown): candidate is FlockEntry {
  if (typeof candidate !== "object" || candidate === null) {
    return false;
  }
  const obj = candidate as Record<string, unknown>;
  return (
    typeof obj.id === "string" &&
    typeof obj.label === "string" &&
    typeof obj.weightG === "number" &&
    Number.isFinite(obj.weightG) &&
    (obj.sourceType === "preset" || obj.sourceType === "upload") &&
    (obj.presetId === null || typeof obj.presetId === "string") &&
    typeof obj.ageStageLabel === "string" &&
    typeof obj.addedAtIso === "string"
  );
}

function loadFlockFromSessionStorage(): FlockEntry[] {
  if (typeof window === "undefined") {
    return [];
  }
  try {
    const raw = window.sessionStorage.getItem(FLOCK_SESSION_STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter(isStoredFlockEntry);
  } catch {
    return [];
  }
}

function createEntryId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `flock-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Manages the stateless client-side simulated flock accumulator (`WEB-02`),
 * syncing safely with `sessionStorage` and computing live uniformity (`CV%`).
 */
export function useFlockBatch(): UseFlockBatchResult {
  const [entries, setEntries] = useState<FlockEntry[]>(loadFlockFromSessionStorage);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      window.sessionStorage.setItem(FLOCK_SESSION_STORAGE_KEY, JSON.stringify(entries));
    } catch {
      // Ignore storage quota or private-mode restrictions cleanly
    }
  }, [entries]);

  const stats = useMemo(() => computeFlockStats(entries), [entries]);

  const addEntry = useCallback((input: NewFlockEntryInput) => {
    const nextEntry: FlockEntry = {
      id: input.id ?? createEntryId(),
      label: input.label,
      weightG: input.weightG,
      sourceType: input.sourceType,
      presetId: input.presetId,
      ageStageLabel: input.ageStageLabel,
      addedAtIso: input.addedAtIso ?? new Date().toISOString(),
    };
    setEntries((prev) => [...prev, nextEntry]);
  }, []);

  const removeEntry = useCallback((id: string) => {
    setEntries((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const clearEntries = useCallback(() => {
    setEntries([]);
  }, []);

  return {
    entries,
    stats,
    addEntry,
    removeEntry,
    clearEntries,
  };
}
