/**
 * Brazilian Portuguese (`pt-BR`) numeric, weight, percentage, nullable (`—`),
 * and singular/plural formatting helpers for AvianScale CV telemetry.
 */

const EM_DASH = "—";
const numberFormatterCache = new Map<number, Intl.NumberFormat>();
const integerFormatter = new Intl.NumberFormat("pt-BR", {
  maximumFractionDigits: 0,
});

function getNumberFormatter(decimals: number): Intl.NumberFormat {
  const safeDecimals = Math.max(0, Math.min(8, Math.trunc(decimals)));
  let formatter = numberFormatterCache.get(safeDecimals);
  if (!formatter) {
    formatter = new Intl.NumberFormat("pt-BR", {
      minimumFractionDigits: safeDecimals,
      maximumFractionDigits: safeDecimals,
    });
    numberFormatterCache.set(safeDecimals, formatter);
  }
  return formatter;
}

function isValidNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Format a finite number in `pt-BR` locale, returning `"—"` on nullish/non-finite inputs. */
export function formatNumberPtBr(value: number | null | undefined, decimals = 2): string {
  if (!isValidNumber(value)) {
    return EM_DASH;
  }
  const normalized = Object.is(value, -0) ? 0 : value;
  return getNumberFormatter(decimals).format(normalized);
}

/** Format weight in grams (`g`) using `pt-BR` locale, returning `"—"` on invalid inputs. */
export function formatGrams(value: number | null | undefined, decimals = 1): string {
  if (!isValidNumber(value)) {
    return EM_DASH;
  }
  return `${formatNumberPtBr(value, decimals)} g`;
}

/** Format weight already expressed in kilograms (`kg`) using `pt-BR` locale. */
export function formatKg(kg: number | null | undefined, decimals = 2): string {
  if (!isValidNumber(kg)) {
    return EM_DASH;
  }
  return `${formatNumberPtBr(kg, decimals)} kg`;
}

/** Format percentage value (`%`) using `pt-BR` locale, returning `"—"` on invalid inputs. */
export function formatPercent(value: number | null | undefined, decimals = 1): string {
  if (!isValidNumber(value)) {
    return EM_DASH;
  }
  return `${formatNumberPtBr(value, decimals)}%`;
}

/** Format a nullable scientific metric with optional unit suffix, returning `"—"` on invalid inputs. */
export function formatNullableMetric(
  value: number | null | undefined,
  decimals = 2,
  suffix = "",
): string {
  if (!isValidNumber(value)) {
    return EM_DASH;
  }
  const formatted = formatNumberPtBr(value, decimals);
  return suffix ? `${formatted}${suffix}` : formatted;
}

/** Format a `pt-BR` integer count paired with singular (`|count| === 1`) or plural noun. */
export function pluralizePtBr(count: number, singular: string, plural: string): string {
  if (!Number.isFinite(count)) {
    return `${EM_DASH} ${plural}`;
  }
  const rounded = Math.trunc(count);
  const noun = Math.abs(rounded) === 1 ? singular : plural;
  return `${integerFormatter.format(rounded)} ${noun}`;
}
