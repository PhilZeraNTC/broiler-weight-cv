import { AlertCircle, X } from "lucide-react";
import { cn } from "../../lib/cn.ts";

export interface InlineErrorBannerProps {
  message: string;
  title?: string;
  actionLabel?: string;
  onAction?: () => void;
  onDismiss?: () => void;
  className?: string;
}

/**
 * Accessible inline error banner (`role="alert"`) rendering sanitized pt-BR copy
 * and optional Verb+Noun recovery CTA (never raw stack traces or file paths).
 */
export function InlineErrorBanner({
  message,
  title,
  actionLabel,
  onAction,
  onDismiss,
  className,
}: InlineErrorBannerProps) {
  return (
    <div
      role="alert"
      className={cn(
        "flex flex-col gap-3 rounded-xl border border-danger-border bg-danger-tint p-4 text-danger sm:flex-row sm:items-start sm:justify-between",
        className,
      )}
    >
      <div className="flex min-w-0 items-start gap-3">
        <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-danger" aria-hidden="true" />
        <div className="min-w-0 space-y-1">
          {title ? (
            <p className="text-[12px] font-semibold leading-[1.33] text-danger">{title}</p>
          ) : null}
          <p className="text-[14px] font-normal leading-[1.5] text-danger">{message}</p>
        </div>
      </div>

      {actionLabel || onDismiss ? (
        <div className="flex shrink-0 items-center gap-2 sm:pl-4">
          {actionLabel && onAction ? (
            <button
              type="button"
              onClick={onAction}
              className="inline-flex min-h-[44px] items-center rounded-lg border border-danger-border bg-surface px-3 text-[14px] font-normal leading-[1.5] text-danger transition-colors hover:bg-danger/10"
            >
              {actionLabel}
            </button>
          ) : null}
          {onDismiss ? (
            <button
              type="button"
              onClick={onDismiss}
              aria-label="Fechar notificação"
              className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-lg text-danger transition-colors hover:bg-danger/10"
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
