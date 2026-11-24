"""Pure stdlib Cobb Avian 48 developmental age-stage mapper (D-08).

Maps predicted live broiler weight (g) into the 4 agronomic developmental stages
grounded in the audited ``116–2,093 g`` (``~7–29 days``) training coverage and
exposes an explicit ``out_of_training_range`` disclosure flag whenever the weight
falls outside ``[116, 2093] g``.

Intentionally imports strictly Python stdlib modules so Phase 3 offline audits
and Phase 4 FastAPI services share one zero-overhead domain contract.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

VALID_TRAINING_WEIGHT_MIN_G: int = 116
VALID_TRAINING_WEIGHT_MAX_G: int = 2093
VALID_AGE_DAYS_RANGE: str = "~7-29 days"

_STAGE_INICIAL_MAX_EXCLUSIVE_G: float = 350.0
_STAGE_CRESC_INICIAL_MAX_INCLUSIVE_G: float = 700.0
_STAGE_CRESC_INTERM_MAX_INCLUSIVE_G: float = 1200.0


@dataclass(frozen=True, slots=True)
class AgeStageResult:
    """Agronomic developmental stage classification for a predicted broiler weight."""

    weight_g: float
    stage_code: str
    stage_label: str
    estimated_age_days_range: str
    out_of_training_range: bool
    valid_training_range_g: tuple[int, int]
    valid_age_days_range: str


def map_weight_to_age_stage(weight_g: float) -> AgeStageResult:
    """Map a strictly positive finite live weight (g) into a Cobb Avian 48 stage (D-08)."""
    if isinstance(weight_g, bool) or not isinstance(weight_g, numbers.Real):
        raise ValueError(f"weight_g must be a finite positive real number, got {weight_g!r}")
    w = float(weight_g)
    if not math.isfinite(w) or w <= 0.0:
        raise ValueError(f"weight_g must be finite and > 0, got {weight_g!r}")

    if w < _STAGE_INICIAL_MAX_EXCLUSIVE_G:
        stage_code = "inicial_pintinho"
        stage_label = "Inicial / Pintinho"
        estimated_age_days_range = "~7-13 dias"
    elif w <= _STAGE_CRESC_INICIAL_MAX_INCLUSIVE_G:
        stage_code = "crescimento_inicial"
        stage_label = "Crescimento Inicial"
        estimated_age_days_range = "~14-19 dias"
    elif w <= _STAGE_CRESC_INTERM_MAX_INCLUSIVE_G:
        stage_code = "crescimento_intermediario"
        stage_label = "Crescimento Intermediário"
        estimated_age_days_range = "~20-24 dias"
    else:
        stage_code = "terminacao_pre_abate"
        stage_label = "Terminação / Pré-Abate"
        estimated_age_days_range = "~25-29 dias"

    out_of_training_range = (w < float(VALID_TRAINING_WEIGHT_MIN_G)) or (
        w > float(VALID_TRAINING_WEIGHT_MAX_G)
    )

    return AgeStageResult(
        weight_g=w,
        stage_code=stage_code,
        stage_label=stage_label,
        estimated_age_days_range=estimated_age_days_range,
        out_of_training_range=out_of_training_range,
        valid_training_range_g=(VALID_TRAINING_WEIGHT_MIN_G, VALID_TRAINING_WEIGHT_MAX_G),
        valid_age_days_range=VALID_AGE_DAYS_RANGE,
    )


__all__ = [
    "VALID_TRAINING_WEIGHT_MIN_G",
    "VALID_TRAINING_WEIGHT_MAX_G",
    "VALID_AGE_DAYS_RANGE",
    "AgeStageResult",
    "map_weight_to_age_stage",
]
