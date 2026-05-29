import statistics
from fusion_config import (
    ABSOLUTE_THRESHOLD,
    PLAUSIBILITY_FLOOR,
    PLAUSIBILITY_CEILING,
    CONFIDENCE_CAP,
    ACTIVITY_FIELDS,
)

# Minimum confidence for a candidate to participate in fusion
MIN_CONFIDENCE = 0.4

def _is_candidate_dict(item) -> bool:
    """Returns True if item is a proper candidate dict with a 'value' key."""
    return isinstance(item, dict) and "value" in item


def _fuse_candidates(candidates: list, field_name: str = "", unit_override: str = None):
    """
    Fuses a list of candidate dicts:
      {"value": ..., "confidence": 0–1, "source_type": "..."}
    OR a plain scalar list (LLM1 forgot to wrap) → returns first element.

    Returns a single fused scalar, or None if nothing survives filtering.
    """
    if not candidates:
        return None

    # ── Plain scalar list — LLM1 didn't wrap properly ──────────────────────
    if not all(_is_candidate_dict(item) for item in candidates):
        # Try to find the first dict candidate; fall back to first raw value
        dict_items = [i for i in candidates if _is_candidate_dict(i)]
        if dict_items:
            candidates = dict_items
        else:
            # All raw scalars — return the first non-None value
            for item in candidates:
                if item is not None:
                    return item
            return None

    # ── Drop low-confidence candidates ─────────────────────────────────────
    candidates = [c for c in candidates if c.get("confidence", 0) >= MIN_CONFIDENCE]
    if not candidates:
        return None

    # ── Unit normalisation (activity fields only) ───────────────────────────
    if field_name in ACTIVITY_FIELDS:
        candidates = _normalize_units(candidates, unit_override=unit_override)

    # ── Cap overconfident candidates ────────────────────────────────────────
    if field_name in CONFIDENCE_CAP:
        cap = CONFIDENCE_CAP[field_name]
        candidates = [
            {**c, "confidence": min(c.get("confidence", 0), cap)}
            for c in candidates
        ]

    # ── Consensus hallucination guard (activity fields only) ────────────────
    if _check_consensus_hallucination(candidates, field_name):
        print(
            f"[FUSION] WARNING: all '{field_name}' candidates have identical values "
            f"— possible consensus hallucination, confidence capped to 0.5"
        )
        candidates = [
            {**c, "confidence": min(c.get("confidence", 0), 0.5)}
            for c in candidates
        ]

    # ── Split by value type ─────────────────────────────────────────────────
    numeric = [
        (c["value"], c["confidence"])
        for c in candidates
        if isinstance(c.get("value"), (int, float))
    ]
    string_candidates = [
        (c["value"], c["confidence"])
        for c in candidates
        if isinstance(c.get("value"), str)
    ]

    # ── Numeric fusion ──────────────────────────────────────────────────────
    if numeric:
        values = [v for v, _ in numeric]
        confs  = [c for _, c in numeric]

        # Plausibility floor
        if field_name in PLAUSIBILITY_FLOOR:
            floor = PLAUSIBILITY_FLOOR[field_name]
            paired = [(v, c) for v, c in zip(values, confs) if v >= floor]
            if not paired:
                return None
            values, confs = zip(*paired)
            values, confs = list(values), list(confs)

        # Plausibility ceiling
        if field_name in PLAUSIBILITY_CEILING:
            ceiling = PLAUSIBILITY_CEILING[field_name]
            paired = [(v, c) for v, c in zip(values, confs) if v <= ceiling]
            if not paired:
                return None
            values, confs = zip(*paired)
            values, confs = list(values), list(confs)

        # Outlier removal (only meaningful with 2+ candidates)
        if len(values) >= 2:
            med = statistics.median(values)
            if field_name in ABSOLUTE_THRESHOLD:
                # Tight absolute band — for descriptors like MW, TPSA
                threshold = ABSOLUTE_THRESHOLD[field_name]
                paired = [
                    (v, c) for v, c in zip(values, confs)
                    if abs(v - med) <= threshold
                ]
            else:
                # Ratio-based filter: keep candidates within 5× of the median.
                # Works correctly for activity values spanning orders of magnitude
                # and also for logP-like fields not in ABSOLUTE_THRESHOLD.
                if abs(med) > 1e-9:
                    paired = [
                        (v, c) for v, c in zip(values, confs)
                        if (max(abs(v), abs(med)) / max(abs(min(abs(v), abs(med))), 1e-9)) <= 5
                    ]
                else:
                    # median is ~0 — keep everything within ±1 absolute
                    paired = [(v, c) for v, c in zip(values, confs) if abs(v) <= 1]

            if paired:
                values, confs = zip(*paired)
                values, confs = list(values), list(confs)

        # Confidence-weighted mean
        total_conf = sum(confs)
        if total_conf == 0:
            return sum(values) / len(values)
        return sum(v * c for v, c in zip(values, confs)) / total_conf

    # ── String fusion — highest-confidence wins ─────────────────────────────
    if string_candidates:
        return max(string_candidates, key=lambda x: x[1])[0]

    return None


def _extract_unit_override(section: dict) -> str:
    """
    Reads the activity_unit field from a raw (pre-fusion) section dict and
    returns the best unit string so activity value candidates can be converted.
    Handles both candidate-list format and plain scalar.
    """
    unit_field = section.get("activity_unit")
    if not unit_field:
        return None
    if isinstance(unit_field, list):
        valid = [c for c in unit_field if _is_candidate_dict(c)]
        if valid:
            best = max(valid, key=lambda c: c.get("confidence", 0))
            return best.get("value", "nm")
    if isinstance(unit_field, str):
        return unit_field
    return None


def fuse(extracted: dict) -> dict:
    """
    Recursively walks the extracted dict.
      • Leaf list  → fused to a single value via _fuse_candidates
      • Branch dict → recurse
      • Scalar      → passed through unchanged

    The interaction_profile section passes its activity_unit as a unit_override
    so conversion works even when LLM1 puts the unit in a separate field.
    """
    unit_override = (
        _extract_unit_override(extracted)
        if "activity_unit" in extracted
        else None
    )

    fused = {}
    for key, value in extracted.items():
        if isinstance(value, dict):
            fused[key] = fuse(value)
        elif isinstance(value, list):
            override = unit_override if key in ACTIVITY_FIELDS else None
            fused[key] = _fuse_candidates(value, field_name=key, unit_override=override)
        else:
            fused[key] = value  # scalar passthrough
    return fused