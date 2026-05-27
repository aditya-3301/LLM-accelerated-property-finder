import statistics
from fusion_config import ABSOLUTE_THRESHOLD, PLAUSIBILITY_FLOOR, CONFIDENCE_CAP

# Minimum confidence to participate in fusion
MIN_CONFIDENCE = 0.4

def _is_candidate_dict(item) -> bool:
    """Returns True if item is a proper candidate dict with a 'value' key."""
    return isinstance(item, dict) and "value" in item

_UNIT_TO_NM = {
    "m":  1e9,
    "mm": 1e6,
    "um": 1e3,
    "µm": 1e3,
    "nm": 1.0,
}

def _normalize_units(candidates: list, unit_override: str = None) -> list:
    """
    Convert activity candidate values to nM.
    Priority: per-candidate 'unit' key > unit_override > assume nM.
    unit_override is the fused activity_unit value from the same section
    (e.g. "uM") when LLM1 puts the unit in a separate field rather than
    per-candidate.
    """
    normalized = []
    for c in candidates:
        # Per-candidate unit takes priority; fall back to section-level override
        raw_unit = c.get("unit") or unit_override or "nm"
        unit = raw_unit.strip().lower()
        factor = _UNIT_TO_NM.get(unit)
        if factor is None:
            print(f"[FUSION] Unknown unit '{raw_unit}' — assuming nM")
            factor = 1.0
        if factor != 1.0 and isinstance(c.get("value"), (int, float)):
            c = dict(c)  # don't mutate original
            original = c["value"]
            c["value"] = c["value"] * factor
            print(f"[FUSION] Converted {original} {raw_unit.upper()} → {c['value']} nM")
        normalized.append(c)
    return normalized

def _check_consensus_hallucination(candidates: list) -> bool:
    """
    Returns True if all numeric candidates have identical values — a hallucination signal.
    A real multi-source extraction always has some variance.
    """
    values = [c["value"] for c in candidates if isinstance(c.get("value"), (int, float))]
    if len(values) < 2:
        return False
    return len(set(values)) == 1


def _fuse_candidates(candidates: list, field_name: str = "", unit_override: str = None):
    """
    Takes a list of candidate dicts {"value": ..., "confidence": ..., "source_type": ...}
    OR a plain scalar list (e.g. ["IC50", "EC50"]) and returns a single fused value.
    unit_override: section-level unit string passed in when LLM1 reports units
    as a separate field rather than per-candidate.
    """
    if not candidates:
        return None

    # Plain scalar list — LLM1 didn't wrap properly; take first element
    if not all(_is_candidate_dict(item) for item in candidates):
        return candidates[0]

    # Drop low-confidence candidates
    candidates = [c for c in candidates if c.get("confidence", 0) >= MIN_CONFIDENCE]
    if not candidates:
        return None

    # Normalize units to nM using per-candidate key or section-level override
    if field_name in ("activity_value", "reported_ic50", "ec50"):
        candidates = _normalize_units(candidates, unit_override=unit_override)

    # Cap overconfident candidates
    if field_name in CONFIDENCE_CAP:
        cap = CONFIDENCE_CAP[field_name]
        candidates = [
            {**c, "confidence": min(c.get("confidence", 0), cap)}
            for c in candidates
        ]

    # Consensus hallucination warning
    if _check_consensus_hallucination(candidates):
        print(f"[FUSION] WARNING: all '{field_name}' candidates have identical values "
              f"— possible consensus hallucination, confidence capped to 0.5")
        candidates = [
            {**c, "confidence": min(c.get("confidence", 0), 0.5)}
            for c in candidates
        ]

    numeric = [(c["value"], c["confidence"]) for c in candidates if isinstance(c["value"], (int, float))]
    string_candidates = [(c["value"], c["confidence"]) for c in candidates if isinstance(c["value"], str)]

    if numeric:
        values = [v for v, c in numeric]
        confs  = [c for v, c in numeric]

        # Plausibility floor
        if field_name in PLAUSIBILITY_FLOOR:
            floor = PLAUSIBILITY_FLOOR[field_name]
            filtered = [(v, c) for v, c in zip(values, confs) if v >= floor]
            if not filtered:
                return None
            values = [v for v, c in filtered]
            confs  = [c for v, c in filtered]

        # Outlier removal
        if len(values) >= 2:
            med = statistics.median(values)
            if field_name in ABSOLUTE_THRESHOLD:
                threshold = ABSOLUTE_THRESHOLD[field_name]
                filtered = [
                    (v, c) for v, c in zip(values, confs)
                    if abs(v - med) <= threshold
                ]
            else:
                if med != 0:
                    filtered = [
                        (v, c) for v, c in zip(values, confs)
                        if abs(med) > 1e-9 and (max(abs(v), abs(med)) / max(min(abs(v), abs(med)), 1e-9)) <= 5
                    ]
                else:
                    filtered = list(zip(values, confs))
            if filtered:
                values = [v for v, c in filtered]
                confs  = [c for v, c in filtered]

        # Confidence-weighted mean
        total_conf = sum(confs)
        if total_conf == 0:
            return sum(values) / len(values)
        return sum(v * c for v, c in zip(values, confs)) / total_conf

    if string_candidates:
        return max(string_candidates, key=lambda x: x[1])[0]

    return None


def _extract_unit_override(section: dict) -> str:
    """
    Reads the activity_unit field from a section dict (before fusion) and
    returns the raw unit string so activity value candidates can be converted.
    Handles both candidate-list format and plain scalar.
    """
    unit_field = section.get("activity_unit")
    if not unit_field:
        return None
    if isinstance(unit_field, list):
        # Candidate list — take highest-confidence unit
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
    Leaf nodes (lists of candidates) → fused to a single value.
    Branch nodes (dicts) → recurse.
    Scalars → passed through directly.

    Special case: interaction_profile section passes the activity_unit
    as a unit_override to activity value fields so unit conversion works
    even when LLM1 reports units as a separate field.
    """
    # Extract section-level unit override before recursing, if present
    unit_override = _extract_unit_override(extracted) if "activity_unit" in extracted else None

    fused = {}
    for key, value in extracted.items():
        if isinstance(value, dict):
            fused[key] = fuse(value)
        elif isinstance(value, list):
            override = unit_override if key in ("activity_value", "reported_ic50", "ec50") else None
            fused[key] = _fuse_candidates(value, field_name=key, unit_override=override)
        else:
            fused[key] = value  # scalar passthrough
    return fused