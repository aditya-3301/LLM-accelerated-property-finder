import statistics

# Fields where absolute deviation is more meaningful than ratio
ABSOLUTE_THRESHOLD = {
    "logP": 1.5,               # candidates must be within 1.5 units of median
    "molecular_weight": 50.0,  # candidates must be within 50 Da of median
}

# Plausibility floors — values below these are physically implausible and dropped
PLAUSIBILITY_FLOOR = {
    "activity_value": 0.01,    # anything below 0.01 nM is sub-picomolar, implausible for drugs
}

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

def _normalize_units(candidates: list) -> list:
    """
    If a candidate dict carries a per-candidate 'unit' key, convert its value to nM.
    Candidates without a 'unit' key are assumed already in nM (LLM1's contract).
    """
    normalized = []
    for c in candidates:
        unit = c.get("unit", "nm").strip().lower()
        factor = _UNIT_TO_NM.get(unit)
        if factor is None:
            print(f"[FUSION] Unknown unit '{c.get('unit')}' on candidate — assuming nM")
            factor = 1.0
        if factor != 1.0 and isinstance(c.get("value"), (int, float)):
            c = dict(c)  # don't mutate original
            c["value"] = c["value"] * factor
            print(f"[FUSION] Converted {c['value'] / factor} {unit.upper()} → {c['value']} nM")
        normalized.append(c)
    return normalized

def _fuse_candidates(candidates: list, field_name: str = ""):
    """
    Takes a list of candidate dicts {"value": ..., "confidence": ..., "source_type": ...}
    OR a plain scalar list (e.g. ["IC50", "EC50"]) and returns a single fused value.
    - Candidate dicts: outlier removal + confidence-weighted mean for numerics,
      highest-confidence for strings
    - Plain scalar list: first element (LLM1 failed to wrap properly)
    """
    if not candidates:
        return None

    # If LLM1 returned a plain scalar list instead of candidate dicts, take the first element
    if not all(_is_candidate_dict(item) for item in candidates):
        return candidates[0]
    # Drop low-confidence candidates first — no point normalizing units on discarded candidates
    candidates = [c for c in candidates if c.get("confidence", 0) >= MIN_CONFIDENCE]
    if not candidates:
        return None
    # Normalize per-candidate units to nM on survivors only
    if field_name in ("activity_value",):
        candidates = _normalize_units(candidates)

    numeric = [(c["value"], c["confidence"]) for c in candidates if isinstance(c["value"], (int, float))]
    string_candidates = [(c["value"], c["confidence"]) for c in candidates if isinstance(c["value"], str)]

    if numeric:
        values = [v for v, c in numeric]
        confs  = [c for v, c in numeric]

        # Plausibility floor — drop physically implausible values before outlier removal
        if field_name in PLAUSIBILITY_FLOOR:
            floor = PLAUSIBILITY_FLOOR[field_name]
            filtered = [(v, c) for v, c in zip(values, confs) if v >= floor]
            if not filtered:
                return None  # all values were physically implausible
            values = [v for v, c in filtered]
            confs  = [c for v, c in filtered]

        # Outlier removal — always run if more than 1 candidate
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


def fuse(extracted: dict) -> dict:
    """
    Recursively walks the extracted dict.
    Leaf nodes (lists of candidates) → fused to a single value.
    Branch nodes (dicts) → recurse.
    Scalars → passed through directly.
    """
    fused = {}
    for key, value in extracted.items():
        if isinstance(value, dict):
            fused[key] = fuse(value)
        elif isinstance(value, list):
            fused[key] = _fuse_candidates(value, field_name=key)
        else:
            fused[key] = value  # scalar passthrough (e.g. cid)
    return fused