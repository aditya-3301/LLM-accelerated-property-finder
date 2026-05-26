import statistics

def _fuse_candidates(candidates: list):
    """
    Takes a list of {"value": ..., "confidence": ..., "source_type": ...}
    and returns a single fused value.
    - Numeric fields: outlier removal + confidence-weighted mean
    - String fields: highest-confidence value wins
    """
    if not candidates:
        return None

    numeric = [(c["value"], c["confidence"]) for c in candidates if isinstance(c["value"], (int, float))]
    string_candidates = [(c["value"], c["confidence"]) for c in candidates if isinstance(c["value"], str)]

    if numeric:
        values = [v for v, c in numeric]
        confs  = [c for v, c in numeric]

        # Outlier removal: drop values outside 5x ratio of median
        if len(values) >= 3:
            med = statistics.median(values)
            if med != 0:
                filtered = [
                    (v, c) for v, c in zip(values, confs)
                    if (max(v, med) / max(min(v, med), 1e-9)) <= 5
                ]
                if filtered:
                    values = [v for v, c in filtered]
                    confs  = [c for v, c in filtered]

        # Confidence-weighted mean
        total_conf = sum(confs)
        if total_conf == 0:
            return sum(values) / len(values)
        return sum(v * c for v, c in zip(values, confs)) / total_conf

    if string_candidates:
        # Highest-confidence string wins
        return max(string_candidates, key=lambda x: x[1])[0]

    return None


def fuse(extracted: dict) -> dict:
    """
    Recursively walks the extracted dict.
    Leaf nodes (lists of candidates) → fused to a single value.
    Branch nodes (dicts) → recurse.
    """
    fused = {}
    for key, value in extracted.items():
        if isinstance(value, dict):
            fused[key] = fuse(value)
        elif isinstance(value, list):
            fused[key] = _fuse_candidates(value)
        else:
            fused[key] = value  # scalar passthrough (e.g. cid)
    return fused