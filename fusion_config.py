# ── Fusion configuration ──────────────────────────────────────────────────────
# Keys must match leaf field names in schema.py exactly.
# This is the ONLY file that needs editing when schema.py gains new fields
# that require custom fusion behaviour.

# Fields where absolute deviation is more meaningful than ratio-based outlier removal
ABSOLUTE_THRESHOLD = {
    "logP": 0.5,               # reputable sources agree within 0.5 units
    "molecular_weight": 50.0,  # candidates must be within 50 Da of median
}

# Plausibility floors — values below these are physically implausible and dropped
PLAUSIBILITY_FLOOR = {
    "activity_value": 1.0,     # below 1 nM is sub-nanomolar, implausible for most drugs
}

# Fields where LLM confidence is structurally overestimated — cap before fusion
CONFIDENCE_CAP = {
    "logP": 0.7,               # computed property; no 8B model should be 100% confident
    "molecular_weight": 0.8,   # usually correct but occasional formula errors
}