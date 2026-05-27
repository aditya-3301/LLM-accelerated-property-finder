# ── Fusion configuration ──────────────────────────────────────────────────────
# Keys must match leaf field names in schema.py exactly.
# This is the ONLY file that needs editing when schema.py gains new fields
# that require custom fusion behaviour.

# ── ABSOLUTE_THRESHOLD ────────────────────────────────────────────────────────
# Fields where absolute deviation from the median is more meaningful than a
# ratio-based outlier filter.  Any source whose value falls outside
#   median ± threshold  is treated as an outlier and down-weighted.

ABSOLUTE_THRESHOLD = {
    # ── v1 (unchanged) ──────────────────────────────────────────────────────
    "logP":             0.5,    # reputable sources agree within 0.5 log units
    "molecular_weight": 50.0,   # candidates must be within 50 Da of median

    # ── Molecular descriptors ────────────────────────────────────────────────
    "exact_mol_weight":             0.005,  # monoisotopic MW is deterministic; tight window
    "alogp":                        0.5,    # same order as logP; variant-dependent
    "molecular_polar_surface_area": 10.0,   # ± 10 Å² is reasonable cross-tool agreement

    # ── Physicochemical / ADMET numerics ────────────────────────────────────
    "e_solubility": 0.5,    # log mol/L; ± 0.5 unit cross-model agreement
    "logkp":        0.3,    # skin-perm log cm/s; tight because range is narrow

    # ── Bioactivity ──────────────────────────────────────────────────────────
    "reported_ic50": 100.0, # nM; wide window — assay conditions vary
    "ec50":          100.0, # nM; same rationale as IC50

    # ── Toxicity ─────────────────────────────────────────────────────────────
    "ld50": 200.0,          # mg/kg; inter-study variability can be large
}


# ── PLAUSIBILITY_FLOOR ────────────────────────────────────────────────────────
# Values strictly below these are physically / biologically implausible and
# are dropped before fusion regardless of reported confidence.

PLAUSIBILITY_FLOOR = {
    # ── v1 (unchanged) ──────────────────────────────────────────────────────
    "activity_value": 1.0,  # sub-1 nM is implausible for most drug candidates

    # ── Bioactivity ──────────────────────────────────────────────────────────
    "reported_ic50": 1.0,   # same rationale as activity_value
    "ec50":          1.0,

    # ── Drug-likeness ────────────────────────────────────────────────────────
    "synthetic_accessibility": 1.0,  # SA score is defined on [1, 10]

    # ── Toxicity ─────────────────────────────────────────────────────────────
    "ld50": 1.0,            # LD50 below 1 mg/kg would be an extreme outlier
}


# ── CONFIDENCE_CAP ────────────────────────────────────────────────────────────
# Fields where LLM confidence is structurally overestimated.  Raw confidence
# scores are clamped to these ceilings before weighted fusion so that a single
# over-confident source cannot dominate.

CONFIDENCE_CAP = {
    # ── v1 (unchanged) ──────────────────────────────────────────────────────
    "logP":             0.7,    # computed property; no 8B model should be 100% confident
    "molecular_weight": 0.8,    # usually correct but occasional formula errors

    # ── Computed physicochemical descriptors ─────────────────────────────────
    "exact_mol_weight":             0.85,  # deterministic once formula is known, but formula can err
    "alogp":                        0.7,   # algorithm-specific; variant confusion is common
    "molecular_polar_surface_area": 0.75,  # conformation-dependent; 2-D approximation

}