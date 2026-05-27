# ── Fusion configuration ──────────────────────────────────────────────────────
ABSOLUTE_THRESHOLD = {
    "logP":                       0.5,
    "molecular_weight":           50.0,
    "exact_mol_weight":           0.005,
    "alogp":                      0.5,
    "molecular_polar_surface_area": 10.0,
    "e_solubility":               0.5,
    "logkp":                      0.3,
    "reported_ic50":              100.0,
    "ec50":                       100.0,
    "ld50":                       200.0,
}

PLAUSIBILITY_FLOOR = {
    # After nM conversion, values below 0.01 nM are sub-picomolar — implausible
    "activity_value":          0.01,
    "reported_ic50":           0.01,
    "ec50":                    0.01,
    "synthetic_accessibility": 1.0,   # SA score defined on [1, 10]
    "ld50":                    1.0,    # LD50 below 1 mg/kg is extreme outlier
}

CONFIDENCE_CAP = {
    "logP":                       0.7,
    "molecular_weight":           0.8,
    "exact_mol_weight":           0.85,
    "alogp":                      0.7,
    "molecular_polar_surface_area": 0.75,
}