ABSOLUTE_THRESHOLD = {
    "molecular_weight":             50.0,
    "exact_mol_weight":             0.005,
    "alogp":                        0.5,
    "molecular_polar_surface_area": 10.0,
    "clean_energy":                 2.0,
}

PLAUSIBILITY_FLOOR = {
    "number_of_heavy_atoms": 1.0,
    "num_rotatable_bonds":   0.0,
}

PLAUSIBILITY_CEILING = {
    "molecular_weight":             2000.0,
    "num_h_acceptors_lipinski":     20.0,
    "num_h_donors_lipinski":        10.0,
    "num_h_acceptors":              20.0,
    "num_h_donors":                 10.0,
    "num_rotatable_bonds":          50.0,
    "molecular_polar_surface_area": 500.0,
    "alogp":                        10.0,
}

CONFIDENCE_CAP = {
    "alogp":                        0.7,
    "molecular_weight":             0.8,
    "exact_mol_weight":             0.85,
    "molecular_polar_surface_area": 0.75,
    "clean_energy":                 0.65,
    "number_of_heavy_atoms":        0.8,   # renamed from number_of_atoms
}

ACTIVITY_FIELDS = set()   # empty — no activity fields in schema anymore