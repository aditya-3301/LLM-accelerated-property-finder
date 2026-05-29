# ---------------------------------------------------------------------------
# REQUIRED_SCHEMA
# Default values encode the *type* each leaf must hold after fusion/LLM2:
#   int   → 0          float → 0.0         str → ""     list → []
#   dict  → {...}      None-able fields use None as sentinel elsewhere
#
# clean_energy removed: no agreed units, no standard computation method,
# and extraction is almost entirely hallucinated/noisy.
#
# molecular_composition is now a structured dict of element → mass-fraction
# so fusion can operate on individual fractions instead of opaque strings.
# ---------------------------------------------------------------------------

REQUIRED_SCHEMA = {
    "cid": 0,

    "identity": {
        "drugbank_id":                 "",
        "secondary_accession_numbers": [],
        "common_name":                 "",
        "cas_number":                  "",
        "unii":                        "",
        "synonyms":                    [],
        "smiles":                      "",
    },

    "molecule": {
        "number_of_heavy_atoms":        0,      # integer — count of all non-H atoms
        "molecular_formula":            "",
        "molecular_composition":        {},     # {element: mass_fraction}  e.g. {"C": 0.600, "H": 0.045, "O": 0.355}
        "molecular_weight":             0.0,
        "exact_mol_weight":             0.0,
        "net_formal_charge":            0,      # integer
        "alogp":                        0.0,
        "num_h_acceptors_lipinski":     0,      # integer
        "num_h_donors_lipinski":        0,      # integer
        "num_rotatable_bonds":          0,      # integer
        "molecular_polar_surface_area": 0.0,
        "num_h_acceptors":              0,      # integer
        "num_h_donors":                 0,      # integer
    },
}

# ---------------------------------------------------------------------------
# FIELD_TYPES
# Controls which fusion strategy is applied to each leaf field.
#
#   "integer"      – discrete count; weighted mean → round to int
#   "float"        – continuous; confidence-weighted mean
#   "categorical"  – string/enum; weighted mode (highest-confidence wins)
#   "list"         – multi-valued string list (synonyms, accessions)
#   "smiles"       – validated string; highest-confidence valid candidate wins
#   "composition"  – dict of {element: float}; each element fused independently
# ---------------------------------------------------------------------------

FIELD_TYPES = {
    # identity
    "drugbank_id":                 "categorical",
    "secondary_accession_numbers": "list",
    "common_name":                 "categorical",
    "cas_number":                  "categorical",
    "unii":                        "categorical",
    "synonyms":                    "list",
    "smiles":                      "smiles",

    # molecule — integers
    "number_of_heavy_atoms":       "integer",
    "net_formal_charge":           "integer",
    "num_h_acceptors_lipinski":    "integer",
    "num_h_donors_lipinski":       "integer",
    "num_rotatable_bonds":         "integer",
    "num_h_acceptors":             "integer",
    "num_h_donors":                "integer",

    # molecule — floats
    "molecular_weight":            "float",
    "exact_mol_weight":            "float",
    "alogp":                       "float",
    "molecular_polar_surface_area":"float",

    # molecule — specials
    "molecular_formula":           "categorical",
    "molecular_composition":       "composition",
}

# ---------------------------------------------------------------------------
# SOURCE_PRIORITY
# Higher number = higher trust.  Used when two candidates have equal confidence,
# and as a tie-breaker in weighted-mode selection for categorical fields.
# PubChem/DrugBank canonical values dominate; literature/other are demoted.
# ---------------------------------------------------------------------------

SOURCE_PRIORITY = {
    "pubchem":    5,
    "drugbank":   5,
    "chembl":     4,
    "bindingdb":  3,
    "literature": 2,
    "other":      1,
}

def source_priority(source_type: str) -> int:
    if not isinstance(source_type, str):
        return 1
    key = source_type.strip().lower()
    # Accept prefixed strings like "PubChem_computed"
    for name, score in SOURCE_PRIORITY.items():
        if name in key:
            return score
    return 1


# ---------------------------------------------------------------------------
# DETERMINISTIC_FIELDS
# These descriptors can be computed exactly from molecular_formula or SMILES;
# they should be extracted for reference but their LLM candidates are capped
# at a lower confidence ceiling because the ground truth is structural, not
# measured.  main.py uses this to trigger formula/SMILES-based recomputation
# when a validated SMILES is available.
# ---------------------------------------------------------------------------

DETERMINISTIC_FIELDS = {
    "number_of_heavy_atoms",
    "molecular_weight",
    "exact_mol_weight",
    "molecular_formula",
    "molecular_composition",
    "net_formal_charge",
    "num_h_acceptors_lipinski",
    "num_h_donors_lipinski",
    "num_rotatable_bonds",
    "num_h_acceptors",
    "num_h_donors",
    "molecular_polar_surface_area",
}
