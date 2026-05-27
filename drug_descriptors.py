# ── Schema definition ─────────────────────────────────────────────────────────
# Leaf-value types define the expected Python type for each field:
#   str  → ""        int → 0        float → 0.0        bool → False
#   list → []  (multi-valued string fields, e.g. synonyms, cyp_inhibitors)
#
# Nested dicts are logical sections — not nested API objects.
# Keys must match leaf field names in fusion_config.py exactly.

REQUIRED_SCHEMA = {
    # ── Top-level identifier ─────────────────────────────────────────────────
    "cid": 0,

    # ── Chemical identity ────────────────────────────────────────────────────
    "identity": {
        "drugbank_id":        "",   # e.g. "DB00001"
        "common_name":        "",
        "cas_number":         "",   # e.g. "50-00-0"
        "synonyms":           [],   # list of alternate names / trade names
        "smiles":             "",   # canonical SMILES string
    },

    # ── Molecular / physicochemical descriptors ──────────────────────────────
    "molecule": {
        "molecular_formula":          "",   # e.g. "C9H8O4"
        "molecular_composition":      "",   # human-readable element breakdown
        "molecular_weight":           0.0,  # average MW  (Da)
        "exact_mol_weight":           0.0,  # monoisotopic MW (Da)
        "net_formal_charge":          0,    # integer charge
        "logP":                       0.0,  # Wildman-Crippen logP (kept from v1)
        "alogp":                      0.0,  # AlogP (Ghose-Crippen variant)
        "num_h_acceptors":            0,    # general H-bond acceptor count
        "num_h_donors":               0,    # general H-bond donor count
        "num_rotatable_bonds":        0,
        "molecular_polar_surface_area": 0.0,  # TPSA (Å²)
    },

    # ── Bioactivity / interaction profile ────────────────────────────────────
    "interaction_profile": {
        "activity_type":  "IC50",  # primary assay type label (kept from v1)
        "activity_value": 0.0,     # primary assay value      (kept from v1)
        "activity_unit":  "nM",    # unit for activity_value  (kept from v1)
        "reported_ic50":  0.0,     # explicit IC50 in nM when available
        "ec50":           0.0,     # EC50 in nM
    },

    # ── ADMET predictions ────────────────────────────────────────────────────
    "admet": {
        "e_solubility":    0.0,   # log mol/L (ESOL predicted aqueous solubility)
        "gi_absorption":   "",    # "High" | "Low"
        "logkp":           0.0,   # skin permeation coefficient (log cm/s)
        "bbb_penetration": "",    # "BBB+" | "BBB-"
        "p_gp_substrate":  "",    # "Yes" | "No"
        "cyp_inhibitors":  [],    # list of inhibited CYPs, e.g. ["CYP3A4", "CYP2D6"]
    },

    # ── Toxicity endpoints ───────────────────────────────────────────────────
    "toxicity": {
        "carcinogenicity":  "",   # "Positive" | "Negative" | "Inconclusive"
        "immunotoxicity":   "",   # "Positive" | "Negative" | "Inconclusive"
        "mutagenicity":     "",   # "Positive" | "Negative" | "Inconclusive"
        "cytotoxicity":     "",   # "Positive" | "Negative" | "Inconclusive"
        "ld50":             0.0,  # acute oral LD50 (mg/kg, rat)
    },
}