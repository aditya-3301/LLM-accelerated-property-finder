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
        "number_of_heavy_atoms" :        0,      # renamed from number_of_atoms — heavy atoms only
        "molecular_formula":            "",
        "molecular_composition":        "",
        "molecular_weight":             0.0,
        "exact_mol_weight":             0.0,
        "net_formal_charge":            0,
        "clean_energy":                 0.0,
        "alogp":                        0.0,
        "num_h_acceptors_lipinski":     0,
        "num_h_donors_lipinski":        0,
        "num_rotatable_bonds":          0,
        "molecular_polar_surface_area": 0.0,
        "num_h_acceptors":              0,
        "num_h_donors":                 0,
    },
}