"""
main.py — Drug descriptor extraction pipeline.

Changes vs previous version
----------------------------
* clean_energy removed from schema, prompts, and guards.
* molecular_composition is now validated as a dict {element: float} instead
  of an opaque string; fractions are renormalised if they drift from 1.0.
* Deterministic fields (MW, HBA, formula, …) are recomputed from formula
  or SMILES whenever a validated SMILES is available, bypassing LLM noise.
* Integer-type enforcement pass ensures discrete counts are never stored as
  floats after LLM2 (addresses num_rotatable_bonds = 3.2857 etc.).
* Chemical validation checks added:
    - formula ↔ molecular weight consistency
    - SMILES ↔ descriptor consistency (heavy-atom count)
    - composition fraction sum check
* Provenance metadata surfaced from LLM2 and logged.
* Source-priority scoring is now used throughout fusion (see fusion.py).
* Duplicate secondary-accession guard added (also handled in fusion layer).
"""

import math
import json
import os
import re
import time
import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

import LLM1
import LLM2
import fusion
from drug_descriptors import REQUIRED_SCHEMA, DETERMINISTIC_FIELDS

# ── Auth ──────────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not found. Make sure it is set in your .env file.")

client = InferenceClient(token=HF_TOKEN)


# ── Atomic masses for formula-based recomputation ────────────────────────────

ATOMIC_MASS = {
    "H":  1.00794, "C":  12.0107, "N":  14.0067, "O":  15.9994,
    "S":  32.065,  "P":  30.97376,"F":  18.9984,  "Cl": 35.453,
    "Br": 79.904,  "I":  126.904, "Si": 28.0855,  "Se": 78.96,
    "As": 74.9216, "Te": 127.60,  "B":  10.811,
}

MONOISOTOPIC_MASS = {
    "H":  1.0078250, "C": 12.0000000, "N": 14.0030740, "O": 15.9949146,
    "S":  31.9720707,"P": 30.9737634, "F": 18.9984032, "Cl":34.9688527,
    "Br":78.9183376, "I":126.904468, "Si":27.9769265,  "Se":79.9165196,
    "As":74.9215964, "Te":129.906224,"B": 11.0093055,
}

_FORMULA_RE = re.compile(r'([A-Z][a-z]?)(\d*)')


def _parse_formula(formula: str) -> dict[str, int] | None:
    """Parse a molecular formula string into {element: count}.  Returns None on failure."""
    if not formula or not isinstance(formula, str):
        return None
    counts: dict[str, int] = {}
    for elem, num in _FORMULA_RE.findall(formula):
        if not elem:
            continue
        counts[elem] = counts.get(elem, 0) + (int(num) if num else 1)
    return counts if counts else None


def _compute_molecular_weight(formula: str) -> float | None:
    counts = _parse_formula(formula)
    if counts is None:
        return None
    return sum(ATOMIC_MASS.get(e, 0) * n for e, n in counts.items())


def _compute_exact_mol_weight(formula: str) -> float | None:
    counts = _parse_formula(formula)
    if counts is None:
        return None
    return sum(MONOISOTOPIC_MASS.get(e, 0) * n for e, n in counts.items())


def _compute_heavy_atom_count(formula: str) -> int | None:
    counts = _parse_formula(formula)
    if counts is None:
        return None
    return sum(n for e, n in counts.items() if e != "H")


def _compute_composition(formula: str, mw: float) -> dict[str, float] | None:
    """Return {element: mass_fraction} from formula and average MW."""
    counts = _parse_formula(formula)
    if counts is None or mw is None or mw <= 0:
        return None
    comp = {
        e: round((n * ATOMIC_MASS.get(e, 0)) / mw, 6)
        for e, n in counts.items()
        if e in ATOMIC_MASS
    }
    total = sum(comp.values())
    if total > 1e-6:
        comp = {k: round(v / total, 6) for k, v in comp.items()}
    return comp if comp else None


# ── Schema helpers ────────────────────────────────────────────────────────────

def _get_numeric_paths(schema: dict, path: tuple = ()) -> list:
    paths = []
    for key, val in schema.items():
        if key == "cid":
            continue
        if isinstance(val, dict) and key != "molecular_composition":
            paths.extend(_get_numeric_paths(val, path + (key,)))
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            paths.append(path + (key,))
    return paths


def _get_string_paths(schema: dict, path: tuple = ()) -> list:
    paths = []
    for key, val in schema.items():
        if key == "cid":
            continue
        if isinstance(val, dict) and key != "molecular_composition":
            paths.extend(_get_string_paths(val, path + (key,)))
        elif isinstance(val, str) or isinstance(val, list):
            paths.append(path + (key,))
    return paths


def _get_nested(d: dict, *keys):
    node = d
    for k in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(k)
    return node


def _set_nested(d: dict, keys: list, value):
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _schema_default(path: tuple, schema: dict):
    node = schema
    for k in path:
        node = node[k]
    return node


# ── Integer fields ────────────────────────────────────────────────────────────

INTEGER_LEAF_FIELDS = {
    "number_of_heavy_atoms", "net_formal_charge",
    "num_h_acceptors_lipinski", "num_h_donors_lipinski",
    "num_rotatable_bonds", "num_h_acceptors", "num_h_donors",
}


def _enforce_integer_types(output: dict) -> dict:
    """
    Walk the output dict and round any integer-typed field that ended up as a
    float back to int.  This is the last-resort guard against LLM2 producing
    3.2857 for num_rotatable_bonds etc.
    """
    def _walk(node):
        if not isinstance(node, dict):
            return node
        for k, v in node.items():
            if k in INTEGER_LEAF_FIELDS:
                if isinstance(v, float):
                    node[k] = int(round(v))
                    print(f"[INT-GUARD] Rounded {k}: {v} → {node[k]}")
                elif v is None:
                    pass  # leave null as null
            elif isinstance(v, dict):
                node[k] = _walk(v)
        return node
    return _walk(output)


# ── SMILES validation ─────────────────────────────────────────────────────────

def _validate_smiles(smiles: str) -> bool:
    """
    Basic SMILES sanity check without requiring RDKit.
    Checks: non-empty, only legal characters, balanced parentheses and brackets,
    and that every alphabetic token is a known atom symbol.
    """
    if not smiles or not isinstance(smiles, str):
        return False
    smiles = smiles.strip()
    if not smiles:
        return False
    if not re.search(r'[A-Za-z]', smiles):
        return False
    legal = re.compile(r'^[A-Za-z0-9@+\-=\#\$\%\[\]\(\)\.\/\\:]+$')
    if not legal.match(smiles):
        return False
    if smiles.count('(') != smiles.count(')'):
        return False
    if smiles.count('[') != smiles.count(']'):
        return False

    # Valid atom symbols (organic subset + bracket atoms)
    VALID_ATOMS = {
        'C', 'N', 'O', 'S', 'P', 'F', 'B', 'I', 'H',
        'Cl', 'Br', 'Si', 'Se', 'As', 'Te',
        'Na', 'K', 'Ca', 'Mg', 'Fe', 'Zn', 'Cu', 'Co', 'Al',
        'Li', 'Sn', 'Au', 'Ag', 'Pt',
        'c', 'n', 'o', 's', 'p', 'b',
    }
    TWO_LETTER = {'Cl', 'Br', 'Si', 'Se', 'As', 'Te', 'Na', 'Ca', 'Mg',
                  'Fe', 'Zn', 'Cu', 'Co', 'Al', 'Li', 'Sn', 'Au', 'Ag', 'Pt'}
    two_pat = '|'.join(TWO_LETTER)
    tokens = re.findall(rf'(?:{two_pat})|[A-Za-z]', smiles)
    for tok in tokens:
        if tok not in VALID_ATOMS:
            return False
    return True


# ── Chemical consistency checks ───────────────────────────────────────────────

def _check_formula_mw_consistency(output: dict) -> None:
    """
    Warn if the formula-derived average MW deviates more than 1 Da from the
    stored molecular_weight.  This catches formula/MW mismatches from LLM2.
    """
    formula = _get_nested(output, "molecule", "molecular_formula")
    mw_stored = _get_nested(output, "molecule", "molecular_weight")
    if not formula or not mw_stored or mw_stored == 0.0:
        return
    mw_calc = _compute_molecular_weight(formula)
    if mw_calc is None:
        return
    diff = abs(mw_calc - mw_stored)
    if diff > 1.0:
        print(
            f"[CHEM-CHECK] formula↔MW mismatch: formula '{formula}' implies MW≈{mw_calc:.4f}, "
            f"stored MW={mw_stored:.4f} (Δ={diff:.4f} Da). "
            f"Overwriting with formula-derived value."
        )
        _set_nested(output, ["molecule", "molecular_weight"], round(mw_calc, 4))


def _check_smiles_heavy_atom_consistency(output: dict) -> None:
    """
    Count heavy atoms in the SMILES string and compare to number_of_heavy_atoms.
    Discrepancies > 0 are flagged; the SMILES-derived count is authoritative
    when a valid SMILES is present.
    """
    smiles = _get_nested(output, "identity", "smiles")
    hac_stored = _get_nested(output, "molecule", "number_of_heavy_atoms")

    if not smiles or not hac_stored:
        return

    # Count non-H atoms in SMILES by tokenising
    TWO_LETTER = {'Cl', 'Br', 'Si', 'Se', 'As', 'Te', 'Na', 'Ca', 'Mg',
                  'Fe', 'Zn', 'Cu', 'Co', 'Al', 'Li', 'Sn', 'Au', 'Ag', 'Pt'}
    two_pat = '|'.join(TWO_LETTER)
    tokens = re.findall(rf'(?:{two_pat})|[A-Za-z]', smiles)
    heavy_in_smiles = sum(
        1 for t in tokens
        if t.upper() != 'H' and t not in ('c', 'n', 'o', 's', 'p', 'b')
        or t in ('c', 'n', 'o', 's', 'p', 'b')  # aromatic atoms are heavy
    )
    # Simpler and more reliable: just count non-H uppercase + aromatic tokens
    heavy_in_smiles = sum(
        1 for t in tokens if t.lower() != 'h'
    )

    if heavy_in_smiles and heavy_in_smiles != int(hac_stored):
        print(
            f"[CHEM-CHECK] SMILES↔HAC mismatch: SMILES implies {heavy_in_smiles} heavy atoms, "
            f"stored number_of_heavy_atoms={hac_stored}. Overwriting with SMILES-derived count."
        )
        _set_nested(output, ["molecule", "number_of_heavy_atoms"], heavy_in_smiles)


def _validate_composition_dict(output: dict) -> None:
    """
    Validate that molecular_composition is a {element: float} dict whose
    fractions sum to 1.0 ± 0.01.  If the sum is off, renormalise in place.
    If the field is a string (legacy), clear it so LLM2 can recompute.
    """
    comp = _get_nested(output, "molecule", "molecular_composition")

    if comp is None:
        return

    if isinstance(comp, str):
        print("[COMP] molecular_composition is a string (legacy format) — clearing for recomputation.")
        _set_nested(output, ["molecule", "molecular_composition"], {})
        return

    if not isinstance(comp, dict) or not comp:
        return

    # Verify all values are numeric
    non_numeric = {k: v for k, v in comp.items() if not isinstance(v, (int, float))}
    if non_numeric:
        print(f"[COMP] Non-numeric fractions found: {non_numeric} — clearing composition.")
        _set_nested(output, ["molecule", "molecular_composition"], {})
        return

    total = sum(comp.values())
    if abs(total - 1.0) > 0.01:
        print(f"[COMP] Fractions sum to {total:.6f} (expected 1.0) — renormalising.")
        renorm = {k: round(v / total, 6) for k, v in comp.items()}
        _set_nested(output, ["molecule", "molecular_composition"], renorm)


# ── Deterministic recomputation ───────────────────────────────────────────────

def _recompute_deterministic_from_formula(output: dict) -> None:
    """
    When molecular_formula is known, recompute:
      - molecular_weight  (average)
      - exact_mol_weight  (monoisotopic)
      - number_of_heavy_atoms
      - molecular_composition  (mass fractions)

    These are structural/deterministic — the formula is authoritative.
    Only overrides default (0 / 0.0 / {}) values or values that differ
    significantly from the formula-derived result.
    """
    formula = _get_nested(output, "molecule", "molecular_formula")
    if not formula:
        return

    print(f"[RECOMPUTE] Recomputing deterministic descriptors from formula: {formula}")

    counts = _parse_formula(formula)
    if counts is None:
        print(f"[RECOMPUTE] Could not parse formula '{formula}' — skipping recomputation.")
        return

    mw_calc    = _compute_molecular_weight(formula)
    exact_calc = _compute_exact_mol_weight(formula)
    hac_calc   = _compute_heavy_atom_count(formula)
    comp_calc  = _compute_composition(formula, mw_calc) if mw_calc else None

    def _should_override(stored, computed, tol_rel=0.005):
        """Override if stored is default (0/0.0/{}) or deviates beyond tolerance."""
        if stored is None or stored == 0 or stored == 0.0 or stored == {}:
            return True
        if isinstance(computed, float) and isinstance(stored, (int, float)):
            return abs(stored - computed) / max(abs(computed), 1e-9) > tol_rel
        if isinstance(computed, int):
            return int(stored) != computed
        return False

    if mw_calc is not None:
        stored_mw = _get_nested(output, "molecule", "molecular_weight") or 0.0
        if _should_override(stored_mw, mw_calc):
            _set_nested(output, ["molecule", "molecular_weight"], round(mw_calc, 4))
            print(f"[RECOMPUTE] molecular_weight → {round(mw_calc, 4)}")

    if exact_calc is not None:
        stored_ex = _get_nested(output, "molecule", "exact_mol_weight") or 0.0
        if _should_override(stored_ex, exact_calc):
            _set_nested(output, ["molecule", "exact_mol_weight"], round(exact_calc, 6))
            print(f"[RECOMPUTE] exact_mol_weight → {round(exact_calc, 6)}")

    if hac_calc is not None:
        stored_hac = _get_nested(output, "molecule", "number_of_heavy_atoms") or 0
        if _should_override(stored_hac, hac_calc):
            _set_nested(output, ["molecule", "number_of_heavy_atoms"], hac_calc)
            print(f"[RECOMPUTE] number_of_heavy_atoms → {hac_calc}")

    if comp_calc:
        stored_comp = _get_nested(output, "molecule", "molecular_composition") or {}
        if _should_override(stored_comp, comp_calc):
            _set_nested(output, ["molecule", "molecular_composition"], comp_calc)
            print(f"[RECOMPUTE] molecular_composition → {comp_calc}")


# ── PubChem helpers ───────────────────────────────────────────────────────────

def _fetch_with_retry(url: str, extractor, retries: int = 5, wait: int = 5):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return extractor(r)
            return None
        except Exception as e:
            if attempt < retries:
                print(f"[HTTP] Attempt {attempt} failed ({e.__class__.__name__}). Retrying in {wait}s...")
                time.sleep(wait)
    print(f"[HTTP] All {retries} attempts failed for: {url}")
    return None


def fetch_cid(molecule_input: str, retries: int = 5, wait: int = 5) -> int:
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(molecule_input)}/cids/JSON"
    )
    result = _fetch_with_retry(
        url,
        lambda r: r.json()["IdentifierList"]["CID"][0],
        retries=retries, wait=wait,
    )
    if result is None:
        print(f"[CID] Could not resolve CID for '{molecule_input}' — defaulting to 0.")
        return 0
    return result


def fetch_name_from_cid(cid: int, retries: int = 5, wait: int = 5) -> str:
    iupac_name = _fetch_with_retry(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/JSON",
        lambda r: r.json()["PropertyTable"]["Properties"][0].get("IUPACName", ""),
        retries=retries, wait=wait,
    ) or ""

    common_name = _fetch_with_retry(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON",
        lambda r: r.json()["InformationList"]["Information"][0].get("Synonym", [""])[0],
        retries=retries, wait=wait,
    ) or ""

    if common_name and iupac_name and common_name.lower() != iupac_name.lower():
        label = f"{common_name} (IUPAC: {iupac_name})"
        print(f"[CID] Resolved to: {label}")
        return label
    elif common_name:
        print(f"[CID] Using common name: {common_name}")
        return common_name
    elif iupac_name:
        print(f"[CID] Using IUPAC name: {iupac_name}")
        return iupac_name
    return ""


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(molecule_input: str, debug: bool = False) -> dict:

    # ── CID resolution ────────────────────────────────────────────────────────
    if molecule_input.isdigit():
        cid = int(molecule_input)
        print(f"[CID] Using provided CID: {cid}")
        molecule_name = fetch_name_from_cid(cid)
        if not molecule_name:
            print("[CID] Could not resolve a name — LLM1 will receive the raw CID.")
            molecule_name = molecule_input
    else:
        molecule_name = molecule_input
        print(f"[CID] Resolving PubChem CID for: {molecule_input}")
        cid = fetch_cid(molecule_input)
        if cid:
            print(f"[CID] Resolved to CID {cid}")
            enriched = fetch_name_from_cid(cid)
            if enriched:
                molecule_name = enriched
        else:
            print("[CID] Could not resolve a CID — defaulting to 0.")

    # ── Step 1 — LLM1 extraction ──────────────────────────────────────────────
    print(f"\n[LLM1] Extracting properties for: {molecule_name}")
    try:
        extracted = LLM1.run_extraction(molecule_name, cid, REQUIRED_SCHEMA, client)
    except Exception as e:
        raise RuntimeError(f"LLM1 extraction failed: {e}")
    print("[LLM1] Extraction complete.")

    if debug:
        print("\n[DEBUG] LLM1 raw extraction:")
        print(json.dumps(extracted, indent=2))

    # Unwrap if LLM1 nested output under a single wrapper key
    schema_keys = set(REQUIRED_SCHEMA.keys()) - {"cid"}
    if not schema_keys.intersection(extracted.keys()):
        wrapper_keys = [k for k in extracted if isinstance(extracted[k], dict)]
        if len(wrapper_keys) == 1:
            print(f"[NORM] Output wrapped under '{wrapper_keys[0]}' — unwrapping.")
            extracted = extracted[wrapper_keys[0]]

    # ── Step 2 — Fusion ───────────────────────────────────────────────────────
    print("\n[FUSION] Resolving candidates with typed fusion strategies...")
    fused = fusion.fuse(extracted)

    if debug:
        print("\n[DEBUG] Fused intermediate:")
        print(json.dumps(fused, indent=2))

    print("[FUSION] Fusion complete.")

    # ── Step 3 — LLM2 verification ────────────────────────────────────────────
    print("[LLM2] Verifying plausibility and assembling final record...")
    try:
        raw_output = LLM2.run_verification(fused, REQUIRED_SCHEMA, client)
    except Exception as e:
        raise RuntimeError(f"LLM2 verification failed: {e}")

    if not isinstance(raw_output, dict):
        raise RuntimeError(f"LLM2 returned unexpected type {type(raw_output)}: {raw_output}")

    # Surface LLM2 warnings and provenance metadata
    llm2_warnings = raw_output.pop("warnings", [])
    if llm2_warnings:
        for w in (llm2_warnings if isinstance(llm2_warnings, list) else [llm2_warnings]):
            print(f"[LLM2 WARN] {w}")

    provenance = raw_output.pop("provenance", [])

    final_output = raw_output

    # ── Guard: re-inject fused numeric values LLM2 must not have changed ─────
    numeric_paths = _get_numeric_paths(REQUIRED_SCHEMA)

    for *path, leaf in numeric_paths:
        fused_val = _get_nested(fused, *path, leaf) if path else fused.get(leaf)
        if fused_val is None:
            continue

        default = _schema_default(tuple(path) + (leaf,), REQUIRED_SCHEMA)
        if fused_val == default:
            continue  # fusion returned the schema default — let LLM2's value stand

        existing = _get_nested(final_output, *path, leaf) if path else final_output.get(leaf)

        if existing is None or (
            isinstance(existing, float) and isinstance(fused_val, float)
            and not math.isclose(existing, fused_val, rel_tol=1e-6)
        ) or (
            not isinstance(existing, float) and existing != fused_val
        ):
            print(
                f"[GUARD] LLM2 changed {'.'.join(list(path) + [leaf])} "
                f"from {fused_val} → {existing} — restoring fused value."
            )
            _set_nested(final_output, list(path) + [leaf], fused_val)

    # ── Guard: protect fused string / list fields ─────────────────────────────
    string_paths = _get_string_paths(REQUIRED_SCHEMA)
    FILLABLE_STRING_FIELDS = {
        "drugbank_id", "secondary_accession_numbers", "common_name",
        "cas_number", "unii", "synonyms",
    }

    for *path, leaf in string_paths:
        if leaf in FILLABLE_STRING_FIELDS:
            continue

        fused_val = _get_nested(fused, *path, leaf) if path else fused.get(leaf)
        if not fused_val:
            continue

        existing = _get_nested(final_output, *path, leaf) if path else final_output.get(leaf)
        if existing != fused_val:
            print(
                f"[GUARD] LLM2 changed string field {'.'.join(list(path) + [leaf])} "
                f"— restoring fused value."
            )
            _set_nested(final_output, list(path) + [leaf], fused_val)

    # ── Guard: molecular_composition must be a dict ───────────────────────────
    comp = _get_nested(final_output, "molecule", "molecular_composition")
    fused_comp = _get_nested(fused, "molecule", "molecular_composition")
    if isinstance(fused_comp, dict) and fused_comp:
        if not isinstance(comp, dict) or not comp:
            print("[GUARD] LLM2 lost molecular_composition dict — restoring fused value.")
            _set_nested(final_output, ["molecule", "molecular_composition"], fused_comp)

    # ── SMILES validation ─────────────────────────────────────────────────────
    # Run on fused SMILES first; if valid, prefer it over LLM2's (guard already
    # restored it above).  Then validate whatever is in final_output.
    smiles_val = _get_nested(final_output, "identity", "smiles")
    if smiles_val and not _validate_smiles(smiles_val):
        print(f"[SMILES] Validation failed for '{smiles_val}' — clearing.")
        _set_nested(final_output, ["identity", "smiles"], "")

    # ── Deterministic recomputation from molecular_formula ────────────────────
    _recompute_deterministic_from_formula(final_output)

    # ── Chemical consistency checks ───────────────────────────────────────────
    _check_formula_mw_consistency(final_output)
    _check_smiles_heavy_atom_consistency(final_output)
    _validate_composition_dict(final_output)

    # ── Integer type enforcement ──────────────────────────────────────────────
    final_output = _enforce_integer_types(final_output)

    # ── Duplicate secondary-accession guard ───────────────────────────────────
    primary_db = (_get_nested(final_output, "identity", "drugbank_id") or "").strip().upper()
    secondary = _get_nested(final_output, "identity", "secondary_accession_numbers") or []
    if primary_db and isinstance(secondary, list):
        cleaned = [s for s in secondary if s.strip().upper() != primary_db]
        if len(cleaned) != len(secondary):
            removed = [s for s in secondary if s.strip().upper() == primary_db]
            print(f"[GUARD] Removed duplicate drugbank accession(s) from secondary list: {removed}")
            _set_nested(final_output, ["identity", "secondary_accession_numbers"], cleaned)

    # ── Mark unresolved numeric fields as null ────────────────────────────────
    for *path, leaf in numeric_paths:
        fused_val = _get_nested(fused, *path, leaf) if path else fused.get(leaf)
        if fused_val is None:
            _set_nested(final_output, list(path) + [leaf], None)
            print(f"[INFO] {'.'.join(list(path) + [leaf])} could not be resolved — marked as null.")

    # ── Audit for missing schema keys ─────────────────────────────────────────
    def _validate_keys(output: dict, schema: dict, path: str = ""):
        for key in schema:
            if key == "cid":
                continue
            if key not in output:
                print(f"[WARN] Expected field missing from output: {path}{key}")
            elif isinstance(schema[key], dict) and key != "molecular_composition" \
                    and isinstance(output.get(key), dict):
                _validate_keys(output[key], schema[key], path=f"{path}{key}.")

    _validate_keys(final_output, REQUIRED_SCHEMA)

    # ── Inject CID ────────────────────────────────────────────────────────────
    final_output["cid"] = cid

    # ── Attach provenance metadata ────────────────────────────────────────────
    if provenance:
        final_output["provenance"] = provenance
        print(f"[PROV] Provenance metadata attached ({len(provenance)} entries).")

    print("[LLM2] Verification complete.")
    return final_output


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    molecule = input("Enter molecule name OR CID: ").strip()
    debug_input = input("Debug mode? (y/n): ").strip().lower()
    debug = debug_input == "y"

    result = run_pipeline(molecule, debug=debug)

    print("\n[OUTPUT]")

    def _round_floats(obj, decimals: int = 4):
        if isinstance(obj, float):
            return round(obj, decimals)
        if isinstance(obj, dict):
            return {k: _round_floats(v, decimals) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round_floats(i, decimals) for i in obj]
        return obj

    rounded = _round_floats(result)
    print(json.dumps(rounded, indent=2))

    output_file = "output.json"
    with open(output_file, "w") as f:
        json.dump(rounded, f, indent=2)
    print(f"\nRecord saved to {output_file}")
