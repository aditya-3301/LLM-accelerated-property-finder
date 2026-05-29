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
from drug_descriptors import REQUIRED_SCHEMA

# ── Auth ──────────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not found. Make sure it is set in your .env file.")

client = InferenceClient(token=HF_TOKEN)


# ── Schema helpers ────────────────────────────────────────────────────────────

def _get_numeric_paths(schema: dict, path: tuple = ()) -> list:
    """Return list of tuples (section_key, ..., leaf_key) for all numeric leaves."""
    paths = []
    for key, val in schema.items():
        if key == "cid":
            continue
        if isinstance(val, dict):
            paths.extend(_get_numeric_paths(val, path + (key,)))
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            paths.append(path + (key,))
    return paths


def _get_string_paths(schema: dict, path: tuple = ()) -> list:
    """Return list of tuples for all string/list leaves (for the string guard)."""
    paths = []
    for key, val in schema.items():
        if key == "cid":
            continue
        if isinstance(val, dict):
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


# ── SMILES validation ─────────────────────────────────────────────────────────

def _validate_smiles(smiles: str) -> bool:
    """
    Basic SMILES sanity check without requiring RDKit.
    Checks: non-empty, only legal characters, balanced parentheses and brackets,
    and that it doesn't look like a peptide sequence or plain English word.
    """
    if not smiles or not isinstance(smiles, str):
        return False
    smiles = smiles.strip()
    if not smiles:
        return False
    # Must contain at least one atom symbol
    if not re.search(r'[A-Za-z]', smiles):
        return False
    # SMILES legal character set
    legal = re.compile(r'^[A-Za-z0-9@+\-=\#\$\%\[\]\(\)\.\/\\:]+$')
    if not legal.match(smiles):
        return False
    # Balanced parentheses
    if smiles.count('(') != smiles.count(')'):
        return False
    # Balanced brackets
    if smiles.count('[') != smiles.count(']'):
        return False
    # Extract every alphabetic atom token from the SMILES string.
    # Valid SMILES atom symbols (case-sensitive):
    #   Uppercase: B, C, N, O, S, P, F, I  (and two-letter: Cl, Br, Si, Se, As, Te)
    #   Lowercase (aromatic): b, c, n, o, s, p
    # Any token not in this set means the string is not valid SMILES.
    VALID_ATOMS = {
        'C', 'N', 'O', 'S', 'P', 'F', 'B', 'I',
        'Cl', 'Br', 'Si', 'Se', 'As', 'Te',
        'c', 'n', 'o', 's', 'p', 'b',
    }
    TWO_LETTER = {'Cl', 'Br', 'Si', 'Se', 'As', 'Te'}
    # Tokenize: greedily consume two-letter symbols first, then single letters.
    # Use a regex that tries two-letter known symbols before single letters.
    two_pat = '|'.join(TWO_LETTER)
    tokens = re.findall(rf'(?:{two_pat})|[A-Za-z]', smiles)
    for tok in tokens:
        if tok not in VALID_ATOMS:
            return False
    return True


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
    """Resolve a molecule name to a PubChem CID."""
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(molecule_input)}/cids/JSON"
    )
    result = _fetch_with_retry(
        url,
        lambda r: r.json()["IdentifierList"]["CID"][0],
        retries=retries,
        wait=wait,
    )
    if result is None:
        print(f"[CID] Could not resolve CID for '{molecule_input}' — defaulting to 0.")
        return 0
    return result


def fetch_name_from_cid(cid: int, retries: int = 5, wait: int = 5) -> str:
    """
    Resolve a PubChem CID to the best human-readable name for LLM1.
    Prefers a short IUPAC name; falls back to first synonym (common name).
    Always returns common_name + IUPAC hint so the LLM has both.
    """
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

    # Always give LLM1 both names when we have them — avoids ambiguity
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
            # Enrich the name with IUPAC hint if entered as common name
            enriched = fetch_name_from_cid(cid)
            if enriched:
                molecule_name = enriched
        else:
            print("[CID] Could not resolve a CID — defaulting to 0.")

    # ── Step 1 — LLM1 extraction ──────────────────────────────────────────────
    print(f"\n[LLM1] Extracting properties for: {molecule_name}")
    try:
        # Pass cid explicitly so LLM1 can anchor to the PubChem entry
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
    print("\n[FUSION] Removing outliers and computing confidence-weighted values...")
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

    # Surface any warnings LLM2 emitted (it's allowed a top-level "warnings" key)
    llm2_warnings = raw_output.pop("warnings", [])
    if llm2_warnings:
        for w in (llm2_warnings if isinstance(llm2_warnings, list) else [llm2_warnings]):
            print(f"[LLM2 WARN] {w}")

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

        # Only restore if LLM2 meaningfully diverged from the fused value
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

    # ── Guard: protect fused string / list fields LLM2 must not corrupt ──────
    string_paths = _get_string_paths(REQUIRED_SCHEMA)
    # Fields LLM2 is allowed to fill/override (empty in fused data)
    FILLABLE_STRING_FIELDS = {
        "drugbank_id", "secondary_accession_numbers", "common_name",
        "cas_number", "unii", "synonyms", "activity_type", "activity_unit",
        "carcinogenicity", "immunotoxicity", "mutagenicity", "cytotoxicity",
    }

    for *path, leaf in string_paths:
        if leaf in FILLABLE_STRING_FIELDS:
            continue  # LLM2 is allowed to set these

        fused_val = _get_nested(fused, *path, leaf) if path else fused.get(leaf)
        if not fused_val:
            continue  # nothing to protect

        existing = _get_nested(final_output, *path, leaf) if path else final_output.get(leaf)
        if existing != fused_val:
            print(
                f"[GUARD] LLM2 changed string field {'.'.join(list(path) + [leaf])} "
                f"— restoring fused value."
            )
            _set_nested(final_output, list(path) + [leaf], fused_val)

    # ── SMILES validation ─────────────────────────────────────────────────────
    smiles_val = _get_nested(final_output, "identity", "smiles")
    if smiles_val and not _validate_smiles(smiles_val):
        print(f"[SMILES] Validation failed for '{smiles_val}' — clearing.")
        _set_nested(final_output, ["identity", "smiles"], "")

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
            elif isinstance(schema[key], dict) and isinstance(output.get(key), dict):
                _validate_keys(output[key], schema[key], path=f"{path}{key}.")

    _validate_keys(final_output, REQUIRED_SCHEMA)

    # ── Inject CID ────────────────────────────────────────────────────────────
    final_output["cid"] = cid
    print("[LLM2] Verification complete.")
    return final_output


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    molecule = input("Enter molecule name OR CID: ").strip()
    debug_input = input("Debug mode? (y/n): ").strip().lower()
    debug = debug_input == "y"

    result = run_pipeline(molecule, debug=debug)

    print("\n[OUTPUT]")
    print(json.dumps(result, indent=2))

    def _round_floats(obj, decimals: int = 4):
        if isinstance(obj, float):
            return round(obj, decimals)
        if isinstance(obj, dict):
            return {k: _round_floats(v, decimals) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round_floats(i, decimals) for i in obj]
        return obj

    output_file = "output.json"
    with open(output_file, "w") as f:
        json.dump(_round_floats(result), f, indent=2)
    print(f"\nRecord saved to {output_file}")