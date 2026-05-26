import math
import json
import os
import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

import LLM1
import LLM2
import fusion
from schema import REQUIRED_SCHEMA

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not found. Make sure it is set in your .env file.")

client = InferenceClient(token=HF_TOKEN)

def _get_numeric_paths(schema: dict, path: tuple = ()) -> list:
    """
    Walk schema and return all paths to numeric leaf nodes (int or float).
    Excludes 'cid' which is handled separately.
    e.g. → [("molecule", "logP"), ("interaction_profile", "activity_value"), ...]
    """
    paths = []
    for key, val in schema.items():
        if key == "cid":
            continue
        if isinstance(val, dict):
            paths.extend(_get_numeric_paths(val, path + (key,)))
        elif isinstance(val, (int, float)):
            paths.append(path + (key,))
    return paths


# ── PubChem CID lookup ────────────────────────────────────────────────────────
def fetch_cid(molecule_input: str) -> int:
    """Resolve molecule name or SMILES to a PubChem CID. Returns 0 if not found."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(molecule_input)}/cids/JSON"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()["IdentifierList"]["CID"][0]

        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(molecule_input)}/cids/JSON"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()["IdentifierList"]["CID"][0]

    except Exception as e:
        print(f"[CID] Lookup failed: {e}")

    return 0

# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(molecule_input: str, debug: bool = False) -> dict:

    # CID resolution
    if molecule_input.isdigit():
        cid = int(molecule_input)
        print(f"[CID] Using provided CID: {cid}")
    else:
        print(f"[CID] Looking up PubChem CID for: {molecule_input}")
        cid = fetch_cid(molecule_input)
        if cid:
            print(f"[CID] Found: {cid}")
        else:
            print("[CID] Not found, defaulting to 0.")

    # STEP 1: Extraction
    print(f"\n[LLM1] Extracting properties for: {molecule_input}")
    try:
        extracted = LLM1.run_extraction(molecule_input, REQUIRED_SCHEMA, client)
    except Exception as e:
        raise RuntimeError(f"LLM1 extraction failed: {e}")
    print("[LLM1] Done.")

    if debug:
        print("\n── LLM1 RAW EXTRACTION ──────────────────────────────────────────────────")
        print(json.dumps(extracted, indent=2))

    # Normalize: LLM1 sometimes wraps output under a CID key instead of flat schema
    # e.g. {"5754": {"molecule": ...}} → {"molecule": ...}
    schema_keys = set(REQUIRED_SCHEMA.keys()) - {"cid"}
    if not schema_keys.intersection(extracted.keys()):
        # No schema keys at top level — check if there's exactly one wrapper key
        wrapper_keys = [k for k in extracted if isinstance(extracted[k], dict)]
        if len(wrapper_keys) == 1:
            print(f"[NORM] LLM1 wrapped output under key '{wrapper_keys[0]}' — unwrapping")
            extracted = extracted[wrapper_keys[0]]

    # STEP 2: Fusion
    print("\n[FUSION] Running outlier removal + confidence-weighted mean...")
    fused = fusion.fuse(extracted)

    if debug:
        print("\n── FUSED INTERMEDIATE ───────────────────────────────────────────────────")
        print(json.dumps(fused, indent=2))

    print("[FUSION] Done.")

    # STEP 3: Verification + Final JSON build
    print("[LLM2] Verifying and building final record...")
    try:
        raw_output = LLM2.run_verification(fused, REQUIRED_SCHEMA, client)
    except Exception as e:
        raise RuntimeError(f"LLM2 verification failed: {e}")

    # LLM2 now returns a parsed dict directly after internal JSON cleaning
    if not isinstance(raw_output, dict):
        raise RuntimeError(f"LLM2 returned unexpected type {type(raw_output)}: {raw_output}")
    final_output = raw_output

    # Derive protected numeric paths directly from schema — updates automatically when schema changes
    numeric_paths = _get_numeric_paths(REQUIRED_SCHEMA)

    # Guard: re-inject numeric values from fused data — LLM2 must not alter these
    for *path, leaf in numeric_paths:
        try:
            src = fused
            for key in path:
                src = src[key]
            fused_val = src[leaf]
        except (KeyError, TypeError):
            continue  # field not in fused at all; nothing to protect

        if fused_val is None:
            continue

        try:
            dst = final_output
            for key in path:
                dst = dst[key]  # don't create — if LLM2 dropped the section, warn below
            existing = dst.get(leaf)
            if not (isinstance(existing, float) and isinstance(fused_val, float)
                    and math.isclose(existing, fused_val, rel_tol=1e-9)):
                if existing != fused_val:
                    print(f"[GUARD] LLM2 altered {'.'.join(path + [leaf])}: "
                          f"{existing} → restoring {fused_val}")
            dst[leaf] = fused_val
        except (KeyError, TypeError):
            print(f"[WARN] Could not restore {'.'.join(path + [leaf])}: path missing in LLM2 output")

    # Null sentinel: where fusion returned None, override LLM2's schema-default zero
    # so the final output honestly signals "unresolvable" rather than a fake 0.0
    for *path, leaf in numeric_paths:
        try:
            src = fused
            for key in path:
                src = src[key]
            if src.get(leaf) is None:
                dst = final_output
                for key in path:
                    dst = dst[key]
                dst[leaf] = None
                print(f"[INFO] {'.'.join(path + [leaf])} was unresolvable — set to null in output")
        except (KeyError, TypeError):
            pass  # path missing entirely; nothing to nullify

    # Validate output keys match schema (top-level and one level deep)
    def _validate_keys(output: dict, schema: dict, path: str = ""):
        for key in schema:
            if key == "cid":
                continue  # injected separately
            if key not in output:
                print(f"[WARN] Missing key in final output: {path}{key}")
            elif isinstance(schema[key], dict) and isinstance(output.get(key), dict):
                _validate_keys(output[key], schema[key], path=f"{path}{key}.")
    _validate_keys(final_output, REQUIRED_SCHEMA)
    
    final_output["cid"] = cid
    print("[LLM2] Done.")
    return final_output


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    molecule = input("Enter molecule name / SMILES / CID: ").strip()
    debug_input = input("Debug mode? Show LLM1 extraction + fused intermediate? (y/n): ").strip().lower()
    debug = debug_input == "y"

    result = run_pipeline(molecule, debug=debug)

    print("\n── FINAL OUTPUT ─────────────────────────────────────────────────────────")
    print(json.dumps(result, indent=2))

    def _round_floats(obj, decimals=4):
        if isinstance(obj, float):
            return round(obj, decimals)
        if isinstance(obj, dict):
            return {k: _round_floats(v, decimals) for k, v in obj.items()}
        return obj

    output_file = "output.json"
    with open(output_file, "w") as f:
        json.dump(_round_floats(result), f, indent=2)
    print(f"\nSaved to {output_file}")