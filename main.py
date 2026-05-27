import math
import json
import os
import time
import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

import LLM1
import LLM2
import fusion
from drug_descriptors import REQUIRED_SCHEMA

# Load environment variables and authenticate with Hugging Face.
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not found. Make sure it is set in your .env file.")

client = InferenceClient(token=HF_TOKEN)


def _get_numeric_paths(schema: dict, path: tuple = ()) -> list:
    paths = []
    for key, val in schema.items():
        if key == "cid":
            continue
        if isinstance(val, dict):
            paths.extend(_get_numeric_paths(val, path + (key,)))
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            paths.append(path + (key,))
    return paths


def fetch_cid(molecule_input: str, retries: int = 5, wait: int = 5) -> int:
    """Resolve a molecule name or SMILES to a PubChem CID. Retries on DNS/network failure."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(molecule_input)}/cids/JSON"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()["IdentifierList"]["CID"][0]
            return 0  # Bad status but not a network error — don't retry
        except Exception as e:
            if attempt < retries:
                print(f"[CID] Attempt {attempt} failed ({e.__class__.__name__}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[CID] All {retries} attempts failed. Could not resolve CID.")
    return 0



def fetch_name_from_cid(cid: int, retries: int = 5, wait: int = 5) -> str:
    """
    Resolve a PubChem CID to the best human-readable name for LLM1.
    Strategy:
    - Fetch both the IUPAC name and the first synonym in parallel
    - If IUPAC name is short enough (<=60 chars), use it — it's unambiguous
    - If IUPAC name is long/complex, use the first synonym (common name) instead
    - Falls back to whatever is available if one request fails
    Retries on DNS/network failure.
    """
    iupac_name = ""
    common_name = ""

    def _fetch_with_retry(url, extractor):
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    return extractor(r)
                return ""  # bad status, don't retry
            except Exception as e:
                if attempt < retries:
                    print(f"[CID] Attempt {attempt} failed ({e.__class__.__name__}). Retrying in {wait}s...")
                    time.sleep(wait)
        print(f"[CID] All {retries} attempts failed for URL.")
        return ""

    iupac_name = _fetch_with_retry(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/JSON",
        lambda r: r.json()["PropertyTable"]["Properties"][0].get("IUPACName", "")
    )

    common_name = _fetch_with_retry(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON",
        lambda r: r.json()["InformationList"]["Information"][0].get("Synonym", [""])[0]
    )

    # Prefer common name if IUPAC is too long for the model to parse reliably
    IUPAC_LENGTH_LIMIT = 60
    if iupac_name and len(iupac_name) <= IUPAC_LENGTH_LIMIT:
        print(f"[CID] Using IUPAC name: {iupac_name}")
        return iupac_name
    elif common_name:
        if iupac_name:
            print(f"[CID] IUPAC name too complex ({len(iupac_name)} chars) — using common name: {common_name}")
        else:
            print(f"[CID] Using common name: {common_name}")
        return common_name
    elif iupac_name:
        print(f"[CID] Only IUPAC name available (long): {iupac_name}")
        return iupac_name

    return ""


def run_pipeline(molecule_input: str, debug: bool = False) -> dict:

    # CID resolution
    if molecule_input.isdigit():
        cid = int(molecule_input)
        print(f"[CID] Using provided CID: {cid}")
        molecule_name = fetch_name_from_cid(cid)
        if molecule_name:
            print(f"[CID] Resolved to name: {molecule_name}")
        else:
            print("[CID] Could not resolve a name — LLM1 will receive the raw CID.")
            molecule_name = molecule_input
    else:
        molecule_name = molecule_input
        print(f"[CID] Resolving PubChem CID for: {molecule_input}")
        cid = fetch_cid(molecule_input)
        if cid:
            print(f"[CID] Resolved to CID {cid}")
        else:
            print("[CID] Could not resolve a CID — defaulting to 0.")

    # Step 1 — LLM1 extracts using the resolved molecule name
    print(f"\n[LLM1] Extracting properties for: {molecule_name}")
    try:
        extracted = LLM1.run_extraction(molecule_name, REQUIRED_SCHEMA, client)
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
            print(f"[NORM] Output was wrapped under key '{wrapper_keys[0]}' — unwrapping.")
            extracted = extracted[wrapper_keys[0]]

    # Step 2 — Fusion
    print("\n[FUSION] Removing outliers and computing confidence-weighted values...")
    fused = fusion.fuse(extracted)

    if debug:
        print("\n[DEBUG] Fused intermediate:")
        print(json.dumps(fused, indent=2))

    print("[FUSION] Fusion complete.")

    # Step 3 — LLM2 verification
    print("[LLM2] Verifying plausibility and assembling final record...")
    try:
        raw_output = LLM2.run_verification(fused, REQUIRED_SCHEMA, client)
    except Exception as e:
        raise RuntimeError(f"LLM2 verification failed: {e}")

    if not isinstance(raw_output, dict):
        raise RuntimeError(f"LLM2 returned unexpected type {type(raw_output)}: {raw_output}")
    final_output = raw_output

    numeric_paths = _get_numeric_paths(REQUIRED_SCHEMA)

    def _schema_default(path, leaf):
        node = REQUIRED_SCHEMA
        for k in path:
            node = node[k]
        return node[leaf]

    # Re-inject fused numeric values where fusion produced a meaningful result
    for *path, leaf in numeric_paths:
        try:
            src = fused
            for key in path:
                src = src[key]
            fused_val = src[leaf]
        except (KeyError, TypeError):
            continue

        if fused_val is None:
            continue

        default = _schema_default(tuple(path), leaf)
        if fused_val == default:
            continue

        try:
            dst = final_output
            for key in path:
                dst = dst[key]
            existing = dst.get(leaf)
            if not (isinstance(existing, float) and isinstance(fused_val, float)
                    and math.isclose(existing, fused_val, rel_tol=1e-9)):
                if existing != fused_val:
                    print(f"[GUARD] LLM2 changed {'.'.join(list(path) + [leaf])} "
                          f"from {existing} to {fused_val} — restoring fused value.")
            dst[leaf] = fused_val
        except (KeyError, TypeError):
            print(f"[WARN] Could not restore {'.'.join(list(path) + [leaf])}: "
                  f"the path is missing from LLM2's output.")

    # Mark fields where fusion returned None as null in final output
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
                print(f"[INFO] {'.'.join(path + [leaf])} could not be resolved — marked as null.")
        except (KeyError, TypeError):
            pass

    # Audit for missing keys
    def _validate_keys(output: dict, schema: dict, path: str = ""):
        for key in schema:
            if key == "cid":
                continue
            if key not in output:
                print(f"[WARN] Expected field missing from output: {path}{key}")
            elif isinstance(schema[key], dict) and isinstance(output.get(key), dict):
                _validate_keys(output[key], schema[key], path=f"{path}{key}.")
    _validate_keys(final_output, REQUIRED_SCHEMA)

    final_output["cid"] = cid
    print("[LLM2] Verification complete.")
    return final_output


if __name__ == "__main__":
    molecule = input("Enter molecule name OR CID: ").strip()
    debug_input = input("Debug mode?").strip().lower()
    debug = debug_input == "y"

    result = run_pipeline(molecule, debug=debug)

    print("\n[OUTPUT]")
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
    print(f"\nRecord saved to {output_file}")