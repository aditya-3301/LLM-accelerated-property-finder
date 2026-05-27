import math
import json
import os
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
    """
    Recursively walks the schema and collects the full path to every numeric
    leaf (int or float). These paths are used later to protect fused values
    from being overwritten by LLM2. 'cid' is excluded because it is injected
    directly and never passes through the fusion engine.
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


# Resolve a molecule name or SMILES string to a PubChem CID.
def fetch_cid(molecule_input: str) -> int:
    """
    Tries a name-based lookup, returns 0 if attempt fails.
    """
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(molecule_input)}/cids/JSON"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()["IdentifierList"]["CID"][0]


    except Exception as e:
        print(f"[CID] Lookup failed: {e}")

    return 0

# Main pipeline: takes a molecule identifier and returns a validated JSON record.
def run_pipeline(molecule_input: str, debug: bool = False) -> dict:

    # CID resolution — accept a raw integer CID or resolve from name/SMILES.
    if molecule_input.isdigit():
        cid = int(molecule_input)
        print(f"[CID] Using provided CID: {cid}")
    else:
        print(f"[CID] Resolving PubChem CID for: {molecule_input}")
        cid = fetch_cid(molecule_input)
        if cid:
            print(f"[CID] Resolved to CID {cid}")
        else:
            print("[CID] Could not resolve a CID — defaulting to 0.")

    # Step 1 — LLM1 extracts raw multi-candidate property values from literature.
    print(f"\n[LLM1] Extracting properties for: {molecule_input}")
    try:
        extracted = LLM1.run_extraction(molecule_input, REQUIRED_SCHEMA, client)
    except Exception as e:
        raise RuntimeError(f"LLM1 extraction failed: {e}")
    print("[LLM1] Extraction complete.")

    if debug:
        print("\n[DEBUG] LLM1 raw extraction:")
        print(json.dumps(extracted, indent=2))

    # LLM1 sometimes nests its output under a CID key rather than returning a
    # flat schema-shaped dict. For example: {"5754": {"molecule": ...}} instead
    # of {"molecule": ...}. Detect this by checking whether any top-level key
    # matches the schema, and unwrap if exactly one wrapper key is present.
    schema_keys = set(REQUIRED_SCHEMA.keys()) - {"cid"}
    if not schema_keys.intersection(extracted.keys()):
        wrapper_keys = [k for k in extracted if isinstance(extracted[k], dict)]
        if len(wrapper_keys) == 1:
            print(f"[NORM] Output was wrapped under key '{wrapper_keys[0]}' — unwrapping.")
            extracted = extracted[wrapper_keys[0]]

    # Step 2 — Fusion collapses the noisy candidate arrays into single values.
    print("\n[FUSION] Removing outliers and computing confidence-weighted values...")
    fused = fusion.fuse(extracted)

    if debug:
        print("\n[DEBUG] Fused intermediate:")
        print(json.dumps(fused, indent=2))

    print("[FUSION] Fusion complete.")

    # Step 3 — LLM2 checks biological plausibility and builds the final record.
    print("[LLM2] Verifying plausibility and assembling final record...")
    try:
        raw_output = LLM2.run_verification(fused, REQUIRED_SCHEMA, client)
    except Exception as e:
        raise RuntimeError(f"LLM2 verification failed: {e}")

    # LLM2 now returns a parsed dict directly after internal JSON cleaning
    if not isinstance(raw_output, dict):
        raise RuntimeError(f"LLM2 returned unexpected type {type(raw_output)}: {raw_output}")
    final_output = raw_output

    # Discover every numeric field path from the schema at runtime, so this
    # guard automatically covers any new fields added to schema.py in future.
    numeric_paths = _get_numeric_paths(REQUIRED_SCHEMA)

    # Re-inject fused numeric values into the final output. LLM2 is instructed
    # not to change these, but instruction-following is not guaranteed — this
    # acts as a hard enforcement layer rather than a trust-but-verify check.
    for *path, leaf in numeric_paths:
        try:
            src = fused
            for key in path:
                src = src[key]
            fused_val = src[leaf]
        except (KeyError, TypeError):
            continue  # this field was not present in the fused data at all

        if fused_val is None:
            continue

        try:
            dst = final_output
            for key in path:
                dst = dst[key]  # intentionally don't create missing sections
            existing = dst.get(leaf)
            if not (isinstance(existing, float) and isinstance(fused_val, float)
                    and math.isclose(existing, fused_val, rel_tol=1e-9)):
                if existing != fused_val:
                    print(f"[GUARD] LLM2 changed {'.'.join(path + [leaf])} "
                          f"from {existing} to {fused_val} — restoring fused value.")
            dst[leaf] = fused_val
        except (KeyError, TypeError):
            print(f"[WARN] Could not restore {'.'.join(path + [leaf])}: "
                  f"the path is missing from LLM2's output.")

    # Where fusion returned None (no plausible value could be resolved), make
    # sure the final output also carries None rather than the schema default of
    # 0.0 that LLM2 may have filled in. A null in the output is an honest
    # signal; a silent zero would be misleading.
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
            pass  # path is missing entirely; nothing to nullify

    # Audit the final output against the schema to catch any keys that LLM2
    # may have silently dropped, checking both top-level and one level deep.
    def _validate_keys(output: dict, schema: dict, path: str = ""):
        for key in schema:
            if key == "cid":
                continue  # cid is injected separately below, not from LLM2
            if key not in output:
                print(f"[WARN] Expected field missing from output: {path}{key}")
            elif isinstance(schema[key], dict) and isinstance(output.get(key), dict):
                _validate_keys(output[key], schema[key], path=f"{path}{key}.")
    _validate_keys(final_output, REQUIRED_SCHEMA)
    
    final_output["cid"] = cid
    print("[LLM2] Verification complete.")
    return final_output


# Entry point
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