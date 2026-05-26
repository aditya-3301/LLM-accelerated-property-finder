import json
import os
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

# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(molecule_input: str) -> dict:

    # STEP 1: Extraction
    print(f"\n[LLM1] Extracting properties for: {molecule_input}")
    try:
        extracted = LLM1.run_extraction(molecule_input, REQUIRED_SCHEMA, client)
    except Exception as e:
        raise RuntimeError(f"LLM1 extraction failed: {e}")
    print("[LLM1] Done.")

    # STEP 2: Fusion
    print("[FUSION] Running outlier removal + confidence-weighted mean...")
    fused = fusion.fuse(extracted)
    print("[FUSION] Done.")

    # STEP 3: Verification + Final JSON build
    print("[LLM2] Verifying and building final record...")
    try:
        raw_output = LLM2.run_verification(fused, REQUIRED_SCHEMA, client)
    except Exception as e:
        raise RuntimeError(f"LLM2 verification failed: {e}")

    # Parse LLM2 output to dict
    if isinstance(raw_output, dict):
        final_output = raw_output
    else:
        try:
            final_output = json.loads(raw_output)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLM2 returned invalid JSON: {e}\nRaw output:\n{raw_output}")

    print("[LLM2] Done.")
    return final_output


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    molecule = input("Enter molecule name / SMILES / CID: ").strip()

    result = run_pipeline(molecule)

    print("\n── FINAL OUTPUT ─────────────────────────────────────────────────────────")
    print(json.dumps(result, indent=2))

    output_file = "output.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_file}")