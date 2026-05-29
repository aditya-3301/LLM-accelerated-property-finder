"""
LLM2.py — Plausibility verification and final record assembly.

Changes vs previous version
----------------------------
* clean_energy removed from all prompts and type-enforcement rules.
* molecular_composition: LLM2 validates and optionally recomputes the
  element-fraction dict (not a string) and re-normalises fractions to 1.0.
* Integer-field enforcement is explicit; LLM2 must not produce floats for
  number_of_heavy_atoms, net_formal_charge, num_h_acceptors_*, num_h_donors*,
  num_rotatable_bonds.
* CAS number preservation: LLM2 must not substitute an alternate CAS.
* Provenance metadata: LLM2 adds a top-level "provenance" key documenting
  which fields were computed vs. LLM-extracted vs. validated-by-SMILES.
* Composition cross-check is now element-level (fractions must sum to 1.0).
"""

import json
import re
import time
from huggingface_hub.errors import HfHubHTTPError


def _clean_json_output(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    matches = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if matches:
        candidate = matches[-1]
    else:
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)
        else:
            raise ValueError("No JSON found in LLM2 output.")
    return json.loads(candidate)


def run_verification(
    fused_data: dict,
    schema: dict,
    client,
    retries: int = 3,
    wait: int = 10,
) -> dict:
    """
    LLM2: plausibility verification and final record assembly.
    Receives the fused intermediate and outputs the final clean record.
    """
    system_prompt = (
        "You are a biomedical verification agent and JSON builder.\n\n"

        "YOUR ROLE: You are a VERIFIER, not an extractor. You receive PRE-FILLED FUSED DATA. "
        "Check plausibility, fill genuinely missing fields, and output the final record.\n\n"

        "STRICT RULES\n"

        "1. PRESERVE NUMERICS: every numeric field in FUSED DATA that is not the schema default "
        "(0.0 for floats, 0 for integers) MUST be copied to your output EXACTLY as-is. "
        "Do not round, zero out, or modify these values.\n\n"

        "2. FILL MISSING: if a numeric field IS the schema default (0.0 / 0), you MAY fill it "
        "from your knowledge if confident; otherwise leave it as the default.\n\n"

        "3. STRING FIELDS: fill missing string and list fields from your knowledge ONLY if certain. "
        "Do not guess. A wrong DrugBank ID or CAS number is worse than an empty string.\n\n"

        "4. CAS NUMBER: do NOT substitute an alternate CAS for the one in fused data. "
        "The fusion layer has already selected the canonical primary CAS. Preserve it.\n\n"

        "5. DUPLICATE ACCESSIONS: secondary_accession_numbers must NOT contain the same value "
        "as drugbank_id. Remove any duplicate before outputting.\n\n"

        "6. SMILES: only fill smiles if 100% certain of the exact canonical structure. "
        "If there is any doubt, output an empty string.\n\n"

        "7. OUTPUT SCHEMA: output ONLY the fields present in the schema. No extra keys "
        "except the permitted 'warnings' and 'provenance' lists described below.\n\n"

        "8. STRICT INTEGER FIELDS — output integers (no decimal point) for:\n"
        "   number_of_heavy_atoms, net_formal_charge, num_h_acceptors_lipinski,\n"
        "   num_h_donors_lipinski, num_rotatable_bonds, num_h_acceptors, num_h_donors.\n\n"

        "9. STRICT FLOAT FIELDS — output floats (must include decimal point) for:\n"
        "   molecular_weight, exact_mol_weight, alogp, molecular_polar_surface_area.\n\n"

        "10. PLAUSIBILITY: if a numeric value grossly violates physical chemistry AND you have "
        "specific knowledge it is wrong, you MAY correct it — but you MUST record the change in "
        "a top-level 'warnings' list. Do not silently change values.\n\n"

        "11. CROSS-CHECK — use your token budget for these verifications:\n"
        "    a) number_of_heavy_atoms: sum all non-H atoms in molecular_formula.\n"
        "       C9H8O4 → 9+4=13. If fused value is wrong, correct it and add a warning.\n"
        "    b) molecular_composition: verify it is a dict {element: fraction}.\n"
        "       Fractions must sum to 1.0 ± 0.005. If not, recompute:\n"
        "         fraction = (atom_count × atomic_mass) / molecular_weight\n"
        "       Atomic masses: C=12.011, H=1.008, N=14.007, O=15.999, S=32.06, P=30.974.\n"
        "       Output the corrected dict; add a warning if you changed it.\n"
        "    c) exact_mol_weight ≠ molecular_weight. They must differ "
        "(monoisotopic vs average). If identical, set exact_mol_weight to null and warn.\n\n"

        "12. PROVENANCE: add a top-level 'provenance' object documenting the origin of key fields. "
        "Use these keys where applicable:\n"
        '    {"field": "<name>", "source": "pubchem|chembl|drugbank|llm_extracted|computed|verified",'
        ' "notes": "<optional detail>"}\n'
        "    Include at minimum: molecular_weight, molecular_formula, cas_number, smiles.\n\n"

        "Return ONLY a raw JSON object. No markdown fences, no explanation."
    )

    user_content = (
        f"SCHEMA:\n{json.dumps(schema, indent=2)}\n\n"
        f"FUSED DATA:\n{json.dumps(fused_data, indent=2)}"
    )

    for attempt in range(1, retries + 1):
        try:
            result = client.chat_completion(
                model="meta-llama/Llama-3.1-70B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=2000,
                temperature=0.1,
            )
            return _clean_json_output(result.choices[0].message.content)

        except HfHubHTTPError as e:
            if attempt < retries:
                print(f"[LLM2] Attempt {attempt} failed (server error). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM2 failed after {retries} attempts: {e}")

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < retries:
                print(f"[LLM2] Attempt {attempt} returned malformed JSON ({e}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM2 returned malformed JSON after {retries} attempts: {e}")
