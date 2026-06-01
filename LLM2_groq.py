"""
LLM2.py — Plausibility verification and final record assembly.

Changes vs previous version
----------------------------
* clean_energy removed from all prompts and type-enforcement rules.
* molecular_composition: LLM2 validates and optionally recomputes the
  element-fraction dict (not a string) and re-normalises fractions to 1.0.
* Integer-field enforcement is explicit; LLM2 must not produce floats for
  number_of_atoms, net_formal_charge, num_h_acceptors_*, num_h_donors*,
  num_rotatable_bonds.
* CAS number preservation: LLM2 must not substitute an alternate CAS.
* Composition cross-check is now element-level (fractions must sum to 1.0).
"""

import json
import re
import time
from groq import InternalServerError as GroqServerError


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
        "You are a biomedical verification agent. You receive PRE-FILLED FUSED DATA. "
        "Verify plausibility, fill genuinely missing fields, output the final record.\n\n"

        "RULES\n"
        "1. PRESERVE: copy every non-default numeric exactly (default=0.0 for floats, 0 for ints). No rounding.\n"
        "2. FILL MISSING: if a numeric equals its schema default, fill from knowledge only if confident; else keep default.\n"
        "3. STRINGS/LISTS: fill only if certain. Wrong DrugBank ID or CAS is worse than empty string.\n"
        "4. CAS: preserve fused CAS exactly. Fusion already selected canonical. Do not substitute.\n"
        "5. secondary_accession_numbers must NOT contain the drugbank_id value. Remove duplicates.\n"
        "6. SMILES: fill only if 100% certain of exact canonical structure; else empty string.\n"
        "7. Output ONLY schema fields plus the permitted 'warnings' key.\n"
        "8. INTEGER fields (no decimal): number_of_atoms, net_formal_charge, "
        "num_h_acceptors_lipinski, num_h_donors_lipinski, num_rotatable_bonds, num_h_acceptors, num_h_donors.\n"
        "9. FLOAT fields (must have decimal): molecular_weight, exact_mol_weight, alogp, molecular_polar_surface_area.\n"
        "10. PLAUSIBILITY: correct a value only if it grossly violates physical chemistry AND you have specific knowledge. "
        "Log every change in top-level 'warnings' list.\n"
        "11. CROSS-CHECK:\n"
        "    a) number_of_atoms = count of non-H elements only (exclude H entirely). "
        "C13H18O2: non-H are C(13)+O(2)=15. H18 is ignored. Correct+warn if fused value differs.\n"
        "    b) molecular_composition: FIRST sum the fused fractions. "
        "If sum is already 1.0±0.005, copy it UNCHANGED. "
        "Only recompute if sum is outside that range: fraction=(atom_count×atomic_mass)/MW, "
        "Masses: C=12.011,H=1.008,N=14.007,O=15.999,S=32.06,P=30.974. Warn only if you actually changed it.\n"
        "    c) exact_mol_weight≠molecular_weight (monoisotopic vs average). If equal, set exact_mol_weight=null and warn.\n"


        "Return ONLY a raw JSON object. No markdown fences, no explanation."
    )

    user_content = (
        f"SCHEMA:\n{json.dumps(schema, indent=2)}\n\n"
        f"FUSED DATA:\n{json.dumps(fused_data, indent=2)}"
    )

    for attempt in range(1, retries + 1):
        try:
            result = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=2000,
                temperature=0.1,
            )
            return _clean_json_output(result.choices[0].message.content)

        except GroqServerError as e:
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