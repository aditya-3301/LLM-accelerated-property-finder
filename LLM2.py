import json
import re
import time
from huggingface_hub.errors import HfHubHTTPError


def _clean_json_output(text: str) -> dict:
    """Strip markdown fences and parse JSON. Mirrors LLM1's clean_json_output."""
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


def run_verification(fused_data: dict, schema: dict, client, retries: int = 3, wait: int = 10) -> dict:
    """
    LLM2: plausibility verification and final record assembly.
    Receives the fused intermediate; should preserve all non-default numeric
    values and only fill genuinely missing fields.
    """
    system_prompt = (
        "You are a biomedical verification agent and JSON builder.\n\n"
        "YOUR ROLE: You are a VERIFIER, not an extractor. You receive PRE-FILLED FUSED DATA. "
        "Your job is to check plausibility, fill genuinely missing fields, and output the final record.\n\n"
        "STRICT RULES:\n"
        "1. PRESERVE NUMERICS: every numeric field in FUSED DATA that is not the schema default "
        "(0.0 for floats, 0 for integers) MUST be copied to your output EXACTLY as-is. "
        "Do not round, zero out, or modify these values for any reason.\n"
        "2. FILL MISSING: if a numeric field IS the schema default (0.0 / 0), you MAY fill it "
        "from your knowledge if confident; otherwise leave it as the default.\n"
        "3. STRING FIELDS: fill missing string and list fields (common_name, cas_number, unii, "
        "drugbank_id, secondary_accession_numbers, synonyms) from your knowledge ONLY if certain. "
        "Do not guess. A wrong DrugBank ID is worse than an empty string.\n"
        "4. SMILES: only fill smiles if you are 100% certain of the exact canonical structure "
        "for this specific molecule. If there is any doubt, output an empty string. "
        "A wrong SMILES is actively harmful.\n"
        "5. OUTPUT SCHEMA: output ONLY the fields present in the schema. Do not add extra keys.\n"
        "6. ACTIVITY TYPE: if activity_type contains '|', multiple options, or is empty, resolve "
        "to a single value from: IC50, EC50, Ki, Kd — based on the molecule's known pharmacology. "
        "If unknown, use IC50 as default.\n"
        "7. UNIT RULE: activity_unit must always be 'nM'. If fused data has M/mM/µM values, "
        "convert them (1M=1e9nM, 1mM=1e6nM, 1µM=1000nM). If already nM, do not touch.\n"
        "8. STRICT TYPES:\n"
        "   - These fields MUST be integers: number_of_heavy_atoms, net_formal_charge, "
        "num_h_acceptors_lipinski, num_h_donors_lipinski, num_h_acceptors, num_h_donors, "
        "num_rotatable_bonds.\n"
        "   - All other numeric fields MUST be floats (include a decimal point).\n"
        "9. TOXICITY ENUMS: carcinogenicity, immunotoxicity, mutagenicity, cytotoxicity must be "
        "exactly one of: 'Positive', 'Negative', 'Inconclusive'. No other values allowed.\n"
        "10. PLAUSIBILITY: if a numeric value grossly violates physical chemistry AND you have "
        "specific knowledge it is wrong, you may correct it — but you MUST print a warning in "
        "a top-level 'warnings' list. Do not silently change values.\n"
        "11. INTERACTION BLOCK: if the molecule is not a pharmacological agent (e.g. it is a "
        "nutrient, amino acid, or cofactor with no known receptor target), set activity_value, "
        "reported_ic50, and ec50 to null and activity_type to an empty string.\n\n"
        "12. CROSS-CHECK COMPUTED FIELDS: you have spare token budget — use it.\n"
        "    a) Verify number_of_heavy_atoms by summing all non-H atoms in molecular_formula. "
        "       Example: C9H8O4 → 9+4=13. If the fused value is wrong, correct it and add a warning.\n"
        "    b) Verify molecular_composition fractions sum to 1.0 ± 0.005. "
        "       If they don't, recompute using fraction = (count × atomic_mass) / molecular_weight. "
        "       Atomic masses: C=12.011, H=1.008, N=14.007, O=15.999, S=32.06, P=30.974.\n"
        "    c) Verify exact_mol_weight ≠ molecular_weight. They must differ (monoisotopic vs average). "
        "       If they are identical, set exact_mol_weight to null and add a warning.\n\n"
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