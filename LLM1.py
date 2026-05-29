import json
import re
import time
from huggingface_hub.errors import HfHubHTTPError


def clean_json_output(text: str) -> dict:
    """
    Extracts and parses JSON from LLM output.
    Tries the last ```json block first, then falls back to outermost { }.
    """
    matches = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if matches:
        candidate = matches[-1]
    else:
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)
        else:
            raise ValueError("No JSON found in LLM1 output.")
    return json.loads(candidate)


def run_extraction(molecule_input: str, cid: int, schema: dict, client, retries: int = 3, wait: int = 10) -> dict:
    """
    LLM1: raw multi-candidate extraction.
    molecule_input — common name (or IUPAC name if short enough).
    cid            — PubChem CID, included in the prompt for unambiguous lookup.
    """
    # Build a compact schema hint — just keys + types so the model doesn't get
    # confused by default values and tries to echo them back.
    def _schema_hint(node, indent=0):
        lines = []
        pad = "  " * indent
        for k, v in node.items():
            if isinstance(v, dict):
                lines.append(f"{pad}{k}:")
                lines.extend(_schema_hint(v, indent + 1))
            elif isinstance(v, list):
                lines.append(f"{pad}{k}: [list of strings]")
            elif isinstance(v, str):
                lines.append(f"{pad}{k}: string")
            elif isinstance(v, float):
                lines.append(f"{pad}{k}: float")
            elif isinstance(v, int):
                lines.append(f"{pad}{k}: integer")
        return lines

    schema_hint = "\n".join(_schema_hint(schema))

    system_prompt = (
        "You are a biomedical extraction agent. Your job is to extract candidate property values "
        "for the given molecule from your knowledge of PubChem, ChEMBL, BindingDB, DrugBank, and "
        "peer-reviewed literature.\n\n"
        "OUTPUT FORMAT:\n"
        "Return a single JSON object that mirrors the exact nested structure of the schema below. "
        "Every leaf field must be a LIST of candidate dicts:\n"
        '  [{"value": <value>, "confidence": <0.0–1.0>, "source_type": "PubChem|ChEMBL|DrugBank|literature|other"}]\n'
        "Multiple candidates per field are encouraged when sources conflict.\n\n"
        "RULES:\n"
        "1. STRUCTURE: mirror the schema's nested structure exactly. Do not flatten nested fields.\n"
        "2. IDENTITY: for drugbank_id, secondary_accession_numbers, cas_number, unii, common_name, "
        "synonyms — use known registry information. Provide one candidate dict per synonym in the synonyms list.\n"
        "3. SMILES: only provide smiles if you are certain of the exact canonical structure. "
        "A wrong SMILES is worse than an empty string — leave it empty if any doubt.\n"
        "4. ACTIVITY UNITS: activity_value, reported_ic50, ec50 MUST be in nM. Always convert: "
        "1 M = 1e9 nM, 1 mM = 1e6 nM, 1 µM = 1e3 nM. "
        "A value like 0.035 is almost certainly 0.035 µM = 35 nM — double-check before outputting. "
        "Values below 1 in nM are almost always a conversion error.\n"
        "5. ACTIVITY TYPE: activity_type must be exactly one of: IC50, EC50, Ki, Kd.\n"
        "6. TOXICITY ENUMS: carcinogenicity, immunotoxicity, mutagenicity, cytotoxicity must be "
        "exactly: 'Positive', 'Negative', or 'Inconclusive'.\n"
        "7. TYPES: number_of_heavy_atoms, net_formal_charge, num_h_acceptors_lipinski, "
        "num_h_donors_lipinski, num_h_acceptors, num_h_donors, num_rotatable_bonds "
        "must have integer values. All other numeric fields must be floats.\n"
        "8. CONFIDENCE: if you are not confident about a value, assign confidence < 0.4. "
        "Do not fabricate values — it is better to omit a field than to guess.\n"
        "9. MOLECULAR COMPOSITION: format as element mass-fraction string. "
        "Compute each fraction as: (atom_count × atomic_mass) / molecular_weight. "
        "Use atomic masses: C=12.011, H=1.008, N=14.007, O=15.999, S=32.06, P=30.974. "
        "Example — aspirin C9H8O4 MW=180.16: "
        "C=9×12.011/180.16=0.600, H=8×1.008/180.16=0.045, O=4×15.999/180.16=0.355 "
        '→ "C: 0.600, H: 0.045, O: 0.355". Always verify fractions sum to ~1.0.\n'
        "10. HEAVY ATOMS: number_of_heavy_atoms is the count of ALL non-hydrogen atoms. "
        "Sum every element except H. Example — aspirin C9H8O4: 9 (C) + 4 (O) = 13, NOT 9. "
        "Do not count only carbon atoms.\n"
        "11. JSON: ensure all strings are properly quoted and all objects are properly closed.\n\n"
        f"SCHEMA:\n{schema_hint}\n\n"
        "Output ONLY valid JSON inside a ```json block. No explanation."
    )

    # Include the CID so the model can anchor to the exact PubChem entry
    cid_hint = f" (PubChem CID: {cid})" if cid else ""
    user_content = (
        f"Extract all candidate property values for: {molecule_input}{cid_hint}\n"
        f"Provide AT LEAST 2 candidates per numeric field, drawing from different sources "
        f"(PubChem, ChEMBL, DrugBank, literature). For well-known molecules this should be easy. "
        f"Single candidates are only acceptable when a field is genuinely single-source."
    )

    temperature = 0.5
    for attempt in range(1, retries + 1):
        try:
            result = client.chat_completion(
                model="meta-llama/Llama-3.1-70B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=2000,
                temperature=temperature,
            )
            raw_text = result.choices[0].message.content
            return clean_json_output(raw_text)

        except HfHubHTTPError as e:
            if attempt < retries:
                print(f"[LLM1] Attempt {attempt} failed (server error). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM1 failed after {retries} attempts: {e}")

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < retries:
                print(f"[LLM1] Attempt {attempt} returned malformed JSON ({e}). Retrying in {wait}s...")
                time.sleep(wait)
                temperature = min(temperature + 0.1, 1.0)  # nudge temperature each retry
            else:
                raise RuntimeError(f"LLM1 returned malformed JSON after {retries} attempts: {e}")