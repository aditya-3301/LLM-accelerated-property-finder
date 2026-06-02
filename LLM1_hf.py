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


def _schema_hint(node, indent=0) -> list[str]:
    """
    Produce a compact, human-readable schema hint from the nested schema dict.
    molecular_composition is rendered as a nested dict hint so the LLM
    understands it must output {element: fraction} pairs.
    """
    lines = []
    pad = "  " * indent
    for k, v in node.items():
        if isinstance(v, dict):
            if k == "molecular_composition":
                # Special rendering for composition
                lines.append(f'{pad}{k}: dict  # {{element: mass_fraction}} e.g. {{"C": 0.600, "H": 0.045, "O": 0.355}}')
            else:
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


def run_extraction(
    molecule_input: str,
    cid: int,
    schema: dict,
    client,
    retries: int = 3,
    wait: int = 10,
) -> dict:
    """
    LLM1: raw multi-candidate extraction.
    molecule_input — human-readable name (common + IUPAC hint when available).
    cid            — PubChem CID, included so the model anchors to the exact entry.
    """
    schema_str = "\n".join(_schema_hint(schema))

    system_prompt = (
        "You are a biomedical extraction agent. Extract molecule property candidates "
        "from PubChem, ChEMBL, BindingDB, DrugBank, and peer-reviewed literature.\n\n"

        "OUTPUT: Single JSON object matching schema structure exactly. "
        "Every leaf = LIST of candidate dicts: "
        '[{"value":<v>,"confidence":<0.0-1.0>,"source_type":"pubchem|chembl|drugbank|bindingdb|literature|other"}]. '
        "Prefer multiple candidates per field when sources may differ.\n\n"

        "RULES\n"
        "1. Mirror schema nesting exactly. No flattening.\n"
        "2. CAS: use canonical CAS of free-acid/free-base neutral form (lowest-numbered accession, NOT a salt/hydrate). "
        "Ibuprofen: 15687-27-1 not 58560-75-1. List canonical first (highest confidence); others as lower-confidence candidates.\n"
        "3. Identity fields (drugbank_id, secondary_accession_numbers, cas_number, unii, common_name, synonyms): "
        "use registry data. Do NOT repeat drugbank_id in secondary_accession_numbers.\n"
        "4. SMILES: before submitting, verify your SMILES heavy-atom count matches molecular_formula. "
        "Ibuprofen C13H18O2 must have exactly 15 heavy atoms in SMILES (13C+2O). "
        "If count mismatches the formula, output empty string instead. Wrong SMILES > no SMILES.\n"
        "5. INTEGER fields (no decimals): number_of_atoms, net_formal_charge, "
        "num_h_acceptors_lipinski, num_h_donors_lipinski, num_rotatable_bonds, num_h_acceptors, num_h_donors.\n"
        "6. FLOAT fields (must have decimal): molecular_weight, exact_mol_weight, alogp, molecular_polar_surface_area.\n"
        "7. molecular_composition: {element: mass_fraction} dict computed from molecular_formula (NOT from SMILES). "
        "fraction=(atom_count×atomic_mass)/MW. Masses: C=12.011,H=1.008,N=14.007,O=15.999,S=32.06,P=30.974. "
        'Sum=1.0±0.005. Ex aspirin C9H8O4 MW=180.16: {"value":{"C":0.600,"H":0.045,"O":0.355},"confidence":0.9,"source_type":"pubchem"}\n'
        "8. number_of_atoms = count of non-H atoms ONLY from molecular_formula. Sum every element except H. "
        "C13H18O2: C(13)+O(2)=15. H18 excluded. C9H8O4: C(9)+O(4)=13.\n"
        "9. Confidence: PubChem/DrugBank structural→0.85-0.95; ChEMBL→0.75-0.85; literature→0.60-0.75; "
        "estimated→0.40-0.60; uncertain→<0.4 (do not fabricate).\n"
        "10. molecular_weight=average MW; exact_mol_weight=monoisotopic mass. Must differ for any multi-heavy-atom molecule.\n"
        "11. Valid JSON only. No extra keys.\n\n"

        f"SCHEMA:\n{schema_str}\n\n"
        "Output ONLY valid JSON inside a ```json block. No explanation."
    )

    cid_hint = f" (PubChem CID: {cid})" if cid else ""
    user_content = (
        f"Extract all candidate property values for: {molecule_input}{cid_hint}\n"
        f"Provide AT LEAST 2 candidates per numeric field from different sources "
        f"(pubchem, chembl, drugbank, literature). "
        f"Single candidates are acceptable only when a field is genuinely single-source."
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
                print(f"[LLM1] Attempt {attempt} returned malformed JSON ({e}). Retrying in {wait}s…")
                time.sleep(wait)
                temperature = min(temperature + 0.1, 1.0)
            else:
                raise RuntimeError(f"LLM1 returned malformed JSON after {retries} attempts: {e}")