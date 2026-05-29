"""
LLM1.py — Multi-candidate raw extraction.

Changes vs previous version
----------------------------
* clean_energy removed from all prompts (no agreed units/method).
* molecular_composition is now extracted as a dict {element: mass_fraction},
  not a string.  Each element is a separate key so fusion can operate per-element.
* Integer fields are explicitly listed; LLM must not produce floats for them.
* CAS number guidance: primary CAS is the lowest-numbered (earliest-registered)
  canonical accession — not a salt or polymorph form.
* Source-type vocabulary aligned with SOURCE_PRIORITY in drug_descriptors.py
  so the fusion layer can apply priority scores correctly.
* Confidence guidance tightened: deterministic descriptors (MW, HBA …) should
  carry confidence ≥ 0.8 when sourced from PubChem/DrugBank because the values
  are structural, not measured.
"""

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
        "You are a biomedical extraction agent. Extract candidate property values "
        "for the given molecule from PubChem, ChEMBL, BindingDB, DrugBank, and "
        "peer-reviewed literature.\n\n"

        "OUTPUT FORMAT\n"
        "Return a single JSON object mirroring the schema's nested structure exactly. "
        "Every leaf field must be a LIST of candidate dicts:\n"
        '  [{"value": <value>, "confidence": <0.0–1.0>, '
        '"source_type": "pubchem|chembl|drugbank|bindingdb|literature|other"}]\n'
        "Multiple candidates per field are strongly encouraged when sources may differ.\n\n"

        "SOURCE TYPE VOCABULARY — use exactly one of these strings:\n"
        "  pubchem, chembl, drugbank, bindingdb, literature, other\n\n"

        "RULES\n"
        "1. STRUCTURE: mirror the schema nesting exactly. Do not flatten nested fields.\n\n"

        "2. CAS NUMBER (critical):\n"
        "   The primary CAS number is the CANONICAL CAS assigned to the free-acid/free-base "
        "neutral form of the molecule — NOT a salt, hydrate, or polymorph. "
        "It is the LOWEST-NUMBERED (earliest-registered) CAS accession for the parent structure.\n"
        "   Example — ibuprofen: canonical CAS is 15687-27-1, NOT 58560-75-1 (a racemate salt).\n"
        "   If multiple CAS numbers are known, place the canonical one first with highest confidence; "
        "list others as lower-confidence candidates so the fusion layer can select correctly.\n\n"

        "3. IDENTITY FIELDS: for drugbank_id, secondary_accession_numbers, cas_number, unii, "
        "common_name, synonyms — use known registry information.\n"
        "   IMPORTANT: do NOT repeat the primary drugbank_id value in secondary_accession_numbers.\n\n"

        "4. SMILES: only provide if you are certain of the exact canonical structure. "
        "A wrong SMILES is worse than an empty string.\n\n"

        "5. INTEGER FIELDS — the following MUST have integer values (no decimals):\n"
        "   number_of_heavy_atoms, net_formal_charge, num_h_acceptors_lipinski,\n"
        "   num_h_donors_lipinski, num_rotatable_bonds, num_h_acceptors, num_h_donors.\n"
        "   Example: num_rotatable_bonds=3, NOT 3.2857.\n\n"

        "6. FLOAT FIELDS — the following MUST have float values (include a decimal):\n"
        "   molecular_weight, exact_mol_weight, alogp, molecular_polar_surface_area.\n\n"

        "7. MOLECULAR COMPOSITION — output as a dict of {element: mass_fraction}.\n"
        "   Compute: fraction = (atom_count × atomic_mass) / molecular_weight.\n"
        "   Atomic masses: C=12.011, H=1.008, N=14.007, O=15.999, S=32.06, P=30.974.\n"
        "   Fractions must sum to 1.0 ± 0.005.\n"
        "   Example for aspirin C9H8O4 MW=180.16:\n"
        '     {"value": {"C": 0.600, "H": 0.045, "O": 0.355}, "confidence": 0.9, "source_type": "pubchem"}\n\n'

        "8. HEAVY ATOMS: number_of_heavy_atoms = count of ALL non-hydrogen atoms.\n"
        "   Sum every element except H. Example — aspirin C9H8O4: 9+4=13.\n\n"

        "9. CONFIDENCE GUIDANCE:\n"
        "   - Structural/deterministic descriptors from PubChem or DrugBank → 0.85–0.95\n"
        "   - Same fields from ChEMBL → 0.75–0.85\n"
        "   - Same fields from literature → 0.60–0.75\n"
        "   - Uncertain or computed/estimated values → 0.40–0.60\n"
        "   - Do not fabricate values — assign confidence < 0.4 if uncertain.\n\n"

        "10. EXACT vs AVERAGE MW:\n"
        "    molecular_weight   = average molecular weight (uses standard atomic weights).\n"
        "    exact_mol_weight   = monoisotopic mass (uses most-abundant isotope masses).\n"
        "    These MUST differ for any molecule with >1 heavy atom.\n\n"

        "11. JSON: ensure all strings are properly quoted and all objects are closed.\n"
        "    No extra keys beyond the schema.\n\n"

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
