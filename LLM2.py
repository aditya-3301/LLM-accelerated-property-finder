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


def run_verification(fused_data, schema, client, retries=3, wait=10):
    system_prompt = (
        "You are a biomedical verification agent and JSON builder.\n"
        "Rules:\n"
        "- You are a VERIFIER, not an extractor. You receive pre-filled FUSED DATA. "
        "Your job is to check plausibility and fill genuinely missing fields — NOT to re-extract.\n"
        "- PRESERVE ALL NUMERIC VALUES: every numeric field in FUSED DATA that is not "
        "the schema default (0.0 for floats, 0 for integers) must be copied to your "
        "output EXACTLY as-is. Do not round, zero out, or change these values for any reason.\n"
        "- If a numeric field in FUSED DATA IS the schema default (0.0 / 0), you may fill "
        "it from your knowledge if you are confident; otherwise leave it as the default.\n"
        "- Fill missing string / list fields (smiles, common_name, cas_number, synonyms, "
        "drugbank_id, biological context) from your knowledge where confident.\n"
        "- Output ONLY the fields present in the schema. Do not add any extra keys.\n"
        "- ACTIVITY TYPE RULE: if activity_type contains '|' or multiple options, resolve "
        "it to a single value: IC50, EC50, Ki, or Kd based on the molecule's known pharmacology.\n"
        "- UNIT RULE: activity_unit must always be nM. If it is M, mM, or uM, convert the "
        "value and set unit to nM (1M=1e9nM, 1mM=1e6nM, 1uM=1000nM). If already nM, do not convert.\n"
        "- Strict types: counts (num_h_acceptors, num_h_donors, num_rotatable_bonds, "
        "net_formal_charge) must be integers. All other numeric fields must be floats.\n"
        "- ENUM CONSTRAINTS — use ONLY these exact strings:\n"
        "    gi_absorption   : 'High' or 'Low'\n"
        "    bbb_penetration : 'BBB+' or 'BBB-'\n"
        "    p_gp_substrate  : 'Yes' or 'No'\n"
        "    carcinogenicity : 'Positive', 'Negative', or 'Inconclusive'\n"
        "    immunotoxicity  : 'Positive', 'Negative', or 'Inconclusive'\n"
        "    mutagenicity    : 'Positive', 'Negative', or 'Inconclusive'\n"
        "    cytotoxicity    : 'Positive', 'Negative', or 'Inconclusive'\n"
        "- PHYSICOCHEMICAL SANITY: logP for drug-like molecules is typically −2 to 6. "
        "molecular_weight is typically 100–900 Da. Only set a value to null if it grossly "
        "violates physical chemistry AND you have specific knowledge it is wrong.\n"
        "- Return ONLY the raw JSON object. No markdown, no explanation.\n"
    )

    user_content = (
        f"SCHEMA:\n{json.dumps(schema)}\n\n"
        f"FUSED DATA:\n{json.dumps(fused_data)}"
    )

    for attempt in range(1, retries + 1):
        try:
            result = client.chat_completion(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            return _clean_json_output(result.choices[0].message.content)

        except HfHubHTTPError as e:
            if attempt < retries:
                print(f"[LLM2] Attempt {attempt} failed (timeout/server error). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM2 failed after {retries} attempts: {e}")
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < retries:
                print(f"[LLM2] Attempt {attempt} returned malformed JSON. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM2 returned malformed JSON after {retries} attempts: {e}")