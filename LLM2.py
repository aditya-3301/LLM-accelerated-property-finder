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
        "- Check biological plausibility of all values.\n"
        "- Flag and correct any structural or functional contradictions.\n"
        "- Fill missing fields (SMILES, IUPAC, biological context) from your knowledge.\n"
        "- Do NOT change any numerically fused activity values.\n"
        "- Output ONLY the fields present in the schema. Do not add any extra keys.\n"
        "- ACTIVITY TYPE RULE: if activity_type contains '|' or multiple options, resolve it to a single value: IC50, EC50, Ki, or Kd based on the molecule's known pharmacology.\n"
        "- UNIT RULE: activity_unit must always be nM. If it is M, mM, or uM, convert the value and set unit to nM (1M=1e9nM, 1mM=1e6nM, 1uM=1000nM). If already nM, do not convert.\n"
        "- Strict types: counts as integers, scores as floats.\n"
        "- Return ONLY the raw JSON object. No markdown, no explanation.\n"
        "- PHYSICOCHEMICAL SANITY: logP for drug-like molecules is typically −2 to 6. "
        "molecular_weight is typically 100–900 Da. If a value falls far outside these ranges "
        "or contradicts the molecule's known chemistry (e.g. a hydrophilic neurotransmitter "
        "with logP > 3), flag it by setting the value to null.\n"
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
                max_tokens=800,
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