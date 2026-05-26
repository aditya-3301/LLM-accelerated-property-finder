import json
import re
import time
from huggingface_hub.errors import HfHubHTTPError

def clean_json_output(text):
    """Extracts JSON from the last ```json block if present, else parses directly."""
    matches = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if matches:
        candidate = matches[-1]
    else:
        # No code block — try to extract the outermost {...} directly
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)
        else:
            raise ValueError("No JSON found in LLM1 output.")

    return json.loads(candidate)

def run_extraction(molecule_input, schema, client, retries=3, wait=10):
    system_prompt = (
        "You are a biomedical extraction agent. Extract molecular properties ONLY for fields in the provided schema.\n"
        "Rules:\n"
        "- Prefer values from PubChem, ChEMBL, BindingDB, or peer-reviewed literature. Deprioritize general web sources.\n"
        "- Multiple candidate values per field where evidence conflicts.\n"
        "- No filtering, averaging, or correctness resolution.\n"
        "- STRUCTURE RULE: your output JSON must mirror the exact nested structure of the schema. Do not flatten nested fields to the top level.\n"
        "- Each leaf node format: [{\"value\": ..., \"confidence\": 0.0-1.0, \"source_type\": \"PubChem|ChEMBL|BindingDB|literature|other\"}]\n"
        "- activity_type must be one of: IC50, EC50, Ki, Kd\n"
        "- UNIT RULE: activity_value MUST be in nM. Always convert: 1M=1e9nM, 1mM=1e6nM, 1uM=1000nM. "
        "A correct nM value for a drug-like molecule is typically between 1 and 1,000,000. "
        "Values below 1 are almost always a conversion error — e.g. 0.035 is likely 0.035 uM = 35 nM. "
        "Values like 1.7e-05 are M-scale — convert them (1.7e-05 M = 17000 nM). "
        "Double-check your conversion before outputting.\n"
        "- If you are not confident about a value, assign confidence below 0.4. Do not fabricate values for unknown fields.\n"
        "- Ensure all strings in the JSON are properly quoted and all objects are properly closed.\n"
        f"- Schema: {json.dumps(schema)}\n"
        "Output ONLY valid JSON in a ```json block."
    )

    user_content = f"Extract all raw scientific claims and candidate property values for: {molecule_input}"
    temperature=0.6
    for attempt in range(1, retries + 1):
        try:
            result = client.chat_completion(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=2000,
                temperature=temperature
            )
            raw_text = result.choices[0].message.content
            return clean_json_output(raw_text)

        except HfHubHTTPError as e:
            if attempt < retries:
                print(f"[LLM1] Attempt {attempt} failed (timeout/server error). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM1 failed after {retries} attempts: {e}")

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < retries:
                print(f"[LLM1] Attempt {attempt} returned malformed JSON. Retrying in {wait}s...")
                time.sleep(wait)
                temperature = min(0.6 + attempt * 0.1, 1.0)  # nudge temperature each retry
            else:
                raise RuntimeError(f"LLM1 returned malformed JSON after {retries} attempts: {e}")