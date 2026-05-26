import json
import time
from huggingface_hub.errors import HfHubHTTPError

def run_verification(fused_data, schema, client, retries=3, wait=10):
    system_prompt = (
        "You are a biomedical verification agent and JSON builder.\n"
        "Rules:\n"
        "- Check biological plausibility of all values.\n"
        "- Flag and correct any structural or functional contradictions.\n"
        "- Fill missing fields (SMILES, IUPAC, biological context) from your knowledge.\n"
        "- Do NOT change any numerically fused activity values.\n"
        "- Strict types: counts as integers, scores as floats.\n"
        "- Return ONLY the raw JSON object. No markdown, no explanation."
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
            return result.choices[0].message.content

        except HfHubHTTPError as e:
            if attempt < retries:
                print(f"[LLM2] Attempt {attempt} failed (timeout/server error). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM2 failed after {retries} attempts: {e}")