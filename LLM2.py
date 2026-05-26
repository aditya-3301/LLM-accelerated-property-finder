import json

def run_verification(fused_data, schema, client):
    system_prompt = (
        "You are the Verification Layer and Final JSON Builder for a biomedical reconstruction pipeline.\n"
        "Your input is mathematically fused molecular properties. Transform this into a verified pharmacological record.\n"
        "Rules:\n"
        "- Validate biological plausibility and flag structural/functional contradictions.\n"
        "- Fill remaining fields (SMILES, IUPAC names, biological context) from your biochemical knowledge.\n"
        "- Do NOT alter any mathematically fused activity values.\n"
        "- Strict data types: counts as integers, scores as floats.\n"
        "- Return ONLY the raw JSON object matching the schema. No markdown, no explanation."
    )

    user_content = (
        f"SCHEMA:\n{json.dumps(schema)}\n\n"
        f"FUSED DATA:\n{json.dumps(fused_data)}"
    )

    result = client.chat_completion(
        model="google/gemma-2-2b-it",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        max_tokens=800,
        temperature=0.1
    )

    return result.choices[0].message.content