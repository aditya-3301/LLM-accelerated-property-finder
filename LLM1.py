import json
import re

def clean_json_output(text):
    """Extracts JSON from the last ```json block if present, else parses directly."""
    matches = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if matches:
        return json.loads(matches[-1])
    return json.loads(text)

def run_extraction(molecule_input, schema, client):
    system_prompt = (
        "You are a biomedical extraction agent. Extract molecular properties ONLY for fields in the provided schema.\n"
        "Rules:\n"
        "- Multiple candidate values per field where evidence conflicts.\n"
        "- No filtering, averaging, or correctness resolution.\n"
        "- Each leaf node format: [{\"value\": ..., \"confidence\": 0.0-1.0, \"source_type\": \"...\"}]\n"
        "- activity_type must be one of: IC50, EC50, Ki, Kd\n"
        f"- Schema: {json.dumps(schema)}\n"
        "Output ONLY valid JSON in a ```json block."
    )

    user_content = f"Extract all raw scientific claims and candidate property values for: {molecule_input}"

    result = client.chat_completion(
        model="deepseek-ai/DeepSeek-R1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        max_tokens=1500,
        temperature=0.6
    )

    raw_text = result.choices[0].message.content
    return clean_json_output(raw_text)