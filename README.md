#  Biomedical Property Reconstruction Pipeline

An automated, three-tier data-refinement architecture designed to extract, resolve, and verify sparse physicochemical and pharmacological properties of small molecules from unstructured scientific literature. By employing a **hybrid sandwich pattern — LLM → Code → LLM** — the system combines the broad extraction capabilities of Large Language Models with the strict reliability of a deterministic mathematical engine in Python. This ensures statistical anomalies are rejected, units are normalized, and biological consistency is enforced before any records are committed to disk.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture & Pipeline Flow](#architecture--pipeline-flow)
- [Repository Structure](#repository-structure)
- [Data Schema](#data-schema)
- [Mathematical Engine — Fusion Algorithm](#mathematical-engine--fusion-algorithm)
  - [Stage 1 — Confidence Threshold Filtering](#stage-1--confidence-threshold-filtering)
  - [Stage 2 — Per-Field Confidence Capping](#stage-2--per-field-confidence-capping)
  - [Stage 3 — Multi-Unit Normalization](#stage-3--multi-unit-normalization)
  - [Stage 4 — Plausibility Floor Enforcement](#stage-4--plausibility-floor-enforcement)
  - [Stage 5 — Consensus Hallucination Detection](#stage-5--consensus-hallucination-detection)
  - [Stage 6 — Dynamic Outlier Rejection](#stage-6--dynamic-outlier-rejection)
  - [Stage 7 — Confidence-Weighted Fusion](#stage-7--confidence-weighted-fusion)
- [Module Reference](#module-reference)
  - [main.py](#mainpy)
  - [LLM1.py](#llm1py)
  - [fusion.py](#fusionpy)
  - [fusion_config.py](#fusion_configpy)
  - [LLM2.py](#llm2py)
  - [schema.py](#schemapy)
- [Guards & Safety Mechanisms](#guards--safety-mechanisms)
- [Fusion Configuration Reference](#fusion-configuration-reference)
- [Output Format](#output-format)
- [Quickstart](#quickstart)
- [Environment Variables](#environment-variables)
- [Dependencies](#dependencies)
- [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)
- [Extending the Schema](#extending-the-schema)
- [Known Limitations](#known-limitations)

---

## Overview

Small-molecule property data is scattered across public databases (PubChem, ChEMBL, BindingDB) and peer-reviewed literature in inconsistent units, with conflicting reported values and varying confidence levels depending on the assay method and reporting source. A naive LLM query over this space will either hallucinate a single confident-sounding value or silently pick one of several contradicting figures.

This pipeline solves that problem by:

1. **Forcing disagreement** — LLM1 is contractually prohibited from resolving conflicts. It must emit every candidate it finds with an individual confidence score.
2. **Deterministically cleaning the noise** — a pure Python fusion engine, with no LLM involvement, applies mathematical filters and a confidence-weighted average.
3. **Verifying biological plausibility** — LLM2 inspects the fused result for domain-level contradictions (e.g. impossible logP for a known hydrophilic neurotransmitter) without being permitted to alter any numeric value already produced by the fusion engine.

The result is a production-grade JSON record where every numeric value is traceable to a deterministic calculation, not an LLM guess.

---

## Architecture & Pipeline Flow

```
[ User Input: Molecule Name / SMILES / CID ]
                       │
                       ▼
       ┌───────────────────────────────┐
       │  PubChem API Resolution Layer │
       │  REST endpoint: /pug/compound │
       │  Tries name → falls back SMILES│
       └───────────────────────────────┘
                       │
          (Resolves canonical CID integer)
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  LAYER 1 — Extraction  (LLM1.py)                         │
│                                                          │
│  Model  : meta-llama/Llama-3.1-8B-Instruct               │
│  Role   : Biomedical extraction agent                    │
│  Output : Nested dict of candidate arrays                │
│           Each leaf → list of                            │
│           {"value": ..., "confidence": 0–1,              │
│            "source_type": PubChem|ChEMBL|...}            │
│  Contract: NO averaging, NO conflict resolution          │
│  Retry  : Up to 3 attempts; temperature nudged           │
│           (+0.1 per retry) on malformed JSON             │
└──────────────────────────────────────────────────────────┘
                       │
      (Noisy multi-candidate nested extraction)
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  LAYER 2 — Deterministic Fusion  (fusion.py)             │
│                                                          │
│  Step A : Drop candidates with confidence < 0.4          │
│  Step B : Normalize units → nM  (M, mM, uM, µM, nM)     │
│  Step C : Apply per-field confidence caps                │
│  Step D : Detect consensus hallucination                 │
│  Step E : Plausibility floor rejection                   │
│  Step F : Median-based outlier removal                   │
│           (absolute threshold OR 5× ratio)               │
│  Step G : Confidence-weighted mean                       │
│  String fields → highest-confidence candidate wins       │
└──────────────────────────────────────────────────────────┘
                       │
     (Single fused value per field — no LLM involved)
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  LAYER 3 — Verification  (LLM2.py)                       │
│                                                          │
│  Model  : meta-llama/Llama-3.1-8B-Instruct               │
│  Role   : Biomedical verification agent & JSON builder   │
│  Checks : Biological plausibility, unit consistency,     │
│           logP sanity (−2 to 6), MW sanity (100–900 Da)  │
│           activity_type disambiguation                   │
│  Output : Final schema-conformant JSON object            │
│  Hard rule: Must NOT alter any fused numeric value       │
│  Retry  : Up to 3 attempts on timeout or malformed JSON  │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │  main.py guard layer    │
         │  • Re-injects fused     │
         │    numeric values       │
         │  • Null-sentinels for   │
         │    unresolvable fields  │
         │  • Schema key audit     │
         │  • CID injection        │
         └─────────────────────────┘
                       │
                       ▼
              [ output.json ]
```

---

## Repository Structure

```
.
├── main.py             # Pipeline orchestrator and entry point
├── LLM1.py             # Layer 1 — multi-candidate literature extraction
├── fusion.py           # Layer 2 — deterministic statistical fusion engine
├── fusion_config.py    # Tune all mathematical thresholds here (no code changes needed)
├── LLM2.py             # Layer 3 — biological plausibility verification
├── schema.py           # Single source of truth for output structure
├── output.json         # Sample output from a completed pipeline run
├── .env                # NOT committed — holds HF_TOKEN (listed in .gitignore)
└── .gitignore          # Excludes .env and __pycache__/
```

---

## Data Schema

Defined in `schema.py` as `REQUIRED_SCHEMA`. This compact structure is intentional — a minimal schema saves significant input/output tokens and allows the pipeline to operate efficiently on free-tier Hugging Face inference endpoints.

```json
{
  "cid": 0,
  "molecule": {
    "molecular_weight": 0.0,
    "logP": 0.0
  },
  "interaction_profile": {
    "activity_type": "IC50",
    "activity_value": 0.0,
    "activity_unit": "nM"
  }
}
```

| Field | Type | Description |
|---|---|---|
| `cid` | `int` | PubChem Compound Identifier, resolved before LLM1 runs |
| `molecule.molecular_weight` | `float` | Molecular weight in Daltons (Da) |
| `molecule.logP` | `float` | Octanol-water partition coefficient |
| `interaction_profile.activity_type` | `str` | Pharmacological activity type: `IC50`, `EC50`, `Ki`, or `Kd` |
| `interaction_profile.activity_value` | `float` | Fused activity value, always in nM |
| `interaction_profile.activity_unit` | `str` | Always `"nM"` — enforced at both LLM1 and LLM2 layers |

The schema serves triple duty: it is the prompt template sent to LLM1, the structure reference sent to LLM2, and the validation target audited in `main.py`. Changing the schema in `schema.py` automatically propagates changes through all three layers without touching any other file — except `fusion_config.py` if the new field requires custom fusion behaviour.

---

## Mathematical Engine — Fusion Algorithm

`fusion.py` is the only component that is entirely deterministic and contains zero LLM calls. It recursively traverses the extracted dict, treating every `list`-valued leaf as a pool of candidates to be fused into a single scalar, and every `dict`-valued node as a branch to recurse into. Scalar values are passed through directly (used for `cid`).

Each candidate in a pool has the shape:

```json
{"value": 312.37, "confidence": 0.85, "source_type": "PubChem"}
```

The following stages are applied in strict order:

### Stage 1 — Confidence Threshold Filtering

Any candidate whose `confidence` score is below the hard minimum is discarded before any other processing occurs. This avoids wasting unit-normalization computation on records that will be dropped anyway.

```
C_min = 0.4
```

Candidates with `confidence < 0.4` are permanently removed from the pool.

If LLM1 returns a plain scalar list (e.g., `["IC50", "EC50"]`) instead of properly structured candidate dicts — a known failure mode of small instruction-tuned models — the fusion engine falls back gracefully to returning the first element rather than crashing.

### Stage 2 — Per-Field Confidence Capping

LLMs are structurally overconfident in certain domains. A field like `logP` is a computed property that an 8B model cannot truly "know" to high certainty. Confidence inputs for such fields are capped to a configured ceiling before the weighted average is calculated.

```
C_adjusted = min(C_extracted, CONFIDENCE_CAP[field])
```

Configured caps (from `fusion_config.py`):

| Field | Cap |
|---|---|
| `logP` | 0.7 |
| `molecular_weight` | 0.8 |

### Stage 3 — Multi-Unit Normalization

Applied exclusively to `activity_value` candidates. Any candidate carrying a per-candidate `unit` key has its value converted to nanomolar (nM) before entering the outlier filter. Candidates without a `unit` key are assumed to already be in nM (enforced by LLM1's system prompt).

Supported conversions:

| Input Unit | Factor → nM |
|---|---|
| `M` | × 1,000,000,000 |
| `mM` | × 1,000,000 |
| `uM` / `µM` | × 1,000 |
| `nM` | × 1 (no-op) |

Unknown units trigger a console warning and fall back to assuming nM.

### Stage 4 — Plausibility Floor Enforcement

Values below a defined physical floor are considered extraction artifacts (e.g., a unit conversion error that wasn't caught by LLM1) and are discarded before outlier removal.

```
Configured floor:
  activity_value: 1.0 nM   (below 1 nM is sub-nanomolar — implausible for most drugs)
```

If every candidate in the pool falls below the floor, the field returns `None` rather than producing a false value.

### Stage 5 — Consensus Hallucination Detection

A well-known failure mode of LLMs generating multiple candidates from the same internal "memory" is that all candidates end up with identical values — zero real-world variance. This is a signal that the model is not actually drawing from multiple independent sources.

When all numeric candidates in a pool share exactly the same value, the fusion engine:

1. Emits a `[FUSION] WARNING` to the console naming the affected field.
2. Caps every candidate's confidence to `0.5`, preventing any single source from dominating the weighted average.

### Stage 6 — Dynamic Outlier Rejection

Applied to all numeric pools with two or more candidates. The strategy is chosen per field type:

**Absolute deviation** (for linear-scale properties with known agreement tolerances):

```
|V_i − Median(V)| ≤ Δ_threshold
```

| Field | Threshold |
|---|---|
| `logP` | ± 0.5 |
| `molecular_weight` | ± 50 Da |

**Ratio-based exclusion** (for logarithmic-scale properties like bioactivity values):

```
max(|V_i|, |Median|) / max(min(|V_i|, |Median|), 1×10⁻⁹) ≤ 5
```

A 5× variance tolerance is applied. The `1×10⁻⁹` denominator floor prevents division-by-zero when the median is at or near zero. If the filtered pool would become empty, the original set is retained.

### Stage 7 — Confidence-Weighted Fusion

All candidates that survive the preceding stages are merged into a single scalar using a weighted average where each value's contribution is proportional to its adjusted confidence:

```
V_fused = Σ(V_i × C_i) / Σ(C_i)
```

If the total confidence is exactly zero (pathological case), a simple arithmetic mean is used as a fallback.

For string-typed fields (e.g., `activity_type`), the candidate with the highest confidence score is returned directly, with no averaging.

---

## Module Reference

### `main.py`

The pipeline orchestrator. Responsibilities:

- Loads `HF_TOKEN` from `.env` via `python-dotenv` and initialises the `InferenceClient`.
- **CID resolution**: If the input is already a digit string, it is cast to `int` directly. Otherwise, `fetch_cid()` queries the PubChem REST API first by compound name (`/compound/name/.../cids/JSON`) and falls back to SMILES lookup (`/compound/smiles/.../cids/JSON`) if the name query returns a non-200. Returns `0` on failure.
- **Schema-driven numeric path discovery**: `_get_numeric_paths()` walks `REQUIRED_SCHEMA` recursively at runtime and returns every path to a numeric leaf (excluding `cid`). This means the guard layer automatically adapts when new numeric fields are added to the schema — no manual path lists to maintain.
- **LLM1 output normalization**: LLM1 sometimes wraps its output under a CID key (e.g., `{"5754": {"molecule": ...}}`) instead of returning a flat schema. The normalizer detects when no top-level schema keys are present and unwraps the single wrapper key.
- **Numeric guard re-injection**: After LLM2 returns, every numeric path produced by the fusion engine is written back into the final output, overwriting any value LLM2 may have altered. An `[GUARD]` message is logged if a discrepancy is detected.
- **Null sentinels**: Where fusion returned `None` (unresolvable field), any schema-default `0.0` that LLM2 may have placed is overwritten with `None`, so the output honestly signals an unresolvable field rather than presenting a fake zero.
- **Schema key audit**: `_validate_keys()` performs a recursive comparison of the final output against `REQUIRED_SCHEMA` and logs `[WARN]` for any missing keys.
- **Float rounding**: All float values in the final output are rounded to 4 decimal places before writing to `output.json`.
- **Debug mode**: When enabled at the prompt, prints the full LLM1 extraction dict and the fused intermediate before LLM2 runs.

### `LLM1.py`

The extraction layer. Key behaviours:

- Uses `meta-llama/Llama-3.1-8B-Instruct` via the Hugging Face Inference API with `max_tokens=2000` and a starting `temperature=0.6`.
- The system prompt enforces a strict **non-resolution contract**: the model must output multiple conflicting candidates per field, never pick a winner or average values itself.
- Preferred source hierarchy enforced in the prompt: PubChem → ChEMBL → BindingDB → peer-reviewed literature → other.
- `activity_type` is restricted to exactly four values: `IC50`, `EC50`, `Ki`, `Kd`.
- Explicit unit conversion rules in the prompt include worked examples of common errors (e.g., `0.035` in the output is likely `0.035 uM = 35 nM`; `1.7e-05` is M-scale and must become `17000 nM`).
- Confidence below 0.4 signals the model should not fabricate a value for an unknown field.
- Retry loop: up to 3 attempts on `HfHubHTTPError` (timeout / server error) with a 10-second wait. On `json.JSONDecodeError` or `ValueError`, the temperature is nudged upward by `+0.1` per retry (capped at 1.0) to encourage more varied JSON formatting.
- `clean_json_output()`: Extracts JSON from the last ` ```json ``` ` code block in the response. Falls back to extracting the outermost `{...}` with a regex if no code fence is present.

### `fusion.py`

The deterministic fusion engine. Key internals:

- `fuse(extracted)`: Top-level recursive walker. Dispatches `dict` branches to recursion, `list` leaves to `_fuse_candidates()`, and scalars to passthrough.
- `_fuse_candidates(candidates, field_name)`: Implements all seven fusion stages described above. The `field_name` parameter is used to look up per-field configuration from `fusion_config.py`.
- `_normalize_units(candidates)`: Converts `activity_value` candidates from any supported unit to nM using the `_UNIT_TO_NM` lookup table. Does not mutate the original candidate dicts — creates a shallow copy via `dict(c)` before modifying.
- `_check_consensus_hallucination(candidates)`: Returns `True` if all numeric values in the pool are identical (requires at least 2 values to make the determination).
- `_is_candidate_dict(item)`: Guard that returns `True` only if the item is a `dict` with a `"value"` key — used to detect the plain scalar list fallback path.
- `MIN_CONFIDENCE = 0.4` is a module-level constant, separate from `fusion_config.py`, as it is a structural minimum rather than a tunable threshold.

### `fusion_config.py`

The single control file for all mathematical policy. Contains three dicts:

```python
ABSOLUTE_THRESHOLD = {
    "logP": 0.5,
    "molecular_weight": 50.0,
}

PLAUSIBILITY_FLOOR = {
    "activity_value": 1.0,
}

CONFIDENCE_CAP = {
    "logP": 0.7,
    "molecular_weight": 0.8,
}
```

Keys must exactly match leaf field names defined in `schema.py`. This is the only file that needs editing when the schema gains new numeric fields requiring custom fusion behaviour.

### `LLM2.py`

The verification and JSON-building layer. Key behaviours:

- Uses `meta-llama/Llama-3.1-8B-Instruct` with `max_tokens=800` and `temperature=0.1` for deterministic, conservative output.
- System prompt rules:
  - Check biological plausibility of all values.
  - Flag and correct structural or functional contradictions.
  - Fill missing string/structural fields (SMILES, IUPAC, biological context) from model knowledge.
  - **Hard rule: Do NOT change any numerically fused activity values.**
  - Output only fields present in the schema — no extra keys.
  - `activity_type` disambiguation: if the fused string contains `|` or multiple options, resolve to a single value based on the molecule's known pharmacology.
  - `activity_unit` enforcement: must always be `nM`. Conversions are specified (1M=1e9nM, 1mM=1e6nM, 1uM=1000nM). If already nM, no conversion.
  - Strict type enforcement: counts as integers, scores as floats.
  - Physicochemical sanity bounds: `logP` typically −2 to 6, `molecular_weight` typically 100–900 Da. Values far outside these ranges, or contradicting known chemistry, are flagged by setting the value to `null`.
- `_clean_json_output()`: mirrors LLM1's `clean_json_output()` — extracts the last ` ```json ``` ` block or falls back to outermost `{...}`.
- Retry loop: up to 3 attempts. `HfHubHTTPError` retries wait 10 seconds. Malformed JSON retries also wait 10 seconds. After exhausting all retries, raises `RuntimeError`.

### `schema.py`

Exports a single constant, `REQUIRED_SCHEMA`. Intentionally compact: a minimal schema reduces both input token cost (schema is included in every LLM prompt) and the surface area for structural errors. The Python dict's leaf value types (`0` for int, `0.0` for float, `"string"` for str) serve as type hints that the guard layer and `_get_numeric_paths()` use at runtime.

---

## Guards & Safety Mechanisms

The pipeline implements several layers of protection against LLM unreliability:

| Guard | Location | What it prevents |
|---|---|---|
| Non-resolution contract | LLM1 system prompt | Averaging/conflict resolution before the math engine runs |
| Unit conversion enforcement | LLM1 system prompt + fusion.py | Mixed-unit candidates distorting the weighted average |
| Confidence gating (`MIN_CONFIDENCE = 0.4`) | fusion.py | Low-quality candidates polluting the mean |
| Plausibility floor | fusion.py + fusion_config.py | Sub-nanomolar artifacts from unit conversion errors |
| Consensus hallucination detection | fusion.py | Zero-variance pools where the model fabricated agreement |
| Numeric guard re-injection | main.py | LLM2 silently altering fused values |
| Null sentinel overwrite | main.py | LLM2 replacing an honest `null` with a schema-default `0.0` |
| Schema key audit | main.py | LLM2 dropping required fields silently |
| LLM1 output normalization | main.py | CID-wrapped output structure not matching schema keys |
| Temperature nudge on retry | LLM1.py | Identical malformed JSON on consecutive retries |
| Physicochemical sanity bounds | LLM2 system prompt | Chemically impossible values reaching the output |

---

## Fusion Configuration Reference

Edit `fusion_config.py` to change statistical behaviour without touching any algorithm code.

### `ABSOLUTE_THRESHOLD`

Fields listed here use absolute deviation from the median as the outlier criterion. Any candidate further than the threshold from the median is discarded.

```python
ABSOLUTE_THRESHOLD = {
    "logP": 0.5,              # reputable sources agree within 0.5 units
    "molecular_weight": 50.0, # candidates must be within 50 Da of median
}
```

Fields **not** listed here fall back to the ratio-based exclusion (5× rule).

### `PLAUSIBILITY_FLOOR`

Values below the floor are physically implausible and dropped before outlier removal runs.

```python
PLAUSIBILITY_FLOOR = {
    "activity_value": 1.0,    # below 1 nM is sub-nanomolar; likely a conversion error
}
```

### `CONFIDENCE_CAP`

Caps the maximum effective confidence for fields where LLMs are structurally overconfident. Reduces over-weighting of hallucinated "certain" values.

```python
CONFIDENCE_CAP = {
    "logP": 0.7,              # computed property; no 8B model should be 100% confident
    "molecular_weight": 0.8,  # usually correct but occasional formula errors
}
```

---

## Output Format

`output.json` is written after every successful pipeline run. All float values are rounded to 4 decimal places. The `cid` field is injected by `main.py` directly (not derived from LLM output) to guarantee correctness.

Sample output for CID 5754 (Ibuprofen):

```json
{
  "cid": 5754,
  "molecule": {
    "molecular_weight": 312.37,
    "logP": 1.34
  },
  "interaction_profile": {
    "activity_type": "IC50",
    "activity_value": 73.3778,
    "activity_unit": "nM"
  }
}
```

Fields set to `null` in the output indicate that fusion was unable to resolve a plausible value from the extracted candidates — this is an intentional honest signal, not a bug.

---

## Quickstart

### Prerequisites

- Python 3.8 or higher
- An active Hugging Face account with a User Access Token (Read permission is sufficient)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/biomedical-property-pipeline.git
cd biomedical-property-pipeline

# 2. Install dependencies
pip install huggingface_hub python-dotenv requests

# 3. Create the environment file
touch .env
```

### Environment Variables

Add your Hugging Face token to `.env`:

```env
HF_TOKEN=hf_your_actual_token_string_goes_here
```

The `.gitignore` is already configured to exclude `.env` and `__pycache__/` from version control.

### Running the Pipeline

```bash
python main.py
```

You will be prompted for two inputs:

```
Enter molecule name / SMILES / CID: Imatinib
Debug mode? Show LLM1 extraction + fused intermediate? (y/n): y
```

**Accepted input formats:**

| Format | Example |
|---|---|
| Common name | `Ibuprofen` |
| IUPAC name | `2-(4-isobutylphenyl)propanoic acid` |
| SMILES string | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` |
| PubChem CID (integer string) | `5754` |

### Console Output During Execution

```
[CID] Looking up PubChem CID for: Imatinib
[CID] Found: 5288963

[LLM1] Extracting properties for: Imatinib
[LLM1] Done.

── LLM1 RAW EXTRACTION ──────────────────────────────────────
{ ... multi-candidate JSON ... }

[FUSION] Running outlier removal + confidence-weighted mean...
[FUSION] Done.

── FUSED INTERMEDIATE ───────────────────────────────────────
{ ... single-value JSON ... }

[LLM2] Verifying and building final record...
[LLM2] Done.

── FINAL OUTPUT ─────────────────────────────────────────────
{ ... validated JSON ... }

Saved to output.json
```

The `[GUARD]`, `[WARN]`, `[INFO]`, and `[FUSION] WARNING` prefixes in console output all indicate specific pipeline events worth inspecting when debugging an unusual result.

---

## Dependencies

| Package | Purpose |
|---|---|
| `huggingface_hub` | Inference API client; `InferenceClient.chat_completion()` for both LLM calls; `HfHubHTTPError` for retry handling |
| `python-dotenv` | Loads `HF_TOKEN` from `.env` without it being hardcoded |
| `requests` | PubChem REST API calls for CID resolution |
| `statistics` | `statistics.median()` used in the fusion engine for outlier filtering |
| `json`, `re`, `math`, `time` | Standard library — JSON parsing, regex extraction, float comparison, retry sleep |

Install all third-party dependencies with:

```bash
pip install huggingface_hub python-dotenv requests
```

---

## Design Decisions & Tradeoffs

**Why a small 8B model?**
The pipeline is designed to run on the Hugging Face free inference tier. The schema is intentionally minimal partly for this reason — fewer tokens in/out means the pipeline stays within rate limits and responds in reasonable time. A larger model (70B+) would likely produce higher-confidence extractions and fewer malformed JSON outputs, but would require a paid endpoint or local GPU.

**Why not use LLM2 to also do the numeric fusion?**
Numeric fusion by an LLM is inherently probabilistic. Two identical prompts with the same candidates can produce different weighted averages because temperature > 0. The deterministic Python engine guarantees reproducible results — running the same extraction through the fusion layer twice always produces the same fused value.

**Why re-inject fused numerics after LLM2?**
LLM2 is instructed not to change numeric values, but it is a language model — instruction following is not guaranteed, especially at low temperature where the model may "correct" a value it perceives as surprising. The guard in `main.py` is a hard enforcement layer, not a trust-but-verify check.

**Why is `MIN_CONFIDENCE = 0.4` a constant rather than config?**
It is a structural quality floor, not a per-field policy. Candidates below 0.4 are by definition too uncertain to be useful regardless of field. Moving it into `fusion_config.py` as a per-field value would add complexity without meaningful benefit in the current schema.

**Why does the schema use `0` and `0.0` as placeholder values?**
The types of the leaf values in `REQUIRED_SCHEMA` are read at runtime by `_get_numeric_paths()` in `main.py` to discover which paths need numeric protection. Using `0` (int) and `0.0` (float) makes this introspection trivially reliable without needing a separate type annotation structure.

---

## Extending the Schema

To add a new field to the output (e.g., `tpsa` — topological polar surface area):

1. **`schema.py`** — Add the field at the appropriate nesting level:
   ```python
   REQUIRED_SCHEMA = {
     "cid": 0,
     "molecule": {
       "molecular_weight": 0.0,
       "logP": 0.0,
       "tpsa": 0.0       # ← new field
     },
     "interaction_profile": { ... }
   }
   ```

2. **`fusion_config.py`** — If `tpsa` needs an absolute threshold instead of the default ratio-based filter, add it:
   ```python
   ABSOLUTE_THRESHOLD = {
       "logP": 0.5,
       "molecular_weight": 50.0,
       "tpsa": 20.0,     # ← reputable sources agree within 20 Å²
   }
   ```

That is the entirety of the change needed. `main.py`, `LLM1.py`, `LLM2.py`, and `fusion.py` all derive their behaviour from the schema and config at runtime.

---

## Known Limitations

- **Model hallucination**: `meta-llama/Llama-3.1-8B-Instruct` is a small model. For obscure molecules with limited literature presence, it may fabricate plausible-sounding but incorrect candidates. The confidence gating and consensus hallucination detector mitigate this but cannot eliminate it entirely.
- **LLM1 unit conversion**: Despite explicit prompt instructions with worked examples, the model occasionally emits unconverted M or uM values without a `unit` key (breaking the per-candidate unit normalization). The LLM2 unit enforcement rule acts as a second-line catch for this.
- **Single-candidate pools**: If only one candidate passes the confidence filter, the "outlier removal" step has no peers to compare against. The single value is accepted without filtering. This is intentional — there is no basis for rejection with n=1.
- **Free-tier rate limits**: The Hugging Face free tier imposes request quotas. The retry logic (3 attempts, 10-second wait) handles transient 429/503 errors but sustained rate-limiting will cause the pipeline to fail after exhausting retries.
- **PubChem CID fallback**: If both the name and SMILES lookups fail, `cid` is set to `0`. This is a valid sentinel value, not a PubChem record. Downstream consumers of `output.json` should treat `cid == 0` as unresolved.
- **`activity_unit` string field**: `activity_unit` is a string in the schema and is not numerically fused or guard-protected. It is set to `"nM"` by LLM2 per the unit enforcement rule, but if LLM2 fails to apply the rule, no numeric guard catches it.