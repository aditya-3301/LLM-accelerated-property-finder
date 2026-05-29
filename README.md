# Biomedical Property Reconstruction Pipeline

An automated, three-tier data-refinement architecture designed to extract, resolve, and verify physicochemical and pharmacological descriptors of small molecules from unstructured scientific literature and public databases. By employing a **hybrid sandwich pattern — LLM → Code → LLM** — the system combines the broad extraction capabilities of Large Language Models with the strict reliability of a deterministic, type-aware mathematical engine in Python. This ensures statistical anomalies are rejected, integer fields are never averaged into floats, and biological consistency is enforced before any records are committed to disk.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture & Pipeline Flow](#architecture--pipeline-flow)
- [Repository Structure](#repository-structure)
- [Data Schema](#data-schema)
- [Mathematical Engine — Fusion Algorithm](#mathematical-engine--fusion-algorithm)
  - [Stage 1 — Confidence Threshold Filtering & Source-Priority Boosting](#stage-1--confidence-threshold-filtering--source-priority-boosting)
  - [Stage 2 — Per-Field Confidence Capping](#stage-2--per-field-confidence-capping)
  - [Stage 3 — Plausibility Window Enforcement](#stage-3--plausibility-window-enforcement)
  - [Stage 4 — Consensus Hallucination Detection](#stage-4--consensus-hallucination-detection)
  - [Stage 5 — Dynamic Outlier Rejection](#stage-5--dynamic-outlier-rejection)
  - [Stage 6 — Type-Aware Fusion](#stage-6--type-aware-fusion)
  - [Stage 7 — Composition Fusion & Renormalisation](#stage-7--composition-fusion--renormalisation)
- [Module Reference](#module-reference)
  - [main.py](#mainpy)
  - [LLM1.py](#llm1py)
  - [fusion.py](#fusionpy)
  - [fusion_config.py](#fusion_configpy)
  - [LLM2.py](#llm2py)
  - [drug_descriptors.py](#drug_descriptorspy)
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

Small-molecule property data is scattered across public databases (PubChem, ChEMBL, DrugBank, BindingDB) and peer-reviewed literature in inconsistent formats, with conflicting reported values and varying confidence levels depending on the assay method and reporting source. A naive LLM query over this space will either hallucinate a single confident-sounding value or silently pick one of several contradicting figures.

This pipeline solves that problem by:

1. **Forcing disagreement** — LLM1 is contractually prohibited from resolving conflicts. It must emit every candidate it finds with an individual confidence score and source type, enabling the downstream math engine to adjudicate.
2. **Deterministically cleaning the noise** — a pure Python fusion engine, with no LLM involvement, applies type-aware mathematical filters and source-priority-weighted fusion strategies. Integer fields are never averaged into floats; composition dicts are fused element-by-element.
3. **Verifying biological plausibility** — LLM2 inspects the fused result for domain-level contradictions, fills any genuinely missing string fields from model knowledge, and produces a final clean record with provenance metadata and a warnings list — without being permitted to silently alter any numeric value already produced by the fusion engine.

The result is a production-grade JSON record where every numeric value is traceable to a deterministic calculation, not an LLM guess.

---

## Architecture & Pipeline Flow

```
[ User Input: Molecule Name / CID ]
                       │
                       ▼
       ┌───────────────────────────────┐
       │  PubChem API Resolution Layer │
       │  REST endpoint: /pug/compound │
       │  Name → CID; CID → IUPAC +   │
       │  common name enrichment       │
       └───────────────────────────────┘
                       │
          (Resolves canonical CID integer +
           enriched molecule label)
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  LAYER 1 — Extraction  (LLM1.py)                         │
│                                                          │
│  Model  : meta-llama/Llama-3.1-70B-Instruct              │
│  Role   : Biomedical extraction agent                    │
│  Output : Nested dict of candidate arrays                │
│           Each leaf → list of                            │
│           {"value": ..., "confidence": 0–1,              │
│            "source_type": pubchem|chembl|drugbank|...}   │
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
│  Step A : Drop candidates with effective confidence      │
│           < 0.4 (conf × source-priority boost)           │
│  Step B : Apply per-field confidence caps                │
│  Step C : Plausibility floor + ceiling enforcement       │
│  Step D : Detect consensus hallucination                 │
│           (skipped for deterministic descriptor fields)  │
│  Step E : Median-based outlier removal                   │
│           (absolute threshold OR 5× ratio)               │
│  Step F : Type-aware fusion dispatch:                    │
│           float     → confidence-weighted mean           │
│           integer   → weighted mode → cast to int        │
│           categorical/smiles → highest-confidence wins   │
│           list      → dedup union by confidence rank     │
│           composition → per-element float fusion + renorm│
│  Step G : CAS number: lowest registry number preferred   │
│           among max-confidence candidates                │
│  Step H : Deduplication of secondary_accession_numbers   │
└──────────────────────────────────────────────────────────┘
                       │
     (Single typed fused value per field — no LLM involved)
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  LAYER 3 — Verification  (LLM2.py)                       │
│                                                          │
│  Model  : meta-llama/Llama-3.1-70B-Instruct              │
│  Role   : Biomedical verification agent & JSON builder   │
│  Checks : Biological plausibility, cross-field           │
│           consistency, composition fraction sums,        │
│           exact_mol_weight ≠ molecular_weight            │
│  Output : Final schema-conformant JSON object with       │
│           optional top-level "warnings" and "provenance" │
│  Hard rule: Must NOT silently alter any fused numeric    │
│  Retry  : Up to 3 attempts on timeout or malformed JSON  │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
         ┌──────────────────────────────┐
         │  main.py guard layer         │
         │  • Re-injects fused          │
         │    numeric values            │
         │  • String-field guard        │
         │    (non-fillable fields)     │
         │  • Composition dict guard    │
         │  • SMILES validation +       │
         │    clearing on failure       │
         │  • Deterministic recompute   │
         │    from molecular_formula    │
         │  • Chemical consistency      │
         │    checks (formula↔MW,       │
         │    SMILES↔HAC, comp sums)    │
         │  • Integer type enforcement  │
         │  • Duplicate accession guard │
         │  • Null sentinels for        │
         │    unresolvable fields       │
         │  • Schema key audit          │
         │  • CID injection             │
         │  • Provenance attachment     │
         └──────────────────────────────┘
                       │
                       ▼
              [ output.json ]
```

---

## Repository Structure

```
.
├── main.py               # Pipeline orchestrator and entry point
├── LLM1.py               # Layer 1 — multi-candidate literature extraction
├── fusion.py             # Layer 2 — deterministic type-aware fusion engine
├── fusion_config.py      # Tune all mathematical thresholds here (no code changes needed)
├── LLM2.py               # Layer 3 — biological plausibility verification
├── drug_descriptors.py   # Schema, field types, source priority, deterministic field sets
├── output.json           # Sample output from a completed pipeline run (git-ignored)
├── .env                  # NOT committed — holds HF_TOKEN (listed in .gitignore)
└── .gitignore            # Excludes .env, output.json, and __pycache__/
```

---

## Data Schema

Defined in `drug_descriptors.py` as `REQUIRED_SCHEMA`. The schema is intentionally structured to cover identity metadata, discrete integer molecular descriptors, and continuous float physicochemical properties — all fields amenable to deterministic or literature-sourced extraction.

```json
{
  "cid": 0,
  "identity": {
    "drugbank_id": "",
    "secondary_accession_numbers": [],
    "common_name": "",
    "cas_number": "",
    "unii": "",
    "synonyms": [],
    "smiles": ""
  },
  "molecule": {
    "number_of_heavy_atoms": 0,
    "molecular_formula": "",
    "molecular_composition": {},
    "molecular_weight": 0.0,
    "exact_mol_weight": 0.0,
    "net_formal_charge": 0,
    "alogp": 0.0,
    "num_h_acceptors_lipinski": 0,
    "num_h_donors_lipinski": 0,
    "num_rotatable_bonds": 0,
    "molecular_polar_surface_area": 0.0,
    "num_h_acceptors": 0,
    "num_h_donors": 0
  }
}
```

| Field | Type | Description |
|---|---|---|
| `cid` | `int` | PubChem Compound Identifier, resolved and injected by `main.py` before LLM1 runs |
| `identity.drugbank_id` | `str` | Primary DrugBank accession (e.g. `DB00945`) |
| `identity.secondary_accession_numbers` | `list[str]` | Additional DrugBank accession numbers; must not duplicate `drugbank_id` |
| `identity.common_name` | `str` | Most widely recognised common/brand name |
| `identity.cas_number` | `str` | Canonical CAS number for the free-acid/free-base neutral form (lowest-numbered, earliest-registered) |
| `identity.unii` | `str` | FDA Unique Ingredient Identifier |
| `identity.synonyms` | `list[str]` | Deduped union of known synonyms across sources |
| `identity.smiles` | `str` | Canonical SMILES string; validated by `main.py`; empty string on validation failure |
| `molecule.number_of_heavy_atoms` | `int` | Count of all non-hydrogen atoms; authoritative value derived from SMILES or formula |
| `molecule.molecular_formula` | `str` | Hill-notation molecular formula (e.g. `C13H18O2`) |
| `molecule.molecular_composition` | `dict` | Element → mass fraction mapping (e.g. `{"C": 0.777, "H": 0.090, "O": 0.133}`); fractions sum to 1.0 |
| `molecule.molecular_weight` | `float` | Average molecular weight in Daltons (Da), using standard atomic weights |
| `molecule.exact_mol_weight` | `float` | Monoisotopic mass in Da, using most-abundant isotope masses; must differ from `molecular_weight` |
| `molecule.net_formal_charge` | `int` | Net formal charge of the molecule |
| `molecule.alogp` | `float` | Calculated octanol-water partition coefficient (Ghose–Crippen) |
| `molecule.num_h_acceptors_lipinski` | `int` | Lipinski hydrogen-bond acceptor count (N + O atoms) |
| `molecule.num_h_donors_lipinski` | `int` | Lipinski hydrogen-bond donor count (NH + OH groups) |
| `molecule.num_rotatable_bonds` | `int` | Number of rotatable bonds |
| `molecule.molecular_polar_surface_area` | `float` | Topological polar surface area in Å² |
| `molecule.num_h_acceptors` | `int` | Extended hydrogen-bond acceptor count |
| `molecule.num_h_donors` | `int` | Extended hydrogen-bond donor count |

The schema serves triple duty: it is the extraction template sent to LLM1, the structure reference sent to LLM2, and the validation target audited in `main.py`. Changing the schema in `drug_descriptors.py` automatically propagates through all three layers without touching any other file — except `fusion_config.py` if the new field requires custom fusion behaviour.

The leaf value types (`0` for int, `0.0` for float, `""` for str, `[]` for list, `{}` for dict) serve as type hints that the guard layer and `_get_numeric_paths()` use at runtime.

---

## Mathematical Engine — Fusion Algorithm

`fusion.py` is the only component that is entirely deterministic and contains zero LLM calls. It recursively traverses the extracted dict, dispatching each list-valued leaf to a **typed fusion strategy** based on `FIELD_TYPES` in `drug_descriptors.py`, and recursing into any nested dict. Scalar values (e.g. `cid`) pass through unchanged.

Each candidate in a pool has the shape:

```json
{"value": 312.37, "confidence": 0.85, "source_type": "pubchem"}
```

The following stages are applied in strict order:

### Stage 1 — Confidence Threshold Filtering & Source-Priority Boosting

Before any other processing, each candidate's raw confidence is converted to an **effective confidence** that folds in a source-priority score:

```
eff = confidence × (1 + 0.1 × (priority − 1))
```

Source priorities (from `drug_descriptors.py`):

| Source | Priority | Effective multiplier |
|---|---|---|
| `pubchem`, `drugbank` | 5 | ×1.4 |
| `chembl` | 4 | ×1.3 |
| `bindingdb` | 3 | ×1.2 |
| `literature` | 2 | ×1.1 |
| `other` | 1 | ×1.0 |

Candidates with effective confidence below `MIN_CONFIDENCE = 0.4` are permanently removed from the pool before any fusion step.

### Stage 2 — Per-Field Confidence Capping

LLMs are structurally overconfident for deterministic descriptor fields. Confidence inputs for such fields are capped to a configured ceiling before the weighted strategies are applied.

```
C_adjusted = min(C_extracted, CONFIDENCE_CAP[field])
```

Configured caps (from `fusion_config.py`):

| Field | Cap |
|---|---|
| `alogp` | 0.70 |
| `molecular_polar_surface_area` | 0.75 |
| `molecular_weight` | 0.80 |
| `number_of_heavy_atoms` | 0.80 |
| `num_h_acceptors_lipinski` | 0.80 |
| `num_h_donors_lipinski` | 0.80 |
| `num_h_acceptors` | 0.80 |
| `num_h_donors` | 0.80 |
| `num_rotatable_bonds` | 0.80 |
| `exact_mol_weight` | 0.85 |
| `net_formal_charge` | 0.85 |

### Stage 3 — Plausibility Window Enforcement

Values outside a defined physical window are considered extraction artifacts and discarded before outlier removal. Both a floor and a ceiling are applied where configured.

**Plausibility floors** (`PLAUSIBILITY_FLOOR` in `fusion_config.py`):

| Field | Floor |
|---|---|
| `number_of_heavy_atoms` | 1 |
| `num_rotatable_bonds` | 0 |
| `molecular_weight` | 10.0 Da |
| `exact_mol_weight` | 10.0 Da |
| `molecular_polar_surface_area` | 0.0 Å² |

**Plausibility ceilings** (`PLAUSIBILITY_CEILING` in `fusion_config.py`):

| Field | Ceiling |
|---|---|
| `molecular_weight` | 2000.0 Da |
| `exact_mol_weight` | 2000.0 Da |
| `num_h_acceptors_lipinski` | 20 |
| `num_h_donors_lipinski` | 10 |
| `num_h_acceptors` | 20 |
| `num_h_donors` | 10 |
| `num_rotatable_bonds` | 50 |
| `molecular_polar_surface_area` | 500.0 Å² |
| `alogp` | 10.0 |
| `number_of_heavy_atoms` | 500 |

If every candidate falls outside the window, the field returns `None` rather than producing a false value.

### Stage 4 — Consensus Hallucination Detection

A well-known failure mode of LLMs generating multiple candidates from the same internal "memory" is that all candidates end up with identical values — zero real-world variance. This is a signal the model is not drawing from multiple independent sources.

Critically, this check is **skipped** for all fields listed in `DETERMINISTIC_FIELDS` (MW, HBA, formula, etc.) because identical values across PubChem, DrugBank, and ChEMBL for these fields are entirely expected — they derive from the same molecular structure, not independent measurements. Flagging them would be a false positive.

When all numeric candidates in a non-deterministic pool share exactly the same value, the fusion engine:

1. Emits a `[FUSION] WARNING` to the console naming the affected field.
2. Caps every candidate's confidence to `0.5`, preventing any single source from dominating the weighted average.

### Stage 5 — Dynamic Outlier Rejection

Applied to all numeric pools with two or more candidates. The strategy is chosen per field type:

**Absolute deviation** (for linear-scale properties with known agreement tolerances):

```
|V_i − Median(V)| ≤ Δ_threshold
```

| Field | Threshold |
|---|---|
| `molecular_weight` | ± 50 Da |
| `exact_mol_weight` | ± 0.005 Da |
| `alogp` | ± 0.5 |
| `molecular_polar_surface_area` | ± 10.0 Å² |

**Ratio-based exclusion** (for all other continuous numeric fields):

```
max(|V_i|, |Median|) / max(min(|V_i|, |Median|), 1×10⁻⁹) ≤ 5
```

A 5× variance tolerance is applied. The `1×10⁻⁹` denominator floor prevents division-by-zero when the median is at or near zero. If the filtered pool would become empty, the original set is retained.

### Stage 6 — Type-Aware Fusion

All candidates that survive the preceding stages are merged using a strategy determined by `FIELD_TYPES`:

**`float`** — confidence-weighted mean using effective confidence:
```
V_fused = Σ(V_i × eff_i) / Σ(eff_i)
```
If the total effective confidence is exactly zero (pathological case), a simple arithmetic mean is used.

**`integer`** — weighted mode: candidates are bucketed by integer value (after rounding), and the bucket with the highest total effective confidence wins. This guarantees the result is always a discrete integer — never `3.2857` for `num_rotatable_bonds`. Ties are broken by source priority.

**`categorical`** (including `molecular_formula`) — highest effective confidence wins. For `cas_number` specifically, among all candidates tied at maximum effective confidence, the lexicographically smallest registry number is preferred (the earliest-registered CAS is the canonical primary accession).

**`smiles`** — highest effective confidence string candidate is returned. Structural validation is deferred to `main.py`.

**`list`** (`synonyms`, `secondary_accession_numbers`) — deduplication union: all unique non-empty strings from candidates above the minimum confidence threshold, ordered by descending effective confidence. Deduplication is case-insensitive; original casing is preserved.

### Stage 7 — Composition Fusion & Renormalisation

`molecular_composition` (field type `"composition"`) receives dedicated handling because it is a dict of `{element: mass_fraction}` rather than a scalar.

Each element's fraction is fused independently using the same `_fuse_float` logic (stages 3–6 above). After all elements are fused, the resulting dict is **renormalised** so all fractions sum exactly to 1.0:

```
fraction_renorm[e] = fraction_fused[e] / Σ(fraction_fused)
```

LLM1 may also return composition as a legacy string (`"C: 0.600, H: 0.045, O: 0.355"`); the fusion engine parses and converts these before fusing.

---

## Module Reference

### `main.py`

The pipeline orchestrator. Responsibilities:

- Loads `HF_TOKEN` from `.env` via `python-dotenv` and initialises the `InferenceClient`.
- **CID resolution**: If the input is already a digit string, it is cast to `int` directly. Otherwise, `fetch_cid()` queries the PubChem REST API by compound name (`/compound/name/.../cids/JSON`). Returns `0` on failure. Unlike previous versions, there is no SMILES fallback — the pipeline expects a name or raw CID.
- **Name enrichment**: `fetch_name_from_cid()` fetches both the IUPAC name and first synonym for the resolved CID, constructing an enriched label passed to LLM1 (e.g. `"Ibuprofen (IUPAC: 2-[4-(2-methylpropyl)phenyl]propanoic acid)"`) to anchor extraction.
- **Schema-driven numeric path discovery**: `_get_numeric_paths()` walks `REQUIRED_SCHEMA` recursively at runtime and returns every path to a numeric leaf (excluding `cid`). The guard layer automatically adapts when new numeric fields are added to the schema — no manual path lists to maintain.
- **LLM1 output normalisation**: LLM1 sometimes wraps its output under a single wrapper key. The normaliser detects when no top-level schema keys are present and unwraps the wrapper key.
- **Numeric guard re-injection**: After LLM2 returns, every numeric path produced by the fusion engine is written back into the final output, overwriting any value LLM2 may have altered. A `[GUARD]` message is logged if a discrepancy is detected.
- **String-field guard**: Non-fillable string fields (e.g. `smiles`, `molecular_formula`) that were resolved by fusion are also re-injected if LLM2 changed them. Fillable identity fields (`drugbank_id`, `cas_number`, `unii`, etc.) are exempt — LLM2 is permitted to fill these from its knowledge when fusion returned empty.
- **Composition dict guard**: If fusion produced a non-empty `molecular_composition` dict and LLM2 lost or cleared it, the fused dict is restored.
- **SMILES validation**: `_validate_smiles()` performs a sanity check (non-empty, legal characters, balanced brackets, known atom symbols) without requiring RDKit. Invalid SMILES are cleared to an empty string with a `[SMILES]` log message.
- **Deterministic recomputation from formula**: `_recompute_deterministic_from_formula()` uses the stored `molecular_formula` to recompute `molecular_weight` (average), `exact_mol_weight` (monoisotopic), `number_of_heavy_atoms`, and `molecular_composition` using hardcoded atomic mass tables (`ATOMIC_MASS`, `MONOISOTOPIC_MASS`). Only overrides the stored value if it is the schema default or deviates beyond a relative tolerance of 0.5%.
- **Chemical consistency checks**: Three post-LLM2 validators run before output:
  - `_check_formula_mw_consistency()` — warns and overwrites if formula-derived MW deviates more than 1 Da from stored MW.
  - `_check_smiles_heavy_atom_consistency()` — tokenises SMILES and compares to stored `number_of_heavy_atoms`; SMILES-derived count is authoritative.
  - `_validate_composition_dict()` — verifies all fractions are numeric and sum to 1.0 ± 0.01; renormalises in place if off; clears legacy string format.
- **Integer type enforcement**: `_enforce_integer_types()` walks the final output and rounds any integer-typed field that ended up as a float back to int. This is the last-resort guard against LLM2 producing `3.2857` for `num_rotatable_bonds`. Logged as `[INT-GUARD]`.
- **Null sentinels**: Where fusion returned `None` (unresolvable field), any schema-default `0.0`/`0` that LLM2 may have placed is overwritten with `None`, so the output signals an unresolvable field rather than presenting a false zero.
- **Duplicate accession guard**: Removes any value from `secondary_accession_numbers` that is identical (case-insensitive) to `drugbank_id`. Also handled upstream in `fusion.py` — this is a second-line catch.
- **Schema key audit**: `_validate_keys()` performs a recursive comparison of the final output against `REQUIRED_SCHEMA` and logs `[WARN]` for any missing keys.
- **Float rounding**: All float values in the final output are rounded to 4 decimal places before writing to `output.json`.
- **Provenance attachment**: `provenance` metadata emitted by LLM2 is popped from the raw output and re-attached to `final_output` after all guards have run. Logged as `[PROV]`.
- **Debug mode**: When enabled at the prompt, prints the full LLM1 extraction dict and the fused intermediate before LLM2 runs.

### `LLM1.py`

The extraction layer. Key behaviours:

- Uses `meta-llama/Llama-3.1-70B-Instruct` via the Hugging Face Inference API with `max_tokens=2000` and a starting `temperature=0.5`.
- The system prompt enforces a strict **non-resolution contract**: the model must output multiple conflicting candidates per field, never pick a winner or average values itself.
- Source type vocabulary is aligned with `SOURCE_PRIORITY` in `drug_descriptors.py`: `pubchem`, `chembl`, `drugbank`, `bindingdb`, `literature`, `other`.
- Integer fields are explicitly listed in the prompt; the model must not produce float values for `number_of_heavy_atoms`, `net_formal_charge`, `num_h_acceptors_lipinski`, `num_h_donors_lipinski`, `num_rotatable_bonds`, `num_h_acceptors`, `num_h_donors`.
- `molecular_composition` must be extracted as a dict `{element: mass_fraction}`, not a string. Fractions must sum to 1.0 ± 0.005. Worked example and atomic masses are provided in the prompt.
- `cas_number` guidance: the primary CAS is the canonical CAS for the free-acid/free-base neutral form — the lowest-numbered (earliest-registered) accession. Salt, hydrate, and polymorph CAS numbers must be listed as lower-confidence candidates.
- `secondary_accession_numbers` must not repeat the value in `drugbank_id`.
- Confidence guidance: deterministic descriptors from PubChem/DrugBank → 0.85–0.95; same from ChEMBL → 0.75–0.85; literature → 0.60–0.75; uncertain/estimated → 0.40–0.60. Confidence < 0.4 signals do not fabricate.
- Retry loop: up to 3 attempts on `HfHubHTTPError` (timeout / server error) with a 10-second wait. On `json.JSONDecodeError` or `ValueError`, temperature is nudged upward by `+0.1` per retry (capped at 1.0) to encourage more varied JSON formatting.
- `clean_json_output()`: Extracts JSON from the last ` ```json ``` ` code block. Falls back to the outermost `{...}` with a regex if no code fence is present.

### `fusion.py`

The deterministic fusion engine. Key internals:

- `fuse(extracted)`: Top-level recursive walker. Dispatches `dict` branches to recursion, `list` leaves to the typed fusion strategy for that field, and scalars to passthrough.
- `_effective_confidence(candidate)`: Blends declared confidence with source-priority score. Formula: `eff = conf × (1 + 0.1 × (priority − 1))`. Priority 5 (PubChem/DrugBank) yields ×1.4; priority 1 (other) yields ×1.0. The multiplier is deliberately small so a highly-confident literature value can still beat a low-confidence PubChem one.
- `_fuse_float(candidates, field_name)`: Confidence-weighted mean after plausibility filtering, confidence capping, hallucination detection, and outlier removal. Uses effective confidence for weighting.
- `_fuse_integer(candidates, field_name)`: Weighted mode — buckets candidates by integer value, returns the bucket with the highest total effective confidence. Never produces a fractional result.
- `_fuse_categorical(candidates, field_name)`: Weighted mode for string fields. Special `cas_number` rule: among tied top candidates, the smallest registry number (lexicographically on the integer prefix) wins.
- `_fuse_list_field(candidates, field_name)`: Deduplication union of string values above minimum confidence, ordered by descending effective confidence. Case-insensitive deduplication, original casing preserved.
- `_fuse_smiles(candidates)`: Returns the string value of the highest effective-confidence candidate. Structural validation is deferred to `main.py`.
- `_fuse_composition(candidates)`: Parses dict or legacy string values into per-element float candidate lists, fuses each element independently via `_fuse_float`, then renormalises the result to sum to 1.0.
- `_check_consensus_hallucination(candidates, field_name)`: Returns `True` only for fields not in `DETERMINISTIC_FIELDS` and not in `EXPERIMENTAL_FIELDS`, where all numeric candidates share the same value. Requires at least 2 values.
- `_deduplicate_secondary_accessions(fused)`: Post-fusion guard that removes any value in `secondary_accession_numbers` matching `drugbank_id` (case-insensitive).
- `MIN_CONFIDENCE = 0.4` is a module-level constant, separate from `fusion_config.py`, as it is a structural minimum rather than a tunable threshold.

### `fusion_config.py`

The single control file for all mathematical policy. Contains five dicts/sets:

```python
ABSOLUTE_THRESHOLD = {
    "molecular_weight":             50.0,
    "exact_mol_weight":             0.005,
    "alogp":                        0.5,
    "molecular_polar_surface_area": 10.0,
}

PLAUSIBILITY_FLOOR = {
    "number_of_heavy_atoms":        1,
    "num_rotatable_bonds":          0,
    "molecular_weight":             10.0,
    "exact_mol_weight":             10.0,
    "molecular_polar_surface_area": 0.0,
}

PLAUSIBILITY_CEILING = {
    "molecular_weight":             2000.0,
    "exact_mol_weight":             2000.0,
    "num_h_acceptors_lipinski":     20,
    "num_h_donors_lipinski":        10,
    "num_h_acceptors":              20,
    "num_h_donors":                 10,
    "num_rotatable_bonds":          50,
    "molecular_polar_surface_area": 500.0,
    "alogp":                        10.0,
    "number_of_heavy_atoms":        500,
}

CONFIDENCE_CAP = {
    "alogp":                        0.70,
    "molecular_weight":             0.80,
    "exact_mol_weight":             0.85,
    "molecular_polar_surface_area": 0.75,
    "number_of_heavy_atoms":        0.80,
    "num_h_acceptors_lipinski":     0.80,
    "num_h_donors_lipinski":        0.80,
    "num_h_acceptors":              0.80,
    "num_h_donors":                 0.80,
    "num_rotatable_bonds":          0.80,
    "net_formal_charge":            0.85,
}

EXPERIMENTAL_FIELDS: set = set()   # no experimental fields in current schema
```

Keys must exactly match leaf field names defined in `drug_descriptors.py`. This is the only file that needs editing when the schema gains new numeric fields requiring custom fusion behaviour.

### `LLM2.py`

The verification and JSON-building layer. Key behaviours:

- Uses `meta-llama/Llama-3.1-70B-Instruct` with `max_tokens=2000` and `temperature=0.1` for deterministic, conservative output.
- System prompt rules:
  - **Preserve numerics**: every numeric field in fused data that is not the schema default must be copied exactly — no rounding, zeroing, or silent modification.
  - **Fill missing**: if a numeric field IS the schema default (0.0 / 0), LLM2 may fill it from its knowledge if confident; otherwise leave as default.
  - **String/list fields**: fill missing fields from knowledge only if certain. A wrong DrugBank ID or CAS number is worse than an empty string.
  - **CAS number**: must not substitute an alternate CAS — the fusion layer has already selected the canonical primary accession.
  - **Duplicate accession guard**: `secondary_accession_numbers` must not contain the same value as `drugbank_id`.
  - **SMILES**: only fill if 100% certain of the exact canonical structure; empty string otherwise.
  - **Integer enforcement**: must output integers (no decimal point) for all integer fields.
  - **Plausibility correction**: may correct a value that grossly violates physical chemistry, but must record the change in a top-level `warnings` list. Silent changes are prohibited.
  - **Cross-checks**: (a) `number_of_heavy_atoms` — recount from `molecular_formula`; correct and warn if wrong. (b) `molecular_composition` — verify element fractions sum to 1.0 ± 0.005; recompute from formula if needed. (c) `exact_mol_weight` — must differ from `molecular_weight`; if identical, set to `null` and warn.
  - **Provenance**: adds a top-level `provenance` list documenting the origin of key fields (at minimum: `molecular_weight`, `molecular_formula`, `cas_number`, `smiles`), using source tags `pubchem`, `chembl`, `drugbank`, `llm_extracted`, `computed`, `verified`.
- `_clean_json_output()`: mirrors LLM1's extractor — last ` ```json ``` ` block or outermost `{...}`.
- Retry loop: up to 3 attempts. Both `HfHubHTTPError` and malformed JSON retries wait 10 seconds. After exhausting all retries, raises `RuntimeError`.

### `drug_descriptors.py`

The single source of truth for schema structure and fusion policy metadata. Exports four constants:

- **`REQUIRED_SCHEMA`**: the nested dict defining all output fields and their typed placeholder values.
- **`FIELD_TYPES`**: maps every leaf field name to its fusion strategy (`"integer"`, `"float"`, `"categorical"`, `"list"`, `"smiles"`, `"composition"`).
- **`SOURCE_PRIORITY`**: maps source type strings to integer priority scores (1–5). Also exports `source_priority(source_type)` — a lookup function that handles prefixed strings like `"PubChem_computed"`.
- **`DETERMINISTIC_FIELDS`**: a set of field names whose values derive from molecular structure rather than experimental measurement. Consensus hallucination detection is suppressed for these fields, and `main.py` will attempt to recompute them from `molecular_formula` when available.

---

## Guards & Safety Mechanisms

The pipeline implements several layers of protection against LLM unreliability:

| Guard | Location | What it prevents |
|---|---|---|
| Non-resolution contract | LLM1 system prompt | Averaging/conflict resolution before the math engine runs |
| Effective confidence gating (`MIN_CONFIDENCE = 0.4`) | fusion.py | Low-quality candidates polluting the fusion |
| Source-priority boosting | fusion.py | Literature/other sources dominating PubChem/DrugBank canonical values |
| Plausibility floor + ceiling | fusion.py + fusion_config.py | Physically impossible values entering the pool |
| Consensus hallucination detection | fusion.py | Zero-variance pools where the model fabricated agreement (skipped for deterministic fields) |
| CAS number canonical selection | fusion.py | Salt/polymorph CAS replacing the primary canonical CAS |
| Duplicate accession deduplication | fusion.py + main.py | `drugbank_id` appearing in `secondary_accession_numbers` |
| Numeric guard re-injection | main.py | LLM2 silently altering fused numeric values |
| String-field guard | main.py | LLM2 overwriting non-fillable string fields resolved by fusion |
| Composition dict guard | main.py | LLM2 losing or clearing the fused `molecular_composition` dict |
| SMILES validation + clearing | main.py | Invalid SMILES strings reaching the output |
| Deterministic recomputation | main.py | LLM noise in MW, exact mass, HAC, composition when formula is known |
| Formula ↔ MW consistency check | main.py | Formula/MW mismatch from LLM2; overwrites with formula-derived value |
| SMILES ↔ HAC consistency check | main.py | Heavy atom count inconsistent with the validated SMILES |
| Composition fraction check | main.py | Element fractions not summing to 1.0; renormalises in place |
| Integer type enforcement (`[INT-GUARD]`) | main.py | LLM2 producing floats for discrete integer fields (e.g. `3.2857` for `num_rotatable_bonds`) |
| Null sentinel overwrite | main.py | LLM2 replacing an honest `null` with a schema-default `0.0`/`0` |
| Schema key audit | main.py | LLM2 dropping required fields silently |
| LLM1 output normalisation | main.py | CID-wrapped output structure not matching schema keys |
| Temperature nudge on retry | LLM1.py | Identical malformed JSON on consecutive retries |
| Plausibility warnings | LLM2 system prompt + `warnings` list | Chemically suspicious values reaching the output without a record of the change |

---

## Fusion Configuration Reference

Edit `fusion_config.py` to change statistical behaviour without touching any algorithm code.

### `ABSOLUTE_THRESHOLD`

Fields listed here use absolute deviation from the median as the outlier criterion. Any candidate further than the threshold from the median is discarded.

```python
ABSOLUTE_THRESHOLD = {
    "molecular_weight":             50.0,   # candidates must be within 50 Da of median
    "exact_mol_weight":             0.005,  # monoisotopic mass; reputable sources agree within 0.005 Da
    "alogp":                        0.5,    # reputable sources agree within 0.5 log units
    "molecular_polar_surface_area": 10.0,   # reputable sources agree within 10 Å²
}
```

Fields **not** listed here fall back to the ratio-based exclusion (5× rule).

### `PLAUSIBILITY_FLOOR`

Values strictly below the floor are physically implausible and dropped before outlier removal runs.

```python
PLAUSIBILITY_FLOOR = {
    "number_of_heavy_atoms":        1,      # must have at least one heavy atom
    "num_rotatable_bonds":          0,      # zero is valid (rigid molecules)
    "molecular_weight":             10.0,   # below 10 Da is not a drug-like molecule
    "exact_mol_weight":             10.0,
    "molecular_polar_surface_area": 0.0,    # zero is valid
}
```

### `PLAUSIBILITY_CEILING`

Values strictly above the ceiling are physically implausible for drug-like molecules and dropped before outlier removal runs.

```python
PLAUSIBILITY_CEILING = {
    "molecular_weight":             2000.0,
    "exact_mol_weight":             2000.0,
    "num_h_acceptors_lipinski":     20,
    "num_h_donors_lipinski":        10,
    "num_h_acceptors":              20,
    "num_h_donors":                 10,
    "num_rotatable_bonds":          50,
    "molecular_polar_surface_area": 500.0,
    "alogp":                        10.0,
    "number_of_heavy_atoms":        500,
}
```

### `CONFIDENCE_CAP`

Caps the maximum effective confidence for fields where LLMs are structurally overconfident. Reduces over-weighting of hallucinated "certain" values for properties that are deterministic from structure.

```python
CONFIDENCE_CAP = {
    "alogp":                        0.70,  # computed property; no model should be 100% confident
    "molecular_weight":             0.80,  # usually correct but occasional formula errors
    "exact_mol_weight":             0.85,
    "molecular_polar_surface_area": 0.75,
    "number_of_heavy_atoms":        0.80,
    "num_h_acceptors_lipinski":     0.80,
    "num_h_donors_lipinski":        0.80,
    "num_h_acceptors":              0.80,
    "num_h_donors":                 0.80,
    "num_rotatable_bonds":          0.80,
    "net_formal_charge":            0.85,
}
```

---

## Output Format

`output.json` is written after every successful pipeline run. All float values are rounded to 4 decimal places. The `cid` field is injected by `main.py` directly (not derived from LLM output) to guarantee correctness. Provenance metadata from LLM2, if present, is attached as a top-level `provenance` key.

Sample output for CID 5754 (Ibuprofen):

```json
{
  "cid": 5754,
  "identity": {
    "drugbank_id": "DB01050",
    "secondary_accession_numbers": ["DBSALT001050"],
    "common_name": "Ibuprofen",
    "cas_number": "15687-27-1",
    "unii": "WK2XYI10QM",
    "synonyms": ["Advil", "Motrin", "2-(4-isobutylphenyl)propanoic acid"],
    "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
  },
  "molecule": {
    "number_of_heavy_atoms": 13,
    "molecular_formula": "C13H18O2",
    "molecular_composition": {"C": 0.7771, "H": 0.0902, "O": 0.1591},
    "molecular_weight": 206.2808,
    "exact_mol_weight": 206.1307,
    "net_formal_charge": 0,
    "alogp": 3.97,
    "num_h_acceptors_lipinski": 2,
    "num_h_donors_lipinski": 1,
    "num_rotatable_bonds": 4,
    "molecular_polar_surface_area": 37.3,
    "num_h_acceptors": 2,
    "num_h_donors": 1
  },
  "provenance": [
    {"field": "molecular_weight", "source": "computed", "notes": "recomputed from C13H18O2"},
    {"field": "molecular_formula", "source": "pubchem"},
    {"field": "cas_number", "source": "drugbank"},
    {"field": "smiles", "source": "pubchem"}
  ]
}
```

Fields set to `null` in the output indicate that fusion was unable to resolve a plausible value from the extracted candidates — this is an intentional honest signal, not a bug.

---

## Quickstart

### Prerequisites

- Python 3.10 or higher
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

The `.gitignore` is already configured to exclude `.env`, `output.json`, and `__pycache__/` from version control.

### Running the Pipeline

```bash
python main.py
```

You will be prompted for two inputs:

```
Enter molecule name OR CID: Ibuprofen
Debug mode? (y/n): y
```

**Accepted input formats:**

| Format | Example |
|---|---|
| Common name | `Ibuprofen` |
| IUPAC name | `2-(4-isobutylphenyl)propanoic acid` |
| PubChem CID (integer string) | `5754` |

### Console Output During Execution

```
[CID] Resolving PubChem CID for: Ibuprofen
[CID] Resolved to CID 5754
[CID] Resolved to: Ibuprofen (IUPAC: 2-[4-(2-methylpropyl)phenyl]propanoic acid)

[LLM1] Extracting properties for: Ibuprofen (IUPAC: ...)
[LLM1] Extraction complete.

[DEBUG] LLM1 raw extraction:
{ ... multi-candidate JSON ... }

[FUSION] Resolving candidates with typed fusion strategies...
[FUSION] Fusion complete.

[DEBUG] Fused intermediate:
{ ... single-value JSON ... }

[LLM2] Verifying plausibility and assembling final record...
[LLM2 WARN] exact_mol_weight was identical to molecular_weight — set to null
[RECOMPUTE] Recomputing deterministic descriptors from formula: C13H18O2
[RECOMPUTE] molecular_weight → 206.2808
[RECOMPUTE] exact_mol_weight → 206.1307
[GUARD] LLM2 changed molecule.alogp from 3.97 → 3.9 — restoring fused value.
[PROV] Provenance metadata attached (4 entries).
[LLM2] Verification complete.

[OUTPUT]
{ ... validated JSON ... }

Record saved to output.json
```

The `[GUARD]`, `[WARN]`, `[INFO]`, `[INT-GUARD]`, `[RECOMPUTE]`, `[CHEM-CHECK]`, `[COMP]`, `[SMILES]`, `[PROV]`, and `[FUSION] WARNING` prefixes in console output all indicate specific pipeline events worth inspecting when debugging an unusual result.

---

## Dependencies

| Package | Purpose |
|---|---|
| `huggingface_hub` | Inference API client; `InferenceClient.chat_completion()` for both LLM calls; `HfHubHTTPError` for retry handling |
| `python-dotenv` | Loads `HF_TOKEN` from `.env` without it being hardcoded |
| `requests` | PubChem REST API calls for CID resolution and name enrichment |
| `statistics` | `statistics.median()` used in the fusion engine for outlier filtering |
| `collections` | `defaultdict` used in fusion for bucketing and composition element accumulation |
| `json`, `re`, `math`, `time`, `os` | Standard library — JSON parsing, regex extraction, float comparison (`math.isclose`), retry sleep, environment access |

Install all third-party dependencies with:

```bash
pip install huggingface_hub python-dotenv requests
```

---

## Design Decisions & Tradeoffs

**Why the 70B model instead of the 8B?**
The pipeline uses `meta-llama/Llama-3.1-70B-Instruct` for both LLM1 and LLM2. The 70B model produces substantially more reliable structured JSON, fewer unit errors, and fewer outright hallucinations on chemical property extraction tasks. The schema is intentionally compact (no interaction profile, no pharmacological activity fields) partly to stay within Hugging Face inference tier token budgets even at this model size.

**Why type-aware fusion instead of a single weighted average?**
A confidence-weighted mean applied to an integer field like `num_rotatable_bonds` would produce `3.2857` — a value that does not correspond to any physical reality. Integer fields use weighted mode instead: the most-agreed-upon integer value wins. This preserves the discrete nature of counts and avoids rounding errors propagating through downstream calculations.

**Why not use LLM2 to also do the numeric fusion?**
Numeric fusion by an LLM is inherently probabilistic. Two identical prompts with the same candidates can produce different weighted averages because temperature > 0. The deterministic Python engine guarantees reproducible results — running the same extraction through the fusion layer twice always produces the same fused value.

**Why re-inject fused numerics after LLM2?**
LLM2 is instructed not to change numeric values, but it is a language model — instruction following is not guaranteed, especially at low temperature where the model may "correct" a value it perceives as surprising. The guard in `main.py` is a hard enforcement layer, not a trust-but-verify check.

**Why recompute deterministic descriptors from formula?**
Fields like `molecular_weight`, `exact_mol_weight`, and `number_of_heavy_atoms` are structural — they can be computed exactly from `molecular_formula`. When the formula is known, LLM-extracted values for these fields are inherently noisier than the direct calculation. The pipeline uses the formula as the authoritative source, with LLM extraction as a fallback when the formula is unavailable.

**Why skip consensus hallucination detection for deterministic fields?**
For a field like `molecular_weight`, PubChem, DrugBank, and ChEMBL will all report the same value — because it is computed from the same molecular structure. Flagging this as a hallucination would be a false positive. The `DETERMINISTIC_FIELDS` set in `drug_descriptors.py` explicitly excludes these fields from the hallucination check. Only fields where independent measurements could legitimately disagree (e.g. experimental assay values) are candidates for the hallucination detector.

**Why does the schema use `0`, `0.0`, `""`, `[]`, `{}` as placeholder values?**
The types of the leaf values in `REQUIRED_SCHEMA` are read at runtime by `_get_numeric_paths()`, `_get_string_paths()`, and the integer type enforcement pass to discover which paths need which treatment. Using typed placeholders makes this introspection trivially reliable without needing a separate type annotation structure.

---

## Extending the Schema

To add a new field (e.g. `tpsa` — topological polar surface area):

1. **`drug_descriptors.py`** — Add the field in `REQUIRED_SCHEMA` at the appropriate nesting level, add its type to `FIELD_TYPES`, and (if it is a deterministic structural descriptor) add it to `DETERMINISTIC_FIELDS`:

   ```python
   # In REQUIRED_SCHEMA:
   "molecule": {
       ...
       "tpsa": 0.0,        # ← new field
   }

   # In FIELD_TYPES:
   "tpsa": "float",        # ← new entry

   # In DETERMINISTIC_FIELDS (if applicable):
   DETERMINISTIC_FIELDS = {
       ...,
       "tpsa",             # ← can be computed from SMILES
   }
   ```

2. **`fusion_config.py`** — Add thresholds if `tpsa` needs custom fusion behaviour:

   ```python
   ABSOLUTE_THRESHOLD = {
       ...,
       "tpsa": 10.0,       # reputable sources agree within 10 Å²
   }

   PLAUSIBILITY_CEILING = {
       ...,
       "tpsa": 500.0,
   }

   CONFIDENCE_CAP = {
       ...,
       "tpsa": 0.75,
   }
   ```

That is the entirety of the change needed. `main.py`, `LLM1.py`, `LLM2.py`, and `fusion.py` all derive their behaviour from the schema and config at runtime.

---

## Known Limitations

- **Model hallucination**: Even `meta-llama/Llama-3.1-70B-Instruct` will fabricate plausible-sounding but incorrect values for obscure molecules with limited literature presence, or for fields like `drugbank_id` and `unii` that require exact registry knowledge. The confidence gating, plausibility windows, and deterministic recomputation mitigate but cannot eliminate this.
- **LLM1 integer compliance**: Despite explicit prompt instructions and worked examples, the model occasionally emits float values for integer fields (e.g. `3.2857` for `num_rotatable_bonds`). The `[INT-GUARD]` pass in `main.py` acts as a last-resort catch for this.
- **Molecular composition accuracy**: Composition fractions are extracted by LLM1 and fused element-by-element. When `molecular_formula` is known, `main.py` recomputes composition deterministically — but if the formula is itself wrong or missing, composition may be inaccurate.
- **SMILES validation without RDKit**: The SMILES validator in `main.py` performs a structural sanity check (legal characters, balanced brackets, known atom symbols) but cannot detect chemically invalid SMILES that pass the syntactic check. Full validation would require RDKit or a similar cheminformatics library.
- **Single-candidate pools**: If only one candidate passes the confidence filter, the outlier removal step has no peers to compare against. The single value is accepted without filtering. This is intentional — there is no basis for rejection with n=1.
- **Free-tier rate limits**: The Hugging Face free tier imposes request quotas. The retry logic (3 attempts, 10-second wait) handles transient 429/503 errors but sustained rate-limiting will cause the pipeline to fail after exhausting retries.
- **PubChem CID fallback**: If the name lookup fails, `cid` is set to `0`. This is a valid sentinel value, not a PubChem record. Downstream consumers of `output.json` should treat `cid == 0` as unresolved. Unlike previous versions, there is no SMILES-based fallback lookup.
- **`secondary_accession_numbers` string field**: This list field is not numerically fused or guard-protected beyond deduplication. LLM2 may add, remove, or reorder entries; only the deduplication guard (removing entries matching `drugbank_id`) is enforced programmatically.