# synthetic_data/prompts.py
"""LLM prompt templates for synthetic data generation."""

# System prompt for clinical expertise
CLINICAL_SYSTEM_PROMPT = """You are an expert clinical researcher and oncologist with deep expertise in:
- Comparative effectiveness research
- Clinical trial design
- Real-world evidence generation
- Causal inference methodology

You provide precise, structured responses that can be parsed programmatically.
Always respond with valid JSON when requested."""


# =============================================================================
# Confounder Generation Prompt
# =============================================================================

CONFOUNDER_GENERATION_PROMPT = """Given the following comparative effectiveness research question:

"{clinical_question}"

Generate a comprehensive list of realistic confounding variables that would influence both treatment assignment and outcome in a real-world clinical setting.

Requirements:
1. Include {num_confounders_instruction} total
2. Mix of categorical (3-5 categories each) and continuous variables
3. Common confounders might include: age, sex (if applicable to the cancer type), performance status, comorbidities, prior treatments, biomarkers, disease stage, etc.
4. Be specific to the clinical context of the question

Respond with a JSON object in this exact format:
{{
  "confounders": [
    {{
      "name": "age",
      "type": "continuous",
      "description": "Patient age in years"
    }},
    {{
      "name": "ecog_performance_status",
      "type": "categorical",
      "categories": ["0", "1", "2", "3"],
      "description": "ECOG performance status score"
    }},
    ...
  ]
}}"""


# =============================================================================
# Regression Equation Generation Prompt
# =============================================================================

REGRESSION_EQUATION_PROMPT = """Given the following confounding variables for a comparative effectiveness study:

Clinical Question: "{clinical_question}"

Confounders:
{confounder_list}

Generate two plausible regression equations for a simulation.

CRITICAL: You must ONLY use the confounders listed above. Do NOT invent or add any variables that are not in the confounder list. Every coefficient name must correspond to a confounder from the list above.

1. TREATMENT ASSIGNMENT equation: Predicts logit(P(treatment=1)) based on confounders
   - Should reflect realistic clinical decision-making
   - Some confounders should have stronger effects than others
   - Include interaction terms ONLY if there are 2+ confounders (use pairs from the list above)

2. OUTCOME equation: Predicts logit(P(outcome=1)) based on confounders AND treatment
   - The treatment coefficient is FIXED at {treatment_coefficient} (do not include treatment in your coefficients)
   - Should reflect known prognostic factors
   - Include interaction terms ONLY if there are 2+ confounders
   - Some confounders may affect outcome differently than treatment assignment

For continuous variables, coefficients represent effect per 1 SD increase.
For categorical variables, coefficients are relative to the reference category (first listed).

Respond with JSON in this exact format (example shows structure only - use YOUR confounders from above):
{{
  "treatment_equation": {{
    "intercept": -0.5,
    "coefficients": {{
      "<confounder_name>": 0.3,
      "<categorical_confounder>_<category2>": -0.2,
      "<categorical_confounder>_<category3>": -0.5
    }},
    "interactions": [
      {{
        "terms": ["<confounder1>", "<confounder2>"],
        "coefficient": 0.1
      }}
    ]
  }},
  "outcome_equation": {{
    "intercept": -1.0,
    "coefficients": {{
      "<confounder_name>": -0.2,
      "<categorical_confounder>_<category2>": 0.3,
      "<categorical_confounder>_<category3>": 0.6
    }},
    "interactions": [
      {{
        "terms": ["<confounder1>", "<confounder2>"],
        "coefficient": -0.15
      }}
    ]
  }}
}}

IMPORTANT RULES:
- For categorical variables with N categories, create N-1 dummy coefficients (excluding the reference/first category)
- Dummy variable names must be: variablename_categoryvalue (e.g., "ecog_status_2" for category "2" of "ecog_status")
- If there is only 1 confounder, the "interactions" arrays should be empty []
- Do NOT include any variables that are not in the confounder list above"""


# =============================================================================
# Summary Statistics Generation Prompt
# =============================================================================

SUMMARY_STATISTICS_PROMPT = """Given the following confounding variables for a study on:

"{clinical_question}"

Confounders:
{confounder_list}

Generate realistic summary statistics that would be observed in a real-world patient population.

For each confounder, provide:
- Categorical variables: proportion of patients in each category (must sum to 1.0)
- Continuous variables: mean and standard deviation

Base these on realistic clinical populations. For example:
- Age distributions typical for the cancer type
- Performance status distributions reflecting real-world (sicker than trials)
- Comorbidity rates appropriate for the demographic

Respond with JSON in this exact format:
{{
  "summary_statistics": {{
    "age": {{
      "type": "continuous",
      "mean": 65.0,
      "std": 12.0
    }},
    "ecog_performance_status": {{
      "type": "categorical",
      "proportions": {{
        "0": 0.25,
        "1": 0.45,
        "2": 0.20,
        "3": 0.10
      }}
    }},
    ...
  }}
}}"""


# =============================================================================
# Patient History Generation Prompt  
# =============================================================================

PATIENT_HISTORY_PROMPT = """You are generating a realistic synthetic clinical history document for a cancer patient.
This document should simulate concatenated clinical notes, radiology reports, and pathology reports.

Patient Characteristics (YOU MUST ACCURATELY REPRESENT THESE VALUES):
{patient_characteristics}

Clinical Context: {clinical_question}

Generate a comprehensive clinical history document that:
1. Simulates at least 5 clinical documents, concatenated together
2. Includes sections from different note types:
   - Initial oncology consultation note
   - At least one radiology report (CT, PET, or MRI)
   - At least one pathology report
   - One or more follow-up notes
3. **CRITICAL: ACCURATELY incorporates ALL the patient characteristics listed above**
4. Uses realistic medical terminology and abbreviations
5. Includes dates (use relative dates like "3 months prior", "at diagnosis")
6. Contains typical clinical details like vital signs, lab values, physical exam findings
7. Reflects the clinical decision-making around treatment selection

CRITICAL REQUIREMENT #1: Do NOT mention which specific treatment the patient was ultimately assigned to or received.
The clinical history should describe the patient's condition, characteristics, and the treatment decision-making process,
but should END BEFORE the actual treatment is selected or administered. This is essential because the document
will be used for causal inference and the treatment assignment must be predicted from the confounders, not read from the text.

CRITICAL REQUIREMENT #2: FAITHFULLY REPRESENT CONFOUNDER VALUES IN NATURAL CLINICAL LANGUAGE.
- If the patient has 2 metastatic sites, describe EXACTLY 2 specific sites (e.g., "bone and liver metastases")
- If the patient's age is 58 years, state the patient is 58 years old
- If ECOG performance status is 1, describe functional status consistent with ECOG 1
- For numeric variables, the count in the text MUST MATCH the specified value
- Do NOT embellish, add extra sites/conditions, or change any values

CRITICAL REQUIREMENT #3: EXPRESS CATEGORICAL VARIABLES NATURALLY - NO UNDERSCORES OR CODES.
- NEVER insert literal category codes with underscores (e.g., do NOT write "completed_more_than_12_months_ago")
- Instead, express the concept in natural clinical prose
- Examples of CORRECT natural expression:
  * "completed more than 12 months ago" → "completed adjuvant letrozole approximately 18 months prior to presentation"
  * "completed 12 months or less ago" → "recently finished adjuvant anastrozole 8 months ago"
  * "none" → "no prior adjuvant endocrine therapy" or "patient declined adjuvant hormonal treatment"
  * "high" → "elevated" or describe the specific finding
- The semantic meaning must be preserved, but expressed as a clinician would naturally write it

The document should read as if it captures the patient's state at the moment of treatment decision, before the treatment is revealed.

Write the document as if it were real clinical notes, imaging reports, and pathology reports that have been concatenated together.
The confounder values should appear naturally in the clinical narrative (e.g., "58-year-old female", "ECOG PS 1", "metastatic disease involving bone and liver").

Begin the clinical history document now:"""


def format_confounder_list(confounders: list) -> str:
    """Format confounders into a readable list for prompts."""
    lines = []
    for c in confounders:
        if c["type"] == "categorical":
            cats = ", ".join(c["categories"])
            lines.append(f"- {c['name']} (categorical): {c['description']}. Categories: [{cats}]")
        else:
            lines.append(f"- {c['name']} (continuous): {c['description']}")
    return "\n".join(lines)


def validate_clinical_text(text: str, characteristics: dict, confounders: list) -> dict:
    """
    Validate that generated clinical text doesn't contain literal category codes.

    Checks for:
    1. Underscore-connected phrases that look like category codes
    2. Exact category values that should have been naturalized

    Args:
        text: Generated clinical text
        characteristics: Patient characteristics dict
        confounders: List of confounder definitions

    Returns:
        Dict with 'valid' bool and 'issues' list of problems found
    """
    import re

    issues = []

    # Check for underscore-connected multi-word phrases (likely category codes)
    underscore_patterns = re.findall(r'\b[a-z]+(?:_[a-z0-9]+){2,}\b', text.lower())
    if underscore_patterns:
        issues.append(f"Found underscore-connected phrases (likely category codes): {underscore_patterns[:5]}")

    # Check for exact category values from categorical confounders
    for conf in confounders:
        if conf.get("type") == "categorical":
            name = conf["name"]
            value = characteristics.get(name, "")
            if isinstance(value, str) and "_" in value:
                # This is a multi-word category code - it should NOT appear verbatim
                if value.lower() in text.lower():
                    issues.append(f"Found literal category code '{value}' for confounder '{name}'")

    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


def format_patient_characteristics(characteristics: dict, confounders: list) -> str:
    """Format patient characteristics into readable text for history generation.

    Values are formatted with emphasis to ensure the LLM faithfully represents them
    while expressing categorical values in natural clinical language.
    """
    lines = []
    confounder_map = {c["name"]: c for c in confounders}

    for name, value in characteristics.items():
        conf = confounder_map.get(name, {})
        desc = conf.get("description", name.replace("_", " ").title())

        if conf.get("type") == "continuous":
            # Format continuous with units if known
            if "age" in name.lower():
                lines.append(f"- {desc}: {value:.0f} years old (use this exact age)")
            elif "metastatic" in name.lower() or "sites" in name.lower():
                # Metastatic sites should be an integer count
                int_value = int(round(value))
                lines.append(f"- {desc}: {int_value} site(s) (use this exact count - do not add or remove any)")
            else:
                lines.append(f"- {desc}: {value:.2f}")
        else:
            # For categorical variables, provide the semantic meaning to express naturally
            # Convert underscored category codes to natural language guidance
            natural_value = value.replace("_", " ")
            lines.append(f"- {desc}: \"{natural_value}\" (express this concept naturally in clinical language - do NOT use underscores or the literal category code)")

    return "\n".join(lines)
