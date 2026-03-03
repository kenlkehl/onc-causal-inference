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


# =============================================================================
# Two-Stage Generation: Event Timeline Prompt
# =============================================================================

EVENT_TIMELINE_PROMPT = """Generate a realistic longitudinal clinical history as a sequence of events for a cancer patient with the following characteristics:

Patient Characteristics (YOU MUST ACCURATELY REPRESENT THESE VALUES):
{patient_characteristics}

Clinical Context: {clinical_question}

Generate a list of {min_events}-{max_events} events that might have occurred along this patient's disease trajectory.
Events should be chronological and cover the patient's cancer journey from diagnosis through the treatment decision point.

Types of events (tag each event with its type at the start of the line):
- <demographics> Patient demographics (name, gender)
- <diagnosis> Cancer diagnosis with TNM stage, histology, site, biomarkers
- <systemic> Initiation of a systemic therapy
- <surgery> Surgical procedure
- <radiation> Radiation treatment
- <adverse_event> Treatment-related adverse event
- <clinical_note> Oncologist progress note (indicate disease status: responding/progressing/stable)
- <imaging_report> Single imaging study (specify type: CT/MRI/PET; indicate findings and disease status)
- <pathology_report> Pathology report (anatomic pathology, molecular testing, etc.)
- <ngs_report> Next-generation sequencing report (diagnosis, specimen site, detailed genomic findings)

Rules:
1. Each event is one line of text, tagged with its event type
2. Events should reference the patient's age at each timepoint
3. CRITICAL: FAITHFULLY represent ALL patient characteristics listed above in the events
4. CRITICAL: Do NOT mention which specific treatment the patient was ultimately assigned to for the clinical question. The history should end BEFORE the treatment decision.
5. Express all characteristics naturally in clinical language (no underscores or codes)
6. Include realistic lab values, imaging findings, and clinical details consistent with the patient profile
7. Imaging reports must specify study type, presence/absence of cancer, response status if applicable, and metastatic sites
8. Clinical notes must indicate disease status (responding, progressing, stable, or no evidence of disease)
9. The disease trajectory should be clinically plausible for the described patient profile
10. Diagnosis events should include TNM stage, summary stage, site description, histology, and all relevant biomarkers
11. NGS reports should be very detailed: include actionable alterations, co-mutations, fusions, and copy number alterations. Genomic findings must be consistent with known mutation/co-mutation patterns (e.g., EGFR mutant lung cancers almost never have KRAS co-mutations)
12. To ensure diversity, vary patient names and genders

Example format (hypothetical, just to illustrate formatting):
<demographics> The patient is a 62-year-old female named Maria Santos.
<diagnosis> At age 62, the patient was diagnosed with stage IIIB (T4N2M0) non-small cell lung cancer, adenocarcinoma histology, of the right upper lobe. EGFR exon 19 deletion positive, PD-L1 TPS 30%.
<pathology_report> At age 62, CT-guided core biopsy of right upper lobe mass showed adenocarcinoma, TTF-1 positive, CK7 positive.
<imaging_report> At age 62, CT chest/abdomen/pelvis showed 4.2cm right upper lobe mass with right hilar and mediastinal lymphadenopathy. No distant metastases.
<clinical_note> At age 62, initial oncology consultation. Cancer present, newly diagnosed.
<ngs_report> At age 62 years, the patient had next generation sequencing performed for lung adenocarcinoma based on a specimen obtained from the right upper lobe, which showed an EGFR exon 19 deletion (p.E746_A750del), a TP53 p.R248W mutation, and an RB1 loss.
<systemic> At age 62, started concurrent chemoradiation with carboplatin/paclitaxel.
<imaging_report> At age 63, CT chest showed partial response with decrease in primary mass to 2.1cm. Lymphadenopathy improved.
<clinical_note> At age 63, follow-up visit. Cancer present, responding to therapy.

Generate the event timeline now:"""


# =============================================================================
# Two-Stage Generation: Note Expansion Prompt
# =============================================================================

NOTE_EXPANSION_PROMPT = """You are a brilliant synthetic clinical document generation bot with encyclopedic knowledge about cancer and its treatment.

Below is a chronological list of events from a patient's clinical history. One event is surrounded by <BEGIN EVENT CORRESPONDING TO SYNTHETIC NOTE> and <END EVENT CORRESPONDING TO SYNTHETIC NOTE> tags.

Generate a detailed, realistic clinical document corresponding to that tagged event.

Rules:
1. The document should be a {note_type} as indicated by the tagged event
2. Incorporate everything you know about the patient's history and cancer generally
3. Don't directly incorporate information about future events as if they have already occurred, but you can use your knowledge of the future to inform what the document might have contained at the time it was written
4. CRITICAL: Ignore your knowledge of today's date. Do not add dates to the synthetic document. These will be added later programmatically.
5. Do NOT mention which specific treatment the patient will receive for: {clinical_question}
6. The document should read as if it were a real clinical document
7. CRITICAL: Do not invent treatments that are not included in the list of events
8. Do not include any disclaimers or notes about the fact that the document is synthetic

Note type guidelines:
- **Pathology reports** (~1 page): Specimen ID, date of procedure, type of specimen, diagnostic findings, any ancillary studies (IHC, molecular), and a description of gross pathology if relevant. Pathology reports should NOT include recommendations about management, since these are not part of real pathology reports.
- **Imaging reports** (~1 page): Scan type, Findings (broken down by organs imaged by the study), and Impression. Imaging reports should NOT make treatment or monitoring recommendations.
- **Clinical progress notes** (~2 pages): Chief complaint, history of present illness, review of systems, physical exam, lab results, imaging results, and assessment/plan. If it is the first clinical note, it is a consult note and should also include past medical history, social history, family history, allergies, and medications. Clinical notes should use a realistic mix of common brand names (e.g., Herceptin, Keytruda, Taxol) and generic drug names (e.g., trastuzumab, pembrolizumab, paclitaxel), and sometimes abbreviations (e.g., pembro, cape). Within clinical notes, sometimes patients should have adverse events of therapy and/or comorbidities described that are consistent with their clinical trajectories.
- **NGS reports** (~1-2 pages): Specimen site, diagnosis, testing methodology, and detailed genomic findings including actionable alterations, co-mutations, fusions, and copy number alterations. Most reports should describe alterations in many genes, even though only some will be clinically relevant. If you do not have information about key biomarkers explicitly provided, you should imagine what they might be based on cancer type, history, and prior treatments. However, these must be consistent with realistic biological patterns (e.g., EGFR mutant lung cancers almost never have concomitant driver mutations in KRAS, BRAF, etc.).

Here is the list of events:
{masked_event_timeline}

Now, generate the synthetic document corresponding to the notated event."""


# =============================================================================
# Drug Perturbation Map (generic -> brand/abbreviation alternatives)
# =============================================================================

DRUG_PERTURBATION_MAP = {
    # PD-(L)1 & CTLA-4
    "pembrolizumab": ["Keytruda", "pembro"],
    "nivolumab": ["Opdivo", "nivo"],
    "ipilimumab": ["Yervoy", "ipi"],
    "atezolizumab": ["Tecentriq", "atezo"],
    "durvalumab": ["Imfinzi", "durva"],
    "cemiplimab": ["Libtayo", "cemi"],
    # Platinums & taxanes
    "carboplatin": ["Paraplatin", "carbo"],
    "cisplatin": ["Platinol", "cis"],
    "oxaliplatin": ["Eloxatin", "oxali"],
    "paclitaxel": ["Taxol", "pacli", "PTX"],
    "docetaxel": ["Taxotere", "doce"],
    "nab-paclitaxel": ["Abraxane", "nab-pac"],
    # Antimetabolites
    "capecitabine": ["Xeloda", "cape"],
    "fluorouracil": ["5-FU", "Adrucil", "5FU"],
    "5-fluorouracil": ["5-FU", "Adrucil", "5FU"],
    "gemcitabine": ["Gemzar", "gem"],
    "pemetrexed": ["Alimta", "peme", "pem"],
    "methotrexate": ["MTX", "Trexall"],
    # Anthracyclines & others
    "doxorubicin": ["Adriamycin", "doxo"],
    "epirubicin": ["Ellence", "epi"],
    "cyclophosphamide": ["Cytoxan", "CTX", "cyclo"],
    "etoposide": ["VP-16", "eto"],
    "irinotecan": ["Camptosar", "iri"],
    "topotecan": ["Hycamtin", "topo"],
    # HER2 axis
    "trastuzumab": ["Herceptin", "trast"],
    "pertuzumab": ["Perjeta", "pertu"],
    "ado-trastuzumab emtansine": ["Kadcyla", "T-DM1"],
    "trastuzumab emtansine": ["Kadcyla", "T-DM1"],
    "trastuzumab deruxtecan": ["Enhertu", "T-DXd"],
    "tucatinib": ["Tukysa", "tuca"],
    "lapatinib": ["Tykerb", "lapa"],
    # CDK4/6
    "palbociclib": ["Ibrance", "palbo"],
    "ribociclib": ["Kisqali", "ribo"],
    "abemaciclib": ["Verzenio", "abema"],
    # PARP
    "olaparib": ["Lynparza", "ola"],
    "niraparib": ["Zejula", "nira"],
    "rucaparib": ["Rubraca", "ruca"],
    "talazoparib": ["Talzenna", "tala"],
    # VEGF axis
    "bevacizumab": ["Avastin", "bev"],
    "ramucirumab": ["Cyramza", "ramu"],
    # EGFR, ALK, etc.
    "osimertinib": ["Tagrisso", "osi"],
    "erlotinib": ["Tarceva", "erlo"],
    "gefitinib": ["Iressa", "gefi"],
    "afatinib": ["Gilotrif", "afat"],
    "dacomitinib": ["Vizimpro", "daco"],
    "alectinib": ["Alecensa", "alec"],
    "ceritinib": ["Zykadia", "ceri"],
    "crizotinib": ["Xalkori", "crizo"],
    "lorlatinib": ["Lorbrena", "lorla"],
    # BRAF/MEK
    "dabrafenib": ["Tafinlar", "dabra"],
    "trametinib": ["Mekinist", "tram"],
    "vemurafenib": ["Zelboraf", "vem"],
    "encorafenib": ["Braftovi", "enco"],
    "binimetinib": ["Mektovi", "bini"],
    # Multi-TKIs
    "lenvatinib": ["Lenvima", "lenva"],
    "sorafenib": ["Nexavar", "sora"],
    "regorafenib": ["Stivarga", "rego"],
    "pazopanib": ["Votrient", "pazo"],
    "sunitinib": ["Sutent", "suni"],
    # Antibodies (other)
    "rituximab": ["Rituxan", "ritux"],
    "cetuximab": ["Erbitux", "cetux"],
    "panitumumab": ["Vectibix", "pani"],
    # GU agents
    "enzalutamide": ["Xtandi", "enza"],
    "abiraterone": ["Zytiga", "abi"],
    "apalutamide": ["Erleada", "apa"],
    "leuprolide": ["Lupron", "leup"],
    "degarelix": ["Firmagon", "dega"],
    "relugolix": ["Orgovyx", "relu"],
    # Endocrine
    "letrozole": ["Femara"],
    "anastrozole": ["Arimidex"],
    "exemestane": ["Aromasin"],
    "tamoxifen": ["Nolvadex", "tam"],
    "fulvestrant": ["Faslodex"],
    # mTOR, alkylators, myeloma, etc.
    "everolimus": ["Afinitor", "evero"],
    "sirolimus": ["Rapamune", "siro"],
    "temozolomide": ["Temodar", "TMZ"],
    "bortezomib": ["Velcade", "bortez"],
    "carfilzomib": ["Kyprolis", "carfil"],
    "ixazomib": ["Ninlaro", "ixa"],
    "daratumumab": ["Darzalex", "dara"],
    "obinutuzumab": ["Gazyva", "obinu"],
}


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
