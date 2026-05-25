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
# Explicit Feature Generation Prompt
# =============================================================================

CONFOUNDER_GENERATION_PROMPT = """Given the following comparative effectiveness research question:

"{clinical_question}"

Generate a comprehensive list of realistic patient-level variables for causal inference.
Each variable must be tagged with one or both causal roles:
- "confounder": plausibly influences treatment assignment and baseline outcome risk
- "effect_modifier": plausibly modifies the treatment effect

Requirements:
1. Include {num_confounders_instruction} total
2. Mix of categorical (3-5 categories each) and continuous variables
3. Common variables might include: age, sex (if applicable to the cancer type), performance status, comorbidities, prior treatments, biomarkers, disease stage, etc.
4. Be specific to the clinical context of the question
5. Some variables may have both roles; assign roles based on clinical plausibility

Respond with a JSON object in this exact format:
{{
  "features": [
    {{
      "name": "age",
      "type": "continuous",
      "description": "Patient age in years",
      "roles": ["confounder"]
    }},
    {{
      "name": "ecog_performance_status",
      "type": "categorical",
      "categories": ["0", "1", "2", "3"],
      "description": "ECOG performance status score",
      "roles": ["confounder", "effect_modifier"]
    }},
    ...
  ]
}}"""


# =============================================================================
# Regression Equation Generation Prompt
# =============================================================================

REGRESSION_EQUATION_PROMPT = """Given the following role-tagged variables for a comparative effectiveness study:

Clinical Question: "{clinical_question}"

Variables:
{confounder_list}

Generate two plausible regression equations for a simulation.

CRITICAL: You must ONLY use the variables listed above. Do NOT invent or add variables that are not in the list. Every coefficient name must correspond to a listed variable.

1. TREATMENT ASSIGNMENT equation: Predicts logit(P(treatment=1)) based on variables with the "confounder" role
   - Should reflect realistic clinical decision-making
   - Some confounder-role variables should have stronger effects than others
   - Include interaction terms ONLY if there are 2+ confounder-role variables

2. OUTCOME equation: Predicts logit(P(outcome=1)) based on confounder-role variables AND treatment
   - The treatment coefficient is FIXED at {treatment_coefficient} (do not include treatment in your coefficients)
   - Should reflect known prognostic factors
   - Include interaction terms ONLY if there are 2+ confounder-role variables
   - Treatment-effect interactions will be generated from effect-modifier-role variables

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
- If there is only 1 eligible confounder-role variable, the "interactions" arrays should be empty []
- Do NOT include any variables that are not in the variable list above"""


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


# =============================================================================
# Structured Data: Reference Schemas for Event Timeline
# =============================================================================

STRUCTURED_DATA_EVENT_TYPES = """
- <encounter> Outpatient or ED encounter with ICD-10 diagnosis codes and CPT/HCPCS procedure codes
- <hospitalization> Hospital admission with principal diagnosis ICD-10 code, length of stay, and discharge disposition
- <lab_result> Laboratory panel results with component names, numeric values, units, and normal/abnormal flags
- <pro_assessment> Patient-reported outcome questionnaire results with subscale scores"""

STRUCTURED_DATA_REFERENCE = """
REFERENCE SCHEMAS FOR STRUCTURED DATA EVENTS:

Common Oncology ICD-10 Diagnosis Codes:
  Cancer sites: C34.90 (lung NOS), C50.919 (breast NOS), C18.9 (colon NOS), C25.9 (pancreas NOS),
    C61 (prostate), C56.9 (ovary), C64.9 (kidney), C67.9 (bladder), C71.9 (brain),
    C43.9 (melanoma NOS), C90.00 (multiple myeloma), C73 (thyroid)
  Treatment encounters: Z51.11 (chemotherapy), Z51.0 (radiation therapy), Z51.12 (immunotherapy)
  Complications: D70.1 (chemotherapy-induced neutropenia), D64.81 (anemia of neoplastic disease),
    N17.9 (acute kidney injury), R50.81 (fever from condition), K52.0 (radiation enteritis),
    J18.9 (pneumonia NOS), R11.2 (nausea with vomiting), G62.0 (drug-induced polyneuropathy),
    I26.99 (pulmonary embolism), I82.409 (DVT of unspecified deep vessels)
  Surveillance: Z08 (encounter for follow-up after cancer treatment), Z85.x (personal history of cancer)

Common CPT/HCPCS Procedure Codes:
  Office visits: 99213 (established, low complexity), 99214 (established, moderate),
    99215 (established, high complexity), 99205 (new patient, high complexity)
  Chemotherapy: 96413 (IV infusion, first hour), 96415 (IV infusion, additional hour),
    96409 (IV push), 96401 (subcutaneous/intramuscular injection)
  Immunotherapy: 96413 (IV infusion, first hour - same code as chemo)
  Radiation: 77427 (radiation treatment management, 5 treatments), 77385 (IMRT delivery),
    77386 (IMRT delivery, complex)
  Imaging: 71260 (CT chest with contrast), 74178 (CT abdomen/pelvis with contrast),
    78816 (PET/CT whole body), 70553 (MRI brain with and without contrast)
  Surgical: 38500 (biopsy/excision lymph node), 32405 (lung biopsy),
    19301 (partial mastectomy), 44140 (partial colectomy)

Laboratory Reference Ranges:
  CBC: WBC 3.7-10.5 k/uL, Hgb 12.0-16.0 g/dL, Plt 150-400 k/uL, ANC 1.5-8.0 k/uL
  CMP: Cr 0.6-1.2 mg/dL, BUN 7-20 mg/dL, Na 136-145 mEq/L, K 3.5-5.0 mEq/L,
    AST 10-40 U/L, ALT 7-56 U/L, ALP 44-147 U/L, Bilirubin 0.1-1.2 mg/dL,
    Albumin 3.5-5.0 g/dL, Glucose 70-100 mg/dL, Ca 8.5-10.5 mg/dL
  Tumor markers: CEA 0-2.5 ng/mL, CA 19-9 0-37 U/mL, CA 125 0-35 U/mL,
    AFP 0-15 ng/mL, PSA 0-4.0 ng/mL, LDH 140-280 U/L

PRO Instruments:
  EORTC QLQ-C30 (0-100 scale, higher = better for functioning, higher = worse for symptoms):
    Functional scales: Physical Function, Role Function, Emotional Function,
      Cognitive Function, Social Function
    Symptom scales: Fatigue, Nausea and Vomiting, Pain, Dyspnea, Insomnia,
      Appetite Loss, Constipation, Diarrhea, Financial Difficulties
    Global: Global Health Status
  PRO-CTCAE (0-4 severity scale: 0=None, 1=Mild, 2=Moderate, 3=Severe, 4=Very Severe):
    Common symptoms: Nausea, Fatigue, Pain, Neuropathy, Diarrhea, Constipation,
      Mouth Sores, Rash, Shortness of Breath, Insomnia, Appetite Loss, Vomiting
"""

STRUCTURED_DATA_RULES = """
Rules for structured data events:
13. Encounter events: Include ICD-10 codes with decimal point and full description in parentheses. Include CPT codes with description. Format: DX: CODE (description). CPT: CODE (description).
14. Lab result events: Include test name, numeric value, unit, and (normal/low/high) flag. Group by panel (CBC, CMP, Tumor Markers). Values should reflect the patient's clinical status.
15. Hospitalization events: Include reason for admission, Principal DX: ICD-10 code (description), LOS: N days, Discharge: disposition (home, rehab, skilled nursing, deceased).
16. PRO assessment events: Include instrument name and all subscale scores. EORTC QLQ-C30 scores are integers 0-100. PRO-CTCAE severity scores are integers 0-4. Scores should be consistent with patient's clinical status and trajectory.
17. Include 3-6 encounter events, 2-4 lab panels, 0-2 hospitalizations, and 1-3 PRO assessments spread across the timeline. These should reflect the patient's disease trajectory.
18. Lab values should change over time (e.g., counts drop during chemotherapy, tumor markers may rise with progression).
19. PRO scores should correlate with clinical status (e.g., worse functioning during active treatment or disease progression).
"""

STRUCTURED_DATA_EXAMPLES = """
<encounter> At age 62, outpatient oncology visit. DX: C34.90 (Malignant neoplasm of unspecified part of unspecified bronchus or lung), Z51.11 (Encounter for antineoplastic chemotherapy). CPT: 99214 (Established patient visit, moderate complexity), 96413 (Chemotherapy IV infusion, first hour).
<lab_result> At age 62, CBC: WBC 8.2 k/uL (normal), Hgb 11.5 g/dL (low), Plt 245 k/uL (normal), ANC 5.1 k/uL (normal). CMP: Cr 0.9 mg/dL (normal), AST 28 U/L (normal), ALT 22 U/L (normal), Albumin 3.8 g/dL (normal).
<hospitalization> At age 63, admitted for febrile neutropenia. Principal DX: D70.1 (Agranulocytosis secondary to cancer chemotherapy). LOS: 4 days. Discharge: home.
<pro_assessment> At age 63, EORTC QLQ-C30: Physical Function 60, Role Function 40, Emotional Function 55, Cognitive Function 70, Social Function 50, Fatigue 65, Nausea 20, Pain 45, Global Health 50. PRO-CTCAE: Nausea severity 2, Fatigue severity 3, Pain severity 2, Neuropathy severity 1.
"""


def build_event_timeline_prompt(
    structured_data_config=None,
) -> str:
    """Build the event timeline prompt, optionally including structured data event types.

    Args:
        structured_data_config: StructuredDataConfig instance, or None to use base prompt

    Returns:
        The complete prompt template string (with format placeholders)
    """
    if structured_data_config is None or not structured_data_config.enabled:
        return EVENT_TIMELINE_PROMPT

    # Build list of enabled structured event types
    enabled_types = []
    if structured_data_config.include_encounters:
        enabled_types.append("encounter")
    if structured_data_config.include_hospitalizations:
        enabled_types.append("hospitalization")
    if structured_data_config.include_labs:
        enabled_types.append("lab_result")
    if structured_data_config.include_pros:
        enabled_types.append("pro_assessment")

    if not enabled_types:
        return EVENT_TIMELINE_PROMPT

    # Filter structured data event types to only enabled ones
    type_lines = STRUCTURED_DATA_EVENT_TYPES.strip().split("\n")
    filtered_types = []
    for line in type_lines:
        for etype in enabled_types:
            if f"<{etype}>" in line:
                filtered_types.append(line)
                break

    # Filter examples to only enabled types
    example_lines = STRUCTURED_DATA_EXAMPLES.strip().split("\n")
    filtered_examples = []
    for line in example_lines:
        for etype in enabled_types:
            if line.strip().startswith(f"<{etype}>"):
                filtered_examples.append(line)
                break

    # Filter PRO reference to only enabled instruments
    reference_text = STRUCTURED_DATA_REFERENCE
    if structured_data_config.include_pros:
        pro_instruments = structured_data_config.pro_instruments
        # The reference already includes both EORTC and PRO-CTCAE, which is fine
        # If we add more instruments later, we'd filter here

    # Build the extended prompt by inserting structured data sections
    structured_types_section = "\n".join(filtered_types)
    structured_examples_section = "\n".join(filtered_examples)

    # Insert into the base prompt
    prompt = EVENT_TIMELINE_PROMPT

    # Add structured event types to the types list
    prompt = prompt.replace(
        "- <ngs_report> Next-generation sequencing report (diagnosis, specimen site, detailed genomic findings)",
        "- <ngs_report> Next-generation sequencing report (diagnosis, specimen site, detailed genomic findings)\n"
        + structured_types_section,
    )

    # Add structured data rules
    prompt = prompt.replace(
        "12. To ensure diversity, vary patient names and genders",
        "12. To ensure diversity, vary patient names and genders\n"
        + STRUCTURED_DATA_RULES.strip(),
    )

    # Add reference schemas before the example section
    prompt = prompt.replace(
        "Example format (hypothetical, just to illustrate formatting):",
        reference_text.strip() + "\n\nExample format (hypothetical, just to illustrate formatting):",
    )

    # Add structured data examples
    prompt = prompt.replace(
        "Generate the event timeline now:",
        structured_examples_section + "\n\nGenerate the event timeline now:",
    )

    return prompt


def format_confounder_list(confounders: list) -> str:
    """Format role-tagged features into a readable list for prompts."""
    lines = []
    for c in confounders:
        roles = ", ".join(c.get("roles", []))
        role_text = f" Roles: [{roles}]." if roles else ""
        if c["type"] == "categorical":
            cats = ", ".join(c["categories"])
            lines.append(f"- {c['name']} (categorical): {c['description']}.{role_text} Categories: [{cats}]")
        else:
            lines.append(f"- {c['name']} (continuous): {c['description']}.{role_text}")
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
