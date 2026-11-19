"""Pretreatment oncology visits and patient-reported symptom histories."""

import numpy as np
from scipy.special import expit

from .pro_ctcae import weekly_assessments
from .structured_data import LAB_REFERENCE_RANGES

DAYS_PER_YEAR = 365.25
LEGACY_STREAMS = ("notes", "labs", "encounters", "vitals")
STREAMS = (*LEGACY_STREAMS, "pros")
DIFFERENTIAL = ("neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils")
LAB_UNITS = {name: info[0] for name, info in LAB_REFERENCE_RANGES.items()}
LAB_UNITS.update(mcv="fL", mch="pg", mchc="g/dL", rdw="%", total_protein="g/dL")
LAB_UNITS.update({f"{cell}_abs": "k/uL" for cell in DIFFERENTIAL})
LAB_UNITS.update({f"{cell}_pct": "%" for cell in DIFFERENTIAL})

# Each entry is an alternative narration of the same performance limitation.
FUNCTION = (
    ("Keeping up with work and all usual activities.", "Remains fully active."),
    (
        "Independent at home but avoids strenuous chores.",
        "Light housework is manageable; heavy tasks are tiring.",
    ),
    (
        "Up for most of the day and manages self-care, but unable to work.",
        "Ambulatory and independent with dressing, though unable to resume usual work.",
    ),
    (
        "Needs help with dressing and spends most waking hours in a chair.",
        "Limited self-care; in bed or a chair for much of the day.",
    ),
)


def _append(stream, day, **observation):
    previous = stream[-1]["day_from_index"] if stream else None
    stream.append(
        dict(
            day_from_index=day,
            delta_years=None if previous is None else (day - previous) / DAYS_PER_YEAR,
            **observation,
        )
    )


def _cbc(rng, burden):
    wbc = round(float(np.clip(rng.normal(7.2 + 1.1 * burden, 0.9), 2.5, 22)), 2)
    # Correlated, compositional five-part differential. Each draw totals 100%.
    fractions = np.array([3.8 * np.exp(0.34 * burden), 1.8, 0.53, 0.18, 0.05])
    fractions = rng.dirichlet(20 * fractions)
    hgb = float(np.clip(rng.normal(12.9 - 0.8 * burden, 0.55), 7, 18))
    mcv = float(np.clip(rng.normal(89, 3.0), 72, 108))
    hct = hgb * rng.normal(3.0, 0.05)
    rbc = hct * 10 / mcv
    values = dict(
        wbc=wbc,
        hemoglobin=round(hgb, 1),
        hematocrit=round(hct, 1),
        rbc=round(rbc, 2),
        mcv=round(mcv, 1),
        mch=round(10 * hgb / rbc, 1),
        mchc=round(100 * hgb / hct, 1),
        rdw=round(float(rng.normal(14, 0.8)), 1),
        platelets=round(float(np.clip(rng.normal(250 + 15 * burden, 30), 80, 650))),
    )
    for cell, fraction in zip(DIFFERENTIAL, fractions):
        values[f"{cell}_abs"] = round(wbc * fraction, 3)
        values[f"{cell}_pct"] = round(100 * fraction, 2)
    return values


def _chemistry(rng, burden, creatinine, comprehensive):
    values = dict(
        sodium=rng.normal(139 - 0.6 * burden, 1.8),
        potassium=rng.normal(4.2, 0.26),
        chloride=rng.normal(103, 2.1),
        co2=rng.normal(25, 1.6),
        bun=max(4, rng.normal(8 + 9 * creatinine, 2)),
        creatinine=creatinine,
        glucose=max(55, rng.normal(99, 13)),
        calcium=rng.normal(9.3, 0.35),
    )
    if comprehensive:
        albumin = float(np.clip(rng.normal(3.9 - 0.22 * burden, 0.2), 2, 5.4))
        values.update(
            albumin=albumin,
            total_protein=albumin + rng.normal(2.9, 0.2),
            bilirubin=rng.lognormal(-0.45, 0.28),
            alp=rng.lognormal(4.45, 0.23),
            ast=rng.lognormal(3.23, 0.24),
            alt=rng.lognormal(3.19, 0.29),
        )
    return {name: round(float(value), 2) for name, value in values.items()}


def _panel(stream, day, panel, values):
    results = [
        {"component": key, "value": float(value), "unit": LAB_UNITS[key]}
        for key, value in values.items()
    ]
    _append(stream, day, panel=panel, results=results)


def _vitals(rng, weight, burden, baseline_bp):
    diastolic = round(float(rng.normal(baseline_bp, 3)))
    return dict(
        weight_kg=round(weight, 1),
        systolic_mmhg=diastolic + round(float(rng.normal(47, 5))),
        diastolic_mmhg=diastolic,
        pulse_bpm=round(float(rng.normal(78 + 4 * burden, 5))),
        respirations_per_min=round(float(np.clip(rng.normal(16 + burden, 1.1), 10, 30))),
        temperature_c=round(float(rng.normal(36.8, 0.25)), 1),
        spo2_pct=round(float(np.clip(rng.normal(96.7 - burden, 1), 85, 100))),
    )


def _note(rng, age, performance, previous_performance, emergency, hemoglobin, population, clinic):
    setting = "Emergency department" if emergency else clinic
    activity = rng.choice(FUNCTION[performance])
    common = rng.choice(
        (
            "Appetite varies from day to day.",
            "Sleep has been interrupted.",
            "Discussed home support.",
            "No new pain reported.",
        )
    )
    text = f"{setting} visit. {age}-year-old with {population}. {activity} {common}"
    if previous_performance is not None:
        if performance > previous_performance:
            text += " Activity tolerance has declined since the previous assessment."
        elif performance < previous_performance:
            text += " Daily activity has improved since the last assessment."
    if hemoglobin is not None and rng.random() < 0.45:
        text += f" Hemoglobin today {hemoglobin:.1f} g/dL."
    return text


# Realistic baseline oncology comorbidities / metastatic codes paired with
# baseline log-odds and frailty coupling so encounter embeddings train broadly.
_COMORBIDITY_CODES = (
    ("I10", -0.1, 0.35),      # Essential hypertension
    ("E11.9", -1.4, 0.35),    # Type 2 diabetes mellitus
    ("E78.5", -0.6, 0.20),    # Hyperlipidemia
    ("J44.9", -1.7, 0.55),    # COPD
    ("N18.3", -2.1, 0.50),    # Chronic kidney disease stage 3
    ("I48.91", -2.2, 0.45),   # Atrial fibrillation
    ("I50.9", -2.5, 0.65),    # Heart failure
    ("F32.9", -1.5, 0.30),    # Depressive disorder
    ("K21.9", -1.0, 0.15),    # GERD
    ("G62.9", -1.8, 0.35),    # Peripheral neuropathy
    ("I82.409", -2.3, 0.45),  # Deep vein thrombosis
    ("C79.51", -1.6, 0.55),   # Secondary malignant neoplasm of bone
    ("C78.7", -1.8, 0.55),    # Secondary malignant neoplasm of liver
    ("C78.00", -1.7, 0.50),   # Secondary malignant neoplasm of lung
    ("C79.31", -2.4, 0.60),   # Secondary malignant neoplasm of brain
    ("C78.6", -2.3, 0.55),    # Secondary malignant neoplasm of retroperitoneum/peritoneum
)


def simulate_history(seed, patient_id, *, include_pros=True, question=None):
    """Return observations up to the treatment decision for one synthetic patient."""
    rng = np.random.default_rng([seed, patient_id, 1])
    population = question["population"] if question else "lung cancer"
    diagnosis = question["icd10cm"] if question else "C34.90"
    clinic = "Medical oncology" if question else "Thoracic oncology"
    age_at_index = float(rng.uniform(40, 86))
    frailty, renal, drift = rng.normal(size=3)
    # Sicker patients have modestly more frequent pretreatment visits.
    mean_visits = float(np.clip(7.5 + 1.1 * frailty, 4.5, 11.5))
    n_visits = int(np.clip(np.rint(rng.normal(mean_visits, 2.0)), 4, 12))
    gap_scale = float(np.clip(1.0 - 0.12 * frailty, 0.65, 1.35))
    gaps = np.clip(np.rint(rng.integers(10, 85, n_visits - 1) * gap_scale), 7, 110).astype(int)
    days = -int(rng.integers(1, 22)) - np.r_[np.cumsum(gaps)[::-1], 0]
    streams = {name: [] for name in (STREAMS if include_pros else LEGACY_STREAMS)}
    visit_burden = []
    base_weight = float(rng.uniform(52, 112))
    weight_slope = rng.normal(-2.5 - 1.7 * drift, 3)
    baseline_bp = rng.normal(77, 7)
    comorbidities = [
        code
        for code, base_logit, slope in _COMORBIDITY_CODES
        if rng.random() < expit(base_logit + slope * frailty + 0.015 * (age_at_index - 63.0))
    ]
    fluctuation = 0.0
    previous_performance = None
    for i, visit_day in enumerate(days):
        day = int(visit_day)
        years = (day - int(days[0])) / DAYS_PER_YEAR
        fluctuation = 0.65 * fluctuation + rng.normal(0, 0.32)
        burden = float(np.clip(frailty + drift * years + fluctuation, -3, 3))
        visit_burden.append(burden)
        creatinine = float(
            np.clip(
                1.02 + 0.22 * renal + (0.03 + 0.13 * drift) * years + rng.normal(0, 0.045),
                0.35,
                4.5,
            )
        )
        weight = float(np.clip(base_weight + weight_slope * years + rng.normal(0, 0.7), 38, 155))
        performance = int(np.clip(np.rint(1.1 + 0.75 * burden), 0, 3))
        emergency = rng.random() < expit(-1.85 + 0.8 * burden)
        inpatient = (not emergency) and (rng.random() < expit(-3.1 + 0.9 * burden))
        if emergency:
            setting = "emergency"
        elif inpatient:
            setting = "inpatient"
        elif rng.random() < 0.16:
            setting = "infusion"
        elif rng.random() < 0.10:
            setting = "telehealth"
        else:
            setting = "outpatient"

        # The endpoints anchor change over time; intermediate visits have
        # severity-informed documentation and ordering patterns.
        anchor = i == 0 or i == n_visits - 1
        cbc_prob = float(expit(0.95 + 0.35 * burden))
        chem_prob = float(expit(0.65 + 0.35 * burden))
        vital_prob = float(expit(1.65 + 0.25 * burden)) if setting != "telehealth" else 0.15
        note_prob = float(expit(1.10 + 0.25 * burden))
        hemoglobin = None
        cbc = None
        chem = None
        comprehensive = False
        if anchor or rng.random() < cbc_prob:
            cbc = _cbc(rng, burden)
            _panel(streams["labs"], day, "CBC_DIFF", cbc)
            hemoglobin = cbc["hemoglobin"]
        if anchor or rng.random() < chem_prob:
            comprehensive = rng.random() < 0.55
            chem = _chemistry(rng, burden, creatinine, comprehensive)
            _panel(streams["labs"], day, "CMP" if comprehensive else "BMP", chem)
        if anchor or rng.random() < vital_prob:
            _append(streams["vitals"], day, **_vitals(rng, weight, burden, baseline_bp))
        if anchor or rng.random() < note_prob:
            text = _note(
                rng,
                int(age_at_index + day / DAYS_PER_YEAR),
                performance,
                previous_performance,
                emergency,
                hemoglobin,
                population,
                clinic,
            )
            _append(streams["notes"], day, text=f"{text} ECOG performance status: {performance}.")
            previous_performance = performance

        diagnoses = [diagnosis]
        if emergency or inpatient:
            diagnoses.append("R06.02" if burden > 0 else "R53.83")
        if burden > 0.6 and rng.random() < 0.45:
            diagnoses.append(str(rng.choice(["R53.83", "G89.3", "R11.0", "R63.4"])))
        if cbc is not None:
            if cbc["hemoglobin"] < 11.2:
                diagnoses.append("D64.9")
            if cbc["neutrophils_abs"] < 1.5:
                diagnoses.append("D70.9")
            if cbc["platelets"] < 140:
                diagnoses.append("D69.6")
        if chem is not None:
            if chem["creatinine"] > 1.5:
                diagnoses.append("N17.9" if burden > 0.8 else "N18.3")
            if chem["sodium"] < 135:
                diagnoses.append("E87.1")
        for code in comorbidities:
            if rng.random() < (0.65 if anchor or inpatient else 0.30):
                diagnoses.append(code)
        diagnoses = list(dict.fromkeys(diagnoses))

        if emergency:
            cpts = ["99284"]
            if burden > 0.8:
                cpts.append("99285")
        elif inpatient:
            cpts = ["99223"]
        elif setting == "infusion":
            cpts = ["96413", "96365"]
        elif setting == "telehealth":
            cpts = ["99213"]
        else:
            cpts = ["99215" if burden > 0.9 else ("99214" if burden > -0.6 else "99213")]
        if cbc is not None:
            cpts.extend(["36415", "85025"])
        if chem is not None:
            cpts.append("80053" if comprehensive else "80048")
        if (anchor or emergency or inpatient) and rng.random() < 0.35:
            cpts.append(str(rng.choice(["71260", "74177", "70553", "78815"])))
        cpts = list(dict.fromkeys(cpts))

        _append(
            streams["encounters"],
            day,
            setting=setting,
            icd10cm=diagnoses,
            cpt=cpts,
        )
    if include_pros:
        survey_rng = np.random.default_rng([seed, patient_id, 3])
        for day, form in weekly_assessments(survey_rng, days, visit_burden):
            _append(streams["pros"], day, **form)
    return streams


def history_features(streams):
    """Summaries read by the treatment and potential-outcome equations."""
    labs = streams["labs"]
    creatinine = [
        (panel["day_from_index"], item["value"])
        for panel in labs
        for item in panel["results"]
        if item["component"] == "creatinine"
    ]
    last_cbc = next(panel for panel in reversed(labs) if panel["panel"] == "CBC_DIFF")
    diff = {r["component"]: r["value"] for r in last_cbc["results"]}
    first, last = streams["vitals"][0], streams["vitals"][-1]
    span = (last["day_from_index"] - first["day_from_index"]) / DAYS_PER_YEAR
    (d0, cr0), (d1, cr1) = creatinine[0], creatinine[-1]
    recorded_ecog = streams["notes"][-1]["text"].rsplit(" ECOG performance status: ", 1)[1]
    features = dict(
        latest_ecog=float(recorded_ecog.rstrip(".")),
        log_nlr=float(np.log(diff["neutrophils_abs"] / diff["lymphocytes_abs"])),
        creatinine_slope=(cr1 - cr0) * DAYS_PER_YEAR / (d1 - d0),
        weight_loss_per_year=100
        * (first["weight_kg"] - last["weight_kg"])
        / first["weight_kg"]
        / span,
        recent_ed_visits=float(
            sum(
                e["setting"] == "emergency" and e["day_from_index"] >= -180
                for e in streams["encounters"]
            )
        ),
    )
    if "pros" in streams:
        # The simulator collects fatigue at the final assessment for every enrollee.
        scores = [
            item["score"]
            for form in streams["pros"]
            for item in form["responses"]
            if item["symptom"] == "fatigue"
            and item["attribute"] == "interference"
            and item["score"] is not None
        ]
        features["mean_fatigue_interference"] = float(np.mean(scores))
    return features
