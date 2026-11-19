"""Irregular pretreatment oncology visits and four synchronized observation streams."""

import numpy as np
from scipy.special import expit

from .structured_data import LAB_REFERENCE_RANGES

DAYS_PER_YEAR = 365.25
STREAMS = ("notes", "labs", "encounters", "vitals")
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


def _note(rng, age, performance, previous_performance, emergency, hemoglobin):
    setting = "Emergency department" if emergency else "Thoracic oncology"
    activity = rng.choice(FUNCTION[performance])
    common = rng.choice(
        (
            "Appetite varies from day to day.",
            "Sleep has been interrupted.",
            "Discussed home support.",
            "No new pain reported.",
        )
    )
    text = f"{setting} visit. {age}-year-old with lung cancer. {activity} {common}"
    if previous_performance is not None:
        if performance > previous_performance:
            text += " Activity tolerance has declined since the previous assessment."
        elif performance < previous_performance:
            text += " Daily activity has improved since the last assessment."
    if hemoglobin is not None and rng.random() < 0.45:
        text += f" Hemoglobin today {hemoglobin:.1f} g/dL."
    return text


def simulate_history(seed, patient_id):
    """Return observations up to the treatment decision for one synthetic patient."""
    rng = np.random.default_rng([seed, patient_id, 1])
    n_visits = int(rng.integers(4, 13))
    # Build backwards from the last pretreatment visit, then emit forwards.
    gaps = rng.integers(10, 85, n_visits - 1)
    days = -int(rng.integers(1, 22)) - np.r_[np.cumsum(gaps)[::-1], 0]
    streams = {name: [] for name in STREAMS}
    age_at_index = float(rng.uniform(40, 86))
    frailty, renal, drift = rng.normal(size=3)
    base_weight = float(rng.uniform(52, 112))
    weight_slope = rng.normal(-2.5 - 1.7 * drift, 3)
    baseline_bp = rng.normal(77, 7)
    fluctuation = 0.0
    previous_performance = None
    for i, visit_day in enumerate(days):
        day = int(visit_day)
        years = (day - int(days[0])) / DAYS_PER_YEAR
        fluctuation = 0.65 * fluctuation + rng.normal(0, 0.32)
        burden = float(np.clip(frailty + drift * years + fluctuation, -3, 3))
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
        diagnoses = ["C34.90"]
        if emergency:
            diagnoses.append("R06.02" if burden > 0 else "R53.83")
        _append(
            streams["encounters"],
            day,
            setting="emergency" if emergency else "outpatient",
            icd10cm=diagnoses,
            cpt=["99284" if emergency else "99214"],
        )

        # The endpoints anchor change over time; intermediate visits have distinct
        # documentation and ordering patterns in the four modalities.
        anchor = i == 0 or i == n_visits - 1
        hemoglobin = None
        if anchor or rng.random() < 0.72:
            cbc = _cbc(rng, burden)
            _panel(streams["labs"], day, "CBC_DIFF", cbc)
            hemoglobin = cbc["hemoglobin"]
        if anchor or rng.random() < 0.66:
            comprehensive = rng.random() < 0.55
            values = _chemistry(rng, burden, creatinine, comprehensive)
            _panel(streams["labs"], day, "CMP" if comprehensive else "BMP", values)
        if anchor or rng.random() < 0.84:
            _append(streams["vitals"], day, **_vitals(rng, weight, burden, baseline_bp))
        if anchor or rng.random() < 0.75:
            text = _note(
                rng,
                int(age_at_index + day / DAYS_PER_YEAR),
                performance,
                previous_performance,
                emergency,
                hemoglobin,
            )
            _append(streams["notes"], day, text=f"{text} ECOG performance status: {performance}.")
            previous_performance = performance
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
    return dict(
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
