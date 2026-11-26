#!/usr/bin/env python3
"""Re-equate existing full-LLM multimodal eval cohorts with complex & 5-50-width DGPs.

Keeps every patient's Nemotron-120B clinical notes and longitudinal streams
(notes, labs, encounters, vitals, PROs) byte-identical while recomputing
treatment, outcome, and oracle potential outcomes under out-of-family complex
DGPs:
  - ``epistatic_s8``: pure-epistatic higher-order HTE (zero 1-D linear modifier terms)
  - ``nonparametric_s10``: Bochner RFF GP + bounded rational + mixed nonlinearities
  - ``composite_var_width``: variable 5..min(50, P//2) confounders & modifiers
  - ``dense_50_50`` (on P>=100 cohorts): 50 confounders + 50 modifiers composite DGP
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_data.reequation import reequation_cohort


def build_scenarios(num_features: int) -> list[tuple[str, dict]]:
    if num_features < 12:
        return []
    max_half = min(50, num_features // 2)
    max_each = min(50, max(5, (num_features - 4) // 2))
    scenarios: list[tuple[str, dict]] = [
        (
            "epistatic_s8",
            {
                "name": f"llm_epistatic_p{num_features}",
                "roles": {
                    "confounder": min(max_each, max(5, num_features // 4)),
                    "effect_modifier": min(max_each, max(5, num_features // 4)),
                    "dual_role": min(4, max(2, num_features // 12)),
                },
                "role_assignment": "random",
                "dgp_mode": "interacting",
                "modifier_main_effects": True,
                "nonlinearity": "mixed",
                "treatment_nonlinearity": "mixed",
                "outcome_interactions": min(12, max(4, num_features // 8)),
                "hte_interactions": min(14, max(6, num_features // 6)),
                "pure_epistatic_hte": True,
                "treatment_logit_std": 2.0,
                "outcome_logit_std": 2.0,
                "treatment_effect": 0.2,
                "modifier_strength": 2.0,
                "target_treatment_rate": 0.4,
                "target_control_outcome_rate": 0.3,
            },
        ),
        (
            "nonparametric_s10",
            {
                "name": f"llm_nonparametric_rff_p{num_features}",
                "roles": {
                    "confounder": min(max_each, max(5, num_features // 3)),
                    "effect_modifier": min(max_each, max(5, num_features // 3)),
                    "dual_role": min(4, max(2, num_features // 15)),
                },
                "role_assignment": "random",
                "dgp_mode": "nonparametric",
                "modifier_main_effects": True,
                "nonlinearity": "mixed",
                "treatment_nonlinearity": "mixed",
                "rff_components": {
                    "treatment": 32,
                    "outcome": 32,
                    "hte": 32,
                    "lengthscale": 1.15,
                    "max_terms": 14,
                },
                "rational_terms": {"treatment": 2, "outcome": 2, "hte": 2},
                "component_weights": {
                    "linear": 0.35,
                    "interaction": 0.4,
                    "subgroup": 0.0,
                    "rff": 1.0,
                    "rational": 0.6,
                },
                "treatment_logit_std": 2.0,
                "outcome_logit_std": 2.0,
                "treatment_effect": 0.2,
                "modifier_strength": 2.0,
                "target_treatment_rate": 0.5,
                "target_control_outcome_rate": 0.5,
            },
        ),
        (
            "composite_var_width",
            {
                "name": f"llm_composite_var_5_to_{max_half}",
                "roles": {
                    "confounder": {"min": 5, "max": max_half},
                    "effect_modifier": {"min": 5, "max": max_half},
                },
                "role_assignment": "random",
                "dgp_mode": "composite",
                "modifier_main_effects": True,
                "nonlinearity": "mixed",
                "treatment_nonlinearity": "mixed",
                "outcome_interactions": {"min": 4, "max": min(15, max(5, max_half // 2))},
                "hte_interactions": {"min": 4, "max": min(15, max(5, max_half // 2))},
                "subgroup_rules": {
                    "treatment": 3,
                    "outcome": 3,
                    "hte": 4,
                    "min_depth": 2,
                    "max_depth": 3,
                    "soft_slope": None,
                },
                "rff_components": {
                    "treatment": 28,
                    "outcome": 28,
                    "hte": 28,
                    "lengthscale": 1.15,
                    "max_terms": 14,
                },
                "rational_terms": {"treatment": 2, "outcome": 2, "hte": 2},
                "treatment_logit_std": 2.0,
                "outcome_logit_std": 2.0,
                "treatment_effect": 0.2,
                "modifier_strength": 2.0,
                "target_treatment_rate": 0.35,
                "target_control_outcome_rate": 0.25,
            },
        ),
    ]
    if num_features >= 100:
        scenarios.append(
            (
                "dense_50_50",
                {
                    "name": f"llm_dense_50_50_p{num_features}",
                    "roles": {"confounder": 50, "effect_modifier": 50},
                    "role_assignment": "random",
                    "dgp_mode": "composite",
                    "modifier_main_effects": True,
                    "nonlinearity": "mixed",
                    "treatment_nonlinearity": "mixed",
                    "outcome_interactions": 15,
                    "hte_interactions": 15,
                    "subgroup_rules": {
                        "treatment": 4,
                        "outcome": 4,
                        "hte": 5,
                        "min_depth": 2,
                        "max_depth": 3,
                        "soft_slope": None,
                    },
                    "rff_components": {
                        "treatment": 32,
                        "outcome": 32,
                        "hte": 32,
                        "lengthscale": 1.2,
                        "max_terms": 14,
                    },
                    "rational_terms": {"treatment": 3, "outcome": 3, "hte": 3},
                    "treatment_logit_std": 2.0,
                    "outcome_logit_std": 2.0,
                    "treatment_effect": 0.2,
                    "modifier_strength": 2.0,
                    "target_treatment_rate": 0.35,
                    "target_control_outcome_rate": 0.25,
                },
            )
        )
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("artifacts/llm_eval_cohorts"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/llm_eval_cohorts_complex")
    )
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    summary = []
    for meta_path in sorted(args.source_root.glob("*/metadata.json")):
        cohort_dir = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        num_features = len(meta.get("features") or [])
        scenarios = build_scenarios(num_features)
        if not scenarios:
            print(f"Skipping narrow cohort {cohort_dir.name} (P={num_features})")
            continue
        for idx, (tag, scenario) in enumerate(scenarios):
            out_dir = args.output_root / tag / cohort_dir.name
            seed = args.seed + idx * 17 + (sum(ord(ch) for ch in cohort_dir.name) % 97)
            reequation_cohort(cohort_dir, scenario, size=None, seed=seed, output_dir=out_dir)
            out_meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            n_c = len(out_meta["confounders"])
            n_m = len(out_meta["effect_modifiers"])
            ite_sd = out_meta["dataset_statistics"].get("std_ite")
            print(
                f"{tag:20s} {cohort_dir.name:24s} P={num_features:3d} "
                f"conf={n_c:2d} mod={n_m:2d} ite_sd={ite_sd:.3f}"
            )
            summary.append(
                {
                    "tag": tag,
                    "cohort": cohort_dir.name,
                    "num_features": num_features,
                    "num_confounders": n_c,
                    "num_effect_modifiers": n_m,
                    "seed": seed,
                    "ite_sd": ite_sd,
                    "output_dir": str(out_dir),
                }
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Materialized {len(summary)} complex full-LLM cohorts in {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
