"""Complex synthetic DGPs, variable 5..50 confounder/modifier dimensionality, and S7-S12b registry tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.evaluation import benchmark
from synthetic_data import cli as synthetic_cli
from synthetic_data import generator as g
from synthetic_data import longitudinal as long_sim
from synthetic_data import reequation as rq
from synthetic_data.config import LLMConfig
from synthetic_data.dgp_terms import NONLINEARITIES, feature_terms, nonlinear_array, nonlinear_value
from synthetic_data.llm_client import LLMClient, is_nemotron_3_5
from synthetic_data.wide_corpus import (
    ONCOLOGY_CONTEXTS,
    build_feature_catalog,
    generate_wide_source_cohort,
)


BUNDLED_NSCLC = (
    Path(__file__).resolve().parents[1]
    / "synthetic_data"
    / "example_synthetic_datasets"
    / "five_confounders_five_effect_modifiers_nsclc_with_structured"
)


def _read(directory: Path):
    return (
        pd.read_parquet(directory / "dataset.parquet"),
        json.loads((directory / "metadata.json").read_text(encoding="utf-8")),
    )


def _equation_inputs(metadata):
    unmeasured = [{"name": u["name"], "type": "continuous"} for u in metadata["unmeasured_confounders"]]
    features = metadata["features"] + unmeasured
    stats = {
        **metadata["summary_statistics"],
        **{u["name"]: {"mean": 0.0, "std": 1.0} for u in unmeasured},
    }
    return features, stats


def _all_terms_in_equation(eq: dict, *, hte_only: bool = False) -> set[str]:
    terms: set[str] = set()
    if not hte_only:
        terms.update(eq.get("coefficients", {}).keys())
        for inter in eq.get("interactions", ()):
            terms.update(inter.get("terms", ()))
    for inter in eq.get("treatment_interactions", ()):
        terms.add(str(inter.get("term", "")))
    for inter in eq.get("hte_interactions", ()):
        terms.update(inter.get("terms", ()))
    for rule in eq.get("subgroup_rules", ()):
        if hte_only and not rule.get("treatment_only"):
            continue
        for cond in rule.get("conditions", ()):
            terms.add(str(cond.get("term", "")))
    for comp in eq.get("rff_components", ()):
        if hte_only and not comp.get("treatment_only"):
            continue
        terms.update(comp.get("terms", ()))
    for rterm in eq.get("rational_terms", ()):
        if hte_only and not rterm.get("treatment_only"):
            continue
        terms.update(rterm.get("num_terms", ()))
        terms.update(rterm.get("den_terms", ()))
    return terms


def _verify_scalar_and_batch_match_oracle(frame: pd.DataFrame, metadata: dict, *, sample_rows: int = 80):
    features, stats = _equation_inputs(metadata)
    subset = frame.iloc[:sample_rows]
    chars_list = [
        {feat["name"]: row[f"true_{feat['name']}"] for feat in features}
        for row in subset.to_dict("records")
    ]
    tx_eq, out_eq = metadata["treatment_equation"], metadata["outcome_equation"]

    tx_scalar = np.array([g._compute_logit(c, features, stats, tx_eq) for c in chars_list])
    lp0_scalar = np.array([g._compute_logit(c, features, stats, out_eq, treatment=0) for c in chars_list])
    lp1_scalar = np.array([g._compute_logit(c, features, stats, out_eq, treatment=1) for c in chars_list])

    tx_batch = g._compute_logits_batch(chars_list, features, stats, tx_eq)
    lp0_batch = g._compute_logits_batch(chars_list, features, stats, out_eq, treatment=0)
    lp1_batch = g._compute_logits_batch(chars_list, features, stats, out_eq, treatment=1)

    np.testing.assert_allclose(tx_batch, tx_scalar, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(lp0_batch, lp0_scalar, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(lp1_batch, lp1_scalar, rtol=1e-10, atol=1e-10)

    e_exp = 1.0 / (1.0 + np.exp(-np.clip(tx_scalar, -500.0, 500.0)))
    p0_exp, p1_exp = g.potential_outcome_probabilities(lp0_scalar, lp1_scalar, out_eq)
    np.testing.assert_allclose(subset["true_treatment_prob"].to_numpy(), e_exp, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(subset["true_y0"].to_numpy(), p0_exp, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(subset["true_y1"].to_numpy(), p1_exp, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(subset["true_ite"].to_numpy(), p1_exp - p0_exp, rtol=1e-9, atol=1e-9)
    return tx_scalar, lp0_scalar, lp1_scalar


def test_all_nonlinearity_shapes_scalar_equals_vectorized():
    xs = np.linspace(-4.5, 4.5, 121)
    for shape in NONLINEARITIES:
        if shape == "mixed":
            continue
        scalar = np.array([nonlinear_value(float(x), shape) for x in xs])
        vector = nonlinear_array(xs, shape)
        np.testing.assert_allclose(vector, scalar, rtol=1e-12, atol=1e-12)


def test_enforce_feature_count_request_scales_linearly_to_125_features():
    features = []
    for i in range(55):
        features.append({"name": f"c_{i}", "type": "continuous", "roles": ["confounder"]})
    for i in range(55):
        features.append({"name": f"m_{i}", "type": "continuous", "roles": ["effect_modifier"]})
    for i in range(15):
        features.append({"name": f"d_{i}", "type": "continuous", "roles": ["confounder", "effect_modifier"]})

    t0 = time.perf_counter()
    selected = g._enforce_feature_count_request(
        features, num_features=100, num_confounders=50, num_effect_modifiers=50
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05
    assert len(selected) == 100
    assert sum("confounder" in f["roles"] for f in selected) == 50
    assert sum("effect_modifier" in f["roles"] for f in selected) == 50
    # Preserves original LLM order
    pos = {f["name"]: idx for idx, f in enumerate(features)}
    assert [pos[f["name"]] for f in selected] == sorted(pos[f["name"]] for f in selected)

    # Overlapping dual-role request: 100 total, 55 confounders, 55 modifiers -> 10 dual-role
    overlap = g._enforce_feature_count_request(
        features, num_features=100, num_confounders=55, num_effect_modifiers=55
    )
    assert len(overlap) == 100
    assert sum(set(f["roles"]) == {"confounder", "effect_modifier"} for f in overlap) == 10

    # Partial role request (only num_confounders given)
    partial = g._enforce_feature_count_request(
        features, num_features=90, num_confounders=50, num_effect_modifiers=None
    )
    assert len(partial) == 90
    assert sum("confounder" in f["roles"] for f in partial) == 50


@pytest.fixture(scope="module")
def wide_source(tmp_path_factory):
    out = tmp_path_factory.mktemp("wide_src") / "corpus_120"
    return generate_wide_source_cohort(
        out,
        num_patients=900,
        num_features=120,
        seed=7,
        within_block_corr=0.35,
        cross_block_corr=0.03,
        render_text=True,
    )


def test_wide_corpus_structure_correlation_and_cli(wide_source, tmp_path):
    frame, metadata = _read(wide_source)
    assert len(frame) == 900
    assert len(metadata["features"]) == 120
    assert metadata["scenario"] is None
    assert metadata["post_treatment"] is None
    assert metadata["unmeasured_confounders"] == []
    assert all(f"true_{f['name']}" in frame.columns for f in metadata["features"])
    assert all(isinstance(t, str) and "<new_note>" in t for t in frame["clinical_text"].iloc[:5])

    # Within-block correlation (block 3 CBC continuous features) exceeds cross-block correlation
    within = abs(frame["true_hemoglobin"].corr(frame["true_hematocrit"]))
    across = abs(frame["true_hemoglobin"].corr(frame["true_pro_fatigue"]))
    assert within > 0.18
    assert within > across + 0.10

    # CLI wide-corpus subcommand works end-to-end
    cli_out = tmp_path / "cli_wide"
    assert synthetic_cli.main([
        "wide-corpus", "--output-dir", str(cli_out), "--num-patients", "60",
        "--num-features", "40", "--seed", "3", "--no-text",
    ]) == 0
    cli_frame, cli_meta = _read(cli_out)
    assert len(cli_frame) == 60 and len(cli_meta["features"]) == 40


def test_registry_loads_s0_through_s12b_and_materializes_s7_to_s11(tmp_path):
    registry = benchmark.load_registry()
    for sid in ("S0", "S1", "S2", "S3", "S4a", "S4b", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12a", "S12b"):
        assert sid in registry.scenarios
    assert registry.scenarios["S12a"].role_feature_count() == 120
    assert registry.scenarios["S12b"].role_feature_count() == 110
    assert [s.id for s in registry.select("S12")] == ["S12a", "S12b"]

    # Materialize S7-S11 directly on the bundled 10-feature NSCLC corpus
    for sid in ("S7", "S8", "S9", "S10", "S11"):
        scenario = registry.scenarios[sid]
        out = rq.reequation_cohort(
            BUNDLED_NSCLC,
            scenario.dgp,
            size=400,
            seed=101,
            output_dir=tmp_path / sid,
        )
        frame, metadata = _read(out)
        assert len(frame) == 400
        tx_lp, lp0, lp1 = _verify_scalar_and_batch_match_oracle(frame, metadata, sample_rows=60)

        # Role-boundary invariants hold across every complex equation block
        tx_terms = _all_terms_in_equation(metadata["treatment_equation"])
        out_terms = _all_terms_in_equation(metadata["outcome_equation"])
        hte_terms = _all_terms_in_equation(metadata["outcome_equation"], hte_only=True)
        by_role = lambda role: {
            t for f in metadata["features"] if role in f["roles"] for t in feature_terms(f)
        }
        conf_t = by_role("confounder")
        mod_t = by_role("effect_modifier")
        inst_t = by_role("instrument")
        prog_t = by_role("prognostic")
        unassigned_t = {
            t for f in metadata["features"] if not f["roles"] for t in feature_terms(f)
        }
        u_names = {u["name"] for u in metadata["unmeasured_confounders"]}
        assert tx_terms <= (conf_t | inst_t | u_names)
        assert not (inst_t & out_terms)
        assert not (prog_t & tx_terms)
        assert hte_terms <= mod_t
        assert not (unassigned_t & (tx_terms | out_terms))

        if sid == "S7":
            assert np.max(np.abs(frame["true_ite"].to_numpy())) == 0.0
            assert frame["true_treatment_prob"].std() > 0.15
        elif sid == "S8":
            assert metadata["outcome_equation"]["treatment_interactions"] == []
            assert len(metadata["outcome_equation"].get("hte_interactions", [])) >= 4
            assert frame["true_ite"].std() > 0.04
            # Check genuine epistasis: 1-D linear regression on all standardized covariates
            # cannot explain the full HTE logit difference.
            features, stats = _equation_inputs(metadata)
            chars_all = [
                {feat["name"]: row[f"true_{feat['name']}"] for feat in features}
                for row in frame.to_dict("records")
            ]
            z_cols, cat_cols = g._build_term_matrix(chars_all, features, stats)
            X_lin = np.column_stack([np.ones(len(frame)), *z_cols.values(), *cat_cols.values()])
            y_hte = (
                g._compute_logits_batch(chars_all, features, stats, metadata["outcome_equation"], 1)
                - g._compute_logits_batch(chars_all, features, stats, metadata["outcome_equation"], 0)
            )
            beta, *_ = np.linalg.lstsq(X_lin, y_hte, rcond=None)
            r2_lin = 1.0 - np.var(y_hte - X_lin @ beta) / max(np.var(y_hte), 1e-12)
            assert r2_lin < 0.40
        elif sid == "S9":
            assert len(metadata["outcome_equation"].get("subgroup_rules", [])) >= 4
            assert 0.18 <= frame["true_treatment_prob"].mean() <= 0.32
            assert 0.09 <= frame["true_y0"].mean() <= 0.22
        elif sid == "S10":
            assert len(metadata["outcome_equation"].get("rff_components", [])) >= 1
            assert len(metadata["outcome_equation"].get("rational_terms", [])) >= 2
            assert "term_nonlinearities" in metadata["outcome_equation"]
            assert frame["true_ite"].std() > 0.05
        elif sid == "S11":
            # Unmeasured confounder enters strictly linearly -> benchmark.short_nuisances compatible
            u_entry = metadata["unmeasured_confounders"][0]
            assert u_entry["name"] == "u_1"
            assert "u_1" not in (metadata["outcome_equation"].get("nonlinear_terms") or [])
            assert "u_1" not in (metadata["treatment_equation"].get("nonlinear_terms") or [])
            e_short, g0, g1 = benchmark.short_nuisances(
                propensity=frame["true_treatment_prob"].to_numpy(),
                mu0=frame["true_y0"].to_numpy(),
                mu1=frame["true_y1"].to_numpy(),
                u=frame["true_u_1"].to_numpy(),
                treatment_coefficients=np.array([u_entry["coefficients"]["treatment"]]),
                outcome_coefficients=np.array([u_entry["coefficients"]["outcome"]]),
                binary=True,
            )
            assert np.all(np.isfinite(e_short)) and np.all(np.isfinite(g1 - g0))


def test_variable_dimensionality_5_to_50_confounders_and_modifiers(wide_source, tmp_path):
    registry = benchmark.load_registry()
    s12a = registry.scenarios["S12a"]
    s12b = registry.scenarios["S12b"]

    # S12a across multiple seeds spans wide variation in confounder and modifier counts in [5, 50]
    conf_counts = []
    mod_counts = []
    for seed in (101, 102, 103, 104, 105, 106):
        out = rq.reequation_cohort(
            wide_source, s12a.dgp, size=400, seed=seed, output_dir=tmp_path / f"s12a_{seed}"
        )
        frame, metadata = _read(out)
        n_conf_only = len(metadata["scenario"]["roles"]["confounder"])
        n_mod_only = len(metadata["scenario"]["roles"]["effect_modifier"])
        assert 5 <= n_conf_only <= 50
        assert 5 <= n_mod_only <= 50
        assert len(metadata["confounders"]) >= 5
        assert len(metadata["effect_modifiers"]) >= 5
        conf_counts.append(n_conf_only)
        mod_counts.append(n_mod_only)
        if seed == 101:
            _verify_scalar_and_batch_match_oracle(frame, metadata, sample_rows=40)
            assert 0.25 <= frame["true_treatment_prob"].mean() <= 0.45
            assert 0.15 <= frame["true_y0"].mean() <= 0.35
            assert frame["true_ite"].std() > 0.05

    assert len(set(conf_counts)) >= 4
    assert len(set(mod_counts)) >= 4
    assert max(conf_counts) - min(conf_counts) >= 15
    assert max(mod_counts) - min(mod_counts) >= 15

    # S12b: dense 50 confounders + 50 modifiers + 5 instruments + 5 prognostic (110 active features)
    t0 = time.perf_counter()
    out_50 = rq.reequation_cohort(
        wide_source, s12b.dgp, size=800, seed=101, output_dir=tmp_path / "s12b_101"
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0  # vectorized Monte Carlo calibration stays fast at 120 features
    frame_50, meta_50 = _read(out_50)
    assert len(meta_50["confounders"]) == 50
    assert len(meta_50["effect_modifiers"]) == 50
    assert len(meta_50["instruments"]) == 5
    assert len(meta_50["prognostic_variables"]) == 5
    # Pairwise treatment interactions stay bounded instead of exploding to 50*49/2+
    assert len(meta_50["treatment_equation"]["interactions"]) <= 150
    _verify_scalar_and_batch_match_oracle(frame_50, meta_50, sample_rows=40)
    assert 0.28 <= frame_50["true_treatment_prob"].mean() <= 0.42
    assert 0.18 <= frame_50["true_y0"].mean() <= 0.32
    assert 0.05 < frame_50["true_ite"].std() < 0.45

    # Extreme asymmetry: (5 confounders, 50 modifiers) and (50 confounders, 5 modifiers) both stay balanced
    for label, c_cnt, m_cnt in (("5c_50m", 5, 50), ("50c_5m", 50, 5)):
        scen = {
            "name": f"asym_{label}",
            "roles": {"confounder": c_cnt, "effect_modifier": m_cnt},
            "dgp_mode": "composite",
            "nonlinearity": "mixed",
            "treatment_nonlinearity": "mixed",
            "treatment_logit_std": 2.0,
            "outcome_logit_std": 2.0,
            "modifier_strength": 2.0,
        }
        out_asym = rq.reequation_cohort(
            wide_source, scen, size=500, seed=11, output_dir=tmp_path / label
        )
        df_a, meta_a = _read(out_asym)
        assert len(meta_a["confounders"]) == c_cnt
        assert len(meta_a["effect_modifiers"]) == m_cnt
        assert 0.05 < df_a["true_ite"].std() < 0.45


def test_risk_difference_effect_scale_and_longitudinal_complex_mode(tmp_path):
    rd_scenario = {
        "name": "rd_scale_complex",
        "roles": {"confounder": 3, "effect_modifier": 3},
        "dgp_mode": "interacting",
        "nonlinearity": "saturating",
        "effect_scale": "risk_difference",
        "treatment_effect": 0.3,
        "modifier_strength": 2.0,
    }
    out = rq.reequation_cohort(
        BUNDLED_NSCLC, rd_scenario, size=300, seed=5, output_dir=tmp_path / "rd"
    )
    frame, metadata = _read(out)
    assert metadata["outcome_equation"]["effect_scale"] == "risk_difference"
    _verify_scalar_and_batch_match_oracle(frame, metadata, sample_rows=50)
    assert frame["true_y1"].min() >= 0.01 - 1e-12
    assert frame["true_y1"].max() <= 0.99 + 1e-12

    # Longitudinal simulator supports complex DGP modes cleanly
    dgp = long_sim.draw_longitudinal_dgp(19, prior="mixed", dgp_mode="composite")
    patient = long_sim.sample_longitudinal_patient(0, seed=7, dgp=dgp)
    chars = {f["name"]: patient[f"true_{f['name']}"] for f in dgp.features}
    lp0 = g._compute_logit(chars, dgp.features, dgp.reference, dgp.outcome, treatment=0)
    lp1 = g._compute_logit(chars, dgp.features, dgp.reference, dgp.outcome, treatment=1)
    p0, p1 = g.potential_outcome_probabilities(lp0, lp1, dgp.outcome)
    assert patient["true_ite"] == pytest.approx(p1 - p0)


def test_nemotron_3_5_sampling_profile_in_llm_client():
    assert is_nemotron_3_5("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
    assert not is_nemotron_3_5("openai/gpt-oss-120b")

    nem_client = LLMClient(
        LLMConfig(
            api_base_url="http://127.0.0.1:8000/v1",
            model_name="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
            max_tokens=4096,
        )
    )
    kwargs = nem_client._request_sampling_kwargs(temperature=0.5, max_tokens=2048)
    assert kwargs["temperature"] == 1.0
    assert kwargs["max_tokens"] == 2048
    assert kwargs["extra_body"]["top_p"] == 0.95
    assert kwargs["extra_body"]["top_k"] == 0
    assert kwargs["extra_body"]["min_p"] == 0.0
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    nem_client.close()

    other_client = LLMClient(
        LLMConfig(
            api_base_url="http://127.0.0.1:8000/v1",
            model_name="openai/gpt-oss-120b",
            max_tokens=4096,
        )
    )
    other_kwargs = other_client._request_sampling_kwargs(temperature=0.5, max_tokens=2048)
    assert other_kwargs["temperature"] == 0.5
    assert "extra_body" not in other_kwargs
    other_client.close()


def test_wide_corpus_supports_all_six_oncology_contexts(tmp_path):
    assert set(ONCOLOGY_CONTEXTS) >= {"nsclc", "myeloma", "hcc", "mcrc", "breast_er_pos", "mcrpc"}
    for idx, context in enumerate(ONCOLOGY_CONTEXTS):
        catalog = build_feature_catalog(120, clinical_context=context)
        names = [f["name"] for f in catalog]
        assert len(names) == 120
        assert len(set(names)) == 120, f"duplicate names in {context}"
        out = generate_wide_source_cohort(
            tmp_path / context,
            num_patients=50,
            num_features=120,
            clinical_context=context,
            seed=10 + idx,
            render_text=False,
        )
        frame, meta = _read(out)
        assert len(frame) == 50
        assert meta["wide_corpus"]["clinical_context"] == context
        assert len(meta["features"]) == 120
