"""Offline unit tests for SparseTabularCausalPFN (up to P = 2,000 candidate features)."""

import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from oci.tabular_pfn.cli import main
from oci.tabular_pfn.model import (
    CHECKPOINT_FORMAT_VERSION,
    ROLE_NAMES,
    SparseTabularCausalPFN,
    TabularModelConfig,
    calibrate_role_probabilities,
    format_role_summary,
)
from oci.tabular_pfn.preprocessing import TabularPreprocessor
from oci.tabular_pfn.prior import DGP_FAMILIES, ROLE_TO_ID, PriorConfig, sample_tabular_study
from oci.tabular_pfn.training import evaluate_tabular, load_tabular_run


def test_prior_high_p_2000_and_null_hte():
    # High-P = 2000 study with instruments, proxies, and missingness
    ep = sample_tabular_study(
        101,
        n_total=160,
        num_features=2000,
        prior=PriorConfig(null_hte_prob=0.0, instrument_prob=1.0),
    )
    assert ep["X"].shape == (160, 2000)
    assert ep["M"].shape == (160, 2000)
    assert ep["roles"].shape == (2000,)
    assert ep["meta"]["p"] == 2000
    assert ep["meta"]["n_inst"] >= 1
    assert np.all((ep["true_propensity"] > 0) & (ep["true_propensity"] < 1))
    assert np.all((ep["true_y0"] > 0) & (ep["true_y0"] < 1))
    assert np.all((ep["true_y1"] > 0) & (ep["true_y1"] < 1))
    assert set(ep["t"].tolist()).issubset({0.0, 1.0})
    assert set(ep["y"].tolist()).issubset({0.0, 1.0})
    # Most columns must be noise (extreme sparsity)
    assert (ep["roles"] == ROLE_TO_ID["noise"]).mean() > 0.95

    # Strictly null-HTE study has constant treatment logit (zero modifier/dual_role columns)
    ep_null = sample_tabular_study(
        202,
        n_total=120,
        num_features=80,
        prior=PriorConfig(null_hte_prob=1.0),
    )
    assert ep_null["meta"]["is_null_hte"] is True
    assert ep_null["meta"]["n_mod"] == 0 and ep_null["meta"]["n_dual"] == 0
    assert not np.any(ep_null["roles"] == ROLE_TO_ID["modifier"])
    assert not np.any(ep_null["roles"] == ROLE_TO_ID["dual_role"])


def test_prior_all_three_oci_families_and_split_disjointness():
    # Every OCI DGP family must produce clean P = 2,000 studies (and exact P = 30 studies)
    for fam in DGP_FAMILIES:
        ep_hi = sample_tabular_study(
            303,
            n_total=120,
            num_features=2000,
            prior=PriorConfig(
                unmeasured_conf_prob=1.0,
                extraction_noise_prob=1.0,
                mnar_prob=1.0,
                rd_scale_prob=1.0,
                design_split="train",
            ),
            family=fam,
        )
        assert ep_hi["X"].shape == (120, 2000)
        assert ep_hi["roles"].shape == (2000,)
        assert ep_hi["meta"]["family"] == fam
        assert ep_hi["meta"]["scale"] == "RD"
        assert ep_hi["meta"]["n_unmeasured"] >= 1
        assert np.all(np.isfinite(ep_hi["true_ite"]))

        ep_lo = sample_tabular_study(304, n_total=90, num_features=30, family=fam)
        assert ep_lo["X"].shape == (90, 30)

    # Strict train vs held_out disjointness on bundled nemotron_5000.json.gz
    tr_ids = {
        sample_tabular_study(1000 + i, 60, num_features=40, prior=PriorConfig(design_split="train"), family="nemotron_design")["meta"]["design_id"]
        for i in range(25)
    }
    ho_ids = {
        sample_tabular_study(2000 + i, 60, num_features=40, prior=PriorConfig(design_split="held_out"), family="nemotron_design")["meta"]["design_id"]
        for i in range(25)
    }
    assert tr_ids.isdisjoint(ho_ids) and len(tr_ids) > 5 and len(ho_ids) > 5


def test_preprocessor_mixed_types_and_three_state_missingness():
    ctx_tbl = pa.table(
        {
            "patient_id": list(range(12)),
            "age": [60, 65, 70, None, 55, 72, 68, 61, 64, 59, 66, 71],
            "biomarker": [
                "present",
                "absent",
                "not_documented",
                "documented_unknown",
                "present",
                "absent",
                "present",
                "absent",
                "not_documented",
                "present",
                "absent",
                "present",
            ],
            "stage": [
                "III",
                "IV",
                "IV",
                "III",
                "IV",
                "not_documented",
                "documented_unknown",
                "III",
                "IV",
                "III",
                "IV",
                "III",
            ],
            "const_col": [1.0] * 12,
        }
    )
    q_tbl = pa.table(
        {
            "patient_id": [100, 101],
            "age": [63, None],
            "biomarker": ["documented_unknown", "present"],
            "stage": ["IV", "unseen_stage"],
            "const_col": [1.0, 1.0],
        }
    )
    pre = TabularPreprocessor(min_category_count=3)
    X_ctx, M_ctx = pre.fit_transform(ctx_tbl, ["age", "biomarker", "stage", "const_col"])
    X_q, M_q = pre.transform(q_tbl)

    # Constant column dropped; documented_unknown flags emitted for biomarker and stage
    assert "const_col" not in pre.feature_names
    assert "biomarker__documented_unknown" in pre.feature_names
    assert "stage__documented_unknown" in pre.feature_names
    assert X_ctx.shape == (12, len(pre.feature_names))
    assert X_q.shape == (2, len(pre.feature_names))
    assert M_ctx[3, pre.feature_names.index("age")] == 1.0
    assert X_q[0, pre.feature_names.index("biomarker__documented_unknown")] == 1.0
    assert M_q[1, pre.feature_names.index("age")] == 1.0


def test_model_p2000_forward_backward_and_invariance():
    torch.manual_seed(0)
    ep = sample_tabular_study(303, n_total=140, num_features=2000)
    cfg = TabularModelConfig(
        col_dim=32,
        study_width=32,
        study_layers=1,
        col_attn_layers=1,
        heads=4,
        bins=17,
        top_k_nuis=24,
        top_k_cate=16,
        attn_slots=8,
        ridge_grid_size=4,
    )
    model = SparseTabularCausalPFN(cfg)

    n_ctx = 100
    X_ctx = torch.from_numpy(ep["X"][:n_ctx])
    M_ctx = torch.from_numpy(ep["M"][:n_ctx])
    t_ctx = torch.from_numpy(ep["t"][:n_ctx])
    y_ctx = torch.from_numpy(ep["y"][:n_ctx])
    X_q = torch.from_numpy(ep["X"][n_ctx:])
    M_q = torch.from_numpy(ep["M"][n_ctx:])
    oracle = torch.from_numpy(np.stack([ep["true_y0"][n_ctx:], ep["true_y1"][n_ctx:]], axis=1))
    true_prop = torch.from_numpy(ep["true_propensity"][n_ctx:])
    roles = torch.from_numpy(ep["roles"])

    # 1. Forward + loss + backward at P = 2,000
    out_logits, prop_logits, scale, mem = model(X_ctx, t_ctx, y_ctx, X_q, M_ctx=M_ctx, M_q=M_q)
    assert out_logits.shape == (40, 2, cfg.bins)
    assert prop_logits.shape == (40,)
    assert mem.role_logits.shape == (2000, len(ROLE_NAMES))
    loss, comps = model.loss_components(
        out_logits,
        prop_logits,
        scale,
        oracle,
        true_prop,
        role_logits=mem.role_logits,
        nuis_screen_logits=mem.nuis_screen_logits,
        mod_screen_logits=mem.mod_screen_logits,
        true_roles=roles,
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) == sum(1 for _ in model.parameters())
    assert all(torch.isfinite(g).all() for g in grads)
    assert model.nuis_screen_head.weight.grad.abs().sum().item() > 0.0
    assert model.mod_screen_head.weight.grad.abs().sum().item() > 0.0

    # 2. Row-permutation invariance AND column-permutation invariance + zero query leakage
    model.eval()
    ep_small = sample_tabular_study(404, n_total=90, num_features=40)
    Xc = torch.from_numpy(ep_small["X"][:70])
    Mc = torch.from_numpy(ep_small["M"][:70])
    tc = torch.from_numpy(ep_small["t"][:70])
    yc = torch.from_numpy(ep_small["y"][:70])
    Xq = torch.from_numpy(ep_small["X"][70:])
    Mq = torch.from_numpy(ep_small["M"][70:])

    with torch.no_grad():
        ol1, pl1, sc1, mem1 = model(Xc, tc, yc, Xq, M_ctx=Mc, M_q=Mq)
        m1, s1 = model.risk_stats(ol1, sc1)
        p1 = pl1.sigmoid()

        # Shuffle context rows AND shuffle all 40 columns simultaneously!
        rng = np.random.default_rng(7)
        row_perm = torch.from_numpy(rng.permutation(70))
        col_perm = torch.from_numpy(rng.permutation(40))
        ol2, pl2, sc2, mem2 = model(
            Xc[row_perm][:, col_perm],
            tc[row_perm],
            yc[row_perm],
            Xq[:, col_perm],
            M_ctx=Mc[row_perm][:, col_perm],
            M_q=Mq[:, col_perm],
        )
        m2, s2 = model.risk_stats(ol2, sc2)
        p2 = pl2.sigmoid()
        torch.testing.assert_close(m1, m2, atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(s1, s2, atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(p1, p2, atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(
            mem1.role_logits[col_perm], mem2.role_logits, atol=5e-4, rtol=5e-4
        )
        torch.testing.assert_close(
            mem1.nuis_screen_logits[col_perm], mem2.nuis_screen_logits, atol=5e-4, rtol=5e-4
        )
        torch.testing.assert_close(
            mem1.mod_screen_logits[col_perm], mem2.mod_screen_logits, atol=5e-4, rtol=5e-4
        )

        # Zero query leakage: corrupting queries does not affect condition() memory
        mem_a, _ = model.condition(Xc, tc, yc, M_ctx=Mc)
        torch.testing.assert_close(mem1.role_logits, mem_a.role_logits)
        report = model.column_role_report(mem1)
        assert len(report) == 40 and "predicted_role" in report[0]


def test_cli_train_resume_evaluate_and_predict(tmp_path):
    run = tmp_path / "tab_run"
    main(
        [
            "train",
            "--run-dir",
            str(run),
            "--steps",
            "2",
            "--min-context-size",
            "40",
            "--context-size",
            "60",
            "--query-size",
            "16",
            "--min-features",
            "20",
            "--max-features",
            "40",
            "--col-dim",
            "32",
            "--study-width",
            "32",
            "--study-layers",
            "1",
            "--col-attn-layers",
            "1",
            "--heads",
            "4",
            "--top-k-nuis",
            "16",
            "--top-k-cate",
            "12",
            "--attn-slots",
            "8",
            "--bins",
            "17",
            "--grad-accum-steps",
            "2",
            "--eval-every",
            "2",
            "--eval-studies",
            "4",
            "--eval-context-size",
            "50",
            "--eval-num-features",
            "30",
            "--save-every",
            "2",
            "--device",
            "cpu",
        ]
    )
    main(["resume", "--run-dir", str(run), "--steps", "3", "--save-every", "3", "--eval-every", "0", "--grad-accum-steps", "2", "--min-context-size", "40", "--context-size", "60", "--query-size", "16", "--device", "cpu"])
    model, saved = load_tabular_run(run, "cpu", prefer_best=False)
    assert saved["step"] == 3 and saved["format_version"] == CHECKPOINT_FORMAT_VERSION

    # Evaluate CLI
    main(
        [
            "evaluate",
            "--run-dir",
            str(run),
            "--studies",
            "4",
            "--context-size",
            "50",
            "--query-size",
            "16",
            "--num-features",
            "30",
            "--device",
            "cpu",
        ]
    )

    # Predict CLI with role report on a synthetic Parquet table
    ep = sample_tabular_study(999, n_total=60, num_features=25)
    cols = {"patient_id": list(range(60)), "treatment_indicator": ep["t"], "outcome_indicator": ep["y"]}
    for j in range(25):
        cols[f"cand_{j:03d}"] = ep["X"][:, j]
    data_path = tmp_path / "stage2_catalog.parquet"
    pq.write_table(pa.table(cols), data_path)

    out_path = tmp_path / "preds.parquet"
    role_path = tmp_path / "roles.json"
    main(
        [
            "predict",
            "--run-dir",
            str(run),
            "--context",
            str(data_path),
            "--context-size",
            "45",
            "--output",
            str(out_path),
            "--role-report",
            str(role_path),
            "--device",
            "cpu",
        ]
    )
    preds = pq.read_table(out_path)
    assert preds.num_rows == 15
    assert set(preds.column_names) == {
        "patient_id",
        "risk_control",
        "risk_treated",
        "risk_difference",
        "risk_control_sd",
        "risk_treated_sd",
        "propensity_score",
        "outside_overlap",
    }
    roles = json.loads(role_path.read_text())
    assert 20 <= len(roles) <= 25 and math.isfinite(roles[0]["p_confounder"])

    # Out-of-fold cross-fitted zero-shot prediction for all N rows (--cv-folds 3)
    cv_out = tmp_path / "preds_cv.parquet"
    cv_roles = tmp_path / "roles_cv.json"
    main(
        [
            "predict",
            "--run-dir",
            str(run),
            "--context",
            str(data_path),
            "--cv-folds",
            "3",
            "--query-batch-size",
            "16",
            "--output",
            str(cv_out),
            "--role-report",
            str(cv_roles),
            "--device",
            "cpu",
        ]
    )
    cv_preds = pq.read_table(cv_out)
    assert cv_preds.num_rows == 60
    assert cv_preds["patient_id"].to_pylist() == list(range(60))
    cv_rep = json.loads(cv_roles.read_text())
    assert 20 <= len(cv_rep) <= 25 and math.isfinite(cv_rep[0]["p_confounder"])
    assert "p_any_confounder" in cv_rep[0] and "p_any_modifier" in cv_rep[0]
    summary = format_role_summary(cv_rep, top_k=5)
    assert "Top 5 Suspected Confounders" in summary and "Top 5 Suspected Effect Modifiers" in summary

    # CSV role report export + Normalized Saerens EM sparsity calibration check
    csv_out = tmp_path / "preds_csv.parquet"
    csv_roles = tmp_path / "roles.csv"
    main(
        [
            "predict",
            "--run-dir",
            str(run),
            "--context",
            str(data_path),
            "--context-size",
            "45",
            "--output",
            str(csv_out),
            "--role-report",
            str(csv_roles),
            "--role-summary-top-k",
            "5",
            "--device",
            "cpu",
        ]
    )
    csv_lines = csv_roles.read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) >= 21 and "p_any_confounder" in csv_lines[0]

    # Calibrated probabilities shrink extreme-P noise while preserving exact rank order
    rng = np.random.default_rng(11)
    syn_logits = np.zeros((500, len(ROLE_NAMES)), dtype=np.float64)
    # Note: logsumexp of 5 zero non-noise logits is log(5) ~= 1.61, so noise has l0 > 1.61
    syn_logits[:, 0] = rng.normal(2.6, 0.6, size=500)
    syn_logits[:10, 3] = 7.0  # 10 strong confounders out of 500
    probs_cal, pi_hat, _ = calibrate_role_probabilities(syn_logits)
    assert probs_cal.shape == (500, len(ROLE_NAMES))
    np.testing.assert_allclose(probs_cal.sum(axis=1), 1.0, atol=1e-6)
    assert 0.005 < pi_hat < 0.08
    assert (1.0 - probs_cal[:10, 0]).mean() > 0.75
    raw_ps = 1.0 - torch.softmax(torch.from_numpy(syn_logits), dim=-1)[:, 0].numpy()
    cal_ps = 1.0 - probs_cal[:, 0]
    assert cal_ps[10:].mean() < 0.05 and cal_ps[10:].mean() < 0.25 * raw_ps[10:].mean()
    # Rank order of p_signal must match raw signal log-odds monotonically
    assert np.corrcoef(np.argsort(np.argsort(raw_ps)), np.argsort(np.argsort(cal_ps)))[0, 1] > 0.9999


def test_v2_decoupled_rlearner_propensity_and_v1_compat():
    # 1. Verify v1 config dicts without decouple_rlearner_propensity still load with default False
    v1_dict = {
        "col_dim": 32,
        "study_width": 32,
        "study_layers": 1,
        "col_attn_layers": 1,
        "heads": 4,
        "bins": 17,
        "top_k_nuis": 16,
        "top_k_cate": 12,
        "attn_slots": 8,
    }
    cfg_v1 = TabularModelConfig(**v1_dict)
    assert cfg_v1.decouple_rlearner_propensity is False
    m_v1 = SparseTabularCausalPFN(cfg_v1)
    assert not hasattr(m_v1, "log_lam_er")

    # 2. Verify v2 decoupled model forward/backward + ZERO prop_loss gradient into g_er / log_lam_er
    torch.manual_seed(13)
    cfg_v2 = TabularModelConfig(
        col_dim=36,
        study_width=36,
        study_layers=2,
        col_attn_layers=2,
        heads=6,
        bins=17,
        top_k_nuis=20,
        top_k_cate=16,
        attn_slots=10,
        decouple_rlearner_propensity=True,
    )
    m_v2 = SparseTabularCausalPFN(cfg_v2)
    ep = sample_tabular_study(505, n_total=90, num_features=60)
    Xc = torch.from_numpy(ep["X"][:70]); Mc = torch.from_numpy(ep["M"][:70])
    tc = torch.from_numpy(ep["t"][:70]); yc = torch.from_numpy(ep["y"][:70])
    Xq = torch.from_numpy(ep["X"][70:]); Mq = torch.from_numpy(ep["M"][70:])
    ora = torch.from_numpy(np.stack([ep["true_y0"][70:], ep["true_y1"][70:]], axis=1))
    tp = torch.from_numpy(ep["true_propensity"][70:])
    roles = torch.from_numpy(ep["roles"])

    # Propensity loss alone must NOT send any gradient to log_lam_er or g_er gate output rows [2:4]
    ol, pl, sc, mem = m_v2(Xc, tc, yc, Xq, M_ctx=Mc, M_q=Mq)
    prop_only = torch.nn.functional.binary_cross_entropy_with_logits(pl, tp)
    prop_only.backward()
    assert m_v2.log_lam_er.grad is None
    assert m_v2.nuis_gate_head[-1].weight.grad[2:4].abs().sum().item() == 0.0
    assert m_v2.log_lam_e.grad is not None and m_v2.log_lam_e.grad.abs().sum().item() > 0.0

    # Full loss must give finite non-zero gradients to 100% of v2 parameters (including log_lam_er!)
    m_v2.zero_grad(set_to_none=True)
    ol, pl, sc, mem = m_v2(Xc, tc, yc, Xq, M_ctx=Mc, M_q=Mq)
    loss, _ = m_v2.loss_components(
        ol,
        pl,
        sc,
        ora,
        tp,
        role_logits=mem.role_logits,
        nuis_screen_logits=mem.nuis_screen_logits,
        mod_screen_logits=mem.mod_screen_logits,
        true_roles=roles,
    )
    loss.backward()
    grads = [p.grad for p in m_v2.parameters() if p.grad is not None]
    assert len(grads) == sum(1 for _ in m_v2.parameters())
    assert all(torch.isfinite(g).all() for g in grads)
    assert m_v2.log_lam_er.grad.abs().sum().item() > 0.0

    # 3. Simultaneous row + column permutation invariance on v2
    m_v2.eval()
    rng = np.random.default_rng(19)
    rp = torch.from_numpy(rng.permutation(70))
    cp = torch.from_numpy(rng.permutation(60))
    with torch.no_grad():
        o1, p1, s1, mem1 = m_v2(Xc, tc, yc, Xq, M_ctx=Mc, M_q=Mq)
        m1, _ = m_v2.risk_stats(o1, s1)
        o2, p2, s2, mem2 = m_v2(
            Xc[rp][:, cp], tc[rp], yc[rp], Xq[:, cp], M_ctx=Mc[rp][:, cp], M_q=Mq[:, cp]
        )
        m2, _ = m_v2.risk_stats(o2, s2)
        torch.testing.assert_close(m1, m2, atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(p1.sigmoid(), p2.sigmoid(), atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(mem1.role_logits[cp], mem2.role_logits, atol=5e-4, rtol=5e-4)


def test_v3_five_family_prior_basis_decoupling_and_invariance():
    # 1. Legacy 3-tuple family_weights still loads cleanly
    legacy_prior = PriorConfig(family_weights=(0.5, 0.25, 0.25))
    assert len(legacy_prior.family_weights) == len(DGP_FAMILIES)
    assert legacy_prior.family_weights[3:] == (0.0, 0.0)

    # 2. Verify new families (tree_subgroup, causal_scm) + fragmented dups + M-bias colliders
    ep_tree = sample_tabular_study(
        601,
        n_total=140,
        num_features=200,
        prior=PriorConfig(null_hte_prob=0.0, fragmented_dup_prob=1.0, missing_study_prob=1.0),
        family="tree_subgroup",
    )
    assert ep_tree["X"].shape == (140, 200)
    assert ep_tree["meta"]["family"] == "tree_subgroup"
    assert ep_tree["meta"]["n_frag_dups"] >= 1
    assert ep_tree["meta"]["n_mod"] + ep_tree["meta"]["n_dual"] >= 1
    assert np.all(np.isfinite(ep_tree["true_ite"]))

    ep_scm = sample_tabular_study(
        602,
        n_total=140,
        num_features=200,
        prior=PriorConfig(
            null_hte_prob=0.0,
            collider_prob=1.0,
            fragmented_dup_prob=1.0,
            missing_study_prob=1.0,
        ),
        family="causal_scm",
    )
    assert ep_scm["X"].shape == (140, 200)
    assert ep_scm["meta"]["family"] == "causal_scm"
    assert ep_scm["meta"]["n_colliders"] >= 1
    assert ep_scm["meta"]["n_frag_dups"] >= 1
    assert np.all(np.isfinite(ep_scm["true_ite"]))

    # 3. v3 model forward/backward, ZERO prop_loss gradient into g_er[4:8] & log_lam_er, 100% grad coverage
    torch.manual_seed(29)
    cfg_v3 = TabularModelConfig(
        col_dim=32,
        study_width=32,
        study_layers=2,
        col_attn_layers=2,
        heads=4,
        bins=17,
        top_k_nuis=24,
        top_k_cate=16,
        attn_slots=10,
        ridge_grid_size=5,
        decouple_rlearner_propensity=True,
        use_v3_solver_basis=True,
    )
    m_v3 = SparseTabularCausalPFN(cfg_v3)
    Xc = torch.from_numpy(ep_scm["X"][:100]); Mc = torch.from_numpy(ep_scm["M"][:100])
    tc = torch.from_numpy(ep_scm["t"][:100]); yc = torch.from_numpy(ep_scm["y"][:100])
    Xq = torch.from_numpy(ep_scm["X"][100:]); Mq = torch.from_numpy(ep_scm["M"][100:])
    ora = torch.from_numpy(np.stack([ep_scm["true_y0"][100:], ep_scm["true_y1"][100:]], axis=1))
    tp = torch.from_numpy(ep_scm["true_propensity"][100:])
    roles = torch.from_numpy(ep_scm["roles"])

    ol, pl, sc, mem = m_v3(Xc, tc, yc, Xq, M_ctx=Mc, M_q=Mq)
    prop_only = torch.nn.functional.binary_cross_entropy_with_logits(pl, tp)
    prop_only.backward()
    assert m_v3.log_lam_er.grad is None
    assert m_v3.nuis_gate_head[-1].weight.grad[4:8].abs().sum().item() == 0.0
    assert m_v3.log_lam_e.grad is not None and m_v3.log_lam_e.grad.abs().sum().item() > 0.0

    m_v3.zero_grad(set_to_none=True)
    ol, pl, sc, mem = m_v3(Xc, tc, yc, Xq, M_ctx=Mc, M_q=Mq)
    loss, _ = m_v3.loss_components(
        ol,
        pl,
        sc,
        ora,
        tp,
        role_logits=mem.role_logits,
        nuis_screen_logits=mem.nuis_screen_logits,
        mod_screen_logits=mem.mod_screen_logits,
        true_roles=roles,
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m_v3.parameters() if p.grad is not None]
    assert len(grads) == sum(1 for _ in m_v3.parameters())
    assert all(torch.isfinite(g).all() for g in grads)
    assert m_v3.log_lam_er.grad.abs().sum().item() > 0.0

    # 4. Exact simultaneous row + column permutation invariance on v3 (including missingness & co-obs corr!)
    m_v3.eval()
    ep_p = sample_tabular_study(
        603,
        n_total=90,
        num_features=50,
        prior=PriorConfig(fragmented_dup_prob=1.0, missing_study_prob=1.0),
        family="tree_subgroup",
    )
    Xc2 = torch.from_numpy(ep_p["X"][:70]); Mc2 = torch.from_numpy(ep_p["M"][:70])
    tc2 = torch.from_numpy(ep_p["t"][:70]); yc2 = torch.from_numpy(ep_p["y"][:70])
    Xq2 = torch.from_numpy(ep_p["X"][70:]); Mq2 = torch.from_numpy(ep_p["M"][70:])
    rng = np.random.default_rng(31)
    rp = torch.from_numpy(rng.permutation(70))
    cp = torch.from_numpy(rng.permutation(50))
    with torch.no_grad():
        o1, p1, s1, mem1 = m_v3(Xc2, tc2, yc2, Xq2, M_ctx=Mc2, M_q=Mq2)
        m1, sd1 = m_v3.risk_stats(o1, s1)
        o2, p2, s2, mem2 = m_v3(
            Xc2[rp][:, cp], tc2[rp], yc2[rp], Xq2[:, cp], M_ctx=Mc2[rp][:, cp], M_q=Mq2[:, cp]
        )
        m2, sd2 = m_v3.risk_stats(o2, s2)
        torch.testing.assert_close(m1, m2, atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(sd1, sd2, atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(p1.sigmoid(), p2.sigmoid(), atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(mem1.role_logits[cp], mem2.role_logits, atol=5e-4, rtol=5e-4)
        rep = m_v3.column_role_report(mem1)
        assert len(rep) == 50 and math.isfinite(rep[0]["gate_propensity"])

    # 5. Verify no missingness-rate shortcut in the de-aliased prior (noise spans >55% missingness too)
    hi_noise_fracs = []
    for s in range(12):
        ep_m = sample_tabular_study(700 + s, n_total=200, num_features=200, prior=PriorConfig())
        mr = ep_m["M"].mean(axis=0)
        noise = ep_m["roles"] == 0
        hi_noise_fracs.append(float((mr[noise] > 0.55).mean()))
    assert np.mean(hi_noise_fracs) > 0.10

    # 6. Verify --init-from warm-start loads weights with a fresh step=0 schedule
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        src_dir = Path(td) / "src"
        dst_dir = Path(td) / "dst"
        main(
            [
                "train", "--run-dir", str(src_dir), "--steps", "1",
                "--min-context-size", "40", "--context-size", "50", "--query-size", "12",
                "--min-features", "20", "--max-features", "30",
                "--col-dim", "32", "--study-width", "32", "--study-layers", "1",
                "--col-attn-layers", "1", "--heads", "4", "--top-k-nuis", "16",
                "--top-k-cate", "12", "--attn-slots", "8", "--bins", "17",
                "--decouple-rlearner-propensity", "--use-v3-solver-basis",
                "--grad-accum-steps", "1", "--eval-every", "0", "--save-every", "1",
                "--device", "cpu",
            ]
        )
        main(
            [
                "train", "--run-dir", str(dst_dir), "--init-from", str(src_dir), "--steps", "1",
                "--min-context-size", "40", "--context-size", "50", "--query-size", "12",
                "--min-features", "20", "--max-features", "30",
                "--grad-accum-steps", "1", "--eval-every", "0", "--save-every", "1",
                "--device", "cpu",
            ]
        )
        m_loaded, sv = load_tabular_run(dst_dir, "cpu", prefer_best=False)
        assert sv["step"] == 1 and m_loaded.cfg.use_v3_solver_basis is True
        assert sv["settings"]["init_from_step"] == 1
