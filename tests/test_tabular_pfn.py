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
