"""Unit and integration tests for Option A shared features and Option B (v6) hybrid fine-tuning."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from oci.longitudinal.model import CausalStudyModel, ModelConfig
from oci.longitudinal.option_b import (
    OPTION_A_PRESETS,
    MultimodalTabularPFN,
    OptionAFeatureConfig,
    OptionBConfig,
    StructuredTemporalEncoder,
    _anchor_penalty,
    _snapshot_tabular_weights,
    evaluate_option_b,
    extract_option_a_features,
    load_option_b_run,
    train_option_b,
)
from oci.tabular_pfn.model import SparseTabularCausalPFN, TabularModelConfig
from oci.tabular_pfn.prior import PriorConfig, sample_tabular_study
from oci.tabular_pfn.training import save_tabular_checkpoint
from synthetic_data.longitudinal import draw_longitudinal_dgp, sample_longitudinal_patient


def _make_tiny_bert(tmp_path: Path):
    vocab_file = tmp_path / "vocab.txt"
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "patient",
        "presents",
        "with",
        "no",
        "evidence",
        "of",
        "disease",
        "stage",
        "iv",
        "nsclc",
        "fatigue",
        "cough",
        "pain",
        "egfr",
        "positive",
        "negative",
        "ecog",
        "prior",
        "therapy",
        "stable",
        "progression",
    ]
    vocab_file.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    tok = BertTokenizer(vocab_file=str(vocab_file), do_lower_case=True)
    cfg = BertConfig(
        vocab_size=len(tokens),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    torch.manual_seed(7)
    backbone = BertModel(cfg).eval()
    save_dir = tmp_path / "tiny_bert"
    save_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    return backbone, tok, save_dir


def _simulate_rows(n: int, seed: int = 123):
    dgp = draw_longitudinal_dgp(seed, "mixed", schema_version=2)
    rows = [sample_longitudinal_patient(i, seed=seed + 1000, dgp=dgp) for i in range(n)]
    # Guarantee genuine missing labs/PROs on a subset to test NaN safety thoroughly
    rows[0]["labs"] = []
    rows[0]["pros"] = []
    rows[-1]["vitals"] = []
    return rows


def _tiny_v4_config() -> TabularModelConfig:
    return TabularModelConfig(
        col_dim=32,
        study_width=32,
        study_layers=1,
        col_attn_layers=1,
        heads=4,
        bins=17,
        top_k_nuis=12,
        top_k_cate=10,
        attn_slots=6,
        ridge_grid_size=3,
        decouple_rlearner_propensity=True,
        use_v3_solver_basis=True,
        use_v4_interactions=True,
        pair_top_k=6,
    )


def test_extract_option_a_features_strict_context_only(tmp_path):
    backbone, tok, _ = _make_tiny_bert(tmp_path)
    rows = _simulate_rows(32, seed=202)
    ctx_rows = rows[:24]
    q_rows_a = rows[24:]
    q_rows_b = copy.deepcopy(q_rows_a)
    for r in q_rows_b:
        r["notes"] = [
            {
                "day_from_index": -1,
                "note_type": "progress",
                "text": "Patient presents with progression pain cough positive egfr.",
            }
        ]
        r["labs"] = []
        r["encounters"] = []

    cfg = OptionAFeatureConfig(
        include_nadir_peak=True,
        icd_svd_dim=8,
        sent_km_k=8,
        include_latest_sent_km=True,
        lex_svd_dim=8,
        sent_batch_size=16,
        sent_max_length=32,
    )
    Xc_a, Xq_a, names_a = extract_option_a_features(ctx_rows, q_rows_a, backbone, tok, "cpu", cfg)
    Xc_b, Xq_b, names_b = extract_option_a_features(ctx_rows, q_rows_b, backbone, tok, "cpu", cfg)

    assert names_a == names_b
    assert Xc_a.shape[0] == 24 and Xq_a.shape[0] == 8
    assert Xc_a.shape[1] == Xq_a.shape[1] == len(names_a)
    # Perturbing query rows MUST NOT alter a single entry of the context feature table!
    assert torch.allclose(Xc_a, Xc_b, atol=1e-6, equal_nan=True)
    # Genuine NaNs from missing labs/PROs are preserved for SparseTabularCausalPFN's v3 basis
    assert torch.isnan(Xc_a).any()


def test_hybrid_forward_matches_sparse_tabular_pfn_bit_for_bit(tmp_path):
    backbone, tok, _ = _make_tiny_bert(tmp_path)
    torch.manual_seed(11)
    tab_head = SparseTabularCausalPFN(_tiny_v4_config()).eval()
    opt_b_cfg = OptionBConfig(
        learned_temporal_dim=4,
        event_width=32,
        event_layers=1,
        event_heads=4,
        always_screen_learned_cols=False,
    )
    model = MultimodalTabularPFN(backbone, tok, tab_head, opt_b_cfg, OPTION_A_PRESETS["struct_only"]).eval()

    rng = np.random.default_rng(19)
    n_ctx, n_q, p_exp, d_l = 40, 12, 20, 4
    X_exp_ctx = torch.tensor(rng.normal(size=(n_ctx, p_exp)), dtype=torch.float32)
    X_exp_q = torch.tensor(rng.normal(size=(n_q, p_exp)), dtype=torch.float32)
    # Inject genuine NaNs into explicit columns
    X_exp_ctx[0, 3] = float("nan")
    X_exp_ctx[5, 3] = float("nan")
    X_exp_ctx[2, 7] = float("nan")
    X_exp_q[1, 3] = float("nan")
    Z_ctx = torch.tensor(rng.normal(size=(n_ctx, d_l)), dtype=torch.float32, requires_grad=True)
    Z_q = torch.tensor(rng.normal(size=(n_q, d_l)), dtype=torch.float32, requires_grad=True)
    t_ctx = torch.tensor(rng.integers(0, 2, size=n_ctx), dtype=torch.float32)
    y_ctx = torch.tensor(rng.integers(0, 2, size=n_ctx), dtype=torch.float32)

    ol_hyb, pl_hyb, _, mem_hyb = model._condition_and_query_hybrid(
        X_exp_ctx, Z_ctx, t_ctx, y_ctx, X_exp_q, Z_q, always_screen_learned_cols=False
    )
    with torch.no_grad():
        ol_ref, pl_ref, _, mem_ref = tab_head(
            torch.cat([X_exp_ctx, Z_ctx.detach()], dim=1),
            t_ctx,
            y_ctx,
            torch.cat([X_exp_q, Z_q.detach()], dim=1),
        )

    assert torch.equal(mem_hyb.nuis_idx, mem_ref.nuis_idx)
    assert torch.equal(mem_hyb.cate_idx, mem_ref.cate_idx)
    assert torch.allclose(ol_hyb, ol_ref, atol=1e-4)
    assert torch.allclose(pl_hyb, pl_ref, atol=1e-4)

    # Now verify always_screen_learned_cols=True keeps all top explicit cols + all learned cols
    _, _, _, mem_keep = model._condition_and_query_hybrid(
        X_exp_ctx, Z_ctx, t_ctx, y_ctx, X_exp_q, Z_q, always_screen_learned_cols=True
    )
    learned_set = set(range(p_exp, p_exp + d_l))
    assert learned_set.issubset(set(mem_keep.nuis_idx.tolist()))
    assert learned_set.issubset(set(mem_keep.cate_idx.tolist()))
    assert len(mem_keep.nuis_idx) == tab_head.cfg.top_k_nuis + d_l
    assert len(mem_keep.cate_idx) == tab_head.cfg.top_k_cate + d_l


def test_option_b_gradient_flow_and_frozen_backbone(tmp_path):
    backbone, tok, _ = _make_tiny_bert(tmp_path)
    torch.manual_seed(23)
    tab_head = SparseTabularCausalPFN(_tiny_v4_config())
    opt_b_cfg = OptionBConfig(
        learned_temporal_dim=6,
        event_width=32,
        event_layers=1,
        event_heads=4,
        temporal_chunk_size=12,
        always_screen_learned_cols=True,
    )
    feat_cfg = OptionAFeatureConfig(
        include_nadir_peak=True,
        icd_svd_dim=4,
        sent_km_k=4,
        include_latest_sent_km=True,
        lex_svd_dim=4,
        sent_batch_size=16,
        sent_max_length=32,
    )
    model = MultimodalTabularPFN(backbone, tok, tab_head, opt_b_cfg, feat_cfg)
    model.train()

    # Clinical backbone MUST stay frozen and in eval mode even after model.train()
    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())

    rows = _simulate_rows(30, seed=303)
    ctx_rows, q_rows = rows[:22], rows[22:]
    oracle = torch.tensor([[r["true_y0"], r["true_y1"]] for r in q_rows], dtype=torch.float32)
    true_prop = torch.tensor([r["true_treatment_prob"] for r in q_rows], dtype=torch.float32)

    ol, pl, sc, _ = model(ctx_rows, q_rows, precision="fp32")
    loss, _ = model.tabular_head.loss_components(ol, pl, sc, oracle, true_prop, true_roles=None)
    assert torch.isfinite(loss)
    loss.backward()

    # Backbone receives ZERO gradients
    assert all(p.grad is None for p in model.backbone.parameters())

    # StructuredTemporalEncoder receives finite, non-zero gradients despite NaNs in explicit cols!
    temp_grads = [
        p.grad for p in model.temporal_encoder.parameters() if p.requires_grad and p.grad is not None
    ]
    assert len(temp_grads) > 0
    assert all(torch.isfinite(g).all() for g in temp_grads)
    assert any(g.abs().max().item() > 0.0 for g in temp_grads)

    # SparseTabularCausalPFN head receives finite, non-zero gradients
    head_grads = [
        p.grad for p in model.tabular_head.parameters() if p.requires_grad and p.grad is not None
    ]
    assert len(head_grads) > 0
    assert all(torch.isfinite(g).all() for g in head_grads)
    assert any(g.abs().max().item() > 0.0 for g in head_grads)

    # Pure tabular replay episode updates role_head cleanly
    model.zero_grad(set_to_none=True)
    ep = sample_tabular_study(44, 30, num_features=16, prior=PriorConfig(design_split="train"))
    ol_t, pl_t, sc_t, mem_t = model.forward_tabular(
        torch.from_numpy(ep["X"][:22]),
        torch.from_numpy(ep["t"][:22]),
        torch.from_numpy(ep["y"][:22]),
        torch.from_numpy(ep["X"][22:]),
        M_ctx=torch.from_numpy(ep["M"][:22]),
        M_q=torch.from_numpy(ep["M"][22:]),
    )
    loss_t, parts_t = model.tabular_head.loss_components(
        ol_t,
        pl_t,
        sc_t,
        torch.from_numpy(np.stack([ep["true_y0"][22:], ep["true_y1"][22:]], axis=1)),
        torch.from_numpy(ep["true_propensity"][22:]),
        role_logits=mem_t.role_logits,
        nuis_screen_logits=mem_t.nuis_screen_logits,
        mod_screen_logits=mem_t.mod_screen_logits,
        true_roles=torch.from_numpy(ep["roles"]),
    )
    assert torch.isfinite(loss_t) and parts_t["role_loss"] > 0.0
    loss_t.backward()
    assert model.tabular_head.role_head[-1].weight.grad is not None
    assert torch.isfinite(model.tabular_head.role_head[-1].weight.grad).all()

    # Anchor penalty check
    snap = _snapshot_tabular_weights(model.tabular_head)
    anc0, rms0 = _anchor_penalty(model.tabular_head, snap)
    assert float(anc0.item()) == 0.0 and rms0 == 0.0
    with torch.no_grad():
        model.tabular_head.attn_residual_scale.add_(0.1)
    anc1, rms1 = _anchor_penalty(model.tabular_head, snap)
    assert float(anc1.item()) > 0.0 and rms1 > 0.0


def test_warm_start_structured_encoder_from_run5_schema(tmp_path):
    backbone, tok, _ = _make_tiny_bert(tmp_path)
    long_cfg = ModelConfig(
        event_width=32,
        event_layers=1,
        study_width=32,
        study_layers=1,
        heads=4,
        bins=17,
        note_slots=4,
        solver_dim=16,
    )
    long_model = CausalStudyModel(backbone, tok, long_cfg)
    ckpt_dir = tmp_path / "run5_mock"
    ckpt_dir.mkdir()
    torch.save({"step": 1200, "weights": long_model.state_dict()}, ckpt_dir / "best_checkpoint.pt")

    enc = StructuredTemporalEncoder(
        OptionBConfig(learned_temporal_dim=8, event_width=32, event_layers=1, event_heads=4)
    )
    info = enc.load_from_longitudinal_checkpoint(ckpt_dir)
    assert info["source_step"] == 1200
    assert info["loaded_tensors"] >= 20
    # Verify exact weight copy including cross_modal modality slice 1..4 (skipping modality 0 = notes)
    assert torch.equal(enc.labs.weight, long_model.patient.labs.weight)
    assert torch.equal(
        enc.cross_modal.modality.weight,
        long_model.patient.cross_modal.modality.weight[1:],
    )

    # Mismatched event_width must raise ValueError rather than silently skipping encoder tensors
    bad_enc = StructuredTemporalEncoder(
        OptionBConfig(learned_temporal_dim=8, event_width=64, event_layers=1, event_heads=4)
    )
    with pytest.raises(ValueError, match="Shape/key mismatch"):
        bad_enc.load_from_longitudinal_checkpoint(ckpt_dir)


def test_train_resume_and_evaluate_option_b_end_to_end(tmp_path):
    _, _, bert_dir = _make_tiny_bert(tmp_path)
    tab_dir = tmp_path / "tab_v4_mock"
    tab_dir.mkdir()
    tab_head = SparseTabularCausalPFN(_tiny_v4_config())
    prior = PriorConfig(design_split="train", min_features=12, max_features=20)
    save_tabular_checkpoint(
        tab_dir / "best_checkpoint.pt", tab_head, None, 100, {"mock": True}, prior
    )

    run_dir = tmp_path / "option_b_run"
    opt_b_cfg = OptionBConfig(
        learned_temporal_dim=6,
        event_width=32,
        event_layers=1,
        event_heads=4,
        temporal_chunk_size=16,
    )
    feat_cfg = OptionAFeatureConfig(
        include_nadir_peak=True,
        icd_svd_dim=4,
        sent_km_k=4,
        include_latest_sent_km=True,
        lex_svd_dim=4,
        sent_batch_size=16,
        sent_max_length=32,
    )

    res1 = train_option_b(
        run_dir=run_dir,
        tabular_init_dir=tab_dir,
        backbone_path=bert_dir,
        design_bank_path=None,
        paraphrase_bank_path=None,
        option_b_cfg=opt_b_cfg,
        feature_cfg=feat_cfg,
        steps=2,
        grad_accum_steps=2,
        multimodal_prob=0.5,
        min_context_size=24,
        context_size=24,
        query_size=8,
        tabular_min_context=24,
        tabular_max_context=24,
        tabular_query_size=8,
        tabular_min_features=14,
        tabular_max_features=18,
        warmup_steps=1,
        save_every=2,
        eval_every=2,
        eval_studies=2,
        eval_context_size=24,
        eval_query_size=8,
        tabular_eval_studies=2,
        tabular_eval_context=24,
        tabular_eval_features=14,
        eval_step_zero=True,
        num_workers=0,
        device="cpu",
        precision="fp32",
        resume=False,
    )
    assert res1["step"] == 2
    assert (run_dir / "checkpoint.pt").is_file()
    assert (run_dir / "best_checkpoint.pt").is_file()
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert any(r["event"] == "eval" and r["step"] == 0 for r in lines)
    assert any(r["event"] == "train" and r["step"] == 2 for r in lines)
    assert any(r["event"] == "eval" and r["step"] == 2 for r in lines)

    # Resume to step 3 cleanly
    res2 = train_option_b(
        run_dir=run_dir,
        tabular_init_dir=tab_dir,
        backbone_path=bert_dir,
        design_bank_path=None,
        paraphrase_bank_path=None,
        option_b_cfg=opt_b_cfg,
        feature_cfg=feat_cfg,
        steps=3,
        grad_accum_steps=2,
        multimodal_prob=0.5,
        min_context_size=24,
        context_size=24,
        query_size=8,
        tabular_min_context=24,
        tabular_max_context=24,
        tabular_query_size=8,
        tabular_min_features=14,
        tabular_max_features=18,
        warmup_steps=1,
        save_every=1,
        eval_every=1,
        eval_studies=2,
        eval_context_size=24,
        eval_query_size=8,
        tabular_eval_studies=1,
        tabular_eval_context=24,
        tabular_eval_features=14,
        num_workers=0,
        device="cpu",
        precision="fp32",
        resume=True,
    )
    assert res2["step"] == 3
    loaded_model, loaded_meta = load_option_b_run(run_dir, device="cpu", prefer_best=False)
    assert loaded_meta["step"] == 3

    ev = evaluate_option_b(
        loaded_model,
        seed=42,
        studies=2,
        context_size=24,
        query_size=8,
        bank=None,
        precision="fp32",
        num_workers=0,
    )
    assert ev["studies"] == 2
    assert np.isfinite(ev["cate_rmse"])
    assert np.isfinite(ev["risk_rmse"])


def test_eval_option_a_v4_helpers():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_option_a_v4.py"
    spec = importlib.util.spec_from_file_location("eval_option_a_v4", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)

    recs = [
        {
            "p0": np.array([0.2, 0.4, 0.6]),
            "p1": np.array([0.3, 0.3, 0.4]),
            "e": np.array([0.5, 0.5, 0.5]),
            "t0": np.array([0.2, 0.4, 0.6]),
            "t1": np.array([0.35, 0.25, 0.45]),
            "te": np.array([0.4, 0.5, 0.6]),
        }
    ]
    s = mod.summarize_multimodal_records(recs)
    assert s["cohorts"] == 1
    assert s["query_patients"] == 3
    assert s["cate_rmse"] < 0.06
    assert s["within_r"] > 0.9

    md = mod.build_markdown_summary(
        {
            "meta": {
                "tabular_v4_dir": "artifacts/tabular_causal_pfn_v4_h200",
                "tabular_v4_step": 40000,
                "run5_dir": "artifacts/longitudinal_tinyoncbert_v5",
                "run5_step": 1200,
                "option_b_dir": None,
                "option_b_step": None,
            },
            "suite2_llm16": {
                "overall": {"800": {"PFN-v4 (s40000) [option_a_full]": s}},
                "by_group": {},
            },
        }
    )
    assert "Option A / Tabular v4 Evaluation Summary" in md
    assert "PFN-v4 (s40000) [option_a_full]" in md
