"""Unit and integration tests for MMAI v23 disjoint long-note corpus & Option B (v7) 3-way mixing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from oci.longitudinal.events import note_sentences, note_sentences_balanced
from oci.longitudinal.mmai_v23 import (
    CANCER_GROUPS,
    TIMELINE_META_DIM,
    V23_DGP_MODES,
    MMAIv23EpisodeStore,
    evaluate_on_v23,
    extract_note_timeline_features,
    featurize_v23_patient,
    fit_v23_factor_vocab,
    load_v23_extraction_records,
    parse_demographics_from_notes,
    prepare_v23_cache,
    project_cached_sent_km,
    resolve_v23_splits,
)
from oci.longitudinal.option_b import (
    OptionAFeatureConfig,
    OptionBConfig,
    load_option_b_run,
    train_option_b,
)
from oci.tabular_pfn.model import SparseTabularCausalPFN, TabularModelConfig
from oci.tabular_pfn.prior import PriorConfig
from oci.tabular_pfn.training import save_tabular_checkpoint


def _make_tiny_bert(tmp_path: Path) -> Path:
    vocab_file = tmp_path / "vocab.txt"
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "patient",
        "age",
        "sex",
        "female",
        "male",
        "stage",
        "iv",
        "ii",
        "lung",
        "breast",
        "colon",
        "prostate",
        "adenocarcinoma",
        "squamous",
        "egfr",
        "kras",
        "tp53",
        "her2",
        "pdl1",
        "positive",
        "negative",
        "progression",
        "remission",
        "carboplatin",
        "pembrolizumab",
        "paclitaxel",
        "osimertinib",
        "impression",
        "ngs",
        "stable",
        "metastatic",
        "biopsy",
        "resection",
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
    torch.manual_seed(17)
    backbone = BertModel(cfg).eval()
    save_dir = tmp_path / "tiny_v23_bert"
    save_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    return save_dir


def _tiny_v5_config() -> TabularModelConfig:
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


def _build_mock_v23_corpus(tmp_path: Path, n_train_seen: int = 12, n_train_unseen: int = 36, n_val: int = 24, n_test: int = 24):
    """Create a realistic miniature MMAI v23 corpus (notes parquet, meta parquet, checkpoint JSONs)."""
    rng = np.random.default_rng(1234)
    cancers = [
        ("Lung, right upper lobe", "Adenocarcinoma", ["EGFR", "KRAS", "TP53", "PD-L1"], ["carboplatin", "pembrolizumab"]),
        ("Breast, left upper outer", "Invasive ductal carcinoma", ["HER2", "ER", "PR", "TP53"], ["paclitaxel", "trastuzumab"]),
        ("Colon, sigmoid", "Adenocarcinoma", ["KRAS", "BRAF", "MSI", "TP53"], ["oxaliplatin", "fluorouracil"]),
        ("Prostate gland", "Acinar adenocarcinoma", ["BRCA2", "TP53", "PTEN", "AR"], ["enzalutamide", "docetaxel"]),
    ]

    note_rows = []
    meta_rows = []
    ckpt_dir = tmp_path / "extraction_records" / ".n5000_all_nemotron.checkpoints" / "patient_records"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ordered_specs = (
        [("train", f"PAT_SEEN_{i:03d}") for i in range(n_train_seen)]
        + [("validation", f"PAT_VAL_{i:03d}") for i in range(n_val)]
        + [("train", f"PAT_UNSEEN_{i:03d}") for i in range(n_train_unseen)]
        + [("test", f"PAT_TEST_{i:03d}") for i in range(n_test)]
    )

    cutoff_notes = 0
    for idx, (split, pid) in enumerate(ordered_specs):
        meta_rows.append({"patient_id": pid, "split": split})
        site, hist, bio_pool, drug_pool = cancers[idx % len(cancers)]
        age = int(rng.integers(42, 84))
        sex = "Female" if rng.random() < 0.5 else "Male"
        stage = "IV" if rng.random() < 0.55 else "II"
        high_grade = bool(rng.random() < 0.45)
        pos_bios = [b for b in bio_pool if rng.random() < 0.45]
        if not pos_bios:
            pos_bios = [bio_pool[0]]
        had_prog = bool(rng.random() < 0.50)
        had_rem = bool(rng.random() < 0.35)
        n_notes = 5

        for n_i in range(n_notes):
            etype = ("clinical_note", "pathology_report", "imaging_report", "ngs_report", "clinical_note")[n_i]
            bio_txt = " ".join(f"{b} positive" for b in pos_bios)
            drug_txt = " ".join(drug_pool)
            prog_txt = "Progression noted on imaging." if had_prog else "Disease remains stable."
            text = (
                f"Age: {age} years. Sex: {sex}. Primary site {site} {hist} Stage {stage}.\n"
                f"NGS biomarkers: {bio_txt}. Prior therapy: {drug_txt}.\n"
                f"Impression: {prog_txt}"
            )
            note_rows.append(
                {
                    "patient_id": pid,
                    "split": split,
                    "event_number": n_i + 1,
                    "event_type": etype,
                    "date": f"2025-0{n_i + 1}-15",
                    "synthetic_note": text,
                }
            )
        # Cutoff right after all PAT_SEEN + half of PAT_VAL notes!
        if split == "train" and pid == f"PAT_SEEN_{n_train_seen - 1:03d}":
            cutoff_notes = len(note_rows)

        # Write nested mmai-clinical-extraction-v4 checkpoint JSON
        state = {
            "diagnoses": [
                {
                    "fields": {
                        "primary_site": site,
                        "histology": hist,
                        "stage_group": f"Stage {stage}",
                        "extent_at_diagnosis": "distant metastatic" if stage == "IV" else "localized",
                        "grade": "high grade" if high_grade else "well differentiated",
                    }
                }
            ],
            "biomarkers": [
                {
                    "fields": {
                        "biomarker": b,
                        "result": "positive" if b in pos_bios else "negative",
                        "method": "NGS" if j % 2 == 0 else "IHC",
                    }
                }
                for j, b in enumerate(bio_pool)
            ],
            "treatments": [
                {
                    "fields": {
                        "treatment_type": "systemic_therapy",
                        "regimen": " + ".join(drug_pool),
                        "agents": drug_pool,
                        "line_of_therapy": 1 + (idx % 3),
                    }
                }
            ],
            "events": (
                [{"fields": {"event_type": "progression", "description": "CT progression"}}] if had_prog else []
            )
            + (
                [{"fields": {"event_type": "remission", "description": "partial remission"}}] if had_rem else []
            ),
            "procedures": [{"fields": {"procedure_group": "biopsy" if idx % 2 == 0 else "resection"}}],
        }
        ckpt_obj = {
            "patient_id": pid,
            "status": "complete",
            "chunk_count": 2,
            "next_chunk": 2,
            "state": state,
        }
        (ckpt_dir / f"{pid}.json").write_text(json.dumps(ckpt_obj), encoding="utf-8")

    notes_pq = tmp_path / "all_synthetic_notes.parquet"
    meta_pq = tmp_path / "synthetic_patient_metadata.parquet"
    pq.write_table(pa.Table.from_pylist(note_rows), notes_pq, row_group_size=40)
    pq.write_table(pa.Table.from_pylist(meta_rows), meta_pq)

    manifest_path = tmp_path / "tinyoncbert_manifest.json"
    manifest_path.write_text(json.dumps({"notes_read": cutoff_notes}), encoding="utf-8")
    return notes_pq, meta_pq, tmp_path / "extraction_records", manifest_path, cutoff_notes


def test_note_sentences_balanced_preserves_all_notes_head_and_tail():
    notes = []
    for n_idx in range(14):
        clauses = [f"Note{n_idx:02d} header line {k} with clinical detail" for k in range(20)]
        clauses.append(f"Note{n_idx:02d} final impression biomarker positive")
        notes.append({"text": ". ".join(clauses) + "."})

    # Legacy chronological cap drops late notes completely once 128 clauses are filled
    legacy = note_sentences(notes, max_sentences=128)
    assert len(legacy) == 128
    assert not any("Note13" in s for s in legacy)

    # Balanced round-robin tail/head keeps EVERY note's impression AND header under a tighter cap!
    balanced = note_sentences_balanced(notes, max_sentences=56, min_chars=12)
    assert len(balanced) == 56
    assert len(set(balanced)) == len(balanced)
    for n_idx in range(14):
        assert any(f"Note{n_idx:02d} final impression" in s for s in balanced), f"Missing tail of note {n_idx}"
        assert any(f"Note{n_idx:02d} header line 0" in s for s in balanced), f"Missing head of note {n_idx}"

    assert note_sentences_balanced([], max_sentences=32) == []


def test_v23_disjoint_split_cutoff_excludes_seen_and_boundary_patients(tmp_path):
    # Include a boundary train patient (PAT_BOUNDARY) whose notes straddle cutoff!
    rows = [
        {"patient_id": "P_SEEN_1", "split": "train", "synthetic_note": "a"},
        {"patient_id": "P_VAL_1", "split": "validation", "synthetic_note": "b"},
        {"patient_id": "P_BOUNDARY", "split": "train", "synthetic_note": "c"},  # row 2 (< cutoff=3)
        {"patient_id": "P_BOUNDARY", "split": "train", "synthetic_note": "d"},  # row 3 (>= cutoff=3)
        {"patient_id": "P_UNSEEN_1", "split": "train", "synthetic_note": "e"},
        {"patient_id": "P_UNSEEN_1", "split": "train", "synthetic_note": "f"},
        {"patient_id": "P_TEST_1", "split": "test", "synthetic_note": "g"},
    ]
    notes_pq = tmp_path / "notes.parquet"
    pq.write_table(pa.Table.from_pylist(rows), notes_pq)

    info = resolve_v23_splits(notes_pq, None, tinyoncbert_notes_cutoff=3)
    sp = info["splits"]
    assert sp["train_seen"] == {"P_SEEN_1", "P_BOUNDARY"}
    assert sp["train_unseen"] == {"P_UNSEEN_1"}
    assert sp["validation"] == {"P_VAL_1"}
    assert sp["test"] == {"P_TEST_1"}
    assert info["tinyoncbert_train_notes_before_cutoff"] == 2
    assert sp["train_unseen"].isdisjoint(sp["train_seen"])
    assert sp["train_unseen"].isdisjoint(sp["validation"] | sp["test"])


def test_v23_dual_format_extraction_parser_and_train_only_vocab(tmp_path):
    # 1. Consolidated patient_records.parquet format (items[dom] flat dicts)
    rec_items = {
        "diagnoses": [{"primary_site": "Lung", "histology": "Adenocarcinoma", "stage_group": "Stage IV"}],
        "biomarkers": [
            {"biomarker": "EGFR", "result": "positive", "method": "NGS"},
            {"biomarker": "PDL1", "result": "positive", "method": "IHC"},
        ],
        "treatments": [{"treatment_type": "systemic", "agents": ["Osimertinib"], "line_of_therapy": 1}],
        "events": [{"event_type": "progression", "description": "new lesion"}],
        "procedures": [{"procedure_group": "biopsy"}],
    }
    pq_path = tmp_path / "patient_records.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"patient_id": "P1", "extraction_status": "complete", "record_json": json.dumps({"items": rec_items})},
                {"patient_id": "P2", "extraction_status": "partial", "record_json": json.dumps({"items": rec_items})},
            ]
        ),
        pq_path,
    )
    loaded_pq = load_v23_extraction_records(pq_path, allow_partial=False)
    assert set(loaded_pq.keys()) == {"P1"}
    loaded_pq_partial = load_v23_extraction_records(pq_path, allow_partial=True)
    assert set(loaded_pq_partial.keys()) == {"P1", "P2"}

    # 2. Live checkpoint JSON format (state[dom][i]["fields"])
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    nested = {dom: [{"fields": entry} for entry in lst] for dom, lst in rec_items.items()}
    (ckpt_dir / "P1.json").write_text(
        json.dumps({"patient_id": "P1", "status": "complete", "chunk_count": 3, "next_chunk": 3, "state": nested}),
        encoding="utf-8",
    )
    (ckpt_dir / "P2.json").write_text(
        json.dumps({"patient_id": "P2", "status": "partial", "chunk_count": 3, "next_chunk": 1, "state": nested}),
        encoding="utf-8",
    )
    loaded_ckpt = load_v23_extraction_records(ckpt_dir, allow_partial=False)
    assert loaded_ckpt == loaded_pq
    loaded_ckpt_partial = load_v23_extraction_records(ckpt_dir, allow_partial=True)
    assert set(loaded_ckpt_partial.keys()) == {"P1", "P2"}

    # 3. Train-only vocab strictly ignores unseen test-only biomarker + demographics parser works
    train_recs = {f"T{i}": loaded_pq["P1"] for i in range(6)}
    vocab = fit_v23_factor_vocab(train_recs, min_count=2)
    assert "EGFR" in vocab["biomarkers"]
    assert "PD-L1" in vocab["biomarkers"]  # PDL1 alias normalized to PD-L1
    assert "RARE_TEST_GENE" not in vocab["biomarkers"]

    age, sex = parse_demographics_from_notes([{"text": "Patient is a 68-year-old female with lung cancer."}])
    assert age == 68.0 and sex == 1.0
    vec, cg_id = featurize_v23_patient(loaded_pq["P1"], vocab, age_years=age, sex_female=sex)
    assert vec.shape == (len(vocab["factor_names"]),)
    assert CANCER_GROUPS[cg_id] == "lung"
    tl = extract_note_timeline_features(
        [{"text": "hello world", "event_type": "ngs_report", "date": "2025-01-10"}]
    )
    assert tl.shape == (TIMELINE_META_DIM,) and np.isfinite(tl).all()


def test_prepare_cache_sample_episode_and_query_blind_sent_km(tmp_path):
    bert_dir = _make_tiny_bert(tmp_path)
    notes_pq, meta_pq, ext_root, manifest_path, cutoff_notes = _build_mock_v23_corpus(tmp_path)
    cache_dir = tmp_path / "v23_cache"

    manifest = prepare_v23_cache(
        output_dir=cache_dir,
        notes_parquet=notes_pq,
        meta_parquet=meta_pq,
        extraction_source=ext_root,
        backbone_path=bert_dir,
        tinyoncbert_manifest=manifest_path,
        include_seen_train=False,
        max_sentences_per_patient=24,
        sent_batch_size=32,
        sent_max_length=32,
        flush_patients=16,
        device="cpu",
        num_io_threads=2,
    )

    assert manifest["disjoint_train_only"] is True
    assert manifest["tinyoncbert_notes_cutoff"] == cutoff_notes
    assert "train_seen" not in manifest["splits"]
    assert manifest["splits"]["train_unseen"]["num_patients"] == 36
    assert manifest["splits"]["validation"]["num_patients"] == 24
    assert manifest["splits"]["test"]["num_patients"] == 24

    store = MMAIv23EpisodeStore(cache_dir, split="train_unseen")
    assert all(pid.startswith("PAT_UNSEEN_") for pid in store.patient_ids)

    # Verify strict zero query leakage in project_cached_sent_km
    ctx_idx = np.arange(0, 20, dtype=np.int64)
    q_idx = np.arange(20, 28, dtype=np.int64)
    s_ctx_a, s_q_a = project_cached_sent_km(
        store.sent_embs, store.sent_offsets, ctx_idx, q_idx, sent_km_k=8, device=torch.device("cpu"), chunk_patients=6
    )
    mut_embs = np.array(store.sent_embs, copy=True)
    for qi in q_idx:
        mut_embs[store.sent_offsets[qi] : store.sent_offsets[qi + 1]] *= -1.0
    s_ctx_b, s_q_b = project_cached_sent_km(
        mut_embs, store.sent_offsets, ctx_idx, q_idx, sent_km_k=8, device=torch.device("cpu"), chunk_patients=6
    )
    assert torch.equal(s_ctx_a, s_ctx_b)
    assert not torch.allclose(s_q_a, s_q_b)

    # Verify all 4 DGP modes & both notes_only and hybrid modes produce strictly finite tensors
    for idx_m, mode in enumerate(V23_DGP_MODES):
        ep = store.sample_episode(
            seed=100 + 2 * idx_m,
            n_ctx=24,
            n_q=6,
            device="cpu",
            sent_km_k=8,
            notes_only_prob=1.0 if idx_m % 2 == 0 else 0.0,
            dgp_mode=mode,
        )
        assert ep["X_ctx"].shape[0] == 24 and ep["X_q"].shape[0] == 6
        assert ep["X_ctx"].shape[1] == ep["X_q"].shape[1]
        assert ep["meta"]["dgp_mode"] == mode
        if mode == "null_hte":
            assert ep["meta"]["n_mod"] == 0
            y0 = ep["true_y0_q"].numpy()
            y1 = ep["true_y1_q"].numpy()
            if ep["meta"]["scale"] == "logOR":
                lor = np.log(y1 / (1.0 - y1)) - np.log(y0 / (1.0 - y0))
                assert float(lor.std()) < 1e-4
            else:
                # On RD scale, unclipped patients have constant risk difference
                unclipped = (y1 > 0.02) & (y1 < 0.98)
                if unclipped.sum() >= 2:
                    assert float((y1[unclipped] - y0[unclipped]).std()) < 1e-4
        if ep["meta"]["obs_mode"] == "notes_only":
            assert ep["X_ctx"].shape[1] == 8 + TIMELINE_META_DIM
            assert torch.isfinite(ep["X_ctx"]).all()
        else:
            assert ep["X_ctx"].shape[1] > 8 + TIMELINE_META_DIM


def test_option_b_three_way_v23_training_resume_and_eval(tmp_path):
    bert_dir = _make_tiny_bert(tmp_path)
    notes_pq, meta_pq, ext_root, manifest_path, _ = _build_mock_v23_corpus(
        tmp_path, n_train_seen=8, n_train_unseen=32, n_val=24, n_test=24
    )
    cache_dir = tmp_path / "v23_cache"
    prepare_v23_cache(
        output_dir=cache_dir,
        notes_parquet=notes_pq,
        meta_parquet=meta_pq,
        extraction_source=ext_root,
        backbone_path=bert_dir,
        tinyoncbert_manifest=manifest_path,
        max_sentences_per_patient=16,
        sent_batch_size=32,
        sent_max_length=32,
        flush_patients=16,
        device="cpu",
        num_io_threads=2,
    )

    tab_dir = tmp_path / "tab_v5_mock"
    tab_dir.mkdir()
    tab_head = SparseTabularCausalPFN(_tiny_v5_config())
    prior = PriorConfig(design_split="train", min_features=12, max_features=20)
    save_tabular_checkpoint(
        tab_dir / "best_checkpoint.pt", tab_head, None, 100, {"mock": True}, prior
    )

    run_dir = tmp_path / "option_b_v7_run"
    opt_b_cfg = OptionBConfig(
        learned_temporal_dim=4,
        event_width=32,
        event_layers=1,
        event_heads=4,
        temporal_chunk_size=16,
    )
    feat_cfg = OptionAFeatureConfig(
        include_nadir_peak=False,
        icd_svd_dim=0,
        sent_km_k=8,
        include_latest_sent_km=False,
        lex_svd_dim=0,
        sent_batch_size=16,
        sent_max_length=32,
    )

    # 3-way mixing with grad_accum_steps=3 -> 1 v23 + 1 multimodal + 1 tabular per step!
    res1 = train_option_b(
        run_dir=run_dir,
        tabular_init_dir=tab_dir,
        backbone_path=bert_dir,
        design_bank_path=None,
        paraphrase_bank_path=None,
        v23_cache_dir=cache_dir,
        v23_prob=0.34,
        v23_train_split="train_unseen",
        v23_eval_split="validation",
        v23_eval_studies=2,
        v23_eval_context=18,
        v23_min_context=18,
        v23_max_context=22,
        v23_query_size=6,
        option_b_cfg=opt_b_cfg,
        feature_cfg=feat_cfg,
        steps=2,
        grad_accum_steps=3,
        multimodal_prob=0.34,
        min_context_size=20,
        context_size=20,
        query_size=6,
        tabular_min_context=20,
        tabular_max_context=20,
        tabular_query_size=6,
        tabular_min_features=14,
        tabular_max_features=18,
        warmup_steps=1,
        save_every=2,
        eval_every=2,
        eval_studies=2,
        eval_context_size=20,
        eval_query_size=6,
        tabular_eval_studies=2,
        tabular_eval_context=20,
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

    # Resume cleanly to step 3
    res2 = train_option_b(
        run_dir=run_dir,
        tabular_init_dir=tab_dir,
        backbone_path=bert_dir,
        design_bank_path=None,
        paraphrase_bank_path=None,
        v23_cache_dir=cache_dir,
        v23_prob=0.34,
        v23_train_split="train_unseen",
        v23_eval_split="validation",
        v23_eval_studies=2,
        v23_eval_context=18,
        v23_min_context=18,
        v23_max_context=22,
        v23_query_size=6,
        option_b_cfg=opt_b_cfg,
        feature_cfg=feat_cfg,
        steps=3,
        grad_accum_steps=3,
        multimodal_prob=0.34,
        min_context_size=20,
        context_size=20,
        query_size=6,
        tabular_min_context=20,
        tabular_max_context=20,
        tabular_query_size=6,
        tabular_min_features=14,
        tabular_max_features=18,
        warmup_steps=1,
        save_every=1,
        eval_every=1,
        eval_studies=2,
        eval_context_size=20,
        eval_query_size=6,
        tabular_eval_studies=2,
        tabular_eval_context=20,
        tabular_eval_features=14,
        eval_step_zero=True,
        num_workers=0,
        device="cpu",
        precision="fp32",
        resume=True,
    )
    assert res2["step"] == 3

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    train_recs = [m for m in metrics if m["event"] == "train"]
    eval_recs = [m for m in metrics if m["event"] == "eval"]
    assert [r["step"] for r in train_recs] == [1, 2, 3]
    assert all(r["v23_loss"] is not None and r["mm_loss"] is not None and r["tab_loss"] is not None for r in train_recs)
    assert all(r["v23_cate_rmse"] is not None and r["selection_cate_rmse"] is not None for r in eval_recs)

    # Verify backward-compatible checkpoint loading and zero-shot v23 test evaluation
    loaded_model, loaded_saved = load_option_b_run(run_dir, device="cpu", prefer_best=True)
    assert loaded_saved["settings"]["v23_train_split"] == "train_unseen"
    test_store = MMAIv23EpisodeStore(cache_dir, split="test")
    ev_test = evaluate_on_v23(
        loaded_model.tabular_head, test_store, studies=4, context_size=18, query_size=6, sent_km_k=8
    )
    assert ev_test["split"] == "test"
    assert np.isfinite(ev_test["cate_rmse"])
    assert np.isfinite(ev_test["prop_rmse"])
