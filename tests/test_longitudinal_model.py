import copy
import json
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertConfig,
    Lfm2Config,
    PreTrainedTokenizerFast,
)

from oci.models.lossless_tokenization import SemanticTruncationError

from oci.longitudinal.cli import main, predict
from oci.longitudinal.events import (
    PRO_INDEX,
    STREAMS,
    binary_observations,
    event_times,
    lab_features,
    pro_features,
    pro_vector,
)
from oci.longitudinal.model import (
    CHECKPOINT_FORMAT_VERSION,
    ArmScale,
    CausalStudyModel,
    ModelConfig,
)
from oci.longitudinal.training import (
    DESIGN_BANK_FILENAME,
    fit_episode,
    load_run,
    simulate_episode,
)
from synthetic_data.longitudinal import generate_longitudinal_cohort


@pytest.fixture(scope="module", autouse=True)
def tiny_model_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


@pytest.fixture(scope="module")
def note_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-lfm")
    tokens = Tokenizer(
        WordLevel({"[PAD]": 0, "[UNK]": 1, "visit": 2, "ECOG": 3}, unk_token="[UNK]")
    )
    tokens.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokens, pad_token="[PAD]", unk_token="[UNK]"
    ).save_pretrained(path)
    config = Lfm2Config(
        vocab_size=4,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention"],
        block_auto_adjust_ff_dim=False,
        max_position_embeddings=256,
    )
    AutoModel.from_config(config).save_pretrained(path)
    return path


@pytest.fixture
def model(note_model):
    return CausalStudyModel(
        AutoModel.from_pretrained(note_model),
        AutoTokenizer.from_pretrained(note_model),
        ModelConfig(
            event_width=8,
            event_layers=1,
            study_width=16,
            study_layers=2,
            bins=33,
            note_slots=2,
        ),
    )


def test_inference_modules_do_not_import_synthetic_data():
    script = (
        "import json, sys; "
        "import oci.longitudinal.events, oci.longitudinal.model; "
        "print(json.dumps([m for m in sys.modules if m == 'synthetic_data' or m.startswith('synthetic_data.')]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == []


def test_gradients_time_and_note_attention_pooling(model):
    rows, _ = simulate_episode(7, 1, False, 8)
    loss, _, _, _, _, parts = fit_episode(model, rows, 5)
    assert all(torch.isfinite(v) for v in parts.values())
    loss.backward()
    for part in [
        model.patient.backbone,
        model.patient.note_pool_norm,
        model.patient.note_key,
        model.patient.note_val,
        model.patient.note_max_proj,
        model.patient.notes,
        model.patient.note_cross_norm,
        model.patient.codes,
        model.patient.code_count,
        model.patient.encounter_summary,
        model.patient.labs,
        model.patient.lab_summary,
        model.patient.vitals,
        model.patient.vital_summary,
        model.patient.pros,
        model.patient.pro_summary,
        model.patient.cross_modal,
        *model.patient.streams.values(),
        model.solver_note_proj,
        model.solver_struct_proj,
        model.solver_mlp,
        model.arm_cal,
        model.prop_cal,
        model.slot_proj,
        model.outcome_token,
        model.layers,
        model.head,
        model.propensity_head,
    ]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    for scalar_param in (
        model.patient.note_queries,
        model.log_lam_e,
        model.log_lam_m,
        model.log_lam_tau,
        model.log_cv_temp,
        model.attn_residual_scale,
        model.prop_residual_scale,
    ):
        assert scalar_param.grad is not None and scalar_param.grad.abs().sum() > 0

    model.eval()
    docs = [{"text": "visit"}, {"text": "visit ECOG visit ECOG"}]
    with torch.no_grad():
        # Right-padding invariance for multi-query attention + max + mean pooling.
        torch.testing.assert_close(
            model.patient._note_vectors(docs)[:1],
            model.patient._note_vectors(docs[:1]),
            atol=1e-5,
            rtol=1e-5,
        )
        assert not torch.allclose(
            model.patient._note_vectors([{"text": "visit"}]),
            model.patient._note_vectors([{"text": "visit ECOG"}]),
        )
        changed = copy.deepcopy(rows[0])
        for event in changed["labs"]:
            event["day_from_index"] *= 2
            if event["delta_years"] is not None:
                event["delta_years"] *= 2
        assert not torch.allclose(model.patient([rows[0]]), model.patient([changed]))
        empty = {key: [] for key in STREAMS}
        assert torch.isfinite(model.patient([empty, rows[0]])).all()
    # Index-day events (day_from_index == 0) and omitted delta_years are allowed for real EHRs.
    times = event_times([{"day_from_index": -10}, {"day_from_index": 0}])
    assert len(times) == 2 and times[1][0] == 0.0
    with pytest.raises(ValueError, match="before the index"):
        event_times([{"day_from_index": 1, "delta_years": None}])


def test_real_ehr_open_vocabulary_labs_pros_and_encounters(model):
    # Open-vocabulary lab panels & components (e.g. tumor markers) and non-PRO-CTCAE instruments
    # should encode cleanly without failing closed.
    lab_event = dict(
        day_from_index=0,
        panel="TUMOR_MARKERS",
        results=[
            {"component": "cea", "value": 18.4, "unit": "ng/mL"},
            {"component": "hemoglobin", "value": 11.2, "unit": "g/dL"},
        ],
    )
    vals, extra = lab_features(lab_event, strict=False)
    assert len(extra) == 2  # panel token + cea token
    pro_event = dict(
        day_from_index=0,
        instrument="EORTC-QLQ-C30",
        recall_days=14,
        responses=[{"symptom": "global_health", "attribute": "score", "score": 65}],
    )
    pvals, pextra = pro_features(pro_event, strict=False)
    assert len(pextra) == 1 and all(v == 0.0 for v in pvals)

    ehr_row = dict(
        notes=[{"day_from_index": 0, "text": "visit ECOG"}],
        labs=[lab_event],
        encounters=[
            {
                "day_from_index": -5,
                "setting": "inpatient",
                "icd10cm": ["C34.90", "I10", "E11.9", "J44.9", "C79.51"],
                "cpt": ["99223", "71260", "85025"],
            }
        ],
        vitals=[{"day_from_index": 0, "weight_kg": 71.5, "systolic_mmhg": 132}],
        pros=[pro_event],
    )
    model.eval()
    with torch.no_grad():
        emb = model.patient([ehr_row])
        assert emb.shape == (1, model.patient.num_slots, model.cfg.event_width)
        assert torch.isfinite(emb).all()


def test_pro_scores_timing_and_empty_history(model):
    def form(score):
        return dict(
            instrument="PRO-CTCAE",
            recall_days=7,
            day_from_index=-12,
            delta_years=None,
            responses=[dict(symptom="fatigue", attribute="interference", score=score)],
        )

    offset = 5 * PRO_INDEX["fatigue", "interference"]
    for score, expected in [(None, [0, 0, 0, 0, 0]), (0, [1, 0, 0, 0, 0]), (2, [1, 1, 1, 0, 0])]:
        assert pro_vector(form(score))[offset : offset + 5] == expected
    for score in (-1, 5, 2.5, True):
        with pytest.raises(ValueError, match="ordinal"):
            pro_vector(form(score))
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        pro_vector({**form(None), "responses": form(None)["responses"] + form(1)["responses"]})
    with pytest.raises(ValueError, match="Unknown"):
        pro_vector(
            {**form(1), "responses": [dict(symptom="nausea", attribute="interference", score=1)]}
        )
    with pytest.raises(ValueError, match="seven-day"):
        pro_vector({**form(1), "recall_days": 30})

    row = {s: [] for s in STREAMS if s != "pros"}
    rows = [
        row,
        {**row, "pros": []},
        {**row, "pros": [form(0)]},
        {**row, "pros": [form(4)]},
        {**row, "pros": [{**form(4), "day_from_index": -70}]},
    ]
    model.eval()
    with torch.no_grad():
        embeddings = model.patient(rows)
        assert embeddings.shape == (len(rows), model.patient.num_slots, model.cfg.event_width)
        torch.testing.assert_close(embeddings[0], embeddings[1])
        # Non-PRO, non-cross-modal slots match identically when only PROs change.
        non_pro_slots = model.cfg.note_slots + 3  # notes + labs + encounters + vitals
        torch.testing.assert_close(
            embeddings[:, :non_pro_slots],
            embeddings[0:1, :non_pro_slots].expand(len(rows), -1, -1),
        )
        for first, second in [(1, 2), (2, 3), (3, 4)]:
            assert not torch.allclose(embeddings[first, non_pro_slots:], embeddings[second, non_pro_slots:])
        context, _ = simulate_episode(8, 2, False, 3)
        logits, _, _ = model(context, rows[2:])
        assert not torch.allclose(logits[0], logits[1])


def test_context_and_query_information_boundaries(model):
    model.eval()
    rows, _ = simulate_episode(8, 2, False, 6)
    with torch.no_grad():

        def scores(context, queries):
            t, y = binary_observations(context, "cpu")
            memory, scale = model.condition(model.patient(context), t, y)
            q_emb = model.patient(queries)
            mean, sd = model.risk_stats(model.query_logits(q_emb, memory, scale), scale)
            prop = model.query_propensity_logits(q_emb, memory, scale).sigmoid()
            return mean, sd, prop

        pred, sd, prop = scores(rows[:3], rows[3:])
        changed = {
            **rows[4],
            "outcome_indicator": 100,
            "treatment_indicator": -123,
            "true_y0": float("nan"),
            "true_treatment_prob": float("nan"),
        }
        rev_pred, rev_sd, rev_prop = scores(list(reversed(rows[:3])), [changed])
        torch.testing.assert_close(pred[1:2], rev_pred, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sd[1:2], rev_sd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(prop[1:2], rev_prop, atol=1e-5, rtol=1e-5)

        new_context = [{**r, "outcome_indicator": 1 - r["outcome_indicator"]} for r in rows[:3]]
        new_pred, _, _ = scores(new_context, rows[3:])
        assert not torch.allclose(pred, new_pred)
        single_arm = ArmScale.from_context(torch.zeros(4), torch.ones(4))
        assert torch.isfinite(single_arm.logit_offsets(torch.tensor([[0.0, 1.0]]))).all()
        assert single_arm.rate[1] == 0.5
        assert (pred >= 0).all() and (pred <= 1).all()
        assert (sd >= 0).all() and (prop >= 0).all() and (prop <= 1).all()


def test_study_replay_from_published_seeds(tmp_path):
    rows, seeds = simulate_episode(52, 3, True, 5)
    _, train_seeds = simulate_episode(52, 3, False, 5)
    assert seeds["dgp_seed"] % 2 == 1 and train_seeds["dgp_seed"] % 2 == 0
    path = generate_longitudinal_cohort(
        tmp_path / "study",
        dataset_size=5,
        prior="mixed",
        seed=seeds["patient_seed"],
        dgp_seed=seeds["dgp_seed"],
    )
    assert rows == pq.read_table(path / "dataset.parquet").to_pylist()


def test_train_resume_and_unlabeled_inference(note_model, tmp_path):
    run = tmp_path / "run"
    data = generate_longitudinal_cohort(tmp_path / "data", dataset_size=12) / "dataset.parquet"
    args = [
        "train",
        "--steps",
        "1",
        "--device",
        "cpu",
        "--context-size",
        "4",
        "--query-size",
        "2",
        "--note-model",
        str(note_model),
        "--run-dir",
        str(run),
        "--train-cohort",
        str(data),
        "--event-width",
        "8",
        "--event-layers",
        "1",
        "--study-width",
        "16",
        "--study-layers",
        "2",
        "--note-slots",
        "2",
        "--bins",
        "17",
    ]
    main(args)
    main(["resume", "--run-dir", str(run), "--device", "cpu", "--steps", "2"])
    model, saved = load_run(run, "cpu")
    assert saved["step"] == 2 and saved["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert (run / "metrics.jsonl").is_file()
    args[args.index("--steps") + 1] = "2"
    args[args.index(str(run))] = str(tmp_path / "uninterrupted")
    main(args)
    uninterrupted, _ = load_run(tmp_path / "uninterrupted", "cpu")
    for resumed, direct in zip(model.parameters(), uninterrupted.parameters(), strict=True):
        torch.testing.assert_close(resumed, direct, rtol=0, atol=0)
    table = pq.read_table(data)
    qpath, cpath = tmp_path / "queries.parquet", tmp_path / "context.parquet"
    # Deliberately give queries a different patient_id type (string) than context (int64).
    q_table = table.slice(5).select(["patient_id", *STREAMS])
    q_table = q_table.set_column(
        0, "patient_id", pa.array([f"q_{i}" for i in q_table["patient_id"].to_pylist()])
    )
    pq.write_table(q_table, qpath)
    pq.write_table(
        table.slice(0, 5).select(
            ["patient_id", *STREAMS, "treatment_indicator", "outcome_indicator"]
        ),
        cpath,
    )
    out = tmp_path / "out.parquet"
    main(
        [
            "predict",
            "--run-dir",
            str(run),
            "--context",
            str(cpath),
            "--queries",
            str(qpath),
            "--output",
            str(out),
            "--batch-size",
            "2",
            "--note-batch-size",
            "4",
            "--device",
            "cpu",
        ]
    )
    p = pq.read_table(out).to_pandas()
    assert p.patient_id.tolist() == [f"q_{i}" for i in range(5, 12)]
    assert (p.risk_treated - p.risk_control == p.risk_difference).all()
    assert {"risk_control_sd", "risk_treated_sd", "propensity_score", "outside_overlap"} <= set(
        p.columns
    )
    assert (p.propensity_score >= 0).all() and (p.propensity_score <= 1).all()
    assert predict(model, data, None, tmp_path / "split.parquet", context_size=4) == 8

    # Verify disjointness check on same-type IDs.
    overlap_q = tmp_path / "overlap_queries.parquet"
    pq.write_table(table.slice(5).select(["patient_id", *STREAMS]), overlap_q)
    with pytest.raises(ValueError, match="disjoint"):
        predict(model, data, overlap_q, tmp_path / "bad.parquet")

    legacy_query = tmp_path / "old_queries.parquet"
    pq.write_table(pq.read_table(qpath).drop(["pros"]), legacy_query)
    assert predict(model, cpath, legacy_query, tmp_path / "old_out.parquet") == 7

    # Materialized cohort evaluation via --eval-cohort.
    main(
        [
            "evaluate",
            "--run-dir",
            str(run),
            "--eval-cohort",
            str(data),
            "--studies",
            "2",
            "--context-size",
            "6",
            "--query-size",
            "4",
            "--device",
            "cpu",
        ]
    )


def test_four_stream_checkpoint_keeps_its_four_stream_architecture(note_model, tmp_path):
    legacy = CausalStudyModel(
        AutoModel.from_pretrained(note_model),
        AutoTokenizer.from_pretrained(note_model),
        ModelConfig(
            event_width=8, event_layers=1, study_width=16, study_layers=1, bins=17, note_slots=2
        ),
        include_pros=False,
    ).eval()
    legacy.patient.tokenizer.save_pretrained(tmp_path / "tokenizer")
    torch.save(
        dict(
            format_version=1,
            model=vars(legacy.cfg),
            backbone=legacy.patient.backbone.config.to_dict(),
            weights=legacy.state_dict(),
        ),
        tmp_path / "checkpoint.pt",
    )
    restored, _ = load_run(tmp_path, "cpu")
    rows, _ = simulate_episode(7, 1, False, 4, schema_version=1)
    assert "pros" not in rows[0] and "pros" not in restored.patient.streams
    with torch.no_grad():
        torch.testing.assert_close(
            restored.eval()(rows[:2], rows[2:])[0], legacy(rows[:2], rows[2:])[0]
        )


def test_linear_lr_resume_reproduces_uninterrupted_updates(
    tmp_path, note_model, monkeypatch, capsys
):
    from oci.longitudinal import training

    run = tmp_path / "linear"
    mid_checkpoint = tmp_path / "step2.pt"
    save = training._save

    def save_with_midpoint(directory, model, optimizer, settings, step, **kwargs):
        save(directory, model, optimizer, settings, step, **kwargs)
        if step == 2 and kwargs.get("filename", "checkpoint.pt") == "checkpoint.pt":
            shutil.copyfile(directory / "checkpoint.pt", mid_checkpoint)

    monkeypatch.setattr(training, "_save", save_with_midpoint)
    main(
        [
            "train",
            "--run-dir",
            str(run),
            "--mixed-prior",
            "--note-model",
            str(note_model),
            "--device",
            "cpu",
            "--context-size",
            "3",
            "--query-size",
            "2",
            "--steps",
            "4",
            "--save-every",
            "2",
            "--lr-scheduler",
            "linear",
            "--note-lr",
            "0.00002",
            "--lr",
            "0.0004",
            "--event-width",
            "8",
            "--event-layers",
            "1",
            "--study-width",
            "16",
            "--study-layers",
            "1",
            "--note-slots",
            "2",
        ]
    )
    full_log = [json.loads(s) for s in capsys.readouterr().out.splitlines() if s.startswith("{")]
    for row, factor in zip(full_log, [1, 0.75, 0.5, 0.25], strict=True):
        assert row["note_lr"] == pytest.approx(2e-5 * factor)
        assert row["lr"] == pytest.approx(4e-4 * factor)
    full, saved = load_run(run, "cpu")
    assert saved["settings"]["lr_schedule_steps"] == 4
    assert all(group["lr"] == 0 for group in saved["optimizer"]["param_groups"])

    mid_checkpoint.replace(run / "checkpoint.pt")
    main(["resume", "--run-dir", str(run), "--device", "cpu", "--steps", "4"])
    resumed_log = [json.loads(s) for s in capsys.readouterr().out.splitlines() if s.startswith("{")]
    assert resumed_log == full_log[2:]
    resumed, _ = load_run(run, "cpu")
    for first, second in zip(full.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_design_bank_externalized_grad_accum_var_ctx_and_rich_evaluation(
    tmp_path, note_model, capsys
):
    from synthetic_data.study_designs import DEFAULT_BANK, load_bank
    from oci.longitudinal.training import split_designs

    bank = load_bank(DEFAULT_BANK)
    train, held = [split_designs(bank, s) for s in ("train", "held_out")]
    assert (len(train), len(held)) == (4000, 1000)
    assert {
        sum(len(items) for items in design["candidates"].values()) for design in bank["designs"]
    } == set(range(3, 101))
    assert {d["question"]["id"] for d in train}.isdisjoint(d["question"]["id"] for d in held)
    assert {d["question"]["cancer"] for d in held} - {d["question"]["cancer"] for d in train}
    run = tmp_path / "designed"
    main(
        [
            "train",
            "--run-dir",
            str(run),
            "--steps",
            "1",
            "--lr-scheduler",
            "linear",
            "--min-context-size",
            "3",
            "--context-size",
            "5",
            "--query-size",
            "2",
            "--grad-accum-steps",
            "2",
            "--eval-every",
            "1",
            "--eval-studies",
            "2",
            "--device",
            "cpu",
            "--note-model",
            str(note_model),
            "--event-width",
            "8",
            "--event-layers",
            "1",
            "--study-width",
            "16",
            "--study-layers",
            "1",
            "--note-slots",
            "2",
        ]
    )
    # Bank is saved once as design_bank.json.gz rather than embedded inside checkpoint.pt.
    assert (run / DESIGN_BANK_FILENAME).is_file()
    assert (run / "best_checkpoint.pt").is_file()
    _, saved = load_run(run, "cpu")
    assert "design_bank" not in saved["settings"]
    assert saved["settings"]["design_bank_file"] == DESIGN_BANK_FILENAME

    main(["resume", "--run-dir", str(run), "--steps", "2", "--device", "cpu"])
    logs = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    assert [d["step"] for d in logs] == [1, 2]
    assert [d["lr"] for d in logs] == pytest.approx([3e-4, 1.5e-4])
    assert all(d["design_id"] in {v["id"] for v in train} for d in logs)
    assert all("eval_cate_rmse" in d and "eval_within_study_cate_pearson" in d for d in logs)
    assert all("eval_linear_t_learner_cate_rmse" in d and "eval_ate_rmse" in d for d in logs)

    main(
        [
            "evaluate",
            "--run-dir",
            str(run),
            "--device",
            "cpu",
            "--studies",
            "4",
            "--context-size",
            "6",
            "--query-size",
            "3",
        ]
    )
    metrics = json.loads(capsys.readouterr().out)
    assert set(metrics["held_out_designs"]) == {d["id"] for d in held[:4]}
    assert all(v["studies"] == 1 for v in metrics["held_out_designs"].values())
    assert {
        "risk_rmse",
        "risk_brier",
        "cate_rmse",
        "ate_rmse",
        "ate_bias",
        "arm_mean_cate_rmse",
        "zero_cate_rmse",
        "linear_t_learner_cate_rmse",
        "propensity_rmse",
        "by_width",
        "by_generalization",
    } <= set(metrics)


def test_freeze_backbone_partial_note_grads_and_worker_prefetch_determinism(
    tmp_path, note_model
):
    # Phase 1: frozen backbone must keep backbone weights untouched while updating heads.
    run = tmp_path / "two_phase"
    base_args = [
        "train",
        "--run-dir",
        str(run),
        "--mixed-prior",
        "--note-model",
        str(note_model),
        "--device",
        "cpu",
        "--context-size",
        "6",
        "--query-size",
        "2",
        "--steps",
        "1",
        "--freeze-note-backbone",
        "--event-width",
        "8",
        "--event-layers",
        "1",
        "--study-width",
        "16",
        "--study-layers",
        "1",
        "--note-slots",
        "2",
    ]
    main(base_args)
    m1, s1 = load_run(run, "cpu")
    assert s1["settings"]["freeze_note_backbone"] is True
    init_backbone = AutoModel.from_pretrained(note_model)
    for trained, orig in zip(m1.patient.backbone.parameters(), init_backbone.parameters(), strict=True):
        torch.testing.assert_close(trained, orig, rtol=0, atol=0)

    # Phase 2: resume with curriculum overrides (unfreeze, larger context, grad_accum, new LR).
    main(
        [
            "resume",
            "--run-dir",
            str(run),
            "--device",
            "cpu",
            "--steps",
            "2",
            "--no-freeze-note-backbone",
            "--note-grad-patients",
            "2",
            "--min-context-size",
            "8",
            "--context-size",
            "10",
            "--query-size",
            "3",
            "--grad-accum-steps",
            "2",
            "--lr",
            "0.0005",
            "--note-lr",
            "0.00005",
        ]
    )
    m2, s2 = load_run(run, "cpu")
    assert s2["settings"]["freeze_note_backbone"] is False
    assert s2["settings"]["note_grad_patients"] == 2
    assert s2["settings"]["min_context_size"] == 8
    assert s2["settings"]["context_size"] == 10
    assert s2["settings"]["query_size"] == 3
    assert s2["settings"]["grad_accum_steps"] == 2
    assert s2["settings"]["lr"] == pytest.approx(5e-4)
    assert s2["settings"]["note_lr"] == pytest.approx(5e-5)
    assert any(
        not torch.allclose(a, b)
        for a, b in zip(m2.patient.backbone.parameters(), init_backbone.parameters(), strict=True)
    )

    # Background CPU prefetching (--num-workers 2) matches synchronous (--num-workers 0) exactly.
    sync_dir, async_dir = tmp_path / "sync", tmp_path / "async"
    common = [
        "train",
        "--mixed-prior",
        "--note-model",
        str(note_model),
        "--device",
        "cpu",
        "--context-size",
        "5",
        "--query-size",
        "2",
        "--steps",
        "2",
        "--grad-accum-steps",
        "2",
        "--note-grad-patients",
        "2",
        "--event-width",
        "8",
        "--event-layers",
        "1",
        "--study-width",
        "16",
        "--study-layers",
        "1",
        "--note-slots",
        "2",
    ]
    main([*common, "--run-dir", str(sync_dir), "--num-workers", "0"])
    main([*common, "--run-dir", str(async_dir), "--num-workers", "2"])
    m_sync, _ = load_run(sync_dir, "cpu")
    m_async, _ = load_run(async_dir, "cpu")
    for p_sync, p_async in zip(m_sync.parameters(), m_async.parameters(), strict=True):
        torch.testing.assert_close(p_sync, p_async, rtol=0, atol=0)


def test_tinybert_encoder_attention_pooling_and_lossless_long_note_windowing(tmp_path):
    bert_dir = tmp_path / "tiny-bert"
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "visit": 4, "ECOG": 5, "anemia": 6}
    tokens = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokens.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokens,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    ).save_pretrained(bert_dir)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    AutoModel.from_config(config).save_pretrained(bert_dir)

    bert_model = CausalStudyModel(
        AutoModel.from_pretrained(bert_dir),
        AutoTokenizer.from_pretrained(bert_dir),
        ModelConfig(
            event_width=8,
            event_layers=1,
            study_width=16,
            study_layers=1,
            bins=17,
            note_slots=2,
            solver_dim=8,
        ),
    )
    assert bert_model.patient._is_bidirectional is True

    # Verify gradients flow through BertModel and attention pooler cleanly.
    rows, _ = simulate_episode(11, 1, False, 6)
    loss, _, _, _, _, _ = fit_episode(bert_model, rows, 4)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in bert_model.patient.backbone.parameters()
    )
    assert bert_model.patient.note_queries.grad is not None
    assert bert_model.patient.note_queries.grad.abs().sum() > 0

    # Frozen backbone must still train note_queries / note_key / note_val.
    bert_model.zero_grad(set_to_none=True)
    loss_frozen, _, _, _, _, _ = fit_episode(bert_model, rows, 4, freeze_note_backbone=True)
    loss_frozen.backward()
    assert all(p.grad is None for p in bert_model.patient.backbone.parameters())
    assert bert_model.patient.note_queries.grad is not None
    assert bert_model.patient.note_queries.grad.abs().sum() > 0

    # Verify right-padding invariance, exact cross-window aggregation, and lossless windowing.
    bert_model.eval()
    docs = [{"text": "visit"}, {"text": "visit ECOG visit ECOG"}]
    with torch.no_grad():
        torch.testing.assert_close(
            bert_model.patient._note_vectors(docs)[:1],
            bert_model.patient._note_vectors(docs[:1]),
            atol=1e-5,
            rtol=1e-5,
        )
        # Verify exact log-sum-exp cross-window attention + max + token-weighted mean equivalence.
        hidden = torch.randn(1, 10, 16)
        mask = torch.ones(1, 10, dtype=torch.long)
        full_stats = bert_model.patient._pool_window_hidden(hidden, mask)
        full_agg = bert_model.patient._aggregate_windows(*full_stats, [1])
        h_parts = torch.cat([hidden[:, :4], torch.zeros(1, 2, 16)], dim=1)
        m_parts = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.long)
        split_hidden = torch.cat([h_parts, hidden[:, 4:]], dim=0)
        split_stats = bert_model.patient._pool_window_hidden(split_hidden, m_parts)
        split_agg = bert_model.patient._aggregate_windows(*split_stats, [2])
        for a, b in zip(full_agg, split_agg, strict=True):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

        # Verify lossless overlapping windowing (32-token overlap at limit=512; clamped at limit=16).
        seq_512 = [2, *range(10, 10 + 988), 3]  # 988 body tokens -> exact 2 windows of 510 with 32 overlap
        w512, c512 = bert_model.patient._window_note_ids([seq_512], 512)
        assert c512 == [2]
        assert len(w512[0]) == 512 and len(w512[1]) == 512
        assert w512[0][-33:-1] == w512[1][1:33]  # exact 32-token body overlap

        # Changing a token past token 30 must change the resulting note representation at limit=16.
        long_a = {"text": " ".join(["visit"] * 35 + ["ECOG"])}
        long_b = {"text": " ".join(["visit"] * 35 + ["anemia"])}
        va = bert_model.patient._note_vectors([long_a])
        vb = bert_model.patient._note_vectors([long_b])
        assert torch.isfinite(va).all() and torch.isfinite(vb).all()
        assert not torch.allclose(va, vb)

    with pytest.raises(SemanticTruncationError):
        bert_model.patient._note_vectors([{"text": ""}])


def test_longitudinal_trajectory_slopes_and_observational_loocv_ridge_baseline():
    from oci.longitudinal.events import LAB_INDEX, lab_stream_summary, vital_stream_summary
    from oci.longitudinal.training import _ridge_t_learner_cate

    # Rising vs falling creatinine over 180 days must produce opposite-signed slopes.
    rising = [
        {"day_from_index": -180, "panel": "BMP", "results": [{"component": "creatinine", "value": 0.8, "unit": "mg/dL"}]},
        {"day_from_index": 0, "panel": "BMP", "results": [{"component": "creatinine", "value": 2.0, "unit": "mg/dL"}]},
    ]
    falling = [
        {"day_from_index": -180, "panel": "BMP", "results": [{"component": "creatinine", "value": 2.0, "unit": "mg/dL"}]},
        {"day_from_index": 0, "panel": "BMP", "results": [{"component": "creatinine", "value": 0.8, "unit": "mg/dL"}]},
    ]
    cr_slope_idx = 4 * LAB_INDEX["creatinine"] + 2
    assert lab_stream_summary(rising)[cr_slope_idx] > 0.5
    assert lab_stream_summary(falling)[cr_slope_idx] < -0.5

    # Weight loss over a year must yield a clearly negative slope.
    w_loss = [
        {"day_from_index": -365, "weight_kg": 85.0},
        {"day_from_index": -180, "weight_kg": 77.0},
        {"day_from_index": 0, "weight_kg": 70.0},
    ]
    assert vital_stream_summary(w_loss)[2] < -0.3

    # Observational LOO-CV Ridge baseline stays well-regularized when N_ctx < d_summary.
    rows, _ = simulate_episode(19, 1, True, 96)
    pred_ite = _ridge_t_learner_cate(rows[:64], rows[64:])
    true_ite = [r["true_y1"] - r["true_y0"] for r in rows[64:]]
    rmse = float(torch.tensor(pred_ite - torch.tensor(true_ite).numpy()).square().mean().sqrt())
    assert rmse < 0.25


def test_study_adaptive_lexical_svd_channel_and_v4_checkpoint_compat(note_model, tmp_path):
    # 1. lexical_dim > 0: gradients, context permutation invariance, and train/resume/predict.
    lex_model = CausalStudyModel(
        AutoModel.from_pretrained(note_model),
        AutoTokenizer.from_pretrained(note_model),
        ModelConfig(
            event_width=8,
            event_layers=1,
            study_width=16,
            study_layers=1,
            bins=17,
            note_slots=2,
            solver_dim=8,
            lexical_dim=4,
        ),
    )
    rows, _ = simulate_episode(13, 1, False, 8)
    loss, _, _, _, _, parts = fit_episode(lex_model, rows, 5)
    assert all(torch.isfinite(v) for v in parts.values())
    loss.backward()
    assert lex_model.lexical_log_weight.grad is not None
    assert lex_model.lexical_log_weight.grad.abs().sum() > 0

    lex_model.eval()
    with torch.no_grad():
        logits_a, prop_a, scale_a = lex_model(rows[:5], rows[5:])
        logits_b, prop_b, scale_b = lex_model(list(reversed(rows[:5])), [rows[6]])
        torch.testing.assert_close(
            lex_model.risks(logits_a, scale_a)[1:2],
            lex_model.risks(logits_b, scale_b),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(prop_a[1:2], prop_b, atol=1e-4, rtol=1e-4)
        # Empty notes remain finite.
        empty_rows = [{**r, "notes": []} for r in rows]
        logits_e, prop_e, _ = lex_model(empty_rows[:5], empty_rows[5:])
        assert torch.isfinite(logits_e).all() and torch.isfinite(prop_e).all()

    data = generate_longitudinal_cohort(tmp_path / "lex_data", dataset_size=10) / "dataset.parquet"
    run = tmp_path / "lex_run"
    main(
        [
            "train",
            "--run-dir",
            str(run),
            "--train-cohort",
            str(data),
            "--note-model",
            str(note_model),
            "--device",
            "cpu",
            "--steps",
            "1",
            "--context-size",
            "5",
            "--query-size",
            "2",
            "--event-width",
            "8",
            "--event-layers",
            "1",
            "--study-width",
            "16",
            "--study-layers",
            "1",
            "--note-slots",
            "2",
            "--solver-dim",
            "8",
            "--lexical-dim",
            "4",
            "--bins",
            "17",
        ]
    )
    restored, _ = load_run(run, "cpu")
    assert restored.cfg.lexical_dim == 4
    assert predict(restored, data, None, tmp_path / "lex_out.parquet", context_size=5) == 5

    # 2. Backward compatibility: v4 checkpoints saved without lexical_dim load identically.
    compat_dir = tmp_path / "v4_without_lexical"
    compat_dir.mkdir()
    base = CausalStudyModel(
        AutoModel.from_pretrained(note_model),
        AutoTokenizer.from_pretrained(note_model),
        ModelConfig(
            event_width=8,
            event_layers=1,
            study_width=16,
            study_layers=1,
            bins=17,
            note_slots=2,
            solver_dim=8,
        ),
    ).eval()
    assert base.lexical_log_weight is None
    assert "lexical_log_weight" not in base.state_dict()
    base.patient.tokenizer.save_pretrained(compat_dir / "tokenizer")
    old_cfg = {k: v for k, v in vars(base.cfg).items() if k != "lexical_dim"}
    torch.save(
        dict(
            format_version=CHECKPOINT_FORMAT_VERSION,
            model=old_cfg,
            backbone=base.patient.backbone.config.to_dict(),
            weights=base.state_dict(),
        ),
        compat_dir / "checkpoint.pt",
    )
    loaded, _ = load_run(compat_dir, "cpu")
    assert loaded.cfg.lexical_dim == 0
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.eval()(rows[:4], rows[4:])[0], base(rows[:4], rows[4:])[0]
        )


def test_per_slot_note_temporal_log_uniform_sampling_and_eval_context_size(
    note_model, tmp_path, capsys
):
    import numpy as np
    from oci.longitudinal.training import _sample_context_size

    # 1. Log-uniform context sampling covers every order of magnitude cleanly.
    rng = np.random.default_rng(7)
    draws = [_sample_context_size(rng, 100, 10000, "log_uniform") for _ in range(2000)]
    assert min(draws) >= 100 and max(draws) <= 10000
    assert 700 <= float(np.median(draws)) <= 1400
    assert sum(d <= 300 for d in draws) > 300 and sum(d >= 5000 for d in draws) > 200

    # 2. Per-slot longitudinal note temporal refinement + 7-point Ridge grid.
    m = CausalStudyModel(
        AutoModel.from_pretrained(note_model),
        AutoTokenizer.from_pretrained(note_model),
        ModelConfig(
            event_width=8,
            event_layers=1,
            study_width=16,
            study_layers=1,
            bins=17,
            note_slots=2,
            solver_dim=8,
            lexical_dim=4,
            ridge_grid_size=7,
            per_slot_note_temporal=True,
        ),
    )
    assert m.log_lam_tau.shape == (7,)
    rows, _ = simulate_episode(17, 1, False, 8)
    loss, _, _, _, _, parts = fit_episode(m, rows, 5)
    assert all(torch.isfinite(v) for v in parts.values())
    loss.backward()
    for part in (
        m.patient.note_slot_clock,
        m.patient.note_slot_temporal_fuse,
        m.patient.note_slot_temporal_norm,
    ):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    assert m.patient.note_slot_time_q.grad is not None
    assert m.patient.note_slot_time_q.grad.abs().sum() > 0

    # Reversing progression order of two distinct notes or shifting timestamps changes note slots.
    m.eval()
    base_row = {s: [] for s in STREAMS}
    p_a = {
        **base_row,
        "notes": [
            {"day_from_index": -100, "text": "visit"},
            {"day_from_index": 0, "text": "visit ECOG visit ECOG"},
        ],
    }
    p_b = {
        **base_row,
        "notes": [
            {"day_from_index": -100, "text": "visit ECOG visit ECOG"},
            {"day_from_index": 0, "text": "visit"},
        ],
    }
    p_c = {
        **base_row,
        "notes": [
            {"day_from_index": -10, "text": "visit"},
            {"day_from_index": 0, "text": "visit ECOG visit ECOG"},
        ],
    }
    p_one = {**base_row, "notes": [{"day_from_index": 0, "text": "visit"}]}
    with torch.no_grad():
        embs = m.patient([p_a, p_b, p_c, p_one, base_row])
        assert torch.isfinite(embs).all()
        assert not torch.allclose(embs[0, :2], embs[1, :2])
        assert not torch.allclose(embs[0, :2], embs[2, :2])

    # 3. CLI end-to-end with --per-slot-note-temporal, --ridge-grid-size 7, log_uniform, --eval-context-size.
    run = tmp_path / "single_stage"
    main(
        [
            "train",
            "--run-dir",
            str(run),
            "--mixed-prior",
            "--note-model",
            str(note_model),
            "--device",
            "cpu",
            "--steps",
            "1",
            "--min-context-size",
            "4",
            "--context-size",
            "8",
            "--context-size-sampling",
            "log_uniform",
            "--query-size",
            "2",
            "--eval-every",
            "1",
            "--eval-studies",
            "2",
            "--eval-context-size",
            "5",
            "--event-width",
            "8",
            "--event-layers",
            "1",
            "--study-width",
            "16",
            "--study-layers",
            "1",
            "--note-slots",
            "2",
            "--per-slot-note-temporal",
            "--solver-dim",
            "8",
            "--lexical-dim",
            "4",
            "--ridge-grid-size",
            "7",
            "--bins",
            "17",
        ]
    )
    log_lines = [json.loads(s) for s in capsys.readouterr().out.splitlines() if s.startswith("{")]
    assert log_lines[-1]["eval_context_size"] == 5
    reloaded, saved = load_run(run, "cpu")
    assert reloaded.cfg.per_slot_note_temporal is True and reloaded.cfg.ridge_grid_size == 7
    assert saved["settings"]["context_size_sampling"] == "log_uniform"
    assert saved["settings"]["eval_context_size"] == 5

    # 4. Backward compatibility with Run 2 v4 checkpoints (lexical_dim > 0, no new fields).
    r2_dir = tmp_path / "run2_compat"
    r2_dir.mkdir()
    r2 = CausalStudyModel(
        AutoModel.from_pretrained(note_model),
        AutoTokenizer.from_pretrained(note_model),
        ModelConfig(
            event_width=8,
            event_layers=1,
            study_width=16,
            study_layers=1,
            bins=17,
            note_slots=2,
            solver_dim=8,
            lexical_dim=4,
        ),
    ).eval()
    r2.patient.tokenizer.save_pretrained(r2_dir / "tokenizer")
    old_r2_cfg = {
        k: v
        for k, v in vars(r2.cfg).items()
        if k not in ("per_slot_note_temporal", "ridge_grid_size")
    }
    torch.save(
        dict(
            format_version=CHECKPOINT_FORMAT_VERSION,
            model=old_r2_cfg,
            backbone=r2.patient.backbone.config.to_dict(),
            weights=r2.state_dict(),
        ),
        r2_dir / "checkpoint.pt",
    )
    r2_loaded, _ = load_run(r2_dir, "cpu")
    assert r2_loaded.cfg.per_slot_note_temporal is False and r2_loaded.cfg.ridge_grid_size == 5
    with torch.no_grad():
        torch.testing.assert_close(r2_loaded.eval()(rows[:4], rows[4:])[0], r2(rows[:4], rows[4:])[0])
