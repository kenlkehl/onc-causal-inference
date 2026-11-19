import copy

import pyarrow.parquet as pq
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModel, AutoTokenizer, Lfm2Config, PreTrainedTokenizerFast

from oci.longitudinal.cli import main, predict
from oci.longitudinal.events import STREAMS, binary_observations, event_times
from oci.longitudinal.model import ArmScale, CausalStudyModel, ModelConfig
from oci.longitudinal.training import fit_episode, load_run, simulate_episode
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
        ModelConfig(event_width=8, event_layers=1, study_width=16, study_layers=2, bins=33),
    )


def test_gradients_time_and_last_nonpadding_token(model):
    rows, _ = simulate_episode(7, 1, False, 8)
    loss, _, _ = fit_episode(model, rows, 5)
    loss.backward()
    for part in [
        model.patient.backbone,
        model.patient.codes,
        model.patient.labs,
        model.patient.vitals,
        model.patient.notes,
        *model.patient.streams.values(),
        model.layers,
        model.head,
    ]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    model.eval()
    docs = [{"text": "visit"}, {"text": "visit ECOG visit ECOG"}]
    with torch.no_grad():
        pad = model.patient.tokenizer([d["text"] for d in docs], padding=True, return_tensors="pt")
        hidden = model.patient.backbone(
            input_ids=pad.input_ids, attention_mask=pad.attention_mask, use_cache=False
        ).last_hidden_state
        expected = model.patient.notes(hidden[[0, 1], [0, 3]])
        torch.testing.assert_close(model.patient._note_vectors(docs), expected)
        torch.testing.assert_close(
            model.patient._note_vectors(docs)[:1], model.patient._note_vectors(docs[:1])
        )
        changed = copy.deepcopy(rows[0])
        for event in changed["labs"]:
            event["day_from_index"] *= 2
            if event["delta_years"] is not None:
                event["delta_years"] *= 2
        assert not torch.allclose(model.patient([rows[0]]), model.patient([changed]))
        empty = {key: [] for key in STREAMS}
        assert torch.isfinite(model.patient([empty, rows[0]])).all()
    with pytest.raises(ValueError, match="before the index"):
        event_times([{"day_from_index": 1, "delta_years": None}])


def test_context_and_query_information_boundaries(model):
    model.eval()
    rows, _ = simulate_episode(8, 2, False, 6)
    with torch.no_grad():

        def scores(context, queries):
            t, y = binary_observations(context, "cpu")
            memory, scale = model.condition(model.patient(context), t, y)
            return model.risks(model.query_logits(model.patient(queries), memory), scale)

        pred = scores(rows[:3], rows[3:])
        changed = {
            **rows[4],
            "outcome_indicator": 100,
            "treatment_indicator": -123,
            "true_y0": float("nan"),
        }
        torch.testing.assert_close(
            pred[1:2], scores(list(reversed(rows[:3])), [changed]), atol=1e-6, rtol=1e-5
        )
        new_context = [{**r, "outcome_indicator": 1 - r["outcome_indicator"]} for r in rows[:3]]
        assert not torch.allclose(pred, scores(new_context, rows[3:]))
        single_arm = ArmScale.from_context(torch.zeros(4), torch.ones(4))
        assert torch.isfinite(single_arm.standardized_logits(torch.tensor([[0.0, 1.0]]))).all()
        assert single_arm.rate[1] == 0.5
        assert (pred >= 0).all() and (pred <= 1).all()


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
    data = generate_longitudinal_cohort(tmp_path / "data", dataset_size=10) / "dataset.parquet"
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
        "--bins",
        "17",
    ]
    main(args)
    main(["resume", "--run-dir", str(run), "--device", "cpu", "--steps", "2"])
    model, saved = load_run(run, "cpu")
    assert saved["step"] == 2
    args[args.index("--steps") + 1] = "2"
    args[args.index(str(run))] = str(tmp_path / "uninterrupted")
    main(args)
    uninterrupted, _ = load_run(tmp_path / "uninterrupted", "cpu")
    for resumed, direct in zip(model.parameters(), uninterrupted.parameters(), strict=True):
        torch.testing.assert_close(resumed, direct, rtol=0, atol=0)
    table = pq.read_table(data)
    qpath, cpath = tmp_path / "queries.parquet", tmp_path / "context.parquet"
    pq.write_table(table.slice(5).select(["patient_id", *STREAMS]), qpath)
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
            "--device",
            "cpu",
        ]
    )
    p = pq.read_table(out).to_pandas()
    assert p.patient_id.tolist() == list(range(5, 10))
    assert (p.risk_treated - p.risk_control == p.risk_difference).all()
    assert predict(model, data, None, tmp_path / "split.parquet", context_size=4) == 6
    with pytest.raises(ValueError, match="disjoint"):
        predict(model, data, qpath, tmp_path / "bad.parquet")
