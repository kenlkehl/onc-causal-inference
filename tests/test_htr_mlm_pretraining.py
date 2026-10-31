import json
import string

import numpy as np
import pandas as pd
import pytest
import torch

from oci.inference import htr_mlm_pretraining, standalone_htr
from oci.models.hierarchical_transformer_extractor import HierarchicalTransformerExtractor

transformers = pytest.importorskip("transformers")

TINY = [
    "architecture.htr_sentence_pooling=token_attention",
    "architecture.htr_require_live_unfrozen_encoder=true",
    "architecture.htr_num_layers=1",
    "architecture.htr_num_heads=2",
    "architecture.htr_transformer_dim=16",
    "architecture.htr_projection_dim=8",
    "architecture.htr_chunk_size_words=8",
    "architecture.htr_chunk_overlap_words=2",
    "architecture.htr_max_chunks=4",
    "architecture.htr_max_chunk_length=64",
    "architecture.htr_prediction_head_hidden_dim=8",
    "training.batch_size=8",
    "training.effect_batch_size=8",
    "training.dataloader_workers=0",
    "run.outer_folds=2",
    "stage1_htr.nuisance_folds=2",
    "stage1_htr.effect_folds=2",
    "stage1_htr.nuisance_epochs=1",
    "stage1_htr.effect_epochs=1",
    "run.mlm_epochs=1",
    "run.mlm_batch_size=8",
]


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    """A random one-layer BERT with masked-LM heads and a character WordPiece vocabulary."""
    directory = tmp_path_factory.mktemp("tiny_bert_mlm")
    characters = list(string.ascii_lowercase + string.digits)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab += characters + list(".,:;-/()[]%") + ["##" + c for c in characters]
    (directory / "vocab.txt").write_text("\n".join(vocab) + "\n")
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=256,
    )
    transformers.BertForPreTraining(config).save_pretrained(directory)
    return directory


def _cohort(tmp_path, n=32):
    rng = np.random.default_rng(0)
    words = ["fatigue", "cough", "stage", "four", "ecog", "nausea", "pain", "egfr"]
    path = tmp_path / "dataset.parquet"
    pd.DataFrame(
        {
            "clinical_text": [
                " ".join(rng.choice(words, size=10)) + f" hgb {h:.1f}"
                for h in rng.uniform(8, 16, n)
            ],
            "treatment_indicator": np.tile([0, 1], n // 2),
            "outcome_indicator": np.tile([0, 0, 1, 1], n // 4),
            "true_ite": np.zeros(n),
        }
    ).to_parquet(path, index=False)
    return path


def _run(dataset, out, model, *extra):
    config, settings, joint = standalone_htr.build_configs(
        outcome_type="binary", htr_model=str(model), overrides=[*TINY, *extra]
    )
    return standalone_htr.run_standalone_htr(
        dataset_path=dataset,
        output_dir=out,
        config=config,
        settings=settings,
        joint=joint,
        devices=["cpu"],
    )


def _unit_rows(out, unit, fold=1):
    (complete,) = (
        out / "folds" / f"fold_{fold:02d}" / "stage1_checkpoints" / "units" / unit
    ).glob("*/complete.json")
    return json.loads(complete.read_text())["rows"]


def test_pretraining_starts_from_the_checkpoint_head_and_changes_the_encoder(tiny_bert):
    torch.manual_seed(1)
    extractor = HierarchicalTransformerExtractor(
        sentence_encoder_model=str(tiny_bert),
        chunk_size_words=8,
        chunk_overlap_words=2,
        max_chunks=4,
        max_chunk_length=64,
        transformer_dim=16,
        projection_dim=8,
        num_attention_heads=2,
    )
    extractor.fit_tokenizer(["warmup"])
    before = {k: v.clone() for k, v in extractor._sentence_encoder.state_dict().items()}
    texts = ["cough and fatigue with hgb 11.2", "stage four egfr pain"] * 4
    result = htr_mlm_pretraining.pretrain_chunk_encoder(
        extractor,
        texts,
        epochs=2,
        mask_probability=0.3,
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=4,
    )
    assert result["head"] == "pretrained"
    assert result["chunks"] == sum(len(c) for c in extractor._chunks_for_texts(texts))
    assert [entry["epoch"] for entry in result["history"]] == [1, 2]
    assert all(np.isfinite(entry["loss"]) and entry["masked_tokens"] > 0 for entry in result["history"])
    changed = [
        key for key, value in result["encoder"].items() if not np.allclose(value, before[key].numpy())
    ]
    assert any(key.startswith("encoder.layer.") for key in changed)


def test_each_network_pretrains_on_its_own_training_rows_and_resumes_identically(
    tmp_path, tiny_bert, monkeypatch
):
    dataset = _cohort(tmp_path)
    out = tmp_path / "run"
    _run(dataset, out, tiny_bert, "run.mlm_scope=network")
    first = pd.read_parquet(out / "predictions.parquet")
    for column in ("propensity", "m_hat", "estimated_cate"):
        assert np.isfinite(first[column]).all()
    for fold in (1, 2):
        for kind, network in (("nuisance", "htr/arm_nuisance"), ("effect", "htr/effect/dina")):
            for inner in (1, 2):
                corpus = _unit_rows(out, f"htr/mlm/{kind}/fold_{inner:03d}", fold)["corpus"]
                fit = _unit_rows(out, f"{network}/fold_{inner:03d}", fold)["fit"]
                # Never the network's validation rows or the outer held-out rows.
                assert corpus == fit
    record = json.loads((out / "folds" / "fold_01" / "fold.json").read_text())
    assert len(record["mlm_units"]) == 4
    history = json.loads((out / "folds" / "fold_01" / "mlm_pretraining.json").read_text())
    assert set(history) == set(record["mlm_units"])
    run = json.loads((out / "run.json").read_text())
    assert run["identity"]["config"]["run"]["mlm_epochs"] == 1

    # Keep only the pretraining units: the networks retrain from reloaded
    # weights and must draw exactly the numbers they drew the first time.
    for fold_dir in (out / "folds").iterdir():
        (fold_dir / "fold.json").unlink()
        units = fold_dir / "stage1_checkpoints" / "units"
        for unit in ("htr/arm_nuisance", "htr/effect", "assembly"):
            for path in sorted((units / unit).rglob("complete.json")):
                path.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("pretraining should have been reloaded")

    monkeypatch.setattr(htr_mlm_pretraining, "pretrain_chunk_encoder", forbidden)
    _run(dataset, out, tiny_bert, "run.mlm_scope=network")
    pd.testing.assert_frame_equal(first, pd.read_parquet(out / "predictions.parquet"))


def test_outer_fold_scope_pretrains_once_per_outer_fold_on_its_training_rows(
    tmp_path, tiny_bert, monkeypatch
):
    corpora = []
    pretrain = htr_mlm_pretraining.pretrain_chunk_encoder

    def counted(extractor, texts, **kwargs):
        corpora.append(len(texts))
        return pretrain(extractor, texts, **kwargs)

    monkeypatch.setattr(htr_mlm_pretraining, "pretrain_chunk_encoder", counted)
    dataset = _cohort(tmp_path)
    out = tmp_path / "run"
    _run(dataset, out, tiny_bert)  # the default scope
    training_rows = []
    for fold in (1, 2):
        units = out / "folds" / f"fold_{fold:02d}" / "stage1_checkpoints" / "units"
        assert [path.name for path in (units / "htr" / "mlm").iterdir()] == ["outer_fold"]
        corpus = _unit_rows(out, "htr/mlm/outer_fold", fold)["corpus"]
        # All of the outer training fold, and none of its held-out rows.
        assert corpus == _unit_rows(out, "assembly/nuisance_ensemble", fold)["train"]
        training_rows.append(corpus["count"])
        record = json.loads((out / "folds" / f"fold_{fold:02d}" / "fold.json").read_text())
        assert record["mlm_units"] == ["htr/mlm/outer_fold"]
    # Two nuisance and two effect networks per outer fold, one pretraining.
    assert corpora == training_rows
    run = json.loads((out / "run.json").read_text())
    assert run["identity"]["config"]["run"]["mlm_scope"] == "outer_fold"


def test_pretraining_settings_are_validated_and_left_out_of_identities_while_off():
    with pytest.raises(ValueError, match="run.mlm_epochs"):
        standalone_htr.StandaloneHTRConfig(mlm_epochs=-1)
    with pytest.raises(ValueError, match="run.mlm_scope"):
        standalone_htr.StandaloneHTRConfig(mlm_scope="everything")
    with pytest.raises(ValueError, match="run.mlm_mask_probability"):
        standalone_htr.StandaloneHTRConfig(mlm_mask_probability=1.0)
    with pytest.raises(ValueError, match="needs run.method=two_stage"):
        standalone_htr.build_configs(
            outcome_type="binary", overrides=["run.method=joint", "run.mlm_epochs=1"]
        )
    def run_record(*overrides):
        configs = standalone_htr.build_configs(outcome_type="binary", overrides=overrides)
        return standalone_htr._config_record(*configs)["run"]

    mlm_keys = {*standalone_htr.MLM_SETTINGS, "mlm_scope"}
    assert not mlm_keys & set(run_record("run.mlm_epochs=0"))
    assert not mlm_keys & set(run_record("run.method=joint"))
    default = run_record()
    assert (default["mlm_epochs"], default["mlm_scope"]) == (3, "outer_fold")
    # Per-network pretraining keeps the identity it had before mlm_scope existed.
    assert "mlm_scope" not in run_record("run.mlm_scope=network")
