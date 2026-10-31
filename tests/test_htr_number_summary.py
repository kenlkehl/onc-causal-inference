import string
from dataclasses import asdict

import json
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from oci.config import ModelArchitectureConfig, omit_inactive_htr_numeric_settings
from oci.inference import standalone_htr
from oci.models.hierarchical_transformer_extractor import HierarchicalTransformerExtractor
from oci.models.htr_number_summary import ChunkNumberSummary, chunk_number_mentions

transformers = pytest.importorskip("transformers")

NOTE = "Hgb: 14.61 g/dL [Reference: 12.0-16.0] R05.9 71-year-old 1,234 5mg BRCA1"


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    """A random one-layer BERT whose WordPiece vocabulary spells words by character."""
    directory = tmp_path_factory.mktemp("tiny_bert")
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
    transformers.BertModel(config).save_pretrained(directory)
    return directory


def _extractor(path, *, numeric, pooling="token_attention", chunk_size_words=8, freeze=False):
    torch.manual_seed(1)
    extractor = HierarchicalTransformerExtractor(
        sentence_encoder_model=str(path),
        freeze_sentence_encoder=freeze,
        numeric_summary=numeric,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=2,
        max_chunks=16,
        max_chunk_length=128,
        transformer_dim=16,
        projection_dim=8,
        num_attention_heads=2,
        sentence_pooling=pooling,
    )
    torch.manual_seed(2)
    extractor.fit_tokenizer(["warmup"])
    return extractor.eval()


def test_numbers_are_anchored_at_their_first_token_and_identifiers_are_skipped(tiny_bert):
    extractor = _extractor(tiny_bert, numeric=True, chunk_size_words=40)
    tokenizer = extractor._tokenizer
    encoded = tokenizer(NOTE, return_offsets_mapping=True)
    mentions = chunk_number_mentions(NOTE, encoded["offset_mapping"])
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

    values = [value for _, value in mentions]
    assert values == pytest.approx([14.61, 12.0, 16.0, 71.0, 1234.0, 5.0])
    assert [tokens[index] for index, _ in mentions] == ["1", "1", "1", "7", "1", "5"]


def test_untrained_summary_leaves_the_model_and_later_random_draws_unchanged(tiny_bert):
    texts = [NOTE + " and more words here", "no digits in this note at all"]
    without = _extractor(tiny_bert, numeric=False)
    state_without = torch.get_rng_state()
    with_numbers = _extractor(tiny_bert, numeric=True)
    assert torch.equal(torch.get_rng_state(), state_without)
    with torch.no_grad():
        assert torch.equal(without(texts), with_numbers(texts))


def test_prepared_batches_match_the_text_path_and_chunks_without_numbers_keep_their_vectors(
    tiny_bert,
):
    extractor = _extractor(tiny_bert, numeric=True)
    torch.nn.init.normal_(extractor._number_summary.output.weight)
    texts = [NOTE + " and more words here", "no digits in this note at all"]
    preprocessor = extractor.make_batch_preprocessor()
    prepared = preprocessor(texts)
    assert prepared["number_chunk_index"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    # Loader workers receive the preprocessor by pickling.
    restored = pickle.loads(pickle.dumps(preprocessor))(texts)
    for key in ("number_chunk_index", "number_token_index", "number_value"):
        assert torch.equal(restored[key], prepared[key])
    with torch.no_grad():
        assert torch.allclose(extractor(texts), extractor(prepared), atol=1e-6)
        chunks = ["no digits here", "hgb 14.61 g/dl"]
        _, summaries = extractor._encode_chunks(
            chunks, extractor._number_mentions_for_chunks(chunks)
        )
    assert torch.count_nonzero(summaries[0]) == 0
    assert torch.count_nonzero(summaries[1]) > 0


@pytest.mark.parametrize("pooling", ["token_attention", "last"])
def test_anchor_positions_follow_left_and_right_padding(tiny_bert, pooling):
    extractor = _extractor(tiny_bert, numeric=True, pooling=pooling)
    chunks = ["a 7", "a much longer chunk before 42 and 3.5"]
    encoded = extractor._tokenize_chunks_for_transformers(chunks)
    numbers = extractor._number_mentions_for_chunks(chunks)
    rows, tokens, _ = extractor._number_anchor_positions(
        encoded["attention_mask"], numbers, 0, len(chunks)
    )
    anchored = encoded["input_ids"][rows, tokens].tolist()
    assert extractor._tokenizer.convert_ids_to_tokens(anchored) == ["7", "4", "3"]


def test_summary_attends_within_each_chunk_only():
    torch.manual_seed(0)
    summary = ChunkNumberSummary(hidden_dim=4, model_dim=8, num_queries=2)
    torch.nn.init.normal_(summary.output.weight)
    context = torch.randn(3, 4)
    values = torch.tensor([14.61, 210.0, 5.2])
    together = summary(context, values, torch.tensor([0, 0, 2]), 3)
    first = summary(context[:2], values[:2], torch.tensor([0, 0]), 1)
    last = summary(context[2:], values[2:], torch.tensor([0]), 1)
    assert torch.allclose(together[0], first[0], atol=1e-6)
    assert torch.allclose(together[2], last[0], atol=1e-6)
    assert torch.count_nonzero(together[1]) == 0


def test_summary_learns_a_value_next_to_a_distractor(tiny_bert):
    # The encoder is frozen so that only the chunk pooling, the document
    # transformer, and the number summary learn.
    rng = np.random.default_rng(0)

    def notes(n):
        hgb, plt = rng.uniform(8, 16, n), rng.uniform(150, 400, n)
        texts = [
            f"labs plt {p:.0f} k/ul hgb {h:.2f} g/dl"
            if first
            else f"labs hgb {h:.2f} g/dl plt {p:.0f} k/ul"
            for h, p, first in zip(hgb, plt, rng.random(n) < 0.5)
        ]
        return texts, torch.tensor((hgb - 12) / 2.3, dtype=torch.float32)

    train_texts, train_y = notes(256)
    test_texts, test_y = notes(256)
    r_squared = {}
    for numeric in (False, True):
        extractor = _extractor(
            tiny_bert, numeric=numeric, chunk_size_words=40, freeze=True
        ).train()
        head = torch.nn.Linear(8, 1)
        optimizer = torch.optim.AdamW(
            [*extractor.parameters(), *head.parameters()], lr=3e-3, weight_decay=0.0
        )
        torch.manual_seed(3)
        for _ in range(250):
            batch = torch.randint(0, len(train_texts), (32,))
            prediction = head(extractor([train_texts[i] for i in batch])).squeeze(1)
            loss = torch.nn.functional.mse_loss(prediction, train_y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        extractor.eval()
        with torch.no_grad():
            prediction = head(extractor(test_texts)).squeeze(1)
        r_squared[numeric] = float(1 - torch.mean((prediction - test_y) ** 2) / torch.var(test_y))
    assert r_squared[True] > 0.9
    assert r_squared[True] > r_squared[False] + 0.2


def test_numeric_summary_needs_token_states_and_valid_settings():
    with pytest.raises(ValueError, match="transformers backend"):
        HierarchicalTransformerExtractor(sentence_encoder_model="hash", numeric_summary=True)
    with pytest.raises(ValueError, match="transformers backend"):
        HierarchicalTransformerExtractor(
            sentence_encoder_model="sentence-transformers/all-MiniLM-L6-v2",
            freeze_sentence_encoder=True,
            numeric_summary=True,
        )
    with pytest.raises(ValueError, match="numeric_queries"):
        HierarchicalTransformerExtractor(sentence_encoder_model="hash", numeric_queries=0)
    with pytest.raises(ValueError, match="htr_numeric_queries"):
        ModelArchitectureConfig(htr_numeric_queries=0)
    with pytest.raises(TypeError, match="htr_numeric_summary"):
        ModelArchitectureConfig(htr_numeric_summary=1)


def test_identities_omit_the_settings_while_summaries_are_off():
    default = omit_inactive_htr_numeric_settings(asdict(ModelArchitectureConfig()))
    assert "htr_numeric_summary" not in default
    assert "htr_numeric_queries" not in default
    enabled = omit_inactive_htr_numeric_settings(
        asdict(ModelArchitectureConfig(htr_numeric_summary=True, htr_numeric_queries=2))
    )
    assert (enabled["htr_numeric_summary"], enabled["htr_numeric_queries"]) == (True, 2)

    # The Stage 1 template turns the summaries on.
    record = standalone_htr._config_record(*standalone_htr.build_configs(outcome_type="binary"))
    assert record["architecture"]["htr_numeric_summary"] is True
    record = standalone_htr._config_record(
        *standalone_htr.build_configs(
            outcome_type="binary", overrides=["architecture.htr_numeric_summary=false"]
        )
    )
    assert "htr_numeric_summary" not in record["architecture"]


def test_standalone_htr_trains_with_number_summaries(tmp_path, tiny_bert):
    rng = np.random.default_rng(0)
    n = 32
    hgb = rng.uniform(8, 16, n)
    platelets = rng.integers(150, 400, n)
    dataset = tmp_path / "dataset.parquet"
    pd.DataFrame(
        {
            "clinical_text": [
                f"labs hgb {h:.2f} g/dl plt {p} k/ul cough" for h, p in zip(hgb, platelets)
            ],
            "treatment_indicator": np.tile([0, 1], n // 2),
            "outcome_indicator": (hgb < 12).astype(int),
        }
    ).to_parquet(dataset, index=False)
    overrides = [
        "architecture.htr_numeric_summary=true",
        "architecture.htr_numeric_queries=2",
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
        "run.mlm_epochs=0",
    ]
    config, settings, joint = standalone_htr.build_configs(
        outcome_type="binary", htr_model=str(tiny_bert), overrides=overrides
    )
    out = tmp_path / "run"
    standalone_htr.run_standalone_htr(
        dataset_path=dataset,
        output_dir=out,
        config=config,
        settings=settings,
        joint=joint,
        devices=["cpu"],
    )
    predictions = pd.read_parquet(out / "predictions.parquet")
    assert len(predictions) == n
    assert np.isfinite(predictions["estimated_cate"]).all()
    run = json.loads((out / "run.json").read_text())
    assert run["identity"]["config"]["architecture"]["htr_numeric_queries"] == 2
