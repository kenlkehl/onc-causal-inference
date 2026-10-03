from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import re
import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

from oci.colbert_config import ColBERTConfig, colbert_config_from_mapping
from oci.extraction import colbert


class Tokenizer:
    def __call__(self, text, **kwargs):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


@pytest.fixture
def backend(monkeypatch, tmp_path):
    from oci.extraction import colbert_encoder

    state = {"documents": 0, "queries": 0, "devices": set(), "active": 0, "peak": 0}
    lock = threading.Lock()

    class Encoder:
        def __init__(self, config, device, folder):
            self.tokenizer = Tokenizer()
            self.device = device

        def encode(self, texts, *, query=False):
            with lock:
                state["queries" if query else "documents"] += len(texts)
                state["devices"].add(self.device)
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.005)
            result = []
            for text in texts:
                axis = 0 if "red" in text.lower() else 1
                result.append(np.eye(2, dtype=np.float32)[[axis]])
            with lock:
                state["active"] -= 1
            return result

    monkeypatch.setattr(colbert_encoder, "ColBERTEncoder", Encoder)
    monkeypatch.setattr(
        colbert_encoder,
        "resolve_checkpoint",
        lambda config: (tmp_path, {"revision": "test", "artifact_sha256": "weights-v1"}),
    )
    actual_score = colbert.maxsim_scores
    monkeypatch.setattr(colbert, "maxsim_scores", lambda q, d, **kwargs: actual_score(q, d))
    # Avoid requiring an installed torch just to test worker scheduling.
    import sys

    if "torch" not in sys.modules:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            from types import SimpleNamespace

            monkeypatch.setitem(
                sys.modules,
                "torch",
                SimpleNamespace(
                    Tensor=type("Tensor", (), {}), cuda=SimpleNamespace(device_count=lambda: 0)
                ),
            )
    yield state
    with colbert._retrievers_lock:
        for retriever in colbert._retrievers.values():
            for worker in retriever.workers:
                worker.executor.shutdown(wait=True)
        colbert._retrievers.clear()


def settings(tmp_path, **kwargs):
    return ColBERTConfig(
        cache_dir=str(tmp_path / "cache"), devices=("cpu",), chunk_size=2, **kwargs
    )


def test_unicode_chunks_cover_source_and_preserve_overlap(tmp_path):
    text = "  red 漢字\nblue — words  "
    chunks = colbert.chunk_text(text, Tokenizer(), settings(tmp_path))
    assert "".join(c["text"] for c in chunks) == text
    assert all(text[c["start"] : c["end"]] == c["text"] for c in chunks)
    overlap = colbert.chunk_text(text, Tokenizer(), replace(settings(tmp_path), chunk_overlap=1))
    assert overlap[1]["start"] < overlap[0]["end"]
    context = colbert.render_context(text, overlap)
    assert context.count(text) == 1


def test_maxsim_uses_token_maxima_and_stable_ties():
    query = np.eye(2, dtype=np.float32)
    docs = [
        np.eye(2, dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
        np.eye(2, dtype=np.float32),
    ]
    assert colbert.maxsim_scores(query, docs).tolist() == [2, 1, 2]


def test_torch_maxsim_masks_padding_even_when_all_real_similarities_are_negative():
    import torch

    query = np.array([[1, 0]], dtype=np.float32)
    docs = [
        np.array([[-1, 0]], dtype=np.float32),
        np.array([[-1, 0], [-0.6, 0.8]], dtype=np.float32),
    ]
    np.testing.assert_allclose(
        colbert.maxsim_scores(query, docs, device=torch.device("cpu"), batch_size=2),
        [-1, -0.6],
        atol=1e-6,
    )


def test_cache_reuse_new_questions_devices_and_top_k(tmp_path, backend):
    config = settings(tmp_path, top_k=1)
    retriever = colbert.get_retriever(config)
    first = retriever.retrieve("red first blue last", [{"name": "red"}], top_k=1)
    assert first["hits"][0][0]["text"] == "red first "
    documents = backend["documents"]
    second = retriever.retrieve("red first blue last", [{"name": "blue"}], top_k=2)
    assert second["cache_hit"] and backend["documents"] == documents
    assert second["index_key"] == first["index_key"] and len(second["hits"][0]) == 2
    other = colbert.get_retriever(replace(config, devices=("cuda:1",), batch_size=1))
    third = other.retrieve("red first blue last", [{"name": "red"}], top_k=1)
    assert third["cache_hit"] and backend["documents"] == documents
    assert Path(first["index_path"]).stat().st_mode & 0o777 == 0o600


def test_source_model_and_chunk_settings_invalidate_vectors(tmp_path, backend, monkeypatch):
    from oci.extraction import colbert_encoder

    config = settings(tmp_path)
    first = colbert.get_retriever(config).retrieve("red first blue last", [{"name": "red"}])
    changed = colbert.get_retriever(config).retrieve("red NEW blue last", [{"name": "red"}])
    assert first["index_key"] != changed["index_key"]
    geometry = colbert.get_retriever(replace(config, chunk_size=3)).retrieve(
        "red first blue last", [{"name": "red"}]
    )
    assert geometry["index_key"] != first["index_key"]
    monkeypatch.setattr(
        colbert_encoder,
        "resolve_checkpoint",
        lambda config: (tmp_path, {"revision": "v2", "artifact_sha256": "weights-v2"}),
    )
    weights = colbert.ColBERTRetriever(config).retrieve("red first blue last", [{"name": "red"}])
    assert weights["index_key"] != first["index_key"]


def test_corrupt_cache_rebuilt_and_no_patient_cross_retrieval(tmp_path, backend):
    retriever = colbert.get_retriever(settings(tmp_path))
    result = retriever.retrieve("red patient", [{"name": "red"}])
    path = Path(result["index_path"])
    path.write_bytes(b"invalid-npz")
    repaired = retriever.retrieve("red patient", [{"name": "red"}])
    assert not repaired["cache_hit"]
    other = retriever.retrieve("blue other", [{"name": "red"}])
    assert "red patient" not in other["context"]
    assert other["hits"][0][0]["text"] == "blue other"


def test_parallel_gpu_workers_and_atomic_shared_patient_cache(tmp_path, backend):
    config = replace(settings(tmp_path), devices=("cuda:0", "cuda:1"))
    retriever = colbert.get_retriever(config)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda i: retriever.retrieve(
                    "red same patient" if i < 4 else f"blue patient {i}", [{"name": "red"}]
                ),
                range(8),
            )
        )
    assert backend["devices"] == {"cuda:0", "cuda:1"} and backend["peak"] == 2
    assert len({r["index_key"] for r in results[:4]}) == 1
    assert sum(not r["cache_hit"] for r in results[:4]) == 1
    assert backend["documents"] == 10  # five unique three-word documents, two chunks each


def test_config_and_feature_query_exclude_roles_and_outcomes(tmp_path):
    assert colbert_config_from_mapping({"devices": "cuda:0,cuda:1"}).devices == ("cuda:0", "cuda:1")
    for options in (
        {"top_k": 0},
        {"chunk_overlap": 64},
        {"devices": []},
        {"devices": ["auto", "cpu"]},
        {"made_up": 3},
    ):
        with pytest.raises((ValueError, TypeError)):
            colbert_config_from_mapping(options)
    query = colbert.feature_query(
        {
            "name": "red",
            "description": "clinical state",
            "roles": ["effect_modifier"],
            "outcome": 42,
            "patient_id": "secret",
        }
    )
    assert "clinical state" in query and all(
        t not in query for t in ("effect_modifier", "42", "secret")
    )


def test_stage2_retrieval_checkpoint_reuse_and_patient_request_parallelism(
    tmp_path, backend, monkeypatch
):
    from oci.inference import plain_handoff_stage2_analysis as analysis
    from tests.stage2_prompt_spy import install, prompt_inputs

    install(monkeypatch)
    definitions = [
        {
            "name": "state",
            "description": "red state",
            "value_type": "categorical",
            "categories_or_unit": ["red", "blue"],
            "measurement_definition": "Use documented state",
            "missing_value_rule": "Return null",
            "conflict_resolution": {"strategy": "single_or_null"},
        }
    ]
    calls = []
    barrier = threading.Barrier(2)

    def request(messages, validate, *, request_kind):
        body = prompt_inputs(messages)
        row = body["patients"][0]
        assert "[oci_colbert_v1]" in row["text"]
        calls.append(row["row_id"])
        barrier.wait(timeout=10)
        return validate({"values": {"state": "red" if "red patient" in row["text"] else "blue"}})

    kwargs = dict(
        dataset=pd.DataFrame({"text": ["red patient", "blue patient"]}),
        row_ids=[1, 0],
        text_column="text",
        definitions=definitions,
        output_dir=tmp_path / "output",
        request_json=request,
        workers=2,
        max_prompt_chars=100_000,
        context_strategy="colbert",
        colbert=settings(tmp_path),
    )
    frame = analysis.extract_rows(**kwargs)
    assert frame["_oci_row_id"].tolist() == [1, 0] and frame["state"].tolist() == ["blue", "red"]
    pd.testing.assert_frame_equal(analysis.extract_rows(**kwargs), frame)
    assert len(calls) == 2
    assert len(list((tmp_path / "output").rglob("retrieval.json"))) == 2
    with pytest.raises(ValueError, match="differ from saved"):
        analysis.extract_rows(**{**kwargs, "colbert": replace(settings(tmp_path), top_k=1)})


def test_standalone_prompt_retrieval_and_overflow_fail_closed(tmp_path, backend):
    from oci.config import ExplicitFeatureSpec, ExplicitFeatureExtractionConfig
    from oci.extraction.explicit_features import build_extraction_prompt, VLLMFeatureExtractor

    config = settings(tmp_path, top_k=1)
    feature = ExplicitFeatureSpec(
        "red", "continuous", description="red value", roles=["confounder"]
    )
    assert ExplicitFeatureExtractionConfig().extraction_context_strategy == "colbert"
    assert VLLMFeatureExtractor([feature], colbert=config).context_strategy == "colbert"
    prompt = build_extraction_prompt(
        "red value blue irrelevant", [feature], context_strategy="colbert", colbert=config
    )
    assert "red value" in prompt and "blue irrelevant" not in prompt
    with pytest.raises(ValueError, match="exceeds"):
        build_extraction_prompt(
            "red value", [feature], context_strategy="colbert", colbert=config, max_text_length=1
        )


def test_stage2_default_roundtrip_and_mode_use_only_retrieved_source(
    tmp_path, backend, monkeypatch
):
    from dataclasses import asdict
    from oci.inference import plain_handoff_stage2 as stage2
    from oci.inference import plain_handoff_stage2_analysis as analysis
    from tests.stage2_prompt_spy import install, prompt_inputs

    install(monkeypatch)
    config = stage2.PlainHandoffStage2Config(
        endpoint="http://unused/v1", model="test", colbert=settings(tmp_path, top_k=1)
    )
    assert config.extraction_context_strategy == "colbert"
    assert colbert_config_from_mapping(asdict(config)["colbert"]) == config.colbert
    selected = analysis._configured_serial_extraction(config)
    assert selected["context_strategy"] == "colbert" and selected["colbert"] == config.colbert
    definition = {
        "name": "state",
        "description": "red state",
        "value_type": "categorical",
        "categories_or_unit": ["red", "blue"],
        "measurement_definition": "Use modal state",
        "missing_value_rule": "Return null",
        "conflict_resolution": {"strategy": "mode"},
    }

    def request(messages, validate, *, request_kind):
        patient = prompt_inputs(messages)["patient"]
        assert "blue blue" not in patient["text"] and "red red" in patient["text"]
        return validate(
            {
                "observations": [
                    {
                        "feature": "state",
                        "value": "red",
                        "quote": "red red",
                        "governing_date_quote": None,
                    }
                ]
            }
        )

    frame = analysis.extract_rows(
        dataset=pd.DataFrame({"text": ["red red blue blue blue"]}),
        row_ids=[0],
        text_column="text",
        definitions=[definition],
        output_dir=tmp_path / "mode",
        request_json=request,
        workers=1,
        max_prompt_chars=100_000,
        **selected,
    )
    assert frame.loc[0, "state"] == "red"


def test_feature_refinement_reuses_document_vectors(tmp_path, backend, monkeypatch):
    from oci.inference import plain_handoff_stage2_analysis as analysis
    from tests.stage2_prompt_spy import install

    install(monkeypatch)
    first = {
        "name": "state",
        "feature_id": "state-feature",
        "description": "red state",
        "value_type": "categorical",
        "categories_or_unit": ["red", "blue"],
        "measurement_definition": "Use stated state",
        "missing_value_rule": "Return null",
        "conflict_resolution": {"strategy": "single_or_null"},
    }
    dataset = pd.DataFrame({"text": ["red patient"]})
    requests = []

    def request(messages, validate, *, request_kind):
        requests.append(messages)
        return validate({"values": {"state": "red"}})

    kwargs = dict(
        dataset=dataset,
        row_ids=[0],
        text_column="text",
        request_json=request,
        workers=1,
        max_prompt_chars=100_000,
        feature_batch_size=10,
        request_identity={},
        tokenizer=None,
        chunk_size_tokens=50000,
        context_window_tokens=128000,
        max_output_tokens=1000,
        context_margin_tokens=100,
        context_strategy="colbert",
        colbert=settings(tmp_path),
    )
    prior = analysis.extract_rows(definitions=[first], output_dir=tmp_path / "initial", **kwargs)
    count = backend["documents"]
    refined, _ = analysis._extract_changed_features_and_merge(
        definitions=[{**first, "description": "A refined red state"}],
        prior_definitions=[first],
        prior_extracted=prior,
        prior_failure_summary=json.loads((tmp_path / "initial/failure_summary.json").read_text()),
        output_dir=tmp_path / "refined",
        **kwargs,
    )
    assert refined.loc[0, "state"] == "red" and len(requests) == 2
    assert backend["documents"] == count


def test_real_local_checkpoint_markers_projection_masks_and_capacity(tmp_path):
    """Exercise the actual encoder without network access or a GPU."""
    import json
    import torch
    from safetensors.torch import save_file
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from oci.extraction.colbert_encoder import ColBERTEncoder, resolve_checkpoint

    vocab = {
        word: i
        for i, word in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[Q]", "[D]", "red", "blue", "."]
        )
    }
    tokenizer = BertTokenizerFast(vocab=vocab)
    tokenizer.save_pretrained(tmp_path)
    model = BertModel(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=64,
        )
    )
    model.save_pretrained(tmp_path, safe_serialization=True)
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {"type": "sentence_transformers.models.Transformer", "path": ""},
                {"type": "pylate.models.Dense.Dense", "path": "1_Dense"},
            ]
        )
    )
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps(
            {
                "query_prefix": "[Q]",
                "document_prefix": "[D]",
                "skiplist_words": ["."],
                "do_query_expansion": True,
                "attend_to_expansion_tokens": False,
            }
        )
    )
    (tmp_path / "1_Dense").mkdir()
    (tmp_path / "1_Dense/config.json").write_text(
        json.dumps(
            {
                "in_features": 8,
                "out_features": 4,
                "bias": False,
                "activation_function": "torch.nn.modules.linear.Identity",
            }
        )
    )
    save_file({"linear.weight": torch.eye(4, 8)}, tmp_path / "1_Dense/model.safetensors")
    config = ColBERTConfig(model_name=str(tmp_path), devices=("cpu",), query_length=8, chunk_size=4)
    folder, signature = resolve_checkpoint(config)
    encoder = ColBERTEncoder(config, "cpu", folder)
    observed = []
    encoder.model.register_forward_pre_hook(
        lambda model, args, kwargs: observed.append(
            {k: v.detach().clone() for k, v in kwargs.items()}
        ),
        with_kwargs=True,
    )
    docs = encoder.encode(["red .", "red blue ."])
    query = encoder.encode(["red"], query=True)[0]
    assert [v.shape for v in docs] == [(4, 4), (5, 4)]
    assert observed[0]["input_ids"][:, 1].tolist() == [vocab["[D]"], vocab["[D]"]]
    assert observed[1]["input_ids"][0, 1].item() == vocab["[Q]"]
    assert query.shape == (8, 4)
    for vectors in [*docs, query]:
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    with pytest.raises(ValueError, match="no text was truncated"):
        encoder.encode(["red " * 20], query=True)
    save_file({"linear.weight": torch.ones(4, 8)}, tmp_path / "1_Dense/model.safetensors")
    assert resolve_checkpoint(config)[1]["artifact_sha256"] != signature["artifact_sha256"]
