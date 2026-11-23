"""Offline CPU unit tests for TinyOncBERT curation, distillation, and model export."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel, BertTokenizer

from oci.longitudinal.tinyoncbert import (
    POLARITY_AFFIRMED,
    POLARITY_NEGATED,
    POLARITY_UNCERTAIN,
    TinyOncBertConfig,
    classify_polarity,
    curate_corpus,
    flip_clinical_negation,
    initialize_word_embeddings_svd,
    precompute_teacher_embeddings,
    strip_markdown,
    train_tinyoncbert,
)


def _build_offline_bert_tokenizer(tmp_path: Path) -> BertTokenizer:
    """Create a tiny self-contained offline WordPiece tokenizer for unit tests."""
    vocab_tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        ".",
        ",",
        ";",
        ":",
        "-",
        "/",
        "patient",
        "denies",
        "reports",
        "reported",
        "no",
        "evidence",
        "of",
        "metastatic",
        "disease",
        "in",
        "the",
        "liver",
        "lung",
        "brain",
        "bone",
        "lymph",
        "node",
        "biopsy",
        "negative",
        "positive",
        "for",
        "malignancy",
        "egfr",
        "her2",
        "mutation",
        "without",
        "with",
        "acute",
        "cardiopulmonary",
        "abnormality",
        "pleural",
        "effusion",
        "noted",
        "unremarkable",
        "abnormal",
        "mri",
        "ct",
        "chest",
        "contrast",
        "shows",
        "hepatic",
        "lesion",
        "fever",
        "chills",
        "fatigue",
        "dyspnea",
        "pain",
        "tumor",
        "stage",
        "iv",
        "adenocarcinoma",
        "chemotherapy",
        "carboplatin",
        "pemetrexed",
        "pembrolizumab",
        "cannot",
        "rule",
        "out",
        "suspicious",
        "pending",
        "follow",
        "up",
        "clinic",
        "three",
        "weeks",
        "stable",
        "response",
        "progression",
        "new",
        "nodules",
        "present",
        "absent",
        "oncology",
        "radiology",
        "pathology",
        "note",
        "assessment",
        "plan",
        "findings",
        "history",
        "imaging",
        "scan",
        "mass",
        "normal",
        "elevated",
        "is",
        "are",
        "was",
        "not",
        "a",
        "an",
        "and",
        "or",
        "on",
        "at",
        "to",
        "from",
        "##s",
        "##ed",
        "##ing",
    ]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
    return BertTokenizer(vocab=str(vocab_file), do_lower_case=True)


def _write_synthetic_notes_parquet(parquet_path: Path) -> None:
    """Create a small multi-row-group local parquet with synthetic oncology notes."""
    templates_by_type = {
        "oncology_note": [
            "### Oncology Assessment\n**Interval History:** Patient denies fever, chills, or acute pain.\n---\nNo evidence of metastatic disease in the liver. Patient reports fatigue on chemotherapy; follow up in clinic in three weeks.",
            "## Progress Note\nPatient reports progressive dyspnea and chest pain. Biopsy positive for EGFR mutation. Tumor is HER2-negative adenocarcinoma; plan to continue carboplatin and pemetrexed.",
            "**Follow-up Visit**\nDenies acute dyspnea or fever. Stable response on pembrolizumab. Cannot rule out bone lesion on bone scan; pending pathology review.",
        ],
        "radiology": [
            "### CT Chest With Contrast\nNo pleural effusion or acute cardiopulmonary abnormality. Unremarkable brain MRI with contrast. Evidence of stable lung nodules in the right lobe.",
            "--- \n**Imaging Findings:** CT chest without contrast shows hepatic lesion and new lung nodules. Lymph node biopsy negative for malignancy. Suspicious for metastatic disease versus benign mass.",
            "## Radiology Report\nWithout acute cardiopulmonary abnormality. Pleural effusion noted on chest imaging. Abnormal liver findings on staging scan.",
        ],
        "pathology": [
            "**Pathology Summary**\nLymph node biopsy negative for malignancy. Tumor is positive for EGFR mutation. No evidence of residual disease in resected mass.",
            "### Surgical Pathology\nBiopsy positive for adenocarcinoma. HER2-negative and EGFR positive for exon mutation. Evidence of focal lymph node involvement noted on histology.",
            "## Molecular Report\nTested negative for ALK rearrangement. Abnormal cellularity noted in lung biopsy; cannot rule out metastatic progression.",
        ],
    }

    notes: list[str] = []
    event_types: list[str] = []
    splits: list[str] = []

    # Generate enough distinct train notes across 3 event types + some test rows to test filtering
    for i in range(60):
        etype = ("oncology_note", "radiology", "pathology")[i % 3]
        base = templates_by_type[etype][i % len(templates_by_type[etype])]
        # Repeat slightly varied clauses so sentences are unique and long notes yield 512-tok windows
        extra = (
            f"Visit {i} oncology update: Patient denies acute fever during cycle {i} therapy. "
            f"CT chest shows stable lung mass on visit {i} imaging. "
            f"No evidence of liver disease on scan {i} review. "
            f"Lymph node biopsy positive for tumor cells in specimen {i} pathology. "
            f"Unremarkable brain MRI at assessment {i} clinic. "
            f"Without pleural effusion on imaging review {i} today. "
            f"Cannot rule out bone lesion in region {i} currently."
        )
        notes.append(f"{base}\n{extra}\n{extra}")
        event_types.append(etype)
        splits.append("train")

    # Add held-out test split rows that MUST be filtered out during curation
    for j in range(6):
        notes.append(f"HELD_OUT_TEST_MARKER_{j} patient reports severe pain and fever.")
        event_types.append("oncology_note")
        splits.append("test")

    table = pa.table(
        {
            "synthetic_note": pa.array(notes, type=pa.string()),
            "event_type": pa.array(event_types, type=pa.string()),
            "split": pa.array(splits, type=pa.string()),
            "unused_heavy_col": pa.array(["x" * 50] * len(notes), type=pa.string()),
        }
    )
    pq.write_table(table, parquet_path, row_group_size=15)


class _DummyCausalTeacher(nn.Module):
    """Minimal causal decoder-style teacher verifying ``use_cache=False`` and last-token pooling."""

    def __init__(self, vocab_size: int, hidden_size: int = 48):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, model_type="qwen3")
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.saw_use_cache_false = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
    ):
        assert use_cache is False, "Causal teacher must be invoked with use_cache=False"
        self.saw_use_cache_false = True
        h = torch.tanh(self.proj(self.embed(input_ids)))
        return SimpleNamespace(last_hidden_state=h)


def test_strip_markdown_removes_formatting_and_preserves_clinical_text():
    raw = (
        "### Oncology Note\n"
        "---\n"
        "**Assessment:** Patient has *stage IV* HER2_positive NSCLC.\n"
        "- Denies `fever` or chills.\n"
        "***\n"
    )
    cleaned = strip_markdown(raw)
    assert "###" not in cleaned
    assert "---" not in cleaned
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert "Oncology Note" in cleaned
    assert "Assessment: Patient has stage IV HER2_positive NSCLC." in cleaned
    assert "Denies fever or chills." in cleaned


@pytest.mark.parametrize(
    "sentence,expected_polarity,expected_substring",
    [
        (
            "Denies fever, chills, or night sweats.",
            POLARITY_NEGATED,
            "Reports fever, chills, or night sweats.",
        ),
        (
            "Patient reports progressive dyspnea on exertion.",
            POLARITY_AFFIRMED,
            "Patient denies progressive dyspnea on exertion.",
        ),
        (
            "No evidence of metastatic disease in the liver.",
            POLARITY_NEGATED,
            "Evidence of metastatic disease in the liver.",
        ),
        (
            "Evidence of recurrent pulmonary nodules on CT.",
            POLARITY_AFFIRMED,
            "No evidence of recurrent pulmonary nodules on CT.",
        ),
        (
            "Lymph node biopsy negative for malignancy.",
            POLARITY_NEGATED,
            "Lymph node biopsy positive for malignancy.",
        ),
        (
            "Tumor is positive for EGFR exon 19 deletion.",
            POLARITY_AFFIRMED,
            "Tumor is negative for EGFR exon 19 deletion.",
        ),
        (
            "Without acute cardiopulmonary abnormality.",
            POLARITY_NEGATED,
            "With acute cardiopulmonary abnormality.",
        ),
        (
            "With significant mediastinal lymphadenopathy.",
            POLARITY_AFFIRMED,
            "Without significant mediastinal lymphadenopathy.",
        ),
        (
            "No pleural effusion or pneumothorax.",
            POLARITY_NEGATED,
            "Pleural effusion or pneumothorax noted.",
        ),
        (
            "Pleural effusion noted on chest radiograph.",
            POLARITY_AFFIRMED,
            "No pleural effusion on chest radiograph.",
        ),
        (
            "Unremarkable brain MRI with contrast.",
            POLARITY_NEGATED,
            "Abnormal brain MRI with contrast.",
        ),
        (
            "Abnormal liver function tests on routine labs.",
            POLARITY_AFFIRMED,
            "Unremarkable liver function tests on routine labs.",
        ),
        (
            "Tumor is HER2-negative invasive ductal carcinoma.",
            POLARITY_NEGATED,
            "Tumor is HER2-positive invasive ductal carcinoma.",
        ),
        (
            "CT chest without contrast shows hepatic metastases.",
            POLARITY_AFFIRMED,
            "CT chest without contrast shows no hepatic metastases.",
        ),
    ],
)
def test_polarity_and_bidirectional_negation_flip(
    sentence: str, expected_polarity: int, expected_substring: str
):
    assert classify_polarity(sentence) == expected_polarity
    flipped = flip_clinical_negation(sentence)
    assert flipped == expected_substring
    assert classify_polarity(flipped) == (1 - expected_polarity)


@pytest.mark.parametrize(
    "sentence,expected_polarity",
    [
        ("Cannot rule out osseous metastasis in T8.", POLARITY_UNCERTAIN),
        ("Suspicious for hepatic lesion versus hemangioma.", POLARITY_UNCERTAIN),
        ("Pending next-generation sequencing panel.", POLARITY_UNCERTAIN),
        ("Follow up in oncology clinic in three weeks.", POLARITY_AFFIRMED),
    ],
)
def test_uncertain_and_nonflippable_sentences_return_none(sentence: str, expected_polarity: int):
    assert classify_polarity(sentence) == expected_polarity
    assert flip_clinical_negation(sentence) is None


def test_curation_and_teacher_precomputation_resumable_and_pooling_modes(tmp_path: Path):
    tokenizer = _build_offline_bert_tokenizer(tmp_path)
    parquet_path = tmp_path / "synthetic_notes.parquet"
    _write_synthetic_notes_parquet(parquet_path)

    cfg = TinyOncBertConfig(
        parquet=str(parquet_path),
        work_dir=str(tmp_path / "work"),
        output_dir=str(tmp_path / "out"),
        num_sentences=90,
        num_windows=12,
        window_max_length=512,
        teacher_model="dummy/causal-teacher",
        teacher_pooling="auto",
        teacher_batch_size=16,
        device="cpu",
        seed=7,
    )

    manifest1 = curate_corpus(cfg, tokenizer=tokenizer)
    assert manifest1["num_sentences_actual"] == 90
    assert manifest1["num_windows_actual"] > 0
    assert manifest1["num_flippable"] > 0
    # Balanced across all 3 event types
    assert set(manifest1["event_type_counts"].keys()) == {"oncology_note", "pathology", "radiology"}
    counts = list(manifest1["event_type_counts"].values())
    assert max(counts) - min(counts) <= 2

    # Ensure split=='test' notes were filtered out
    curated_dir = Path(cfg.work_dir) / "curated"
    sent_table = pq.read_table(curated_dir / manifest1["sentence_shards"][0])
    texts = sent_table["text"].to_pylist()
    assert not any("HELD_OUT_TEST_MARKER" in t for t in texts)
    # Ensure deduplication (case-insensitive)
    assert len({t.lower() for t in texts}) == len(texts)

    # Second call reuses cache immediately
    manifest2 = curate_corpus(cfg, tokenizer=tokenizer)
    assert manifest2 == manifest1

    # Causal teacher precomputation with auto -> last pooling + use_cache=False
    causal_teacher = _DummyCausalTeacher(vocab_size=len(tokenizer), hidden_size=48)
    emb_path, t_dim = precompute_teacher_embeddings(
        cfg,
        texts,
        manifest1["signature"],
        teacher_model_obj=causal_teacher,
        teacher_tokenizer_obj=tokenizer,
    )
    assert causal_teacher.saw_use_cache_false
    assert t_dim == 48
    embs = np.load(emb_path, mmap_mode="r")
    assert embs.shape == (len(texts), 48)
    assert embs.dtype == np.float16
    norms = np.linalg.norm(embs.astype(np.float32), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-2)

    # Re-calling with matching cache does not recompute (passing None model would fail if it tried)
    emb_path_cached, t_dim_cached = precompute_teacher_embeddings(
        cfg,
        texts,
        manifest1["signature"],
        teacher_model_obj=None,
        teacher_tokenizer_obj=None,
    )
    assert emb_path_cached == emb_path
    assert t_dim_cached == 48

    # Mean-pooling bidirectional teacher test
    cfg_mean = TinyOncBertConfig(
        parquet=str(parquet_path),
        work_dir=str(tmp_path / "work_mean"),
        output_dir=str(tmp_path / "out_mean"),
        teacher_model="dummy/bert-teacher",
        teacher_pooling="auto",
        teacher_batch_size=16,
        device="cpu",
    )
    bert_teacher = BertModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=40,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=80,
            max_position_embeddings=128,
        )
    )
    emb_path_m, t_dim_m = precompute_teacher_embeddings(
        cfg_mean,
        texts,
        manifest1["signature"],
        teacher_model_obj=bert_teacher,
        teacher_tokenizer_obj=tokenizer,
    )
    assert t_dim_m == 40
    meta_m = json.loads((Path(cfg_mean.work_dir) / "teacher_meta.json").read_text(encoding="utf-8"))
    assert meta_m["teacher_pooling_resolved"] == "mean"
    assert np.load(emb_path_m).shape == (len(texts), 40)


def test_svd_word_embedding_warm_start(tmp_path: Path):
    tokenizer = _build_offline_bert_tokenizer(tmp_path)
    student = BertModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=512,
        )
    )
    torch.manual_seed(0)
    src_emb = torch.randn(len(tokenizer), 64) * 0.05
    before = student.embeddings.word_embeddings.weight.detach().clone()
    ok = initialize_word_embeddings_svd(student, src_emb, student_tokenizer=tokenizer)
    assert ok is True
    after = student.embeddings.word_embeddings.weight.detach()
    assert after.shape == (len(tokenizer), 32)
    assert not torch.allclose(before, after)
    assert torch.isfinite(after).all()
    assert abs(float(after.std()) - float(src_emb.std())) < 1e-2


def test_end_to_end_training_and_clean_automodel_export(tmp_path: Path):
    tokenizer = _build_offline_bert_tokenizer(tmp_path)
    parquet_path = tmp_path / "synthetic_notes.parquet"
    _write_synthetic_notes_parquet(parquet_path)

    teacher = _DummyCausalTeacher(vocab_size=len(tokenizer), hidden_size=48)
    torch.manual_seed(123)
    svd_source = torch.randn(len(tokenizer), 64) * 0.04

    output_dir = tmp_path / "tinyoncbert_out"
    cfg = TinyOncBertConfig(
        parquet=str(parquet_path),
        work_dir=str(tmp_path / "work"),
        output_dir=str(output_dir),
        num_sentences=96,
        num_windows=12,
        val_sentences=16,
        val_windows=4,
        sentence_max_length=64,
        window_max_length=512,
        teacher_model="dummy/causal-teacher",
        teacher_pooling="last",
        teacher_batch_size=32,
        svd_init_model="",
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        lambda_emb=1.0,
        lambda_rel=0.5,
        lambda_neg=0.5,
        lambda_pol=0.1,
        lambda_mlm=0.25,
        neg_margin=0.25,
        window_mlm_every=2,
        epochs=6,
        batch_size=16,
        window_batch_size=4,
        lr=3e-3,
        warmup_ratio=0.05,
        log_every=1,
        eval_every=10,
        num_workers=0,
        device="cpu",
        seed=42,
    )

    result = train_tinyoncbert(
        cfg,
        tokenizer_obj=tokenizer,
        teacher_model_obj=teacher,
        teacher_tokenizer_obj=tokenizer,
        svd_init_source=svd_source,
    )

    assert result["total_steps"] >= 20
    assert result["window_mlm_steps"] >= 10

    train_records = [r for r in result["history"] if r["type"] == "train"]
    val_records = [r for r in result["history"] if r["type"] == "val"]
    assert len(train_records) >= 20
    assert len(val_records) >= 2
    assert any("window_mlm_loss" in r for r in train_records)

    # All 4 loss terms ran cleanly and total loss decreased meaningfully
    for rec in train_records:
        for key in ("loss", "emb_loss", "rel_loss", "neg_loss", "pol_loss", "mlm_loss"):
            assert np.isfinite(rec[key]), f"Non-finite {key} at step {rec['step']}"
    first_3 = np.mean([r["loss"] for r in train_records[:3]])
    last_3 = np.mean([r["loss"] for r in train_records[-3:]])
    assert last_3 < first_3, f"Expected loss to decrease ({first_3:.4f} -> {last_3:.4f})"

    # Verify metrics.jsonl
    metrics_path = output_dir / "metrics.jsonl"
    assert metrics_path.exists()
    parsed = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    val_lines = [r for r in parsed if r["type"] == "val"]
    assert val_lines
    assert {"val_teacher_cos", "val_neg_flip_cos", "val_mlm_loss"}.issubset(val_lines[-1].keys())

    # Clean HuggingFace AutoModel & AutoTokenizer load with zero missing/unexpected keys
    loaded_model, loading_info = AutoModel.from_pretrained(
        output_dir, output_loading_info=True
    )
    loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert loaded_model.config.num_hidden_layers == 2
    assert loaded_model.config.hidden_size == 32
    assert loaded_model.config.max_position_embeddings == 512

    loaded_model.eval()
    with torch.no_grad():
        # Encode a 64-token sentence batch
        sent_batch = loaded_tokenizer(
            [
                "Denies fever, chills, or acute pain.",
                "Lymph node biopsy positive for EGFR mutation.",
            ],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        sent_out = loaded_model(**sent_batch)
        assert sent_out.last_hidden_state.shape == (2, 64, 32)
        assert torch.isfinite(sent_out.last_hidden_state).all()

        # Encode a full 512-token clinical note batch (verifying positional embeddings up to 512)
        long_note = " ".join(
            ["Patient reports stable disease without acute pleural effusion on staging CT scan."] * 60
        )
        note_batch = loaded_tokenizer(
            [long_note, long_note],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        assert note_batch["input_ids"].shape == (2, 512)
        note_out = loaded_model(**note_batch)
        assert note_out.last_hidden_state.shape == (2, 512, 32)
        assert torch.isfinite(note_out.last_hidden_state).all()
