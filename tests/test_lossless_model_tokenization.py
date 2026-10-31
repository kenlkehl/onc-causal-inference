from __future__ import annotations

import ast
from pathlib import Path

import pytest

from oci.models.lossless_tokenization import SemanticTruncationError


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def num_special_tokens_to_add(self, pair=False):
        del pair
        return 0


def test_concept_embedding_cache_rejects_token_chunk_overflow(tmp_path):
    from oci.models.concept_embedding_cache import ConceptEmbeddingCache

    cache = ConceptEmbeddingCache(
        cache_dir=str(tmp_path),
        sentence_model_name="unused-model",
        dataset_path=str(tmp_path / "dataset.parquet"),
        chunk_size_words=100,
        chunk_overlap_words=0,
        max_chunks=1,
    )
    with pytest.raises(SemanticTruncationError, match="token-bounded chunk plan"):
        cache._chunk_texts(
            ["abcdefghi"],
            tokenizer=_CharacterTokenizer(),
            max_seq_length=5,
        )


def test_production_sources_have_no_literal_truncation_true():
    repository = Path(__file__).resolve().parents[1]
    source_paths = tuple((repository / "oci").rglob("*.py")) + tuple(
        (repository / "scripts").rglob("*.py")
    )
    offenders: list[str] = []
    for path in sorted(source_paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "truncation"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    offenders.append(f"{path.relative_to(repository)}:{node.lineno}")
    assert offenders == []
