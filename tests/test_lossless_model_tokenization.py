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


class _VerboseWarningCharacterTokenizer(_CharacterTokenizer):
    def __init__(self) -> None:
        self.verbose_values: list[bool] = []

    def encode(self, text, add_special_tokens=False, verbose=True):
        import warnings

        self.verbose_values.append(bool(verbose))
        if verbose and len(text) > 8:
            warnings.warn("sequence length exceeds model_max_length", UserWarning)
        return super().encode(text, add_special_tokens=add_special_tokens)


class _RoundTripExpandingTokenizer:
    """Mimic WordPiece fragments that expand after slice decoding."""

    marker = "[D] "

    def encode(self, text, add_special_tokens=False, verbose=True):
        del verbose
        value = str(text)
        marker_tokens = []
        if value.startswith(self.marker):
            marker_tokens = [900]
            value = value[len(self.marker) :]
        content_tokens = [ord(character) for character in value]
        tokens = marker_tokens + content_tokens
        if add_special_tokens:
            return [101, *tokens, 102]
        return tokens

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        values = [int(token_id) for token_id in token_ids]
        decoded = "".join(chr(token_id) for token_id in values)
        # A slice beginning inside the source resembles a WordPiece continuation:
        # its decoded ``##`` prefix consumes two extra tokens on re-encoding.
        return decoded if values[0] == ord("a") else f"##{decoded}"

    def num_special_tokens_to_add(self, pair=False):
        del pair
        return 2


def test_concept_word_chunking_rejects_prefix_or_suffix_selection():
    from oci.models.concept_embedding_utils import chunk_text_words

    text = " ".join(f"word{index}" for index in range(10))
    for selection in ("first", "last"):
        with pytest.raises(SemanticTruncationError, match="requires 4 chunks"):
            chunk_text_words(
                text,
                chunk_size_words=3,
                chunk_overlap_words=0,
                max_chunks=2,
                chunk_selection=selection,
            )


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


def test_token_chunk_boundaries_suppress_length_warning_without_truncating():
    import warnings

    from oci.models.concept_embedding_utils import split_text_to_token_chunks

    tokenizer = _VerboseWarningCharacterTokenizer()
    text = "abcdefghijklmnopqrstuvwxyz"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chunks = split_text_to_token_chunks(
            text,
            tokenizer,
            max_seq_length=7,
            chunk_overlap_tokens=0,
        )

    assert tokenizer.verbose_values
    assert all(value is False for value in tokenizer.verbose_values)
    assert "".join(chunks) == text
    assert max(map(len, chunks)) <= 7


def test_token_chunks_are_bounded_after_framed_decode_encode_round_trip():
    from oci.models.concept_embedding_utils import split_text_to_token_chunks

    tokenizer = _RoundTripExpandingTokenizer()
    chunks = split_text_to_token_chunks(
        "abcdefghij",
        tokenizer,
        max_seq_length=8,
        chunk_overlap_tokens=1,
        encoding_prefix=tokenizer.marker,
    )

    # The verified ends, rather than the too-large proposed ends, drive each
    # subsequent boundary.  Every original token consequently remains covered.
    assert chunks == ["abcde", "##efg", "##ghi", "##ij"]
    assert all(
        len(tokenizer.encode(f"{tokenizer.marker}{chunk}", add_special_tokens=True))
        <= 8
        for chunk in chunks
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
