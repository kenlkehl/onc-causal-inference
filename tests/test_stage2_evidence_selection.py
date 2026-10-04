from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from oci.inference.neural_query_evidence_contract import query_retrieval_policy, query_term_normalization
from oci.inference.plain_handoff_stage2 import _interpretation_prompt
from oci.inference.plain_handoff_stage2_evidence import (
    compile_stage2_handoff_evidence,
    extract_stage1_architecture_occurrences,
)
from oci.inference.stage2_evidence_selection import (
    SupportingSentenceIndex,
    annotate_strength,
    redundant,
    select_exemplars,
)


def member(text, score=None, *, query=0, inner=1, metric="tfidf_contrast", kind="lexical_term"):
    return {
        "member_id": text, "text": text, "evidence_kind": kind,
        "evidence_axes": ["outcome"], "polarities": ["positive"],
        "source_architectures": ["neural_query_moments"],
        "raw_references": [{
            "source": "neural_queries", "scope": "inner_train", "inner_fold": inner,
            "json_path": f"evidence[{query}].{'top_chunks' if kind == 'clinical_text' else 'top_contrastive_ngrams'}[0]",
            "query_id": f"q{query}", "bank": "outcome",
            "scores": {metric: score} if score is not None else {},
        }],
    }


def test_strength_beats_geometric_diversity_and_does_not_compare_score_scales():
    rows = [
        member("creatinine", -0.8, metric="coefficient"),
        member("random fragment", 0.001, metric="coefficient"),
        member("ECOG status", 900, query=1), member("other fragment", 10, query=1),
    ]
    annotate_strength(rows)
    assert rows[0]["selection_strength"]["mean_context_percentile"] == rows[2]["selection_strength"]["mean_context_percentile"]
    # Both center proximity and the old farthest-first rule favor weak fragments.
    selected = select_exemplars(rows, np.array([[2.], [0.], [3.], [100.]]), np.array([0.]), limit=2)
    assert {m["text"] for m in selected} == {"creatinine", "ECOG status"}


def test_query_fit_is_not_phrase_strength_and_repeated_queries_do_not_inflate_support():
    strong, weak = member("strong", 0.9), member("weak", 0.1)
    for index in range(20):
        ref = copy.deepcopy(weak["raw_references"][0])
        ref["scores"]["fit_standardized_score"] = 1e9
        weak["raw_references"].append(ref)
    unscored = member("unscored")
    unscored["raw_references"][0]["scores"] = {"fit_standardized_score": 1e12, "r_loss": 100}
    annotate_strength([strong, weak, unscored])
    assert weak["selection_strength"]["scored_context_count"] == 1
    assert unscored["selection_strength"]["mean_context_percentile"] is None
    assert select_exemplars([weak, unscored, strong], None, None, limit=1) == [strong]
    # Source-native ranks are a valid fallback, and lower rank is stronger.
    rows = [member("first", 1, metric="rank"), member("last", 1500, metric="rank")]
    annotate_strength(rows)
    assert select_exemplars(rows, None, None, limit=1)[0]["text"] == "first"


@pytest.mark.parametrize("left,right,expected", [
    ("spo₂ 94", "spo₂ 94 room", True),
    ("acetaminophen 500", "acetaminophen 500 mg", True),
    ("creatinine", "creatinine clearance", False),
    ("creatinine clearance", "estimated GFR", False),
    ("ECOG 1", "ECOG 2", False),
    ("fever", "no fever", False),
    ("change -1", "change 1", False),
])
def test_overlap_preserves_distinct_measurements_values_and_negation(left, right, expected):
    assert redundant(member(left), member(right)) is expected


def test_duplicate_slots_are_replaced_by_next_strong_distinct_item():
    rows = [member("spo₂ 94", 1), member("spo₂ 94 room", 0.9), member("dyspnea", 0.8)]
    annotate_strength(rows)
    assert [m["text"] for m in select_exemplars(rows, None, None, limit=3)] == ["spo₂ 94", "dyspnea"]


def test_repeated_note_template_does_not_hide_a_different_lab():
    template = " ".join(f"word{i}" for i in range(150))
    assert not redundant(
        member(template + " creatinine was reported."),
        member(template + " creatinine clearance was reported."),
    )


def test_context_is_local_exact_and_match_centered():
    phrase = member("22 mg", 0.4)
    wrong_query = member("WRONG QUERY BUN 22 mg/dL.", 1, query=1, kind="clinical_text")
    wrong_inner = member("WRONG INNER BUN 22 mg/dL.", 1, inner=2, kind="clinical_text")
    source = member("Other sentence. Laboratory BUN: 22 mg/dL. Next sentence.", 0.8, kind="clinical_text")
    index = SupportingSentenceIndex([wrong_query, wrong_inner, source])
    found = index.find(phrase)
    assert [r["text"] for r in found] == ["Laboratory BUN: 22 mg/dL."]
    assert source["text"][found[0]["source_char_start"]:found[0]["source_char_end"]] == found[0]["text"]
    assert index.find(member("2 mg", 0.4)) == []  # no substring matching inside 22
    assert index.find(member("absent lab", 0.4)) == []
    long = member("filler " * 300 + "BUN 22 mg/dL " + "tail " * 300, 1, kind="clinical_text")
    clipped = SupportingSentenceIndex([long]).find(phrase)[0]
    assert clipped["clipped"] and "BUN 22 mg/dL" in clipped["text"]
    assert len(clipped["text"]) < 900


@pytest.mark.parametrize("phrase,source,match", [
    ("brain metastases remained", "His brain metastases have remained stable.", "brain metastases have remained"),
    ("metastases remained", "His brain metastases have remained stable.", "metastases have remained"),
    ("stay febrile", "Follow-up after a hospital stay for febrile neutropenia.", "stay for febrile"),
    ("fatigued uses", "Appears fatigued, uses a walker.", "fatigued, uses"),
    ("metastases remained", "His metastases have not remained stable.", "metastases have not remained"),
])
def test_source_ngrams_recover_original_sentences_and_negation(phrase, source, match):
    # Generate the term with the real producer before exercising the matcher.
    from oci.inference.neural_query_agentic_forest import NeuralQueryAgenticForestConfig, _contrastive_ngrams
    config = NeuralQueryAgenticForestConfig()
    assert phrase in {item["term"] for item in _contrastive_ngrams([source], [], limit=None, config=config)}
    phrase_member = member(phrase, 0.9)
    wrong_query = member(source, 1, query=2, kind="clinical_text")
    wrong_fold = member(source, 1, inner=2, kind="clinical_text")
    chunk = member(source, 0.8, kind="clinical_text")
    result, = SupportingSentenceIndex([wrong_query, wrong_fold, chunk]).find(phrase_member)
    assert result["text"] == source
    assert source[result["source_char_start"]:result["source_char_end"]] == source
    assert source[result["matched_source_char_start"]:result["matched_source_char_end"]] == match
    assert result["reference"]["query_id"] == "q0"
    assert result["reference"]["inner_fold"] == 1


def test_normalization_respects_custom_settings_and_maps_unicode_offsets():
    phrase = member("metastases remained", 0.9)
    phrase["raw_references"][0]["term_normalization"] = {**query_term_normalization(), "stop_words": None}
    text = member("Metastases have remained stable.", 1, kind="clinical_text")
    assert SupportingSentenceIndex([text]).find(phrase) == []
    phrase["raw_references"][0]["term_normalization"]["stop_words"] = ["have"]
    result, = SupportingSentenceIndex([text]).find(phrase)
    assert not result["normalization_defaults_assumed"]
    assert result["term_normalization"]["stop_words"] == ["have"]
    # English-stop-word matching never allows arbitrary intervening content.
    assert SupportingSentenceIndex([member("Metastases rapidly remained stable.", 1, kind="clinical_text")]).find(phrase) == []
    phrase = member("cafe intact", 1)
    phrase["raw_references"][0]["term_normalization"] = {**query_term_normalization(), "strip_accents": "unicode"}
    source = "İnitial examination: café is intact."
    result, = SupportingSentenceIndex([member(source, 1, kind="clinical_text")]).find(phrase)
    assert result["text"] == source
    assert source[result["matched_source_char_start"]:result["matched_source_char_end"]] == "café is intact"


def test_custom_capturing_token_pattern_preserves_source_offsets():
    phrase = member("Cr 2", 1)
    phrase["raw_references"][0]["term_normalization"] = {
        **query_term_normalization(), "token_pattern": r"(?u)\b(\w+)\b",
        "lowercase": False, "stop_words": ["was"],
    }
    source = "Cr was 2."
    result, = SupportingSentenceIndex([member(source, 1, kind="clinical_text")]).find(phrase)
    assert result["text"] == source
    assert source[result["matched_source_char_start"]:result["matched_source_char_end"]] == "Cr was 2"


def _query_row(outer, *, inner=1):
    return {
        "source": "neural_queries", "outer_fold": outer, "inner_fold": inner,
        "scope": "inner_train", "evidence": {"evidence": [{
            "query_id": "q0", "bank": "outcome", "retrieval_policy": query_retrieval_policy(1),
            "fit_standardized_score": 123456.0,
            "top_chunks": [{
                "_oci_row_id": outer, "chunk_index": 0, "similarity": 0.8,
                "text": "BUN was 22 mg/dL." if outer == 1 else "DIFFERENT FOLD BUN was 22 mg/dL.",
            }],
            "top_contrastive_ngrams": [{"term": "22 mg", "tfidf_contrast": 0.9}],
        }]},
    }


def test_compilation_preserves_strength_context_and_prompt_isolation_through_canonical_handoff(tmp_path):
    rows = [_query_row(1), _query_row(2), _query_row(1, inner=2)]
    def compile(rows):
        return compile_stage2_handoff_evidence(rows, handoff_path=tmp_path / "handoff.jsonl")
    raw = compile(rows)
    canonical = []
    for outer, occurrences in extract_stage1_architecture_occurrences(rows).items():
        for occurrence in occurrences:
            canonical.append({
                "outer_fold": outer, "source": "stage1_architecture",
                "evidence": {"architecture": occurrence["architecture"], "occurrence": occurrence},
            })
    compact = compile(canonical)
    assert raw.cards_by_outer_fold == compact.cards_by_outer_fold
    phrase_member = next(m for m in raw.members_by_outer_fold[1] if m["evidence_kind"] == "lexical_term")
    assert phrase_member["selection_strength"]["metrics"] == ["tfidf_contrast"]
    assert phrase_member["selection_strength"]["scored_context_count"] == 2
    assert all(ref["scores"]["rank"] == 1 for ref in phrase_member["raw_references"])
    packet = next(p for p in raw.packets if p["outer_fold"] == 1 and p["content"]["evidence_kind"] == "lexical_term")
    messages = _interpretation_prompt(architecture="ignored", packets=[packet])
    prompt = json.dumps(messages)
    assert "BUN was 22 mg/dL." in prompt
    for hidden in ("DIFFERENT FOLD", "tfidf_contrast", "123456", "percentile", "scores", "source_char", "member_id"):
        assert hidden not in prompt
    lineage = next(l for l in raw.lineage_by_outer_fold[1] if l["card_id"] == packet["packet_id"])
    assert lineage["selected_exemplars"][0]["supporting_sentences"][0]["reference"]["row_id"] == 1
