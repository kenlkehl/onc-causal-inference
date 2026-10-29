from __future__ import annotations

import json


from oci.inference.neural_query_evidence_contract import query_retrieval_policy
from oci.inference.plain_handoff_stage2 import _interpretation_prompt
from oci.inference.plain_handoff_stage2_evidence import (
    compile_stage2_handoff_evidence,
    extract_stage1_architecture_occurrences,
)


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
