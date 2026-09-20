"""Read-only concept-recall audit of saved raw evidence and prompt card text.

No clustering, LLM requests, model fitting, or production-artifact writes.
Raw scanning stops after finding a valid source witness for every concept/fold;
it does not estimate how many raw occurrences mention each concept.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "artifacts/research_all_evidence/five_conf_five_mod_nsclc_full"
RUN = BASE / "stage2"
STEM = "five_conf_five_mod_card_recall_2026-09-19"
sys.path.insert(0, str(ROOT))
from oci.inference.plain_handoff_stage2_evidence import (
    _extract_embedding_occurrences,
    _extract_htr_occurrences,
    _extract_sparse_occurrences,
    _extract_tfidf_occurrences,
    _extract_neural_query_occurrences,
)

PATTERNS = {
    "age": r"\bage\b|\baged\b|\byears?[- ]old\b",
    "sex": r"\b(?:sex|gender|male|female|man|woman|nonbinary)\b",
    "ecog_performance_status": r"\becog\b|eastern cooperative oncology group|performance status",
    "creatinine_clearance": r"creatinine.{0,15}clearance|\bcr\s*cl\b|\bclcr\b|cockcroft.{0,5}gault",
    "prior_platinum_therapy": r"\bplatinum\b|\bcisplatin\b|\bcarboplatin\b",
    "histology_type": r"\bhistolog\w*|\badenocarcinoma\b|\bsquamous\b|\blarge[- ]cell\b",
    "egfr_mutation_status": r"\begfr\b|epidermal growth factor receptor",
    "baseline_nlr": r"\bnlr\b|neutrophil.{0,24}lymphocyte.{0,12}ratio",
    "brain_metastases_status": r"\b(?:brain|cerebral|intracranial|cns)\b.{0,100}\b(?:metasta\w*|mets?)\b|\b(?:metasta\w*|mets?)\b.{0,100}\b(?:brain|cerebral|intracranial|cns)\b",
    "baseline_hemoglobin": r"\b(?:ha?emoglobin|hgb|hb)\b",
}
STRICT = dict(PATTERNS)
STRICT["egfr_mutation_status"] = (
    r"\begfr\b.{0,60}\b(?:mutation\w*|mutant|mutated|wild[- ]?type|exon|positive|negative|unknown|activat\w*|l858r|t790m|tki|status|alteration\w*|variant\w*)\b"
    r"|\b(?:activating|negative|positive|mutant|wild[- ]type)\b.{0,30}\begfr\b"
    r"|epidermal growth factor receptor"
)
HISTORY = r"\b(?:prior|previous\w*|history|received|completed|after|adjuvant|first[- ]line|second[- ]line|multiple|progress\w*|refractory|pretreat\w*|exposure|exposed|treated|induction)\b|\bs/p\b"
DRUG = PATTERNS["prior_platinum_therapy"]
STRICT["prior_platinum_therapy"] = rf"(?:{HISTORY}).{{0,160}}(?:{DRUG})|(?:{DRUG}).{{0,160}}(?:{HISTORY})"
BROAD_RE = {k: re.compile(v, re.I | re.S) for k, v in PATTERNS.items()}
STRICT_RE = {k: re.compile(v, re.I | re.S) for k, v in STRICT.items()}


def normalize(text):
    return re.sub(r"[\u2010-\u2015\u2212]", "-", unicodedata.normalize("NFKC", str(text)))


def snippet(text, match, margin=90):
    return text[max(0, match.start() - margin): min(len(text), match.end() + margin)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Verify freshly frozen card sets before the script consults oracle labels.
    card_manifest = json.loads((OUT / "five_conf_five_mod_card_sets_frozen_2026-09-19.json").read_text())
    for relative, identity in card_manifest.items():
        assert digest(ROOT / relative) == identity["sha256"], relative
    compilation = json.loads((RUN / "evidence_compilation/summary.json").read_text())
    completion = json.loads((RUN / "complete.json").read_text())
    metadata_path = ROOT / "synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/metadata.json"
    metadata_sha = digest(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    features = metadata["features"]
    assert {f["name"] for f in features} == set(PATTERNS)
    result = {
        "date": "2026-09-19", "run": str(RUN.relative_to(ROOT)),
        "completed_at": completion["completed_at"],
        "oracle_metadata": str(metadata_path.relative_to(ROOT)), "oracle_metadata_sha256": metadata_sha,
        "oracle_features": features, "broad_patterns": PATTERNS, "qualified_patterns": STRICT,
        "handoff_sha256_recorded_by_compiler": compilation["handoff_sha256"],
        "card_hashes_verified_before_oracle_labels": True,
        "method": "Concept-level lexical presence in readable representative text only; no metadata/details/lineage-only matches. EGFR requires molecular context; platinum requires treatment-history context. Labels, baseline timing, patient-level accuracy, association strength, and causal roles are not validated by presence. Raw witnesses establish presence, not raw frequency.",
        "folds": [],
    }
    for fold in range(1, 6):
        cards_path = RUN / "evidence_compilation" / f"outer_{fold:03d}" / "cards.jsonl"
        cards = [json.loads(line) for line in cards_path.open()]
        inputs_path = RUN / f"outer_{fold:03d}" / "input_packets.jsonl"
        packets = [json.loads(line) for line in inputs_path.open()]
        card_by_id = {c["card_id"]: c for c in cards}
        assert len(cards) == len(packets) == len(card_by_id) == 400
        assert {p["packet_id"] for p in packets} == set(card_by_id)
        assert all(p["content"] == card_by_id[p["packet_id"]] for p in packets)
        concepts = {}
        for feature in features:
            name = feature["name"]
            broad_ids, qualified_ids, witnesses = [], [], []
            architectures = set()
            for card in cards:
                texts = [normalize(r["text"]) for r in card["representative_evidence"]]
                if any(BROAD_RE[name].search(t) for t in texts):
                    broad_ids.append(card["card_id"])
                matched = [(t, STRICT_RE[name].search(t)) for t in texts]
                matched = [(t, m) for t, m in matched if m]
                if matched:
                    qualified_ids.append(card["card_id"])
                    architectures.update(card["source_architectures"])
                    # Favor a clinical passage over an isolated bag-of-words phrase.
                    text, match = min(matched, key=lambda tm: (len(tm[0]) < 150, len(tm[0])))
                    witnesses.append({
                        "card_id": card["card_id"], "architecture": card["source_architectures"],
                        "evidence_kind": card["evidence_kind"], "snippet": snippet(text, match),
                        "representative_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    })
            witnesses.sort(key=lambda w: (w["evidence_kind"] != "clinical_text", len(w["snippet"])))
            concepts[name] = {
                "roles": feature["roles"], "broad_card_count": len(broad_ids),
                "qualified_card_count": len(qualified_ids), "qualified_card_ids": qualified_ids,
                "architectures": sorted(architectures), "card_witnesses": witnesses[:3],
            }
        result["folds"].append({
            "outer_fold": fold, "cards": len(cards),
            "representatives": sum(len(c["representative_evidence"]) for c in cards),
            "raw_occurrences": compilation["outer_folds"][str(fold)]["raw_occurrences"],
            "exact_members": compilation["outer_folds"][str(fold)]["exact_members"],
            "input_packets_match_cards": True, "concepts": concepts,
        })
        print("CARD_COUNTS", fold, {k: v["qualified_card_count"] for k, v in concepts.items()}, flush=True)

    # Read the real handoff until all fifty concept/fold presence queries have
    # a witness from the compiler's allowlisted scientific text projection.
    # A tail read identifies envelope fields without decoding irrelevant rows.
    tail_pattern = re.compile(rb',\s*"inner_fold":\s*(null|\d+),\s*"outer_fold":\s*(\d+),\s*"scope":\s*"([^"]+)",\s*"source":\s*"([^"]+)"\s*}\s*$')
    raw_witnesses = {i: {} for i in range(1, 6)}
    examined_rows = []
    handoff_path = BASE / "handoff/evidence.jsonl"
    with handoff_path.open("rb") as handle:
        for number, line in enumerate(handle, 1):
            envelope = tail_pattern.search(line[-1024:])
            if envelope:
                fold = int(envelope[2])
                if len(raw_witnesses[fold]) == len(features):
                    continue
            row = json.loads(line)
            fold = int(row["outer_fold"])
            if len(raw_witnesses[fold]) == len(features):
                continue
            source = row["source"]
            extractors = {
                "text_models": [_extract_embedding_occurrences, _extract_sparse_occurrences, _extract_htr_occurrences],
                "tfidf": [_extract_tfidf_occurrences],
                "neural_queries": [_extract_neural_query_occurrences],
            }[source]
            examined_rows.append({"handoff_row": number, "outer_fold": fold, "inner_fold": row.get("inner_fold"), "source": source})
            for extract in extractors:
                if len(raw_witnesses[fold]) == len(features):
                    break
                occurrences = extract(row, row["evidence"], handoff_row=number)
                for occurrence in occurrences:
                    text = normalize(occurrence["text"])
                    for name in set(PATTERNS) - set(raw_witnesses[fold]):
                        match = STRICT_RE[name].search(text)
                        if match:
                            raw_witnesses[fold][name] = {
                                "reference": occurrence["reference"], "architecture": occurrence["architecture"],
                                "evidence_kind": occurrence["evidence_kind"], "snippet": snippet(text, match),
                                "normalized_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            }
                    if len(raw_witnesses[fold]) == len(features):
                        break
                del occurrences
            print("RAW_WITNESSES", fold, len(raw_witnesses[fold]), "handoff row", number, flush=True)
            del row
            if all(len(v) == len(features) for v in raw_witnesses.values()):
                break
    result["raw_presence_rows_examined"] = examined_rows
    missing = []
    for fold in result["folds"]:
        number = fold["outer_fold"]
        for name, concept in fold["concepts"].items():
            concept["raw_witness"] = raw_witnesses[number].get(name)
            concept["raw_presence_confirmed"] = name in raw_witnesses[number]
            if concept["raw_presence_confirmed"] and concept["qualified_card_count"] == 0:
                missing.append({"outer_fold": number, "feature": name, "roles": concept["roles"]})
    result["raw_to_card_losses"] = missing
    result["raw_positive_concept_fold_combinations"] = sum(len(v) for v in raw_witnesses.values())
    result["card_positive_concept_fold_combinations"] = sum(c["qualified_card_count"] > 0 for f in result["folds"] for c in f["concepts"].values())
    result["full_handoff_hash_recomputed_and_verified"] = False
    manifest_path = OUT / "five_conf_five_mod_card_recall_manifest_2026-09-19.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        identity = manifest["sources"][str(handoff_path.relative_to(ROOT))]
        assert identity["sha256"] == compilation["handoff_sha256"]
        assert handoff_path.stat().st_size == identity["bytes"]
        assert handoff_path.stat().st_mtime_ns == identity["mtime_ns"]
        result["full_handoff_hash_recomputed_and_verified"] = True
        result["source_manifest"] = str(manifest_path.relative_to(ROOT))
    target = OUT / (STEM + ".json")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("RESULT", result["raw_positive_concept_fold_combinations"], "raw-positive combinations;", len(missing), "lost", flush=True)
    print(target, flush=True)


if __name__ == "__main__":
    main()
