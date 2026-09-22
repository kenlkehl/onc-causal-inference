"""Warm the native adjudicator's exact checkpoints with four independent requests.

Payloads, validators, ordering, hierarchy, and fingerprints reproduce the native
adjudicator. The native adjudicator subsequently replays every checkpoint with
network requests disabled: a mismatch fails instead of silently changing review.
"""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

from oci.inference import stage2_multi_model_adjudication as m


def prefetch(*, definitions, statistical_report, request_json, output_dir, policy, workers=4):
    evidence = m.build_multi_model_role_evidence(
        definitions=definitions, statistical_report=statistical_report, policy=policy
    )
    maximum_chars = int(statistical_report["policy"]["multi_model"]["max_prompt_chars"])
    maximum_items = max(2, int(policy.max_candidates_per_request))
    directory = Path(output_dir)
    identity = {
        "evidence": m._fingerprint(evidence), "policy": policy.public_dict(),
        "prompt": m.PROMPT_VERSION, "model": statistical_report.get("adjudication_model_identity"),
        "source_sha256": sha256(Path(m.__file__).read_bytes()).hexdigest(),
    }
    cards = evidence["candidates"]
    m._write_json(directory / "evidence.json", evidence)

    def theme_payload(batch):
        return {
            "task": "review_stage2_multi_model_themes", "prompt_version": m.PROMPT_VERSION,
            "score_meaning": evidence["score_meaning"], "candidates": batch,
            "required_response": {"themes": [{
                "name": "theme", "member_feature_ids": ["candidate IDs"],
                "evidence_ids": ["at most 12 representative supplied modeling evidence IDs"],
                "interpretation": "common or complementary evidence",
                "disagreements": "contradictions, weak signals, and proxy distinctions",
            }]},
            "coverage": "Cover every supplied candidate, including weak/unevaluable candidates.",
        }

    def execute(job):
        name, payload, validate = job
        return m._request(directory, name, payload, validate, request_json=request_json,
                          maximum_chars=maximum_chars, identity=identity)

    # executor.map preserves original batch order regardless of response timing.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        batches = m._bounded_batches(cards, theme_payload, maximum_items=maximum_items,
                                     maximum_chars=maximum_chars)
        jobs = [
            (f"themes/initial_{i:03d}", theme_payload(batch), m._themes_validator(
                [c["feature_id"] for c in batch],
                {r["evidence_id"] for c in batch for r in c["modeling_evidence"]}, maximum=len(batch)))
            for i, batch in enumerate(batches)
        ]
        themes = [theme for response in executor.map(execute, jobs) for theme in response["themes"]]
        level = 0
        while len(themes) > maximum_items:
            def merge_payload(batch):
                return {
                    "task": "merge_stage2_multi_model_themes", "themes": batch,
                    "maximum_output_themes": max(1, len(batch) // 2),
                    "instructions": "Preserve all candidate IDs and distinctions. Broader parent themes organize evidence; they do not imply measurement equivalence.",
                    "required_response": theme_payload([])["required_response"],
                }

            batches = m._bounded_batches(themes, merge_payload, maximum_items=maximum_items,
                                         maximum_chars=maximum_chars)
            if all(len(batch) == 1 for batch in batches):
                raise ValueError("theme merge cannot fit two summaries; increase max_prompt_chars")
            jobs, positions = [], []
            for i, batch in enumerate(batches):
                if len(batch) > 1:
                    positions.append(i)
                    jobs.append((f"themes/merge_{level:03d}_{i:03d}", merge_payload(batch), m._themes_validator(
                        {k for t in batch for k in t["member_feature_ids"]},
                        {k for t in batch for k in t["evidence_ids"]}, maximum=max(1, len(batch) // 2))))
            responses = dict(zip(positions, executor.map(execute, jobs)))
            themes = [theme for i, batch in enumerate(batches)
                      for theme in (responses[i]["themes"] if i in responses else batch)]
            level += 1
        m._write_json(directory / "themes.json", {"themes": themes, "all_candidate_ids_preserved": True})
        by_id = {m._feature_id(d): d for d in definitions}

        def role_payload(batch):
            ids = {c["feature_id"] for c in batch}
            relevant = [t for t in themes if ids & set(t["member_feature_ids"])]
            return {
                "task": "adjudicate_stage2_multi_model_roles", "prompt_version": m.PROMPT_VERSION,
                "score_meaning": evidence["score_meaning"], "candidates": batch, "themes": relevant,
                "required_response": {
                    "summary": "overall interpretation", "decisions": [{
                        "feature_id": "each supplied candidate exactly once",
                        "roles": ["confounder and/or effect_modifier, or empty"],
                        "evidence_ids": ["this candidate's evidence IDs; modifiers cite effect evidence; confounders cite treatment and outcome evidence"],
                        "evidence_for": ["specific facts"], "evidence_against": ["specific facts"],
                        "inner_fold_consistency": "compare folds and subset exposures",
                        "cross_method_reconciliation": "reconcile disagreements",
                        "rationale": "justify roles from evidence", "stability": "consistent, mixed, or insufficient",
                    }],
                },
            }

        batches = m._bounded_batches(cards, role_payload,
                                     maximum_items=int(policy.max_candidates_per_request), maximum_chars=maximum_chars)
        jobs = [(f"roles/batch_{i:03d}", role_payload(batch),
                 m._decision_validator([by_id[c["feature_id"]] for c in batch], batch))
                for i, batch in enumerate(batches)]
        list(executor.map(execute, jobs))
    return {"parallel_workers": workers, "native_checkpoint_replay_required": True,
            "role_batches": len(batches), "themes": len(themes), "merge_levels": level}
