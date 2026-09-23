import json
from collections import Counter

def validator(payload, evidence_index):
    allowed = {c["feature_id"] for c in payload["candidates"]}

    def nonempty(value, label):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be nonempty text")
        return value.strip()

    def validate(value):
        if not isinstance(value, dict):
            raise ValueError("Return one JSON object")
        # Gather independent errors first. Large reviews should not regenerate
        # the whole concept map once per omitted field or missing citation.
        proposed = value.get("concepts")
        issues = []
        if isinstance(proposed, list) and all(isinstance(row, dict) for row in proposed):
            memberships = Counter()
            for position, row in enumerate(proposed, 1):
                cid = row.get("concept_id") or f"concept_at_position_{position}"
                for field in ("concept_id", "name", "relationship_summary", "modifier_rationale", "limitations"):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        issues.append(f"{cid}: {field} must be nonempty text")
                members = row.get("member_feature_ids")
                if isinstance(members, list) and all(isinstance(key, str) for key in members):
                    memberships.update(members)
                else:
                    issues.append(f"{cid}: member_feature_ids must be a nonempty list of supplied IDs")
                    members = []
                citations = row.get("evidence")
                cited_members = set()
                if not isinstance(citations, list) or not citations:
                    issues.append(f"{cid}: cite at least one supplied own-member effect-evidence record, including for uncertain or excluded concepts")
                    citations = []
                for cite in citations:
                    if not isinstance(cite, dict) or cite.get("evidence_id") not in evidence_index:
                        issues.append(f"{cid}: each evidence citation must use a supplied effect-evidence ID")
                        continue
                    key = evidence_index[cite["evidence_id"]]
                    if key not in members:
                        issues.append(f"{cid}: citation {cite['evidence_id']} is not a member's evidence")
                    cited_members.add(key)
                for rep in row.get("representatives") or []:
                    if isinstance(rep, dict) and rep.get("feature_id") not in cited_members:
                        issues.append(f"{cid}: representative {rep.get('feature_id')} needs its own effect-evidence citation")
            missing = allowed - set(memberships)
            unknown = set(memberships) - allowed
            duplicated = [key for key, count in memberships.items() if count > 1]
            if missing:
                issues.append(f"Assign all omitted candidates to a concept: {sorted(missing)}")
            if unknown:
                issues.append(f"Remove unknown candidate IDs: {sorted(unknown)}")
            if duplicated:
                issues.append(f"Assign each candidate to exactly one concept; duplicates: {sorted(duplicated)}")
        if issues:
            raise ValueError("Correct ALL of these validation issues together: " + json.dumps(issues))
        overall = nonempty(value.get("overall_interpretation"), "overall_interpretation")
        rows = value.get("concepts")
        if not isinstance(rows, list) or not rows:
            raise ValueError("concepts must be a nonempty list")
        seen, concept_ids, selected, clean = set(), set(), set(), []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each concept must be an object")
            cid = nonempty(row.get("concept_id"), "concept_id")
            if cid in concept_ids:
                raise ValueError(f"Duplicate concept_id {cid}")
            concept_ids.add(cid)
            members = row.get("member_feature_ids")
            if not isinstance(members, list) or not members or any(not isinstance(x, str) for x in members):
                raise ValueError(f"{cid}: member_feature_ids must be a nonempty string list")
            if len(set(members)) != len(members) or not set(members) <= allowed:
                raise ValueError(f"{cid}: unknown or repeated member IDs")
            if set(members) & seen:
                raise ValueError(f"{cid}: candidates assigned to multiple concepts: {sorted(set(members) & seen)}")
            seen.update(members)
            decision = row.get("decision")
            if decision not in {"retain", "uncertain", "exclude"}:
                raise ValueError(f"{cid}: invalid decision")
            reps = row.get("representatives")
            if not isinstance(reps, list) or bool(reps) != (decision == "retain"):
                raise ValueError(f"{cid}: representatives must be nonempty exactly when decision is retain")
            cleaned_reps = []
            for rep in reps:
                if not isinstance(rep, dict) or rep.get("feature_id") not in members:
                    raise ValueError(f"{cid}: representative must be an existing concept member")
                key = rep["feature_id"]
                if key in selected:
                    raise ValueError(f"Duplicate representative {key}")
                selected.add(key)
                cleaned_reps.append({"feature_id": key, "reason": nonempty(rep.get("reason"), "representative reason")})
            citations = row.get("evidence")
            if not isinstance(citations, list) or not citations:
                raise ValueError(f"{cid}: cite at least one supplied effect-evidence record")
            cited_members, cleaned_cites = set(), []
            for citation in citations:
                if not isinstance(citation, dict) or citation.get("evidence_id") not in evidence_index:
                    raise ValueError(f"{cid}: unknown evidence ID")
                key = evidence_index[citation["evidence_id"]]
                if key not in members:
                    raise ValueError(f"{cid}: cite only the concept's own member evidence")
                folds = citation.get("inner_folds")
                if (not isinstance(folds, list) or not folds or any(type(n) is not int for n in folds)
                        or len(set(folds)) != len(folds) or not set(folds) <= set(range(1, 6))):
                    raise ValueError(f"{cid}: inner_folds must be distinct integers from 1 to 5")
                cited_members.add(key)
                cleaned_cites.append({"evidence_id": citation["evidence_id"], "inner_folds": folds})
            if any(rep["feature_id"] not in cited_members for rep in cleaned_reps):
                raise ValueError(f"{cid}: every representative needs an own-feature effect-evidence citation")
            clean.append({"concept_id": cid,
                          **{field: nonempty(row.get(field), f"{cid}: {field}") for field in
                             ("name", "relationship_summary", "modifier_rationale", "limitations")},
                          "member_feature_ids": members, "decision": decision,
                          "representatives": cleaned_reps, "evidence": cleaned_cites})
        if seen != allowed:
            raise ValueError(f"Concepts must cover every supplied candidate. Missing: {sorted(allowed - seen)}")
        return {"overall_interpretation": overall, "concepts": clean}

    return validate
