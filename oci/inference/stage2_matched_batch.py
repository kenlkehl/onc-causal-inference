"""Matched-batch modifier evidence on existing measurements.

Adapted from stage2_batch_core.py in the htr-batch-delta-20260927 worktree.
The caller supplies honest nuisance predictions and overlap-eligible patients.
This component does not fit nuisances, choose representations, or select roles.
"""
from collections import Counter

import numpy as np

from .stage2_elastic_net_selection import _encode_design

VERSION = "matched_batch_evidence_v1"


class NoBatches(ValueError):
    """An expected lack of statistical support, not a pipeline failure."""


class MatchedBatches:
    """Shuffle each treatment arm within bins fitted on reference patients."""

    def __init__(self, treatment, outcome, nuisance, fit_rows, rows, seed, cfg):
        self.t, self.y, self.scores = map(np.asarray, (treatment, outcome, nuisance))
        self.fit, self.rows = np.asarray(fit_rows, dtype=int), np.asarray(rows, dtype=int)
        if len(set(self.fit)) != len(self.fit) or len(set(self.rows)) != len(self.rows):
            raise ValueError("Duplicate patient rows")
        self.same = set(self.fit) == set(self.rows)
        if not self.same and set(self.fit) & set(self.rows):
            raise ValueError("Reference and evaluation patients must be disjoint")
        if self.scores.shape != (len(self.t), 2) or self.y.shape != self.t.shape:
            raise ValueError("Matched batches require aligned treatment, outcome and two nuisance predictions")
        used = np.r_[self.fit, self.rows]
        if (not np.isin(self.t[used], [0, 1]).all() or
                not np.isfinite(self.y[used]).all() or not np.isfinite(self.scores[used]).all()):
            raise ValueError("Matched batches require finite labels/predictions and binary treatment")
        if not len(self.fit):
            raise NoBatches("no_reference_patients")
        self.cfg, self.rng = cfg, np.random.default_rng(seed)
        self.scale = np.maximum(self.scores[self.fit].std(0), 1e-8)
        self.cuts = [np.unique(np.quantile(self.scores[self.fit, j],
            np.linspace(0, 1, cfg.bins_per_nuisance + 1)[1:-1])) for j in range(2)]
        codes = [np.searchsorted(self.cuts[j], self.scores[:, j], side="right") for j in range(2)]
        self.groups = codes[0] * (len(self.cuts[1]) + 1) + codes[1]
        self.references = {}
        for group in np.unique(self.groups[self.fit]):
            arms = [self.fit[(self.groups[self.fit] == group) & (self.t[self.fit] == arm)]
                    for arm in (0, 1)]
            if min(map(len, arms)) < cfg.min_reference_arm:
                continue
            self.references[int(group)] = {
                "contrast": float(self.y[arms[1]].mean() - self.y[arms[0]].mean()),
                "counts": list(map(len, arms)),
                "variances": [float(self.y[r].var(ddof=1)) for r in arms],
            }
        if not self.references:
            raise NoBatches("no_supported_nuisance_bins")
        self.group_ids = sorted(self.references)
        self.group_index = {group: i for i, group in enumerate(self.group_ids)}
        self.stats = Counter()

    def one_pass(self):
        half = self.cfg.batch_size // 2
        for group in self.rng.permutation(self.group_ids):
            rows = self.rows[self.groups[self.rows] == group]
            control, treated = [self.rng.permutation(rows[self.t[rows] == arm]) for arm in (0, 1)]
            for start in range(0, min(len(control), len(treated)) - half + 1, half):
                a, b = control[start:start + half], treated[start:start + half]
                self.stats["proposed"] += 1
                smd = np.abs(self.scores[a].mean(0) - self.scores[b].mean(0)) / self.scale
                if max(smd) > self.cfg.max_smd:
                    self.stats["imbalance_rejected"] += 1
                    continue
                ref = self.references[group]
                deviation = float(self.y[b].mean() - self.y[a].mean() - ref["contrast"])
                # Training batches are subsets of their references; validation
                # batches are disjoint. This SE is only a training-tail heuristic.
                sign = -1 if self.same else 1
                variance = sum(v * max(0, 1 / half + sign / n)
                               for v, n in zip(ref["variances"], ref["counts"]))
                threshold = max(self.cfg.min_delta, self.cfg.z_threshold * np.sqrt(variance))
                tail = abs(deviation) >= threshold
                self.stats["matched"] += 1
                self.stats["tail"] += int(tail)
                yield np.r_[a, b], self.group_index[group], deviation, tail


class BlockRidge:
    """Candidate-wise ridge with bin intercepts and a mean-loss penalty."""

    def __init__(self, blocks, p, bins, penalty):
        self.blocks, self.penalty = blocks, penalty
        self.n = np.zeros(bins)
        self.sx, self.sy = np.zeros((bins, p)), np.zeros(bins)
        self.xx = [np.zeros((len(b), len(b))) for b in blocks]
        self.xy = np.zeros(p)

    def add(self, x, y, groups):
        for group in np.unique(groups):
            keep = groups == group
            self.n[group] += keep.sum()
            self.sx[group] += x[keep].sum(0)
            self.sy[group] += y[keep].sum()
        self.xy += x.T @ y
        for k, block in enumerate(self.blocks):
            self.xx[k] += x[:, block].T @ x[:, block]

    def finalize(self):
        if self.n.sum() == 0:
            raise NoBatches("no_eligible_training_batches")
        safe = np.maximum(self.n, 1)
        self.xmean, self.ymean = self.sx / safe[:, None], self.sy / safe
        self.beta, self.informative = [], []
        for block, xx in zip(self.blocks, self.xx):
            gram = xx - (self.sx[:, block].T / safe) @ self.sx[:, block]
            cov = self.xy[block] - (self.sx[:, block].T / safe) @ self.sy
            self.beta.append(np.linalg.solve(gram / self.n.sum() + self.penalty * np.eye(len(block)),
                                             cov / self.n.sum()))
            self.informative.append(bool(len(block) and np.trace(gram) / self.n.sum() > 1e-12))
        return self

    def predict(self, x, groups):
        return self.ymean[groups, None] + np.column_stack([
            (x[:, block] - self.xmean[groups][:, block]) @ beta
            for block, beta in zip(self.blocks, self.beta)])


def score_candidates(train, valid, definitions, t, y, tv, yv, *, nuisance, policy, seed):
    """Return one modifier-evidence record per candidate, with no hard gate."""
    cfg = policy.multi_model.matched_batch
    cfg.validate()
    primary = "filtered" if cfg.filter_training_batches else "unfiltered"
    audit = {"schema_version": VERSION, "primary": primary,
             "reference": "observed_training_bin_contrast",
             "validation_scope": "matched_validation_batches_without_outcome_filter",
             "nuisances": "shared_inner_fold_cross_fitted_predictions",
             "representation_search": False, "shuffles_are_independent_patients": False}

    def unavailable(reason):
        return {"records": [{"feature_id": str(d.get("feature_id") or d["name"]),
                "role": "effect", "status": "not_evaluable", "selected": None, "score": None}
                for d in definitions], "audit": {**audit, "status": reason}}

    encoded = _encode_design(train, valid, definitions, categorical_min_count=policy.categorical_min_count)
    # Keep the frozen clinical encodings. Missingness by itself is not a clinical
    # modifier measurement; no cutpoints, thresholds or new categories are tried.
    keep = np.array([not name.endswith(":missing") for name in encoded.column_names], dtype=bool)
    x = np.vstack([encoded.train[:, keep], encoded.valid[:, keep]])
    ids = np.asarray(encoded.column_feature_ids)[keep]
    blocks = [np.flatnonzero(ids == str(d.get("feature_id") or d["name"])) for d in definitions]
    if not x.shape[1]:
        return unavailable("no_variable_measurements")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite encoded measurements in matched-batch evidence")
    q = np.column_stack([np.r_[nuisance["e"], nuisance["ev"]],
                         np.r_[nuisance["m"], nuisance["mv"]]])
    treatment, outcome = np.r_[t, tv], np.r_[y, yv]
    fit, hold = np.arange(len(train)), np.arange(len(train), len(x))
    try:
        sampler = MatchedBatches(treatment, outcome, q, fit, fit, seed, cfg)
    except NoBatches as exc:
        return unavailable(str(exc))
    models = {name: BlockRidge(blocks, x.shape[1], len(sampler.group_ids), cfg.ridge_penalty)
              for name in ("filtered", "unfiltered")}
    seen_training = {name: set() for name in models}
    for _ in range(cfg.train_passes):
        bags = list(sampler.one_pass())
        if not bags:
            continue
        rows = np.array([bag[0] for bag in bags])
        xb = x[rows].mean(1)
        groups = np.array([bag[1] for bag in bags])
        target = np.array([bag[2] for bag in bags])
        tails = np.array([bag[3] for bag in bags])
        for name, eligible in (("unfiltered", np.ones(len(bags), bool)), ("filtered", tails)):
            models[name].add(xb[eligible], target[eligible], groups[eligible])
            seen_training[name].update(rows[eligible].ravel().tolist())
    available = {name: model.finalize() for name, model in models.items() if model.n.sum()}
    evaluation = MatchedBatches(treatment, outcome, q, fit, hold, seed + 991, cfg)
    totals = {name: np.zeros(len(definitions)) for name in models}
    baseline, counts = Counter(), Counter()
    seen_validation = {name: set() for name in models}
    # If both models are available, compare them on the same bins and patients.
    common_bins = (np.logical_and.reduce([model.n > 0 for model in available.values()])
                   if available else np.zeros(len(sampler.group_ids), dtype=bool))
    for _ in range(cfg.validation_passes):
        bags = [bag for bag in evaluation.one_pass() if common_bins[bag[1]]]
        if not bags:
            continue
        rows = np.array([bag[0] for bag in bags])
        xb = x[rows].mean(1)
        groups = np.array([bag[1] for bag in bags])
        target = np.array([bag[2] for bag in bags])
        for name, model in available.items():
            totals[name] += ((model.predict(xb, groups) - target[:, None]) ** 2).sum(0)
            baseline[name] += float(((model.ymean[groups] - target) ** 2).sum())
            counts[name] += len(bags)
            seen_validation[name].update(rows.ravel().tolist())
    audit.update(
        status="complete" if counts[primary] else "no_usable_primary_batches",
        bin_cuts=[cuts.tolist() for cuts in sampler.cuts], nuisance_scales=sampler.scale.tolist(),
        bin_references={str(k): v for k, v in sampler.references.items()},
        training_sampling=dict(sampler.stats), validation_sampling=dict(evaluation.stats),
        primary_training_patients=len(seen_training[primary]),
        primary_validation_patients=len(seen_validation[primary]),
        primary_training_batches=int(models[primary].n.sum()), primary_validation_batches=counts[primary],
        scoring_bins=[sampler.group_ids[i] for i in np.flatnonzero(common_bins)],
        variants={name: {"training_batches": int(model.n.sum()), "validation_batches": counts[name],
            "training_patients": len(seen_training[name]), "validation_patients": len(seen_validation[name]),
            "bin_only_loss": baseline[name] / counts[name] if counts[name] else None}
            for name, model in models.items()})
    records = []
    for j, definition in enumerate(definitions):
        scores = {}
        for name, model in models.items():
            usable = (counts[name] > 0 and baseline[name] / counts[name] > 1e-12
                      and name in available and model.informative[j])
            scores[name] = float((baseline[name] - totals[name][j]) / baseline[name]) if usable else None
        score = scores[primary]
        records.append({"feature_id": str(definition.get("feature_id") or definition["name"]),
            "role": "effect", "status": "ok" if score is not None else "not_evaluable",
            "selected": score > 0 if score is not None else None, "score": score,
            "filtered_score": scores["filtered"], "unfiltered_score": scores["unfiltered"],
            "filter_agreement": ((scores["filtered"] > 0) == (scores["unfiltered"] > 0)
                                 if all(v is not None for v in scores.values()) else None)})
    return {"records": records, "audit": audit}
