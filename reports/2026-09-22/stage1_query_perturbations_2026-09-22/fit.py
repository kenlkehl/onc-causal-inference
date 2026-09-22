"""Fit edits/probes and freeze predictions without opening validation outcomes."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/query-perturbation-mpl")

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import traceback
import numpy as np
import pandas as pd
import torch

from common import (RUN, SOURCE, POLICY, BANKS, assert_scope, labels, embeddings,
                    chunk_texts, nuisance_path, read, sha, split, write, status, now, verify_manifest)
from perturb import activation, optimize, random_at_distance
from probes import Probe
from oci.inference.neural_cohort_witness import pad_chunk_embeddings


def project(chunks, mask, vectors, size=20):
    with torch.no_grad():
        return np.column_stack([activation(chunks, mask, torch.as_tensor(vectors[k:k + size]),
                                           POLICY["temperature"]).numpy()
                                for k in range(0, len(vectors), size)])


def evidence_packets(position, row_ids, matrices, all_text, original_vectors, all_vectors,
                     variants, train_activations, contribution, weights, query_ids):
    packets = []
    for j in range(5):
        original_index = j
        targeted_index = next(i for i, v in enumerate(variants) if v["query"] == j and v["kind"] == "targeted" and v["radius"] == POLICY["llm_radius"])
        random_index = next(i for i, v in enumerate(variants) if v["query"] == j and v["kind"] == "random" and v["radius"] == POLICY["llm_radius"] and v["control"] == POLICY["llm_random_control"])
        a0 = train_activations[:, original_index]
        centered0 = a0 - np.average(a0, weights=weights)
        for condition, index in [("original", original_index), ("targeted", targeted_index), ("random", random_index)]:
            a1 = train_activations[:, index]
            if condition == "original":
                order = np.argsort(-a0, kind="stable")
            else:
                centered1 = a1 - np.average(a1, weights=weights)
                importance = np.abs((centered0 - centered1) * contribution)
                order = np.argsort(-importance, kind="stable")
            records = []
            for loc in order[:POLICY["llm_evidence_patients"]]:
                row = int(row_ids[loc])
                z = matrices[loc] / np.maximum(np.linalg.norm(matrices[loc], axis=1, keepdims=True), 1e-12)
                s0 = z @ original_vectors[j]
                s1 = z @ all_vectors[index]
                if condition == "original":
                    choices = [(int(c), "original_high_similarity") for c in np.argsort(-s0, kind="stable")[:4]]
                else:
                    choices = [(int(s0.argmax()), "original_peak"), (int(s1.argmax()), "edited_peak"),
                               (int((s1-s0).argmin()), "largest_similarity_decrease"),
                               (int((s1-s0).argmax()), "largest_similarity_increase")]
                merged = {}
                for c, reason in choices:
                    merged.setdefault(c, []).append(reason)
                for c, reasons in merged.items():
                    records.append({"evidence_id": f"inner{position}_row{row}_chunk{c}", "row_id": row,
                                    "chunk_index": c, "selection_reasons": reasons,
                                    "original_similarity": float(s0[c]), "edited_similarity": float(s1[c]),
                                    "text": all_text[row][c]})
            packets.append({"packet_id": f"inner_{position:03d}_query_{j:03d}_{condition}",
                            "inner_fold": position, "query": j, "query_id": query_ids[j], "condition": condition,
                            "distance_budget": POLICY["llm_radius"] if condition != "original" else 0,
                            "actual_distance": variants[index]["distance"], "evidence": records,
                            "evidence_scope": "inner_training_only", "patient_outcomes_in_prompt": False,
                            "interpretation": "Training-guided semantic retrieval evidence. No validated causal role is established."})
    return packets


def fit_one(position):
    torch.set_num_threads(POLICY["torch_threads_per_fold"])
    torch.set_num_interop_threads(1)
    verify_manifest()
    outer = split()
    part = outer["inner_splits"][position - 1]
    fit_ids, valid_ids = part["fit_row_ids"], part["heldout_row_ids"]
    assert_scope(fit_ids, valid_ids, outer)
    folder = RUN / "folds" / f"inner_{position:03d}"
    frozen = folder / "frozen.json"
    if frozen.exists():
        record = read(frozen)
        assert record["manifest_sha256"] == sha(RUN / "manifest.json")
        for name, expected in record["files"].items():
            assert sha(folder / name) == expected
        return record
    status(folder / "status.json", "load_training", inner_fold=position)
    observed = labels(fit_ids)
    t, y = observed.treatment.to_numpy(), observed.outcome.to_numpy()
    nuisance = pd.read_parquet(nuisance_path(position)).set_index("_oci_row_id")
    e, m = nuisance.loc[fit_ids, ["treatment_stacked", "outcome_stacked"]].to_numpy().T
    u, v = t - e, y - m
    tau0 = float(u @ v / (u @ u))
    contribution, weights = u * (v - tau0 * u), u ** 2
    matrices = embeddings(fit_ids)
    texts = chunk_texts(fit_ids)
    padded, mask = pad_chunk_embeddings(matrices)
    train_chunks, train_mask = torch.from_numpy(padded), torch.from_numpy(mask)
    source = SOURCE / f"components/neural_queries/outer_001_inner_{position:03d}"
    query_records = read(source / "query_records.json")
    with np.load(source / "queries.npz", allow_pickle=False) as saved:
        original_all = np.vstack([saved[b + "_queries"] for b in BANKS])
        expected_train = np.column_stack([saved[b + "_train_activations"] for b in BANKS])
        expected_valid = np.column_stack([saved[b + "_heldout_activations"] for b in BANKS])
    base_train = project(train_chunks, train_mask, original_all)
    train_error = float(np.max(np.abs(base_train - expected_train)))
    assert train_error < 5e-6, train_error
    validation_matrices = embeddings(valid_ids)
    valid_padded, valid_mask = pad_chunk_embeddings(validation_matrices)
    validation_chunks, validation_mask = torch.from_numpy(valid_padded), torch.from_numpy(valid_mask)
    base_valid = project(validation_chunks, validation_mask, original_all)
    valid_error = float(np.max(np.abs(base_valid - expected_valid)))
    assert valid_error < 5e-6, valid_error
    del validation_matrices
    originals = original_all[10:].copy()
    status(folder / "status.json", "optimize", inner_fold=position, train_activation_error=train_error,
           validation_activation_error=valid_error)
    chosen, history = optimize(train_chunks, train_mask, originals, contribution, weights, POLICY,
                               POLICY["seed"] + position * 10000,
                               progress=lambda value: status(folder / "status.json", "optimize", inner_fold=position, **value))
    variants = [{"query": j, "kind": "original", "radius": 0.0, "distance": 0.0, "control": -1} for j in range(5)]
    vectors = list(originals)
    for item in chosen:
        j, radius = item["query"], item["radius"]
        vector = item["vector"]
        distance = float(np.linalg.norm(vector - originals[j]))
        assert distance <= radius + 2e-6
        variants.append({"query": j, "kind": "targeted", "radius": radius, "distance": distance,
                         "control": -1, "restart": item["restart"], "training_criterion": item["criterion"]})
        vectors.append(vector)
        for control in range(POLICY["random_controls"]):
            seed = POLICY["seed"] + position * 100000 + j * 1000 + round(radius * 1000) + control
            random = random_at_distance(originals[j], distance, seed)
            assert abs(np.linalg.norm(random - originals[j]) - distance) < 2e-6
            variants.append({"query": j, "kind": "random", "radius": radius, "distance": distance,
                             "control": control, "seed": seed})
            vectors.append(random)
    vectors = np.asarray(vectors, dtype=np.float32)
    edit_train = project(train_chunks, train_mask, vectors)
    edit_valid = project(validation_chunks, validation_mask, vectors)
    for i, variant in enumerate(variants):
        ratio = np.std(edit_train[:, i]) / np.std(edit_train[:, variant["query"]])
        variant["training_sd_ratio"] = float(ratio)
        if variant["kind"] == "targeted":
            assert POLICY["relative_sd_bounds"][0] - 1e-5 <= ratio <= POLICY["relative_sd_bounds"][1] + 1e-5
    del padded, mask, train_chunks, train_mask, validation_chunks, validation_mask, valid_padded, valid_mask
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "activations.npz", vectors=vectors, train=edit_train, validation=edit_valid,
                        original_train=base_train, original_validation=base_valid,
                        fit_row_ids=np.asarray(fit_ids), validation_row_ids=np.asarray(valid_ids))
    write(folder / "variants.json", variants)
    write(folder / "optimization.json", {"history": history, "tau0": tau0, "nuisance_policy": POLICY["nuisances"],
                                          "activation_reproduction_max_abs_error": {"fit": train_error, "validation": valid_error}})
    packets = evidence_packets(position, fit_ids, matrices, texts, originals, vectors, variants, edit_train,
                               contribution, weights, [x["query_id"] for x in query_records["effect"]])
    for packet in packets:
        assert {x["row_id"] for x in packet["evidence"]} <= set(fit_ids)
        write(RUN / "packets" / (packet["packet_id"] + ".json"), packet)
    del matrices, texts

    predictions, prediction_metadata = [], []

    def save_pred(meta, pred):
        assert np.asarray(pred).shape == (len(valid_ids),) and np.isfinite(pred).all()
        prediction_metadata.append(meta)
        predictions.append(pred)

    for target, constant in (("treatment", float(t.mean())), ("outcome", float(y.mean())), ("effect", tau0)):
        save_pred({"query": -1, "target": target, "family": "constant", "scope": "constant", "mode": "constant",
                   "variant": -1}, np.full(len(valid_ids), constant))
    for j in range(5):
        for scope in ("solo", "conditional"):
            original_train = base_train[:, [10 + j]] if scope == "solo" else base_train
            original_valid = base_valid[:, [10 + j]] if scope == "solo" else base_valid
            for family in ("linear", "spline"):
                for target in ("treatment", "outcome", "effect"):
                    head = Probe(family, target, POLICY).fit(original_train, t, y, u, v)
                    meta = {"query": j, "scope": scope, "family": family, "target": target}
                    save_pred({**meta, "mode": "original", "variant": j}, head.predict(original_valid))
                    if scope == "conditional":
                        keep = [k for k in range(15) if k != 10 + j]
                        remaining = Probe(family, target, POLICY).fit(base_train[:, keep], t, y, u, v)
                        save_pred({**meta, "mode": "without_query", "variant": -1}, remaining.predict(base_valid[:, keep]))
                    for index, variant in enumerate(variants):
                        if variant["query"] != j or variant["kind"] == "original":
                            continue
                        if scope == "solo":
                            x_train, x_valid = edit_train[:, [index]], edit_valid[:, [index]]
                        else:
                            x_train, x_valid = base_train.copy(), base_valid.copy()
                            x_train[:, 10 + j], x_valid[:, 10 + j] = edit_train[:, index], edit_valid[:, index]
                        save_pred({**meta, "mode": "frozen", "variant": index}, head.predict(x_valid))
                        refit = Probe(family, target, POLICY).fit(x_train, t, y, u, v)
                        save_pred({**meta, "mode": "refit", "variant": index}, refit.predict(x_valid))
        status(folder / "status.json", "probe_fitting", inner_fold=position, queries_complete=j + 1)
    np.savez_compressed(folder / "probe_predictions.npz", predictions=np.asarray(predictions), row_ids=valid_ids)
    write(folder / "probe_metadata.json", prediction_metadata)
    files = {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name not in {"status.json", "frozen.json"}}
    frozen_record = {"at": now(), "inner_fold": position, "manifest_sha256": sha(RUN / "manifest.json"), "files": files,
                     "packets": {p["packet_id"]: sha(RUN / "packets" / (p["packet_id"] + ".json")) for p in packets},
                     "validation_outcomes_loaded": False, "outer_test_rows_used": [], "fit_row_ids": fit_ids,
                     "validation_row_ids": valid_ids, "prediction_rows": len(predictions)}
    write(frozen, frozen_record)
    status(folder / "status.json", "complete", inner_fold=position, variants=len(variants), probes=len(predictions))
    return frozen_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    args = parser.parse_args()
    positions = [args.fold] if args.fold else range(1, 6)
    if args.fold:
        fit_one(args.fold)
    else:
        with ProcessPoolExecutor(max_workers=POLICY["fold_workers"]) as pool:
            for result in as_completed([pool.submit(fit_one, i) for i in positions]):
                result.result()
        write(RUN / "numerical_frozen.json", {"at": now(), "manifest_sha256": sha(RUN / "manifest.json"),
              "folds": {f"inner_{i:03d}": sha(RUN / "folds" / f"inner_{i:03d}" / "frozen.json") for i in range(1, 6)}})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(RUN / "numerical_failed.json", {"at": now(), "error": str(error), "traceback": traceback.format_exc()})
        raise
