#!/usr/bin/env python
"""Run CausalForestDML with raw-text neural mention slots as X.

The experiment has two unsupervised text branches:

* X branch: generic label/value mentions -> frozen-LM embeddings ->
  train-fold slot clustering -> continuous patient-slot activations.
* W branch: raw clinical text -> train-fold TF-IDF/SVD nuisance summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from oci.models.neural_mention_slot_extractor import (  # noqa: E402
    MentionRecord,
    NeuralMentionSlotConfig,
    NeuralMentionSlotExtractor,
    embed_mention_texts,
    extract_mention_records,
)
from run_oracle_experiments import _resolve_parquet_file, compute_metrics  # noqa: E402


def _fit_text_svd_w(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    w_dim: int,
    max_features: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)
    n_components = min(int(w_dim), train_tfidf.shape[1] - 1, len(train_texts) - 2)
    if n_components < 1:
        raise ValueError("Not enough text features or rows to fit W SVD")
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    W_train = svd.fit_transform(train_tfidf).astype(np.float32)
    W_test = svd.transform(test_tfidf).astype(np.float32)
    scaler = StandardScaler().fit(W_train)
    return (
        scaler.transform(W_train).astype(np.float32),
        scaler.transform(W_test).astype(np.float32),
    )


def _fit_predict_nuisance(
    W_train: np.ndarray,
    W_test: np.ndarray,
    treatment_train: np.ndarray,
    outcome_train: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    propensity_model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
    )
    propensity_model.fit(W_train, treatment_train)
    propensity = propensity_model.predict_proba(W_test)[:, 1]

    pred_y: List[np.ndarray] = []
    for arm in (0, 1):
        mask = treatment_train == arm
        model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=seed + 100 + arm,
            n_jobs=-1,
        )
        model.fit(W_train[mask], outcome_train[mask])
        pred_y.append(np.clip(model.predict(W_test), 1e-3, 1 - 1e-3))

    return (
        np.clip(propensity, 1e-3, 1 - 1e-3),
        pred_y[0],
        pred_y[1],
    )


def _metrics_from_predictions(results_df: pd.DataFrame) -> Dict[str, float]:
    return compute_metrics(
        pred_ite=results_df["pred_ite_prob"].values,
        true_ite=results_df["true_ite_prob"].values,
        pred_propensity=results_df["pred_propensity"].values,
        true_treatment=results_df["treatment_indicator"].values,
        pred_y0=results_df["pred_y0_prob"].values,
        pred_y1=results_df["pred_y1_prob"].values,
        true_y0=results_df["true_y0_prob"].values,
        true_y1=results_df["true_y1_prob"].values,
        true_outcome=results_df["outcome_indicator"].values,
    )


def _records_for_patients(
    records: Sequence[MentionRecord],
    patient_set: set[int],
) -> Tuple[List[MentionRecord], np.ndarray]:
    indices = [
        idx
        for idx, record in enumerate(records)
        if int(record.patient_index) in patient_set
    ]
    return [records[idx] for idx in indices], np.asarray(indices, dtype=np.int64)


def _slot_examples(
    extractor: NeuralMentionSlotExtractor,
    records: Sequence[MentionRecord],
    embeddings: np.ndarray,
    max_slots: int = 8,
    per_slot: int = 3,
) -> Dict[str, List[str]]:
    if extractor.kmeans_ is None or len(records) == 0:
        return {}
    centers = extractor.kmeans_.cluster_centers_.astype(np.float32)
    distances = (
        np.sum(embeddings ** 2, axis=1, keepdims=True)
        - 2.0 * embeddings @ centers.T
        + np.sum(centers ** 2, axis=1)
    )
    examples: Dict[str, List[str]] = {}
    for slot in range(min(max_slots, centers.shape[0])):
        nearest = np.argsort(distances[:, slot])[:per_slot]
        examples[str(slot)] = [
            f"{records[idx].label}: {records[idx].value}"[:240]
            for idx in nearest
        ]
    return examples


def _embed_with_tfidf_svd(
    texts: Sequence[str],
    dim: int,
    seed: int,
) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    n_components = min(int(dim), matrix.shape[1] - 1, len(texts) - 2)
    if n_components < 1:
        return matrix.toarray().astype(np.float32)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    embeddings = svd.fit_transform(matrix).astype(np.float32)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-6)
    return (embeddings / norm).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--mention-encoder", choices=["frozen_lm", "tfidf_svd"], default="frozen_lm")
    parser.add_argument("--encoder-model", default="answerdotai/ModernBERT-large")
    parser.add_argument("--slot-text-mode", choices=["label", "label_value"], default="label")
    parser.add_argument("--value-feature-dim", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--encoder-max-length", type=int, default=128)
    parser.add_argument("--embedding-cache-dir", default="../pcori_experiments/mention_embedding_cache")
    parser.add_argument("--tfidf-embedding-dim", type=int, default=256)

    parser.add_argument("--n-slots", type=int, default=64)
    parser.add_argument("--assignment-temperature", type=float, default=0.35)
    parser.add_argument("--top-assignments", type=int, default=4)
    parser.add_argument("--max-mentions-per-patient", type=int, default=160)
    parser.add_argument("--w-dim", type=int, default=64)
    parser.add_argument("--w-max-features", type=int, default=40000)
    parser.add_argument("--cf-n-estimators", type=int, default=400)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=10)
    parser.add_argument("--max-patients", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = _resolve_parquet_file(args.dataset)
    if parquet_file is None:
        raise FileNotFoundError(f"No parquet dataset found under {args.dataset}")
    df = pd.read_parquet(parquet_file).reset_index(drop=True)
    if args.max_patients is not None:
        df = df.iloc[: int(args.max_patients)].reset_index(drop=True)
    texts = df[args.text_column].astype(str).tolist()

    records = extract_mention_records(
        texts,
        max_mentions_per_patient=args.max_mentions_per_patient,
    )
    if not records:
        raise RuntimeError("No generic mention records were extracted from text")
    if args.slot_text_mode == "label":
        slot_texts = [f"label: {record.label}" for record in records]
    else:
        slot_texts = [record.encoder_text for record in records]
    value_texts = [f"value: {record.value}" for record in records]

    unique_slot_texts = list(dict.fromkeys(slot_texts))
    unique_slot_index = {text: idx for idx, text in enumerate(unique_slot_texts)}
    slot_inverse = np.asarray(
        [unique_slot_index[text] for text in slot_texts],
        dtype=np.int64,
    )
    unique_value_texts = list(dict.fromkeys(value_texts))
    unique_value_index = {text: idx for idx, text in enumerate(unique_value_texts)}
    value_inverse = np.asarray(
        [unique_value_index[text] for text in value_texts],
        dtype=np.int64,
    )

    print(
        f"parsed_mentions={len(records)} unique_slot_texts={len(unique_slot_texts)} "
        f"unique_value_texts={len(unique_value_texts)} patients={len(df)}",
        flush=True,
    )
    if args.mention_encoder == "frozen_lm":
        unique_slot_embeddings = embed_mention_texts(
            unique_slot_texts,
            model_name=args.encoder_model,
            batch_size=args.batch_size,
            max_length=args.encoder_max_length,
            device=args.device,
            cache_dir=args.embedding_cache_dir,
        )
        unique_value_embeddings = embed_mention_texts(
            unique_value_texts,
            model_name=args.encoder_model,
            batch_size=args.batch_size,
            max_length=args.encoder_max_length,
            device=args.device,
            cache_dir=args.embedding_cache_dir,
        )
    else:
        unique_slot_embeddings = _embed_with_tfidf_svd(
            unique_slot_texts,
            dim=args.tfidf_embedding_dim,
            seed=args.random_state,
        )
        unique_value_embeddings = _embed_with_tfidf_svd(
            unique_value_texts,
            dim=args.tfidf_embedding_dim,
            seed=args.random_state,
        )
    mention_slot_embeddings = unique_slot_embeddings[slot_inverse].astype(np.float32)
    mention_value_embeddings = unique_value_embeddings[value_inverse].astype(np.float32)

    slot_config = NeuralMentionSlotConfig(
        n_slots=args.n_slots,
        assignment_temperature=args.assignment_temperature,
        top_assignments=args.top_assignments,
        max_mentions_per_patient=args.max_mentions_per_patient,
        value_feature_dim=args.value_feature_dim,
        random_state=args.random_state,
    )

    kf = KFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=args.random_state,
    )
    predictions: List[pd.DataFrame] = []
    diagnostics: List[Dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
        train_set = set(int(idx) for idx in train_idx)
        test_set = set(int(idx) for idx in test_idx)
        train_records, train_record_indices = _records_for_patients(
            records,
            train_set,
        )
        test_records, test_record_indices = _records_for_patients(
            records,
            test_set,
        )
        train_slot_embeddings = mention_slot_embeddings[train_record_indices]
        test_slot_embeddings = mention_slot_embeddings[test_record_indices]
        train_value_embeddings = mention_value_embeddings[train_record_indices]
        test_value_embeddings = mention_value_embeddings[test_record_indices]

        fold_config = NeuralMentionSlotConfig(
            **{**asdict(slot_config), "random_state": args.random_state + fold}
        )
        slot_extractor = NeuralMentionSlotExtractor(fold_config)
        X_train = slot_extractor.fit_transform(
            train_records,
            train_slot_embeddings,
            patient_indices=[int(idx) for idx in train_idx],
            value_embeddings=train_value_embeddings,
        )
        X_test = slot_extractor.transform(
            test_records,
            test_slot_embeddings,
            patient_indices=[int(idx) for idx in test_idx],
            value_embeddings=test_value_embeddings,
        )
        W_train, W_test = _fit_text_svd_w(
            [texts[idx] for idx in train_idx],
            [texts[idx] for idx in test_idx],
            w_dim=args.w_dim,
            max_features=args.w_max_features,
            seed=args.random_state + 100 + fold,
        )

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        treatment_train = train_df["treatment_indicator"].to_numpy(dtype=int)
        outcome_train = train_df["outcome_indicator"].to_numpy(dtype=float)

        cf_seed = args.random_state * 1000 + fold
        cf = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=20,
                random_state=cf_seed + 1,
                n_jobs=-1,
            ),
            model_t=RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=20,
                random_state=cf_seed + 2,
                n_jobs=-1,
            ),
            discrete_treatment=True,
            n_estimators=args.cf_n_estimators,
            min_samples_leaf=args.cf_min_samples_leaf,
            random_state=cf_seed + 3,
            n_jobs=-1,
            cv=3,
        )
        cf.fit(Y=outcome_train, T=treatment_train, X=X_train, W=W_train)
        pred_tau = cf.effect(X_test).reshape(-1)

        propensity, pred_y0, pred_y1 = _fit_predict_nuisance(
            W_train=W_train,
            W_test=W_test,
            treatment_train=treatment_train,
            outcome_train=outcome_train,
            seed=cf_seed + 100,
        )

        fold_preds = test_df.copy()
        fold_preds["pred_ite_prob"] = pred_tau
        fold_preds["pred_tau"] = pred_tau
        fold_preds["pred_propensity"] = propensity
        fold_preds["pred_y0_prob"] = pred_y0
        fold_preds["pred_y1_prob"] = pred_y1
        fold_preds["cv_fold"] = fold
        predictions.append(fold_preds)

        fold_diag = {
            "fold": fold,
            "train_mentions": len(train_records),
            "test_mentions": len(test_records),
            **slot_extractor.diagnostics(),
            "slot_examples": _slot_examples(
                slot_extractor,
                train_records,
                train_slot_embeddings,
            ),
        }
        diagnostics.append(fold_diag)
        print(
            f"fold={fold} X={X_train.shape[1]} W={W_train.shape[1]} "
            f"train_mentions={len(train_records)} test_mentions={len(test_records)}",
            flush=True,
        )

    results_df = pd.concat(predictions).sort_index()
    metrics = _metrics_from_predictions(results_df)
    result = {
        "config": vars(args),
        "slot_config": asdict(slot_config),
        "parquet_file": str(parquet_file),
        "mention_count": len(records),
        "unique_slot_text_count": len(unique_slot_texts),
        "unique_value_text_count": len(unique_value_texts),
        "diagnostics": diagnostics,
        "metrics": metrics,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame([metrics]).to_csv(output_dir / "summary.csv", index=False)
    results_df.to_csv(output_dir / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
