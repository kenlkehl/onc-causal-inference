"""Candidate-variable X features with unsupervised text-SVD W features.

This extractor assumes an upstream model has already converted clinical text
into candidate variable columns, for example ``llm_extracted_*`` fields. It
does not use causal roles or a fixed number of variables: every matching
candidate column is encoded as an X feature, while W is learned from raw text
with an unsupervised TF-IDF/SVD pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class CandidateVariableTextSVDConfig:
    """Configuration for candidate-variable X plus unsupervised text-SVD W."""

    candidate_prefix: str = "llm_extracted_"
    text_column: str = "clinical_text"
    numeric_min_nonmissing: float = 0.5
    w_dim: int = 64
    max_features: int = 40000
    min_df: int = 2
    max_df: float = 0.98
    ngram_range: Tuple[int, int] = (1, 2)
    sublinear_tf: bool = True


class CandidateVariableTextSVDExtractor:
    """Fold-local transformer that yields ``X`` and ``W`` matrices.

    ``X`` is a supervised-stage input made from all discovered candidate
    variables. ``W`` is an unsupervised nuisance representation from raw text.
    The class intentionally does not know which candidates are confounders or
    effect modifiers.
    """

    def __init__(
        self,
        config: Optional[CandidateVariableTextSVDConfig] = None,
        candidate_columns: Optional[Sequence[str]] = None,
        random_state: int = 0,
    ):
        self.config = config or CandidateVariableTextSVDConfig()
        self.candidate_columns = list(candidate_columns) if candidate_columns else None
        self.random_state = int(random_state)

        self.numeric_columns_: List[str] = []
        self.categorical_columns_: List[str] = []
        self.candidate_columns_: List[str] = []
        self.x_preprocessor_: Optional[ColumnTransformer] = None
        self.text_vectorizer_: Optional[TfidfVectorizer] = None
        self.w_svd_: Optional[TruncatedSVD] = None
        self.w_scaler_: Optional[StandardScaler] = None

    def _discover_candidate_columns(self, df: pd.DataFrame) -> List[str]:
        if self.candidate_columns is not None:
            missing = [col for col in self.candidate_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Candidate columns not found: {missing}")
            return list(self.candidate_columns)

        columns = [
            col
            for col in df.columns
            if col.startswith(self.config.candidate_prefix)
        ]
        if not columns:
            raise ValueError(
                "No candidate variable columns found with prefix "
                f"{self.config.candidate_prefix!r}"
            )
        return columns

    def _split_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        numeric: List[str] = []
        categorical: List[str] = []
        for col in self.candidate_columns_:
            parsed = pd.to_numeric(df[col], errors="coerce")
            if parsed.notna().mean() >= self.config.numeric_min_nonmissing:
                numeric.append(col)
            else:
                categorical.append(col)
        return numeric, categorical

    def _prepare_candidate_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df[self.candidate_columns_].copy()
        for col in self.numeric_columns_:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        for col in self.categorical_columns_:
            frame[col] = frame[col].astype("string").fillna("__missing__")
        return frame

    def fit(self, df: pd.DataFrame) -> "CandidateVariableTextSVDExtractor":
        if self.config.text_column not in df.columns:
            raise ValueError(f"Text column not found: {self.config.text_column}")

        self.candidate_columns_ = self._discover_candidate_columns(df)
        self.numeric_columns_, self.categorical_columns_ = self._split_column_types(df)
        candidate_frame = self._prepare_candidate_frame(df)

        transformers = []
        if self.numeric_columns_:
            transformers.append(
                (
                    "num",
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                    ),
                    self.numeric_columns_,
                )
            )
        if self.categorical_columns_:
            transformers.append(
                (
                    "cat",
                    make_pipeline(
                        SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    self.categorical_columns_,
                )
            )
        if not transformers:
            raise ValueError("No usable candidate variable columns were found")

        self.x_preprocessor_ = ColumnTransformer(
            transformers,
            sparse_threshold=0.0,
        )
        self.x_preprocessor_.fit(candidate_frame)

        self.text_vectorizer_ = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            max_features=self.config.max_features,
            sublinear_tf=self.config.sublinear_tf,
            dtype=np.float32,
        )
        text_features = self.text_vectorizer_.fit_transform(
            df[self.config.text_column].astype(str).tolist()
        )
        n_components = min(
            int(self.config.w_dim),
            text_features.shape[1] - 1,
            len(df) - 2,
        )
        if n_components < 1:
            raise ValueError("Not enough text features or samples to fit W SVD")
        self.w_svd_ = TruncatedSVD(
            n_components=n_components,
            random_state=self.random_state,
        )
        w = self.w_svd_.fit_transform(text_features).astype(np.float32)
        self.w_scaler_ = StandardScaler().fit(w)
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if self.x_preprocessor_ is None:
            raise RuntimeError("Extractor must be fit before transform")
        if self.text_vectorizer_ is None or self.w_svd_ is None or self.w_scaler_ is None:
            raise RuntimeError("W text pipeline must be fit before transform")

        candidate_frame = self._prepare_candidate_frame(df)
        x = self.x_preprocessor_.transform(candidate_frame).astype(np.float32)

        text_features = self.text_vectorizer_.transform(
            df[self.config.text_column].astype(str).tolist()
        )
        w = self.w_svd_.transform(text_features).astype(np.float32)
        w = self.w_scaler_.transform(w).astype(np.float32)
        return {"X": x, "W": w}

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        return self.fit(df).transform(df)

    def diagnostics(self) -> Dict[str, object]:
        if self.x_preprocessor_ is None:
            x_dim = None
        else:
            x_dim = int(len(self.x_preprocessor_.get_feature_names_out()))
        return {
            "candidate_columns": list(self.candidate_columns_),
            "numeric_columns": list(self.numeric_columns_),
            "categorical_columns": list(self.categorical_columns_),
            "x_dim": x_dim,
            "w_dim": None if self.w_svd_ is None else int(self.w_svd_.n_components),
        }
