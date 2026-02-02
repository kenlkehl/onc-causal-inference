# cdt/models/numeric_features.py
"""Numeric feature extraction and embedding for clinical text.

Clinical text contains numbers critical for causal inference (lab values, vitals,
scores, doses, ages) but they receive no special treatment in standard tokenizers.
This module provides magnitude-aware numeric features that can be injected into
any feature extractor's pipeline.

Two modules are provided:

1. NumericEmbedding: Position-aligned embeddings for token-level extractors (GRU, CNN).
   Adds learnable magnitude and type embeddings at numeric token positions.

2. NumericFeatureVector: Fixed-size aggregate vector for chunk/document-level extractors
   (BERT, LLM, hierarchical). Summarizes all numbers in a text as a histogram.
"""

import logging
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default log-scale magnitude bin boundaries
DEFAULT_MAGNITUDE_BINS = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

# Clinical numeric type categories detected by preceding keyword
NUMERIC_TYPE_KEYWORDS: Dict[str, int] = {
    # Vitals
    'bp': 1, 'sbp': 1, 'dbp': 1, 'systolic': 1, 'diastolic': 1,
    'hr': 2, 'heartrate': 2, 'pulse': 2,
    'temp': 3, 'temperature': 3,
    'rr': 4, 'respiratory': 4,
    'spo2': 5, 'o2sat': 5, 'saturation': 5,
    # Labs
    'creatinine': 6, 'cr': 6, 'egfr': 6, 'gfr': 6, 'bun': 6,
    'hemoglobin': 6, 'hgb': 6, 'hb': 6, 'wbc': 6, 'platelets': 6, 'plt': 6,
    'albumin': 6, 'bilirubin': 6, 'alt': 6, 'ast': 6, 'ldh': 6,
    'sodium': 6, 'potassium': 6, 'calcium': 6, 'glucose': 6,
    'psa': 6, 'cea': 6, 'ca125': 6, 'afp': 6,
    # Clinical scores
    'ecog': 7, 'kps': 7, 'karnofsky': 7, 'ps': 7,
    'stage': 7, 'grade': 7, 'gleason': 7,
    'tnm': 7, 'ajcc': 7,
    # Demographics
    'age': 8, 'bmi': 8, 'weight': 8, 'height': 8,
    # Doses
    'mg': 9, 'dose': 9, 'gy': 9, 'mg/m2': 9, 'mcg': 9, 'units': 9,
}

# Number of type categories (0 = unknown/other, 1-9 = specific types)
NUM_TYPE_CATEGORIES = 10

# Regex patterns for number extraction
_NUMBER_PATTERN = re.compile(
    r'(?<!\w)'                   # not preceded by word char
    r'(\d+(?:\.\d+)?)'          # integer or decimal
    r'(?:\s*/\s*(\d+(?:\.\d+)?))?'  # optional /denominator (e.g., 120/80)
    r'(?!\w)',                   # not followed by word char
    re.IGNORECASE
)

_WORD_PATTERN = re.compile(r'[a-z0-9/]+', re.IGNORECASE)


@dataclass
class NumericPattern:
    """A detected numeric value with context."""
    value: float
    magnitude_bin: int
    type_category: int
    char_start: int
    char_end: int


def _magnitude_bin(value: float, bins: List[float]) -> int:
    """Assign a value to a log-scale magnitude bin.

    Returns bin index in [0, len(bins)-1].
    """
    abs_val = abs(value)
    for i, boundary in enumerate(bins):
        if abs_val < boundary:
            return max(0, i - 1)
    return len(bins) - 1


def _get_preceding_word(text: str, match_start: int) -> Optional[str]:
    """Get the word immediately preceding a match position."""
    # Look back from match_start, skipping whitespace
    pos = match_start - 1
    while pos >= 0 and text[pos] in ' \t':
        pos -= 1
    if pos < 0:
        return None
    # Find word boundary
    end = pos + 1
    while pos >= 0 and (text[pos].isalnum() or text[pos] in '/-'):
        pos -= 1
    word = text[pos+1:end].lower().strip('/-')
    return word if word else None


def extract_numeric_patterns(
    text: str,
    magnitude_bins: Optional[List[float]] = None
) -> List[NumericPattern]:
    """Extract numeric patterns from clinical text.

    Detects integers, decimals, and fractions (e.g., 120/80 for BP).
    Assigns magnitude bins and type categories based on preceding keywords.

    Args:
        text: Clinical text string
        magnitude_bins: Log-scale bin boundaries (default: DEFAULT_MAGNITUDE_BINS)

    Returns:
        List of NumericPattern instances
    """
    if magnitude_bins is None:
        magnitude_bins = DEFAULT_MAGNITUDE_BINS

    patterns = []
    for match in _NUMBER_PATTERN.finditer(text):
        numerator = float(match.group(1))
        denominator_str = match.group(2)

        # Get preceding word for type detection
        preceding = _get_preceding_word(text, match.start())
        type_cat = NUMERIC_TYPE_KEYWORDS.get(preceding, 0) if preceding else 0

        # Primary number
        patterns.append(NumericPattern(
            value=numerator,
            magnitude_bin=_magnitude_bin(numerator, magnitude_bins),
            type_category=type_cat,
            char_start=match.start(),
            char_end=match.end(),
        ))

        # If fraction (e.g., BP 120/80), add denominator too
        if denominator_str is not None:
            denom = float(denominator_str)
            patterns.append(NumericPattern(
                value=denom,
                magnitude_bin=_magnitude_bin(denom, magnitude_bins),
                type_category=type_cat,
                char_start=match.start(),
                char_end=match.end(),
            ))

    return patterns


class NumericEmbedding(nn.Module):
    """Position-aligned numeric embeddings for token-level extractors.

    For each token position in a sequence, produces a numeric embedding vector
    that is zero for non-numeric tokens and a learned magnitude+type embedding
    for numeric tokens. This is added to word embeddings before the encoder
    (GRU, CNN).

    The alignment works by matching detected numbers in the raw text to
    token positions via character offset mapping.

    Args:
        num_magnitude_bins: Number of log-scale magnitude bins
        num_type_categories: Number of numeric type categories
        embedding_dim: Dimension of numeric embeddings (should match word embedding dim)
    """

    def __init__(
        self,
        num_magnitude_bins: int = 8,
        num_type_categories: int = NUM_TYPE_CATEGORIES,
        embedding_dim: int = 32
    ):
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.num_type_categories = num_type_categories
        self.embedding_dim = embedding_dim

        self.magnitude_embed = nn.Embedding(num_magnitude_bins, embedding_dim)
        self.type_embed = nn.Embedding(num_type_categories, embedding_dim)

        # Projection to combine magnitude + type (2*dim -> dim)
        self.combine = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)

        # Scale factor so addition to word embeddings starts small
        self.scale = nn.Parameter(torch.tensor(0.1))

        logger.info(f"NumericEmbedding: {num_magnitude_bins} bins, "
                    f"{num_type_categories} types, dim={embedding_dim}")

    def forward(
        self,
        texts: List[str],
        token_ids: torch.Tensor,
        tokenize_fn,
        word_to_id: dict
    ) -> torch.Tensor:
        """Compute position-aligned numeric embeddings.

        Args:
            texts: Raw text strings
            token_ids: (batch, seq_len) token ID tensor
            tokenize_fn: Function that tokenizes text to word list
            word_to_id: Vocabulary mapping

        Returns:
            numeric_embs: (batch, seq_len, embedding_dim) tensor
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        result = torch.zeros(batch_size, seq_len, self.embedding_dim, device=device)

        for b, text in enumerate(texts):
            # Extract numeric patterns from raw text
            patterns = extract_numeric_patterns(text)
            if not patterns:
                continue

            # Tokenize to get token list
            tokens = tokenize_fn(text)

            # Build a mapping from token index to numeric patterns
            # by matching number strings to token positions
            for pattern in patterns:
                # Find the token that contains this number
                value_str = str(int(pattern.value)) if pattern.value == int(pattern.value) else str(pattern.value)
                for tok_idx, tok in enumerate(tokens):
                    if tok_idx >= seq_len:
                        break
                    if tok == value_str or tok == value_str.rstrip('0').rstrip('.'):
                        mag_emb = self.magnitude_embed(
                            torch.tensor(pattern.magnitude_bin, device=device)
                        )
                        type_emb = self.type_embed(
                            torch.tensor(pattern.type_category, device=device)
                        )
                        combined = self.combine(torch.cat([mag_emb, type_emb]))
                        result[b, tok_idx] = combined
                        break

        return result * self.scale


class NumericFeatureVector(nn.Module):
    """Fixed-size numeric feature vector for chunk/document-level extractors.

    Aggregates all detected numbers in a text into a histogram of magnitude bins
    and type counts, then projects to a fixed-size vector. This is concatenated
    to chunk or document embeddings.

    Args:
        num_magnitude_bins: Number of log-scale magnitude bins
        num_type_categories: Number of numeric type categories
        output_dim: Dimension of output feature vector
    """

    def __init__(
        self,
        num_magnitude_bins: int = 8,
        num_type_categories: int = NUM_TYPE_CATEGORIES,
        output_dim: int = 32
    ):
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.num_type_categories = num_type_categories
        self.output_dim = output_dim

        # Input: magnitude histogram + type counts + count features
        input_dim = num_magnitude_bins + num_type_categories + 1  # +1 for total count
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

        # Scale factor for smooth integration
        self.scale = nn.Parameter(torch.tensor(0.1))

        logger.info(f"NumericFeatureVector: {num_magnitude_bins} bins, "
                    f"{num_type_categories} types, output_dim={output_dim}")

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Extract numeric feature vectors from texts.

        Args:
            texts: List of text strings (can be chunks or full documents)

        Returns:
            features: (batch, output_dim) numeric feature vectors
        """
        batch_size = len(texts)
        device = next(self.parameters()).device

        # Build histograms
        histograms = torch.zeros(
            batch_size,
            self.num_magnitude_bins + self.num_type_categories + 1,
            device=device
        )

        for b, text in enumerate(texts):
            patterns = extract_numeric_patterns(text)
            if not patterns:
                continue

            for p in patterns:
                histograms[b, p.magnitude_bin] += 1
                histograms[b, self.num_magnitude_bins + p.type_category] += 1

            # Total count (log-scaled to avoid dominating)
            histograms[b, -1] = math.log1p(len(patterns))

        # Normalize magnitude histogram by total count
        total = histograms[:, :self.num_magnitude_bins].sum(dim=1, keepdim=True).clamp(min=1)
        histograms[:, :self.num_magnitude_bins] /= total

        # Normalize type histogram by total count
        total_type = histograms[:, self.num_magnitude_bins:-1].sum(dim=1, keepdim=True).clamp(min=1)
        histograms[:, self.num_magnitude_bins:-1] /= total_type

        features = self.projection(histograms)
        return features * self.scale

    def forward_batch_chunks(self, chunk_texts_list: List[List[str]]) -> List[torch.Tensor]:
        """Extract numeric features for a batch of document chunks.

        Args:
            chunk_texts_list: List of lists of chunk strings, one per document

        Returns:
            List of (num_chunks, output_dim) tensors, one per document
        """
        results = []
        for chunk_texts in chunk_texts_list:
            if chunk_texts:
                features = self.forward(chunk_texts)
                results.append(features)
            else:
                device = next(self.parameters()).device
                results.append(torch.zeros(0, self.output_dim, device=device))
        return results
