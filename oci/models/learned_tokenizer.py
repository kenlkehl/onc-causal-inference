# oci/models/learned_tokenizer.py
"""Simple word-level tokenizer for trainable-from-scratch extractors.

Builds vocabulary from training texts via word frequency counting.
Used by hierarchical_cnn, hierarchical_gru, and simple_cnn extractors.
"""

import re
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import torch

logger = logging.getLogger(__name__)

# Special tokens
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]


class LearnedTokenizer:
    """Word-level tokenizer that builds vocabulary from training data.

    Tokenization: lowercase + regex word splitting (alphanumeric sequences and
    individual non-whitespace characters). Top vocab_size tokens by frequency
    are kept; the rest map to [UNK].
    """

    def __init__(self):
        self._word2idx: Dict[str, int] = {}
        self._idx2word: Dict[int, str] = {}
        self._fitted = False

    @property
    def vocab_size(self) -> int:
        if not self._fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit() first.")
        return len(self._word2idx)

    @property
    def pad_token_id(self) -> int:
        return 0  # [PAD] is always index 0

    @property
    def unk_token_id(self) -> int:
        return 1  # [UNK] is always index 1

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        texts: List[str],
        vocab_size: int = 50000,
        min_freq: int = 2,
    ) -> None:
        """Build vocabulary from training texts.

        Args:
            texts: Training text corpus.
            vocab_size: Maximum vocabulary size (including special tokens).
            min_freq: Minimum word frequency to include in vocabulary.
        """
        counter = Counter()
        for text in texts:
            tokens = _tokenize(text)
            counter.update(tokens)

        # Filter by min_freq, sort by frequency descending
        max_words = vocab_size - len(SPECIAL_TOKENS)
        filtered = [
            (word, count) for word, count in counter.most_common()
            if count >= min_freq
        ][:max_words]

        # Build mappings
        self._word2idx = {}
        for i, token in enumerate(SPECIAL_TOKENS):
            self._word2idx[token] = i

        for word, _ in filtered:
            self._word2idx[word] = len(self._word2idx)

        self._idx2word = {v: k for k, v in self._word2idx.items()}
        self._fitted = True

        logger.info(
            f"LearnedTokenizer fitted: {len(self._word2idx)} tokens "
            f"(from {len(counter)} unique words in {len(texts)} texts, "
            f"min_freq={min_freq})"
        )

    def encode(self, text: str, max_length: int) -> List[int]:
        """Tokenize and encode a single text to token IDs.

        Args:
            text: Input text string.
            max_length: Maximum number of tokens (truncates if longer).

        Returns:
            List of integer token IDs.
        """
        if not self._fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit() first.")

        tokens = _tokenize(text)
        ids = [
            self._word2idx.get(t, self.unk_token_id)
            for t in tokens[:max_length]
        ]
        return ids

    def encode_batch(
        self,
        texts: List[str],
        max_length: int,
        padding: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of texts with optional padding.

        Args:
            texts: List of input texts.
            max_length: Maximum sequence length.
            padding: Whether to pad to the longest sequence in the batch.

        Returns:
            input_ids: (B, seq_len) long tensor
            attention_mask: (B, seq_len) float tensor (1=real, 0=pad)
        """
        encoded = [self.encode(text, max_length) for text in texts]

        if padding:
            max_len = max(len(ids) for ids in encoded) if encoded else 0
            max_len = max(max_len, 1)  # at least length 1
        else:
            max_len = max_length

        input_ids = torch.full((len(texts), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), max_len)

        for i, ids in enumerate(encoded):
            length = min(len(ids), max_len)
            input_ids[i, :length] = torch.tensor(ids[:length], dtype=torch.long)
            attention_mask[i, :length] = 1.0

        return input_ids, attention_mask

    def get_state(self) -> Dict[str, Any]:
        """Serialize tokenizer state for checkpointing."""
        return {
            'word2idx': self._word2idx,
            'fitted': self._fitted,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore tokenizer state from checkpoint."""
        self._word2idx = state['word2idx']
        self._idx2word = {v: k for k, v in self._word2idx.items()}
        self._fitted = state['fitted']


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word tokens.

    Splits on word boundaries: alphanumeric sequences and individual
    non-whitespace characters (punctuation, symbols).
    """
    return re.findall(r'[a-z0-9]+|[^\s]', text.lower())
