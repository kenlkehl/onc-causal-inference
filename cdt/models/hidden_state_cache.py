# cdt/models/hidden_state_cache.py
"""Pre-computed hidden state cache for FrozenLLMPoolerExtractor.

When the LLM is frozen, its hidden states are deterministic for a given input.
This module pre-computes hidden states once for the entire dataset and caches
them to disk as numpy memmap files. During training, only the lightweight
trainable layers (~200K params) need to be loaded to GPU.

Cache is keyed on (model_name, max_length, dataset_path) only, so it is
reusable across experiments with different causal heads, learning rates,
fold counts, etc.

Storage format (variable-length, no padding waste):
    {dataset_dir}/.cdt_cache/flp_hidden_states_{hash}/
        hidden_states.npy   - float16 memmap (total_tokens, hidden_size)
        offsets.npy          - int64 array (N+1,) sample boundaries
        metadata.json        - cache metadata with storage_format="variable_length"

Each sample's hidden states span [offsets[i], offsets[i+1]) in the flat array.
Per-batch padding to the batch-local max length happens during collation,
avoiding the global-max padding waste of the old format.
"""

import gc
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VariableLengthArray:
    """Array-like wrapper for variable-length sequences stored in a flat array.

    Supports integer indexing: arr[i] returns the slice for sample i.
    Used by CachedHiddenStateDataset to access per-sample hidden states.
    """

    def __init__(self, flat_array: np.ndarray, offsets: np.ndarray):
        self.flat = flat_array      # (total_tokens, hidden_size)
        self.offsets = offsets       # (N+1,) int64

    def __getitem__(self, idx: int) -> np.ndarray:
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        return self.flat[start:end]  # (seq_len_i, hidden_size)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def shape(self):
        """For logging compatibility."""
        return (len(self), 'variable', self.flat.shape[-1])


class VariableLengthMaskArray:
    """Generates all-ones attention masks of the correct length per sample.

    No data stored on disk -- masks are created on the fly since all stored
    tokens are real (non-padding).
    """

    def __init__(self, offsets: np.ndarray):
        self.offsets = offsets

    def __getitem__(self, idx: int) -> np.ndarray:
        length = int(self.offsets[idx + 1] - self.offsets[idx])
        return np.ones(length, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.offsets) - 1


class HiddenStateCache:
    """Pre-computed hidden state cache for frozen LLM extractors.

    Pre-computes LLM hidden states once for the entire dataset, stores them
    as float16 numpy memmap files in variable-length flat format, and provides
    batch loading for training.

    Args:
        cache_dir: Directory to store cache files.
        model_name: HuggingFace model name.
        max_length: Maximum sequence length for tokenization.
        dataset_path: Path to the dataset file (used for cache key).
    """

    def __init__(
        self,
        cache_dir: str,
        model_name: str,
        max_length: int,
        dataset_path: str,
    ):
        self._cache_dir = Path(cache_dir)
        self._model_name = model_name
        self._max_length = max_length
        self._dataset_path = dataset_path
        self._cache_hash = self.compute_cache_hash(model_name, max_length, dataset_path)

        # Actual cache location
        self._cache_path = self._cache_dir / f"flp_hidden_states_{self._cache_hash}"

        # Memmap handles (lazy, opened by open())
        self._hs_mmap = None
        self._mask_mmap = None
        self._flat_mmap = None
        self._offsets = None
        self._metadata = None
        self._preloaded = False

    @staticmethod
    def compute_cache_hash(model_name: str, max_length: int, dataset_path: str) -> str:
        """Compute deterministic hash for cache identification."""
        key = f"{model_name}|{max_length}|{os.path.abspath(dataset_path)}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    @property
    def cache_path(self) -> Path:
        """Return the cache directory path."""
        return self._cache_path

    @property
    def hidden_size(self) -> int:
        """Return hidden size from metadata."""
        if self._metadata is None:
            self._load_metadata()
        return self._metadata['hidden_size']

    @property
    def cache_size_gb(self) -> float:
        """Return approximate cache size in GB."""
        if self._metadata is None:
            self._load_metadata()
        total_tokens = self._metadata.get('total_tokens')
        if total_tokens is not None:
            hs = self._metadata['hidden_size']
            return total_tokens * hs * 2 / 1e9  # float16 only
        # Fallback for legacy padded format
        n = self._metadata['num_samples']
        seq_len = self._metadata['actual_max_len']
        hs = self._metadata['hidden_size']
        return (n * seq_len * hs * 2 + n * seq_len) / 1e9

    @property
    def hidden_states_array(self):
        """Return the hidden states array (VariableLengthArray or numpy)."""
        if self._hs_mmap is None:
            self.open()
        return self._hs_mmap

    @property
    def attention_mask_array(self):
        """Return the attention mask array (VariableLengthMaskArray or numpy)."""
        if self._mask_mmap is None:
            self.open()
        return self._mask_mmap

    @property
    def actual_max_len(self) -> int:
        """Return actual max tokenized length from metadata."""
        if self._metadata is None:
            self._load_metadata()
        return self._metadata['actual_max_len']

    def _load_metadata(self):
        """Load metadata from disk."""
        meta_path = self._cache_path / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Cache metadata not found: {meta_path}")
        with open(meta_path) as f:
            self._metadata = json.load(f)

    def is_valid(self, expected_num_samples: int) -> bool:
        """Check if cache exists and is valid.

        Requires variable_length storage format. Old padded-format caches
        are treated as invalid and will be regenerated.

        Args:
            expected_num_samples: Expected number of samples in the cache.

        Returns:
            True if cache is valid and matches expected size.
        """
        try:
            meta_path = self._cache_path / "metadata.json"
            hs_path = self._cache_path / "hidden_states.npy"
            offsets_path = self._cache_path / "offsets.npy"

            if not all(p.exists() for p in [meta_path, hs_path, offsets_path]):
                return False

            self._load_metadata()

            # Require variable_length format (old padded caches are invalid)
            if self._metadata.get('storage_format') != 'variable_length':
                logger.warning("Cache uses legacy padded format, will regenerate")
                return False

            if self._metadata.get('num_samples') != expected_num_samples:
                logger.warning(
                    f"Cache sample count mismatch: "
                    f"expected {expected_num_samples}, got {self._metadata.get('num_samples')}"
                )
                return False

            if self._metadata.get('cache_hash') != self._cache_hash:
                logger.warning("Cache hash mismatch")
                return False

            # Verify offsets
            offsets = np.load(str(offsets_path))
            if len(offsets) != expected_num_samples + 1:
                return False

            # Verify flat memmap shape
            hs = np.load(str(hs_path), mmap_mode='r')
            expected_total = int(offsets[-1])
            if hs.shape[0] != expected_total:
                return False

            logger.info(
                f"Valid hidden state cache found: {expected_num_samples} samples, "
                f"total_tokens={expected_total}, hash={self._cache_hash}"
            )
            return True

        except Exception as e:
            logger.warning(f"Cache validation failed: {e}")
            return False

    def precompute(
        self,
        texts: List[str],
        device: torch.device,
        batch_size: int = 4,
    ) -> None:
        """Pre-compute LLM hidden states for all texts and save to disk.

        Stores hidden states in variable-length flat format: only real tokens
        are saved (no padding). An offsets array indexes sample boundaries.

        Args:
            texts: All texts in the dataset (in order).
            device: Device to run the LLM on.
            batch_size: Batch size for LLM inference.
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        num_samples = len(texts)
        logger.info(f"Pre-computing hidden states for {num_samples} texts...")
        logger.info(f"  Model: {self._model_name}")
        logger.info(f"  Max length: {self._max_length}")
        logger.info(f"  Cache path: {self._cache_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # First pass: compute per-sample tokenized lengths
        logger.info("Pass 1/2: Computing per-sample tokenized lengths...")
        sequence_lengths = []
        for i in range(0, num_samples, batch_size * 4):
            batch_texts = texts[i:i + batch_size * 4]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                max_length=self._max_length,
                padding=False,
                return_length=True,
            )
            sequence_lengths.extend(encodings['length'])

        total_tokens = sum(sequence_lengths)
        actual_max_len = max(sequence_lengths)
        mean_len = total_tokens / num_samples
        padded_total = num_samples * actual_max_len
        savings_pct = 1 - total_tokens / padded_total

        logger.info(f"  Total tokens: {total_tokens:,} across {num_samples} samples")
        logger.info(f"  Mean length: {mean_len:.0f}, Max: {actual_max_len}")
        logger.info(f"  Savings vs padded: {savings_pct:.1%} "
                     f"({total_tokens:,} vs {padded_total:,} tokens)")

        # Compute offsets array
        offsets = np.zeros(num_samples + 1, dtype=np.int64)
        for i, length in enumerate(sequence_lengths):
            offsets[i + 1] = offsets[i] + length

        # Load model
        logger.info("Loading LLM for hidden state extraction...")
        hf_config = AutoConfig.from_pretrained(self._model_name, trust_remote_code=True)
        hidden_size = hf_config.hidden_size

        # Load to CPU first, then move to device.
        # Using device_map triggers accelerate's dispatch_model which loads
        # via meta tensors — this fails for models with tied weights (e.g. Qwen3).
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name, config=hf_config, trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model = model.to(device)

        if tokenizer.pad_token == '[PAD]':
            model.resize_token_embeddings(len(tokenizer))

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Create cache directory
        self._cache_path.mkdir(parents=True, exist_ok=True)

        # Create flat memmap (no padding)
        hs_path = self._cache_path / "hidden_states.npy"
        offsets_path = self._cache_path / "offsets.npy"

        hs_mmap = np.lib.format.open_memmap(
            str(hs_path), mode='w+', dtype=np.float16,
            shape=(total_tokens, hidden_size)
        )

        # Save offsets
        np.save(str(offsets_path), offsets)

        # Second pass: compute hidden states and write only real tokens
        logger.info("Pass 2/2: Computing hidden states...")
        processed = 0
        for i in range(0, num_samples, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_end = min(i + batch_size, num_samples)
            batch_lengths = sequence_lengths[i:batch_end]
            batch_max_len = max(batch_lengths)

            encoding = tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=batch_max_len,
                return_tensors="pt",
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states[-1]  # (batch, batch_max_len, hidden_size)

            # Write only real (non-padding) tokens to flat memmap
            hs_cpu = hidden_states.cpu().to(torch.float16).numpy()
            for j in range(len(batch_texts)):
                sample_len = batch_lengths[j]
                sample_offset = int(offsets[i + j])
                hs_mmap[sample_offset:sample_offset + sample_len] = hs_cpu[j, :sample_len]

            processed += len(batch_texts)
            if processed % (batch_size * 10) == 0 or processed == num_samples:
                logger.info(f"  Processed {processed}/{num_samples} texts")

        # Flush to disk
        hs_mmap.flush()
        del hs_mmap

        # Write metadata
        metadata = {
            'model_name': self._model_name,
            'max_length': self._max_length,
            'hidden_size': hidden_size,
            'num_samples': num_samples,
            'actual_max_len': actual_max_len,
            'total_tokens': total_tokens,
            'sequence_lengths': sequence_lengths,
            'storage_format': 'variable_length',
            'dataset_path': os.path.abspath(self._dataset_path),
            'cache_hash': self._cache_hash,
            'created_at': datetime.now().isoformat(),
            'dtype': 'float16',
        }
        with open(self._cache_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        self._metadata = metadata

        # Unload LLM
        logger.info("Unloading LLM from GPU...")
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log cache size
        total_bytes = total_tokens * hidden_size * 2  # float16
        padded_bytes = num_samples * actual_max_len * hidden_size * 2
        logger.info(
            f"Hidden state cache created: {total_bytes / 1e9:.2f} GB "
            f"(vs {padded_bytes / 1e9:.2f} GB if padded), "
            f"total_tokens={total_tokens:,}, savings={savings_pct:.1%}"
        )

    def open(self) -> None:
        """Open memmap files for reading (lazy).

        Wraps the flat memmap in VariableLengthArray/VariableLengthMaskArray
        so callers can index by sample: arr[i] -> (seq_len_i, hidden_size).
        """
        if self._hs_mmap is not None:
            return

        if self._metadata is None:
            self._load_metadata()

        hs_path = self._cache_path / "hidden_states.npy"
        offsets_path = self._cache_path / "offsets.npy"

        self._flat_mmap = np.load(str(hs_path), mmap_mode='r')
        self._offsets = np.load(str(offsets_path))

        self._hs_mmap = VariableLengthArray(self._flat_mmap, self._offsets)
        self._mask_mmap = VariableLengthMaskArray(self._offsets)

        logger.info(
            f"Hidden state cache opened: {len(self._hs_mmap)} samples, "
            f"total_tokens={self._metadata.get('total_tokens', 'unknown')}"
        )

    def preload_to_ram(self) -> None:
        """Load entire cache from disk into RAM for faster random access.

        After preloading, all reads come from RAM instead of disk-backed memmap,
        eliminating I/O latency for random batch access. The flat array is shared
        across forked DataLoader workers via copy-on-write.
        """
        if self._preloaded:
            return

        if self._hs_mmap is None:
            self.open()

        size_gb = self.cache_size_gb
        logger.info(f"Preloading hidden state cache to RAM ({size_gb:.2f} GB)...")

        # Copy flat memmap to RAM and rebuild wrapper
        ram_flat = np.array(self._flat_mmap)
        self._flat_mmap = ram_flat
        self._hs_mmap = VariableLengthArray(ram_flat, self._offsets)
        # _mask_mmap stays as VariableLengthMaskArray (generates on-the-fly, no I/O)
        self._preloaded = True

        logger.info(
            f"Hidden state cache preloaded to RAM: "
            f"{len(self._hs_mmap)} samples, {size_gb:.2f} GB"
        )

    def load_batch(
        self,
        indices: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load cached hidden states for a batch of indices.

        Gathers variable-length sequences and pads to the batch-local max.

        Args:
            indices: List of global cache indices.
            device: Device to place tensors on.

        Returns:
            Tuple of (hidden_states, attention_mask) tensors on device.
            hidden_states: (batch, batch_max_len, hidden_size) float32
            attention_mask: (batch, batch_max_len) float32
        """
        if self._hs_mmap is None:
            self.open()

        # Gather variable-length sequences
        sequences = [self._hs_mmap[i] for i in indices]
        lengths = [seq.shape[0] for seq in sequences]
        max_len = max(lengths)
        hidden_size = sequences[0].shape[-1]

        # Pad to batch max
        hs = np.zeros((len(indices), max_len, hidden_size), dtype=np.float16)
        mask = np.zeros((len(indices), max_len), dtype=np.float32)
        for i, (seq, length) in enumerate(zip(sequences, lengths)):
            hs[i, :length] = seq
            mask[i, :length] = 1.0

        # Transfer float16 to GPU, then cast to float32 on GPU
        hs_tensor = torch.from_numpy(hs).to(device).float()
        mask_tensor = torch.from_numpy(mask).to(device)

        return hs_tensor, mask_tensor

    def invalidate(self) -> None:
        """Delete cache directory."""
        self.close()
        if self._cache_path.exists():
            shutil.rmtree(self._cache_path)
            logger.info(f"Cache invalidated: {self._cache_path}")

    def close(self) -> None:
        """Release memmap handles."""
        self._hs_mmap = None
        self._mask_mmap = None
        self._flat_mmap = None
        self._offsets = None
