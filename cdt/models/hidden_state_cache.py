# cdt/models/hidden_state_cache.py
"""Pre-computed hidden state cache for FrozenLLMPoolerExtractor.

When the LLM is frozen, its hidden states are deterministic for a given input.
This module pre-computes hidden states once for the entire dataset and caches
them to disk as numpy memmap files. During training, only the lightweight
trainable layers (~200K params) need to be loaded to GPU.

Cache is keyed on (model_name, max_length, dataset_path) only, so it is
reusable across experiments with different causal heads, learning rates,
fold counts, etc.

Storage format:
    {dataset_dir}/.cdt_cache/flp_hidden_states_{hash}/
        hidden_states.npy   - float16 memmap (N, actual_max_len, hidden_size)
        attention_mask.npy   - uint8 memmap (N, actual_max_len)
        metadata.json        - cache metadata
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


class HiddenStateCache:
    """Pre-computed hidden state cache for frozen LLM extractors.

    Pre-computes LLM hidden states once for the entire dataset, stores them
    as float16 numpy memmap files, and provides batch loading for training.

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
        self._metadata = None

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

        Args:
            expected_num_samples: Expected number of samples in the cache.

        Returns:
            True if cache is valid and matches expected size.
        """
        try:
            meta_path = self._cache_path / "metadata.json"
            hs_path = self._cache_path / "hidden_states.npy"
            mask_path = self._cache_path / "attention_mask.npy"

            if not all(p.exists() for p in [meta_path, hs_path, mask_path]):
                return False

            self._load_metadata()

            if self._metadata.get('num_samples') != expected_num_samples:
                logger.warning(
                    f"Cache sample count mismatch: "
                    f"expected {expected_num_samples}, got {self._metadata.get('num_samples')}"
                )
                return False

            if self._metadata.get('cache_hash') != self._cache_hash:
                logger.warning("Cache hash mismatch")
                return False

            # Verify memmap files can be opened
            hs = np.load(str(hs_path), mmap_mode='r')
            if hs.shape[0] != expected_num_samples:
                return False

            logger.info(
                f"Valid hidden state cache found: {expected_num_samples} samples, "
                f"shape {hs.shape}, hash={self._cache_hash}"
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

        Loads the LLM temporarily, processes all texts in batches, writes
        hidden states (float16) and attention masks (uint8) as memmap files,
        then unloads the LLM.

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

        # First pass: find actual max tokenized length
        logger.info("Pass 1/2: Computing actual max tokenized length...")
        actual_max_len = 0
        for i in range(0, num_samples, batch_size * 4):
            batch_texts = texts[i:i + batch_size * 4]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                max_length=self._max_length,
                padding=False,
                return_length=True,
            )
            batch_max = max(encodings['length'])
            actual_max_len = max(actual_max_len, batch_max)

        logger.info(f"  Actual max tokenized length: {actual_max_len} "
                    f"(vs configured max_length={self._max_length})")

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

        # Create memmap files
        hs_path = self._cache_path / "hidden_states.npy"
        mask_path = self._cache_path / "attention_mask.npy"

        hs_mmap = np.lib.format.open_memmap(
            str(hs_path), mode='w+', dtype=np.float16,
            shape=(num_samples, actual_max_len, hidden_size)
        )
        mask_mmap = np.lib.format.open_memmap(
            str(mask_path), mode='w+', dtype=np.uint8,
            shape=(num_samples, actual_max_len)
        )

        # Second pass: compute hidden states
        logger.info("Pass 2/2: Computing hidden states...")
        processed = 0
        for i in range(0, num_samples, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_end = min(i + batch_size, num_samples)

            encoding = tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=actual_max_len,
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
                hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

            # Write to memmap (convert to float16)
            hs_mmap[i:batch_end] = hidden_states.cpu().to(torch.float16).numpy()
            mask_mmap[i:batch_end] = attention_mask.cpu().numpy().astype(np.uint8)

            processed += len(batch_texts)
            if processed % (batch_size * 10) == 0 or processed == num_samples:
                logger.info(f"  Processed {processed}/{num_samples} texts")

        # Flush to disk
        hs_mmap.flush()
        mask_mmap.flush()
        del hs_mmap, mask_mmap

        # Write metadata
        metadata = {
            'model_name': self._model_name,
            'max_length': self._max_length,
            'hidden_size': hidden_size,
            'num_samples': num_samples,
            'actual_max_len': actual_max_len,
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
        total_bytes = (
            num_samples * actual_max_len * hidden_size * 2  # float16
            + num_samples * actual_max_len  # uint8
        )
        logger.info(
            f"Hidden state cache created: {total_bytes / 1e9:.2f} GB, "
            f"shape=({num_samples}, {actual_max_len}, {hidden_size})"
        )

    def open(self) -> None:
        """Open memmap files for reading (lazy)."""
        if self._hs_mmap is not None:
            return

        if self._metadata is None:
            self._load_metadata()

        hs_path = self._cache_path / "hidden_states.npy"
        mask_path = self._cache_path / "attention_mask.npy"

        self._hs_mmap = np.load(str(hs_path), mmap_mode='r')
        self._mask_mmap = np.load(str(mask_path), mmap_mode='r')

        logger.info(f"Hidden state cache opened: shape={self._hs_mmap.shape}")

    def load_batch(
        self,
        indices: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load cached hidden states for a batch of indices.

        Args:
            indices: List of global cache indices.
            device: Device to place tensors on.

        Returns:
            Tuple of (hidden_states, attention_mask) tensors on device.
            hidden_states: (batch, actual_max_len, hidden_size) float32
            attention_mask: (batch, actual_max_len) float32
        """
        if self._hs_mmap is None:
            self.open()

        # Read from memmap (copies data from disk)
        idx_array = np.array(indices)
        hs = self._hs_mmap[idx_array]  # (batch, seq_len, hidden_size) float16
        mask = self._mask_mmap[idx_array]  # (batch, seq_len) uint8

        # Convert to tensors on device (float16 -> float32)
        hs_tensor = torch.from_numpy(hs.astype(np.float32)).to(device)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(device)

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
