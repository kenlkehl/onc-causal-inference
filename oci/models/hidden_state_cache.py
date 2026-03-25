# oci/models/hidden_state_cache.py
"""Pre-computed hidden state cache for FrozenLLMPoolerExtractor.

When the LLM is frozen, its hidden states are deterministic for a given input.
This module pre-computes hidden states once for the entire dataset and caches
them to disk as numpy memmap files. During training, only the lightweight
trainable layers (~200K params) need to be loaded to GPU.

Cache is keyed on (model_name, max_length, dataset_path) only, so it is
reusable across experiments with different causal heads, learning rates,
fold counts, etc.

Storage format (variable-length, no padding waste):
    {dataset_dir}/.oci_cache/flp_hidden_states_{hash}/
        hidden_states.npy   - float16 memmap (total_tokens, hidden_size)
        offsets.npy          - int64 array (N+1,) sample boundaries
        metadata.json        - cache metadata with storage_format="variable_length"

Each sample's hidden states span [offsets[i], offsets[i+1]) in the flat array.
Per-batch padding to the batch-local max length happens during collation,
avoiding the global-max padding waste of the old format.
"""

import concurrent.futures
import gc
import hashlib
import json
import logging
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .gpu_hidden_state_store import _get_hidden_size, _get_model_dtype, _make_downprojection

logger = logging.getLogger(__name__)


def _sanitize_hidden_states(hidden_states: torch.Tensor, context: str = "") -> torch.Tensor:
    """Replace NaN/Inf values in hidden states with 0.

    Some models (e.g., MedGemma) produce NaN or Inf in float16 hidden states
    when activation magnitudes exceed the float16 range (~65504).
    """
    nan_mask = torch.isnan(hidden_states)
    inf_mask = torch.isinf(hidden_states)
    bad_mask = nan_mask | inf_mask
    if bad_mask.any():
        n_bad = bad_mask.sum().item()
        total = hidden_states.numel()
        logger.warning(
            f"Hidden states contain {n_bad}/{total} NaN/Inf values "
            f"({n_bad/total:.4%}){' [' + context + ']' if context else ''}. "
            f"Replacing with 0."
        )
        hidden_states = hidden_states.clone()
        hidden_states[bad_mask] = 0.0
    return hidden_states


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
        random_projection_dim: If set, apply a random linear projection to reduce
            hidden state dimensionality before caching. Uses a deterministic random
            matrix seeded on (model_name, hidden_size, projection_dim) for
            reproducibility. Reduces cache size proportionally.
        downprojection_dim: If set, apply a frozen nn.Linear downprojection during
            precomputation. Mutually exclusive with random_projection_dim.
    """

    def __init__(
        self,
        cache_dir: str,
        model_name: str,
        max_length: int,
        dataset_path: str,
        random_projection_dim: Optional[int] = None,
        downprojection_dim: Optional[int] = None,
    ):
        if random_projection_dim is not None and downprojection_dim is not None:
            raise ValueError(
                "Cannot use both random_projection_dim and downprojection_dim. "
                "Use one or the other for dimensionality reduction."
            )
        self._cache_dir = Path(cache_dir)
        self._model_name = model_name
        self._max_length = max_length
        self._dataset_path = dataset_path
        self._random_projection_dim = random_projection_dim
        self._downprojection_dim = downprojection_dim
        self._cache_hash = self.compute_cache_hash(
            model_name, max_length, dataset_path, random_projection_dim,
            downprojection_dim,
        )

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
    def compute_cache_hash(
        model_name: str,
        max_length: int,
        dataset_path: str,
        random_projection_dim: Optional[int] = None,
        downprojection_dim: Optional[int] = None,
    ) -> str:
        """Compute deterministic hash for cache identification."""
        key = f"{model_name}|{max_length}|{os.path.abspath(dataset_path)}"
        if random_projection_dim is not None:
            key += f"|rp{random_projection_dim}"
        if downprojection_dim is not None:
            key += f"|dp{downprojection_dim}"
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

            # Check random projection dim matches
            cached_rp = self._metadata.get('random_projection_dim')
            if cached_rp != self._random_projection_dim:
                logger.warning(
                    f"Cache random_projection_dim mismatch: "
                    f"expected {self._random_projection_dim}, got {cached_rp}"
                )
                return False

            # Check downprojection dim matches
            cached_dp = self._metadata.get('downprojection_dim')
            if cached_dp != self._downprojection_dim:
                logger.warning(
                    f"Cache downprojection_dim mismatch: "
                    f"expected {self._downprojection_dim}, got {cached_dp}"
                )
                return False

            logger.info(
                f"Valid hidden state cache found: {expected_num_samples} samples, "
                f"total_tokens={expected_total}, hash={self._cache_hash}"
            )
            return True

        except Exception as e:
            logger.warning(f"Cache validation failed: {e}")
            return False

    @staticmethod
    def _make_random_projection(
        hidden_size: int,
        projection_dim: int,
        model_name: str,
    ) -> np.ndarray:
        """Generate a deterministic random projection matrix.

        Uses a seed derived from (model_name, hidden_size, projection_dim) so
        the same matrix is always produced for the same configuration.

        Returns:
            float32 numpy array of shape (hidden_size, projection_dim), scaled
            by 1/sqrt(projection_dim) for variance preservation.
        """
        seed_str = f"rp|{model_name}|{hidden_size}|{projection_dim}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        W = rng.randn(hidden_size, projection_dim).astype(np.float32)
        W *= 1.0 / np.sqrt(projection_dim)
        return W

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
            padding_side="right",
            truncation_side="left",  # Keep end of long documents
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
        hidden_size = _get_hidden_size(hf_config)
        compute_dtype = _get_model_dtype(hf_config)
        logger.info(f"  Compute dtype: {compute_dtype}")

        # Load model directly to target device.
        # Using device_map={"": device} avoids meta tensors that break .to()
        # for models with tied weights (e.g. Qwen3.5).
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name, config=hf_config, trust_remote_code=True,
            torch_dtype=compute_dtype,
            device_map={"": device},
        )
        # Remove accelerate dispatch hooks so the model behaves like a normal nn.Module
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)

        if tokenizer.pad_token == '[PAD]':
            model.resize_token_embeddings(len(tokenizer))

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Random projection setup
        rp_matrix = None
        rp_matrix_gpu = None
        store_dim = hidden_size
        if self._random_projection_dim is not None and self._random_projection_dim < hidden_size:
            rp_matrix = self._make_random_projection(
                hidden_size, self._random_projection_dim, self._model_name
            )
            rp_matrix_gpu = torch.from_numpy(rp_matrix).to(device)
            store_dim = self._random_projection_dim
            logger.info(
                f"  Random projection: {hidden_size} -> {store_dim} "
                f"({store_dim / hidden_size:.1%} of original)"
            )

        # Frozen downprojection setup (mutually exclusive with random projection)
        downproj_layer = None
        if self._downprojection_dim is not None and self._downprojection_dim < hidden_size:
            downproj_layer = _make_downprojection(
                hidden_size, self._downprojection_dim, self._model_name
            ).float().to(device)
            store_dim = self._downprojection_dim
            logger.info(
                f"  Downprojection: {hidden_size} -> {store_dim} "
                f"(frozen, {store_dim / hidden_size:.1%} of original)"
            )

        # Create cache directory
        self._cache_path.mkdir(parents=True, exist_ok=True)

        # Create flat memmap (no padding)
        hs_path = self._cache_path / "hidden_states.npy"
        offsets_path = self._cache_path / "offsets.npy"

        hs_mmap = np.lib.format.open_memmap(
            str(hs_path), mode='w+', dtype=np.float16,
            shape=(total_tokens, store_dim)
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

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=compute_dtype):
                # Use base transformer (model.model) to skip lm_head logits
                # computation, which would allocate vocab_size * seq_len floats
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.last_hidden_state  # (batch, batch_max_len, hidden_size)

                # Sanitize NaN/Inf (safety net for models that still overflow)
                hidden_states = _sanitize_hidden_states(hidden_states, context="precompute")

                # Apply random projection on GPU before transfer
                if rp_matrix_gpu is not None:
                    hidden_states = hidden_states.float() @ rp_matrix_gpu

            # Apply frozen downprojection on GPU before transfer
            if downproj_layer is not None:
                with torch.no_grad():
                    hidden_states = downproj_layer(hidden_states.float()).half()

            # Write only real (non-padding) tokens to flat memmap
            # Always store as float16 for cache efficiency
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
        del hs_mmap, rp_matrix_gpu, downproj_layer

        # Write metadata
        metadata = {
            'model_name': self._model_name,
            'max_length': self._max_length,
            'hidden_size': store_dim,
            'original_hidden_size': hidden_size,
            'random_projection_dim': self._random_projection_dim,
            'downprojection_dim': self._downprojection_dim,
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
        total_bytes = total_tokens * store_dim * 2  # float16
        original_bytes = total_tokens * hidden_size * 2
        padded_bytes = num_samples * actual_max_len * hidden_size * 2
        msg = f"Hidden state cache created: {total_bytes / 1e9:.2f} GB"
        if store_dim != hidden_size:
            msg += f" (vs {original_bytes / 1e9:.2f} GB without projection)"
        msg += (
            f", total_tokens={total_tokens:,}, savings={savings_pct:.1%}"
        )
        logger.info(msg)

    def precompute_multi_gpu(
        self,
        texts: List[str],
        devices: List[torch.device],
        batch_size: int = 4,
    ) -> None:
        """Pre-compute LLM hidden states in parallel across multiple GPUs.

        Splits the dataset into contiguous shards (one per device), loads a
        separate LLM copy on each GPU, and writes results to a shared memmap.
        Each thread writes to non-overlapping offsets so no locking is needed.

        Falls back to single-GPU ``precompute()`` when only one device is given.

        Args:
            texts: All texts in the dataset (in order).
            devices: List of GPU devices to use.
            batch_size: Batch size for LLM inference per GPU.
        """
        if len(devices) <= 1:
            device = devices[0] if devices else torch.device('cuda:0')
            return self.precompute(texts, device, batch_size)

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        num_samples = len(texts)
        num_devices = len(devices)
        logger.info(f"Pre-computing hidden states for {num_samples} texts "
                     f"across {num_devices} GPUs...")
        logger.info(f"  Model: {self._model_name}")
        logger.info(f"  Max length: {self._max_length}")
        logger.info(f"  Devices: {[str(d) for d in devices]}")
        logger.info(f"  Cache path: {self._cache_path}")

        # --- Pass 1: tokenize to get sequence lengths (CPU, single-threaded) ---
        logger.info("Pass 1/2: Computing per-sample tokenized lengths...")
        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            padding_side="right",
            truncation_side="left",
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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

        # Compute offsets
        offsets = np.zeros(num_samples + 1, dtype=np.int64)
        for i, length in enumerate(sequence_lengths):
            offsets[i + 1] = offsets[i] + length

        # Get hidden size and compute dtype from config (no model load yet)
        hf_config = AutoConfig.from_pretrained(self._model_name, trust_remote_code=True)
        hidden_size = _get_hidden_size(hf_config)
        compute_dtype = _get_model_dtype(hf_config)
        logger.info(f"  Compute dtype: {compute_dtype}")
        needs_resize = tokenizer.pad_token == '[PAD]'
        vocab_size = len(tokenizer)

        # Random projection setup
        rp_matrix = None
        store_dim = hidden_size
        if self._random_projection_dim is not None and self._random_projection_dim < hidden_size:
            rp_matrix = self._make_random_projection(
                hidden_size, self._random_projection_dim, self._model_name
            )
            store_dim = self._random_projection_dim
            logger.info(
                f"  Random projection: {hidden_size} -> {store_dim} "
                f"({store_dim / hidden_size:.1%} of original)"
            )

        # Frozen downprojection flag (each thread creates its own layer)
        use_downprojection = (
            self._downprojection_dim is not None
            and self._downprojection_dim < hidden_size
        )
        if use_downprojection:
            store_dim = self._downprojection_dim
            logger.info(
                f"  Downprojection: {hidden_size} -> {store_dim} "
                f"(frozen, {store_dim / hidden_size:.1%} of original)"
            )

        # --- Create memmap and save offsets ---
        self._cache_path.mkdir(parents=True, exist_ok=True)
        hs_path = self._cache_path / "hidden_states.npy"
        offsets_path = self._cache_path / "offsets.npy"

        hs_mmap = np.lib.format.open_memmap(
            str(hs_path), mode='w+', dtype=np.float16,
            shape=(total_tokens, store_dim)
        )
        np.save(str(offsets_path), offsets)

        # --- Shard texts across devices ---
        shard_size = (num_samples + num_devices - 1) // num_devices
        shards = []
        for d_idx in range(num_devices):
            start = d_idx * shard_size
            end = min(start + shard_size, num_samples)
            if start >= num_samples:
                break
            shards.append((devices[d_idx], start, end))

        logger.info(f"Pass 2/2: Computing hidden states across {len(shards)} shards...")
        for dev, s, e in shards:
            logger.info(f"  {dev}: samples [{s}, {e}) ({e - s} samples)")

        progress_lock = threading.Lock()
        progress = [0]

        def _compute_shard(device, shard_start, shard_end):
            """Load LLM on device, process shard, write to shared memmap."""
            shard_texts = texts[shard_start:shard_end]
            shard_lengths = sequence_lengths[shard_start:shard_end]
            n_shard = len(shard_texts)

            # Each thread loads its own tokenizer and model
            tok = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=True,
                padding_side="right",
                truncation_side="left",
            )
            if tok.pad_token is None:
                if tok.eos_token is not None:
                    tok.pad_token = tok.eos_token
                    tok.pad_token_id = tok.eos_token_id
                else:
                    tok.add_special_tokens({'pad_token': '[PAD]'})

            # Load directly to target device to avoid meta tensors with
            # tied-weight models (e.g. Qwen3.5).
            mdl = AutoModelForCausalLM.from_pretrained(
                self._model_name, config=hf_config, trust_remote_code=True,
                torch_dtype=compute_dtype,
                device_map={"": device},
            )
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(mdl, recurse=True)
            if needs_resize:
                mdl.resize_token_embeddings(vocab_size)
            mdl.eval()
            for param in mdl.parameters():
                param.requires_grad = False

            # Per-device random projection matrix on GPU
            rp_gpu = None
            if rp_matrix is not None:
                rp_gpu = torch.from_numpy(rp_matrix).to(device)

            # Per-device frozen downprojection layer
            dp_layer = None
            if use_downprojection:
                dp_layer = _make_downprojection(
                    hidden_size, self._downprojection_dim, self._model_name
                ).float().to(device)

            for i in range(0, n_shard, batch_size):
                batch_texts = shard_texts[i:i + batch_size]
                batch_end = min(i + batch_size, n_shard)
                batch_lengths = shard_lengths[i:batch_end]
                batch_max_len = max(batch_lengths)

                encoding = tok(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=batch_max_len,
                    return_tensors="pt",
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=compute_dtype):
                    # Use base transformer (mdl.model) to skip lm_head logits
                    outputs = mdl.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states = outputs.last_hidden_state

                    # Sanitize NaN/Inf (some models overflow in float16)
                    hidden_states = _sanitize_hidden_states(
                        hidden_states, context=f"precompute_multi_gpu/{device}"
                    )

                    # Apply random projection on GPU before transfer
                    if rp_gpu is not None:
                        hidden_states = hidden_states.float() @ rp_gpu

                # Apply frozen downprojection on GPU before transfer
                if dp_layer is not None:
                    with torch.no_grad():
                        hidden_states = dp_layer(hidden_states.float()).half()

                hs_cpu = hidden_states.cpu().to(torch.float16).numpy()
                global_start = shard_start + i
                for j in range(len(batch_texts)):
                    sample_len = batch_lengths[j]
                    sample_offset = int(offsets[global_start + j])
                    hs_mmap[sample_offset:sample_offset + sample_len] = hs_cpu[j, :sample_len]

                with progress_lock:
                    progress[0] += len(batch_texts)
                    if progress[0] % (batch_size * 10) == 0 or progress[0] == num_samples:
                        logger.info(f"  Processed {progress[0]}/{num_samples} texts "
                                     f"(total across all GPUs, reported by {device})")

            # Unload model from this GPU
            del mdl, tok, rp_gpu, dp_layer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- Launch threads ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(shards)) as executor:
            futures = [
                executor.submit(_compute_shard, dev, s, e)
                for dev, s, e in shards
            ]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()  # raise any exceptions

        # Flush memmap
        hs_mmap.flush()
        del hs_mmap

        # Write metadata
        metadata = {
            'model_name': self._model_name,
            'max_length': self._max_length,
            'hidden_size': store_dim,
            'original_hidden_size': hidden_size,
            'random_projection_dim': self._random_projection_dim,
            'downprojection_dim': self._downprojection_dim,
            'num_samples': num_samples,
            'actual_max_len': actual_max_len,
            'total_tokens': total_tokens,
            'sequence_lengths': sequence_lengths,
            'storage_format': 'variable_length',
            'dataset_path': os.path.abspath(self._dataset_path),
            'cache_hash': self._cache_hash,
            'created_at': datetime.now().isoformat(),
            'dtype': 'float16',
            'num_gpus_used': len(shards),
        }
        with open(self._cache_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        self._metadata = metadata

        total_bytes = total_tokens * store_dim * 2
        original_bytes = total_tokens * hidden_size * 2
        padded_bytes = num_samples * actual_max_len * hidden_size * 2
        msg = (
            f"Hidden state cache created ({len(shards)} GPUs): {total_bytes / 1e9:.2f} GB"
        )
        if store_dim != hidden_size:
            msg += f" (vs {original_bytes / 1e9:.2f} GB without projection)"
        msg += f", total_tokens={total_tokens:,}, savings={savings_pct:.1%}"
        logger.info(msg)

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
