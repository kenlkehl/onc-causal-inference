# oci/models/gpu_hidden_state_store.py
"""GPU-resident hidden state store for frozen LLM extractors.

When the LLM is frozen, hidden states are deterministic for a given input.
This module pre-computes hidden states once for the entire dataset and keeps
them in GPU VRAM as a flat float16 tensor.  During training only the
lightweight trainable layers (~200K params) are loaded, and batch access
is a pure GPU gather+pad with zero CPU-GPU transfer.

Storage layout (variable-length, no padding waste):
    _flat_tensor  : (total_tokens, hidden_size) float16 on GPU
    _offsets      : (N+1,) int64 on CPU  -- sample boundaries

Each sample's hidden states span [offsets[i], offsets[i+1]) in the flat
tensor.  Per-batch padding to the batch-local max length happens during
load_batch().
"""

import gc
import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _get_hidden_size(config) -> int:
    """Resolve hidden_size from an HF config, handling multimodal models.

    Multimodal models like Qwen3.5 nest the text config under a
    ``text_config`` attribute instead of exposing ``hidden_size`` at the
    top level.
    """
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise AttributeError(
        f"Cannot determine hidden_size from config type {type(config).__name__}. "
        f"Neither config.hidden_size nor config.text_config.hidden_size exists."
    )


def _make_downprojection(
    hidden_size: int,
    downprojection_dim: int,
    model_name: str,
) -> torch.nn.Linear:
    """Create a deterministic frozen downprojection layer.

    Uses Kaiming uniform init with a deterministic seed derived from
    (model_name, hidden_size, downprojection_dim) for reproducibility.
    """
    seed_str = f"downproj|{model_name}|{hidden_size}|{downprojection_dim}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)

    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    layer = torch.nn.Linear(hidden_size, downprojection_dim, bias=False)

    torch.random.set_rng_state(rng_state)

    for p in layer.parameters():
        p.requires_grad = False
    layer.eval()

    return layer


class GPUHiddenStateStore:
    """GPU-resident hidden state store for frozen LLM extractors.

    Pre-computes LLM hidden states once for the entire dataset, stores them
    as a flat float16 tensor on GPU, and provides zero-copy batch access.

    Usage::

        store = GPUHiddenStateStore()
        store.precompute(texts, "Qwen/Qwen3-0.6B-Base", 8192, device, batch_size=4)
        # LLM is now unloaded; only the flat tensor remains on GPU.

        hs, mask = store.load_batch([0, 5, 12])
        # hs: (3, batch_max_len, hidden_size) float32 on device
        # mask: (3, batch_max_len) float32 on device

        store.free()  # release GPU memory
    """

    def __init__(self):
        self._flat_tensor: Optional[torch.Tensor] = None   # (total_tokens, hidden_size) fp16
        self._offsets: Optional[np.ndarray] = None          # (N+1,) int64
        self._hidden_size: int = 0
        self._device: Optional[torch.device] = None
        self._num_samples: int = 0

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_vram_gb(
        texts: List[str],
        model_name: str,
        max_length: int,
        downprojection_dim: Optional[int] = None,
    ) -> float:
        """Estimate VRAM needed without loading the model.

        Tokenizes all texts to get actual sequence lengths, then computes
        the float16 storage requirement.

        Args:
            texts: All texts in the dataset.
            model_name: HuggingFace model name.
            max_length: Maximum sequence length for tokenization.
            downprojection_dim: If set, use this instead of hidden_size
                for storage estimation (frozen downprojection applied
                during precomputation).

        Returns:
            Estimated VRAM in GB for the hidden state tensor.
        """
        from transformers import AutoConfig, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="right",
            truncation_side="left",  # Keep end of long documents
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = _get_hidden_size(config)

        store_dim = hidden_size
        if downprojection_dim is not None and downprojection_dim < hidden_size:
            store_dim = downprojection_dim

        # Compute total tokens
        total_tokens = 0
        batch_size = 512
        for i in range(0, len(texts), batch_size):
            encodings = tokenizer(
                texts[i : i + batch_size],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_length=True,
            )
            total_tokens += sum(encodings["length"])

        # float16: 2 bytes per element
        return total_tokens * store_dim * 2 / 1e9

    # ------------------------------------------------------------------
    # Precompute
    # ------------------------------------------------------------------

    def precompute(
        self,
        texts: List[str],
        model_name: str,
        max_length: int,
        device: torch.device,
        batch_size: int = 4,
        downprojection_dim: Optional[int] = None,
    ) -> None:
        """Pre-compute LLM hidden states for all texts and store on GPU.

        Loads the LLM, runs a batched forward pass, stores the results as a
        flat float16 tensor on ``device``, and then unloads the LLM to free
        VRAM for training.

        When ``downprojection_dim`` is set, a frozen linear projection
        (hidden_size -> downprojection_dim) is applied during precomputation
        so the cached tensor is smaller. The downprojection weights are
        deterministically initialized from the model name for reproducibility.

        Args:
            texts: All texts in the dataset (in order).
            model_name: HuggingFace model name.
            max_length: Maximum sequence length for tokenization.
            device: GPU device to store results on.
            batch_size: Batch size for LLM inference.
            downprojection_dim: If set, apply a frozen linear projection to
                reduce hidden states from hidden_size to this dimension.
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        num_samples = len(texts)
        self._device = device
        self._num_samples = num_samples

        logger.info(f"GPUHiddenStateStore: pre-computing hidden states for {num_samples} texts")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Max length: {max_length}")
        logger.info(f"  Device: {device}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="right",
            truncation_side="left",  # Keep end of long documents
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # First pass: compute per-sample tokenized lengths
        logger.info("  Pass 1/2: Computing per-sample tokenized lengths...")
        sequence_lengths: List[int] = []
        for i in range(0, num_samples, batch_size * 4):
            batch_texts = texts[i : i + batch_size * 4]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_length=True,
            )
            sequence_lengths.extend(encodings["length"])

        total_tokens = sum(sequence_lengths)
        actual_max_len = max(sequence_lengths)
        mean_len = total_tokens / num_samples

        logger.info(
            f"  Total tokens: {total_tokens:,} across {num_samples} samples "
            f"(mean {mean_len:.0f}, max {actual_max_len})"
        )

        # Compute offsets
        offsets = np.zeros(num_samples + 1, dtype=np.int64)
        for i, length in enumerate(sequence_lengths):
            offsets[i + 1] = offsets[i] + length
        self._offsets = offsets

        # Load model
        logger.info("  Loading LLM for hidden state extraction...")
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = _get_hidden_size(hf_config)
        self._hidden_size = hidden_size

        # Load directly to target device to avoid meta tensors with
        # tied-weight models (e.g. Qwen3.5).
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=hf_config,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map={"": device},
        )
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)

        if tokenizer.pad_token == "[PAD]":
            model.resize_token_embeddings(len(tokenizer))

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Frozen downprojection (applied during precomputation)
        downproj_layer = None
        store_dim = hidden_size
        if downprojection_dim is not None and downprojection_dim < hidden_size:
            downproj_layer = _make_downprojection(
                hidden_size, downprojection_dim, model_name
            ).half().to(device)
            store_dim = downprojection_dim
            logger.info(
                f"  Downprojection: {hidden_size} -> {store_dim} "
                f"(frozen, {store_dim / hidden_size:.1%} of original)"
            )
        self._hidden_size = store_dim

        # Allocate flat GPU tensor
        estimated_gb = total_tokens * store_dim * 2 / 1e9
        logger.info(
            f"  Allocating GPU tensor: ({total_tokens:,}, {store_dim}) "
            f"float16 ≈ {estimated_gb:.2f} GB"
        )
        flat_tensor = torch.zeros(
            total_tokens, store_dim, dtype=torch.float16, device=device
        )

        # Second pass: compute hidden states and write to flat tensor
        logger.info("  Pass 2/2: Computing hidden states...")
        processed = 0
        for i in range(0, num_samples, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_end = min(i + batch_size, num_samples)
            batch_lengths = sequence_lengths[i:batch_end]
            batch_max_len = max(batch_lengths)

            encoding = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=batch_max_len,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden)

            # Sanitize NaN/Inf (some models overflow in float16)
            from .hidden_state_cache import _sanitize_hidden_states
            hidden_states = _sanitize_hidden_states(hidden_states, context="gpu_store")

            # Apply frozen downprojection before storing
            if downproj_layer is not None:
                with torch.no_grad():
                    hidden_states = downproj_layer(hidden_states.float()).half()

            # Write only real (non-padding) tokens to flat tensor
            hs_fp16 = hidden_states.to(torch.float16)
            for j in range(len(batch_texts)):
                sample_len = batch_lengths[j]
                sample_offset = int(offsets[i + j])
                flat_tensor[sample_offset : sample_offset + sample_len] = hs_fp16[
                    j, :sample_len
                ]

            processed += len(batch_texts)
            if processed % (batch_size * 10) == 0 or processed == num_samples:
                logger.info(f"    Processed {processed}/{num_samples} texts")

        self._flat_tensor = flat_tensor

        # Unload LLM
        logger.info("  Unloading LLM from GPU...")
        del model, tokenizer, downproj_layer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            f"GPUHiddenStateStore ready: {num_samples} samples, "
            f"{total_tokens:,} tokens, {self.estimated_vram_gb:.2f} GB VRAM"
        )

    # ------------------------------------------------------------------
    # Batch access
    # ------------------------------------------------------------------

    def load_batch(
        self,
        indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load hidden states for a batch of indices from GPU.

        Gathers variable-length sequences and pads to the batch-local max.
        All operations are on GPU — no CPU-GPU transfer.

        Args:
            indices: List of global dataset indices.

        Returns:
            Tuple of (hidden_states, attention_mask) tensors on GPU.
            hidden_states: (batch, batch_max_len, hidden_size) float32
            attention_mask: (batch, batch_max_len) float32
        """
        if self._flat_tensor is None:
            raise RuntimeError("GPUHiddenStateStore has no data. Call precompute() first.")

        # Compute batch-local lengths and max
        lengths = []
        for idx in indices:
            start = int(self._offsets[idx])
            end = int(self._offsets[idx + 1])
            lengths.append(end - start)
        max_len = max(lengths)

        batch_size = len(indices)
        hidden_size = self._hidden_size
        device = self._flat_tensor.device

        # Allocate output tensors on GPU
        hs = torch.zeros(
            batch_size, max_len, hidden_size, dtype=torch.float16, device=device
        )
        mask = torch.zeros(batch_size, max_len, dtype=torch.float32, device=device)

        # Gather: copy each sample's tokens from flat tensor
        for i, idx in enumerate(indices):
            start = int(self._offsets[idx])
            length = lengths[i]
            hs[i, :length] = self._flat_tensor[start : start + length]
            mask[i, :length] = 1.0

        return hs.float(), mask

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        """Return hidden size of the stored representations."""
        return self._hidden_size

    @property
    def estimated_vram_gb(self) -> float:
        """Return current VRAM usage of stored hidden states in GB."""
        if self._flat_tensor is None:
            return 0.0
        return self._flat_tensor.nelement() * self._flat_tensor.element_size() / 1e9

    @property
    def num_samples(self) -> int:
        """Return number of samples stored."""
        return self._num_samples

    @property
    def device(self) -> Optional[torch.device]:
        """Return device the store is on."""
        return self._device

    # ------------------------------------------------------------------
    # Load from disk cache
    # ------------------------------------------------------------------

    def load_from_disk_cache(self, disk_cache, device: torch.device) -> None:
        """Load hidden states from a pre-computed HiddenStateCache into GPU.

        This avoids re-running the LLM when a disk cache is already available
        (e.g., from multi-GPU precomputation).

        Args:
            disk_cache: A HiddenStateCache that has been opened and optionally
                preloaded to RAM.
            device: GPU device to store the tensor on.
        """
        self._device = device

        # Access the underlying flat array and offsets
        hs_array = disk_cache.hidden_states_array
        self._offsets = hs_array.offsets.copy()
        self._num_samples = len(hs_array)
        self._hidden_size = hs_array.flat.shape[-1]

        total_tokens = int(self._offsets[-1])
        estimated_gb = total_tokens * self._hidden_size * 2 / 1e9
        logger.info(
            f"GPUHiddenStateStore: loading from disk cache "
            f"({self._num_samples} samples, {total_tokens:,} tokens, "
            f"~{estimated_gb:.2f} GB) to {device}"
        )

        # Transfer flat array to GPU as float16
        self._flat_tensor = torch.from_numpy(
            np.array(hs_array.flat, dtype=np.float16)
        ).to(device)

        logger.info(
            f"GPUHiddenStateStore ready: {self._num_samples} samples, "
            f"{total_tokens:,} tokens, {self.estimated_vram_gb:.2f} GB VRAM"
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def free(self) -> None:
        """Release GPU memory held by this store."""
        if self._flat_tensor is not None:
            vram_gb = self.estimated_vram_gb
            del self._flat_tensor
            self._flat_tensor = None
            self._offsets = None
            self._num_samples = 0
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"GPUHiddenStateStore freed ~{vram_gb:.2f} GB VRAM")
