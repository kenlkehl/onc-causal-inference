"""ColBERT token encoder using Transformers and a checkpoint's trained projection.

Supports the Transformer + identity Dense layout published by PyLate, without
depending on PyLate or any other clinical application. No remote code executes.
"""

import hashlib
import json
from pathlib import Path


def resolve_checkpoint(config):
    import transformers

    folder = Path(config.model_name).expanduser()
    local = folder.is_dir()
    if not local:
        from huggingface_hub import snapshot_download

        folder = Path(
            snapshot_download(
                config.model_name,
                revision=config.revision,
                allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Dense/*"],
            )
        )
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".txt", ".safetensors"}:
            digest.update(str(path.relative_to(folder)).encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return folder, {
        "revision": "local" if local else folder.name,
        "artifact_sha256": digest.hexdigest(),
        "transformers_version": transformers.__version__,
    }


class ColBERTEncoder:
    def __init__(self, config, device, folder):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        self.config, self.device = config, device
        modules = json.loads((folder / "modules.json").read_text())
        if (
            len(modules) != 2
            or modules[0].get("path") != ""
            or modules[0].get("type") != "sentence_transformers.models.Transformer"
            or modules[1].get("path") != "1_Dense"
            or modules[1].get("type")
            not in {"pylate.models.Dense.Dense", "sentence_transformers.models.Dense"}
        ):
            raise ValueError(
                "ColBERT requires a Transformer + trained 1_Dense projection checkpoint"
            )
        self.settings = json.loads((folder / "config_sentence_transformers.json").read_text())
        dense = json.loads((folder / "1_Dense/config.json").read_text())
        if dense.get("activation_function") != "torch.nn.modules.linear.Identity":
            raise ValueError("Unsupported ColBERT projection activation")
        if self.settings.get("prompts") or self.settings.get("default_prompt_name"):
            raise ValueError("Custom ColBERT checkpoint prompts are unsupported")
        self.tokenizer = AutoTokenizer.from_pretrained(
            folder, use_fast=True, local_files_only=True, trust_remote_code=False
        )
        if not self.tokenizer.is_fast or self.tokenizer.padding_side != "right":
            raise ValueError("ColBERT needs a fast tokenizer with right padding")
        self.model = (
            AutoModel.from_pretrained(
                folder, local_files_only=True, use_safetensors=True, trust_remote_code=False
            )
            .to(device)
            .eval()
        )
        self.projection = torch.nn.Linear(
            dense["in_features"], dense["out_features"], bias=dense["bias"]
        )
        self.projection.load_state_dict(
            {
                k.removeprefix("linear."): v
                for k, v in load_file(folder / "1_Dense/model.safetensors").items()
            },
            strict=True,
        )
        self.projection.to(device=device, dtype=next(self.model.parameters()).dtype).eval()
        if (
            max(config.query_length, config.chunk_size + 8)
            > self.model.config.max_position_embeddings
        ):
            raise ValueError(
                "ColBERT query/chunk length exceeds the checkpoint's position capacity"
            )
        self.markers = {}
        for kind in ("query", "document"):
            token = self.settings[f"{kind}_prefix"]
            if token and token not in self.tokenizer.get_vocab():
                raise ValueError("Missing trained ColBERT marker token")
            self.markers[kind] = self.tokenizer.convert_tokens_to_ids(token) if token else None
        for token in (
            self.tokenizer.mask_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        ):
            if token is not None:
                self.tokenizer.pad_token_id = token
                break
        self.skip_ids = [
            self.tokenizer.convert_tokens_to_ids(w) for w in self.settings["skiplist_words"]
        ]

    def encode(self, texts, *, query=False):
        import torch

        if not texts:
            return []
        marker = self.markers["query" if query else "document"]
        limit = self.config.query_length if query else self.config.chunk_size + 8
        expansion = query and self.settings["do_query_expansion"]
        inputs = self.tokenizer(
            [t.strip() for t in texts],
            return_tensors="pt",
            truncation=False,
            padding="max_length" if expansion else True,
            **({"max_length": limit - int(marker is not None)} if expansion else {}),
        )
        if marker is not None:
            for key, tensor in list(inputs.items()):
                fill = marker if key == "input_ids" else int(key == "attention_mask")
                inputs[key] = torch.cat(
                    (
                        tensor[:, :1],
                        torch.full((len(texts), 1), fill, dtype=tensor.dtype),
                        tensor[:, 1:],
                    ),
                    dim=1,
                )
        if inputs["input_ids"].shape[1] > limit:
            raise ValueError("ColBERT input exceeds configured capacity; no text was truncated")
        if expansion and self.settings["attend_to_expansion_tokens"]:
            inputs["attention_mask"].fill_(1)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            vectors = self.projection(self.model(**inputs).last_hidden_state)
            keep = inputs["attention_mask"].bool()
            if expansion:
                keep = torch.ones_like(keep)
            elif not query:
                for token in self.skip_ids:
                    keep &= inputs["input_ids"] != token
            return [
                torch.nn.functional.normalize(v[m], dim=-1).float().cpu().numpy()
                for v, m in zip(vectors, keep, strict=True)
            ]
