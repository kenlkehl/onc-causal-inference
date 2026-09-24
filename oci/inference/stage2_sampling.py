"""Publisher sampling recommendations, verified 2026-09-23.

Profiles are checked in for reproducible/offline runs, rather than scraping a
mutable model card during inference. Unrecognized models use server defaults.
Neutral penalties/filters fill unspecified controls for recognized profiles so
server generation_config.json files cannot silently add a different penalty.
"""

from __future__ import annotations

import re
from typing import Any

SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
)


def is_qwen_flash_next(model: str) -> bool:
    return bool(re.search(r"qwen[-_ ]?3[._]8[-_ ]flash[-_ ]next", str(model), re.I))


def default_reasoning(model: str, request_kind: str) -> str:
    if is_qwen_flash_next(model):
        return "xhigh"
    return "none" if request_kind == "extraction" else "high"


def sampling_provenance(model: str, family: str) -> dict[str, Any]:
    if is_qwen_flash_next(model):
        return {"profile": "qwen3.8_flash_next", "verified_on": "2026-09-23",
                "source": "https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices"}
    if family == "gemma4":
        return {"profile": "gemma4", "verified_on": "2026-09-23",
                "source": "https://huggingface.co/google/gemma-4-31B-it#best-practices"}
    return {"profile": family if family in {"qwen3", "lfm2.5"} else "server_defaults",
            "verified_on": "2026-09-12" if family in {"qwen3", "lfm2.5"} else None}


def recommended_sampling(model: str, family: str, thinking: bool) -> dict[str, Any]:
    compact = re.sub(r"[^a-z0-9]+", "", model.lower())
    defaults: dict[str, Any] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    if family == "gemma4":
        # https://huggingface.co/google/gemma-4-31B-it#best-practices
        defaults.update(top_p=0.95, top_k=64)
    elif family == "qwen3":
        # https://huggingface.co/Qwen/Qwen3-32B#best-practices
        # https://huggingface.co/Qwen/Qwen3.5-27B#best-practices
        # https://huggingface.co/Qwen/Qwen3.6-27B#best-practices
        # https://huggingface.co/Qwen/Qwen3.8-27B#best-practices
        # Keep the version separator: Qwen3-8B is not Qwen3.8.
        version_match = re.search(r"qwen[-_ ]?3[._]([568])(?:[-_/]|$)", model.lower())
        version = version_match.group(1) if version_match else ""
        modern = bool(version)
        defaults.update(
            temperature=(1.0 if modern else 0.6) if thinking else 0.7,
            top_p=0.95 if thinking else 0.8,
            top_k=20,
            presence_penalty=1.5
            if (version == "5" or (modern and not thinking))
            else 0.0,
        )
    elif family == "lfm2.5":
        # https://huggingface.co/LiquidAI/LFM2.5-2.6B
        # https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
        defaults.update(
            temperature=0.1,
            top_k=50,
            repetition_penalty=1.05 if "12b" in compact else 1.1,
        )
    else:
        return {}
    return defaults
