"""Lightweight configuration for non-generative Stage 2 measurement."""

from dataclasses import asdict, dataclass, replace
import math
from collections.abc import Mapping


LETTERS = "ABCDEFGHIJKLMNOP"
PLUMB_REVISION = "24f7bf77e7ee258a2d158c61ea2dce2b60321010"


@dataclass(frozen=True)
class DecisionExtractionConfig:
    enabled: bool = False
    tokenizer_name: str = "crh225/plumb-4b"
    tokenizer_revision: str = PLUMB_REVISION
    temperature: float = 2.07
    max_prompt_tokens: int = 3000
    numeric_bins: int = 6
    numeric_passes: int = 3
    verification_relative_tolerance: float = 0.05
    verification_min_probability: float = 0.8
    none_above_fraction: float = 0.2
    none_above_min_patients: int = 3

    def validate(self):
        if type(self.enabled) is not bool:
            raise ValueError("decision_extraction.enabled must be boolean")
        if not isinstance(self.tokenizer_name, str) or not self.tokenizer_name.strip():
            raise ValueError("decision_extraction.tokenizer_name is required")
        if not isinstance(self.tokenizer_revision, str):
            raise ValueError("decision_extraction.tokenizer_revision must be a string")
        for name, lo, hi in (("max_prompt_tokens", 128, 3000), ("numeric_bins", 2, 6),
                             ("numeric_passes", 1, 8), ("none_above_min_patients", 1, 10**9)):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"decision_extraction.{name} must be an integer in [{lo}, {hi}]")
        for name, lo, hi in (("temperature", 0, 100), ("verification_relative_tolerance", 0, 1),
                             ("verification_min_probability", 0.5, 1), ("none_above_fraction", 0, 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"decision_extraction.{name} must be finite")
            if not lo <= value <= hi or (name == "temperature" and value == 0):
                raise ValueError(f"invalid decision_extraction.{name}")
            if name == "verification_relative_tolerance" and value == 1:
                raise ValueError("verification_relative_tolerance must be below one")

    def public_dict(self):
        return asdict(self)


def decision_config_from_mapping(raw=None):
    if isinstance(raw, DecisionExtractionConfig):
        config = raw
    else:
        if raw is not None and not isinstance(raw, Mapping):
            raise ValueError("stage2.decision_extraction must be an object")
        try:
            config = DecisionExtractionConfig(**dict(raw or {}))
        except TypeError as error:
            raise ValueError(f"invalid stage2.decision_extraction: {error}") from error
    config.validate()
    return config


def classifier_vllm_config(config):
    """Use next-token A–P weights as a pooling head, with no generation."""
    import json

    reserved = {"--runner", "--convert", "--hf-overrides", "--pooler-config"}
    if any(str(arg).split("=", 1)[0] in reserved for arg in config.extra_args):
        raise ValueError("decision extraction manages runner/convert/hf-overrides/pooler-config")
    return replace(config, extra_args=(*config.extra_args,
        "--runner", "pooling", "--convert", "classify",
        "--hf-overrides", json.dumps({"classifier_from_token": list(LETTERS),
            "method": "no_post_processing", "num_labels": len(LETTERS)}),
        "--pooler-config", json.dumps({"pooling_type": "LAST", "use_activation": False})))
