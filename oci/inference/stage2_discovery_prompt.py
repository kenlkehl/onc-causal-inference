"""Standard atomic clinical-variable discovery instructions."""

from .stage2_prompt_catalog import PROMPT_VERSION, SYSTEM_PROMPTS

DISCOVERY_PROMPT_VERSION = PROMPT_VERSION + "_atomic_discovery"
DISCOVERY_SYSTEM_PROMPT = SYSTEM_PROMPTS["01_discover"]
