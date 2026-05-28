#!/usr/bin/env python
"""Expanded oracle runner for agentic explicit-feature forests only.

This is a constrained version of run_oracle_experiments.py. It always runs
only the agentic_explicit_feature_forest method, but uses a broader default
grid and more repeats:

    base conditions per dataset:
      agentic_iterations: 1, 2, 3
      initial_feature_counts: 0, 2, 5
      initial_feature_strategies: true_first, modifiers_first, mixed, distractors

The count=0 condition uses the "none" strategy internally, so this produces
27 base conditions per dataset. With the default 30 repeats, that is 810
experiments per dataset before --max-experiments or --resume filtering.

The default LLM budgets are intentionally large:
    agent proposal context chars: 200000
    agent proposal generation tokens: 50000
    extraction text chars: 200000
    extraction generation tokens: 50000

For OpenAI-compatible server mode, the serving process must also be launched
with a context length large enough for these requests.
"""

import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_oracle_experiments as oracle_runner


DEFAULT_ARGS: Dict[str, List[str]] = {
    "--output-dir": ["../pcori_experiments/oracle_agentic_explicit_forest_expanded"],
    "--n-repeats": ["30"],
    "--agentic-iterations": ["1", "2", "3"],
    "--agentic-initial-feature-counts": ["0", "2", "5"],
    "--agentic-initial-feature-strategies": [
        "true_first",
        "modifiers_first",
        "mixed",
        "distractors",
    ],
    "--agentic-inner-folds": ["3"],
    "--agentic-max-additions-per-iter": ["6"],
    "--agentic-max-removals-per-iter": ["2"],
    "--agentic-agent-model-name": ["nvidia/Gemma-4-31B-IT-NVFP4"],
    "--agentic-vllm-model-name": ["nvidia/Gemma-4-31B-IT-NVFP4"],
    "--agentic-agent-max-tokens": ["50000"],
    "--agentic-agent-context-chars": ["200000"],
    "--agentic-agent-context-examples": ["20"],
    "--agentic-vllm-max-model-len": ["200000"],
    "--agentic-extraction-max-tokens": ["50000"],
    "--agentic-extraction-max-text-length": ["200000"],
    "--agentic-extraction-max-retries": ["5"],
    "--agentic-extraction-batch-size": ["16"],
}

FORCED_AGENTIC_ONLY_ARGS = [
    "--model-types",
    "agentic_explicit_feature_forest",
    "--filter-extractor-types",
    "agentic_explicit_features",
]


def _option_present(args: List[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def _with_expanded_agentic_defaults(argv: List[str]) -> List[str]:
    args = list(argv[1:])
    expanded_args = list(args)

    for option, values in DEFAULT_ARGS.items():
        if not _option_present(args, option):
            expanded_args.extend([option, *values])

    # Keep this last so this runner cannot accidentally launch the full oracle
    # model grid if a caller passes --model-types or --filter-extractor-types.
    expanded_args.extend(FORCED_AGENTIC_ONLY_ARGS)
    return [argv[0], *expanded_args]


def main() -> None:
    sys.argv = _with_expanded_agentic_defaults(sys.argv)
    oracle_runner.main()


if __name__ == "__main__":
    main()
