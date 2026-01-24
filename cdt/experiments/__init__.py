# cdt/experiments/__init__.py

"""Experiment orchestration modules."""

from .runner import ExperimentRunner
from .matched_pair_runner import (
    MatchedPairExperimentRunner,
    run_matched_pair_experiment,
    run_matched_pair_from_config
)

__all__ = [
    'ExperimentRunner',
    'MatchedPairExperimentRunner',
    'run_matched_pair_experiment',
    'run_matched_pair_from_config',
]
