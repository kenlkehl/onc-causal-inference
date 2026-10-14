"""Resumable final-fit predictions and independent penalty-CV paths.

Only JSON predictions and optimizer diagnostics are cached, never executable
model pickles. The caller supplies the complete estimation input identity.
"""

from functools import partial

import numpy as np

from . import stage2_multi_model_selection as numerical


def _prediction(compute, rows):
    values = np.asarray(compute(), dtype=float)
    if values.shape != (rows,) or not np.isfinite(values).all():
        raise ValueError("final nuisance predictions must be finite and match validation rows")
    return {"prediction": values.tolist()}


def prediction_future(directory, name, fingerprint, compute, rows):
    """Submit one independent fit, or read its verified completed prediction."""
    if directory is not None and fingerprint == "uncached":
        raise ValueError("final modeling checkpoints require an input fingerprint")
    return numerical.checkpoint_future(
        directory, name, fingerprint, partial(_prediction, compute, rows)
    )


def prediction_result(future, rows):
    values = np.asarray(future.result()["prediction"], dtype=float)
    if values.shape != (rows,) or not np.isfinite(values).all():
        raise ValueError("cached final nuisance prediction has invalid shape or values")
    return values


def penalty_path_submitter(directory, fingerprint):
    """Each path keeps its original penalty order and warm starts in one worker."""

    def submit(index, compute):
        return numerical.checkpoint_future(
            directory, f"interaction_cv/fold_{index + 1:03d}", fingerprint, compute
        )

    return submit
