"""Resumable final-fit predictions and independent penalty-CV paths.

These checkpoints hold only JSON predictions and optimizer diagnostics. The
caller supplies the complete estimation input identity. (The inner-fold
nuisance models of the orthogonal modifier tests are stored separately with
joblib by ``plain_handoff_stage2_analysis`` and loaded only after their
fingerprint and file hash are verified.)
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
    """Each path keeps its original penalty order and warm starts in one worker.

    ``variant`` names a different penalty structure of the same interaction
    model (the selected-modifier penalty paths); its paths are checkpointed in
    ``interaction_cv_<variant>/`` so no structure reuses another's losses.
    """

    def submit(index, compute, variant=None):
        prefix = "interaction_cv" if variant is None else f"interaction_cv_{variant}"
        return numerical.checkpoint_future(
            directory, f"{prefix}/fold_{index + 1:03d}", fingerprint, compute
        )

    return submit
