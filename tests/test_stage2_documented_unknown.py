"""The reserved documented_unknown measurement state in Stage 2."""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_documented_unknown as du
from oci.inference.stage2_extraction_failures import extraction_failure_column

UNKNOWN = du.DOCUMENTED_UNKNOWN


def _definition(name, value_type, categories=(), **extra):
    return {
        "feature_id": f"f_{name}",
        "name": name,
        "description": f"{name} status.",
        "value_type": value_type,
        "categories_or_unit": list(categories),
        "measurement_definition": f"Extract {name}.",
        "missing_value_rule": "Return null when undocumented.",
        **extra,
    }


DEFINITIONS = [
    _definition("egfr", "binary", ["Mutant", "Wild-type"]),
    _definition("histology", "categorical", ["Adenocarcinoma", "Squamous", "Other"]),
    _definition("ecog", "ordinal", ["0", "1", "2"]),
    _definition("albumin", "continuous", ["g/dL"]),
    _definition("free_text", "ambiguous"),
]


# ---------------------------------------------------------------------------
# Validators


# ---------------------------------------------------------------------------
# Mode resolution and harmonization


# ---------------------------------------------------------------------------
# Encoders


def _frame():
    return pd.DataFrame({
        "_oci_row_id": np.arange(8),
        "egfr": ["Mutant", "Wild-type", UNKNOWN, None, "Mutant", UNKNOWN, "Wild-type", None],
        "albumin": [3.0, 4.0, UNKNOWN, None, 3.5, UNKNOWN, 4.5, None],
        extraction_failure_column("albumin"): [0, 0, 0, 1, 0, 0, 0, 0],
    })


def test_final_encoder_adds_a_separate_grouped_indicator():
    egfr = DEFINITIONS[0]
    albumin = {**DEFINITIONS[3], "extraction_failure_indicator": True}
    frame = _frame()
    encoder = analysis._FeatureEncoder([albumin]).fit(frame)
    x = encoder.transform(frame)
    # value, missing, failure, documented_unknown
    assert x.shape == (8, 4)
    np.testing.assert_array_equal(x[:, 3], [0, 0, 1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(x[:, 1], [0, 0, 0, 0, 0, 0, 0, 1])  # failure excluded too
    np.testing.assert_array_equal(x[:, 2], [0, 0, 0, 1, 0, 0, 0, 0])
    assert x[2, 0] == pytest.approx(0.0)  # median-imputed like a missing value
    categorical = analysis._FeatureEncoder([egfr]).fit(frame).transform(frame)
    # Mutant, Wild-type, __missing__, documented_unknown
    assert categorical.shape == (8, 4)
    np.testing.assert_array_equal(categorical[:, 3], [0, 0, 1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(categorical[:, 2], [0, 0, 0, 1, 0, 0, 0, 1])
    assert categorical[[2, 5], :3].sum() == 0
    # Without training occurrences a held-out documented_unknown is missing.
    train = frame.assign(egfr=["Mutant", "Wild-type"] * 4)
    held = analysis._FeatureEncoder([egfr]).fit(train).transform(frame)
    assert held.shape == (8, 3)
    np.testing.assert_array_equal(held[:, 2], [0, 0, 1, 1, 0, 1, 0, 1])


# ---------------------------------------------------------------------------
# Informativeness filter and reporting


# ---------------------------------------------------------------------------
# End to end
