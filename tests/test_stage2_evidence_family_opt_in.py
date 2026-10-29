"""The treatment/outcome prediction families are opt-in diagnostics (M-12).

penalized_main and predictive_forest write only treatment- and outcome-role
records. Modifier ranking reads effect-role records, nuisance selection reads
the fold elastic-net supports, and concept-review prompts list effect-role rows
only. Turning the two families on or off must therefore leave every decision
unchanged. The comparisons run the same code twice, so no numerical value is
hard-coded.
"""

import json
from pathlib import Path

import pytest

from oci.inference.stage2_elastic_net_selection import (
    statistical_selection_config_from_mapping,
)
from oci.inference.stage2_multi_model_config import (
    DIAGNOSTIC_FAMILIES,
    ModifierCountConfig,
    multi_model_config_from_mapping,
)


@pytest.mark.parametrize("raw, message", [
    ({"diagnostic_families": ["causal_forest"]}, "diagnostic_families"),
])
def test_invalid_family_policy_is_rejected(raw, message):
    with pytest.raises(ValueError, match=message):
        multi_model_config_from_mapping(raw)


def test_example_config_parses_without_diagnostics():
    path = Path(__file__).resolve().parents[1] / "example_configs/research_all_evidence_multi_model.json"
    raw = json.loads(path.read_text())["stage2"]["statistical_selection"]
    config = statistical_selection_config_from_mapping(raw)
    assert config.multi_model.diagnostic_families == ()
    assert set(config.multi_model.active_families("binary")).isdisjoint(DIAGNOSTIC_FAMILIES)
    assert ModifierCountConfig().concept_review is False
