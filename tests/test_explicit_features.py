from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from oci.config import ExplicitFeatureSpec
from oci.extraction import (
    ExplicitFeatureValue,
    VLLMFeatureExtractor,
    build_extraction_prompt,
    compact_contract_lexical_context,
)
from oci.models.explicit_feature_featurizer import (
    filter_specs_by_role,
    get_raw_explicit_features,
)


def test_complete_paged_provider_processes_and_reconciles_every_page(tmp_path: Path):
    from oci.extraction import COMPLETE_PAGED_RESPONSE_SCHEMA, CompletePageResponse
    from oci.inference.agentic_explicit_feature_forest import (
        VLLMExplicitFeatureExtractionProvider,
    )

    note = ("baseline text " * 12) + "ECOG 2" + (" later text" * 12)
    spec = ExplicitFeatureSpec(
        name="performance_status",
        type="categorical",
        categories=["ECOG 0", "ECOG 1", "ECOG 2"],
        description="Baseline ECOG score",
        roles=["confounder"],
    )
    provider = object.__new__(VLLMExplicitFeatureExtractionProvider)
    provider.output_dir = tmp_path
    provider.config = SimpleNamespace(text_column="clinical_text")
    provider.feature_config = SimpleNamespace(
        complete_page_core_chars=80,
        complete_page_context_chars=10,
        complete_page_max_chars=100,
        complete_reconciliation_fan_in=3,
        extraction_batch_size=4,
    )

    class FakeExtractor:
        def __init__(self):
            self.pages = []

        def extract_complete_page(self, *, text, page, feature, geometry):
            self.pages.append(page)
            marker_start = text.find("ECOG 2", page.context_start, page.context_end)
            if marker_start >= 0:
                return CompletePageResponse.validate(
                    {
                        "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
                        "status": "positive",
                        "normalized_value": "ECOG 2",
                        "reason": None,
                        "citations": [
                            {
                                "start": marker_start,
                                "end": marker_start + len("ECOG 2"),
                                "text": "ECOG 2",
                            }
                        ],
                    },
                    text=text,
                    page=page,
                )
            return CompletePageResponse.validate(
                {
                    "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
                    "status": "negative",
                    "normalized_value": None,
                    "reason": None,
                    "citations": [],
                },
                text=text,
                page=page,
            )

        def reconcile_complete_pages(self, *, text, feature, children):
            positive = next(
                (
                    child["response"]
                    for child in children
                    if child["response"]["status"] == "positive"
                ),
                None,
            )
            response = positive or {
                "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
                "status": "negative",
                "normalized_value": None,
                "reason": None,
                "citations": [],
            }
            return {
                "child_ids": [child["node_id"] for child in children],
                **response,
            }

    extractor = FakeExtractor()
    frame = provider._extract_complete_paged_spec(
        dataset=pd.DataFrame({"clinical_text": [note]}),
        specs=[spec],
        extractor=extractor,
    )

    cores = sorted(extractor.pages, key=lambda page: page.core_start)
    assert [(page.core_start, page.core_end) for page in cores] == [
        (start, min(len(note), start + 80)) for start in range(0, len(note), 80)
    ]
    assert frame.loc[0, "explicit_feat_performance_status"] == "ECOG 2"
    assert not bool(frame.loc[0, "explicit_feat_performance_status_missing"])
    assert list((tmp_path / "complete_paged_ledgers").glob("*.json"))


def test_explicit_feature_roles_are_valid_and_deduped():
    spec = ExplicitFeatureSpec(
        name="ecog_status",
        type="categorical",
        categories=["0", "1", "2"],
        roles=["confounder", "effect_modifier", "confounder"],
    )

    assert spec.roles == ["confounder", "effect_modifier"]

    with pytest.raises(ValueError, match="roles required"):
        ExplicitFeatureSpec(name="age", type="continuous")

    with pytest.raises(ValueError, match="invalid roles"):
        ExplicitFeatureSpec(name="age", type="continuous", roles=["instrument"])


def test_raw_explicit_features_split_by_role_with_overlap():
    specs = [
        ExplicitFeatureSpec(
            name="ecog_status",
            type="categorical",
            categories=["0", "1", "2"],
            roles=["confounder", "effect_modifier"],
        ),
        ExplicitFeatureSpec(
            name="age",
            type="continuous",
            roles=["confounder"],
        ),
        ExplicitFeatureSpec(
            name="marker",
            type="continuous",
            roles=["effect_modifier"],
        ),
    ]
    values = [
        {
            "ecog_status": "1",
            "ecog_status_missing": False,
            "age": 60.0,
            "age_missing": False,
            "marker": 2.0,
            "marker_missing": False,
        },
        {
            "ecog_status": "2",
            "ecog_status_missing": False,
            "age": 70.0,
            "age_missing": False,
            "marker": 4.0,
            "marker_missing": False,
        },
    ]

    confounder_specs = filter_specs_by_role(specs, "confounder")
    effect_specs = filter_specs_by_role(specs, "effect_modifier")
    assert [s.name for s in confounder_specs] == ["ecog_status", "age"]
    assert [s.name for s in effect_specs] == ["ecog_status", "marker"]

    w_features, w_names = get_raw_explicit_features(values, specs, role="confounder")
    x_features, x_names = get_raw_explicit_features(values, specs, role="effect_modifier")

    assert w_names == [
        "ecog_status_1",
        "ecog_status_2",
        "ecog_status_missing",
        "age_normalized",
        "age_missing",
    ]
    assert x_names == [
        "ecog_status_1",
        "ecog_status_2",
        "ecog_status_missing",
        "marker_normalized",
        "marker_missing",
    ]
    assert len(w_features[0]) == 5
    assert len(x_features[0]) == 5
    assert w_features[0][0] == 1.0
    assert x_features[0][0] == 1.0


@pytest.mark.parametrize("mode", ["server"])
@pytest.mark.parametrize("maximum", [4])
def test_standalone_extraction_batches_variables_and_preserves_patient_values(monkeypatch, mode, maximum):
    specs = [ExplicitFeatureSpec(name=f"f_{i}", type="continuous", roles=["confounder"]) for i in range(23)]
    extractor = VLLMFeatureExtractor(specs, mode=mode, max_variables_per_extraction_request=maximum)
    monkeypatch.setattr(extractor, "_ensure_initialized", lambda: None)
    calls = []

    def group(text, group_specs):
        calls.append((text, [s.name for s in group_specs]))
        return {s.name: ExplicitFeatureValue(s.name, s.type, int(text) + int(s.name[2:]), False) for s in group_specs}

    monkeypatch.setattr(extractor, "_extract_single_server_group", group)
    monkeypatch.setattr(extractor, "_extract_batch_python_api_group", lambda texts, specs: [group(t, specs) for t in texts])
    results = extractor.extract(["100", "200"], batch_size=2, show_progress=False)
    assert len(calls) == 2 * ((23 + maximum - 1) // maximum)
    assert all(len(names) <= maximum for _, names in calls)
    assert [len(names) for text, names in calls if text == "100"] == [maximum] * (23 // maximum) + [23 % maximum]
    for text, result in zip([100, 200], results):
        assert list(result) == [s.name for s in specs]
        assert [result[s.name].value for s in specs] == [text + i for i in range(23)]


def test_build_extraction_prompt_truncates_to_note_tail():
    specs = [
        ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
    ]

    prompt = build_extraction_prompt(
        "beginning age 44. " + ("middle " * 20) + "end age 71.",
        specs,
        max_text_length=30,
        context_strategy="tail",
    )

    assert "end age 71" in prompt
    assert "beginning age 44" not in prompt


def test_contract_lexical_context_keeps_matching_passages_verbatim_in_order_under_budget():
    specs = [
        ExplicitFeatureSpec(
            name="ecog_performance_status",
            type="categorical",
            categories=["0", "1", "2", "3", "4"],
            description="ECOG performance status",
            roles=["confounder"],
        ),
        ExplicitFeatureSpec(
            name="brain_metastases",
            type="categorical",
            categories=["yes", "no"],
            description="Brain metastases on imaging",
            roles=["effect_modifier"],
            value_aliases={"yes": ["intracranial lesions"]},
        ),
    ]
    filler = "The patient walked in the garden and discussed travel plans with family. "
    ecog = "ECOG performance status 1 at the visit. "
    metastases = "MRI showed intracranial lesions consistent with brain metastases. "
    note = (
        "Opening remark. " + filler * 20 + ecog + filler * 20 + "Unrelated midpoint remark. "
        + filler * 20 + metastases + filler * 20 + "Closing remark."
    )

    context = compact_contract_lexical_context(note, specs, max_chars=3000)

    assert len(context.text) <= 3000
    assert not context.fallback_tail_used
    assert ecog in context.text and metastases in context.text
    assert context.text.index(ecog) < context.text.index(metastases)
    for remark in ("Opening remark", "Unrelated midpoint remark", "Closing remark"):
        assert remark not in context.text
    spans = [(excerpt.start, excerpt.end) for excerpt in context.selected_excerpts]
    assert spans == sorted(spans)
    assert all(end <= next_start for (_, end), (next_start, _) in zip(spans, spans[1:]))
    for start, end in spans:
        assert note[start:end] in context.text
    prompt = build_extraction_prompt(
        note, specs, max_text_length=3000, context_strategy="contract_lexical_rag"
    )
    assert context.text in prompt
    assert "Unrelated midpoint remark" not in prompt

    # With nothing to match, the end of the note is kept.
    unmatched = "Opening remark. " + filler * 30 + "Closing remark."
    fallback = compact_contract_lexical_context(unmatched, specs, max_chars=500)
    assert fallback.fallback_tail_used
    assert len(fallback.text) <= 500
    assert fallback.text.endswith("Closing remark.")
    assert "Opening remark" not in fallback.text


def test_unmatched_categorical_value_is_missing_not_the_reference_level(caplog):
    specs = [
        ExplicitFeatureSpec(
            name="stage",
            type="categorical",
            categories=["I", "II", "III"],
            roles=["confounder"],
        ),
    ]
    values = [
        {"stage": "I", "stage_missing": False},  # reference level
        {"stage": "stage IV", "stage_missing": False},  # matches no category
        {"stage": None, "stage_missing": False},  # inconsistent null
        {"stage": "III", "stage_missing": False},
    ]
    with caplog.at_level("WARNING"):
        features, names = get_raw_explicit_features(values, specs)

    assert names == ["stage_II", "stage_III", "stage_missing"]
    assert features == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
    assert "matched no declared category" in caplog.text
