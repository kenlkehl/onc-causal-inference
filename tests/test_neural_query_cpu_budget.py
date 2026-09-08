from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.config import BoWViewConfig, TfidfNuisanceStackScientificConfig
from oci.inference import neural_query_discovery_runtime as runtime
from oci.inference import research_all_evidence_workflow as workflow
from oci.inference.neural_query_agentic_forest import NeuralQueryAgenticForestConfig


@pytest.mark.parametrize(
    "devices,expected", [(("cuda:0",), [30]), (("cuda:0", "cuda:1"), [15, 15])]
)
def test_workflow_divides_neural_query_cpu_budget(
    tmp_path, monkeypatch, devices, expected
):
    import joblib

    specs = [{"scope_id": f"outer_{i:03d}_full", "outer_fold": i} for i in range(1, 4)]
    context = SimpleNamespace(
        config=SimpleNamespace(devices=devices, workers=30),
        dataset=None,
        applied_config=None,
        neural_query_config=None,
    )
    calls = []

    def run_lane(**kwargs):
        calls.append((kwargs["device"], kwargs["cpu_workers"]))
        return kwargs["specs"]

    monkeypatch.setattr(workflow, "_stage1_context_specs", lambda _: specs)
    monkeypatch.setattr(workflow, "_run_neural_query_context_lane", run_lane)
    monkeypatch.setattr(
        joblib,
        "Parallel",
        lambda **_: lambda tasks: [fn(*args, **kwargs) for fn, args, kwargs in tasks],
    )
    result = workflow._neural_queries_component(context, tmp_path)
    assert calls == list(zip(devices, expected))
    assert result["contexts"] == 3


def test_lane_passes_budget_through_context_to_discovery(tmp_path, monkeypatch):
    from oci.inference import embedding_contrast_discovery

    class FakeGenerator:
        def __init__(self, **kwargs):
            pass

        def prepare(self, dataset):
            pass

        def chunk_matrices(self, rows):
            return [np.ones((1, 2), dtype=np.float32) for _ in rows]

        def chunk_texts(self, rows):
            return [["text"] for _ in rows]

    class ReachedDiscovery(Exception):
        pass

    def discover(**kwargs):
        assert kwargs["cpu_workers"] == 30
        assert kwargs["devices"] == ("cpu",)
        raise ReachedDiscovery

    monkeypatch.setattr(
        embedding_contrast_discovery,
        "EmbeddingContrastEvidenceGenerator",
        FakeGenerator,
    )
    monkeypatch.setattr(runtime, "fit_context_query_discovery", discover)
    with pytest.raises(ReachedDiscovery):
        workflow._run_neural_query_context_lane(
            dataset=pd.DataFrame({"text": ["a", "b"], "t": [0, 1], "y": [1, 0]}),
            config=SimpleNamespace(
                output_dir=tmp_path,
                text_column="text",
                treatment_column="t",
                outcome_column="y",
                outcome_type="binary",
                seed=42,
            ),
            applied_config=SimpleNamespace(
                architecture=SimpleNamespace(
                    multi_model_forest=SimpleNamespace(
                        bow_views=[],
                        nuisance_folds=5,
                        tfidf_topic=SimpleNamespace(
                            nuisance_stack_scientific=TfidfNuisanceStackScientificConfig()
                        ),
                    )
                )
            ),
            neural_query_config=NeuralQueryAgenticForestConfig(),
            specs=[
                dict(
                    scope_id="outer_001_full",
                    outer_fold=1,
                    fold_key=1,
                    train_idx=[0],
                    heldout_idx=[1],
                )
            ],
            component_dir=tmp_path,
            device="cpu",
            cpu_workers=30,
        )


@pytest.mark.parametrize(
    "cpu_workers,devices,expected_nested",
    [
        (1, ("cpu",), [1, 1]),
        (30, ("cpu",), [30, 30]),
        (5, ("cuda:0", "cuda:1"), [2, 3]),
        (1, ("cuda:0", "cuda:1"), [1, 1]),
    ],
)
def test_context_and_nested_nuisances_use_budget_with_real_fits(
    monkeypatch,
    cpu_workers,
    devices,
    expected_nested,
):
    # Exercise real TF-IDF nuisance fits; stub GPU query optimization only.
    calls = []
    original = runtime.fit_joint_cross_fitted_nuisance_stacks

    def fit_nuisances(**kwargs):
        assert kwargs["tfidf_workers"] == kwargs["owner_cpu_budget"]
        calls.append(kwargs["tfidf_workers"])
        return original(**kwargs)

    def query_fit(*args, **kwargs):
        return dict(
            queries=np.zeros((1, 2), dtype=np.float32),
            train_standardized_scores=[0.0],
            query_drift=[0.0],
            loss_history=[],
            objective="test",
        )

    def final_bank(**kwargs):
        i = kwargs["bank_index"]
        return dict(
            bank=kwargs["bank"],
            bank_index=i,
            result={},
            consensus_seed=kwargs["seed"] + 1000 + i,
            final_refit_seed=kwargs["seed"] + 2000 + i,
        )

    monkeypatch.setattr(
        runtime, "fit_joint_cross_fitted_nuisance_stacks", fit_nuisances
    )
    monkeypatch.setattr(runtime, "fit_soft_target_queries", query_fit)
    monkeypatch.setattr(runtime, "fit_soft_contrast_queries", query_fit)
    monkeypatch.setattr(
        runtime,
        "soft_retrieval_activations",
        lambda chunks, *args, **kwargs: np.zeros((len(chunks), 1)),
    )
    monkeypatch.setattr(
        runtime,
        "standardized_direct_target_contrasts",
        lambda *args, **kwargs: {"standardized_scores": [0.0]},
    )
    monkeypatch.setattr(
        runtime,
        "standardized_cohort_moments",
        lambda *args, **kwargs: {"standardized_scores": [0.0]},
    )
    monkeypatch.setattr(runtime, "_fit_final_bank", final_bank)
    n = 48
    result = runtime.fit_context_query_discovery(
        row_ids=tuple(range(n)),
        chunks=[np.ones((1, 2), dtype=np.float32) for _ in range(n)],
        texts=tuple(
            f"patient group_{i % 3} arm_{i % 2} outcome_{(i // 2) % 2}"
            for i in range(n)
        ),
        treatment=np.asarray([i % 2 for i in range(n)], dtype=float),
        outcome=np.asarray([(i // 2) % 2 for i in range(n)], dtype=float),
        outcome_binary=True,
        nuisance_views=[
            BoWViewConfig(
                name="linear", min_df=1, max_df=1.0, max_features=32, bow_model="linear"
            )
        ],
        nuisance_stack_config=TfidfNuisanceStackScientificConfig(),
        query_config=NeuralQueryAgenticForestConfig(query_inner_folds=2),
        nuisance_folds=2,
        devices=devices,
        seed=42,
        cpu_workers=cpu_workers,
    )
    assert calls[0] == cpu_workers
    assert sorted(calls[1:]) == expected_nested
    assert [row["fold"] for row in result["subfold_audit"]] == [1, 2]
    assert result["validation_audits_used_for_selection"] is False


@pytest.mark.parametrize(
    "budget,error",
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_invalid_cpu_budget_fails_before_fitting(budget, error):
    with pytest.raises(error, match="cpu_workers must be a positive integer"):
        runtime.fit_context_query_discovery(
            row_ids=(),
            chunks=(),
            texts=(),
            treatment=np.array([]),
            outcome=np.array([]),
            outcome_binary=True,
            nuisance_views=[],
            nuisance_stack_config=None,
            query_config=None,
            nuisance_folds=2,
            devices=("cpu",),
            seed=42,
            cpu_workers=budget,
        )
