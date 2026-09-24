import pytest


@pytest.fixture(autouse=True)
def stage2_caller_spies(request, monkeypatch):
    if request.module.__name__.split('.')[-1] in {
        "test_plain_handoff_stage2", "test_stage2_extraction_reliability",
        "test_stage2_role_adjudication", "test_stage2_multi_model",
        "test_stage2_modifier_count", "test_stage2_independent_tasks",
    }:
        from tests.stage2_prompt_spy import install
        install(monkeypatch)
