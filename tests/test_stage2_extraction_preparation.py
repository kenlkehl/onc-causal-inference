
import pytest

from oci.inference import plain_handoff_stage2_analysis as analysis


class CharacterTokenizer:
    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.calls += 1
        return range(sum(len(m["content"]) + len(m["role"]) + 2 for m in messages) + 11)


def definition(index=0):
    return {
        "name": f"state_{index}", "description": "Documented clinical state.",
        "value_type": "categorical", "categories_or_unit": ["red", "blue"],
        "measurement_definition": "Use the most frequently documented state.",
        "missing_value_rule": "Return null when undocumented.",
        "conflict_resolution": {"strategy": "mode"},
    }


@pytest.mark.parametrize("token_limited", [True])
def test_prepared_page_prompts_preserve_lossless_coverage_and_every_batch_budget(token_limited):
    batches = [[definition(0)], [dict(definition(1), description="Long definition. " * 40)]]
    templates = [analysis._page_extraction_prompt_template(definitions=b) for b in batches]
    tokenizer = CharacterTokenizer()
    overhead = max(analysis.prompt_token_count(tokenizer, analysis._render_page_extraction_prompt(p, ""))
                   for p in templates)
    budget = overhead + 200
    char_budget = 100_000 if token_limited else budget
    token_budget = budget if token_limited else 100_000
    source = "red 漢字\n\nblue — repeated observation. " * 50
    pages = analysis._lossless_extraction_pages(
        {"row_id": 5, "text": source}, definition_batches=batches,
        max_prompt_chars=char_budget, tokenizer=tokenizer,
        input_token_budget=token_budget, prompt_templates=templates,
    )
    assert len(pages) > 1
    assert "".join(p["text"] for p in pages) == source
    for page in pages:
        meta = page["page"]
        assert source[meta["char_start"]:meta["char_end"]] == page["text"]
        for batch in batches:
            prompt = analysis._page_extraction_prompt(definitions=batch, row=page)
            assert analysis._prompt_chars(prompt) <= char_budget
            assert analysis.prompt_token_count(tokenizer, prompt) <= token_budget
    # Reusing shared prefixes across patients must not append source text to them.
    assert templates == [analysis._page_extraction_prompt_template(definitions=b) for b in batches]
