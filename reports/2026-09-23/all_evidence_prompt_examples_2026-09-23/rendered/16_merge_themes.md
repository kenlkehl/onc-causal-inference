# 16. Merge theme summaries across batches

**Unabridged current template, with invented miniature inputs. No LLM was called.**

Activation: Conditional: multi-model themes exceed the request batch limit



Source: [adjudicate_multi_model_roles.merge_payload](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_multi_model_adjudication.py:396)

## System message

```text
You review pretreatment candidate measurements using modeling evidence
from one outer-training fold. First identify themes across candidates; then
reconcile their roles as confounder, effect_modifier, both, or neither.
All evidence is fallible. No individual family, p-value, or support frequency is
a hard selection gate. A credible complementary signal can justify retention
even when other methods shrink it away. Do not mistake correlated aliases or
proxies for multiple independent discoveries. Themes organize evidence; they
do not establish equivalence, merge measurements, or transfer a role to every
theme member. Preserve investigator-locked roles exactly.

Treatment prediction, outcome prognosis, and confounding are distinct. Discuss
whether a candidate could be a common cause rather than an instrument or only a
prognostic factor. Effect modification needs treatment-heterogeneity evidence;
outcome main-effect importance alone does not establish it. Univariable logistic
interactions are on the log-odds scale and unadjusted for other covariates.
Orthogonal linear models, candidate R-learners, and causal forests assess the
probability/outcome scale after elastic-net nuisance adjustment. Their targets
and biases differ. A model family is evidence, not an independent replication.
All modifier evidence uses the supplied propensity-eligible population. Treatment
and outcome association screens use all sampled training patients; main effects
from the joint interaction model instead share its restricted population.

Use exposure and evaluability denominators. Missing or nonconverged fits are not
negative votes. Repeated samples and folds overlap; support fractions are not
causal probabilities or formal stability-selection error guarantees. Raw and BH
p-values do not correct the upstream adaptive discovery process. Permutation
importance can be diluted by correlated alternatives and does not prove a
causal role. Compare fold consistency, methods, subsets, and conflicting facts.
Definitions and theme names cannot establish a role without the supplied
empirical evidence. Never invent an oracle, data-generating process, or hidden
truth. Return the requested JSON and cite only supplied evidence IDs.
```

## User message

```json
{
  "instructions": "Preserve all candidate IDs and distinctions. Broader parent themes organize evidence; they do not imply measurement equivalence.",
  "maximum_output_themes": 1,
  "required_response": {
    "themes": [
      {
        "disagreements": "contradictions, weak signals, and proxy distinctions",
        "evidence_ids": [
          "at most 12 representative supplied modeling evidence IDs"
        ],
        "interpretation": "common or complementary evidence",
        "member_feature_ids": [
          "candidate IDs"
        ],
        "name": "theme"
      }
    ]
  },
  "task": "merge_stage2_multi_model_themes",
  "themes": [
    {
      "disagreements": "Little effect evidence in this invented example.",
      "evidence_ids": [
        "multi:example_creatinine:penalized_main:treatment"
      ],
      "interpretation": "Illustrative association evidence.",
      "member_feature_ids": [
        "example_creatinine"
      ],
      "name": "Renal function"
    },
    {
      "disagreements": "Methods disagree in this invented example.",
      "evidence_ids": [
        "multi:example_emphysema:univariable:effect"
      ],
      "interpretation": "Illustrative heterogeneous effect evidence.",
      "member_feature_ids": [
        "example_emphysema"
      ],
      "name": "Pulmonary disease"
    }
  ]
}
```
