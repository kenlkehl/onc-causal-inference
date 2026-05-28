# May 28, 2026 Iteration Summary

This summarizes the long feature-discovery and causal-inference iteration from
today. The main thread was trying to discover useful confounder (`W`) and effect
modifier (`X`) features from clinical text without assuming the true simulation
variables in advance.

## Starting Problem

The core concern was that many successful synthetic-data runs still depended on
knowing what feature concepts to look for. We explored whether an all-neural or
weakly specified text method could discover:

- nuisance/confounding structure useful for treatment and outcome prediction;
- treatment-effect heterogeneity structure useful for CATE estimation;
- a clean split between `W` features for adjustment and `X` features for effect
  modification.

The target downstream estimator remained causal forest, because it gave a
strong and interpretable CATE model once the right structured inputs were
available.

## Methods Explored

### Token Hash / Bag-of-Ngrams Path

We reviewed and extended a sparse token-hash style approach. The basic idea was:

- hash tokenizer IDs or text ngrams into sparse columns;
- use unsupervised SVD on that sparse text matrix to create dense `W` nuisance
  features;
- score candidate sparse columns as potential `X` effect modifiers by comparing
  treated-vs-untreated outcome contrasts when the feature is present versus
  absent;
- penalize columns that mostly shift baseline outcome risk in the control arm;
- optionally turn selected columns into soft hash gates.

This behaved a lot like a bag-of-ngrams or TF-IDF pipeline. The useful piece was
not really the neural framing; it was the explicit effect-modifier contrast score
plus the unsupervised `W` representation. The SVD `W` worked better than the
neural `W` attempts because it preserved broad text variation without trying to
solve nuisance prediction too early. The causal forest then handled nuisance and
heterogeneity modeling downstream.

### Neural Hash Gates / Causal-Purity Objective

We sketched and implemented neural versions with an `X` branch built from sparse
CNN/hash gates and a causal-purity objective. The goal was for learned gates to
identify text features whose treatment interaction signal was large while
ordinary prognostic signal was penalized.

This was conceptually close to the hand-scored token-hash method, but it was
harder to optimize. The neural objective had to learn extraction, nuisance
structure, and effect modification at the same time, which made it unstable and
less convincing than the simpler SVD plus causal forest pipeline.

### Matched / Contrastive Ideas

We considered a nuisance model that predicts both outcome and treatment, then
forms treated/untreated matched pairs with similar propensities and trains a
contrastive representation to explain outcome differences. This aligns with the
earlier binned-batch and matched-pair R-learner thread.

The idea remains plausible, but in these synthetic tests it did not become the
clear winning path. It also inherited a difficult chicken-and-egg problem:
matching quality depends on already having good nuisance features.

### Candidate Variable / Text-SVD Extractor

We tried a more explicit middle ground: generate or define candidate variables,
represent their text evidence, and use SVD-like vectorization to feed the causal
forest. This made the role of candidate variables clearer, but it still required
some prior notion of what variables to consider. That was the main unresolved
weakness.

### Residual Effect-Modifier Search

We designed an approach where user-provided structured variables first train a
baseline causal forest, then an extra text-derived residual `X` representation
tries to improve on the baseline tau predictions. In simulation sweeps, adding
the residual `X` features often made causal forest performance worse, including
cases where the original structured `X` features were still present.

The likely issue was that the residual features added noisy degrees of freedom
without a strong enough out-of-sample acceptance mechanism. The lesson was that
residual discovery needs aggressive validation before adding features to the
final forest.

### Low-Dimensional LLM Extractor Variants

We revisited the original end-to-end R-learner neural design:

```text
clinical text -> LLM embedding -> W branch -> outcome / propensity
                          \-> X branch -> tau
```

We considered whether much lower-dimensional `W` and `X` bottlenecks might help.
The requested direction was specifically LLM extraction rather than token-hash
extraction. This still left the same optimization problem: the model had to
learn the causal role split from weak supervision. It did not replace the need
for explicit, validated variable extraction.

## Main Conclusion

The durable conclusion was that fully unsupervised or end-to-end neural discovery
of clean causal `W` and `X` variables is very hard in this setting. The best
empirical behavior came from methods that either:

- preserved broad nuisance information without supervising it too tightly
  (`SVD W`), or
- used explicit candidate variables and let causal forest do the CATE modeling.

But the most realistic production direction is not to pretend the system can
discover every causal variable from scratch. Instead:

1. Let users provide candidate confounders and effect modifiers in structured
   form.
2. Use an LLM to extract those variables from each patient record.
3. Fit causal forest with role-tagged explicit features:
   - confounder-role features as `W`;
   - effect-modifier-role features as `X`.
4. Add an agentic loop that proposes additional baseline variables, validates
   them by nested CV, and only accepts changes that improve non-oracle metrics
   consistently.

## Agentic Explicit-Feature Path

The final production path from today is the agentic explicit-feature forest. It
does nested cross-validation:

- outer folds estimate honest held-out performance;
- inner folds decide whether an agent-proposed feature add/remove/re-role action
  improves the current feature set;
- acceptance uses R-loss, outcome AUROC, treatment AUROC, extraction coverage,
  and fold-consistency thresholds;
- true ITE is used only for simulation reporting, not for feature acceptance.

The agent can start from:

- user-provided variables;
- an empty feature set;
- simulated partial user lists in oracle experiments.

This path now has production-facing code in the main repo and archived copies in
this folder.

## Cleanup Decision

After the exploration, the main repo was narrowed back down to keep only the
agentic explicit iteration path as the active production direction. Earlier
experimental branches were archived here under `codex_iteration/`, including:

- neural causal hash gates;
- token/window/hash extractor experiments;
- candidate-variable text-SVD forest scripts;
- neural mention-slot extractors;
- residual modifier forest experiments;
- older oracle runner snapshots.

The current active repo keeps the agentic explicit-feature forest integration,
config support, extraction/cache changes, docs, example config, tests, and oracle
runner support for evaluating the agentic path.

## Latest Oracle Runner Addition

At the end of the session, `oracle_experiment_scripts/run_oracle_experiments.py`
was extended so the oracle grid can include `agentic_explicit_feature_forest`.
It now supports agentic grid hyperparameters such as:

- number of search iterations;
- number of initial structured variables;
- starting-variable strategy:
  - `true_first`;
  - `modifiers_first`;
  - `distractors`;
  - `mixed`.

This lets the synthetic-data oracle runner simulate cases where a user supplies
some true confounders/effect modifiers, only modifiers, plausible distractors, or
nothing at all.

## Practical Takeaway

For this project, the strongest path is now:

```text
clinical text
  -> LLM explicit variable extraction
  -> role-tagged structured feature table
  -> nested-CV agentic variable search
  -> causal forest using W and X roles
  -> oracle-only simulation metrics when true ITE exists
```

The unresolved research question is still whether a neural system can discover
the relevant causal variables without user-specified candidates. Today strongly
suggested that, at least for the current synthetic setup and available training
signals, explicit variable extraction plus rigorous nested-CV feature acceptance
is the more reliable engineering route.
