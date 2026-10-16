# Stage 1 architectures

Stage 1 generates evidence about which parts of the clinical record relate to
treatment choice, outcome, and variation in the treatment effect. It does so
with up to ten independent [architectures](glossary.md#architecture), each a
different way of reading text. Every architecture is fitted separately within
each outer fold's training patients, so the evidence for an outer fold never
uses that fold's held-out patients. Stage 2 later reads the evidence through
the [handoff](glossary.md#handoff); see [How OCI works](how_it_works.md) for the
overall flow.

Evidence from Stage 1 is a set of clues, not variables. A phrase, topic, or
retrieved excerpt may be a proxy, a documentation habit, or a mixture of several
clinical constructs. Stage 2 decides whether it supports a measurable patient
variable.

## The ten architectures

The canonical order below is the order used by every selector, manifest, and
report.

| # | ID | Default | What it looks for | Evidence it writes |
|---|---|---|---|---|
| 1 | `bow_nuisance` | on | Words and phrases that predict treatment or outcome | Ranked terms with scores, per text view |
| 2 | `bow_r_loss` | on | Words and phrases that explain treatment-effect variation after nuisance adjustment | Ranked terms with scores |
| 3 | `matched_pair_uplift` | on | Words and phrases that modify the treatment effect within matched treated/untreated pairs | Signed ranked terms, per text view |
| 4 | `htr_neural` | on | Context-dependent passages used by neural treatment, outcome, and effect heads | Attended text chunks per head |
| 5 | `embedding_whole_cohort` | on | Semantic directions separating patient groups across the cohort | Text chunks at both ends of each direction |
| 6 | `embedding_clustered` | on | The same directions computed within clusters of similar patients | Text chunks at both ends of each cluster-based direction |
| 7 | `tfidf_semantic_retrieval_contrasts` | off | A lexical summary of the excerpts retrieved by architectures 5 and 6 | Terms that distinguish the two sides of each direction |
| 8 | `tfidf_topics` | on | Groups of co-occurring terms associated with treatment, outcome, or effect variation | Topics (term groups) per bank |
| 9 | `tfidf_orphan_ngrams` | off | Effect-associated phrases that no topic captured | Clusters of orphan phrases |
| 10 | `neural_query_moments` | on | Learned semantic detectors in the embedding space | Witness text chunks and contrastive phrases per query |

The architectures belong to three computational components: architectures 1 to 7
write under `components/text_models/`, architectures 8 and 9 under
`components/tfidf/`, and architecture 10 under `components/neural_queries/`. A
shared chunk-embedding cache (`components/embedding_cache/`) serves
architectures 5, 6, 7, and 10. Most components also write row-level prediction
files that the [oracle-safe evaluator](#oracle-safe-evaluation) reads.

For binary outcomes, every effect-variation screen below (architectures 2, 4,
5, 6, 8, 9, and 10) uses the [DINA](glossary.md#dina) objective, built from
cross-fitted propensity and arm-specific outcome predictions; effects are
reported on the log-odds-ratio scale inside the model and converted to risk
differences for prediction. For continuous outcomes the same screens use the
residualized relation `Y − m(X) ≈ τ(X)·(T − e(X))` that underlies the
[R-loss](glossary.md#r-loss). Architecture 3 instead compares the members of
matched treated/untreated pairs.

### 1. Sparse treatment and outcome associations (`bow_nuisance`)

Several [TF-IDF](glossary.md#bow-and-tf-idf) bag-of-words views (unigrams,
bigrams, trigrams, and longer phrases, with linear and tree-based learners) are
fitted to predict treatment and outcome. The terms with the largest coefficients
or importances are recorded, with their sign when the learner provides one. The
model is sensitive to explicit chart language such as diagnoses, performance
status, prior therapies, laboratory abnormalities, and administrative patterns.
A characteristic that predicts both treatment and outcome is a candidate
confounder, but the term itself remains only a clue.

### 2. Sparse residual-effect associations (`bow_r_loss`)

The same bag-of-words views are fitted to the residual-effect target: a sparse
DINA model for binary outcomes, or pseudo-outcome and weighted R-loss
regressions for continuous outcomes. Terms that explain this residual relation
are candidates for effect modification. They give a different signal from
ordinary outcome prediction, but they do not show that the named characteristic
modifies the treatment effect. The [R-learner weighting note](design_notes/r_learner_weighting.md)
discusses the weighted and unweighted forms.

### 3. Matched-pair effect modification (`matched_pair_uplift`)

Within each training context, treated patients are matched one-to-one, without
replacement, to untreated patients. A pair is allowed only when both
cross-fitted scores lie within a caliper: the propensity `ê` (absolute
caliper `matched_pair_propensity_caliper`, 0.05 by default) and the
[prognostic score](glossary.md#prognostic-score) `μ̂0`, the predicted outcome
without treatment (`matched_pair_prognostic_caliper`: absolute on the
probability scale for binary outcomes, 0.05 by default; in standard deviations
of `μ̂0` over the training patients for continuous outcomes, 0.2 by default).
Using a prognostic score alongside the propensity follows Hansen (2008). The
matching is a maximum bipartite matching (Hopcroft–Karp) in which
each treated patient tries its closest eligible controls first; a treated
patient with no control inside both calipers stays unmatched. Each treated
patient's candidate list keeps at most its 256 closest eligible controls,
which matters only in large cohorts.

Both scores are out of fold for every training patient:

- `ê` is the mean of the bag-of-words nuisance propensities (architecture 1),
  each predicted by a model fitted on the other nuisance folds. The HTR
  propensity is not used, because its calibration step is fitted on
  predictions from models that saw the patient.
- For binary outcomes, `μ̂0` is the untreated-arm probability from the
  context's cross-fitted DINA nuisance models (a ridge logistic regression on
  untreated patients, fitted on the other folds). For continuous outcomes it is
  a cross-fitted elastic-net linear regression on the untreated patients of the
  other folds, using the first bag-of-words view's vocabulary.

A pair is described by the average of its two members' bag-of-words vectors:
one block of features per pair, with no treated or control side. One model is
fitted per distinct vocabulary among the configured bag-of-words views, so
views that differ only in their learner share a fit.

**Binary outcomes: conditional logistic regression on discordant pairs.**
Only pairs in which exactly one member had the event inform a matched-pair
comparison; pairs in which both or neither had it are discarded (Breslow and
Day 1980, chapters 5 and 7). With `D = 1` when the treated member had the
event, the model is

```text
logit P(D = 1 | discordant pair) = [logit μ̂0(x_t) − logit μ̂0(x_c)] + α + β·x̄_pair
```

The bracketed term is a fixed offset that absorbs prognostic imbalance left
inside the pair. The intercept `α` is the average conditional log odds ratio of
treatment (features are centered, so it applies at the average pair), and each
coefficient is effect modification on the log-odds-ratio scale.

**Continuous outcomes: adjusted paired differences.** Every matched pair is
kept, and the target is
`Δ = (Y_t − Y_c) − (μ̂0(x_t) − μ̂0(x_c))`, regressed on the pair features. The
intercept is the average effect and the coefficients are additive effect
modification.

Both fits are elastic nets (`matched_pair_l1_ratio`, 0.5 by default) on
standardized pair features with an unpenalized intercept. The penalty is chosen
from a 30-value path by `matched_pair_cv_folds`-fold cross-validation over
pairs (5 by default), minimizing the held-out conditional negative
log-likelihood (binary) or squared error (continuous). The vocabulary and the
column scaling are fitted on the analysis pairs' records without outcome
information. The evidence is the `top_n_features` terms with the largest
absolute standardized coefficients, kept with their sign: a positive term
means a higher conditional odds of the outcome (binary) or a larger outcome
(continuous) for a treated patient than for a similar untreated patient. Stage
2 attaches supporting sentences to these terms the same way it does for the
other bag-of-words architectures.

**Validation.** A nested cross-fit over the same pair folds, in which each
fold's penalty is chosen by an inner cross-validation on the other folds'
pairs, reports the held-out log-likelihood gain (binary) or squared-error
reduction (continuous) of the fitted model over the intercept-only,
constant-effect model with the same offset. It is reported in the evidence and
in the evaluator's metrics and never filters evidence. For the outer held-out
patients the architecture also writes a prediction
`τ̂(x) = intercept + coefficients · x` from each patient's own features; nothing
is fitted on those patients.

**Power.** Only discordant pairs inform the binary fit, roughly `2p(1 − p)` of
the matched pairs when the outcome rate is `p`, and patients without a match
within the calipers contribute nothing. When a context has fewer than
`matched_pair_min_discordant_pairs` discordant pairs (binary) or
`matched_pair_min_pairs` matched pairs (continuous), both 30 by default, the
architecture emits no evidence for that context and records the reason.
Matching balances only what the nuisance models capture, so a term is evidence
for a candidate effect modifier, not an estimate of an individual treatment
effect.

### 4. Hierarchical transformer (`htr_neural`)

The hierarchical transformer splits a record into overlapping chunks (96 words
with a 24-word overlap by default), encodes each chunk with a small pretrained
encoder, and combines the chunks with a document-level transformer. Separate
heads learn treatment, outcome, and effect objectives; the effect heads use
nuisance predictions fitted and frozen inside the effect-training fold. Its
attention identifies the chunks the model relied on. Unlike a bag-of-words
model, it can distinguish some uses of the same vocabulary and combine
information spread across a long history. Attention describes what the model
used, not a causal explanation. `--disable-htr` turns off this architecture.

### 5. Whole-cohort embedding contrasts (`embedding_whole_cohort`)

Each record is cut into chunks (256 words with a 64-word overlap by default) and
encoded by a frozen sentence-embedding model. A patient's embedding is the mean
of the patient's chunk embeddings, normalized to unit length. Within each
training context, the architecture computes directions in embedding space:

| Direction | Definition |
|---|---|
| Treatment | `mean(T = 1) − mean(T = 0)` |
| Outcome | `mean(outcome present) − mean(outcome absent)` (upper versus lower outcome quantile for continuous outcomes) |
| Within-arm outcome | The outcome direction computed separately in each arm |
| Treatment-by-outcome interaction | `mean(T=1,Y=1) − mean(T=1,Y=0) − mean(T=0,Y=1) + mean(T=0,Y=0)` |
| Confounder vector | The normalized average of the normalized treatment and outcome directions |
| Residualized interaction | The interaction direction with its projection on the treatment and outcome directions removed |
| Effect score | Upper versus lower quantile (20%) of a per-patient effect score: the DINA score for binary outcomes, with and without Fisher-information weights; the R pseudo-outcome `(Y − m̂)/(T − ê)` weighted by `(T − ê)²` and the unweighted orthogonal score `(Y − m̂)(T − ê)` for continuous outcomes |

For each direction, the chunks most and least aligned with it are retrieved (up
to 12 per side, at most 2 per patient). Both tails are evidence: a scalar or
monotone characteristic often appears as an opposition between the two tails
rather than as a cluster of similar positive excerpts. The retrieved chunks are
what make a numerical direction interpretable; they let a reader judge whether
the direction reflects a coherent patient characteristic.

### 6. Cluster-based embedding contrasts (`embedding_clustered`)

A single cohort-wide direction can be dominated by common documentation
patterns and can average away signals that live in different parts of the
cohort. This architecture first clusters patient embeddings with mini-batch
k-means (10 clusters by default), then:

1. In each cluster with at least 24 patients and at least 8 patients per arm, it
   computes the local treatment direction
   `d_s = mean(T = 1, cluster s) − mean(T = 0, cluster s)`.
2. In each cluster with at least 4 patients in every treatment-by-outcome cell,
   it computes the local interaction direction and removes its projection on the
   local treatment and outcome directions.
3. Each local direction is normalized and weighted by `√n_s`, where `n_s` is the
   cluster size. The weighted directions of each kind are stacked into a matrix
   and decomposed by singular value decomposition; the top right singular
   vectors (up to 5 per kind) become retrieval directions.

This is an uncentered principal component analysis of the local contrasts: it
finds the leading eigenvectors of `Σ_s n_s · normalize(d_s) normalize(d_s)ᵀ`.
Where a whole-cohort contrast asks "what is the average difference between
groups?", the cluster basis asks "what kinds of group differences recur across
patient subgroups?" Each component records which clusters contributed most
(`cluster_component_loadings`), with their sizes and local contrast norms.
Chunks are retrieved for each component as in architecture 5.

A cluster is a computational neighborhood, not a disease subtype. Small cells
amplify noise, which is why the minimum cluster, arm, and cell counts apply; for
binary outcomes, interaction components are best treated as hypotheses and
judged by cross-fold stability and by whether the retrieved chunks suggest a
plausible pretreatment variable. If a training context cannot support the
configured clustering (fewer usable patients than the cluster count times the
minimum cluster size, or fewer than two qualifying clusters for either kind of
direction), the architecture stops with an error instead of producing partial
evidence. The settings are in
`science.stage1.architecture.multi_model_forest.embedding_contrast.cluster_local_scientific`.

### 7. Semantic-retrieval vocabulary (`tfidf_semantic_retrieval_contrasts`, optional)

For each direction from architectures 5 and 6, a TF-IDF model is fitted to the
excerpts retrieved from the two tails, and the terms that best distinguish the
sides are reported (up to 30 per side). Each term keeps the identity of its
parent direction, so terms from a whole-cohort direction stay distinguishable
from terms from a cluster-based one. This is a readable projection of
architectures 5 and 6, not a separate model. The terms can show whether a
direction concerns, for example, disease burden, functional status, toxicity,
or a documentation artifact; nonspecific vocabulary is a reason to keep the
direction's meaning open.

### 8. Consensus TF-IDF topics (`tfidf_topics`)

This architecture does not use embeddings. A fold-local TF-IDF representation
is screened separately for terms associated with treatment, with outcome, and
with residual effect (three banks), using cross-fitted nuisance predictions from
TF-IDF models. Non-negative matrix factorization ([NMF](glossary.md#nmf)) is fitted
over several random seeds, and topics that recur across seeds are kept. A topic
is a co-occurrence pattern, not necessarily one variable: a topic containing
frailty, oxygen use, and hospitalization terms may be one severity construct or
several measurements that should stay separate.

### 9. Orphan n-grams (`tfidf_orphan_ngrams`, optional)

Starting from the effect-associated terms of architecture 8's screen, this
architecture removes every term already represented in the effect-topic bank.
The remaining phrases, the "orphans", can carry residual-effect signal even
though no stable topic captured them. Retained phrases are grouped only into
small clusters of similar wording (at most 15 terms each), because a rare but
precise measurement may matter even when it belongs to no broad topic.

### 10. Learned neural queries (`neural_query_moments`)

Neural queries are trainable vectors that search the chunk-embedding space for
recurring patterns. Separate banks (five queries each by default) are optimized
for treatment, outcome, and effect objectives, with penalties that encourage
useful activation and diversity among queries. Each patient is scored by the
maximum cosine similarity between any of the patient's chunks and a query; the
highest-scoring patients' best-matching chunks (one per patient per query by
default) are saved as witnesses, together with phrases that distinguish those
chunks from background text. Queries can learn several distinct detectors for
the same objective, which a single mean-difference direction cannot.

### What agreement and disagreement mean

Agreement across architectures increases confidence that a concept is not an
artifact of one representation. Disagreement is also informative: a lexical
model may find an explicit laboratory term that an embedding model smooths away,
while an embedding model may recognize a concept expressed through many
paraphrases. Stage 2 receives the differences intact rather than a single
global importance ranking.

## Selecting architectures

The selector is `science.stage1_architectures` in the configuration file, or
`--architectures` on the command line (`STAGE1_ARCHITECTURES` for
`run_synthetic.sh`).

| Selector | Architectures that run and enter the handoff |
|---|---|
| Omitted | The architectures enabled in the Stage 1 template `example_configs/all_evidence_stage1_model.json`. With the bundled template these are the eight marked "on" above, for binary and continuous outcomes. |
| `all` | All ten |
| Comma-separated list, for example `bow_nuisance,tfidf_topics` | Exactly those, in canonical order |

Rules that apply to an explicit selection:

- **Frozen for the run.** The selection is recorded in
  `stage1_architectures/manifest.json` and must match on every later command
  for the same output directory; a different selection is rejected. A selector
  cannot be added to an output directory that already contains completed work
  from a run without one. Use a different output directory to change the set.
- **Private prerequisites.** Some architectures need another architecture's
  models. For example, the embedding architectures need nuisance predictions
  from a text model, so selecting only embedding architectures still fits the
  bag-of-words nuisance models (or the HTR models when bag-of-words is disabled
  in the template), `matched_pair_uplift` needs the bag-of-words views and
  their nuisance propensities, and `neural_query_moments` needs the embedding
  cache.
  Prerequisites run as private support; only the selected architectures'
  evidence enters the handoff and Stage 2.
- **Availability checks.** Naming either optional architecture enables it. A
  named architecture that its template settings disable (for example,
  `tfidf_semantic_retrieval_contrasts` when embedding contrasts are disabled, or
  `htr_neural` with `--disable-htr`) is an error.
- **Per-architecture sidecars.** The handoff step writes each selected
  architecture's evidence to `stage1_architectures/<id>/evidence.jsonl`, with
  provenance back to the producing component, and fails if a selected
  architecture produced no evidence.

Without a selector, the handoff contains the raw evidence of every enabled
architecture and no per-architecture sidecars are written until the evaluator
needs them.

## Oracle-safe evaluation

For synthetic cohorts, `oci-evaluate-stage1` compares each frozen architecture's
outputs with the data-generating truth, without refitting or reselecting
anything:

```bash
uv run oci-evaluate-stage1 \
  --run-dir /path/to/output \
  --architectures all
```

| Option | Meaning |
|---|---|
| `--run-dir` (required) | A run directory with a completed handoff |
| `--architectures` | Comma-separated IDs, or `all` (default) for every architecture present in the run |
| `--dataset` | The dataset with [oracle columns](glossary.md#oracle-columns); defaults to the path in the run's `run_config.json` |
| `--metadata` | The generator's `metadata.json`, which names the true confounders and modifiers; defaults to `metadata.json` beside the dataset |
| `--output-dir` | Defaults to `<run-dir>/evaluations/stage1` |

The evaluator hashes every Stage 1 evidence and score file it will use before it
opens the oracle-bearing dataset, keeps the loaded copies in memory, and
records the hashes in `evaluation_manifest.json`. It never changes pipeline
checkpoints or completion markers. If the run has no per-architecture
sidecars, it builds them from the handoff first.

Metrics are computed on outer held-out rows where row-level scores exist. They
include:

- **Recovery:** whether evidence text mentions the true features (best lexical
  token recall), and how strongly row-level scores associate with each true
  feature's value;
- **Nuisance quality:** the treatment-prediction AUC of nuisance scores;
- **Effect quality:** the best normalized R-loss gain and the association of
  effect scores with the true individual effect;
- **Stability:** overlap of evidence across inner folds, plus
  architecture-specific measures such as topic, query, or cluster stability and
  retrieval coverage;
- **Matched pairs** (`matched_pair_uplift`, read from the frozen text-model
  evidence, which is hashed with the other inputs): the numbers of matched and
  discordant pairs, the counts of pairs in which only the treated member
  (`n10`) or only the control (`n01`) had the event, the McNemar log odds ratio
  `log(n10/n01)` (binary) or the mean adjusted paired difference (continuous),
  the mean and maximum within-pair `|Δê|` and `|Δμ̂0|`, the held-out gain over
  the constant-effect model, and the number of nonzero terms, per context.

Outputs: `metrics.jsonl` (all metrics), `architectures/<id>/metrics.jsonl`,
`comparison.csv` (one summary row per architecture and metric), `summary.json`,
and `evaluation_manifest.json`. See [Synthetic data](synthetic_data.md) for the
oracle columns and the bundled cohorts.

## References

- Breslow NE, Day NE. *Statistical Methods in Cancer Research, Volume I: The
  Analysis of Case-Control Studies.* IARC Scientific Publications No. 32. Lyon:
  International Agency for Research on Cancer; 1980.
- Hansen BB. The prognostic analogue of the propensity score. *Biometrika*
  2008;95(2):481–488.
