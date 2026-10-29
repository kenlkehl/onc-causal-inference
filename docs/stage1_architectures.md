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
files that the [oracle-safe evaluator](#oracle-safe-evaluation) reads; for
architectures 5 and 6 these hold each context's held-out patients projected on
exactly the directions whose excerpts enter the handoff. Every component also
records the wall time of each context's sections, attributed to the
architecture they serve (see [Cost per architecture](#cost-per-architecture)).

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

The effect models are cross-fitted on `effect_folds` folds of the training
patients. The ridge penalty of each penalized linear effect model (DINA, on
every view whatever its learner, for binary outcomes; a linear view's
pseudo-outcome and weighted R-loss ridge regressions for continuous outcomes)
is chosen by cross-validation over `bow_effect_ridge_alpha_grid` (`ridge_alpha`
from 0.1 to 1000 by default; the DINA regularization is `ridge_alpha / n`).
Every effect fold fits each value, and the value with the lowest loss on the
effect folds' held-out patients, pooled over the folds, is kept: the DINA
loss (scored with nuisances fitted without those patients), the pseudo-outcome
squared error, or the R-loss. Ties go to the larger penalty. The context's
held-out patients never enter the choice; the training patients' out-of-fold
effect predictions and the in-context metrics built from them use the chosen
value. The full-training-patient fit whose coefficients rank the view's terms
uses the same value, and each view's evidence records the grid, the pooled
held-out losses, and the chosen value under `effect_penalty` (a choice at an
end of the grid is also logged as a warning). An empty grid uses each view's
fixed `ridge_alpha`; tree regressors have no penalty.

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
to 12 per side, at most 2 per patient). Retrieval skips chunks with fewer than
12 letters or digits: the empty chunk of an empty record, a short tail that the
previous chunk's overlap already contains, or a run of symbols such as soft
hyphens. The directions themselves use every chunk. Both tails are evidence: a
scalar or monotone characteristic often appears as an opposition between the
two tails rather than as a cluster of similar positive excerpts. The retrieved
chunks are what make a numerical direction interpretable; they let a reader
judge whether the direction reflects a coherent patient characteristic.

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
useful activation and diversity among queries. Queries see only chunks with at
least 12 letters or digits, the rule embedding retrieval uses. A record without
such a chunk, for example an empty record, takes the mean activation of the
other patients scored with it: a neutral value that no query can be fitted
toward and that adds nothing to the moment statistics. It never contributes
witnesses or background chunks. A patient's activation for a query is a smooth
maximum of the cosine similarities `s_j` between the patient's `n` remaining
chunks and the query: the log-mean-exp `τ · (log Σ_j exp(s_j / τ) − log n)`,
with temperature `τ` (`temperature`, 0.05 by default). It equals the common
similarity when every chunk matches equally and approaches the best chunk's
similarity minus `τ · log n` when one chunk dominates, so a passage found in
only a few chunks activates a long record less than a short one, although a
long record also has more chances to contain a closely matching chunk. The
objectives and held-out statistics use this activation.

Witnesses are chosen by the single best chunk instead: the patients whose
best-matching chunk has the highest cosine (ten by default) contribute that
chunk (one per patient per query by default). The query's phrases are the 30
terms (`evidence_ngram_top_k`) with the largest positive TF-IDF contrast
between those chunks and the other training patients' best-matching chunks;
`term_selection` in each query's evidence records how many terms had a
positive contrast. Queries can learn several distinct detectors for the same
objective, which a single mean-difference direction cannot.

Each bank is fitted in two steps within a training context:

1. **Candidates.** The context's training rows are split into inner folds
   (`query_inner_folds`, 5 by default). On each inner fold the bank is fitted
   on the other folds' rows, and every candidate query is then scored on the
   rows it did not see: a held-out standardized moment statistic (z) for the
   bank's objective and its two-sided p-value.
2. **Consensus and refit.** All candidates of a bank are grouped by k-means on
   their patient activation patterns into exactly as many groups as the bank
   has queries. Each group's medoid initializes one query, and the bank is
   refitted on all of the context's training rows.

Groups are ordered by the median absolute held-out z of their members, then
by the median absolute training statistic, then by the medoid candidate's ID.
A group whose members have no held-out statistic follows every group that has
one, and the number of such groups is recorded. The order only sets the query
IDs (`<bank>_query_001` ranks first): no query is gated or dropped, so each
bank keeps its configured number of queries. The held-out rows of this step
are inner partitions of the context's own training rows; the context's
held-out rows never enter it. Each query's evidence records its rank, both
medians, the median held-out p-value, and every member's held-out z, held-out
p-value, and training statistic (`member_heldout_statistics`);
`components/neural_queries/<context>/query_records.json` holds the same values
for every group and member.

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
| `--output-dir` | Defaults to `<run-dir>/evaluation/stage1` |
| `--null-draws` | Random-chunk null draws per architecture and context for the lexical recovery metric (default 200; 0 disables the null) |

The evaluator hashes every Stage 1 evidence, score, split, and timing file it
will use before it opens the oracle-bearing dataset, keeps the loaded copies in
memory, and records the hashes in `evaluation_manifest.json`. It never changes
pipeline checkpoints or completion markers. If the run has no per-architecture
sidecars, it builds them from the handoff first.

Metrics are computed on outer held-out rows where row-level scores exist. They
include:

- **Recovery:** whether evidence text mentions the true features (best lexical
  token recall, compared with a random-chunk null below), and how strongly
  row-level scores associate with each true feature's value. For the embedding
  architectures the row-level scores are the held-out patients' mean and
  maximum chunk cosine with each handed-off direction (columns
  `direction__<contrast>__mean_chunk_cosine` and `__max_chunk_cosine`), and
  `handed_off_direction_score_coverage` reports the share of an outer fold's
  handed-off directions that have scores;
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

### Random-chunk null for lexical recovery

Long excerpts contain many words, so an architecture that emits long text can
reach a high token recall by chance. For each architecture and discovery
context (an outer fold's full training set or one of its inner training
folds), the evaluator draws `--null-draws` sets of random text windows from
that context's training rows (from `components/tfidf/split_provenance.jsonl`).
Each set has one window per evidence item of the architecture in that context,
with the same number of characters; a window starts at a word boundary of a
training row long enough to hold it (an item longer than every training row
takes a whole row, counted as `length_unmatched_items`). The same best token
recall is computed on every set. The draws are seeded from the run seed, the
architecture, and the context, so a repeated evaluation reproduces them. For
each true feature and context the evaluator reports the observed value, the
null mean, the null 95th percentile, and the empirical p-value
`(1 + #{null ≥ observed}) / (draws + 1)`.

### Cost per architecture

Each Stage 1 context records the wall time of its sections in
`components/<component>/<context>/timing.json`, and each component collects its
contexts in `components/<component>/timing.jsonl`
(`components/embedding_cache/timing.jsonl` holds the one cache build). A
section that serves one architecture is attributed to it; a section that
several architectures need, such as the bag-of-words nuisance fits, the
nuisance ensemble, the TF-IDF nuisance stacks, and the embedding cache, is
attributed to `shared` and lists its consumers. A context whose sections
restored checkpoints is marked `partially_restored` with the restored units; a
context restored whole keeps the record written when it was computed; a
context restored from a checkpoint written without timing is marked
`unknown`, and the evaluator reports a completed component without a
`timing.jsonl` the same way. Timing never enters a checkpoint identity or any
fitted value.

### Outputs

The evaluator writes to `<run-dir>/evaluation/stage1/` (or `--output-dir`):

| File | Content |
|---|---|
| `metrics.jsonl` | Every metric row, including the null comparisons (`oracle_feature_recovery_null`) and per-context wall seconds (`cost`) |
| `architectures/<id>/metrics.jsonl` | The same rows for one architecture |
| `comparison.csv` | One summary row per architecture, metric, and target (mean, standard deviation, and count over estimates) |
| `lexical_recovery_null.csv` | One row per architecture, context, and true feature: observed recall, null mean, null 95th percentile, empirical p-value, draws, evidence items, length-unmatched items, and seed |
| `timing.csv` | One row per timed section and context: component, context, status, unit, attribution, consumers, and seconds |
| `summary.json` | Counts, the comparison table, the null settings, and the timing summary (per-architecture and shared seconds, context status counts, and contexts with unknown timing) |
| `evaluation_manifest.json` | Input paths and hashes, the oracle columns read, and the number of null draws |

See [Synthetic data](synthetic_data.md) for the oracle columns and the bundled
cohorts.

## Attribution and ablation

Two post-hoc tools measure what each architecture contributes to Stage 2. Both
read frozen outputs and change nothing a run reads; the ablation tool only
prepares inputs for separate Stage 2 runs.

### Attribution of Stage 2 variables

`oci-evaluate-attribution` credits each Stage 2 variable to the architectures
whose evidence supported it, fold by fold:

```bash
uv run oci-evaluate-attribution --run-dir /path/to/output
uv run oci-evaluate-attribution --run-dir /path/to/output --oracle   # synthetic cohorts
```

It follows the lineage Stage 2 records. Each evidence card lists its
`source_architectures` (`stage2/evidence_compilation/outer_00k/cards.jsonl`);
each candidate cites the cards it was proposed from
(`outer_00k/interpreted_candidates.json`); and each defined variable lists its
cards and, through `candidate_dispositions`, its candidates
(`outer_00k/feature_definitions.json`). An architecture supports a variable
when it produced one of the variable's cards or one of its candidates' cards,
or when the variable lists it in `supporting_architectures`. Variables created
later inherit lineage: a variable split by the extraction review, or an
estimand-search variant, from its parent definition; a consolidated latent from
all of its measurement dependencies. A variable is **sole-source** when exactly
one architecture supports it and it is not investigator-specified
(`stage2.explicit_features`); every supporting architecture gets
**any-source** credit.

For each fold, in total, and for each architecture, the report gives the
any-source and sole-source count at each level:

| Level | Items |
|---|---|
| `cards` | Evidence cards compiled for the fold |
| `candidates` | Candidates proposed by interpretation |
| `variables_defined` | Variables in `feature_definitions.json` |
| `variables_passing_filter` | Variables kept by the informativeness and extraction-failure filter (`extraction/candidate_missingness_filter.json`, and the estimand-search filter when it ran) |
| `nuisance_set` | Selected variables with the `confounder` role (`final_definitions.json`) |
| `selected_modifiers` | Selected variables with the `effect_modifier` role |

Each level also counts its items with several supporting architectures, its
investigator-specified items, and items without recorded lineage. Every card
and candidate comes from one architecture, so credit starts to be shared where
candidates merge into variables.

**Oracle recovery (`--oracle`).** For synthetic cohorts, the report then
measures which data-generating variables the selected variables capture. It
first reads and hashes every Stage 2 artifact it uses and closes its reader;
only then does it load the `true_<variable>` columns named in the generator's
`metadata.json` (without metadata, every `true_*` column except the outcome-model
and effect columns such as `true_ite`). On each fold's outer held-out rows, it
regresses each oracle column on each variable's columns from the frozen
held-out extraction (`estimation/nuisance_candidate_heldout.csv`, encoded as in
the Stage 2 diagnostics: a numeric value or one indicator per level, with
not-documented, documented-unknown, and extraction-failure indicators) by
ordinary least squares with an intercept. A categorical oracle column is
one-hot encoded, and its R² is 1 − ΣSSE/ΣSST over its levels. A variable
recovers an oracle concept when R² ≥ `--recovery-r2`; a variable's name plays
no part. The report lists, per fold and architecture, the concepts recovered
by the selected variables it supported, any-source and sole-source, for all
selected variables, the nuisance set, and the modifiers. The R² is computed on
the same held-out rows it describes, so it is slightly optimistic for variables
with many levels.

| Option | Default | Meaning |
|---|---|---|
| `--run-dir DIR` | required | Workflow output directory, or an ablation directory written by `prepare` |
| `--oracle` | off | Add oracle recovery |
| `--recovery-r2 R` | 0.5 | Held-out R² at which a variable recovers an oracle concept |
| `--dataset PATH` | `input/analysis_cohort.parquet` when present, else `dataset` from `run_config.json` | Dataset with oracle columns (`--oracle`); for an ablation directory, its source run's |
| `--metadata PATH` | `metadata.json` beside the run's dataset | Generator metadata (`--oracle`) |
| `--out DIR` | `RUN_DIR/evaluation/attribution` | Output directory; inside the run directory it must be under `evaluation/` |

Outputs: `attribution.json` (all counts, and recovery with `--oracle`),
`attribution_counts.csv` (one row per fold, level, and architecture),
`variables.csv` (each variable's cards, candidates, architectures, and levels),
`attribution_manifest.json` (the SHA-256 of every Stage 2 artifact read, and
the oracle inputs), and with `--oracle`, `oracle_recovery.csv` (recovered
concepts per fold, level, and architecture) and `oracle_recovery_r2.csv`
(every variable–concept R²).

### Leave-one-architecture-out ablation

`scripts/ablate_stage1_architectures.py` measures what Stage 2 loses without
each architecture by rerunning Stage 2 on the run's handoff minus that
architecture's evidence. Its `prepare` step writes the inputs and prints the
commands; it never starts an LLM request. Each ablation is a
[`run_stage2_from_artifacts.py`](configuration.md#stage-2-in-a-separate-directory-from-preserved-stage-1-artifacts)
launch with the run's split file, dataset, Stage 2 settings, and seed.

```bash
python scripts/ablate_stage1_architectures.py prepare \
  --run-dir /path/to/output --out /path/to/ablations \
  --stage2-config endpoints.json --include-control
# run the printed launcher commands, then:
python scripts/ablate_stage1_architectures.py compare \
  --run-dir /path/to/output --ablation-dir /path/to/ablations --oracle
```

`prepare` writes `without_<architecture>/` for each architecture of the run, or
for those named by `--architectures`. The run's architectures are its explicit
selection (`stage1_architectures/manifest.json`), else the architectures its
Stage 2 compiled, else those with evidence in the handoff.

| File | Content |
|---|---|
| `handoff/evidence.jsonl` | The run's handoff without the architecture's evidence |
| `handoff/index.json` | The remaining architectures, and the component evidence files Stage 2 reads to check the split file |
| `components/tfidf/split_provenance.jsonl` | Copy of the run's split file |
| `components/embedding_cache` | Link to the run's embedding cache, when the run has one, so evidence cards are grouped as in the run |
| `stage2_from_artifacts.json` | Launcher configuration; Stage 2 writes to `stage2/` and the launcher's status files go to `stage2_report/` |
| `ablation.json` | Row counts, the filtering rule, the launch commands, and the SHA-256 of every input |

`ABLATION_DIR/ablations.json` indexes the prepared and skipped ablations.

Filtering works on architecture rows. In a handoff written for an explicit
selection, each row holds one architecture's evidence, so the ablated
architecture's rows are dropped and every other line is copied unchanged.
Without a selection, the handoff streams component rows, and one text-model row
can hold several architectures' evidence (BoW nuisance and R-loss views,
matched pairs, embedding contrasts, HTR). `prepare` then first projects the
handoff into one row per architecture occurrence with the extractor Stage 2
uses, writes the projection to `ABLATION_DIR/projection/`, and drops the
ablated architecture's rows, so the other architectures keep their evidence
from shared rows. The projection compiles to the same evidence cards as the
original handoff. In each launcher configuration, `stage2.included_architectures`
lists the remaining architectures and `stage2.required_architectures` those of
them present in every outer fold. An ablation that would leave an empty handoff
is skipped with a message.

Stage 2's LLM steps vary from run to run, so an ablation's differences from the
full run combine the removed evidence with that variation. `--include-control`
adds `full_control/`, a rerun with every architecture, which measures the
variation alone.

The Stage 2 settings come from the run's `run_config.json`. `--stage2-config`
names a JSON file whose `stage2` block is merged over them, and which may also
set `concurrency_control_path` and `tokenizer_cache_dir`. Use it to point at
running servers: the launcher does not start managed vLLM pools, and an
endpoint in the template replaces the run's `vllm` block. API keys stay
`<redacted>` unless the template sets them; the launcher then reads
`OCI_STAGE2_API_KEY`, `OCI_STAGE2_EXTRACTION_API_KEY`, and
`OCI_STAGE2_EXTRACTION_RUNTIME_API_KEY` when it starts. A key in the template
is copied into every launcher configuration, so prefer the environment
variables. The dataset is `input/analysis_cohort.parquet` when the run used
dated notes, else the run's dataset; a CSV dataset is written to
`ABLATION_DIR/shared/stage2_cohort.parquet` with only the four declared
columns, because the launcher reads Parquet.

`compare` reads each finished ablation (one whose `stage2/` holds
`causal_estimate.json` and `heterogeneity_diagnostics.json`) and lists the
others as pending, with the launcher's last status. Fold by fold, it matches
the ablation's selected modifiers, and separately its nuisance set, to the full
run's by concept with the cross-fold matching rule of the
[outer-fold confirmation](stage2_selection.md#outer-fold-confirmation)
(TF-IDF similarity of name, description, and measurement definition, threshold
0.7, at most one variable per side in a concept), and reports the Jaccard
overlap of the concepts (1 when both sets are empty) and the concepts lost and
gained. It also reports the ATE and ATO with their 95% CIs, the pooled BLP β2
with the primary heterogeneity test's p-value, per fold where available, and,
with `--oracle`, the oracle concepts each side's selected variables recover, as
in the attribution report. It writes `comparison.json`, `summary.csv` (one row
per ablation, with the full run as reference), `folds.csv` (one row per
ablation and fold), and `summary.md` to `ABLATION_DIR/comparison/`, or to
`--out`. `compare` accepts the same `--oracle`, `--recovery-r2`, `--dataset`,
and `--metadata` options as `oci-evaluate-attribution`.

## References

- Breslow NE, Day NE. *Statistical Methods in Cancer Research, Volume I: The
  Analysis of Case-Control Studies.* IARC Scientific Publications No. 32. Lyon:
  International Agency for Research on Cancer; 1980.
- Hansen BB. The prognostic analogue of the propensity score. *Biometrika*
  2008;95(2):481–488.
