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
| 3 | `matched_pair_uplift` | on (binary outcomes only) | Text that separates outcomes within matched treated/untreated pairs | Ranked terms and attended text chunks |
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
[R-loss](glossary.md#r-loss).

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

### 3. Matched-pair uplift (`matched_pair_uplift`)

Treated patients are matched one-to-one to untreated patients whose
cross-fitted propensity and outcome probability both lie within a caliper (0.05
by default), using a maximum bipartite matching. A bag-of-words model and an
[HTR](glossary.md#htr) model then predict each treated patient's outcome with the
matched control's predicted risk as an offset, so the text features explain the
difference within pairs. The architecture asks what differs in the records of
otherwise similar patients who received different treatments. Matching balances
only what the nuisance models capture, so the result is evidence for a candidate
characteristic, not an individual treatment effect. It runs only for binary
outcomes.

### 4. Hierarchical transformer (`htr_neural`)

The hierarchical transformer splits a record into overlapping chunks (96 words
with a 24-word overlap by default), encodes each chunk with a small pretrained
encoder, and combines the chunks with a document-level transformer. Separate
heads learn treatment, outcome, and effect objectives; the effect heads use
nuisance predictions fitted and frozen inside the effect-training fold. Its
attention identifies the chunks the model relied on. Unlike a bag-of-words
model, it can distinguish some uses of the same vocabulary and combine
information spread across a long history. Attention describes what the model
used, not a causal explanation. `--disable-htr` turns off this architecture and
the HTR half of `matched_pair_uplift`.

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
| Omitted | The architectures enabled in the Stage 1 template `example_configs/all_evidence_stage1_model.json`. With the bundled template these are the eight marked "on" above, or seven for continuous outcomes because `matched_pair_uplift` requires a binary outcome. |
| `all` | All ten. Because `matched_pair_uplift` requires a binary outcome, `all` is rejected for a continuous outcome; list the other nine instead. |
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
  in the template), and `neural_query_moments` needs the embedding cache.
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
  matching or retrieval coverage.

Outputs: `metrics.jsonl` (all metrics), `architectures/<id>/metrics.jsonl`,
`comparison.csv` (one summary row per architecture and metric), `summary.json`,
and `evaluation_manifest.json`. See [Synthetic data](synthetic_data.md) for the
oracle columns and the bundled cohorts.

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
