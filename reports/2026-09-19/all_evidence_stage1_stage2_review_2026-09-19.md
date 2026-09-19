# Stage 1 → Stage 2 all-evidence workflow review — 2026-09-19

1. **Scope, interpretation, and principal findings**
    1. **Review scope.** This is a review of the current repository working tree at Git base `aca1828`, including the existing local modifications and untracked Stage 2 support modules. It traces the supported `scripts/run_all_evidence.py` workflow from cohort input through discovery, measurement, selection, estimation, checkpointing, and evaluation. It also examines existing saved synthetic-run summaries. Source locations refer to the working tree reviewed on September 19, 2026.
    2. **Overall assessment.** The workflow has a substantial foundation for interpretable causal modeling: shared folds, complementary text representations, traceable evidence, explicit measurement definitions, isolated patient extraction, protected outer evaluation, and separate nuisance and effect models. The largest opportunities for better CATE estimation concern which information survives discovery and selection, how accurately it is measured, and how nuisance and effect models are validated.
    3. **Highest-priority findings.**
        - **Evidence recall:** “all evidence” means all selected architecture lanes and all compiled cards; the interpretation model does not read every raw evidence member.
        - **Modifier recall:** a nonlinear final forest is preceded by predominantly linear interaction screens. A modifier rejected there cannot be recovered by the forest.
        - **Selection stability:** retaining a variable selected in any inner fold favors sensitivity, but can produce large, unstable feature sets.
        - **Estimand:** a patient-level forest prediction is an estimated conditional average effect. It does not identify that patient's realized counterfactual outcome difference.
        - **Validation:** the current outputs support much stronger evaluation of CATE calibration, ranking, subgroup effects, and uncertainty than is presently reported.
    4. **Strength of evidence.**
        - **Implemented behavior** below is grounded in the active call chain and configuration, rather than inferred solely from documentation or module names.
        - **Observed results** are read-only summaries of saved runs, with prediction hashes checked against their post-hoc evaluation records.
        - **Recommendations** are proposed experiments or changes. Their improvement in causal accuracy has not been demonstrated by this review.
        - Sixty focused tests passed. No new full-cohort discovery, GPU training, LLM extraction, or production Stage 2 run was launched. Details appear in item 14.

2. **Scientific objects and the end-to-end structure**
    1. **Inputs and notation.**
        - Each row supplies pretreatment clinical text, a unit identifier, binary treatment `T ∈ {0,1}`, and outcome `Y`, declared binary or continuous.
        - Let `Z` denote all measured adjustment information, `e(Z) = P(T=1 | Z)`, and `m(Z) = E[Y | Z]`.
        - Let `μt(Z) = E[Y | T=t, Z]`. These treatment-specific outcome regressions differ from the marginal outcome regression `m` used for residualization.
        - The intended CATE is `τ(x) = E[Y(1) − Y(0) | X=x]`. The ATE averages causal effects over a specified population.
        - For binary outcomes, the effect scale is a difference in outcome probabilities, with direction determined by the outcome coding. A positive effect does not automatically mean clinical benefit.
    2. **Identification requirements.** Consistency, adequate pretreatment confounding measurement, positivity, and the appropriate independence/interference assumptions remain scientific requirements. Predictive associations, semantic plausibility, LLM causal labels, and cross-validation do not establish these conditions.
    3. **Two distinct transformations.**
        - Stage 1 transforms text and training-fold treatment/outcome information into evidence about potential patient characteristics.
        - Stage 2 transforms that evidence into scalar measurements and then estimates effects from those measurements.
        - In this supported final path, raw Stage 1 numerical predictions are not automatically stacked into the Stage 2 causal forest. Their principal contribution is discovery evidence. Existing helper classes or older workflows should not be mistaken for active final-estimation behavior.
    4. **Workflow map.**

        ```mermaid
        flowchart TD
          A["Cohort, configuration, shared folds"] --> B["Stage 1: ten evidence lanes"]
          B --> C["Frozen handoff and fold-local evidence cards"]
          C --> D["Interpret concepts; merge aliases; define ontologies"]
          D --> E["Extract all candidates on outer-training records"]
          E --> F["Repair and supervise measurements; freeze mappings"]
          F --> G["Optional measurement consolidation"]
          G --> H["Inner-fold numerical evidence and selection"]
          H --> I["Freeze selected definitions; extract outer-heldout records"]
          I --> J["External nuisances and AIPW ATE"]
          I --> K["Internal forest nuisances and CATE predictions"]
          J --> L["Pool outer-heldout outputs; diagnostics; post-hoc oracle evaluation"]
          K --> L
        ```

3. **Entry points, configuration, and fold design**
    1. **Supported launch path.**
        - `scripts/run_all_evidence.py` delegates to `ResearchAllEvidenceWorkflow`.
        - `run_one_conf_one_mod.sh` and `run_five_conf_five_mod.sh` wrap the synthetic examples, hardware selection, environment setup, and resume behavior through `scripts/run_synthetic_all_evidence.sh`.
        - `oci run` remains for retained explicit-feature workflows; it is not a second all-evidence orchestrator.
        - Without a configured Stage 2 endpoint or managed vLLM pool, the researcher workflow stops after the Stage 1 handoff. Dataset-backed Stage 2 also requires an extraction-model configuration.
    2. **Configuration resolution.**
        - JSON/YAML configuration supplies paths, column names, clinical question, outcome type, fold counts, seed, model locations, compute settings, and Stage 2 policies.
        - Stage 1 and neural-query overrides are merged into their templates. Paths are resolved relative to the configuration file.
        - `run_config.json`, `resolved_stage1_model_config.json`, and `resolved_neural_query_config.json` record resolved settings, with credential fields redacted.
        - An explicit architecture subset is frozen on resume. Dependencies can execute privately, while only selected lanes enter its architecture handoff. With no explicit selector, the resolved historical enable flags determine the lanes.
    3. **Shared outer and inner partitions.**
        - The canonical split plan is persisted under `components/tfidf/split_provenance.jsonl`, including when a targeted run does not otherwise need TF-IDF discovery.
        - Default splitting stratifies jointly on treatment and outcome when support permits. Continuous outcomes use outcome quantile bins. Small joint strata trigger a recorded shuffled-KFold fallback.
        - An explicit split registry can supply alternative partitions. The ordinary generated folds operate on rows; the unit identifier is not automatically a patient-grouping instruction.
        - With five outer folds and five inner folds, each outer fold has a full outer-training discovery context plus five exact inner-training discovery contexts: 30 discovery contexts in total, before deeper model-specific cross-fitting.
        - For a 1,000-row cohort, an ordinary outer context has 800 training and 200 held-out rows; each five-way inner training partition contains 640 of those 800 rows.
    4. **What fold protection does and does not mean.**
        - Architecture fitting and discovery use the permitted training rows. Held-out text can be transformed to make predictions without fitting to its labels.
        - Labels are used to stratify the split design; they must not subsequently guide outer-heldout feature discovery, role selection, or hyperparameter choice.
        - Stage 2 compiles the full and inner evidence for an entire outer fold before defining its common candidate set. Consequently, its inner screens evaluate an already adaptively discovered representation. Those inner p-values and rankings are exploratory evidence, not fresh confirmatory tests of independently specified hypotheses.
        - Outer-heldout predictions remain the main protection against evaluating the complete adaptive procedure on its own training outcomes.
    5. **Source anchors.** [Configuration and orchestration](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/research_all_evidence_workflow.py:386); [context construction](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/research_all_evidence_workflow.py:1063); [canonical split generation](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/tfidf_topic_stage1.py:135); [architecture registry](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage1_architectures.py:42).

4. **Stage 1: infrastructure and all ten evidence architectures**
    1. **Shared infrastructure.**
        - A frozen embedding model encodes clinical-record chunks. Cached chunk embeddings, offsets, text metadata, and patient representations can be reused across compatible components.
        - Computing a frozen, patient-specific embedding once does not itself fit a supervised model across folds. Supervised contrast directions, vocabularies, clusters, queries, and prediction models still need the appropriate training scope.
        - Text models build cross-fitted treatment and outcome predictions. Within the text-model producer, available nuisance views also feed an averaged nuisance ensemble used by several downstream discovery objectives.
        - Long-input tokenization and configured capacity checks aim to prevent silent loss of record content. Stage 2's later evidence-card compression is a separate operation with different limits.
        - Context checkpoints save expensive substeps and their row/configuration identities. Architecture outputs include readable evidence, numerical sidecars, diagnostics, and provenance.
    2. **Lane 1 — `bow_nuisance`: sparse treatment and outcome prediction.**
        - Fits configured bag-of-words/TF-IDF views with linear and tree-based predictors.
        - Learns treatment assignment and observed-outcome prediction separately, using training-only fitted vocabularies and cross-fitted predictions.
        - Exports influential words/phrases and prediction diagnostics for each task.
        - **Contribution to CATE:** helps discover adjustment and prognostic measurements. A treatment predictor alone is not proof of confounding; an outcome predictor alone is not proof of effect modification.
    3. **Lane 2 — `bow_r_loss`: sparse residual-effect evidence.**
        - Constructs treatment residuals `u = T − ê` and outcome residuals `v = Y − m̂`.
        - A residual-effect learner models `v ≈ u τ(text)`. The equivalent ratio-target form uses `v/u` with weight `u²`, with numerical protections in the producing path.
        - Sparse feature importance and effect predictions reveal words associated with this residual relationship. The producer also supports configured effect-objective variants, so the resolved settings are needed to interpret a particular result.
        - **Contribution to CATE:** surfaces candidate modifiers that ordinary prognostic modeling can miss. Small treatment residuals and binary-outcome noise can make discovery weak or unstable.
    4. **Lane 3 — `matched_pair_uplift`: matched-patient contrasts.**
        - Constructs treated/control comparisons satisfying propensity and predicted-outcome calipers. Training matching uses a maximum-cardinality bipartite matching procedure.
        - Sparse and HTR pair models learn contrast signals. The implementation includes offset-logit and ridge-difference machinery; the training-pair record contains the treated outcome and a matched control prediction as a baseline.
        - Exports matching coverage, pair-model information, and language associated with uplift signals.
        - This is not simply an observed treated-minus-control outcome difference relabeled as an individual causal effect. Binary-outcome restrictions affect availability; continuous-outcome runs do not enable this lane in the normal registry logic.
        - **Contribution to CATE:** offers another discovery view. Its usefulness depends on nuisance quality and actual covariate balance, which score calipers alone cannot guarantee.
    5. **Lane 4 — `htr_neural`: hierarchical transformer evidence.**
        - Splits long records into overlapping chunks, encodes them, and combines their ordered representations with a document-level transformer.
        - Fits nuisance and effect objectives within the context; encoder training, pooling, heads, calibration, and loss choices are controlled by the Stage 1 template.
        - Exports predictions plus attention/span/chunk evidence that can be translated back into readable clinical concepts.
        - **Contribution to CATE:** captures context that lexical counts miss. Attention is evidence of model use, not a causal attribution or guaranteed explanation.
    6. **Lane 5 — `embedding_whole_cohort`: semantic contrast directions.**
        - Uses frozen embeddings to form context-level treatment, outcome, combined treatment/outcome, residual-effect, and residualized interaction directions, according to configuration.
        - Some directions are marginal contrasts; others use nuisance residuals or pseudo-targets. They should not all be described as causally adjusted effect vectors.
        - Retrieves positive/negative text witnesses aligned with each direction. Optional specified residualization variables and retrieval settings affect the geometry.
        - **Contribution to CATE:** brings together paraphrases and diffuse clinical descriptions. A coherent direction still requires a valid patient-level measurement definition.
    7. **Lane 6 — `embedding_clustered`: local semantic contrast structure.**
        - Fits training-context patient clusters and constructs supported local treatment and treatment–outcome interaction contrasts.
        - The native implementation also combines local directions using SVD: rows contain support-weighted normalized local contrasts, and leading singular directions become retrieval axes.
        - Cluster-count, cell-support, and numerical-rank requirements can fail explicitly when the configured scientific geometry is infeasible.
        - **Contribution to CATE:** can expose local patterns cancelled in a whole-cohort contrast. A semantic cluster is not automatically a valid clinical subgroup.
    8. **Lane 7 — `tfidf_semantic_retrieval_contrasts`: lexical descriptions of semantic retrieval.**
        - Starts with text retrieved along the embedding contrast directions.
        - Uses TF-IDF terminology to describe distinctions between their sides, preserving the parent contrast and whether it came from global or clustered evidence.
        - **Contribution to CATE:** makes semantic evidence easier to operationalize. This lane shares upstream evidence with lanes 5–6; agreement is not three statistically independent replications.
    9. **Lane 8 — `tfidf_topics`: stable topic banks.**
        - Starts independently from fold-local text, fitting its own TF-IDF vocabulary and cross-fitted nuisance stack.
        - Screens vocabulary for treatment, outcome, and residual-effect evidence. An effect moment centers the residual relationship around a constant effect, using terms of the form `u(v − τ̂constant u)`.
        - Repeated NMF fits and component alignment construct consensus topic banks. Saved terms, loadings, stability measures, scores, and supported excerpts describe each topic.
        - Nested calibration contexts prevent an external held-out outcome from selecting the topic representation used to evaluate that outcome.
        - **Contribution to CATE:** finds recurring multivariate language patterns. A topic can contain several distinct variables, requiring decomposition in Stage 2.
    10. **Lane 9 — `tfidf_orphan_ngrams`: effect-language outside topic coverage.**
        - Begins with the TF-IDF branch's effect-associated n-gram inventory and removes terms already represented in its effect-topic inventory.
        - Retains eligible leftover phrases as separate evidence rather than requiring membership in a broad topic.
        - **Contribution to CATE:** offers a recovery path for precise or unusual modifier language, although support and upstream screening still limit recall.
    11. **Lane 10 — `neural_query_moments`: learned semantic detectors.**
        - Learns query vectors over frozen chunk embeddings, with separate treatment, outcome, and effect-oriented query banks.
        - Soft retrieval produces patient activations; training objectives seek informative cohort moments while controlling redundancy, activation scale, and movement from initialization/consensus anchors.
        - Nested subfold discovery, consensus construction, and final refitting produce query activations, moment diagnostics, and readable witness passages.
        - **Contribution to CATE:** adapts semantic detectors to weak heterogeneous signals. Numerical moments alone do not establish what clinical variable a query represents.
    12. **Important boundary.** The ten lanes generate candidate evidence. Their scores need not be on a common scale, and they are neither ten independent causal estimates nor a ten-model final CATE ensemble.
    13. **Source anchors.** [Active text-model producer](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/multi_model_forest_stage1.py:1686); [feature bundle](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/multi_model_forest_stage1.py:2029); [pair construction](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/multi_model_pair_uplift.py:164); [embedding discovery](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/embedding_contrast_discovery.py:193); [TF-IDF context](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/tfidf_topic_discovery.py:2023); [query discovery](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/neural_query_discovery_runtime.py:538).

5. **The Stage 1 → Stage 2 handoff and evidence compiler**
    1. **Plain handoff construction.**
        - Component evidence is assembled into `handoff/evidence.jsonl`; `handoff/index.json` records source references and configured cohort columns.
        - Explicit architecture runs materialize architecture-specific envelopes and a manifest. The omitted-selector route preserves the older combined format.
        - A completed handoff is required before Stage 2. Selected prerequisite computations do not authorize unselected evidence lanes to enter interpretation.
    2. **Scientific projection.**
        - `semantic_cluster_cards_v2` is the supported compiler. It reuses allowlisted evidence adapters from `all_evidence_fusion.py`.
        - Those adapters separate concept-bearing text from operational metadata and inappropriate numerical/row-level content.
        - Stage 2 checks required architecture coverage within each outer fold before making interpretation requests. This catches missing lanes, not missing clinical concepts within an otherwise present lane.
    3. **Deduplication and compression.**
        - Exact recurring evidence is aggregated within an outer fold, retaining links to its contexts and raw occurrences.
        - Members are stratified by evidence kind, axis, polarity, architecture, and embedding availability before clustering.
        - Compatible cached embeddings support semantic grouping; other evidence uses lexical projections. Cards remain architecture-specific.
        - Typical settings request 400 cards per fold, four exemplars per card, and 2,400 characters per exemplar. The card allocation preserves groups and can exceed the nominal target when needed to give each group representation.
        - Packet limits can remove exemplars or shorten representative text. Raw evidence and member/card lineage remain available outside the prompt.
    4. **What the LLM actually receives.**
        - Each prompt item contains a local integer and readable representative text.
        - Architecture labels, scores, direction labels, support counts, patient identifiers, fold labels, and raw provenance are not included in this discovery prompt.
        - Python maps returned `supporting_items` integers back to the original packets and their provenance.
    5. **CATE implication.** Exhaustively interpreting every card is not equivalent to exhaustive interpretation of all raw members. Rare modifiers can be absent from selected exemplars even if all architectures are present. This is a plausible information-loss mechanism to measure, not a finding that a particular modifier was lost.
    6. **Source anchors.** [Handoff writer](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/research_all_evidence_workflow.py:1439); [compiler](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_evidence.py:1626); [card construction and packet limits](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_evidence.py:1483); [prompt projection](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2.py:3679).

6. **Stage 2 discovery, alias consolidation, and ontology definition**
    1. **Concept discovery.**
        - Interpretation requests are grouped by architecture and bounded prompt batches.
        - The primary model enumerates atomic patient-level characteristics explicitly supported or reasonably implied by the supplied text. One card may support several variables or none.
        - At this point, it supplies names, descriptions, rationales, and evidence citations—not final causal roles or extraction schemas.
        - `interpreted_candidates.json` retains the assembled results. There is no active global top-K candidate gate between this interpretation and ontology construction.
    2. **Merge-only alias consolidation.**
        - Exact-name groups and model-proposed genuine aliases are merged while preserving their evidence and origin dispositions.
        - Shifted alphabetical batches and seeded shuffled batches give potential aliases repeated opportunities to meet. Example settings use batches of 20 and up to 55 rounds, including five alphabetical rounds.
        - Clinically distinct measurements should remain separate. Investigator-defined features have additional identity/definition protections.
        - A structurally invalid optional consolidation response can fall back to retaining the supplied candidates unchanged, with an audit record.
    3. **Operationalization.**
        - The primary model defines one extractable scalar ontology for each consolidated candidate, using its name and directly cited readable evidence.
        - Definitions specify value type, categories or units, measurement instructions, missing-value handling, and longitudinal conflict resolution.
        - Closed binary ontologies require exactly two distinct categories; categorical/ordinal ontologies require at least two. Invalid schemas trigger validation/repair rather than arbitrary coercion.
        - Conflicting longitudinal observations need rules such as latest, earliest, minimum, maximum, mode, any-positive, or single-or-null. When policy is inferred or defaulted, that fact is recorded.
    4. **Investigator-supplied features.**
        - `stage2.explicit_features` supplies complete measurement definitions and configured roles: confounder, effect modifier, or both.
        - These features bypass discovery and statistical inclusion gates. Their configured roles remain locked.
        - Force inclusion retains the input; it does not force a coefficient to be nonzero, guarantee that the text contains the measurement, or eliminate extraction error.
    5. **Temporal boundary.** `input_temporal_scope='pre_index_treatment'` is an upstream input requirement. The system does not infer the treatment index and remove post-treatment evidence by semantic judgment. Correct pretreatment record construction must happen before this workflow.
    6. **Source anchors.** [Outer-fold discovery](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2.py:7666); [merge-only consolidation](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2.py:7235); [operationalization](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2.py:7103); [explicit features](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2.py:503); [conflict rules](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_analysis.py:845).

7. **Stage 2 measurement: extraction, repair, supervision, and consolidation**
    1. **Outer-training extraction.**
        - A separately configured small model extracts all candidate features on outer-training records before supervised selection.
        - Each prompt contains exactly one patient's text. Feature batches, commonly ten definitions, control request size; patients do not share a clinical inference prompt.
        - Treatment and outcome values are not provided as extraction targets. Each result must cover the expected row and feature names with valid scalar values or explicit missingness.
    2. **Long records.**
        - With a tokenizer, records that exceed source-token, total-input-token, or character limits are processed in contiguous, ordered chunks.
        - Each chunk updates validated cumulative values plus a bounded per-feature carry-forward state. That state supports date/order comparisons and the declared conflict rule.
        - The planner reserves room for instructions, definitions, carried state, and output. It fails if even one source character cannot fit.
        - The older no-tokenizer fallback uses lossless character pages and validated quoted observations, then deterministic conflict resolution.
        - Ordinary scalar/serial extraction does not universally require a quoted source span for every final value. Complete text coverage and valid JSON therefore do not by themselves prove clinical measurement accuracy.
    3. **Repair versus missingness.**
        - Transport failures, malformed responses, category violations, scalar violations, and unsupported observations have distinct handling and audit records.
        - Category repair may map an extracted value to an existing declared category; unresolved scientific/validation failures may be recorded as null.
        - Infrastructure failures are not supposed to become ordinary missing patient measurements. The code can identify and supersede older checkpoints that conflated these conditions.
    4. **Ontology feedback and aggregate supervision.**
        - Repeated feature-specific failures on training patients can trigger bounded schema refinements; common example limits are at least three affected patients and two refinement rounds.
        - A primary-model supervisor sees aggregate extracted values and validation failures, without patient text, treatment/outcome values, causal performance statistics, or p-values.
        - It may revise the same candidate's measurement schema but cannot freely add, delete, rename, or assign causal roles to candidates.
        - Only changed prompt-facing definitions require re-extraction; unchanged columns can be reused.
        - If the final allowed review revises a definition, the implementation applies that revision through extraction before selection. Exhausting the review budget is recorded separately from achieving convergence.
    5. **Harmonization.**
        - Training aggregates guide explicit mappings for mixed numeric/categorical measurements, fallback strings, category normalization, or bins.
        - These mappings are frozen and applied to held-out measurements. Held-out clinical values should not be used to redesign the representation.
        - Extraction-health checks catch inadequate coverage; they are not substitutes for a chart-validated measurement study.
    6. **Optional second consolidation pass.**
        - This is distinct from the earlier text-only alias merge. It operates on extracted outer-training measurements before supervised selection.
        - A small embedding model retrieves nearby active feature definitions. Mixed-type association and row-agreement checks accompany an LLM decision about equivalent measurements.
        - The policy supports replacing disjoint sets of empirically concordant aliases with reconstructable canonical derived features, called latents in the implementation. It excludes broader composites and general/specific rollups.
        - Defaults include ten neighbors and a 0.85 minimum pairwise association when enabled. Association is necessary, not sufficient, for semantic equivalence.
        - Derived features enter the active retrieval pool; source columns and dependency state remain available for held-out reconstruction. This pass does not consume treatment/outcome labels.
    7. **Source anchors.** [Extraction entry point](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_analysis.py:3369); [serial prompts](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_analysis.py:1759); [harmonization](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_analysis.py:5246); [active analysis and supervision](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_analysis.py:9500); [measurement consolidation](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_sequential_consolidation.py:1461).

8. **Stage 2 numerical evidence and the two selection modes**
    1. **Common numerical preparation.**
        - Inner-training encoders impute continuous/ordinal values, standardize usable columns, encode nominal reference contrasts, pool rare categories as applicable, and retain feature identities across encoded columns.
        - Missingness indicators share the measurement's penalty group. Group penalties prevent a factor with many dummy variables from being selected merely by treating each level as an unrelated feature.
        - The selector uses a custom group-lasso-plus-ridge solver. This differs from the ordinary coefficient-wise elastic nets used in the final nuisance wrappers.
        - Default numerical settings include an L1/group share of 0.8, three internal CV folds, 16 regularization values, and an any-inner-fold union rule. Treatment and outcome models choose their own regularization strength.
        - The one-standard-error rule is enabled for nuisance screening by default; nuisance prediction and joint modifier fitting use minimum-loss choices by default.
    2. **Treatment and outcome screens.**
        - Separate grouped logistic models predict treatment and binary marginal outcome; continuous outcomes use squared-error models.
        - Nonzero group coefficients generate per-task fold votes. Prediction losses, calibration, selected penalties, convergence, and group coefficient norms are saved.
        - Candidate-wise treatment/outcome tests, including outcome adjusted for treatment, provide additional descriptive evidence. Benjamini–Hochberg adjustments are within the corresponding inner-fold test families.
        - These p/q values do not establish causal roles and do not account for the entire upstream adaptive discovery procedure.
    3. **Residual nuisance prediction.**
        - The legacy mode builds a treatment/outcome screen union across inner folds and uses it to restrict both base nuisance designs.
        - Independent-task mode instead makes all candidates available to each fold-local nuisance fit, with separately selected regularization. This removes the specific cross-fold screen-union gate.
        - Nested nuisance predictions on inner-training rows and predictions on inner-heldout rows support the effect screens. An ordinary out-of-fold model fit does not make an upstream selected representation fully independent of that fold.
    4. **Candidate-specific modifier evidence.**
        - For each candidate, the method augments/calibrates nuisance predictions with that candidate, using nested fits for training predictions.
        - It compares a ridge-stabilized reduced residual model against a model adding residual-treatment × candidate interactions, scored on the inner-heldout partition.
        - Continuous interactions are winsorized; missingness interactions are excluded. Categorical interactions need minimum support in both treatment arms.
        - The default provisional ranking retains up to ten evaluable candidates per inner fold, then unions them. It does not require a positive R-loss gain, so a top-N position can still represent unfavorable evidence.
    5. **Joint modifier evidence.**
        - A separate multivariable grouped elastic-net model fits interactions jointly across candidate groups.
        - With `u=T−ê` and `v=Y−m̂`, it first estimates a constant residual effect, centers candidate columns using `u²` weights, and penalizes the additional interaction terms.
        - The model records nonzero groups, held-out R-loss, support exclusions, and optimizer status. The same missingness and categorical-arm support protections apply.
    6. **Mode A — `selection_mode='llm_roles'` (default).**
        - Numerical decisions are provisional when role adjudication is enabled.
        - The primary LLM receives an allowlisted bundle of definitions and aggregate treatment/outcome, candidate-specific, and joint-interaction evidence.
        - It decides confounder, effect modifier, both, or neither, subject to investigator locks. It can retain candidates outside the provisional sets or decline provisional selections.
        - Its prompt explicitly distinguishes prognosis, treatment-only prediction, and plausible confounding, but a prompt cannot identify an adjustment set from observational associations alone.
        - If role adjudication is disabled, the statistical provisional decisions become the final routing decisions.
    7. **Mode B — `selection_mode='independent_tasks'` (opt-in).**
        - Treatment, outcome, and effect are three independently regularized selection tasks.
        - A group selected in any inner fold enters that task's support. Effect support comes from the **joint** R-loss model; neither nuisance support nor candidate top-N placement is a prerequisite.
        - LLM role interpretation, if enabled, is advisory. It cannot veto, promote, or reroute numerical selections, and annotation failures do not veto estimation.
        - No task support means exclusion, unless an investigator definition forces inclusion.
        - The retained `confounder`/`effect_modifier` fields serve as compatibility routing labels. The authoritative interpretation is in `selection_authority`, `modeling_tasks`, and `taskwise_routing`.
    8. **Actual downstream routing.**

        | Destination | Inputs in independent-task mode |
        | --- | --- |
        | External propensity model | Treatment-task features, with investigator overrides |
        | External treatment-specific outcome models | Outcome-task features **plus every selected effect modifier**, with investigator overrides |
        | Forest `X` | Effect-task features |
        | Forest `W` | Treatment/outcome feature union, excluding features already in `X` |
        | Forest internal nuisance models | Both independently regularize over `X + W`; there are no separate hard task masks inside EconML |

        - The outcome-model augmentation is an implementation detail easy to miss in the independent-task documentation. The review checked it directly with an effect-only feature.
        - The legacy `final_role_assignment='llm_all_evidence_adjudication'` string can still appear in independent-task artifacts. It names the executed branch, not binding selection authority.
    9. **Source anchors.** [Selection policy](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_elastic_net_selection.py:51); [candidate R-learner](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_elastic_net_selection.py:1254); [joint modifier model](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_elastic_net_selection.py:1688); [selector](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_elastic_net_selection.py:1824); [task routing](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_taskwise_policy.py:32); [role evidence](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_role_adjudication.py:362); [actual external nuisance inputs](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_analysis.py:5523).

9. **Final estimation: held-out measurement, overlap, CATE, and ATE**
    1. **Freeze first, then measure outer-heldout patients.**
        - Only selected measurements and the source dependencies of selected derived features are extracted from held-out text.
        - Training-derived harmonization and latent states are applied without relearning them from held-out outcomes.
        - The final training/held-out matrices retain row alignment and extraction-health audits.
    2. **Final encoding is a separate implementation.**
        - The estimator's `_FeatureEncoder` uses training medians/scales for continuous values and missingness columns; mixed continuous/fallback values have additional representation flags.
        - Closed categorical and ordinal definitions are expanded categorically. This differs from the selector's ordinal score, reference contrasts, standardization, and rare-level rules.
        - Effect-design and control encoders are fitted on eligible training rows when propensity bounds are enabled. External nuisance encoders use all outer-training rows.
    3. **External nuisance models.**
        - An elastic-net logistic model estimates propensity; two arm-specific elastic-net outcome models estimate `μ0` and `μ1`.
        - Binary outcomes use probability predictions; continuous outcomes use elastic-net regression. Degenerate designs/classes have specified constant-model fallbacks.
        - Separate inner out-of-fold fits supply training propensities for eligibility and diagnostic predictions. Full outer-training fits supply predictions on the outer-heldout patients.
        - These final wrappers choose their own penalties. Changing the selection policy's alpha grid or one-standard-error flags does not automatically change all final nuisance-model settings.
    4. **Propensity eligibility is different from denominator clipping.**
        - `min_propensity`/`max_propensity` default to no exclusion. With bounds, eligibility is determined using un-clipped external propensities; boundaries are inclusive.
        - Training eligibility uses inner out-of-fold propensity predictions. Held-out eligibility uses a model trained on the outer-training partition; held-out outcomes do not determine it.
        - Modifier fitting/scoring and final forest fitting respect configured eligibility. External final nuisance fits still train on all outer-training patients.
        - Excluded held-out rows stay in the prediction file, flagged `effect_eligible=False`, with missing CATE, interval, and AIPW values.
        - The estimand becomes the average effect in an estimated, fold-specific eligible population. Insufficient eligible rows/treatment arms or an empty eligible held-out set raises an error.
        - `propensity_clip`, typically 0.02, separately limits AIPW denominators. Clipping does not exclude rows or automatically repair poor overlap.
    5. **Honest causal forest for CATE.**
        - `CausalForestHead` wraps EconML `CausalForestDML`, using `X` for allowed effect variation and `W` for additional controls.
        - Its own cross-fitted nuisance models estimate marginal outcome and treatment on `X+W`; they do not reuse the external AIPW nuisance predictions.
        - Both internal nuisance families are cross-validated elastic nets, with a binary/continuous outcome contract and fitted-clone parameter/iteration audits.
        - The active call uses 200 trees by default, unlimited depth, minimum leaf size 10, `max_features='sqrt'`, honesty and inference enabled, one compute thread per forest, and **`tune_model=False`**.
        - The subforest size is chosen to divide the tree count. The wrapper does not explicitly pass an EconML residualization `cv`; the review environment's default is two folds. This is distinct from the workflow's five inner folds and the nuisance wrappers' three penalty-tuning folds.
        - If no modifier survives, a constant `X` design produces a fold-level constant effect. Additional `W` controls may still be used.
        - Held-out predictions include effect, standard error, and pointwise 95% interval when inference succeeds. The contract is explicitly treatment 1 versus treatment 0.
    6. **AIPW for the population average.**
        - For each eligible outer-heldout row, the code computes the following using external nuisance predictions and clipped propensity `ẽ`:

            ```text
            ψi = μ̂1(Zi) − μ̂0(Zi)
                 + Ti [Yi − μ̂1(Zi)] / ẽi
                 − (1−Ti) [Yi − μ̂0(Zi)] / (1−ẽi).
            ```

        - The pooled ATE is `mean(ψi)` over eligible rows. Its reported standard error is the sample standard deviation of those scores divided by the square root of their count; the interval uses ±1.96 standard errors.
        - Each cohort row must appear exactly once in pooled outer-heldout predictions. Outer-fold training records must not be pooled as if they were unique independent evaluation patients.
        - AIPW's double-robust interpretation requires identification and appropriate nuisance/regularity conditions. Fixed clipping can alter the usual consistency argument if the true propensity lies outside the clip bounds and the outcome models are misspecified.
    7. **Why forest CATE and AIPW ATE need not agree.**
        - The forest and AIPW paths use different nuisance fits, feature exposure, training populations, and objectives. Their averages are not forced to agree.
        - There is also a target issue if omitted effect variation remains in `W`. **Inference from the documented local-moment objective:** with correct nuisances and `Z=(X,W)`, a residual-slope estimator restricted to `X` targets

            ```text
            E[e(Z)(1−e(Z)) τ(Z) | X=x]
            --------------------------------,
                  E[e(Z)(1−e(Z)) | X=x]
            ```

        - This equals the ordinary conditional average effect when the treatment effect is adequately represented by `X`, or under other conditions removing the weighting difference. Otherwise it is a variance-weighted projection over omitted heterogeneity. The same issue matters for constant `X`. This is a mathematical implication, not a measured diagnosis of the saved cohort. [EconML local-moment specification](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html).
    8. **Interval interpretation.**
        - Honest trees separate split construction from within-tree effect estimation; they do not make the entire adaptive discovery pipeline unbiased in finite samples.
        - Forest intervals are intervals for an estimated conditional mean effect under the model's assumptions, not prediction intervals for a patient's realized `Y(1)−Y(0)`.
        - These intervals do not explicitly propagate all LLM sampling, extraction, candidate-selection, ontology, and split uncertainty. The ATE standard error also treats the learned eligibility boundary as fixed.
        - Averaging pointwise lower/upper CATE endpoints, as some diagnostics do, does not yield a valid confidence interval for mean CATE or ATE.
    9. **Source anchors.** [Final estimation](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_analysis.py:8431); [encoders and external models](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_analysis.py:5418); [forest construction](/data1/ken/pcori_dev/causal-dragonnet-text/oci/models/causal_forest_head.py:423); [AIPW calculation](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_analysis.py:5704); [pooled estimates](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2.py:8251).

10. **Diagnostics, evaluation, runtime, and resume behavior**
    1. **Current nuisance diagnostics.**
        - Binary diagnostics include Brier score, log loss, AUROC, observed/predicted means, mean error, reliability bins, and logistic calibration intercept/slope.
        - Continuous diagnostics include prediction error, MSE/RMSE, and linear calibration intercept/slope.
        - Propensity summaries include quantiles, extreme-score counts, eligibility counts, and clipping counts. Outcome diagnostics include factual predictions by observed treatment arm.
        - These reported calibration diagnostics are descriptive; they do not recalibrate the final nuisance models. Counterfactual outcome calibration is not directly observable in ordinary cohort data.
    2. **Stage 2 oracle evaluation.**
        - All outer predictions are written and hashed before the evaluator joins the default `true_ite_prob` column.
        - It reports Pearson/Spearman correlation, MAE, RMSE, mean error, predicted/truth means, and predicted/truth standard deviations, overall and by fold.
        - Excluded patients contribute no finite effect pairs. `n` can remain the total cohort size while `finite_pairs` is the actual effect-evaluation count.
        - In the synthetic generator, `true_ite_prob` is `P(Y(1)=1) − P(Y(0)=1)`: an oracle risk difference, not the realized difference between two binary potential outcomes.
        - **Naming issue:** this evaluator's `ate_bias` is mean **forest CATE** minus mean truth. It is not the bias of the separately reported AIPW ATE.
    3. **Stage 1 oracle evaluation.**
        - `scripts/evaluate_stage1_architectures.py` evaluates frozen architecture artifacts without refitting them.
        - It freezes/hashes source evidence and scores before loading oracle-bearing data, and reports architecture-appropriate nuisance, R-loss, lexical/semantic association, recovery, matching, and stability metrics where available.
        - Native scores and lexical recovery are useful for locating failure stages, but cannot by themselves establish downstream CATE accuracy.
    4. **LLM runtime.**
        - Primary interpretation/supervision and extraction requests use separate configurations and can use external OpenAI-compatible endpoints or pipeline-managed vLLM pools.
        - The serving layer handles model identity, bounded concurrency, per-attempt and logical-request timeouts, transport retries, JSON validation, and bounded response repairs.
        - Reasoning and sampling policies depend on request type and checked-in model profiles. Explicit overrides and resolved policy identities matter for reproducibility.
        - Managed pools can switch models or keep them resident on separate GPU allocations according to the serving configuration. These operational choices do not substitute for scientific selection settings.
    5. **Checkpoint and migration behavior.**
        - Ordinary files plus completion markers enable restart at component, context, request, extraction, selection, and estimation boundaries.
        - Fine-grained checks compare relevant schemas, rows, data/measurement fingerprints, policies, and model identities. Stage 2 revalidates selection/estimation inputs rather than trusting a coarse outer-fold completion marker alone.
        - Operational changes such as endpoint addresses and worker counts can be compatible with reuse; scientific changes may require new downstream work.
        - Guarded `--stage2-reselect` archives selection/estimation results and reuses compatible frozen preselection measurements. Newly required held-out measurements may still need extraction.
        - Saved-run launchers and `--preflight-only` validate compatibility without fitting or endpoint calls. Archives should be handled through these supported paths, not by manually copying completion markers.
    6. **Documentation boundaries.**
        - Older descriptions of simple p-value gates, active candidate registries, arbitrary composite latents, or a single all-purpose Stage 2 LLM do not fully describe this active path.
        - Some retained configuration fields are compatibility-only. Changing an inactive field may not change the science; inspect the serialized policy and active call site.
        - A status file saying “complete” can outlive a renamed output directory. Artifact presence and compatible fine-grained state are more informative than that label alone.
    7. **Source anchors.** [Nuisance diagnostics](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/nuisance_diagnostics.py); [post-hoc Stage 2 evaluation](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2.py:368); [Stage 1 evaluator](/data1/ken/pcori_dev/causal-dragonnet-text/oci/evaluation/stage1.py:1146); [guarded reselection](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/research_all_evidence_workflow.py:2351); [preflight](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_preflight.py); [sampling policy](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_sampling.py).

11. **What the saved runs currently show**
    1. **Read-only snapshot.** The table summarizes existing saved outputs, not newly fitted comparisons. Their populations and policies differ. The snapshot records exact source paths and SHA-256 hashes in [saved_run_evidence_2026-09-19.json](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/saved_run_evidence_2026-09-19.json).

        | Saved analysis | Effect pairs | Pearson / Spearman | CATE RMSE | Predicted / oracle effect SD | AIPW ATE | Oracle mean effect on evaluated rows |
        | --- | ---: | ---: | ---: | ---: | ---: | ---: |
        | One confounder/one modifier, archived LLM-role run | 1,000 | 0.420 / 0.267 | 0.165 | 0.086 / 0.163 | 0.119 | 0.063 |
        | Five confounders/five modifiers, archived LLM-role run | 1,000 | 0.297 / 0.271 | 0.206 | 0.077 / 0.181 | 0.027 | −0.074 |
        | Five confounders/five modifiers, active independent-task run, propensity 0.10–0.90 | 728 | 0.246 / 0.232 | 0.212 | 0.098 / 0.188 | 0.0067 | −0.0813 |

    2. **Active five-confounder/five-modifier analysis.**
        - It excludes 272 of 1,000 rows under its learned propensity bounds. The 728-patient target differs from the archived full-cohort target.
        - Its mean forest CATE is 0.0146, versus an oracle mean of −0.0813: a mean-CATE error of +0.0959. The separate AIPW ATE error is approximately +0.0880.
        - Predicted CATE standard deviation is about 52% of oracle effect standard deviation. This is consistent with compressed heterogeneity, but does not isolate its cause; selection, measurement, nuisance error, and forest regularization can all contribute.
        - Across folds, selected feature counts are **130, 121, 30, 130, 139**; effect-design feature counts are **28, 2, 2, 5, 35**. These are original feature counts, not expanded matrix dimensions.
        - Eligible held-out counts are **143, 111, 186, 158, 130**. Comparisons across these folds involve materially different fitted selection/eligibility behavior.
        - External treatment calibration slope is approximately **0.693**, and factual outcome calibration slope **0.651** on all 1,000 held-out rows. The below-one slopes flag prediction extremity/calibration concerns despite AUROCs of approximately 0.803 and 0.730; AUROC alone is insufficient for causal nuisance assessment.
    3. **Discovery and measurement observations.**
        - The active five-variable-pair run's compiler has approximately **693,000–744,000 exact evidence members per fold**, summarized into **400 cards per fold**. Raw occurrence counts are approximately 4.35–4.47 million per fold.
        - Those counts demonstrate substantial compression, not a measured clinical-feature recall rate. Four exemplars per card cannot expose every member individually.
        - Candidate pools entering selection range from 304 to 419 features. The optional post-extraction consolidation is disabled in these saved active/archived runs; it therefore has not been evaluated by this comparison.
        - The active and archived five-variable-pair runs mark all five ontology-supervision folds as nonconverged within their review budget. The archived one-variable-pair run marks folds 2–5 nonconverged. The final revisions were intended to be applied before fitting; nonconvergence alone does not establish invalid measurements.
    4. **What cannot be concluded.**
        - The table does not demonstrate that one selection mode is superior: policy, trimming, and evaluated population differ, and these are individual datasets/runs.
        - It does provide concrete reasons to investigate heterogeneity compression, feature-set instability, and nuisance calibration.
        - The one-variable-pair cohort currently has archived Stage 2 directories but no active `stage2/` directory, despite a historical “complete” progress record. Its table row is explicitly the `stage2_pre_roles_refactor` archive.
        - Frozen prediction hashes matched the oracle-evaluation records for all three reported analyses. This checks artifact consistency, not statistical validity.

12. **Prioritized opportunities to improve CATE/ITE estimation**
    1. **P1 — Measure modifier loss at each stage before changing the estimator.**
        - **Reason:** the final forest cannot recover a measurement that never reaches `X`; hundreds of thousands of evidence members are reduced before interpretation.
        - **Change:** create a recall audit from raw evidence → card exemplar → named candidate → ontology → extracted column → selected modifier. Preserve rare, diverse, and residual-effect witnesses through a coverage-oriented second pass or a reserved exemplar budget.
        - **Evaluate:** use synthetic truth only after artifacts are frozen, and independently annotated clinical concepts for real data. Compare card/exemplar budgets on the same training partitions and held-out evaluation protocol. Report cost per recovered valid modifier, not just candidate count.
        - **Code focus:** compiler allocation/exemplar selection, interpretation coverage, and post-hoc architecture evaluation.
    2. **P1 — Give nonlinear and interaction-only modifiers a route into `X`.**
        - **Reason:** continuous and ordinal modifier screens mainly examine linear interactions; joint grouped selection still uses a linear basis. Threshold, U-shaped, or pure pairwise modification can have little marginal linear signal.
        - **Change:** compare grouped spline/threshold bases, a small prespecified interaction library, and a broad-candidate forest with training-only regularization. Include a branch that admits scientifically eligible candidates to the effect learner without requiring a nonzero linear interaction.
        - **Evaluate:** known threshold, smooth nonlinear, and XOR-like effect simulations, including null-effect controls. Tune using training-only CATE criteria; compare PEHE/RMSE and false heterogeneity across independent replications.
        - A flexible second-stage residual learner is consistent with the R-learner framework; the present linear screen is a design choice, not a requirement of residualization. [Nie and Wager](https://arxiv.org/abs/1712.04912).
    3. **P1 — Improve nuisance approximation and calibration while preserving honest comparisons.**
        - **Reason:** linear elastic nets on scalar encodings can miss nonlinear prognosis and assignment mechanisms. The saved calibration slopes show that useful discrimination does not ensure suitable probabilities.
        - **Change:** first compare training-fitted spline/interaction expansions and penalty grids within the existing elastic-net constraint. Test calibration using strictly nested predictions, with the calibrator refitted inside each outer-training partition.
        - **Evaluate:** held-out log loss/Brier/calibration, propensity tails, treatment-arm support, and downstream CATE/ATE error jointly. A higher treatment AUROC is not an objective by itself, because increasingly deterministic assignment can worsen overlap.
        - Comparing flexible nuisance families would be a deliberate scientific-policy extension: the current repository guide and forest implementation explicitly restrict forest nuisances to elastic nets.
    4. **P1 — Define the target conditional effect explicitly and benchmark a DR learner.**
        - **Reason:** restricting heterogeneity to `X` while putting possible modifiers only in `W` can change the residual forest's target; forest CATE and AIPW ATE also use different nuisance paths.
        - **Change:** compare the current forest with a learner that regresses cross-fitted AIPW/DR pseudo-outcomes on the intended effect covariates. Construct its training scores entirely within the outer-training partition, then predict untouched held-out rows.
        - **Evaluate:** the same covariates, target population, and evaluation rows; examine both error and instability near poor overlap. A DR learner is a benchmark, not an assumption of superiority. [Kennedy's analysis of doubly robust CATE estimation](https://arxiv.org/abs/2004.14497).
        - Keep realized individual effects, oracle risk differences, conditional effects, overlap-weighted projections, and population averages distinctly named in outputs.
    5. **P1 — Validate extraction as a measurement model.**
        - **Reason:** extraction error can attenuate real effect modification, create artificial modification, or leave residual confounding. Schema validity and nonmissingness checks do not measure this error.
        - **Change:** require source spans and dates for high-impact measurements, validate deterministic units/ranges, and chart-review a stratified sample. Preserve unknown, undocumented, contradictory, and observed-negative states where scientifically appropriate.
        - **Evaluate:** sensitivity/specificity or numerical error by treatment arm, documentation intensity, subgroup, and fold. Compare errors for selected modifiers and adjustment features, not just overall extraction agreement.
        - Use known structured measurements as a post-hoc upper-bound benchmark on synthetic data. If stochastic extraction or missingness is propagated through multiple fits, describe it as a sensitivity procedure unless its uncertainty coverage is validated.
    6. **P1 — Align selection and estimation representations.**
        - **Reason:** ordinal scores in selection become categorical columns in final fitting; rare-level handling also changes. Missingness interactions excluded during modifier selection can reappear as forest split variables in `X`.
        - **Change:** define one explicit representation contract per feature, or document intentional differences. Decide separately whether documentation/missingness should be permitted to modify effects, and enforce that decision in the final design.
        - **Evaluate:** missingness-shift, rare-level, and nonmonotone ordinal simulations; inspect whether apparent clinical heterogeneity is predominantly driven by recording patterns.
        - Fit preprocessing within the deepest relevant training split where practical. Current internal penalty/nuisance subfolds can operate on arrays encoded at the enclosing training scope; outer protection is stronger than full end-to-end nested preprocessing independence.
    7. **P2 — Replace arbitrary sensitivity settings with measured selection stability.**
        - **Reason:** any-fold unions accumulate chance selections; candidate top-N can select negative gains in the legacy provisional route. The active run's effect count ranges from 2 to 35 across folds.
        - **Change:** compare any-fold union, repeated-split stability, nested selection on improvement over a constant-effect baseline, and broad-input shrinkage. Retain the independent-task mode as an ablation that isolates LLM routing from numerical routing.
        - **Evaluate:** feature/role consistency after aligning semantic definitions, downstream CATE stability, and null-effect false discoveries. Do not substitute an arbitrary high-frequency threshold without testing its loss of weak modifiers.
        - For strict inner validation, rebuild supervised discovery/selection within that inner training partition. At minimum, avoid interpreting cross-fold-selected representations as independently validated inner-fold hypotheses.
    8. **P2 — Tune the forest and report conditional support.**
        - **Reason:** the active estimator disables tuning and fixes leaf size and feature subsampling. With hundreds of possible encodings and variable eligible sample sizes, one configuration may oversmooth some runs and overfit others.
        - **Change:** expose training-only selection of leaf size, depth, feature subsampling, tree count, and appropriate local treatment-variation constraints. Explicitly configure and record internal residualization folds.
        - **Evaluate:** nested R-loss/DR criteria, stability across seeds, interval Monte Carlo stability, and local arm balance/effective sample support. More trees mainly reduce Monte Carlo variability; they do not by themselves recover omitted heterogeneity.
        - Persist fitted preprocessing/forest artifacts if prediction for new patients is an intended deliverable. The reviewed final call primarily emits evaluation predictions and diagnostics rather than a dedicated deployment bundle.
    9. **P2 — Treat overlap as part of the estimand and study design.**
        - **Reason:** 27.2% exclusion in the active run changes who is represented, and a scalar propensity can hide weak support in modifier subgroups.
        - **Change:** prespecify clinically meaningful eligibility; add covariate balance, per-arm effective sample size, weight-tail, and subgroup-support diagnostics. Investigate calibration before interpreting extreme scores as immutable clinical nonoverlap.
        - **Evaluate:** prespecified clipping/trimming sensitivity on both each method's target population and a common comparison population. Never select bounds by whichever oracle correlation or ATE is most favorable.
        - If overlap weighting is explored, label the different target population explicitly. Do not simply remove the `u²` weighting from a final R learner: that changes its objective and gives unstable ratio targets disproportionate influence.
    10. **P2 — Add CATE-specific evaluation for cohorts without oracle truth.**
        - **Reason:** ordinary outcome AUROC and ATE confidence intervals do not assess individualized effect predictions.
        - **Change:** add held-out constant-effect versus heterogeneous R-loss, calibration of DR scores against predicted effects, prespecified effect-ranked subgroup contrasts, policy value/regret diagnostics, and uncertainty-aware ranking summaries.
        - **Evaluate:** use outer-heldout scores or a separate evaluation sample. Account for dependencies from shared training and for outcome direction when ranking benefit. Avoid repeatedly choosing algorithms on the same outer evaluation outputs.
        - RATE/TOC-style evaluation assesses whether a fixed ranking prioritizes larger treatment effects and is applicable beyond ordinary outcome prediction. [Yadlowsky and colleagues](https://arxiv.org/abs/2111.07966). Repeated-split BLP/GATES work also motivates evaluating stable summaries of heterogeneity, while its randomized-experiment guarantees should not be transferred unchanged to this observational workflow. [Chernozhukov and colleagues](https://arxiv.org/abs/1712.04802).
    11. **P2 — Make uncertainty correspond to the reported claim.**
        - **Reason:** forest pointwise intervals condition on a learned representation; the score-based ATE interval does not capture every source of selection, extraction, or boundary uncertainty.
        - **Change:** report pointwise CATE intervals separately from average-effect intervals. Add repeated complete-pipeline splits and a prespecified patient-level resampling or external-validation strategy, with coverage checked by simulation.
        - **Evaluate:** empirical coverage and width for CATE summaries and ATE across data-generating replications. Re-running only the final forest quantifies a narrower uncertainty source than repeating discovery and extraction.
        - Honest-forest asymptotic inference is valuable, but its assumptions and estimator-specific variance calculation do not automatically validate intervals for an arbitrary adaptive upstream pipeline. [Generalized Random Forests](https://arxiv.org/abs/1610.01271).
    12. **P2 — Strengthen causal design, grouping, and reporting contracts.**
        - **Reason:** row-based folds allow leakage if multiple decisions belong to one patient. Outcome-driven feature discovery cannot prove that retained variables form a sufficient adjustment set.
        - **Change:** require explicit patient/site/time grouping where appropriate, using the split registry; prespecify key adjustment variables and time zero; add negative-control and unmeasured-confounding sensitivity analyses appropriate to the study.
        - **Reporting fixes:** rename post-hoc `ate_bias` to `mean_cate_bias` and separately calculate AIPW bias; make selection authority prominent; list actual final nuisance inputs and encoder policies; label nonconverged ontologies and eligible populations clearly.
        - **Reproducibility:** retain code/config/model identities and actual package versions. Existing setup-version metadata and the interpreter available for this review report different package versions, so a historical environment snapshot is not proof of the current runtime.

13. **A practical evaluation sequence and artifact guide**
    1. **First experiment: separate the main sources of error.**
        - On controlled synthetic replications, compare: known structured measurements + intended adjustment/effect covariates; extracted measurements + the same intended covariates; extracted measurements + current selection; and the complete discovery/extraction/selection workflow.
        - Keep oracle-informed models strictly in a separately labeled benchmark branch. Their purpose is to locate the performance ceiling and the discovery/measurement/selection losses, never to guide the purportedly oracle-free fit on the same evaluation data.
        - Include a constant-effect baseline and null-heterogeneity data, not only simulations with known modifiers.
    2. **Then compare bounded changes on frozen measurements.**
        - Contrast the legacy role mode, independent tasks, nonlinear modifier bases, and broader effect-input sets.
        - Tune forest and nuisance policies entirely within outer training; compare a DR learner using the same intended target population.
        - Use guarded reselection to reuse compatible extraction. Changing extraction policies or discovery coverage requires a fresh compatible upstream branch rather than stale measurement reuse.
    3. **Finally repeat the complete workflow.**
        - Vary sample size, effect strength/form, overlap, confounding nonlinearity, documentation noise, missingness, and extraction variability.
        - Freeze a final evaluation cohort or untouched simulation seeds after method development. Repeated post-hoc oracle inspection otherwise becomes a source of development overfitting even if each individual run is internally cross-fitted.
        - Report CATE RMSE/PEHE, calibration, ranking/policy value, ATE error, interval coverage, subgroup support, feature recall/stability, and compute/extraction cost.
    4. **Most useful run artifacts.**

        | Location relative to a run root | What it establishes |
        | --- | --- |
        | `run_config.json`, resolved model configs | Requested/resolved scientific and runtime settings |
        | `components/tfidf/split_provenance.jsonl` | Exact outer/inner row assignments and data identity |
        | `components/{text_models,tfidf,neural_queries}/` | Native evidence and numerical sidecars |
        | `handoff/evidence.jsonl`, `handoff/index.json` | What was handed to Stage 2 and where it originated |
        | `stage2/evidence_compilation/` | Cards, exact members, lineage, and compression counts |
        | `stage2/outer_NNN/interpreted_candidates.json` | Named concepts before alias consolidation |
        | `stage2/outer_NNN/feature_definitions.json` | Initial operationalized measurement definitions |
        | `stage2/outer_NNN/ontology_supervision/` and extraction directories | Revisions, training measurements, failures, harmonization, and convergence evidence |
        | `stage2/outer_NNN/selection/statistical_evidence.json` | Numerical evidence before final role/task routing |
        | `stage2/outer_NNN/selection/elastic_net_selection.json` | Final selection authority, decisions, and counts |
        | `stage2/outer_NNN/selection/selected_definitions.json` | Frozen retained modeling definitions |
        | `stage2/outer_NNN/selection/measurement_definitions.json`, `selected_latent_states.json` | Raw extraction dependencies and derived-feature reconstruction |
        | `stage2/outer_NNN/selection/nuisance_predictions.csv` | Inner out-of-fold diagnostics; rows repeat across outer files |
        | `stage2/outer_NNN/estimation/` | Fold predictions, final nuisance diagnostics, overlap, and forest audits |
        | `stage2/cross_fitted_predictions.csv` | One outer-heldout prediction row per cohort row |
        | `stage2/causal_estimate.json` | Pooled AIPW estimate, eligible population, and diagnostics |
        | `stage2/posthoc_oracle_ite_metrics.json` | Frozen-prediction oracle CATE/risk-difference evaluation |
        | `stage2/reselection_archives/` and `preselection/` under relevant folds | Preserved downstream versions and reusable frozen measurements |

14. **Review validation, limitations, and companion files**
    1. **Checks performed.**
        - Traced the active orchestration, ten-lane registry and producers, evidence compiler, discovery prompts, measurement workflow, numerical selection, role/task routing, final estimator, oracle evaluator, and saved-run compatibility mechanisms.
        - Read the example configuration and current workflow/independent-task documentation, comparing them with active code where their descriptions differ.
        - Ran six focused test modules: architecture registry, TF-IDF Stage 1 honesty, independent tasks, propensity overlap, role adjudication, and causal-forest runtime. **Result: 60 passed, four warnings, 20.86 seconds.**
        - The warnings concern plotting-library deprecations and incomplete out-of-bag predictions in a deliberately small forest test. They do not establish a production model failure.
        - Checked the current EconML default `cv=2` and demonstrated that an effect-only retained feature enters the external outcome model but not the external treatment model.
        - Verified the frozen prediction hashes of the three saved analyses summarized above.
    2. **Environment and scope limits.**
        - The repository `.venv` lacked pytest; the checks used the existing interpreter configured in the local saved-run launchers, `/home/klkehl/thisenv/bin/python`, without installing dependencies.
        - This review interpreter reported Python 3.13.7, EconML 0.16.0, and scikit-learn 1.6.1. Existing setup metadata records a different environment, including EconML 0.17.0. Saved numerical outputs were inspected, not reproduced under the review interpreter.
        - Focused tests establish selected contracts and execution behavior; they do not demonstrate causal identification or improved CATE accuracy. The full project suite, package build, and full LLM/GPU pipeline were not rerun for this report.
        - No production code, run configuration, extraction matrix, or saved estimate was intentionally changed. This deliverable consists of the report and its review evidence.
    3. **Companion files in this dated folder.**
        - [Saved-run aggregate evidence and hashes](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/saved_run_evidence_2026-09-19.json).
        - [Successful focused-validation log](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/focused_validation_run_environment_2026-09-19.log).
        - [Initial `.venv` test-run attempt](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/focused_validation_2026-09-19.log).
        - [Actual review runtime and routing checks](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/runtime_review_checks_2026-09-19.json).
        - [Reviewed-source inventory and report checks](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/review_manifest_2026-09-19.json).
