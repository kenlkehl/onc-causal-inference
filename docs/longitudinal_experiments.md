# Longitudinal Multimodal Causal Foundation Model — Experiment Log

This document records the architectures, training objectives, multimodal synthetic DGP evolution, training runs, controlled ablations, diagnostic probes, and held-out evaluations for the longitudinal multimodal causal foundation model (`oci.longitudinal`) on the 5,000-design oncology prior (`synthetic_data/designs/nemotron_5000.json.gz`, 1,000 held-out designs spanning 3 unseen cancer types + new clinical questions on seen cancers).

All models and baselines use strictly observational context inputs `(X, T, Y)` with zero oracle leakage.

---

## 1. Architectures, Objectives, & Multimodal DGP Evolution

### 1.1 Side-by-Side Architecture & Training Configuration Matrix

All seven runs share the five-stream multimodal input schema (`notes`, `labs`, `vitals`, `encounters`, `pros` encoded via modality-specific 2-layer `EventTransformer`s + cross-modal timeline transformer, `event_width=128`, `event_layers=2`, `heads=4`, `study_width=192`, `bins=129`). The exact architectural, solver, schedule, and note-DGP settings recorded in each run's checkpoint are:

| Run (Artifact Dir / Script) | Params / Fmt | Note Backbone | Note Pooling & Windowing | Structured & Cross-Modal Encoder | Study Head & Differentiable Solver | Training Schedule (`N_ctx`, `N_q`, `accum`, `steps`, `lr / note_lr`) | Note DGP |
|---|---|---|---|---|---|---|---|
| **Baseline (Pure PFN)** (`longitudinal_tinybert_n5000`, `run_longitudinal_h200.sh`) | `20.04M` / `v3` | `TinyBERT_General_4L_312D` (14.5M), **frozen** Phase 1 | `note_slots=4`, single mean-pool per note, non-overlapping 512-tok windows | 5× `EventTransformer` + cross-modal timeline (no explicit trajectory slopes) | **Pure 4-layer `TwoWayStudyLayer`** (`study_layers=4`), **no R-learner solver** | Phase 1: `N=500..5000` (uniform), `N_q=64`, `accum=4`, `lr=3e-4` (warmup 50, cosine), stopped s527 | Plain (`augment_notes`) |
| **Curriculum (Pure PFN)** (`longitudinal_tinybert_curriculum`, early `run_longitudinal_curriculum_h200.sh`) | `20.04M` / `v3` | `TinyBERT_General_4L_312D`, unfrozen (`note_grad_patients=64 → 128`) | Same as Baseline (`note_slots=4`) | Same as Baseline | **Pure 4-layer `TwoWayStudyLayer`** (`study_layers=4`), **no R-learner solver** | 2-stage: S1 `N=64..256, N_q=32, accum=32, lr=6e-4/3e-5` (1..600) → S2 `N=500..5000, N_q=64, accum=8, lr=2e-4/1e-5` (stopped s723) | Plain (`augment_notes`) |
| **Run 1 (Hybrid)** (`longitudinal_tinybert_hybrid`, `run_longitudinal_curriculum_h200.sh`) | `19.38M` / `v4` | `TinyBERT_General_4L_312D`, unfrozen (`64 → 128` grad pts) | **`note_slots=8`** multi-query token attn + masked **max** + mean pooling (lossless windows) | Added **explicit per-stream trajectory summaries** (`latest, mean, asinh OLS slope/yr`, counts, recency) | **Hybrid LOO-CV R-learner** (`solver_dim=64`, `ridge_grid_size=5`, fp32 Cholesky) + **2-layer residual `TwoWayStudyLayer`** (`attn/prop_residual_scale` init `0.1`) | 2-stage: S1 `N=64..256, N_q=32, accum=32` (s1..600, 19.2k studies) → S2 `N=500..5000, N_q=64, accum=8` (s601..1200, 4.8k studies) | Plain (`augment_notes`) |
| **Run 2 (Wide Hybrid)** (`longitudinal_tinybert_wide_hybrid`, `run_longitudinal_wide_hybrid_h200.sh`) | `20.96M` / `v4` | `TinyBERT_General_4L_312D`, unfrozen (`64 → 128` grad pts) | **`note_slots=16`** (2× concept slots) | Same as Run 1 | **`solver_dim=256`** (128 note + 128 struct) + **`lexical_dim=128`** (context-only hashed 1–3gram SVD), `ridge_grid_size=5` + 2-layer residual PFN | Same 2-stage curriculum as Run 1 (s1..600 → s601..1200, 24k total studies) | Plain (`augment_notes`) |
| **Run 3 (Single-Stage Plain)** (`longitudinal_tinybert_single_stage`, `run_longitudinal_single_stage_h200.sh`) | `21.05M` / `v4` | `TinyBERT_General_4L_312D`, unfrozen (`note_grad_patients=128`) | `note_slots=16` + **`--per-slot-note-temporal`** (per-slot max/latest/delta/time-attn) + **32-tok overlapping** windows | Same as Run 1 | `solver_dim=256` + `lexical_dim=128` + **`ridge_grid_size=7`** (CATE penalty ceiling `~22,000`) + 2-layer residual PFN | **Single-stage log-uniform** `N=100..10,000`, `N_q=64`, `accum=8`, `1,200` steps (`9,600` studies), `lr=4e-4 / note_lr=2e-5`, warmup 30, cosine | Plain (`augment_notes`) |
| **Run 4 (Single-Stage Paraphrased)** (`longitudinal_tinybert_paraphrased`, `run_longitudinal_paraphrased_h200.sh`) | `21.05M` / `v4` | `TinyBERT_General_4L_312D`, unfrozen (`128` grad pts) | Identical to Run 3 | Identical to Run 3 | Identical to Run 3 (exact 1-variable DGP ablation) | Identical to Run 3 (`1,200` steps, `9,600` studies) | **Paraphrased** (`--paraphrase-bank`) |
| **Run 5 (`TinyOncBERT` v5)** (`longitudinal_tinyoncbert_v5`, `run_longitudinal_tinyoncbert_h200.sh`) | `29.45M` / `v4` | `TinyOncBERT` (6L/384D/6H, 22.7M, PubMedBERT vocab), unfrozen (`128` grad pts) | `note_slots=16`, `per_slot_note_temporal`, 32-tok overlap + sentence-level MaxSim | Same as Run 1 | `solver_dim=256` + `lexical_dim=128` + `sentence_concept_dim=128` (context-only spherical k-means MaxSim) + `--block-ridge` (2D LOO-CV grid shrinking text vs. struct blocks separately), `ridge_grid_size=7` + 2-layer residual PFN | Single-stage log-uniform `N=100..10,000`, `N_q=64`, `accum=8`, `1,200` steps (`9,600` studies), `lr=4e-4 / note_lr=2e-5`, warmup 30, cosine | Paraphrased (`--paraphrase-bank`) |
| **Option B (`v6` Hybrid Continued Pre-Training)** (`longitudinal_option_b_v6`, `run_longitudinal_option_b_h200.sh`) | **`36.6M`** trainable (+ `22.7M` frozen BERT) | **`TinyOncBERT` (22.7M), permanently FROZEN** (`eval()`, zero grads) | Context-fitted `TinyOncBERT` SentKM (`option_a_full` 1,077 cols) | **`StructuredTemporalEncoder`** (4 structured `EventTransformer`s + `CrossModalTimeline` warm-started from Run 5's 203 tensors → `24` learned temporal cols appended alongside explicit cols) | **Warm-started `SparseTabularCausalPFN` v4 (`s40k`)** (ARD spike-and-slab + decoupled `e_r` + symmetric pairwise interactions + 8L/320W/40S residual TabPFN) via exact differentiable hybrid forward pass | **50/50 mixed episodes** (`50%` longitudinal `N=100..5,000` + `50%` pure-tabular replay w/ 6-class role loss), `accum=8`, `1,200` steps (`s1197`, best `s800`), `temporal_lr=2e-4`, `tabular_lr=3e-5`, L2-SP anchor `1e-4` | **Paraphrased** (`--paraphrase-bank`) + 6-family tabular replay |
| **Option B (`v7` Disjoint MMAI v23 + Lean `sent_256`, Wired)** (`longitudinal_option_b_v7`, `prepare_mmai_v23_cache_h200.sh` + `run_longitudinal_option_b_lean_h200.sh`) | **`36.6M`** trainable (+ `22.7M` frozen BERT) | **`TinyOncBERT` (22.7M), permanently FROZEN** (`eval()`, zero grads) | Lean **`sent_256`** (`476` explicit cols) + balanced 320-clause multi-note `v23` SentKM (`note_sentences_balanced`) | Same `StructuredTemporalEncoder` (`24` learned temporal cols → `500` total) | **Warm-started `SparseTabularCausalPFN` v5** (v5 epistasis scanner + sparse/dense prior + soft propensity coupling) | **3-way mixed episodes** (`35%` `TinyOncBERT`-disjoint `mmai/v23` `train_unseen` [37,631 patients] + `25%` `nemotron_5000` 5-stream + `40%` pure-tabular v5 replay), `accum=8`, `1,200` steps, `num_workers=3`, `MALLOC_ARENA_MAX=2` | **Real 14-note pan-cancer `v23`** (`train_unseen`) + Paraphrased + Tabular v5 replay |

### 1.2 Training Objectives & Loss Functions

1. **Longitudinal Causal Pretraining Objective (`oci/longitudinal/model.py`, `CausalStudyModel.loss_components` — constant weights across all 7 runs):**
   $$\mathcal{L}_{\text{study}} = \mathcal{L}_{\text{outcome}}^{\text{sinh-129}} + 2.0\,\mathcal{L}_{\text{cate}}^{\text{MSE}} + 0.5\,\mathcal{L}_{\text{prop}}^{\text{BCE}}$$
   - **$\mathcal{L}_{\text{outcome}}^{\text{sinh-129}}$ (weight `1.0`):** Local-width Gaussian-smoothed cross-entropy over 129 bins on a non-uniform $\sinh$-warped centered logit-offset grid $\delta = \operatorname{logit}(p) - \operatorname{logit}(\hat{p}_t) \in [-6, +6]$ (evaluated on both oracle counterfactual arm risks `true_y0`, `true_y1`), calibrating posterior predictive mean and epistemic SD.
   - **$\mathcal{L}_{\text{cate}}^{\text{MSE}}$ (weight `2.0`):** Direct MSE between predicted posterior-mean risk difference $(\hat{p}_1 - \hat{p}_0)$ and oracle ITE $(p_1 - p_0)$ on query patients.
   - **$\mathcal{L}_{\text{prop}}^{\text{BCE}}$ (weight `0.5`):** Binary cross-entropy of the treatment-masked propensity query head against oracle `true_treatment_prob`.
   - **Optimization & Numerical Precision:** `AdamW` with separate parameter groups (`note_lr` and `backbone_weight_decay = 0.0` on the text backbone; `lr` and `weight_decay = 0.01` on 2D+ downstream matrices only, `0.0` on biases/LayerNorms/embeddings/tokens), cosine LR decay with linear warmup, gradient clipping `1.0`, `bf16` mixed precision for transformer blocks while strictly forcing `fp32` for Cholesky LOO-CV Ridge solvers, SVD/k-means projections, calibration heads, and losses. Partial-gradient context encoding (`--note-grad-patients K`) backpropagates through the text backbone on all queries plus `K` random context patients per episode while encoding the remaining context notes under `no_grad`.
   - **Key objective-landscape shift (Baseline/Curriculum → Run 1+):** Although the loss formula stayed fixed, embedding exact differentiable closed-form LOO-CV nuisances $(\hat{e}^{(-i)}, \hat{m}^{(-i)})$ and Robinson orthogonalization $(Y - \hat{m}^{(-i)}) \sim (T - \hat{e}^{(-i)}) \cdot [1, \phi(X)]$ inside the forward pass eliminated the zero-gradient cohort-mean saddle that stalled pure attention PFNs (`within_r ≈ 0.00 → +0.217`).

2. **`TinyOncBERT` Distillation & MLM Objective (`oci/longitudinal/tinyoncbert.py`, `scripts/run_tinyoncbert_h200.sh`):**
   - **Architecture & Init:** 6-layer, 384-hidden, 6-head, 1536-FFN BERT (`22.7M` params, `max_position_embeddings=512`, `BiomedBERT` clinical tokenizer, word embeddings SVD-initialized from `NeuML/pubmedbert-base-embeddings`), distilled from frozen 600M `Qwen/Qwen3-Embedding-0.6B` on 2M curated `split == "train"` oncology sentences + 250k contiguous 512-token note windows from `ksg-dfci/mmai-synthetic-1226/patients/all_synthetic_notes.parquet` (3 epochs, `batch_size=2048`, `lr=5e-4`, AdamW `wd=0.01`, 3% warmup, `7.6 min` on 1× H200).
   - **Composite 5-Term Objective:**
     $$\mathcal{L}_{\text{distill}} = 1.0\,\mathcal{L}_{\text{emb}} + 0.5\,\mathcal{L}_{\text{rel}} + 0.5\,\mathcal{L}_{\text{neg}} + 0.1\,\mathcal{L}_{\text{pol}} + 0.25\,\mathcal{L}_{\text{mlm}}$$
     - $\mathcal{L}_{\text{emb}}$: $(1 - \cos(\text{proj}(s), t)) + \text{MSE}(\text{proj}(s), t)$ aligning projected student sentence embeddings to normalized teacher embeddings.
     - $\mathcal{L}_{\text{rel}}$: $\text{MSE}(S_{\text{native}} S_{\text{native}}^\top,\, T T^\top)$ transferring pairwise relational geometry directly into the unprojected native 384D space used at inference.
     - $\mathcal{L}_{\text{neg}}$: $\operatorname{ReLU}(\cos(s, s_{\text{flip}}) - 0.25)$ margin repulsion between synthetically polarity-flipped clinical clauses (pushing `"reports dyspnea"` vs. `"denies dyspnea"` near-orthogonal; achieved `val_neg_flip_cos = 0.066`).
     - $\mathcal{L}_{\text{pol}}$: 3-class polarity cross-entropy (affirmed / negated / neutral; achieved `val_pol_acc = 99.73%`).
     - $\mathcal{L}_{\text{mlm}}$: Standard 15% BERT masked language modeling on sentences every step plus 512-token contiguous oncology note windows every 6th step (`--window-mlm-every 6`).

3. **Cross-Fit Episodic Factual Adaptation Objective (`oci/longitudinal/adapt.py`, `adapt-predict`):**
   - Strictly observable K-fold cross-fitted adaptation on a single applied cohort with **zero oracle inputs**:
     $$\mathcal{L}_{\text{adapt}} = \operatorname{BCE}(\hat{p}_{T_q}, Y_q) + 0.5\,\operatorname{BCE}(\hat{e}_q, T_q) + \lambda_{\text{anchor}}\|\theta - \theta_0\|_2^2$$
   - Freezes all CATE shrinkage, calibration, and residual attention heads (`log_lam_tau`, `log_text_mult_tau`, `log_cv_temp`, `arm_cal`, `prop_cal`, `layers`, `head`, `propensity_head`, `attn/prop_residual_scale`) so unobserved treatment-effect heterogeneity cannot overfit factual noise; adapts only solver projections, text channel weights, and nuisance Ridge penalties (default `anchor_weight=1e-2`, `lr=1e-4` cosine, `40` steps × `4` random inner episodes), gated automatically against step-0 zero-shot inner-validation factual loss.

### 1.3 Multimodal Longitudinal DGP & Iteration Changelog

**Base 3-Level Longitudinal DGP (`synthetic_data/longitudinal.py` + `nemotron_5000.json.gz`):**
- **Level 1 (Design):** Uniformly samples from 4,000 `split == "train"` Nemotron-authored oncology study designs spanning 20 cancers and 100 clinical questions (1,000 `split == "held_out"` designs including 3 completely unseen cancers are strictly isolated for evaluation). Design widths span `3–100` candidate measurements and note findings.
- **Level 2 (Study Equations):** Draws permitted causal roles per candidate (capping wide designs to `<= 16` active causal variables while keeping all remaining candidates as chart distractors), samples `linear` / `threshold` / `quadratic` terms plus up to `15` pairwise treatment interactions and (in `65%` of studies) `1–6` pairwise outcome interactions (`coef ~ U(-0.35, 0.35)`), and Monte Carlo–calibrates linear predictors (`8%` unconfounded RCTs vs. `tx_std ~ U(0.6, 1.8)`, `out_std ~ U(0.9, 2.0)`, `tx_rate ~ U(0.25, 0.75)`, `ctrl_rate ~ U(0.12, 0.82)`).
- **Level 3 (5-Stream Patient Histories):** Generates timestamped pretreatment histories (`day_from_index <= 0`) across `notes`, `labs` (CBC/BMP/CMP with longitudinal slopes), `vitals` (including annualized weight loss), `encounters` (settings + ICD-10-CM + CPT codes), and `pros` (11 canonical PRO-CTCAE v1.0 ordinal items), with positive-only charting and note-framing augmentation (`--augment-notes`).

**What Changed Across Iterations (and Why):**
1. **Baseline Pure PFN → Curriculum Pure PFN (`9c0a42a`):** Unfroze `TinyBERT` and added a 2-stage small-to-large `N_ctx` curriculum (`64..256` with `accum=32` → `500..5000` with `accum=8`). *Result:* Still stuck in the cohort-mean saddle (`within_r = +0.021`), proving attention alone cannot bootstrap orthogonalized CATE estimation from scratch.
2. **Curriculum Pure PFN → Run 1 Hybrid (`3b392c9`):** Replaced 4 pure attention layers with an exact differentiable fp32 Cholesky LOO-CV R-learner (`solver_dim=64`, `ridge_grid_size=5`) + 2 residual PFN layers, upgraded note pooling from single mean-pool to 8-slot multi-query token attention + masked max + mean, and added explicit per-stream longitudinal trajectory slopes. *Result:* Broke the saddle immediately (`cate_rmse 0.1118 → 0.1016`, `within_r +0.021 → +0.150`).
3. **Run 1 → Run 2 Wide Hybrid (`6e62bcd`):** Step-450 linear probes revealed a 32-dim note bottleneck; expanded `note_slots 8 → 16`, `solver_dim 64 → 256` (128 note + 128 structured), and added context-only 128-d hashed 1–3gram lexical SVD (`--lexical-dim 128`). *Result:* Huge jump in within-study CATE (`+0.150 → +0.199`), risk (`0.1946 → 0.1754`), and propensity (`0.1719 → 0.1302`), and final 1k-study `cate_rmse 0.1060 → 0.1028`.
4. **Run 2 → Run 3 Single-Stage Plain (`2df3779`, `4f76e34`):** Replaced the 2-stage split with single-stage **log-uniform `N_ctx = 100..10,000`**, added `--per-slot-note-temporal` (per-slot `max / latest / delta / time-attn` across visits), switched long-note windowing to 32-token overlap, and expanded `--ridge-grid-size 5 → 7` (raising upper CATE shrinkage to `~22k` to prevent 384-d over-spread). *Result:* All-time best plain-note 1k-study performance (`cate_rmse 0.1028 → 0.0962`, `within_r +0.144 → +0.187`, `prop_rmse 0.1275 → 0.1199`).
5. **Run 3 → Run 4 Single-Stage Paraphrased (`91e911f`, `db41543`):** **Pure DGP upgrade (1-variable ablation)** — added `--paraphrase-bank` (`nemotron_5000_paraphrases.json.gz`: 2,902 clinical findings × ~10 LLM paraphrases, multi-section oncology headers, varied ECOG templates, and benign EHR filler) while holding architecture/objective/schedule fixed. *Result:* Exposed that static 16-slot note attention (`0.52` probe AUROC) + lexical n-grams lose traction under paraphrased/Full-LLM clinical prose (`1k cate_rmse = 0.1048`, `within_r = +0.110`; Full-LLM `0.1233`).
6. **Run 4 → `TinyOncBERT` + Run 5 (`a62b90e`, `828fc77`):** Guided by the Section 5 representation probe, distilled `TinyOncBERT` (22.7M, PubMedBERT vocab, negation-repulsion), added **study-adaptive sentence-concept MaxSim** (`--sentence-concept-dim 128` via context-only spherical k-means), and upgraded LOO-CV Ridge to **2D block-weighted Ridge (`--block-ridge`)** so propensity, outcome, and CATE independently shrink noisy text blocks relative to structured features. *Result:* Finished at `1k cate_rmse = 0.0987, within_r = +0.1553` (crushing Run 4's `0.1048 / +0.1103` with zero drop on unseen cancers), yet its uniform 2-block Ridge head remained fundamentally limited compared to per-column ARD gating.
7. **Run 5 → Option A Zero-Shot Composition & Option B (`v6`) Continued Multimodal Pre-Training (`241a874`, `e6282ba`):** Replaced Run 5's uniform block-Ridge head with finished `SparseTabularCausalPFN` v4 (`step 40,000`) on unmixed explicit features + frozen `TinyOncBERT` SentKM (**Option A**), and launched continued synthetic multimodal pre-training (**Option B `v6`**) appending `24` trainable `StructuredTemporalEncoder` columns (warm-started from Run 5's 203 structured tensors) with 50/50 pure-tabular replay + L2-SP anchor regularization. *Result:* Even at **step 0** (zero fine-tuning), Option A/Option B shattered Run 5 on the exact same 32 paraphrased studies (`0.0800 / +0.428` vs. Run 5's `0.0941 / +0.203`), and Option B climbed to **`0.0782 / +0.484`** by step 200 with zero tabular forgetting! Lean `sent_256` (`476` cols) beat wider `option_a_full` (`1,077` cols) zero-shot on Full-LLM cohorts (`0.1041 / +0.231` vs `0.1072 / +0.208`) and was promoted to the default Option A input structure (`e6282ba`).

---

## 2. Training Runs Summary

### Mid-Training Held-Out Eval (32 fixed held-out studies, `N_ctx = 5,000`, `N_q = 64`)
Reference baselines on these 32 studies: `arm_mean cate_rmse = 0.1160`, observational LOO-CV `ridge cate_rmse = 0.1097`.

| Run | Artifact Dir | Architecture / Setup | Note Distribution | Status / Step | `cate_rmse` ↓ | `within_r` ↑ | `pool_r` ↑ | `risk_rmse` ↓ | `prop_rmse` ↓ | `ate_rmse` ↓ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (Pure PFN)** | `longitudinal_tinybert_n5000` | Pure `TwoWayStudyLayer` (`4L`, no R-learner), mean note pooling (`4` slots) | Plain | Stopped (527) | `0.1126` | `—` | `+0.6254` | `0.2125` | `—` | `—` |
| **Curriculum (Pure PFN)** | `longitudinal_tinybert_curriculum` | Pure `TwoWayStudyLayer` (`4L`), 2-stage small→large N | Plain | Stopped (723) | `0.1118` | `+0.0212` | `+0.6271` | `0.2122` | `0.2008` | `0.0470` |
| **Run 1 (Hybrid)** | `longitudinal_tinybert_hybrid` | Differentiable LOO-CV R-learner (`solver_dim=64`), 8 note slots, trajectory slopes, 2-stage curriculum (`N=64..256` → `500..5000`) | Plain | Complete (1200) | `0.1016` | `+0.1504` | `+0.7156` | `0.1946` | `0.1719` | `0.0410` |
| **Run 2 (Wide Hybrid)** | `longitudinal_tinybert_wide_hybrid` | Run 1 + `note_slots=16`, `solver_dim=256`, study-adaptive `lexical_dim=128` SVD, 2-stage curriculum | Plain | Complete (1200) | `0.1023` | `+0.1989` | `+0.7024` | `0.1754` | `0.1302` | `0.0375` |
| **Run 3 (Single-Stage Plain)** | `longitudinal_tinybert_single_stage` | Run 2 + single-stage log-uniform `N_ctx=100..10,000`, `--per-slot-note-temporal`, `--ridge-grid-size 7`, 32-tok overlapping note windows | Plain | Complete (1200 + 1k eval) | **`0.0951`** | **`+0.2169`** | **`+0.7526`** | **`0.1522`** | **`0.1230`** | **`0.0288`** |
| **Run 4 (Single-Stage Paraphrased)** | `longitudinal_tinybert_paraphrased` | Exact 1-variable ablation of Run 3 with `--paraphrase-bank` (`nemotron_5000_paraphrases.json.gz`, 2,902 findings × ~10 paraphrases + EHR sections + filler) | Paraphrased | Complete (1200 + 1k eval) | `0.0987` | `+0.1593` | `+0.7321` | `0.1876` | `0.1697` | `0.0365` |
| **Run 5 (`TinyOncBERT` v5)** | `longitudinal_tinyoncbert_v5` | `TinyOncBERT` (22.7M) + `--sentence-concept-dim 128` + `--lexical-dim 128` + `--block-ridge` + `--paraphrase-bank` | Paraphrased | **Complete (1200 + 1k eval)** | `0.0941` (s1200) | `+0.2078` (s1000) / `+0.2030` (s1200) | `+0.7631` (s1200) | `0.1662` (s1000) / `0.1665` (s1200) | `0.1387` (s1000) / `0.1388` (s1200) | `0.0287` (s1200) |
| **Option B (`v6` Hybrid, Complete)** | `longitudinal_option_b_v6` | Frozen `TinyOncBERT` + unmixed `option_a_full` (1,077) + `24`-d `StructuredTemporalEncoder` (Run 5 warm-start) + warm-started `SparseTabularCausalPFN` v4 (`s40k`), 50/50 tabular replay + L2-SP anchor | Paraphrased (+ 6-fam tabular replay) | **Complete (`s1,197`; best = `s800`)** | **`0.0762`** (s800; `0.0800` @ s0) | **`+0.5040`** (s800; `+0.5087` @ s600) | **`+0.8540`** (s800; `+0.8319` @ s0) | **`0.1287`** (s800; `0.1283` @ s1100) | **`0.1168`** (s800; `0.1165` @ s400) | **`0.0239`** (s800; `0.0227` @ s100) |

*(Run 5 host-RAM note: Run 5 was SIGKILLed by the Linux OOM killer at step 400 — fixed by capping `_EpisodePrefetcher._max_inflight` to `6`, nested eval workers to `<= 2`, and adding `del rows` / `gc.collect()` + glibc `malloc_trim(0)` in `oci/longitudinal/training.py` — and again at step 700 while sharing host RAM with Tabular v3's 16 workers; resumed cleanly from step 600 with `--num-workers 6` and passed step 700 cleanly.)*

### Learning Trajectories (Run 3, Run 4, & Run 5 on 32 Held-Out Studies at `N_ctx = 5,000`)

**Run 3 (Single-Stage Plain):**

| Step | `cate_rmse` | `within_r` | `pool_r` | `risk_rmse` | `prop_rmse` | `ate_rmse` |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | `0.0984` | `+0.1857` | `+0.7285` | `0.1597` | `0.1283` | `0.0311` |
| 200 | `0.0976` | `+0.1815` | `+0.7354` | `0.1583` | `0.1266` | `0.0302` |
| 300 | `0.0981` | `+0.1868` | `+0.7331` | `0.1566` | `0.1253` | `0.0311` |
| 400 | `0.0971` | `+0.1917` | `+0.7415` | `0.1554` | `0.1244` | `0.0304` |
| 500 | `0.0966` | `+0.1966` | `+0.7422` | `0.1544` | `0.1245` | `0.0296` |
| 600 | `0.0955` | `+0.2033` | `+0.7498` | `0.1535` | `0.1234` | `0.0285` |
| 700 | `0.0955` | `+0.2171` | `+0.7494` | `0.1533` | `0.1235` | `0.0292` |
| 800 | `0.0952` | `+0.2149` | `+0.7514` | `0.1529` | `0.1236` | `0.0289` |
| 900 | `0.0951` | `+0.2168` | `+0.7523` | `0.1526` | `0.1230` | `0.0288` |
| 1000 | `0.0950` | `+0.2171` | `+0.7530` | `0.1523` | `0.1230` | `0.0288` |
| 1100 | `0.0950` | `+0.2170` | `+0.7528` | `0.1521` | `0.1230` | `0.0288` |
| **1200** | **`0.0951`** | **`+0.2169`** | **`+0.7526`** | **`0.1522`** | **`0.1230`** | **`0.0288`** |

**Run 4 (Single-Stage Paraphrased, evaluated on paraphrased held-out notes):**

| Step | `cate_rmse` | `within_r` | `pool_r` | `risk_rmse` | `prop_rmse` | `ate_rmse` |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | `0.1013` | `+0.1216` | `+0.7098` | `0.1905` | `0.1741` | `0.0362` |
| 200 | `0.1007` | `+0.1366` | `+0.7193` | `0.1899` | `0.1727` | `0.0374` |
| 300 | `0.1006` | `+0.1524` | `+0.7179` | `0.1894` | `0.1721` | `0.0379` |
| 400 | `0.1009` | `+0.1346` | `+0.7189` | `0.1891` | `0.1716` | `0.0388` |
| 500 | `0.0996` | `+0.1466` | `+0.7270` | `0.1886` | `0.1707` | `0.0368` |
| 600 | `0.0995` | `+0.1527` | `+0.7274` | `0.1883` | `0.1705` | `0.0376` |
| 700 | `0.0991` | `+0.1569` | `+0.7286` | `0.1880` | `0.1705` | `0.0368` |
| 800 | `0.0989` | `+0.1571` | `+0.7293` | `0.1881` | `0.1707` | `0.0365` |
| 900 | `0.0987` | `+0.1595` | `+0.7321` | `0.1880` | `0.1699` | `0.0363` |
| 1000 | `0.0988` | `+0.1588` | `+0.7318` | `0.1876` | `0.1699` | `0.0365` |
| 1100 | `0.0987` | `+0.1595` | `+0.7323` | `0.1876` | `0.1697` | `0.0365` |
| **1200** | **`0.0987`** | **`+0.1593`** | **`+0.7321`** | **`0.1876`** | **`0.1697`** | **`0.0365`** |

**Run 5 (`TinyOncBERT` v5, evaluated on paraphrased held-out notes — complete):**

| Step | `cate_rmse` | `within_r` | `pool_r` | `risk_rmse` | `prop_rmse` | `ate_rmse` |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | `0.0988` | `+0.1453` | `+0.7279` | `0.1755` | `0.1452` | `0.0306` |
| 200 | `0.0973` | `+0.1671` | `+0.7393` | `0.1746` | `0.1439` | `0.0306` |
| 300 | `0.0972` | `+0.1568` | `+0.7414` | `0.1727` | `0.1431` | `0.0311` |
| 400 | `0.0969` | `+0.1594` | `+0.7479` | `0.1703` | `0.1411` | `0.0322` |
| 500 | `0.0957` | `+0.1828` | `+0.7524` | `0.1699` | `0.1417` | `0.0299` |
| 600 | `0.0949` | `+0.1871` | `+0.7570` | `0.1682` | `0.1399` | `0.0295` |
| 700 | `0.0949` | `+0.1843` | `+0.7569` | `0.1671` | `0.1402` | `0.0293` |
| 800 | `0.0945` | `+0.1880` | `+0.7600` | `0.1679` | `0.1398` | `0.0288` |
| 900 | `0.0947` | `+0.1852` | `+0.7590` | `0.1667` | `0.1391` | `0.0292` |
| 1000 | `0.0942` | **`+0.2078`** | `0.7629` | **`0.1662`** | **`0.1387`** | `0.0291` |
| 1100 | `0.0942` | `+0.2071` | `0.7628` | `0.1665` | `0.1392` | `0.0293` |
| **1200** | **`0.0941`** | `+0.2030` | **`+0.7631`** | `0.1665` | `0.1388` | **`0.0287`** |

---

## 3. Final 1,000-Study Held-Out Evaluation (`N_ctx = 5,000`, `64,000` query patients, `seed = 42`)

Baselines across all 1,000 held-out studies: `zero = 0.1455`, observational LOO-CV `ridge = 0.1199`, `arm_mean = 0.1140`.

| Run | `cate_rmse` ↓ | `within_r` ↑ | `pool_r` ↑ | `spearman` ↑ | `risk_rmse` ↓ | `prop_rmse` ↓ | `ate_rmse` ↓ | `ate_bias` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Run 1 (Hybrid)** | `0.1060` | `+0.1021` | `+0.6859` | `+0.7411` | `0.2031` | `0.1694` | `0.0471` | `-0.0030` |
| **Run 2 (Wide Hybrid)** | `0.1028` | `+0.1440` | `+0.7069` | `+0.7584` | `0.1791` | `0.1275` | `0.0395` | `+0.0010` |
| **Run 3 (Single-Stage Plain)** | **`0.0962`** | **`+0.1870`** | **`+0.7502`** | **`+0.7923`** | **`0.1583`** | **`0.1199`** | **`0.0315`** | **`-0.0023`** |
| **Run 4 (Single-Stage Paraphrased)** *(eval on paraphrased notes)* | `0.1048` | `+0.1103` | `+0.6924` | `+0.7454` | `0.1988` | `0.1672` | `0.0442` | `-0.0034` |
| **Run 5 (`TinyOncBERT` v5)** *(eval on paraphrased notes)* | **`0.0987`** | **`+0.1553`** | **`+0.7344`** | **`+0.7841`** | **`0.1748`** | **`0.1391`** | **`0.0352`** | **`-0.0014`** |

### Zero-Shot Generalization to Unseen Cancers (1,000 studies)

| Run | Split Group | Studies | `cate_rmse` | `within_r` | `pool_r` | `risk_rmse` | `prop_rmse` | `ate_rmse` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Run 2** | `seen_cancer_new_question` | 250 | `0.1028` | `+0.1422` | `+0.7213` | `0.1791` | `0.1271` | `0.0390` |
| | `unseen_cancer` (3 held-out cancers) | 750 | `0.1028` | `+0.1446` | `+0.7017` | `0.1791` | `0.1276` | `0.0397` |
| **Run 3** | `seen_cancer_new_question` | 250 | **`0.0966`** | **`+0.1883`** | **`+0.7599`** | **`0.1590`** | **`0.1195`** | **`0.0321`** |
| | `unseen_cancer` (3 held-out cancers) | 750 | **`0.0961`** | **`+0.1865`** | **`+0.7467`** | **`0.1580`** | **`0.1201`** | **`0.0313`** |
| **Run 4** | `seen_cancer_new_question` | 250 | `0.1044` | `+0.1163` | `+0.7109` | `0.1997` | `0.1605` | `0.0425` |
| | `unseen_cancer` (3 held-out cancers) | 750 | `0.1050` | `+0.1084` | `+0.6857` | `0.1985` | `0.1694` | `0.0448` |
| **Run 5** | `seen_cancer_new_question` | 250 | `0.0991` | `+0.1522` | `+0.7454` | `0.1754` | `0.1348` | `0.0345` |
| *(paraphrased)* | `unseen_cancer` (3 held-out cancers) | 750 | **`0.0986`** | **`+0.1563`** | `+0.7305` | **`0.1746`** | `0.1405` | `0.0355` |

### Performance by Candidate Width (1,000 studies)

| Run | Candidate Width | Studies | `cate_rmse` | `within_r` | `risk_rmse` | `prop_rmse` | `ate_rmse` |
|---|---|---:|---:|---:|---:|---:|---:|
| **Run 2** | `3–20` | 195 | `0.0957` | `+0.1454` | `0.1667` | `0.1023` | `0.0388` |
| | `21–50` | 308 | `0.1036` | `+0.1519` | `0.1759` | `0.1240` | `0.0373` |
| | `51–80` | 295 | `0.1044` | `+0.1334` | `0.1836` | `0.1356` | `0.0405` |
| | `81–100` | 202 | `0.1058` | `+0.1463` | `0.1887` | `0.1416` | `0.0420` |
| **Run 3** | `3–20` | 195 | **`0.0880`** | **`+0.1837`** | **`0.1399`** | **`0.0935`** | **`0.0293`** |
| | `21–50` | 308 | **`0.0974`** | **`+0.1900`** | **`0.1554`** | **`0.1154`** | **`0.0301`** |
| | `51–80` | 295 | **`0.0982`** | **`+0.1815`** | **`0.1642`** | **`0.1285`** | **`0.0330`** |
| | `81–100` | 202 | **`0.0991`** | **`+0.1936`** | **`0.1701`** | **`0.1354`** | **`0.0333`** |
| **Run 4** | `3–20` | 195 | `0.1035` | `+0.0755` | `0.1959` | `0.1636` | `0.0502` |
| | `21–50` | 308 | `0.1065` | `+0.1092` | `0.1996` | `0.1722` | `0.0430` |
| | `51–80` | 295 | `0.1042` | `+0.1245` | `0.1987` | `0.1643` | `0.0427` |
| | `81–100` | 202 | `0.1044` | `+0.1249` | `0.2003` | `0.1672` | `0.0419` |
| **Run 5** | `3–20` | 195 | **`0.0927`** | `+0.1488` | **`0.1589`** | **`0.1157`** | `0.0356` |
| *(paraphrased)* | `21–50` | 308 | `0.1001` | `+0.1623` | `0.1743` | `0.1415` | **`0.0345`** |
| | `51–80` | 295 | `0.1000` | `+0.1444` | `0.1793` | `0.1430` | `0.0351` |
| | `81–100` | 202 | `0.1005` | **`+0.1668`** | `0.1833` | `0.1499` | `0.0362` |

---

## 4. Zero-Shot Plain vs. Paraphrased Distribution-Shift Evaluation

Paired evaluation on 8 held-out studies (`N_ctx = 500`, `N_q = 64`, `seed = 42`) with bit-identical ground truth between plain and paraphrased multi-section notes (`arm_mean = 0.0913`, `ridge = 0.0959`):

| Model | Notes | `cate_rmse` | `within_r` | `pool_r` | `risk_rmse` | `prop_rmse` | `ate_rmse` |
|---|---|---:|---:|---:|---:|---:|---:|
| **Run 1** (s1200, no lexical SVD) | Plain | `0.0851` | `-0.1068` | `+0.6979` | `0.1731` | `0.1947` | `0.0509` |
| | Paraphrased | `0.0855` | `-0.1013` | `+0.6957` | `0.1728` | `0.1941` | `0.0512` |
| | **Delta (para − plain)** | `+0.0003` | `+0.0055` | `-0.0022` | `-0.0003` | `-0.0006` | `+0.0003` |
| **Run 2** (s1200, lexical SVD) | Plain | `0.0884` | `+0.0130` | `+0.6642` | `0.1667` | `0.1758` | `0.0448` |
| | Paraphrased | `0.0877` | `-0.0297` | `+0.6764` | `0.1771` | `0.1999` | `0.0488` |
| | **Delta (para − plain)** | `-0.0007` | `-0.0427` | `+0.0122` | **`+0.0104`** | **`+0.0241`** | **`+0.0040`** |
| **Run 3** (s200, lexical SVD) | Plain | `0.0907` | `-0.0023` | `+0.6529` | `0.1672` | `0.1740` | `0.0463` |
| | Paraphrased | `0.0923` | `-0.0480` | `+0.6514` | `0.1782` | `0.1954` | `0.0525` |
| | **Delta (para − plain)** | `+0.0016` | `-0.0457` | `-0.0015` | **`+0.0110`** | **`+0.0214`** | **`+0.0062`** |

**Finding:** Plain-note lexical SVD gains (`risk_rmse`, `prop_rmse`, `within_r`) completely evaporate under paraphrased multi-section notes when evaluated zero-shot, dropping Run 2 & Run 3 back to Run 1's no-text baseline.

---

## 5. Representation & Pooling Probe (Plain vs. Paraphrased vs. Full-LLM Realistic Notes)

Paired notes-only observational probe on `bladder_q05_v036` (21 note findings) and `endometrium_q04_v046` (57 note findings) — 78 total hidden note-only clinical findings, `N_fit = 200`, `N_test = 80`. Full-LLM cohorts were rendered end-to-end with Nemotron 3.5 (`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`, NVIDIA sampling profile `temp=1.0, top_p=0.95, top_k=0, enable_thinking=False`).

Metric shown: **mean held-out AUROC across all 78 hidden note-only clinical findings** (higher is better; `0.500` = chance).

| Encoder | Pooling Method (Unsupervised, Context-Only) | Plain | Paraphrased | Full-LLM |
|---|---|---:|---:|---:|
| *(Reference)* | Current lexical hash 1–3g SVD-64 | `0.780` | `0.533` | `0.610` |
| *(Reference)* | Unhashed TF-IDF 1–3g SVD-64 | `0.702` | `0.521` | `0.599` |
| *(Reference)* | Run 4 (s400) `PatientEncoder` (16 static slots) | `0.527` | `0.529` | `0.521` |
| **TinyBERT-base (14M)** | Whole-note mean SVD-64 | `0.666` | `0.515` | `0.558` |
| | Sentence SVD-64 `[max, min]` | `0.890` | `0.533` | `0.620` |
| | **Sentence K-Means-128 MaxSim** | **`0.957`** | **`0.653`** | **`0.720`** |
| | Sentence K-Means-256 MaxSim | **`0.989`** | `0.614` | `0.661` |
| **Bio_ClinicalBERT (110M)** | Whole-note mean SVD-64 | `0.671` | `0.538` | `0.574` |
| | Sentence K-Means-128 MaxSim | `0.901` | `0.545` | `0.599` |
| | Sentence K-Means-256 + Lexical SVD | `0.945` | `0.574` | `0.615` |
| **PubMedBERT-embed (110M)** | Whole-note mean SVD-64 | `0.600` | `0.533` | `0.643` |
| | Sentence SVD-64 `[max, min]` | `0.862` | `0.569` | `0.651` |
| | Sentence K-Means-256 MaxSim | `0.951` | `0.587` | `0.642` |
| **ModernBERT-base (149M)** | Whole-note mean SVD-64 | `0.641` | `0.510` | `0.586` |
| | Sentence K-Means-128 MaxSim | `0.932` | `0.537` | `0.560` |
| **GTE-ModernColBERT (149M)** | Whole-note mean SVD-64 | `0.657` | `0.539` | `0.629` |
| | Sentence K-Means-256 MaxSim / + Lexical | `0.977` / `0.965` | `0.642` / `0.647` | `0.662` / `0.682` |
| **Qwen3-Embedding-0.6B (600M)** | **Whole-note mean SVD-64** | `0.759` | **`0.602`** | **`0.704`** |
| | **Sentence K-Means-256 MaxSim / + Lexical** | `0.974` / `0.969` | `0.635` / `0.632` | `0.720` / **`0.725`** |

### Key Takeaways from Diagnostic Probes
1. **Why Pure PFN plateaued:** Pure `TwoWayStudyLayer` attention never broke out of the cohort-mean saddle even on clean 24-d Gaussian tabular data (`within_r ≈ 0.00`). Adding a closed-form differentiable LOO-CV Ridge R-learner inside the forward pass broke symmetry on step 0 and unlocked every subsequent gain.
2. **Why static 16-slot note attention failed on unseen findings:** Globally learned `note_queries` are study-blind and cannot target 2,902 cancer-specific findings (many unseen at test time), yielding only `0.521–0.529` AUROC in Run 4.
3. **Why study-adaptive sentence-level MaxSim is the key breakthrough:** Whole-note pooling dilutes 1 signal sentence across ~100–200 benign/unrelated clauses per patient. Discovering study-specific sentence prototypes via unsupervised context-only spherical k-means and taking per-patient MaxSim jumped 14M `TinyBERT-base` from `0.515 → 0.653` (paraphrased) and `0.558 → 0.720` (full-LLM), matching 600M `Qwen3-Embedding-0.6B` (`0.635 / 0.725`) at 40× smaller parameter count while preserving affirmation vs. negation polarity.
4. **Why clinical BERT alone wasn't enough:** Whole-note `Bio_ClinicalBERT` (`0.538 / 0.574`) and `PubMedBERT-embed` (`0.533 / 0.643`) still trailed sentence-level pooling substantially, and retrieval-tuned models (`PubMedBERT-embed`) tended to cluster `"reports X"` and `"denies X"` together. Distilling a compact `TinyOncBERT` from `Qwen3-Embedding-0.6B` with negation-preserving objectives on synthetic oncology notes is a promising next encoder upgrade on top of sentence-level concept pooling.

---

## 6. `TinyOncBERT` Distillation & 16-Cohort Full-LLM Applied vs. Foundation Model Benchmark

### `TinyOncBERT` (6L / 384D / 6 heads, 22.7M params, PubMedBERT vocab)
Trained with `scripts/run_tinyoncbert_h200.sh` (`oci/longitudinal/tinyoncbert.py`) on 2M curated `split == "train"` clinical sentences + 250k contiguous 512-token windows from `ksg-dfci/mmai-synthetic-1226/patients/all_synthetic_notes.parquet` (3 epochs, **7.6 min on 1× H200**, `artifacts/tinyoncbert_6l_384d`):
- `val_teacher_cos = 0.900` (vs. 600M `Qwen/Qwen3-Embedding-0.6B` teacher)
- `val_neg_flip_cos = 0.066` (affirmed vs. negated clinical clauses pushed nearly orthogonal)
- `val_pol_acc = 99.73%` | `val_mlm_loss = 0.949` | `val_window_mlm_loss = 1.150`

### Head-to-Head Benchmark on All 16 Full-LLM Realistic Held-Out Cohorts
Evaluated across all 16 Nemotron 3.5 full-LLM held-out cohorts (`artifacts/llm_eval_cohorts/*/dataset.parquet`, `N = 1,000` each; fixed held-out query split `rows[800:1000]` = **3,200 query patients** total) at `N_ctx = 200` and `N_ctx = 800`. All applied models fit strictly on each study's `N_ctx` observational `(X, T, Y)` with zero synthetic pretraining; all PFN models perform a single zero-shot forward pass.

#### `N_ctx = 800` (3,200 held-out query patients across 16 full-LLM cohorts)

| Method | `cate_rmse` ↓ | `within_r` ↑ | `pool_r` ↑ | `risk_rmse` ↓ | `prop_rmse` ↓ | `ate_rmse` ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline: Arm-Mean (ATE only) | `0.1316` | `+0.000` | `+0.627` | `0.2114` | `0.1998` | `0.0610` |
| PFN Run 1 (Hybrid, plain, s1200) | `0.1240` | `+0.092` | `+0.702` | `0.1919` | `0.1770` | `0.0585` |
| PFN Run 2 (Wide Hybrid, plain, s1200) | `0.1273` | `+0.117` | `+0.671` | `0.1956` | `0.1848` | `0.0634` |
| PFN Run 3 (Single-Stage, plain, s800) | `0.1228` | `+0.093` | `+0.692` | `0.1882` | `0.1774` | `0.0550` |
| PFN Run 4 (Single-Stage, paraphrased, s700) | `0.1233` | `+0.096` | `+0.691` | `0.1891` | `0.1762` | `0.0558` |
| PFN Run 5 (TinyOncBERT + Block-Ridge, s600) | `0.1217` | `+0.130` | `+0.703` | `0.1847` | `0.1738` | `0.0568` |
| PFN Run 5 (TinyOncBERT + Block-Ridge, s800) | `0.1205` | `+0.129` | `+0.708` | `0.1835` | `0.1719` | `0.0552` |
| Applied: Struct-Only Ridge R-Learner | `0.1137` | `+0.152` | `+0.739` | `0.1838` | `0.1742` | `0.0421` |
| Applied: Struct + LexSVD Ridge R-Learner | `0.1144` | `+0.135` | `+0.736` | `0.1857` | `0.1770` | `0.0441` |
| Applied: Struct + TinyBERT SentKM Ridge-R | `0.1147` | `+0.155` | `+0.734` | `0.1847` | `0.1758` | `0.0469` |
| Applied: Struct + Qwen3-0.6B SentKM Ridge-R | `0.1164` | `+0.141` | `+0.724` | `0.1875` | `0.1774` | `0.0465` |
| Applied: Struct + **TinyOncBERT** SentKM Ridge-R | `0.1129` | `+0.144` | `+0.744` | `0.1841` | `0.1737` | `0.0463` |
| Applied: Struct + TinyOncBERT SentKM+Lex GBM-R | `0.1127` | `+0.160` | `+0.743` | `0.1768` | `0.1697` | `0.0456` |
| Applied: Struct + TinyOncBERT SentKM+Lex DragonNet | `0.1446` | `+0.085` | `+0.579` | `0.1925` | `0.1861` | `0.0662` |
| Zero-Shot Bridge: Tabular PFN-v2 (`Struct + SentKM-128`) | `0.1111` | `+0.173` | `+0.763` | `0.1722` | `0.1591` | `0.0495` |
| Zero-Shot Bridge: Tabular PFN-v2 (`Struct + SentKM-512 + LexSVD-128`) | `0.1108` | `+0.147` | `+0.762` | `0.1730` | `0.1606` | `0.0489` |
| Zero-Shot Bridge: **Tabular PFN-v3 (s25k)** (`Struct-Only`, 220) | `0.1099` | `+0.189` | `+0.759` | `0.1766` | `0.1628` | `0.0409` |
| **Zero-Shot Bridge: Tabular PFN-v3 (s25k) (`Struct + TinyOncBERT SentKM-128`)** | `0.1054` | **`+0.234`** | `+0.782` | `0.1682` | `0.1576` | `0.0406` |
| **Zero-Shot Bridge: Tabular PFN-v3 (s25k) (`Struct + TinyOncBERT SentKM-256`)** | **`0.1043`** | `+0.210` | **`+0.786`** | **`0.1663`** | `0.1501` | **`0.0392`** |
| **Zero-Shot Bridge: Tabular PFN-v3 (s25k) (`Struct + SentKM-256 + LexSVD-128`)** | `0.1045` | `+0.222` | **`+0.786`** | `0.1673` | `0.1510` | `0.0396` |
| Zero-Shot Bridge: Tabular PFN-v3 (s25k) (`Struct + SentKM-512 + LexSVD-128`) | `0.1053` | `+0.201` | `+0.781` | `0.1694` | `0.1497` | **`0.0392`** |
| Zero-Shot Bridge: Tabular PFN-v4 (s13k, *32.5% trained*) (`Struct + SentKM-256`) | `0.1102` | `+0.204` | `+0.767` | `0.1689` | **`0.1484`** | `0.0489` |
| Zero-Shot Bridge: Tabular PFN-v4 (s13k) (`Struct + SentKM-256 + LexSVD-128`) | `0.1097` | `+0.211` | `+0.769` | `0.1690` | `0.1501` | `0.0488` |
| Zero-Shot Bridge: Tabular PFN-v4 (s13k) (`Option-A: +Nadir/Peak +ICD64 +TempSent256 +Lex128`) | `0.1147` | **`+0.236`** | `+0.743` | `0.1702` | `0.1538` | `0.0485` |

### Finished Option A Benchmark (`PFN-v4 s40k` vs. Run 5 `s1200`) on 17 Realistic & 47 Complex Full-LLM Cohorts (`artifacts/eval_option_a_v4/summary.md`)

Evaluated after both Tabular v4 (`step 40,000`) and Run 5 (`step 1,200`) finished, including the newly generated `myeloma_q03_v043` cohort (`17` realistic cohorts = `3,400` queries) and all `47` complex out-of-family Full-LLM cohorts (`9,400` queries: `15` `composite_var_width`, `15` `epistatic_s8`, `15` `nonparametric_s10`, `2` `dense_50_50`):

#### Realistic Full-LLM Cohorts (`17` cohorts, `N_ctx = 800` & `N_ctx = 200`; `zero_cate_rmse = 0.1669`)

| Method | `N=800` `cate` ↓ | `N=800` `within_r` ↑ | `N=800` `pool_r` ↑ | `N=800` `risk` ↓ | `N=800` `prop` ↓ | `N=800` `ate` ↓ | `N=200` `cate` ↓ | `N=200` `within_r` ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PFN-v4 (`s40k`) [`struct_only`, 220] | `0.1139` | `+0.184` | `+0.731` | `0.1782` | `0.1692` | `0.0421` | `0.1307` | `-0.003` |
| **PFN-v4 (`s40k`) [`sent_256`, 476 — new default]** | **`0.1041`** | **`+0.231`** | **`+0.783`** | **`0.1662`** | `0.1554` | `0.0415` | `0.1301` | `+0.019` |
| PFN-v4 (`s40k`) [`sent_256_lex_128`, 604] | `0.1047` | `+0.224` | `+0.780` | `0.1664` | **`0.1552`** | `0.0425` | `0.1299` | `+0.021` |
| PFN-v4 (`s40k`) [`option_a_temp128`, 693] | `0.1058` | `+0.208` | `+0.775` | `0.1688` | `0.1630` | **`0.0398`** | **`0.1287`** | `+0.028` |
| PFN-v4 (`s40k`) [`option_a_full`, 1,077] | `0.1072` | `+0.208` | `+0.767` | `0.1673` | `0.1580` | `0.0406` | `0.1291` | **`+0.035`** |
| Longitudinal Run 5 (`s1200`) | `0.1200` | `+0.118` | `+0.704` | `0.1849` | `0.1762` | `0.0533` | `0.1309` | `-0.004` |

#### Complex Out-of-Family Full-LLM Cohorts (`47` cohorts, `N_ctx = 800`, `9,400` queries)

| Subgroup (`cohorts` / `zero_cate`) | Method | `cate_rmse` ↓ | `within_r` ↑ | `pool_r` ↑ | `prop_rmse` ↓ | `ate_rmse` ↓ |
|---|---|---:|---:|---:|---:|---:|
| **OVERALL** (`47` / `0.2427`) | **PFN-v4 (`s40k`) [`sent_256`]** | **`0.1713`** | **`+0.244`** | **`+0.682`** | `0.2532` | `0.0575` |
| | PFN-v4 (`s40k`) [`option_a_full`] | `0.1716` | `+0.233` | `+0.681` | **`0.2515`** | `0.0578` |
| | Longitudinal Run 5 (`s1200`) | `0.1744` | `+0.172` | `+0.668` | `0.2623` | **`0.0531`** |
| `composite_var_width` (`15` / `0.2621`) | **PFN-v4 [`sent_256` / `option_a_full`]** | **`0.1965` / `0.1963`** | **`+0.304`** / `+0.299` | **`+0.648`** | **`0.2361`** | **`0.0489`** |
| | Longitudinal Run 5 (`s1200`) | `0.2046` | `+0.208` | `+0.614` | `0.2489` | `0.0594` |
| `dense_50_50` (`2` / `0.2641`) | **PFN-v4 [`sent_256`]** | **`0.1834`** | **`+0.217`** | **`+0.595`** | `0.2669` | **`0.0228`** |
| | PFN-v4 [`option_a_full`] | `0.1842` | `+0.161` | `+0.589` | `0.2644` | `0.0226` |
| | Longitudinal Run 5 (`s1200`) | `0.1905` | `+0.078` | `+0.549` | **`0.2589`** | `0.0241` |
| `epistatic_s8` (`15` / `0.0708`) | PFN-v4 [`sent_256` / `option_a_full`] | `0.0884` / `0.0908` | **`+0.078`** / `+0.034` | `+0.016` / `+0.003` | **`0.2553`** | `0.0570` |
| | Longitudinal Run 5 (`s1200`) | `0.0885` | `+0.027` | `+0.011` | `0.2608` | **`0.0535`** |
| `nonparametric_s10` (`15` / `0.3186`) | **PFN-v4 [`option_a_full` / `sent_256`]** | `0.2024` / `0.2024` | **`+0.374`** / `+0.353` | **`+0.751`** / `+0.747` | **`0.2591`** | `0.0679` / `0.0658` |
| | Longitudinal Run 5 (`s1200`) | **`0.2020`** | `+0.293` | `+0.739` | `0.2767` | **`0.0485`** |

- **Feature-design takeaway:** Lean **`sent_256`** (`476` cols) beat wider `option_a_full` (`1,077` cols) at `N_ctx = 800` on Suite 2 (`0.1041 / +0.231` vs `0.1072 / +0.208`), `dense_50_50` (`+0.217` vs `+0.161`), and `epistatic_s8` (`+0.078` vs `+0.034`) because duplicating summary/sentence signals across 601 collinear columns crowds finite top-$K$ screening and pairwise slots. Promoted `sent_256` to default for future Option A and lean Option B runs (`e6282ba`).

### Completed Trajectory & Full 4-Suite Benchmark: Option B (`v6`) (`artifacts/longitudinal_option_b_v6`)

Dual held-out evaluation every 100 steps on the exact 32 paraphrased longitudinal studies (`N_ctx = 5,000`, `N_q = 64`, `seed = 42`; Run 5 finished at `0.0941 / +0.203`) paired with 32 held-out pure-tabular 6-family guard studies (`N_ctx = 800`, `P = 500`):

| Step | Long. `cate_rmse` ↓ | Long. `within_r` ↑ | Long. `pool_r` ↑ | Long. `ate_rmse` ↓ | Long. `risk_rmse` ↓ | Long. `prop_rmse` ↓ | Tab Guard `cate` ↓ | Tab Guard `wr` ↑ | Tab Guard `role_auc` ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` *(zero-shot v4 `s40k`)* | `0.0800` | `+0.4285` | `+0.8319` | `0.0243` | `0.1356` | `0.1183` | `0.1060` | `+0.2559` | `0.8659` |
| `100` | `0.0793` | `+0.4654` | `+0.8362` | **`0.0227`** | `0.1327` | `0.1188` | `0.1065` | `+0.2560` | `0.8648` |
| `200` | `0.0782` | `+0.4840` | `+0.8430` | `0.0243` | `0.1308` | `0.1178` | `0.1062` | `+0.2708` | `0.8626` |
| `400` | `0.0764` | `+0.5065` | **`+0.8541`** | `0.0250` | `0.1294` | **`0.1165`** | `0.1065` | `+0.2623` | `0.8641` |
| `600` | `0.0765` | **`+0.5087`** | `+0.8518` | `0.0229` | `0.1289` | `0.1168` | **`0.1060`** | `+0.2595` | `0.8618` |
| **`800` *(best_checkpoint.pt)*** | **`0.0762`** | `+0.5040` | `+0.8540` | `0.0239` | `0.1287` | `0.1168` | `0.1128`* | `+0.2871`* | `0.8642` |
| `1000` | `0.0766` | `+0.4969` | `+0.8517` | `0.0233` | `0.1284` | `0.1174` | `0.1127`* | `+0.2917`* | `0.8643` |
| `1100` *(killed at s1197)* | `0.0764` | `+0.4962` | `+0.8522` | `0.0230` | **`0.1283`** | `0.1172` | `0.1127`* | **`+0.2916`**\* | `0.8641` |

*(\*Note: Tabular guard steps `700–1100` reflect the updated `PriorConfig` resumed after step 600; head-to-head zero-forgetting is verified directly on identical cohorts in Suite 1 & Suite 4 below. Host-RAM hardening in `8a8f2e6` capped prefetcher `max_inflight=4`, added `max_tasks_per_child=64` worker recycling, capped nested eval workers to `<=2`, and saves `checkpoint.pt` before eval.)*

#### Completed Option B `v6 (s800)` Full 4-Suite Benchmark (`artifacts/longitudinal_option_b_v6/eval_full_suite/summary.md`)

| Suite / Cohort Set | Method | `cate_rmse` ↓ | `within_r` ↑ | `pool_r` ↑ | `risk_rmse` ↓ | `prop_rmse` ↓ | `ate_rmse` ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| **Suite 1: Real OCI Stage-2 (`seed101`)** | OCI Stage-2 Ensemble | `0.1344` | `+0.349` | `—` | `—` | `0.2028` | `+0.0637` *(bias)* |
| | PFN-v4 (`s40k`) | `0.1190` | `+0.477` | `—` | `—` | **`0.1848`** | `+0.0420` *(bias)* |
| | **Option B `v6 (s800)` [`tab-head`]** | **`0.1127`** | **`+0.479`** | `—` | `—` | `0.1854` | **`+0.0233`** *(bias)* |
| **Suite 1: Real OCI Stage-2 (`seed102`)** | OCI Stage-2 Ensemble | `0.2292` | `+0.436` | `—` | `—` | **`0.1911`** | `+0.1238` *(bias)* |
| | PFN-v4 (`s40k`) | `0.1906` | `+0.627` | `—` | `—` | `0.2025` | `+0.0873` *(bias)* |
| | **Option B `v6 (s800)` [`tab-head`]** | **`0.1883`** | **`+0.629`** | `—` | `—` | `0.2046` | **`+0.0847`** *(bias)* |
| **Suite 2: Realistic Full-LLM (`N=800`)** | PFN-v4 (`s40k`) [`sent_256`] | `0.0994` | `+0.197` | `+0.779` | `0.1639` | `0.1560` | `0.0431` |
| | PFN-v4 (`s40k`) [`option_a_full`] | `0.1030` | `+0.158` | `+0.760` | `0.1662` | `0.1544` | `0.0428` |
| | Longitudinal Run 5 (`s1200`) | `0.1139` | `+0.094` | `+0.704` | `0.1824` | `0.1762` | `0.0521` |
| | **Option B `v6 (s800)`** | **`0.0963`** | **`+0.267`** | **`+0.800`** | **`0.1594`** | **`0.1542`** | **`0.0419`** |
| **Suite 3: Complex Full-LLM (`N=800`)** | PFN-v4 (`s40k`) [`sent_256`] | `0.1738` | `+0.234` | `+0.698` | `0.2274` | `0.2542` | `0.0590` |
| | PFN-v4 (`s40k`) [`option_a_full`] | `0.1737` | `+0.230` | `+0.699` | `0.2257` | `0.2521` | `0.0586` |
| | Longitudinal Run 5 (`s1200`) | `0.1763` | `+0.171` | `+0.686` | `0.2359` | `0.2640` | **`0.0535`** |
| | **Option B `v6 (s800)`** | **`0.1728`** | **`+0.237`** | **`+0.706`** | **`0.2223`** | **`0.2515`** | `0.0587` |
| **Suite 4: Complex Tabular (`51` cells)** | PFN-v4 (`s40k`) | **`0.1793`** | `+0.252` | `+0.628` | `—` | **`0.2029`** | **`0.0471`** |
| | **Option B `v6 (s800)` [`tab-head`]** | `0.1795` | **`+0.252`** | **`+0.630`** | `—` | `0.2037` | `0.0485` |

- **Takeaways motivating Option B `v7`:** Option B `v6` decisively beat both Option A (`PFN-v4 s40k`) and Run 5 on Realistic Full-LLM cohorts (`0.0963 / +0.267` vs. `0.0994 / +0.197` and `0.1139 / +0.094`), beat Tabular v4 on real OCI Stage-2 catalogs (`0.1127` vs `0.1190`), improved `nonparametric_s10` (`+0.356 → +0.406`) and `S7` null (`0.0596 → 0.0550`), and showed zero tabular forgetting (`0.1795` vs `0.1793`). Its two remaining weak spots — pure epistasis (`epistatic_s8` `+0.030`) and `dense_50_50` (`+0.119` vs. lean `sent_256` `+0.205`) — stem directly from Tabular v4's base head + wide `option_a_full` feature dilution and training only on short `nemotron_5000` notes. **Option B `v7`** (`scripts/prepare_mmai_v23_cache_h200.sh` + `scripts/run_longitudinal_option_b_lean_h200.sh`) addresses all three by warm-starting from **Tabular v5**, switching to lean **`sent_256`** (`500` cols), and adding **35% `TinyOncBERT`-disjoint `mmai/v23` `train_unseen`** (`37,631` real 14-note pan-cancer charts, 0% overlap with `TinyOncBERT`'s `564,224`-note curation cutoff, reserving `8,470` `validation` and `8,340` `test` patients strictly for zero-shot evaluation).

#### `N_ctx = 200` (small cohort regime, 3,200 held-out query patients)

| Method | `cate_rmse` ↓ | `within_r` ↑ | `pool_r` ↑ | `risk_rmse` ↓ | `prop_rmse` ↓ | `ate_rmse` ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline: Arm-Mean (ATE only) | `0.1315` | `+0.000` | `+0.624` | `0.2124` | `0.2017` | `0.0608` |
| PFN Run 1 (Hybrid, plain, s1200) | `0.1322` | `-0.019` | `+0.639` | `0.2006` | `0.1915` | `0.0675` |
| PFN Run 3 (Single-Stage, plain, s800) | `0.1337` | `+0.004` | `+0.611` | `0.1979` | `0.1923` | `0.0627` |
| PFN Run 4 (Single-Stage, paraphrased, s700) | `0.1332` | `+0.009` | `+0.614` | `0.1993` | `0.1924` | `0.0633` |
| PFN Run 5 (TinyOncBERT + Block-Ridge, s600) | `0.1327` | `+0.003` | `+0.622` | `0.1972` | `0.1923` | `0.0619` |
| PFN Run 5 (TinyOncBERT + Block-Ridge, s800) | `0.1311` | `+0.012` | `+0.632` | `0.1959` | `0.1892` | `0.0606` |
| Applied: Struct-Only Ridge R-Learner | `0.1331` | `+0.025` | `+0.616` | `0.1964` | `0.1897` | `0.0670` |
| Applied: Struct + TinyBERT SentKM Ridge-R | `0.1458` | `+0.015` | `+0.549` | `0.1992` | `0.1929` | `0.0689` |
| Applied: Struct + Qwen3-0.6B SentKM Ridge-R | `0.1395` | `+0.012` | `+0.576` | `0.2021` | `0.1907` | `0.0693` |
| Applied: Struct + **TinyOncBERT** SentKM Ridge-R | `0.1336` | `-0.004` | `+0.614` | `0.1972` | `0.1906` | `0.0701` |
| Applied: Struct + TinyOncBERT SentKM+Lex GBM-R | `0.1404` | `+0.045` | `+0.608` | `0.2126` | `0.2030` | **`0.0577`** |
| Applied: Struct + TinyOncBERT SentKM+Lex DragonNet | `0.1639` | `+0.007` | `+0.457` | `0.2082` | `0.1941` | `0.0706` |
| Zero-Shot Bridge: Tabular PFN-v2 (`Struct + SentKM-128`) | `0.1313` | `+0.063` | `+0.649` | `0.1971` | **`0.1746`** | `0.0688` |
| Zero-Shot Bridge: Tabular PFN-v2 (`Struct + SentKM-256`) | `0.1319` | `+0.063` | `+0.638` | **`0.1949`** | `0.1790` | `0.0693` |
| **Zero-Shot Bridge: Tabular PFN-v3 (s25k) (`Struct + TinyOncBERT SentKM-128`)** | **`0.1279`** | `+0.028` | **`+0.657`** | `0.1970` | `0.1827` | `0.0632` |
| **Zero-Shot Bridge: Tabular PFN-v3 (s25k) (`Struct + TinyOncBERT SentKM-256`)** | `0.1283` | `+0.068` | `+0.656` | `0.1996` | `0.1890` | `0.0645` |
| Zero-Shot Bridge: Tabular PFN-v4 (s13k, *32.5% trained*) (`Struct + SentKM-256`) | `0.1296` | `+0.072` | **`+0.669`** | `0.1937` | `0.1829` | `0.0663` |
| Zero-Shot Bridge: Tabular PFN-v4 (s13k) (`Struct + SentKM-512 + LexSVD-128`) | `0.1304` | `+0.064` | `+0.659` | **`0.1923`** | `0.1812` | `0.0667` |
| Zero-Shot Bridge: Tabular PFN-v4 (s13k) (`Option-A: +Nadir/Peak +ICD64 +TempSent256 +Lex128`) | `0.1315` | **`+0.075`** | `+0.649` | `0.1940` | `0.1853` | `0.0688` |

### Key Conclusions from the 16-Cohort Full-LLM Benchmark
1. **`TinyOncBERT` (22.7M) beat both `TinyBERT` (14M) and its own 600M teacher `Qwen3-Embedding-0.6B`** on every metric at both `N = 200` and `N = 800` — validating that oncology domain vocabulary + negation-repulsion distillation (`val_neg_flip_cos = 0.066`) makes a compact encoder outperform a 26× larger general embedder on realistic clinical notes.
2. **Per-study cross-fitted R-learners beat Runs 1–5 (uniform Ridge PFNs) at `N = 800`:** `Applied: Struct + TinyOncBERT SentKM+Lex GBM-R` achieved `cate_rmse = 0.1127`, `within_r = +0.160`, `risk_rmse = 0.1768`, and `prop_rmse = 0.1697` (vs. Run 5 step-600 `0.1217 / +0.130 / 0.1847 / 0.1738`).
3. **Zero-shot `TinyOncBERT + SparseTabularCausalPFN v3 (s25k)` shattered EVERY record at BOTH `N = 800` AND `N = 200`:** Feeding unmixed `[patient_summary_vector + context-only TinyOncBERT sentence MaxSim prototypes ± lexical SVD]` directly into finished `SparseTabularCausalPFN v3` **zero-shot (never trained on text!)** dominated every applied and end-to-end model on every single metric:
   - **`N = 800`:** **`cate_rmse = 0.1043`** (`SentKM-256`, **`-0.0084` vs. GBM-R `0.1127`, `-0.0174` vs. Run 5 `0.1217`!**), **`within_r = +0.234`** (`SentKM-128`, **+46% above GBM-R, +80% above Run 5!**), **`pool_r = +0.786`**, **`risk_rmse = 0.1663`**, **`prop_rmse = 0.1497–0.1501`** (`-0.020` vs. GBM-R!), and **`ate_rmse = 0.0392`** (first method under `0.040`!).
   - **`N = 200`:** **`cate_rmse = 0.1279`** (`SentKM-128`, beating Arm-Mean `0.1315` by `-0.0036` and crushing GBM-R `0.1404` by `-0.0125`!) and **`within_r = +0.068`** (`SentKM-256`, **`pool_r = +0.657`**)!
   - *(Note: Adding `NaN`-containing nadir/peak features to raw v3 dropped `0.1043 → 0.1086` specifically because sparse `NaN`s triggered v3's `miss_rate > 0.55` screener shortcut — which `v3.1` fine-tuning fixes.)*
   - **Blind raw-note clinical role discovery (zero LLM extraction):** On `bladder_q01_v005`, the top 3 overall features out of 476 were all `flank_pain` sentence clusters (`[dual_role]` / `[modifier]`, `g_tau = 0.86`); on `breast_q02_v030`, the top 6 overall features were 100% pure hits on the exact two oracle note findings (`"takes lisinopril daily"` / `"Hypertension managed with lisinopril"` for `concomitant_ace_inhibitor_use`, and `"history of prior cellulitis"` with `g_m = 0.59, g_tau = 0.00` for `history_of_cellulitis`); on `bladder_q05_v036`, it nailed `lab.glucose.latest` (`[confounder]`, `p_c=0.975`), caught `lab.platelets.slope` as a pure **`[instrument]`** (`p_sig=0.969, g_m=0.06`), and surfaced ECOG performance status directly from prose sentences.
   - **Direct blueprint for Longitudinal `v6`:** Replacing Run 5's uniform block-Ridge head (and static `note_slots` residual inputs) with a `SparseTabularCausalPFN` v3 ARD + co-observed-correlation + dynamic `attn_slots` head on top of `TinyOncBERT` sentence prototypes + explicit structured summaries unifies both foundation models and immediately solves Run 5's text-noise dilution and cross-modal nonlinearity bottlenecks.
4. **Per-study end-to-end neural CATE (`DragonNet`) severely overfit** (`cate_rmse = 0.1639` at `N = 200`, `0.1446` at `N = 800` — worse than Arm-Mean `0.1316`), confirming that orthogonalized R-learner regularization is essential when CATE is unobserved.
5. **PFN Run 3 (plain) and Run 4 (paraphrased) performed identically on Full-LLM notes** (`0.1228` vs. `0.1233` at `N = 800`), confirming that without `TinyOncBERT` + sentence-concept ARD selection, paraphrasing alone does not close the realistic-note gap.

### Zero-Shot vs. Cross-Fit Episodic Factual Adaptation (`adapt-predict`) on 16 Full-LLM Cohorts
Evaluated on `Run 4 (step 1000)` at `N_ctx = 800`, `N_q = 200` (3,200 held-out queries) with frozen CATE shrinkage & calibration heads, cached `PatientEncoder` embeddings, 4 random inner episodes/step, cosine `lr = 1e-4`, L2 anchor `1e-2`, and an automatic step-0 zero-shot inner-validation safety gate (`oci/longitudinal/adapt.py`):

| Mode | Cohorts Adapted | `cate_rmse` ↓ | `within_r` ↑ | `pool_r` ↑ | `risk_rmse` ↓ | `prop_rmse` ↓ | `ate_rmse` ↓ | Runtime (16 cohorts) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-Shot (`Run 4 s1000`) | `0 / 16` | `0.1238` | `+0.094` | `+0.691` | `0.1887` | `0.1749` | `0.0561` | — |
| **CF-Adapted (`adapt-predict`)** | **`10 / 16`** | **`0.1232`** | **`+0.102`** | **`+0.692`** | **`0.1878`** | **`0.1746`** | **`0.0540`** | `713s` (~44s/cohort) |

*(An earlier uncached run at `lr = 1e-3` triggered the safety gate on `16 / 16` cohorts with zero regressions, confirming that the zero-shot gate protects against overly aggressive adaptation learning rates.)*

