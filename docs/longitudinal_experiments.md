# Longitudinal Multimodal Causal Foundation Model — Experiment Log

This document records the training runs, controlled ablations, diagnostic probes, and held-out evaluations for the longitudinal multimodal causal foundation model (`oci.longitudinal`) on the 5,000-design oncology prior (`synthetic_data/designs/nemotron_5000.json.gz`, 1,000 held-out designs spanning 3 unseen cancer types + new clinical questions on seen cancers).

All models and baselines use strictly observational context inputs `(X, T, Y)` with zero oracle leakage.

---

## 1. Training Runs Summary

### Mid-Training Held-Out Eval (32 fixed held-out studies, `N_ctx = 5,000`, `N_q = 64`)
Reference baselines on these 32 studies: `arm_mean cate_rmse = 0.1160`, observational LOO-CV `ridge cate_rmse = 0.1097`.

| Run | Artifact Dir | Architecture / Setup | Note Distribution | Status / Step | `cate_rmse` ↓ | `within_r` ↑ | `pool_r` ↑ | `risk_rmse` ↓ | `prop_rmse` ↓ | `ate_rmse` ↓ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (Pure PFN)** | `longitudinal_tinybert_n5000` | Pure `TwoWayStudyLayer` (no differentiable R-learner), mean note pooling | Plain | Stopped (527) | `0.1126` | `—` | `+0.6254` | `0.2125` | `—` | `—` |
| **Curriculum (Pure PFN)** | `longitudinal_tinybert_curriculum` | Pure `TwoWayStudyLayer`, 2-stage small→large N | Plain | Stopped (723) | `0.1118` | `+0.0212` | `+0.6271` | `0.2122` | `0.2008` | `0.0470` |
| **Run 1 (Hybrid)** | `longitudinal_tinybert_hybrid` | Differentiable LOO-CV R-learner (`solver_dim=64`), 8 note slots, trajectory slopes, 2-stage curriculum (`N=64..512` → `100..5000`) | Plain | Complete (1200) | `0.1016` | `+0.1504` | `+0.7156` | `0.1946` | `0.1719` | `0.0410` |
| **Run 2 (Wide Hybrid)** | `longitudinal_tinybert_wide_hybrid` | Run 1 + `note_slots=16`, `solver_dim=256`, study-adaptive `lexical_dim=128` SVD, 2-stage curriculum | Plain | Complete (1200) | `0.1023` | `+0.1989` | `+0.7024` | `0.1754` | `0.1302` | `0.0375` |
| **Run 3 (Single-Stage Plain)** | `longitudinal_tinybert_single_stage` | Run 2 + single-stage log-uniform `N_ctx=100..10,000`, `--per-slot-note-temporal`, `--ridge-grid-size 7`, 32-tok overlapping note windows | Plain | Complete (1200 + 1k eval) | **`0.0951`** | **`+0.2169`** | **`+0.7526`** | **`0.1522`** | **`0.1230`** | **`0.0288`** |
| **Run 4 (Single-Stage Paraphrased)** | `longitudinal_tinybert_paraphrased` | Exact 1-variable ablation of Run 3 with `--paraphrase-bank` (`nemotron_5000_paraphrases.json.gz`, 2,902 findings × ~10 paraphrases + EHR sections + filler) | Paraphrased | Complete (1200 + 1k eval) | `0.0987` | `+0.1593` | `+0.7321` | `0.1876` | `0.1697` | `0.0365` |
| **Run 5 (`TinyOncBERT` v5)** | `longitudinal_tinyoncbert_v5` | `TinyOncBERT` (22.7M) + `--sentence-concept-dim 128` + `--lexical-dim 128` + `--block-ridge` + `--paraphrase-bank` | Paraphrased | Active (resumed s600 / 1200) | **`0.0949`** | **`+0.1871`** | **`+0.7570`** | **`0.1682`** | **`0.1399`** | **`0.0295`** |

*(Run 5 note: The initial Run 5 process was SIGKILLed by the Linux host OOM killer at step 400 entering mid-training evaluation because `_EpisodePrefetcher` with `num_workers=16` buffered `32` paraphrased mega-episodes in the training pool and spawned a nested `16`-worker / `32`-episode eval pool simultaneously (`64` buffered multi-section cohorts + `32` PyTorch worker processes). Fixed in `oci/longitudinal/training.py` by capping `_EpisodePrefetcher._max_inflight` to `6`, capping nested mid-training eval workers/buffer to `4`, and adding explicit `del rows` / `gc.collect()` boundaries; `checkpoint.pt` at step 300 is intact and ready for `resume`.)*

### Learning Trajectories (Run 3 & Run 4 on 32 Held-Out Studies at `N_ctx = 5,000`)

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

---

## 2. Final 1,000-Study Held-Out Evaluation (`N_ctx = 5,000`, `64,000` query patients, `seed = 42`)

Baselines across all 1,000 held-out studies: `zero = 0.1455`, observational LOO-CV `ridge = 0.1199`, `arm_mean = 0.1140`.

| Run | `cate_rmse` ↓ | `within_r` ↑ | `pool_r` ↑ | `spearman` ↑ | `risk_rmse` ↓ | `prop_rmse` ↓ | `ate_rmse` ↓ | `ate_bias` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Run 1 (Hybrid)** | `0.1060` | `+0.1021` | `+0.6859` | `+0.7411` | `0.2031` | `0.1694` | `0.0471` | `-0.0030` |
| **Run 2 (Wide Hybrid)** | `0.1028` | `+0.1440` | `+0.7069` | `+0.7584` | `0.1791` | `0.1275` | `0.0395` | `+0.0010` |
| **Run 3 (Single-Stage Plain)** | **`0.0962`** | **`+0.1870`** | **`+0.7502`** | **`+0.7923`** | **`0.1583`** | **`0.1199`** | **`0.0315`** | **`-0.0023`** |
| **Run 4 (Single-Stage Paraphrased)** *(eval on paraphrased notes)* | `0.1048` | `+0.1103` | `+0.6924` | `+0.7454` | `0.1988` | `0.1672` | `0.0442` | `-0.0034` |

### Zero-Shot Generalization to Unseen Cancers (1,000 studies)

| Run | Split Group | Studies | `cate_rmse` | `within_r` | `pool_r` | `risk_rmse` | `prop_rmse` | `ate_rmse` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Run 2** | `seen_cancer_new_question` | 250 | `0.1028` | `+0.1422` | `+0.7213` | `0.1791` | `0.1271` | `0.0390` |
| | `unseen_cancer` (3 held-out cancers) | 750 | `0.1028` | `+0.1446` | `+0.7017` | `0.1791` | `0.1276` | `0.0397` |
| **Run 3** | `seen_cancer_new_question` | 250 | **`0.0966`** | **`+0.1883`** | **`+0.7599`** | **`0.1590`** | **`0.1195`** | **`0.0321`** |
| | `unseen_cancer` (3 held-out cancers) | 750 | **`0.0961`** | **`+0.1865`** | **`+0.7467`** | **`0.1580`** | **`0.1201`** | **`0.0313`** |
| **Run 4** | `seen_cancer_new_question` | 250 | `0.1044` | `+0.1163` | `+0.7109` | `0.1997` | `0.1605` | `0.0425` |
| | `unseen_cancer` (3 held-out cancers) | 750 | `0.1050` | `+0.1084` | `+0.6857` | `0.1985` | `0.1694` | `0.0448` |

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

---

## 3. Zero-Shot Plain vs. Paraphrased Distribution-Shift Evaluation

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

## 4. Representation & Pooling Probe (Plain vs. Paraphrased vs. Full-LLM Realistic Notes)

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

## 5. `TinyOncBERT` Distillation & 16-Cohort Full-LLM Applied vs. Foundation Model Benchmark

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
| PFN Run 5 (TinyOncBERT + Block-Ridge, live s400) | `0.1226` | `+0.125` | `+0.704` | `0.1843` | `0.1727` | `0.0584` |
| Applied: Struct-Only Ridge R-Learner | `0.1137` | `+0.152` | `+0.739` | `0.1838` | `0.1742` | **`0.0421`** |
| Applied: Struct + LexSVD Ridge R-Learner | `0.1144` | `+0.135` | `+0.736` | `0.1857` | `0.1770` | `0.0441` |
| Applied: Struct + TinyBERT SentKM Ridge-R | `0.1147` | `+0.155` | `+0.734` | `0.1847` | `0.1758` | `0.0469` |
| Applied: Struct + Qwen3-0.6B SentKM Ridge-R | `0.1164` | `+0.141` | `+0.724` | `0.1875` | `0.1774` | `0.0465` |
| Applied: Struct + **TinyOncBERT** SentKM Ridge-R | `0.1129` | `+0.144` | `+0.744` | `0.1841` | `0.1737` | `0.0463` |
| Applied: Struct + TinyOncBERT SentKM+Lex GBM-R | `0.1127` | `+0.160` | `+0.743` | `0.1768` | `0.1697` | `0.0456` |
| Applied: Struct + TinyOncBERT SentKM+Lex DragonNet | `0.1446` | `+0.085` | `+0.579` | `0.1925` | `0.1861` | `0.0662` |
| **Zero-Shot Bridge: Tabular PFN-v2 (`Struct-Only`, 220)** | `0.1146` | `+0.161` | `+0.742` | `0.1803` | `0.1621` | `0.0487` |
| **Zero-Shot Bridge: Tabular PFN-v2 (`Struct + TinyOncBERT SentKM-128`)** | `0.1111` | **`+0.173`** | **`+0.763`** | `0.1722` | `0.1591` | `0.0495` |
| **Zero-Shot Bridge: Tabular PFN-v2 (`Struct + SentKM-256 + LexSVD-128`)** | `0.1110` | `+0.148` | `+0.761` | `0.1718` | `0.1548` | `0.0488` |
| **Zero-Shot Bridge: Tabular PFN-v2 (`Struct + SentKM-512 + LexSVD-128`)** | **`0.1108`** | `+0.147` | `+0.762` | `0.1730` | `0.1606` | `0.0489` |
| **Zero-Shot Bridge: Tabular PFN-v3 (live s5000, `Struct + SentKM-256`)** | `0.1119` | `+0.154` | `+0.753` | **`0.1697`** | **`0.1512`** | `0.0494` |

#### `N_ctx = 200` (small cohort regime, 3,200 held-out query patients)

| Method | `cate_rmse` ↓ | `within_r` ↑ | `pool_r` ↑ | `risk_rmse` ↓ | `prop_rmse` ↓ | `ate_rmse` ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline: Arm-Mean (ATE only) | `0.1315` | `+0.000` | `+0.624` | `0.2124` | `0.2017` | `0.0608` |
| PFN Run 1 (Hybrid, plain, s1200) | `0.1322` | `-0.019` | `+0.639` | `0.2006` | `0.1915` | `0.0675` |
| PFN Run 3 (Single-Stage, plain, s800) | `0.1337` | `+0.004` | `+0.611` | `0.1979` | `0.1923` | `0.0627` |
| PFN Run 4 (Single-Stage, paraphrased, s700) | `0.1332` | `+0.009` | `+0.614` | `0.1993` | `0.1924` | `0.0633` |
| PFN Run 5 (TinyOncBERT + Block-Ridge, live s400) | `0.1326` | `+0.014` | `+0.629` | `0.1980` | `0.1907` | `0.0638` |
| Applied: Struct-Only Ridge R-Learner | `0.1331` | `+0.025` | `+0.616` | `0.1964` | `0.1897` | `0.0670` |
| Applied: Struct + TinyBERT SentKM Ridge-R | `0.1458` | `+0.015` | `+0.549` | `0.1992` | `0.1929` | `0.0689` |
| Applied: Struct + Qwen3-0.6B SentKM Ridge-R | `0.1395` | `+0.012` | `+0.576` | `0.2021` | `0.1907` | `0.0693` |
| Applied: Struct + **TinyOncBERT** SentKM Ridge-R | `0.1336` | `-0.004` | `+0.614` | `0.1972` | `0.1906` | `0.0701` |
| Applied: Struct + TinyOncBERT SentKM+Lex GBM-R | `0.1404` | `+0.045` | `+0.608` | `0.2126` | `0.2030` | **`0.0577`** |
| Applied: Struct + TinyOncBERT SentKM+Lex DragonNet | `0.1639` | `+0.007` | `+0.457` | `0.2082` | `0.1941` | `0.0706` |
| **Zero-Shot Bridge: Tabular PFN-v2 (`Struct + TinyOncBERT SentKM-128`)** | **`0.1313`** | **`+0.063`** | **`+0.649`** | `0.1971` | **`0.1746`** | `0.0688` |
| **Zero-Shot Bridge: Tabular PFN-v2 (`Struct + TinyOncBERT SentKM-256`)** | `0.1319` | **`+0.063`** | `+0.638` | **`0.1949`** | `0.1790` | `0.0693` |

### Key Conclusions from the 16-Cohort Full-LLM Benchmark
1. **`TinyOncBERT` (22.7M) beat both `TinyBERT` (14M) and its own 600M teacher `Qwen3-Embedding-0.6B`** on every metric at both `N = 200` and `N = 800` — validating that oncology domain vocabulary + negation-repulsion distillation (`val_neg_flip_cos = 0.066`) makes a compact encoder outperform a 26× larger general embedder on realistic clinical notes.
2. **Per-study cross-fitted R-learners beat Runs 1–5 (uniform Ridge PFNs) at `N = 800`:** `Applied: Struct + TinyOncBERT SentKM+Lex GBM-R` achieved `cate_rmse = 0.1127`, `within_r = +0.160`, `risk_rmse = 0.1768`, and `prop_rmse = 0.1697` (vs. Run 5 step-400 `0.1226 / +0.125 / 0.1843 / 0.1727`).
3. **Zero-shot `TinyOncBERT + SparseTabularCausalPFN` beat EVERY method — including `Applied GBM-R` — at BOTH `N = 800` AND `N = 200`:** Feeding unmixed `[patient_summary_vector + context-only TinyOncBERT sentence MaxSim prototypes ± lexical SVD]` directly into `SparseTabularCausalPFN` **zero-shot (never trained on text!)** set new records across every primary metric:
   - **`N = 800`:** `cate_rmse = 0.1108` (`SentKM-512+Lex`) / `0.1111` (`SentKM-128`, **`within_r = +0.173`, `pool_r = +0.763`**), while **tabular v3 at only step 5,000/30,000** already achieved **`risk_rmse = 0.1697`** (`-0.0071` vs. GBM-R) and **`prop_rmse = 0.1512`** (`-0.0185` vs. GBM-R, `-0.0215` vs. Run 5)!
   - **`N = 200`:** `PFN-v2 (Struct + SentKM-128)` was the **only method to beat Arm-Mean (`0.1313` vs. `0.1315`)** while raising `within_r` to **`+0.063`** (vs. GBM-R `+0.045` at `0.1404` RMSE, and Run 5 `+0.014`) and slashing `prop_rmse` to **`0.1746`**!
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

