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
| **Run 3 (Single-Stage Plain)** | `longitudinal_tinybert_single_stage` | Run 2 + single-stage log-uniform `N_ctx=100..10,000`, `--per-slot-note-temporal`, `--ridge-grid-size 7`, 32-tok overlapping note windows | Plain | Active (step 700 / 1200) | **`0.0955`** | **`+0.2171`** | **`+0.7494`** | **`0.1533`** | **`0.1235`** | **`0.0292`** |
| **Run 4 (Single-Stage Paraphrased)** | `longitudinal_tinybert_paraphrased` | Exact 1-variable ablation of Run 3 with `--paraphrase-bank` (`nemotron_5000_paraphrases.json.gz`, 2,902 findings × ~10 paraphrases + EHR sections + filler) | Paraphrased | Active (step 500 / 1200) | `0.0996` | `+0.1466` | `+0.7270` | `0.1886` | `0.1707` | `0.0368` |

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

**Run 4 (Single-Stage Paraphrased, evaluated on paraphrased held-out notes):**

| Step | `cate_rmse` | `within_r` | `pool_r` | `risk_rmse` | `prop_rmse` | `ate_rmse` |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | `0.1013` | `+0.1216` | `+0.7098` | `0.1905` | `0.1741` | `0.0362` |
| 200 | `0.1007` | `+0.1366` | `+0.7193` | `0.1899` | `0.1727` | `0.0374` |
| 300 | `0.1006` | `+0.1524` | `+0.7179` | `0.1894` | `0.1721` | `0.0379` |
| 400 | `0.1009` | `+0.1346` | `+0.7189` | `0.1891` | `0.1716` | `0.0388` |
| 500 | `0.0996` | `+0.1466` | `+0.7270` | `0.1886` | `0.1707` | `0.0368` |

---

## 2. Final 1,000-Study Held-Out Evaluation (`N_ctx = 5,000`, `64,000` query patients, `seed = 42`)

Baselines across all 1,000 held-out studies: `zero = 0.1455`, observational LOO-CV `ridge = 0.1199`, `arm_mean = 0.1140`.

| Run | `cate_rmse` ↓ | `within_r` ↑ | `pool_r` ↑ | `spearman` ↑ | `risk_rmse` ↓ | `prop_rmse` ↓ | `ate_rmse` ↓ | `ate_bias` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Run 1 (Hybrid)** | `0.1060` | `+0.1021` | `+0.6859` | `+0.7411` | `0.2031` | `0.1694` | `0.0471` | `-0.0030` |
| **Run 2 (Wide Hybrid)** | **`0.1028`** | **`+0.1440`** | **`+0.7069`** | **`+0.7584`** | **`0.1791`** | **`0.1275`** | **`0.0395`** | **`+0.0010`** |

### Zero-Shot Generalization to Unseen Cancers (Run 2, 1,000 studies)

| Split Group | Studies | `cate_rmse` | `within_r` | `pool_r` | `risk_rmse` | `prop_rmse` | `ate_rmse` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `seen_cancer_new_question` | 250 | `0.1028` | `+0.1422` | `+0.7213` | `0.1791` | `0.1271` | `0.0390` |
| `unseen_cancer` (3 held-out cancers) | 750 | `0.1028` | `+0.1446` | `+0.7017` | `0.1791` | `0.1276` | `0.0397` |

### Performance by Candidate Width (Run 2, 1,000 studies)

| Candidate Width | Studies | `cate_rmse` | `within_r` | `risk_rmse` | `prop_rmse` | `ate_rmse` |
|---|---:|---:|---:|---:|---:|---:|
| `3–20` | 195 | `0.0957` | `+0.1454` | `0.1667` | `0.1023` | `0.0388` |
| `21–50` | 308 | `0.1036` | `+0.1519` | `0.1759` | `0.1240` | `0.0373` |
| `51–80` | 295 | `0.1044` | `+0.1334` | `0.1836` | `0.1356` | `0.0405` |
| `81–100` | 202 | `0.1058` | `+0.1463` | `0.1887` | `0.1416` | `0.0420` |

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
