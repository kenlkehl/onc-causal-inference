# Nemotron oncology study design bank

The default longitudinal training bank contains **5,000 designs**. Nemotron authored five clinical questions for each of 20 cancers and selected features and permitted causal roles separately for 50 variants of each question. There are **3 through 100 distinct candidate features** per design. Widths 3 and 4 occur 52 times each. Every other width occurs 51 times. The bank contains 257,404 feature selections in total. [Frozen bank](../synthetic_data/designs/nemotron_5000.json.gz)

## What the model generated

On November 19, 2026, the vLLM server at `http://172.24.216.113:8002/v1` served `RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8`. The inputs were the following 20 cancer seeds, their diagnosis codes, vocabulary topics, and the simulator's existing executable measurements. [Seeds, prompts, and generation code](../synthetic_data/nemotron_bank.py)

| Cancer seed | ICD-10-CM | Cancer seed | ICD-10-CM |
|---|---|---|---|
| Lung | C34.90 | Esophageal | C15.9 |
| Colon | C18.9 | Rectal | C20 |
| Breast | C50.919 | Hepatocellular carcinoma | C22.0 |
| Pancreatic | C25.9 | Intrahepatic cholangiocarcinoma | C22.1 |
| Ovarian | C56.9 | Cervical | C53.9 |
| Prostate | C61 | Endometrial | C54.1 |
| Cutaneous melanoma | C43.9 | Thyroid | C73 |
| Kidney | C64.9 | Testicular | C62.90 |
| Bladder | C67.9 | Malignant glioma | C71.9 |
| Gastric | C16.9 | Multiple myeloma | C90.00 |

1. **Questions.** One request per cancer produced five different comparisons, each with an eligible population, control and treatment arms, a binary event, and a horizon of 7–1,825 days. For example, `colon_q01` compares FOLFOX plus cetuximab with FOLFOX for grade 3 or higher diarrhea by day 180 in adults starting first-line treatment for metastatic RAS wild-type colon cancer with ECOG 0–1.
2. **Feature vocabulary.** Nemotron proposed binary pretreatment findings for eight shared oncology topics and twelve topics per cancer (248 requests). Each proposal included a simulated prevalence and two sentences for charting presence and absence. Python filtered proposals for schema, valid prevalence, distinct chart sentences, overlap with measured physiology, and study-label leakage. It deduplicated identifiers and gave a cancer-specific definition priority over a shared one. Each cancer's catalog contains 312–335 candidates, including 61 executable measurements from the patient histories. The same feature can appear in multiple questions and splits. [Measurement definitions and validation](../synthetic_data/study_designs.py)
3. **Individual designs.** Python assigned each design its feature count and offered a seeded menu of distinct alternatives for each slot. Nemotron received the clinical question, the cancer-specific catalog, and the menus. In a separate completion for each design it selected one feature and its role policy per slot. The compact output uses one letter per selection. For example, `A` selects the first alternative as a potential confounder and `H` selects the second as a potential effect modifier. Disjoint menus guarantee distinct feature names within a design. Python expands the selections into the simulator schema and rejects repeated feature-and-role sets within a question.

For example, the three candidates in `lung_q02_v002` are a contralateral lung nodule, an ALK rearrangement, and vertebral column invasion. Nemotron allowed the first two to act as effect modifiers or prognostic variables, and the third as an instrument. Each can also be unused in a particular simulated study.

The six policies permit a confounder or prognostic role, an effect-modifier or prognostic role, a dual or single confounder/modifier role, prognostic only, instrument only, or always unused. Each of the first five also permits an unused draw. The width counts **all** selected candidates, including distractors. At study simulation time a separate DGP seed determines realized roles and equations, and a patient seed determines patient histories and label draws. Numerical effects, prevalence and eligibility text are synthetic. The patient simulator samples chart findings independently and does not enforce every eligibility criterion in the question text. [Study and patient generation](synthetic_data.md#generating-a-bank-of-study-designs)

Across the bank, there are 2,409 distinct feature identifiers. Chart findings account for 208,524 selections (81.0%) and executable measurements for 48,880 (19.0%). The model designated 68,976 selections (26.8%) as always unused. In the 51 designs with exactly 100 candidates, the number permitted an active role ranges from 5 to 100, with a median of 80. Different DGP draws can activate different subsets.

## Seeds and held-out questions

The master seed was **20261119**. Seed derivation takes the first four bytes of the SHA-256 digest of the compact JSON array `[master, *parts]`, reads them as a big-endian unsigned integer, and masks to 31 bits. The generator uses distinct parts for each operation.

| Operation | Seed parts after the master |
|---|---|
| Sample cancer and question holdouts | `"split"` |
| Shuffle the cyclic 3–100 width schedule across the 5,000 design IDs | `"widths"` |
| Draw non-overlapping feature menus from the ordered cancer catalog | `"menu", design_id` |
| Sample a vLLM completion | `"request", request_key, zero_based_attempt` |

All 50 variants of a question share a split. **Training** has 80 questions and 4,000 designs. **Held-out evaluation** has 20 questions and 1,000 designs. Of those, 750 designs cover three cancers entirely absent from training (bladder, gastric, endometrial). The other 250 cover five new questions in represented cancers (`breast_q02`, `liver_q05`, `cervix_q04`, `glioma_q05`, `myeloma_q03`). Designs are ordered by variant and then by question so that a short held-out run visits different questions.

## Reproducing the bank

The 5,268 accepted responses comprise 20 question batches, 248 vocabulary batches, and 5,000 design completions. The receipt archive records each raw accepted response, the model and server fingerprint, request ID, attempt number, sampling seed, token usage, and content hashes. The server used vLLM 0.31.0. The generation client used Python 3.13.7 and openai 2.52.0. Requests used temperature 0.75, top-p 0.95, top-k 40, and disabled thinking. Questions and vocabulary used JSON output with a 2,400-token limit. Design selections used a width-specific regex and a limit of `4 × width + 32` tokens. [Generation receipts](nemotron_5000_receipts.json.gz)

For **exact offline reconstruction**, run from the repository root with the project dependencies installed. Replay revalidates each accepted response against its request identity and checksum, then checks the SHA-256 of the reconstructed JSON against the archived digest.

```bash
python -m synthetic_data.nemotron_bank \
  --replay docs/nemotron_5000_receipts.json.gz \
  --output /tmp/replayed-nemotron-5000.json.gz
cmp synthetic_data/designs/nemotron_5000.json.gz /tmp/replayed-nemotron-5000.json.gz
```

To make a **fresh live generation** with the original settings, use a new work directory. Reusing a work directory resumes its accepted requests. The initial run piloted widths 3, 10, 30, 60, and 100, then reused those five completions in the full bank. Fresh server responses can vary with serving configuration and batching even when sampling seeds match.

```bash
python -m synthetic_data.nemotron_bank \
  --api-url http://172.24.216.113:8002/v1 \
  --model RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8 \
  --seed 20261119 --workers 48 --work-dir /tmp/nemotron-live/work \
  --output /tmp/nemotron-live/bank.json.gz \
  --receipts /tmp/nemotron-live/receipts.json.gz
```

## Checksums and provenance

All 5,268 accepted responses reported the requested model and the server fingerprint `vllm-0.31.0-ep-8831b8f7`. Each had a distinct request ID. The generator accepted 5,243 responses on their first attempt and 25 on their second. Accepted responses account for 24,075,066 prompt tokens and 386,317 completion tokens.

Offline replay reproduced the compressed bank byte for byte. A simulator check drew one study and one patient for every design, using `20261119 + i` for both the DGP seed and patient seed, where `i` is the zero-based position in the bank and the patient ID is 0. All 5,000 patients had five history streams, every requested oracle feature, a binary observed treatment and outcome, and finite probabilities in [0, 1] for treatment assignment and both potential outcomes.

| File or content | SHA-256 |
|---|---|
| Frozen bank, gzipped | `bd34d3b2096b9250bbf56eba938609c9071884da696f444aec6daa2b88f7282e` |
| Bank JSON, uncompressed | `5accc3b6310632f6bb9309a49aa1bc78228afffc31733979bed41b2a67f739f6` |
| Generation receipts, gzipped | `0ba8da2167f58063e54002883dc01da579a4e3be4b5b1ad87c77944527e6b71e` |
