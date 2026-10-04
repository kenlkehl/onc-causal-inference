# Plumb-4B: adaptive continuous-value extraction

## Finding

Three-pass bin selection can substantially improve the typical numeric estimate,
but Plumb was not reliable enough for unrestricted continuous extraction in this
experiment. Compact inequalities improved final-bin accuracy over verbal range
descriptions, yet only 52/68 numeric values remained inside the final eight-bin
search interval. Narrower intervals could confidently concentrate on a wrong
value. Unit conversion produced large errors, and one plain dollar amount also
failed severely. No pipeline extraction code was changed.

## Experiment

Run on 2026-10-04 using the previously downloaded model revision
`24f7bf77e7ee258a2d158c61ea2dce2b60321010`, the same JevK5 v0.2.0 runtime,
BF16, temperature 2.07, eager execution, and physical A6000 GPU 1. The GPU
remained shared with the pre-existing vLLM worker. Each choice used one forward
pass and zero generated tokens.

The fixture contains 48 seeded direct numeric statements (seed 20261004),
20 additional numeric challenges, and eight missing, ambiguous, or out-of-range
cases. All questions and expected answers were frozen before inference. The
synthetic fields are age, weight, tumor diameter, creatinine, sensor temperature,
invoice amount, percentage, and a signed instrument reading. These are short
invented examples, not clinical validation.

Two strategies were tested, each with **at most eight total choices per call**:

1. Eight equal-width numeric bins, evaluated on the 68 numeric in-range cases.
2. Six equal-width numeric bins plus `unresolved` and `outside`, evaluated on
   all 76 cases. These explicit exits are available at every pass.

The selected interval is subdivided twice more. Each pass receives the original
evidence, the target field, and the new options. The model selects the next
interval; gold values never guide a branch, retry, or correction. The midpoint
of the selected interval is the point estimate. Bounds are lower-inclusive and
upper-exclusive, except that the global maximum is included. A separate CPU
check verified boundary membership, nonoverlap, nested intervals, midpoint
arithmetic, and the eight-option limit.

Initial domains are fixed per field, independent of the individual answer:

| Field | Initial domain | Units | Final eight-bin width |
|---|---:|---|---:|
| age | 0 to 128 | years | 0.25 |
| weight | 0 to 256 | kg | 0.5 |
| tumor | 0 to 16 | cm | 0.03125 |
| creatinine | 0 to 8 | mg/dL | 0.015625 |
| temperature | -40 to 80 | degrees C | 0.234375 |
| amount | 0 to 1024 | dollars | 2 |
| percentage | 0 to 100 | percent | 0.1953125 |
| signed | -64 to 64 | units | 0.25 |

After three correct choices, eight bins divide the initial width by 512;
six bins divide it by 216. The latter intervals are therefore 2.37 times wider,
so their containment rates are not comparisons at equal numerical precision.
Six-way decimal edges are rounded to eight decimal places, and evaluation uses
exactly those displayed edges.

## Results

Each cell below counts numeric cases whose selected interval still contains the
true value, out of **all 68 eligible numeric cases**. Abstentions count as no
successful numeric extraction. This criterion becomes stricter each pass; its
decline does not imply that point-estimate error generally increases.

| Range format | Choices | Pass 1 | Pass 2 | Pass 3 | Numeric outputs after pass 3 | Median time per numeric case |
|---|---|---:|---:|---:|---:|---:|
| verbal | 8 numeric | 58/68 (85.3%) | 52/68 (76.5%) | 44/68 (64.7%) | 68/68 | 176.4 ms |
| verbal | 6 numeric + 2 exits | 63/68 (92.6%) | 54/68 (79.4%) | 45/68 (66.2%) | 65/68 | 202.2 ms |
| compact | 8 numeric | 62/68 (91.2%) | 59/68 (86.8%) | 52/68 (76.5%) | 68/68 | 170.4 ms |
| compact | 6 numeric + 2 exits | 67/68 (98.5%) | 60/68 (88.2%) | 53/68 (77.9%) | 64/68 | 185.6 ms |

The compact format uses descriptions such as `64 <= x < 96; x is in kg.`
It was tried once after inspecting failures with verbal descriptions. It is an
exploratory comparison on the same fixture, not a separately held-out improvement.
The original runs and all failures are retained.

For the compact eight-bin version, median absolute error normalized by each
field's initial range width decreased from **3.906% to 0.547% to 0.091%**.
However, normalized mean absolute error remained **2.73%** at the last pass,
reflecting large outliers. Do not combine raw errors across different units.

With compact ranges and six numeric bins, all six missing/ambiguous cases and
both out-of-range cases were correctly rejected. Four of 68 valid numeric cases
also stopped early. Of 64 returned final numeric intervals, 53 contained the
truth (82.8%); 11 still missed it. Thus adding exits helped avoid forced answers
but did not make fine numeric extraction consistently reliable. The verbal
version correctly rejected 7/8 special cases.

Timing excludes model loading and representative warm-up. New token-length
shapes can still incur compilation overhead, so these are indicative timings.
Median individual calls were about 57–64 ms. Peak PyTorch allocation was about
7.91 GiB; `causal_conv1d` used the reference implementation. CUDA graphs,
batching, and a generative-extraction speed baseline were not evaluated.

## Representative traces

Compact eight-bin results, with midpoint estimates after each pass:

| Evidence / target | True value | Pass 1 estimate | Pass 2 estimate | Pass 3 estimate |
|---|---:|---:|---:|---:|
| Pretreatment weight, kg | 72.6 | 80 | 74 | 72.75 |
| Creatinine, mg/dL | 1.93 | 1.5 | 1.9375 | 1.929688 |
| Sensor temperature, degrees C | 10.8 | 12.5 | 9.6875 | 10.039062 |
| Invoice total, dollars | 79.95 | 832 | 792 | 797 |
| 27 mm diameter, target cm | 2.7 | 3 | 2.375 | 2.296875 |
| 72500 grams weight, target kg | 72.5 | 240 | 254 | 255.75 |
| 180 pounds weight, target kg | 81.6466266 | 176 | 166 | 167.75 |

The successful 72.6 kg trace narrowed `[64, 96)` → `[72, 76)` → `[72.5, 73)`,
ending at 72.75 kg. The successful creatinine trace narrowed `[1, 2)` →
`[1.875, 2)` → `[1.921875, 1.9375)`, ending at 1.9296875 mg/dL for a true 1.93.

In contrast, the 10.8-degree sensor reading was first placed correctly in
`[5, 20)`, then incorrectly in `[8.75, 10.625)`. The final interval
`[9.921875, 10.15625)` had 85.5% probability despite excluding the truth.
The $79.95 example began in `[768, 896)` at 79.7% probability and ended at $797.
These narrow search intervals are **not statistical confidence intervals**.

Unit-conversion constants were supplied explicitly in each conversion question.
All three conversion cases missed the final correct interval in all four
configurations. Conversion should therefore not be delegated to this model on
the strength of these results. The major invoice error also shows that unit
normalization alone would not fix the method.

Some same-unit fields did much better than the aggregate. With compact eight-bin
ranges, age MAE was 0.125 years over 10 cases, creatinine MAE was 0.00510 mg/dL
over nine cases, and weight MAE excluding its two conversion cases was 0.25 kg
over 11 cases. These small subsets do not establish general reliability.

## Implications

The proposed approach is technically viable and fast for approximate estimates,
and extra passes often improve a correct initial branch. The experiments do
not support treating increasing displayed precision as validated extraction
accuracy, or replacing Stage 2 continuous extraction with this greedy procedure
as it stands. The six-bin option can detect that an earlier narrowing step
excluded the answer, but can also incorrectly abstain when the current ranges
still contain it; both reduce numeric output coverage.

For a next prototype, deterministic unit normalization and explicit
missing/outside options would be necessary, followed by field-specific accuracy
validation and a fallback for unreliable measurements. Another candidate is to
extract literal numeric spans with code and have Plumb select the relevant span
among a small candidate set. That would avoid asking it to repeatedly perform
fine numeric comparisons, but remains untested here. No beam search, alternate
branch recovery, confidence threshold fitting, or additional prompt tuning was
performed.

## Reproduction and saved outputs

The fixture is `experiments/plumb_feasibility/continuous_questions.json`. The
runner is `scripts/plumb_continuous_test.py`, using the isolated environment from
`docs/plumb_feasibility.md`. The original verbal runner is also archived with
its results, and its SHA-256 matches the recorded experiment fingerprint.

```bash
artifacts/plumb_feasibility/venv/bin/python scripts/plumb_continuous_test.py \
  --interval-style verbal --output artifacts/plumb_feasibility/new_continuous_verbal
artifacts/plumb_feasibility/venv/bin/python scripts/plumb_continuous_test.py \
  --interval-style compact --output artifacts/plumb_feasibility/new_continuous_compact
```

Existing results are never overwritten. The default GPU selector is the UUID
for physical GPU 1 from this machine. Model weights and the fixture remain
unchanged between the two runs.

- `artifacts/plumb_feasibility/continuous_results/`: original verbal ranges.
- `artifacts/plumb_feasibility/continuous_compact_results/`: compact inequalities.
- Each directory contains `fixture.json`, `results.jsonl`, `summary.json`, and
  `run.log`; every actual question, full distribution, interval, and timing is
  retained. Together the two runs made 833 scored forward passes (417 + 416).
- `artifacts/plumb_feasibility/continuous_analysis.json`: supplementary
  median-error and midpoint-tolerance calculations.

Verification rechecked all 288 case/strategy records, every chosen interval,
midpoint, expected containment result, nested range, option count, fixture hash,
script hash, and aggregate case count against the saved raw outputs.
