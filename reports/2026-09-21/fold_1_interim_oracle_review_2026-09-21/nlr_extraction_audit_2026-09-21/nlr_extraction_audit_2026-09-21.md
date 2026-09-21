# Fold 1 NLR extraction audit — September 21, 2026

1. **Finding**
   1. Of 45 missing NLR values among 200 held-out patients, 26 records contained paired absolute neutrophil and lymphocyte counts, and one contained an explicit positive NLR. These are documented opportunities to recover a measurement that the saved extraction left null.
   2. Fourteen records lacked a documented ratio or usable lymphocyte count; two records had entirely empty clinical text and zero notes in the source dataset. Two more contained negative NLR values, creating a source-validity problem.
   3. All 45 missing values were present at the validated extraction checkpoint and remained missing through raw CSV materialization, harmonization, and the final shared measurement matrix. None came from transport exhaustion, exhausted structural repairs, or serial-chunk state loss.

2. **Breakdown**
   | Source/extraction finding | Patients |
   | --- | ---: |
   | Paired counts documented, ratio left missing | 26 |
   | Explicit positive NLR documented, left missing | 1 |
   | No ratio or usable lymphocyte count found in nonempty text | 14 |
   | Empty clinical text; source dataset has zero notes | 2 |
   | Negative NLR values in the notes | 2 |
   | **Total missing** | **45** |

3. **Concrete examples**
   1. Row 986: the same CBC reported ANC 7.0 k/µL and lymphocytes 1.4 k/µL. Their ratio is 5.0; the saved NLR extraction was null. The latent oracle was approximately 5.001, but it was not supplied to extraction.
   2. Row 606: a pretreatment CBC reported neutrophils 6.8 k/µL and lymphocytes 2.0 k/µL, supporting an NLR of 3.4. The extraction was null.
   3. Row 617: the record repeatedly documented NLR 6.47, including a table entry and a narrative description of elevated neutrophil-to-lymphocyte ratio. Its checkpoint still recorded null. The response finished normally in 14.4 seconds with 167 completion tokens and passed validation on the first attempt, with reasoning disabled.
   4. Rows 528 and 752 had empty input text and zero source notes. Missing extraction was expected for their unrecorded information.
   5. Row 841 explicitly called its NLR −0.23 a data-entry error; row 467 documented −0.56. Both had negative latent oracle values. These are not straightforward failures to copy a valid physiological measurement. Row 467 also contained positive cell counts, so resolving conflicting observations matters.

4. **Why repair did not catch this**
   1. The validator checks rows, keys, scalar types, and allowed categories. It accepts a null for continuous NLR without checking the source for an explicit ratio or paired counts.
   2. Forty-two of the 45 feature-batch responses passed on their first response. Three needed format repairs; two of those reached high reasoning. All ultimately produced a valid response with NLR missing. A clinically unjustified null is currently not a validation error, so it does not itself trigger repair or reasoning escalation.
   3. These were single-request feature groups, not serial extraction chunks. No value was found in an earlier chunk and then erased.

5. **Prompt ambiguity and its limits**
   1. The feature-specific definition explicitly permits computing NLR as absolute neutrophils divided by absolute lymphocytes, and permits null only when neither the ratio nor both components are documented.
   2. The shared scalar rule says: “if the definition requests multiple components, return null rather than a ratio string or aggregate.” That rule is intended to prevent composite outputs, but can be read as discouraging a derived numeric ratio. This is a plausible contributor to the missed calculations; the saved records do not prove the model interpreted it that way.
   3. That ambiguity does not explain the missed explicit NLR in row 617. Its exact cause is not recorded; the evidence supports a semantic extraction miss rather than a token or transport failure.

6. **Improvement targets**
   1. Clarify the generic scalar rule so explicitly authorized numeric derivations are allowed.
   2. For derived features such as NLR, extract the component measurements with units, governing time, and evidence; calculate the ratio deterministically only for compatible observations, with a valid nonzero denominator.
   3. Add targeted review of a null when the record contains evidence that should support the requested measurement. Keep absence of source evidence distinct from an extraction failure.
   4. Address empty records, omitted quantities, and invalid negative latent NLR values in synthetic-data generation. Better extraction cannot faithfully reproduce a latent value that the text does not support.
   5. This audit sent no model requests and changed no fitted result, input, prompt, setting, or extraction checkpoint. Any implementation change requires a separately recorded experiment revision or later run.

7. **Audit limits and source ledger**
   1. Source searches included NLR, spelled-out ratio names, ANC/ALC, neutrophil/lymphocyte count labels, and common abbreviations. Count-pair examples were checked within the same note/laboratory record. Searching did not treat references to lymph nodes as lymphocyte counts.
   2. The 26 count-pair cases identify calculation opportunities. They do not establish that an arbitrary pair is the correct baseline/latest measurement under the conflict-resolution policy. Nor does a computed ratio necessarily equal the simulation’s latent oracle.
   3. The saved result is the validated extraction response, not a complete raw-provider transcript. The accepted nulls have no patient-specific explanation attached.
   4. [Row-level findings, source excerpts, request-event counts, and checkpoint hashes](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_interim_oracle_review_2026-09-21/nlr_extraction_audit_2026-09-21/nlr_missingness_audit_2026-09-21.json).
