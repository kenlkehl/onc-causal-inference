"""ColBERT context selection feeding Stage 2's existing validated extraction."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json

import pandas as pd


def policy_identity(config):
    from ..extraction.colbert import retrieval_identity

    return {"method": "colbert", **retrieval_identity(config)}


def claim_output_method(output_dir, config, *, fold_root=False):
    from .plain_handoff_stage2_analysis import _write_json

    identity = policy_identity(config)
    path = output_dir / "measurement_method.json"
    if path.is_file():
        previous = json.loads(path.read_text())
    elif any(
        (output_dir / name).exists()
        for name in (
            ("ontology_supervision", "extraction", "preselection")
            if fold_root
            else ("batches", "pages", "by_strategy", "extracted.csv", "colbert")
        )
    ):
        previous = {"method": "full_record"}
    else:
        previous = identity
    if previous != identity:
        raise ValueError(
            "Extraction method/settings differ from saved measurements. "
            "Use a fresh output directory or select the saved extraction method."
        )
    _write_json(path, identity)
    return identity


def extract_rows(
    *,
    dataset,
    row_ids,
    text_column,
    definitions,
    output_dir,
    request_json,
    workers,
    max_prompt_chars,
    feature_batch_size,
    request_identity,
    tokenizer,
    chunk_size_tokens,
    context_window_tokens,
    max_output_tokens,
    context_margin_tokens,
    deferred_retry_passes,
    colbert,
):
    from ..extraction.colbert import get_retriever
    from . import plain_handoff_stage2_analysis as analysis

    identity = claim_output_method(output_dir, colbert)
    retriever = get_retriever(colbert)
    # Each question gets its own top-k evidence. Grouping changes LLM packing only.
    batches = analysis._partition_feature_definitions(
        definitions, feature_batch_size=feature_batch_size
    )
    tasks = iter((int(row_id), batch) for row_id in row_ids for batch in batches)
    values = {int(row_id): {} for row_id in row_ids}
    if len(values) != len(row_ids):
        raise ValueError("ColBERT extraction requires distinct row IDs")

    def run(task):
        row_id, batch = task
        source = dataset.iloc[row_id][text_column]
        source = "" if pd.isna(source) else str(source)
        evidence = retriever.retrieve(source, batch, top_k=colbert.top_k)
        directory = (
            output_dir
            / "colbert"
            / f"row_{row_id:08d}"
            / analysis._value_fingerprint(analysis._prompt_feature_definitions(batch))[:24]
        )
        # The original cohort row identity survives every nested checkpoint.
        analysis._write_json(directory / "retrieval.json", {"row_id": row_id, **evidence})
        frame = analysis.extract_rows(
            dataset=dataset,
            row_ids=[row_id],
            text_column=text_column,
            definitions=batch,
            output_dir=directory,
            request_json=request_json,
            workers=1,
            max_prompt_chars=max_prompt_chars,
            feature_batch_size=feature_batch_size,
            request_identity={
                **(request_identity or {}),
                "retrieval": identity,
                "source_sha256": evidence["source_sha256"],
            },
            tokenizer=tokenizer,
            chunk_size_tokens=chunk_size_tokens,
            context_window_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
            context_margin_tokens=context_margin_tokens,
            deferred_retry_passes=deferred_retry_passes,
            context_strategy="full_record",
            _source_rows=[{"row_id": row_id, "text": evidence["context"]}],
        )
        return row_id, frame.iloc[0].drop(labels=["_oci_row_id"]).to_dict()

    # Keep only a bounded number of patient/feature tasks in flight. Existing
    # request_json admission/routing still enforces every LLM endpoint's limits.
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        pending = set()
        try:
            for _ in range(max(1, int(workers))):
                task = next(tasks, None)
                if task is not None:
                    pending.add(pool.submit(run, task))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row_id, measured = future.result()
                    values[row_id].update(measured)
                    task = next(tasks, None)
                    if task is not None:
                        pending.add(pool.submit(run, task))
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    frame = pd.DataFrame(
        [{"_oci_row_id": int(row_id), **values[int(row_id)]} for row_id in row_ids],
        columns=["_oci_row_id", *[d["name"] for d in definitions]],
    )
    analysis._write_frame(output_dir / "extracted.csv", frame)
    summary = analysis._summarize_extraction_failures(
        output_dir=output_dir, definitions=definitions
    )
    analysis._write_json(
        output_dir / "complete.json",
        {
            "status": "complete",
            "completed_at": analysis._now(),
            "rows": len(frame),
            "features": len(definitions),
            "measurement_method": identity,
            "scope": "retrieved_excerpts",
            "feature_failure_patterns": len(summary["feature_failure_patterns"]),
            "structural_failure_patients": summary["structural_failure_patient_count"],
        },
    )
    return frame
