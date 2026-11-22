"""Author and load LLM-generated clinical paraphrase & EHR-scaffold sidecars for design banks.

Keeps ``nemotron_5000.json.gz`` untouched (preserving its SHA-256 receipt) while enriching
every unique ``(present, absent)`` note finding with diverse clinical paraphrases, section
assignments, naturalistic ECOG phrasings, and benign multi-section EHR prose.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import time

from .saved_dgp import _checkpoint, _digest, _read_checkpoint
from .study_designs import DEFAULT_BANK, load_bank

PARAPHRASE_BANK_VERSION = 1
DEFAULT_API_URL = "http://sn4622130540:8000/v1"
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
SECTIONS = (
    "oncology_history",
    "prior_therapy",
    "imaging_pathology",
    "comorbidities",
    "symptoms_exam",
    "social_meds",
)
SECTION_HEADERS = {
    "oncology_history": ("ONCOLOGY HISTORY:", "Disease History:", "Oncologic Background:"),
    "prior_therapy": ("PRIOR THERAPIES:", "Treatment History:", "Prior Oncology Interventions:"),
    "imaging_pathology": ("IMAGING / PATHOLOGY:", "Diagnostic Findings:", "Pathology & Imaging:"),
    "comorbidities": ("PAST MEDICAL HISTORY:", "Comorbidities:", "Medical History:"),
    "symptoms_exam": ("INTERVAL SYMPTOMS & EXAM:", "Clinical Findings:", "Review of Systems / Exam:"),
    "social_meds": ("SOCIAL & MEDICATION HISTORY:", "Medications / Social:", "Additional History:"),
}
FILLER_CATEGORIES = ("hpi_intro", "ros_general", "exam_general", "plan_followup")
_LEAK_RE = re.compile(r"\b(confounder|effect modifier|propensity|counterfactual|oracle|synthetic)\b", re.I)


def finding_key(present: str, absent: str) -> str:
    return f"{present.strip()}|||{absent.strip()}"


def collect_unique_findings(bank: dict) -> list[tuple[str, str]]:
    """Return sorted unique ``(present, absent)`` pairs across all designs in ``bank``."""
    pairs = set()
    for design in bank["designs"]:
        for spec in design["candidates"]["note_findings"]:
            pairs.add((spec["present"].strip(), spec["absent"].strip()))
    return sorted(pairs)


def _derive_seed(master: int, *parts) -> int:
    data = json.dumps([master, *parts], separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big") & 0x7FFFFFFF


def _clean_variants(items, fallback, *, min_count=4, max_chars=240):
    if not isinstance(items, list):
        return None
    out = []
    seen = set()
    for raw in items:
        if not isinstance(raw, str):
            continue
        text = " ".join(raw.split()).strip()
        if not text or len(text) > max_chars or _LEAK_RE.search(text):
            continue
        low = text.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(text)
    if len(out) < min_count:
        return None
    if fallback.strip().lower() not in seen:
        out.insert(0, fallback.strip())
    return out


def _batch_prompt(batch: list[tuple[int, str, str]], variants_per_polarity: int) -> str:
    rows = "\n".join(
        f'- id={idx} | present="{p}" | absent="{a}"' for idx, p, a in batch
    )
    sections = ", ".join(SECTIONS)
    return (
        "You are authoring realistic oncology EHR chart paraphrases for medical NLP.\n"
        f"For EACH numbered clinical finding below, generate:\n"
        f"1. `section`: one of [{sections}] indicating where this fact belongs in an oncology note.\n"
        f"2. `present`: {variants_per_polarity} distinct, realistic physician-chart ways to affirm ONLY "
        "the exact `present` fact. Vary clinical synonyms, standard oncology abbreviations "
        "(e.g., hx, s/p, PMH, bx, CT, w/), concise shorthand fragments, and full sentences.\n"
        f"3. `absent`: {variants_per_polarity} distinct, realistic physician-chart ways to negate ONLY "
        "the exact `absent` fact (e.g., denies, negative for, no evidence of, no h/o, ruled out, absent).\n\n"
        "Rules:\n"
        "- Preserve exact clinical meaning and polarity; never mix present and absent statements.\n"
        "- Do NOT invent unrelated labs, ECOG scores, dates, cancer stages, drugs, or outcomes.\n"
        "- Keep each variant under 25 words and realistic for an outpatient oncology note.\n\n"
        f"Findings:\n{rows}\n\n"
        'Return valid JSON only: {"items": [{"id": 0, "section": "...", "present": [...], "absent": [...]}]}'
    )


def _ecog_prompt() -> str:
    return (
        "Generate 12 distinct, realistic outpatient oncology chart phrases documenting each ECOG "
        "performance status score from 0 to 4 (keys \"0\", \"1\", \"2\", \"3\", \"4\").\n"
        "Vary shorthand and prose (e.g. 'ECOG PS 1', 'Performance status: ECOG 1, ambulatory', "
        "'Functional status ECOG 0 (fully active)'). Every phrase MUST include the exact integer "
        "digit (0, 1, 2, 3, or 4) and must NOT invent cancer type, labs, or treatments.\n"
        'Return valid JSON only: {"0": [...], "1": [...], "2": [...], "3": [...], "4": [...]}'
    )


def _filler_prompt() -> str:
    return (
        "Generate realistic, clinically benign, generic outpatient medical oncology note snippets "
        "that add authentic EHR prose WITHOUT mentioning any specific cancer type, stage, biomarker, "
        "comorbidity, specific drug name, lab number, vital number, or ECOG score.\n"
        "Provide 24 distinct snippets (1-2 sentences each) for EACH of these 4 categories:\n"
        '- "hpi_intro": opening clinic follow-up/consult framing sentences.\n'
        '- "ros_general": generic review-of-systems documentation lines.\n'
        '- "exam_general": generic physical exam documentation lines (alert, oriented, non-focal).\n'
        '- "plan_followup": generic supportive follow-up/coordination closing lines.\n'
        'Return valid JSON only: {"hpi_intro": [...], "ros_general": [...], "exam_general": [...], "plan_followup": [...]}'
    )


class _NemotronParaphraser:
    def __init__(self, api_url: str, model: str, seed: int, work_dir: Path):
        from openai import OpenAI

        self.api_url = api_url
        self.model = model
        self.seed = seed
        self.work_dir = Path(work_dir)
        self.client = OpenAI(
            base_url=api_url,
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or "local",
            timeout=300,
            max_retries=0,
        )

    def close(self):
        self.client.close()

    def request_json(self, key: str, prompt: str, validator, *, max_tokens: int = 3200):
        path = self.work_dir / "requests" / f"{key}.json"
        identity = _digest([PARAPHRASE_BANK_VERSION, self.model, self.seed, prompt, max_tokens])
        cached = _read_checkpoint(path, identity)
        if cached is not None:
            return validator(cached["parsed"])

        from openai import APIConnectionError, APIStatusError

        last_err = None
        feedback = ""
        for attempt in range(8):
            req_seed = _derive_seed(self.seed, key, attempt)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt + feedback}],
                    seed=req_seed,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 40},
                )
            except (APIConnectionError, APIStatusError) as exc:
                last_err = exc
                time.sleep(min(2**attempt, 20))
                continue
            choice = resp.choices[0]
            content = (choice.message.content or "").strip()
            if choice.finish_reason != "stop" or not content:
                last_err = ValueError(f"Incomplete generation ({choice.finish_reason})")
                continue
            try:
                parsed = json.loads(content)
                validated = validator(parsed)
                _checkpoint(path, identity, {"parsed": parsed})
                return validated
            except Exception as exc:
                last_err = exc
                feedback = (
                    f"\n\nPrevious response was rejected ({exc}). "
                    "Please make sure you include ALL requested items and at least the requested count."
                )
        raise RuntimeError(f"Failed paraphrase request {key} after retries: {last_err}")


def _validate_batch_response(parsed, batch, min_variants):
    items = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        raise ValueError("Missing 'items' list")
    by_id = {}
    for entry in items:
        if isinstance(entry, dict) and isinstance(entry.get("id"), int):
            by_id[entry["id"]] = entry
    out = {}
    for idx, p_orig, a_orig in batch:
        entry = by_id.get(idx)
        if entry is None:
            raise ValueError(f"Missing batch id {idx}")
        section = entry.get("section", "oncology_history")
        if section not in SECTIONS:
            section = "oncology_history"
        p_vars = _clean_variants(entry.get("present"), p_orig, min_count=min_variants)
        a_vars = _clean_variants(entry.get("absent"), a_orig, min_count=min_variants)
        if p_vars is None or a_vars is None:
            raise ValueError(f"Insufficient valid paraphrases for id {idx}")
        # Ensure present and absent pools remain strictly disjoint.
        p_low = {s.lower() for s in p_vars}
        a_vars = [s for s in a_vars if s.lower() not in p_low]
        if not a_vars:
            a_vars = [a_orig.strip()]
        out[finding_key(p_orig, a_orig)] = {
            "section": section,
            "present": p_vars,
            "absent": a_vars,
        }
    return out


def _validate_ecog_response(parsed):
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object for ECOG templates")
    out = {}
    for score in ("0", "1", "2", "3", "4"):
        raw = parsed.get(score)
        cleaned = _clean_variants(
            [s for s in (raw or []) if isinstance(s, str) and score in s],
            f"ECOG performance status: {score}.",
            min_count=3,
            max_chars=180,
        )
        if cleaned is None:
            raise ValueError(f"Insufficient valid ECOG {score} templates")
        out[score] = cleaned
    return out


def _validate_filler_response(parsed):
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object for filler paragraphs")
    out = {}
    for cat in FILLER_CATEGORIES:
        cleaned = _clean_variants(
            parsed.get(cat),
            "Patient returns to medical oncology clinic for interval evaluation.",
            min_count=6,
            max_chars=260,
        )
        if cleaned is None:
            raise ValueError(f"Insufficient filler snippets for {cat}")
        out[cat] = cleaned
    return out


def _write_gzip_json(path: Path, payload: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    digest = hashlib.sha256(raw).hexdigest()
    with NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as handle:
        tmp = Path(handle.name)
    try:
        with tmp.open("wb") as out, gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as gz:
            gz.write(raw)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return digest


def load_paraphrase_bank(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != PARAPHRASE_BANK_VERSION:
        raise ValueError(f"Unsupported paraphrase bank format in {path}")
    if not isinstance(data.get("findings"), dict):
        raise ValueError(f"Paraphrase bank missing findings dictionary: {path}")
    return data


def build_paraphrase_bank(
    bank_path: Path,
    output_path: Path,
    *,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    seed: int = 20261121,
    workers: int = 16,
    batch_size: int = 6,
    variants: int = 10,
    work_dir: Path | None = None,
) -> dict:
    bank_path = Path(bank_path)
    output_path = Path(output_path)
    work_dir = Path(work_dir) if work_dir is not None else output_path.with_suffix(".work")
    bank = load_bank(bank_path)
    source_sha = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    unique_pairs = collect_unique_findings(bank)
    batches = [
        [(i, p, a) for i, (p, a) in enumerate(unique_pairs[start : start + batch_size])]
        for start in range(0, len(unique_pairs), batch_size)
    ]
    min_variants = max(3, min(4, variants // 2))
    print(
        f"[paraphrase-bank] {len(unique_pairs)} unique findings across {len(bank['designs'])} "
        f"designs -> {len(batches)} batches (workers={workers}, model={model})",
        flush=True,
    )

    author = _NemotronParaphraser(api_url, model, seed, work_dir)
    t0 = time.monotonic()
    try:
        ecog_templates = author.request_json(
            "meta_ecog", _ecog_prompt(), _validate_ecog_response, max_tokens=2000
        )
        filler_paragraphs = author.request_json(
            "meta_filler", _filler_prompt(), _validate_filler_response, max_tokens=3200
        )
        findings_map = {}

        def _run_batch(b_idx, batch):
            key = f"batch_{b_idx:04d}"
            prompt = _batch_prompt(batch, variants)
            return author.request_json(
                key,
                prompt,
                lambda parsed: _validate_batch_response(parsed, batch, min_variants),
                max_tokens=3600,
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_batch, b_idx, batch): b_idx
                for b_idx, batch in enumerate(batches)
            }
            done = 0
            for fut in as_completed(futures):
                findings_map.update(fut.result())
                done += 1
                if done % 25 == 0 or done == len(batches):
                    elapsed = time.monotonic() - t0
                    eta = elapsed * (len(batches) - done) / done
                    print(
                        f"[paraphrase-bank] {done}/{len(batches)} batches "
                        f"({len(findings_map)}/{len(unique_pairs)} findings, "
                        f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
                        flush=True,
                    )
    finally:
        author.close()

    payload = {
        "version": PARAPHRASE_BANK_VERSION,
        "source_bank": bank_path.name,
        "source_bank_sha256": source_sha,
        "model": model,
        "seed": seed,
        "sections": list(SECTIONS),
        "ecog_templates": ecog_templates,
        "filler_paragraphs": filler_paragraphs,
        "findings": findings_map,
    }
    digest = _write_gzip_json(output_path, payload)
    print(
        f"[paraphrase-bank] Wrote {output_path} ({len(findings_map)} findings, sha256={digest[:12]})",
        flush=True,
    )
    return payload


def paraphrase_bank_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build an LLM clinical paraphrase + EHR-scaffold sidecar for a study design bank.",
        prog="python -m synthetic_data.cli paraphrase-bank",
    )
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BANK.with_name("nemotron_5000_paraphrases.json.gz"),
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    build_paraphrase_bank(
        args.bank,
        args.output,
        api_url=args.api_url,
        model=args.model,
        seed=args.seed,
        workers=args.workers,
        batch_size=args.batch_size,
        variants=args.variants,
        work_dir=args.work_dir,
    )
    return 0
