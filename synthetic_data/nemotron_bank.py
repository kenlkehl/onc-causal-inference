"""Generate a reproducible candidate-count-stratified bank on a seeded vLLM endpoint."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import string
from tempfile import NamedTemporaryFile
import time

from openai import APIConnectionError, APIStatusError, OpenAI

from .saved_dgp import _checkpoint, _digest, _read_checkpoint
from .study_designs import (
    MEASUREMENTS,
    question_text,
    validate_bank,
    validate_candidates,
    validate_question,
)

MODEL = "RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8"
ENDPOINT = "http://172.24.216.113:8002/v1"
SEED = 20261119
VERSION = 1
SITES = {
    "lung": ("lung cancer", "C34.90"),
    "colon": ("colon cancer", "C18.9"),
    "breast": ("breast cancer", "C50.919"),
    "pancreas": ("pancreatic cancer", "C25.9"),
    "ovary": ("ovarian cancer", "C56.9"),
    "prostate": ("prostate cancer", "C61"),
    "melanoma": ("cutaneous melanoma", "C43.9"),
    "kidney": ("kidney cancer", "C64.9"),
    "bladder": ("bladder cancer", "C67.9"),
    "stomach": ("gastric cancer", "C16.9"),
    "esophagus": ("esophageal cancer", "C15.9"),
    "rectum": ("rectal cancer", "C20"),
    "liver": ("hepatocellular carcinoma", "C22.0"),
    "bile_duct": ("intrahepatic cholangiocarcinoma", "C22.1"),
    "cervix": ("cervical cancer", "C53.9"),
    "endometrium": ("endometrial cancer", "C54.1"),
    "thyroid": ("thyroid cancer", "C73"),
    "testis": ("testicular cancer", "C62.90"),
    "glioma": ("malignant glioma", "C71.9"),
    "myeloma": ("multiple myeloma", "C90.00"),
}
SHARED_THEMES = (
    "pre-existing cardiovascular and pulmonary diagnoses",
    "other chronic comorbid diagnoses",
    "baseline concomitant medication exposures",
    "previous treatment complications and reactions",
    "social support, transportation and care access",
    "prior procedures and implanted devices",
    "documented infection or immune history",
    "prior malignancies and family cancer history",
)
SITE_THEMES = (
    "histology and pathology",
    "tumor molecular subtypes and tumor biomarkers on pathology",
    "anatomic involvement and routes of spread",
    "imaging morphology and extent of involvement",
    "prior systemic treatments and response history",
    "past local procedures and surgical findings",
    "prior radiation fields and exposure",
    "disease-specific clinical presentation and complications",
    "diagnostic workup findings",
    "surgical anatomy and feasibility of local therapy",
    "relapse patterns and disease course before the index decision",
    "background conditions and prior exposures specific to this cancer",
)
ROLE_PROFILES = (
    ("confounder", "prognostic", "unused"),
    ("effect_modifier", "prognostic", "unused"),
    ("dual_role", "confounder", "effect_modifier", "unused"),
    ("prognostic", "unused"),
    ("instrument", "unused"),
    ("unused",),
)
ALPHABET = string.ascii_uppercase
SITE_QUESTIONS = 5
VARIANTS = 50
MIN_WIDTH, MAX_WIDTH = 3, 100


def derive_seed(master, *parts):
    data = json.dumps([master, *parts], separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big") & 0x7FFFFFFF


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def question_prompt(site):
    return f"""Design five distinct hypothetical comparative effectiveness studies in adult {site}.
Choose five different clinical decisions, e.g. systemic regimen, timing, duration, extent of a local treatment or supportive care. Each comparison must be plausible within the specified eligible population.
Return JSON {{"questions":[{{"population":"...", "control":"...", "treatment":"...", "outcome":"...", "horizon_days":180}}]}} with exactly five objects. Specify the cancer in every population. Use concise arm names and a single observable binary adverse event, progression or death by a fixed integer number of days (7 to 1825). Avoid composite scores, continuous endpoints and explanations. These are hypothetical studies; do not state which treatment works better."""


def finding_prompt(site, theme):
    return f"""Create a vocabulary for simulation of pretreatment oncology records. Population: {site}. Focus: {theme}.
Return JSON {{"findings":[["snake_case_finding",0.25,"Chart sentence affirming it.","Chart sentence explicitly negating it."]]}} with 14 DISTINCT binary findings. Include a hypothetical prevalence strictly between 0 and 1. Use specific, short clinical phrases. Each pair of chart sentences must express the SAME single fact with opposite polarity. Limit sentences to 12 words each. Use distinct, lowercase ASCII identifiers. These variables can be recorded before the index decision.
Do not use values of routine blood tests, vital signs, performance scores, questionnaires, treatment assignment, future outcomes or causal terminology. Avoid fabricated numerical timelines, dates, and extraneous clinical facts. Return only JSON."""


class Author:
    def __init__(self, url, model, seed, work):
        self.url, self.model, self.seed, self.work = url, model, seed, Path(work)
        self.client = OpenAI(
            base_url=url, api_key=os.getenv("VLLM_API_KEY") or "local", timeout=300, max_retries=0
        )

    def request(self, key, prompt, parse, *, regex=None, max_tokens=2400):
        path = self.work / "requests" / (key + ".json")
        identity = _digest([VERSION, self.url, self.model, self.seed, prompt, regex, max_tokens])
        record = _read_checkpoint(path, identity)
        if record is not None:
            return parse(record["content"])
        feedback = ""
        for attempt in range(10):
            seed = derive_seed(self.seed, "request", key, attempt)
            message = prompt + feedback
            extra = {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 40}
            if regex:
                extra["structured_outputs"] = {"regex": regex}
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": message}],
                    seed=seed,
                    temperature=0.75,
                    top_p=0.95,
                    max_tokens=max_tokens,
                    response_format=None if regex else {"type": "json_object"},
                    extra_body=extra,
                )
            except (APIConnectionError, APIStatusError) as exc:
                if (
                    getattr(exc, "status_code", 500) not in (408, 429)
                    and getattr(exc, "status_code", 500) < 500
                ):
                    raise
                if attempt == 9:
                    raise
                time.sleep(min(2**attempt, 30))
                continue
            choice = response.choices[0]
            try:
                if response.model != self.model or choice.finish_reason != "stop":
                    raise ValueError(
                        f"Expected a complete response from the specified model ({choice.finish_reason})"
                    )
                value = parse(choice.message.content or "")
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                if attempt == 9:
                    raise RuntimeError(f"Invalid completion for {key}: {exc}") from exc
                feedback = f"\nThe previous response failed validation: {exc}. Please correct it."
                continue
            _checkpoint(
                path,
                identity,
                dict(
                    content=choice.message.content,
                    seed=seed,
                    attempt=attempt,
                    request_id=response.id,
                    fingerprint=response.system_fingerprint,
                    usage=response.usage.model_dump(),
                    completion_model=response.model,
                ),
            )
            return value


def parse_questions(text):
    rows = json.loads(text)["questions"]
    if not isinstance(rows, list) or len(rows) != SITE_QUESTIONS:
        raise ValueError("Return exactly five questions")
    fields = {"population", "control", "treatment", "outcome", "horizon_days"}
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("Use the five specified question fields")
        validate_question(
            {**row, "id": "test", "split": "train", "cancer": "test", "icd10cm": "C00"}
        )
        if not 7 <= row["horizon_days"] <= 1825:
            raise ValueError("Invalid outcome horizon")
        arms = frozenset(row[k].strip().casefold() for k in ("control", "treatment"))
        if arms in seen:
            raise ValueError("Return five different comparisons")
        seen.add(arms)
    return rows


def parse_findings(text):
    rows = json.loads(text)["findings"]
    if not isinstance(rows, list):
        raise ValueError("Return an array of findings")
    valid = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 4:
            continue
        name, prevalence, present, absent = row
        if not isinstance(name, str):
            continue
        item = dict(name=name, prevalence=prevalence, present=present, absent=absent)
        try:
            validate_candidates(
                dict(measurements=[], note_findings=[{**item, "roles": ["prognostic"]}])
            )
        except (ValueError, TypeError):
            continue
        valid.setdefault(name, item)
    if len(valid) < 6:
        raise ValueError(
            "Need at least six distinct valid pretreatment findings in this vocabulary batch"
        )
    return list(valid.values())


def parallel(tasks, fn, workers, label):
    tasks = list(tasks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = []
        for i, result in enumerate(pool.map(fn, tasks), 1):
            results.append(result)
            if i % max(20, min(100, len(tasks) // 5)) == 0 or i == len(tasks):
                print(f"{label} {i}/{len(tasks)}", flush=True)
        return results


def prepare(author, workers):
    questions = dict(
        parallel(
            SITES,
            lambda site: (
                site,
                author.request(
                    "questions/" + site, question_prompt(SITES[site][0]), parse_questions
                ),
            ),
            workers,
            "question batches",
        )
    )
    tasks = [("shared", i, t) for i, t in enumerate(SHARED_THEMES)]
    tasks += [(s, i, t) for s in SITES for i, t in enumerate(SITE_THEMES)]

    def get_theme(task):
        s, i, t = task
        population = (
            "adults with cancer, applicable across tumor types" if s == "shared" else SITES[s][0]
        )
        return s, author.request(
            f"findings/{s}/{i:02d}", finding_prompt(population, t), parse_findings, max_tokens=2400
        )

    findings = {s: {} for s in (*SITES, "shared")}
    for s, batch in parallel(tasks, get_theme, workers, "finding batches"):
        for item in batch:
            findings[s].setdefault(item["name"], item)
    return questions, findings


def make_questions(rows, master):
    rng = random.Random(derive_seed(master, "split"))
    unseen_sites = set(rng.sample(list(SITES), 3))
    remaining = [s for s in SITES if s not in unseen_sites]
    held_questions = {(s, rng.randrange(SITE_QUESTIONS)) for s in rng.sample(remaining, 5)}
    result = []
    for site, (cancer, code) in SITES.items():
        for i, row in enumerate(rows[site]):
            split = "held_out" if site in unseen_sites or (site, i) in held_questions else "train"
            q = dict(id=f"{site}_q{i+1:02d}", split=split, cancer=cancer, icd10cm=code, **row)
            validate_question(q)
            result.append((site, q))
    return result


def make_catalog(findings, site):
    # A site-specific definition takes precedence over a shared definition of the same name.
    notes = {**findings["shared"], **findings[site]}
    reserved = {key.replace(".", "_") for key in MEASUREMENTS}
    catalog = [("measurements", dict(name=key)) for key in sorted(MEASUREMENTS)]
    catalog.extend(("note_findings", notes[key]) for key in sorted(notes) if key not in reserved)
    if len(catalog) < 2 * MAX_WIDTH:
        raise ValueError(
            f"{site} has {len(catalog)} candidates; need two distinct options per slot even at width 100"
        )
    return catalog


def make_plans(questions, master):
    total = len(questions) * VARIANTS
    widths = [MIN_WIDTH + i % (MAX_WIDTH - MIN_WIDTH + 1) for i in range(total)]
    random.Random(derive_seed(master, "widths")).shuffle(widths)
    return [
        (site, q, v, widths[i * VARIANTS + v])
        for i, (site, q) in enumerate(questions)
        for v in range(VARIANTS)
    ]


def slot_options(size, width, seed):
    count = min(4, size // width)
    if count < 2:
        raise ValueError("Each slot needs two or more distinct options")
    ids = random.Random(seed).sample(range(size), count * width)
    return [ids[i : i + count] for i in range(0, len(ids), count)]


def design_request(author, plan, catalog):
    _, q, v, width = plan
    name = f"{q['id']}_v{v+1:03d}"
    slots = slot_options(len(catalog), width, derive_seed(author.seed, "menu", name))
    count = len(slots[0])
    symbols = ALPHABET[: count * len(ROLE_PROFILES)]
    prefix = f"""Author one synthetic oncology study design. {question_text(q)}
Catalog IDs below denote distinct candidate pretreatment features. Dotted lab/vital/pro IDs summarize recorded values. Other clinical findings are binary chart observations.
{', '.join(f"{i}={item['name']}" for i,(_,item) in enumerate(catalog))}
Select one clinically plausible candidate from EACH slot and specify its permitted roles across replicate studies. Decide which candidates are confounders, potential effect modifiers, prognostic factors, instruments or distractors. Especially in wide designs, keep a modest core of potentially causal features and use many distractors. Include at least one potentially causal candidate. Avoid relying solely on the first alternative. These are hypothetical mechanisms.
One symbol encodes both your candidate choice and role policy. A/B/C/D/E/F select the first candidate in the slot, G/H/I/J/K/L the second, M/N/O/P/Q/R the third, S/T/U/V/W/X the fourth if available. In each block the roles are, respectively: potential confounder; potential effect modifier; possibly both; prognostic only when active; instrument only when active; always unused. Other than always-unused, each policy allows the indicated feature to be inactive in a replicate."""
    rows = "; ".join("/".join(map(str, slot)) for slot in slots)
    prompt = f"{prefix}\nDesign {name}. {width} slots in order: {rows}\nReturn exactly {width} space-separated symbols from {symbols} in slot order."
    regex = f"[{symbols}]( [{symbols}]){{{width-1}}}"

    def decode(content):
        codes = content.strip().split()
        if len(codes) != width or any(len(c) != 1 or c not in symbols for c in codes):
            raise ValueError("Select exactly one valid symbol per slot")
        body = dict(measurements=[], note_findings=[])
        for slot, code in zip(slots, codes, strict=True):
            alternative, role = divmod(symbols.index(code), len(ROLE_PROFILES))
            kind, item = catalog[slot[alternative]]
            body[kind].append({**item, "roles": list(ROLE_PROFILES[role])})
        validate_candidates(body)
        return dict(id=name, question=q, candidates=body)

    return author.request("designs/" + name, prompt, decode, regex=regex, max_tokens=4 * width + 32)


def gzip_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
    try:
        with (
            tmp.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz,
        ):
            gz.write(compact(value).encode())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class Replay:
    def __init__(self, archive):
        with gzip.open(archive, "rt") as f:
            data = json.load(f)
        self.model, self.url, self.seed = data["model"], data["endpoint"], data["seed"]
        self.records = data["records"]
        self.expected_hash = data["bank_content_sha256"]

    def request(self, key, prompt, parse, *, regex=None, max_tokens=2400):
        record = self.records[key]
        value = record["value"]
        identity = _digest([VERSION, self.url, self.model, self.seed, prompt, regex, max_tokens])
        if (
            record["identity"] != identity
            or record["sha256"] != _digest(value)
            or value["completion_model"] != self.model
            or value["seed"] != derive_seed(self.seed, "request", key, value["attempt"])
        ):
            raise ValueError(f"Receipt mismatch: {key}")
        return parse(value["content"])


def generate(author, raw_questions, findings, args):
    questions = make_questions(raw_questions, author.seed)
    catalogs = {s: make_catalog(findings, s) for s in SITES}
    plans = make_plans(questions, author.seed)
    if args.pilot:
        plans = [next(p for p in plans if p[3] == n) for n in (3, 10, 30, 60, 100)]
    designs = parallel(
        plans, lambda p: design_request(author, p, catalogs[p[0]]), args.workers, "designs"
    )
    if args.pilot:
        print(
            "Pilot verified candidate counts:",
            [sum(map(len, d["candidates"].values())) for d in designs],
            flush=True,
        )
        return
    # Interleave clinical questions so short held-out evaluation runs visit different questions.
    designs = [designs[i * VARIANTS + v] for v in range(VARIANTS) for i in range(len(questions))]
    bank = validate_bank(
        dict(
            version=1,
            generation=dict(
                model=author.model,
                endpoint=author.url,
                seed=author.seed,
                recipe_version=VERSION,
                question_count=len(questions),
                variants=VARIANTS,
                min_features=MIN_WIDTH,
                max_features=MAX_WIDTH,
            ),
            designs=designs,
        )
    )
    signatures = set()
    for d in designs:
        entries = sorted(
            (f["name"], tuple(f["roles"])) for items in d["candidates"].values() for f in items
        )
        sig = (d["question"]["id"], tuple(entries))
        if sig in signatures:
            raise ValueError(f"Duplicate semantic design in {d['id']}")
        signatures.add(sig)
    bank_hash = hashlib.sha256(compact(bank).encode()).hexdigest()
    if isinstance(author, Replay):
        if bank_hash != author.expected_hash:
            raise ValueError("Offline replay does not match the frozen bank")
    elif args.receipts:
        keys = [author.work / "requests" / (f"questions/{s}.json") for s in SITES]
        keys += list((author.work / "requests" / "findings").glob("*/*.json"))
        keys += [author.work / "requests" / "designs" / (d["id"] + ".json") for d in designs]
        records = {
            str(p.relative_to(author.work / "requests").with_suffix("")): json.loads(p.read_text())
            for p in sorted(keys)
        }
        gzip_json(
            args.receipts,
            dict(
                model=author.model,
                endpoint=author.url,
                seed=author.seed,
                records=records,
                bank_content_sha256=bank_hash,
            ),
        )
    gzip_json(args.output, bank)
    print(f"Published {len(designs)} designs. Uncompressed SHA-256 {bank_hash}", flush=True)


@contextmanager
def locked(work):
    work.mkdir(exist_ok=True, parents=True)
    with (work / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=ENDPOINT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--receipts", type=Path, help="Archive successful raw model responses and sampling seeds"
    )
    parser.add_argument("--replay", type=Path, help="Replay an archived generation fully offline")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.seed < 0:
        parser.error("workers must be positive and seed nonnegative")
    if not args.prepare_only and not args.pilot and not args.output:
        parser.error("--output is required for a complete bank")
    archive = args.receipts or args.replay
    if archive and args.output and args.output.resolve() == archive.resolve():
        parser.error("Use distinct paths for the design bank and generation receipts")
    if args.replay:
        if not args.output or args.receipts or args.prepare_only or args.pilot:
            parser.error("--replay requires --output and no live generation options")
        author = Replay(args.replay)
        generate(author, *prepare(author, args.workers), args)
        return
    if not args.work_dir:
        parser.error("Live generation requires --work-dir")
    with locked(args.work_dir):
        author = Author(args.api_url, args.model, args.seed, args.work_dir)
        try:
            models = author.client.models.list()
            if args.model not in {m.id for m in models.data}:
                raise ValueError(f"Server does not expose {args.model}")
            questions, findings = prepare(author, args.workers)
            if not args.prepare_only:
                generate(author, questions, findings, args)
        finally:
            author.client.close()


if __name__ == "__main__":
    main()
