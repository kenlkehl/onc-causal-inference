"""MMAI v23 long-note pan-cancer oncology corpus integration for Option B (v7).

Provides:
  1. Exact patient-disjoint split resolution against TinyOncBERT's curated note cutoff
     (``artifacts/tinyoncbert_6l_384d/work/curated/manifest.json``, ``notes_read = 564,224``),
     isolating the ``37,631`` ``v23`` ``split == "train"`` patients never read by TinyOncBERT
     (``train_unseen``) while keeping ``validation`` (``8,470``) and ``test`` (``8,340``)
     strictly reserved for zero-shot evaluation.
  2. Dual-format ``mmai-clinical-extraction-v4`` structured patient record parsing
     (supporting both consolidated ``patient_records.parquet`` and live checkpoint JSONs in
     ``.n5000_all_nemotron.checkpoints/patient_records/``) with strict train-only clinical
     vocabulary construction (~130+ note-verified covariates across demographics, stage,
     organ-system site, histology, biomarkers, prior therapies/agents, trajectory events,
     and procedures).
  3. Streaming one-time frozen ``TinyOncBERT`` sentence pre-embedding cache builder using
     balanced multi-note sentence extraction (``note_sentences_balanced``, cap ``320`` clauses
     across all ~14 notes/patient so late/impression/NGS clauses are never truncated) written
     as memory-mapped fp16 arrays (``< 2 GB`` peak host RAM during build, ``< 20 MB`` at load).
  4. Fast GPU-accelerated on-the-fly causal study re-equater (``MMAIv23EpisodeStore``) that
     samples within-cancer (65%) or pan-cancer (35%) cohorts, computes strictly context-only
     chunked spherical k-means MaxSim (``< 50 MB`` VRAM), and re-equates diverse causal DGPs
     (sparse clinical, strictly null-HTE, pure-epistatic ``M1 x M2`` / biomarker conjunctions,
     and dense nonlinear surfaces) in both notes-only and hybrid structured+long-note modes.

CLI:
  python -m oci.longitudinal.mmai_v23 prepare --output-dir artifacts/mmai_v23_option_b_cache
  python -m oci.longitudinal.mmai_v23 evaluate --cache-dir artifacts/mmai_v23_option_b_cache ...
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from oci.tabular_pfn.prior import (
    PriorConfig,
    _apply_nonlinear,
    _eval_rule_ensemble,
    _finalize_outcomes,
    _orthogonalize_interaction,
    _rbf_kernel_surface,
    _standardize_vec,
)

from .events import note_sentences_balanced
from .model import _BIDIRECTIONAL_MODELS

DEFAULT_V23_ROOT = Path("/ksg/kehl_mm_data/mmai/v23/data/no_phi")
DEFAULT_V23_NOTES_PARQUET = (
    DEFAULT_V23_ROOT
    / "synthetic_dataset_n5000_screened_20261106"
    / "patient_summaries"
    / "all_synthetic_notes.parquet"
)
DEFAULT_V23_META_PARQUET = (
    DEFAULT_V23_ROOT
    / "synthetic_dataset_n5000_screened_20261106"
    / "patient_summaries"
    / "synthetic_patient_metadata.parquet"
)
DEFAULT_V23_EXTRACTION_ROOT = DEFAULT_V23_ROOT / "dashboard_synthetic" / "extraction_records"
DEFAULT_TINYONCBERT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "tinyoncbert_6l_384d"
    / "work"
    / "curated"
    / "manifest.json"
)
DEFAULT_BACKBONE_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "tinyoncbert_6l_384d"

CACHE_FORMAT_VERSION = 1
TIMELINE_META_DIM = 16
V23_DGP_MODES = ("sparse_clinical", "null_hte", "epistatic_subgroup", "dense_nonlinear")

# Canonical cancer organ-system buckets for realistic within-cancer cohort sampling
CANCER_GROUPS: tuple[str, ...] = (
    "lung",
    "breast",
    "prostate",
    "colorectal",
    "hematologic",
    "pancreas",
    "liver_biliary",
    "gastric_esophageal",
    "ovary_gyn",
    "renal_urothelial",
    "head_neck",
    "cns_brain",
    "melanoma_skin",
    "sarcoma",
    "endocrine_thyroid",
    "other",
)
CANCER_GROUP_TO_ID = {name: idx for idx, name in enumerate(CANCER_GROUPS)}

_CANCER_GROUP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "hematologic",
        re.compile(
            r"\b(?:bone marrow|lymph node|myeloma|leukemia|lymphoma|mds\b|myelodysplastic|"
            r"myeloproliferative|cll\b|aml\b|b-all\b|t-all\b|acute lymphoblastic|cml\b|"
            r"dlbcl\b|follicular|mantle cell|hodgkin|plasma cell)",
            re.I,
        ),
    ),
    (
        "lung",
        re.compile(
            r"\b(?:lung|pulmonary|bronch|hilar|pleura|mesothelioma|nsclc\b|sclc\b|"
            r"(?:right|left)\s+(?:upper|middle|lower)\s+lobe)",
            re.I,
        ),
    ),
    ("breast", re.compile(r"\b(?:breast|mammary|ductal carcinoma|lobular carcinoma)", re.I)),
    ("prostate", re.compile(r"\b(?:prostat|acinar adenocarcinoma|crpc\b|mcrpc\b)", re.I)),
    (
        "colorectal",
        re.compile(r"\b(?:colon|colorectal|rectum|rectal|cecum|sigmoid|appendix|anal\b|anus\b)", re.I),
    ),
    ("pancreas", re.compile(r"\bpancrea", re.I)),
    (
        "liver_biliary",
        re.compile(r"\b(?:liver|hepat|cholangio|biliary|gallbladder|bile duct|hcc\b)", re.I),
    ),
    (
        "gastric_esophageal",
        re.compile(r"\b(?:gastric|stomach|esophag|gastroesophageal|gej\b|small bowel|duoden)", re.I),
    ),
    (
        "ovary_gyn",
        re.compile(r"\b(?:ovar|endometr|uter|cervi[cx]|fallopian|vulva|vagina|peritone)", re.I),
    ),
    (
        "renal_urothelial",
        re.compile(r"\b(?:kidney|renal|bladder|urothel|ureter|testic|penile)", re.I),
    ),
    (
        "head_neck",
        re.compile(
            r"\b(?:tongue|tonsil|oropharyn|laryn|pharyn|oral\b|nasopharyn|salivary|parotid|"
            r"hypopharyn|sinus|head and neck|mandib|buccal|floor of mouth)",
            re.I,
        ),
    ),
    (
        "cns_brain",
        re.compile(
            r"\b(?:brain|glioma|glioblastoma|astrocytoma|oligodendroglioma|meningioma|"
            r"medulloblastoma|cns\b|cerebr|frontal lobe|temporal lobe|parietal lobe|spinal cord)",
            re.I,
        ),
    ),
    (
        "melanoma_skin",
        re.compile(
            r"\b(?:melanoma|skin|cutaneous|basal cell|merkel cell|scalp|dermis)",
            re.I,
        ),
    ),
    (
        "sarcoma",
        re.compile(
            r"\b(?:sarcoma|gist\b|gastrointestinal stromal|osteosarcoma|liposarcoma|leiomyosarcoma|"
            r"synovial|chondrosarcoma|soft tissue)",
            re.I,
        ),
    ),
    (
        "endocrine_thyroid",
        re.compile(r"\b(?:thyroid|adrenal|neuroendocrine|pituitary|parathyroid|pheochromocytoma)", re.I),
    ),
)

_HISTOLOGY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hist_adenocarcinoma", re.compile(r"\badenocarcinoma\b", re.I)),
    ("hist_squamous", re.compile(r"\bsquamous\b", re.I)),
    ("hist_ductal", re.compile(r"\bductal\b", re.I)),
    ("hist_small_cell", re.compile(r"\bsmall cell\b", re.I)),
    ("hist_melanoma", re.compile(r"\bmelanoma\b", re.I)),
    ("hist_hcc", re.compile(r"\bhepatocellular\b", re.I)),
    ("hist_lymphoma", re.compile(r"\blymphoma\b", re.I)),
    ("hist_leukemia", re.compile(r"\bleukemia\b", re.I)),
    ("hist_myeloma", re.compile(r"\b(myeloma|plasma cell)\b", re.I)),
    ("hist_glioma", re.compile(r"\b(glioma|glioblastoma|astrocytoma)\b", re.I)),
    ("hist_sarcoma", re.compile(r"\bsarcoma\b", re.I)),
    ("hist_urothelial", re.compile(r"\b(urothelial|transitional cell)\b", re.I)),
    ("hist_neuroendocrine", re.compile(r"\b(neuroendocrine|carcinoid)\b", re.I)),
    ("hist_serous", re.compile(r"\bserous\b", re.I)),
)

_DRUG_CLASS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "tx_cls_platinum",
        re.compile(r"\b(cisplatin|carboplatin|oxaliplatin)\b", re.I),
    ),
    (
        "tx_cls_immunotherapy",
        re.compile(
            r"\b(pembrolizumab|nivolumab|atezolizumab|durvalumab|ipilimumab|cemiplimab|"
            r"avelumab|tremelimumab|dostarlimab|relatlimab)\b",
            re.I,
        ),
    ),
    (
        "tx_cls_taxane",
        re.compile(r"\b(paclitaxel|docetaxel|nab-paclitaxel|cabazitaxel)\b", re.I),
    ),
    (
        "tx_cls_anti_vegf",
        re.compile(r"\b(bevacizumab|ramucirumab|aflibercept|lenvatinib|cabozantinib|axitinib|sorafenib|sunitinib)\b", re.I),
    ),
    (
        "tx_cls_targeted_tki",
        re.compile(
            r"\b(osimertinib|erlotinib|gefitinib|afatinib|alectinib|lorlatinib|crizotinib|"
            r"dabrafenib|trametinib|vemurafenib|encorafenib|binimetinib|sotorasib|adagrasib|"
            r"imatinib|dasatinib|ibrutinib|acalabrutinib|zanubrutinib|venetoclax|olaparib|"
            r"niraparib|rucaparib|palbociclib|ribociclib|abemaciclib|alpelisib|trastuzumab|"
            r"pertuzumab|tucatinib|lapatinib|cetuximab|panitumumab)\b",
            re.I,
        ),
    ),
    (
        "tx_cls_hormonal",
        re.compile(
            r"\b(tamoxifen|letrozole|anastrozole|exemestane|fulvestrant|leuprolide|"
            r"goserelin|enzalutamide|abiraterone|apalutamide|darolutamide|bicalutamide)\b",
            re.I,
        ),
    ),
    (
        "tx_cls_anthracycline",
        re.compile(r"\b(doxorubicin|epirubicin|daunorubicin|idarubicin)\b", re.I),
    ),
    (
        "tx_cls_fluoropyrimidine",
        re.compile(r"\b(fluorouracil|5-fu|5fu|capecitabine|leucovorin|folfox|folfiri|folfirinox)\b", re.I),
    ),
    (
        "tx_cls_antimetabolite",
        re.compile(r"\b(gemcitabine|pemetrexed|methotrexate|cytarabine|azacitidine|decitabine)\b", re.I),
    ),
    (
        "tx_cls_alkylator",
        re.compile(r"\b(cyclophosphamide|ifosfamide|temozolomide|bendamustine|melphalan|lomustine)\b", re.I),
    ),
)

_AGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bage\s*[:=]\s*(\d{2,3})(?:\s*(?:years?|yo|y/o|yrs?))?\b", re.I),
    re.compile(r"\baged?\s+(\d{2,3})\s*(?:years?|yo|y/o)\b", re.I),
    re.compile(r"\b(\d{2,3})\s*-\s*year\s*-\s*old\b", re.I),
    re.compile(r"\b(\d{2,3})\s*(?:yo|y/o)\s+(?:male|female|man|woman|patient)\b", re.I),
)
_SEX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:sex|gender)\s*[:=]\s*(female|male)\b", re.I),
    re.compile(r"\b\d{2,3}\s*-\s*year\s*-\s*old\s+(female|male|woman|man)\b", re.I),
    re.compile(r"\bthe patient is a\s+(female|male)\b", re.I),
)

_STAGE_NUM_RE = re.compile(r"\b(?:stage\s*)?(iv|iii|ii|i|0|4|3|2|1)[a-c]?\b", re.I)
_ROMAN_STAGE = {"0": 0.0, "i": 1.0, "1": 1.0, "ii": 2.0, "2": 2.0, "iii": 3.0, "3": 3.0, "iv": 4.0, "4": 4.0}
_HIGH_GRADE_RE = re.compile(r"\b(high[\s-]*grade|poorly differentiated|grade\s*[34]|grade\s*iii|grade\s*iv|undifferentiated)\b", re.I)
_BIO_CLEAN_RE = re.compile(r"[^A-Z0-9+-]+")
_EVENT_TYPES_ORDER = ("clinical_note", "imaging_report", "pathology_report", "ngs_report")


# ==============================================================================
# 1. EXACT TINYONCBERT-DISJOINT SPLIT RESOLUTION
# ==============================================================================
def load_tinyoncbert_notes_cutoff(manifest_path: str | Path | None = DEFAULT_TINYONCBERT_MANIFEST) -> int:
    """Read exact ``notes_read`` cutoff from TinyOncBERT's curation ``manifest.json``."""
    if manifest_path is None:
        return 0
    p = Path(manifest_path)
    if not p.is_file():
        return 0
    data = json.loads(p.read_text(encoding="utf-8"))
    return int(data.get("notes_read", 0))


def resolve_v23_splits(
    notes_parquet: str | Path,
    meta_parquet: str | Path | None = None,
    *,
    tinyoncbert_notes_cutoff: int | None = None,
    tinyoncbert_manifest: str | Path | None = DEFAULT_TINYONCBERT_MANIFEST,
) -> dict[str, Any]:
    """Resolve exact patient splits ensuring zero overlap with TinyOncBERT's training notes.

    Because ``oci.longitudinal.tinyoncbert`` streamed ``all_synthetic_notes.parquet``
    sequentially from row group 0 and stopped after ``notes_read`` rows (564,224), any
    ``patient_id`` appearing in rows ``[0, cutoff)`` is placed into ``train_seen`` (or
    flagged if non-train) so ``train_unseen`` contains strictly ``split == "train"``
    patients with **zero** notes ever read by TinyOncBERT.
    """
    notes_pq = Path(notes_parquet)
    cutoff = (
        int(tinyoncbert_notes_cutoff)
        if tinyoncbert_notes_cutoff is not None
        else load_tinyoncbert_notes_cutoff(tinyoncbert_manifest)
    )

    pf = pq.ParquetFile(notes_pq)
    cols = pf.schema_arrow.names
    read_cols = ["patient_id"] + (["split"] if "split" in cols else [])
    tbl = pf.read(columns=read_cols)
    pids = tbl["patient_id"].to_pylist()
    note_splits = tbl["split"].to_pylist() if "split" in read_cols else ["train"] * len(pids)

    note_counts: Counter[str] = Counter(pids)
    pid_to_split: dict[str, str] = {}
    if meta_parquet is not None and Path(meta_parquet).is_file():
        mtbl = pq.read_table(meta_parquet, columns=["patient_id", "split"])
        for pid, sp in zip(mtbl["patient_id"].to_pylist(), mtbl["split"].to_pylist()):
            pid_to_split[str(pid)] = str(sp)
    for pid, sp in zip(pids, note_splits):
        pid_s = str(pid)
        if pid_s not in pid_to_split:
            pid_to_split[pid_s] = str(sp)

    eff_cutoff = max(0, min(cutoff, len(pids)))
    seen_any_pids = {str(p) for p in pids[:eff_cutoff]}
    seen_train_notes = sum(1 for sp in note_splits[:eff_cutoff] if str(sp) == "train")

    splits: dict[str, set[str]] = {
        "train_unseen": set(),
        "train_seen": set(),
        "validation": set(),
        "test": set(),
    }
    for pid in note_counts:
        pid_s = str(pid)
        sp = pid_to_split.get(pid_s, "train")
        if sp == "train":
            if eff_cutoff > 0 and pid_s in seen_any_pids:
                splits["train_seen"].add(pid_s)
            else:
                splits["train_unseen"].add(pid_s)
        elif sp in ("validation", "val"):
            splits["validation"].add(pid_s)
        elif sp == "test":
            splits["test"].add(pid_s)
        else:
            # Unknown split string defaults conservatively out of train_unseen
            splits.setdefault(sp, set()).add(pid_s)

    return {
        "splits": splits,
        "note_counts": dict(note_counts),
        "total_notes": len(pids),
        "tinyoncbert_notes_cutoff": eff_cutoff,
        "tinyoncbert_train_notes_before_cutoff": seen_train_notes,
        "counts": {k: len(v) for k, v in splits.items()},
    }


# ==============================================================================
# 2. MMAI-CLINICAL-EXTRACTION-V4 PARSER & TRAIN-ONLY FACTOR FEATURIZER
# ==============================================================================
def _normalize_extraction_items(raw_obj: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Normalize both ``patient_records.parquet`` and checkpoint JSON extraction formats."""
    domains = ("diagnoses", "biomarkers", "treatments", "events", "procedures")
    src = None
    if isinstance(raw_obj.get("items"), dict):
        src = raw_obj["items"]
    elif isinstance(raw_obj.get("state"), dict):
        src = raw_obj["state"]
    elif isinstance(raw_obj.get("record"), dict):
        return _normalize_extraction_items(raw_obj["record"])
    out: dict[str, list[dict[str, Any]]] = {d: [] for d in domains}
    if not isinstance(src, dict):
        return out
    for dom in domains:
        lst = src.get(dom) or ()
        norm_list: list[dict[str, Any]] = []
        for entry in lst:
            if not isinstance(entry, dict):
                continue
            fields = entry.get("fields")
            norm_list.append(fields if isinstance(fields, dict) else entry)
        out[dom] = norm_list
    return out


def _resolve_extraction_source(extraction_source: str | Path) -> tuple[str, Path]:
    """Resolve ``extraction_source`` to ``('parquet', path)`` or ``('checkpoint_dir', path)``."""
    p = Path(extraction_source)
    if p.is_file() and p.suffix == ".parquet":
        return "parquet", p
    if p.is_dir():
        for cand in (
            p / "patient_records.parquet",
            p / "n5000_all_nemotron" / "patient_records.parquet",
        ):
            if cand.is_file():
                return "parquet", cand
        for cand in (
            p / ".n5000_all_nemotron.checkpoints" / "patient_records",
            p / "patient_records",
            p,
        ):
            if cand.is_dir() and any(cand.glob("*.json")):
                return "checkpoint_dir", cand
    raise FileNotFoundError(
        f"Could not find patient_records.parquet or patient checkpoint JSONs under {p}"
    )


def load_v23_extraction_records(
    extraction_source: str | Path,
    target_pids: set[str] | None = None,
    *,
    allow_partial: bool = False,
    num_threads: int = 16,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load patient-level ``mmai-clinical-extraction-v4`` items keyed by ``patient_id``."""
    kind, resolved = _resolve_extraction_source(extraction_source)
    allowed = {"complete", "partial"} if allow_partial else {"complete"}
    records: dict[str, dict[str, list[dict[str, Any]]]] = {}

    if kind == "parquet":
        tbl = pq.read_table(
            resolved, columns=["patient_id", "extraction_status", "record_json"]
        )
        for pid, status, rec_json in zip(
            tbl["patient_id"].to_pylist(),
            tbl["extraction_status"].to_pylist(),
            tbl["record_json"].to_pylist(),
        ):
            pid_s = str(pid)
            if target_pids is not None and pid_s not in target_pids:
                continue
            if str(status or "").lower() not in allowed:
                continue
            if not rec_json:
                continue
            try:
                obj = json.loads(rec_json)
            except Exception:
                continue
            records[pid_s] = _normalize_extraction_items(obj)
        return records

    # Checkpoint directory of per-patient JSON files
    json_files = sorted(resolved.glob("*.json"))

    def _read_one(fpath: Path):
        try:
            obj = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            return None
        pid_s = str(obj.get("patient_id") or "")
        if not pid_s:
            return None
        if target_pids is not None and pid_s not in target_pids:
            return None
        status = str(obj.get("status") or obj.get("extraction_status") or "").lower()
        if status not in allowed:
            return None
        if not allow_partial and obj.get("chunk_count") is not None and obj.get("next_chunk") is not None:
            if int(obj["next_chunk"]) < int(obj["chunk_count"]):
                return None
        return pid_s, _normalize_extraction_items(obj)

    workers = max(1, min(int(num_threads), len(json_files) or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_read_one, json_files, chunksize=64):
            if res is not None:
                records[res[0]] = res[1]
    return records


def _clean_biomarker_name(name: Any) -> str:
    raw = str(name or "").upper().strip()
    if not raw:
        return ""
    # Normalize common aliases
    if raw in ("PDL1", "PD-L1", "CD274"):
        return "PD-L1"
    if raw in ("ERBB2", "HER-2", "HER2/NEU", "HER2"):
        return "HER2"
    if raw in ("KI67", "KI-67", "MKI67"):
        return "KI-67"
    if raw in ("MSI-H", "MSI", "MMR", "DMMR"):
        return "MSI"
    if raw in ("ESR1", "ER", "ESTROGEN RECEPTOR"):
        return "ER"
    if raw in ("PGR", "PR", "PROGESTERONE RECEPTOR"):
        return "PR"
    cleaned = _BIO_CLEAN_RE.sub("_", raw).strip("_")
    return cleaned[:24]


def _clean_agent_name(agent: Any) -> str:
    s = re.sub(r"[^a-z0-9-]+", "_", str(agent or "").lower().strip()).strip("_")
    return s[:28]


def _classify_cancer_group(items: dict[str, list[dict[str, Any]]]) -> str:
    blobs: list[str] = []
    for d in items.get("diagnoses") or ():
        blobs.append(str(d.get("primary_site") or ""))
        blobs.append(str(d.get("histology") or ""))
    for ev in items.get("events") or ():
        if str(ev.get("event_type") or "").lower() == "initial_diagnosis":
            blobs.append(str(ev.get("description") or ""))
    text = " ".join(blobs)
    for group_name, pat in _CANCER_GROUP_PATTERNS:
        if pat.search(text):
            return group_name
    return "other"


def parse_demographics_from_notes(notes: list[dict[str, Any]]) -> tuple[float, float]:
    """Parse ``(age_years, sex_female)`` from pretreatment note headers (NaN if absent)."""
    age_val = float("nan")
    sex_val = float("nan")
    for note in notes:
        head = str(note.get("text") or "")[:450]
        if not head:
            continue
        if math.isnan(age_val):
            for pat in _AGE_PATTERNS:
                m = pat.search(head)
                if m:
                    v = float(m.group(1))
                    if 18.0 <= v <= 105.0:
                        age_val = v
                        break
        if math.isnan(sex_val):
            for pat in _SEX_PATTERNS:
                m = pat.search(head)
                if m:
                    tok = m.group(1).lower()
                    sex_val = 1.0 if tok in ("female", "woman") else 0.0
                    break
        if (not math.isnan(age_val)) and (not math.isnan(sex_val)):
            break
    return age_val, sex_val


def _parse_iso_day(date_val: Any) -> float | None:
    if not date_val:
        return None
    s = str(date_val).strip()[:10]
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return float(dt.toordinal())
    except Exception:
        return None


def extract_note_timeline_features(notes: list[dict[str, Any]]) -> np.ndarray:
    """Compute 16 safe operational chart-timeline features (zero clinical/oracle leakage)."""
    n_notes = len(notes)
    if n_notes == 0:
        return np.zeros(TIMELINE_META_DIM, dtype=np.float32)

    type_counts = [0, 0, 0, 0]
    words: list[int] = []
    days: list[float] = []
    for n in notes:
        et = str(n.get("event_type") or "").lower()
        for i, ref in enumerate(_EVENT_TYPES_ORDER):
            if et == ref:
                type_counts[i] += 1
                break
        words.append(len(str(n.get("text") or "").split()))
        d = _parse_iso_day(n.get("date"))
        if d is not None:
            days.append(d)

    w_arr = np.asarray(words, dtype=np.float64)
    if days:
        max_d = max(days)
        min_d = min(days)
        span_days = max(0.0, max_d - min_d)
        n_last_90 = sum(1 for d in days if (max_d - d) <= 90.0)
        n_last_180 = sum(1 for d in days if (max_d - d) <= 180.0)
    else:
        span_days = 0.0
        n_last_90 = 0
        n_last_180 = 0

    inv_n = 1.0 / max(n_notes, 1)
    feats = [
        math.log1p(n_notes),
        math.log1p(type_counts[0]),
        math.log1p(type_counts[1]),
        math.log1p(type_counts[2]),
        math.log1p(type_counts[3]),
        type_counts[0] * inv_n,
        type_counts[1] * inv_n,
        type_counts[2] * inv_n,
        type_counts[3] * inv_n,
        math.log1p(span_days),
        math.log1p(n_last_90),
        math.log1p(n_last_180),
        math.log1p(float(w_arr.mean())),
        math.log1p(float(w_arr.std())),
        math.log1p(float(w_arr.max())),
        math.log1p(float(w_arr.sum())),
    ]
    return np.asarray(feats, dtype=np.float32)


def fit_v23_factor_vocab(
    train_records: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    top_k_biomarkers: int = 60,
    top_k_agents: int = 20,
    min_count: int = 10,
) -> dict[str, Any]:
    """Fit biomarker & drug-agent factor vocabulary strictly on the training split."""
    bio_pos_cnt: Counter[str] = Counter()
    agent_cnt: Counter[str] = Counter()
    eff_min = max(2, min(min_count, max(2, len(train_records) // 25)))

    for items in train_records.values():
        seen_b: set[str] = set()
        for b in items.get("biomarkers") or ():
            bname = _clean_biomarker_name(b.get("biomarker"))
            res = str(b.get("result") or "").lower().strip()
            if bname and res == "positive" and bname not in seen_b:
                bio_pos_cnt[bname] += 1
                seen_b.add(bname)
        seen_a: set[str] = set()
        for t in items.get("treatments") or ():
            for ag in t.get("agents") or ():
                aname = _clean_agent_name(ag)
                if aname and aname not in seen_a:
                    agent_cnt[aname] += 1
                    seen_a.add(aname)

    biomarkers = [k for k, c in bio_pos_cnt.most_common(top_k_biomarkers) if c >= eff_min]
    agents = [k for k, c in agent_cnt.most_common(top_k_agents) if c >= eff_min]

    names: list[str] = []
    is_binary: list[bool] = []

    def _add(name: str, binary: bool):
        names.append(name)
        is_binary.append(binary)

    # Demographics (2)
    _add("demo_age", False)
    _add("demo_sex_female", True)
    # Diagnosis / staging core (4)
    _add("diag_stage_ordinal", False)
    _add("diag_stage_iv_metastatic", True)
    _add("diag_high_grade", True)
    _add("diag_count", False)
    # Cancer organ groups (15 — all except 'other')
    for cg in CANCER_GROUPS[:-1]:
        _add(f"site_{cg}", True)
    # Histologies (14)
    for hname, _ in _HISTOLOGY_PATTERNS:
        _add(hname, True)
    # Aggregate biomarker summary (4)
    _add("bio_n_tested", False)
    _add("bio_n_positive", False)
    _add("bio_has_ngs", True)
    _add("bio_has_ihc", True)
    # Specific canonical biomarkers (up to 60)
    for bname in biomarkers:
        _add(f"bio_{bname}_pos", True)
    # Treatment summary (6) + drug classes (10) + specific agents (up to 20)
    _add("tx_count", False)
    _add("tx_max_line", False)
    _add("tx_had_systemic", True)
    _add("tx_had_surgery", True)
    _add("tx_had_radiation", True)
    _add("tx_had_transplant", True)
    for cname, _ in _DRUG_CLASS_PATTERNS:
        _add(cname, True)
    for aname in agents:
        _add(f"agent_{aname}", True)
    # Trajectory events (7)
    _add("ev_had_progression", True)
    _add("ev_n_progressions", False)
    _add("ev_had_remission", True)
    _add("ev_had_new_metastasis", True)
    _add("ev_had_recurrence", True)
    _add("ev_had_hospitalization", True)
    _add("ev_had_hospice", True)
    # Procedures (5)
    _add("proc_count", False)
    _add("proc_had_resection", True)
    _add("proc_had_bone_marrow", True)
    _add("proc_had_biopsy", True)
    _add("proc_had_cytology", True)

    return {
        "biomarkers": biomarkers,
        "agents": agents,
        "factor_names": names,
        "factor_is_binary": is_binary,
    }


def featurize_v23_patient(
    items: dict[str, list[dict[str, Any]]],
    vocab: dict[str, Any],
    *,
    age_years: float = float("nan"),
    sex_female: float = float("nan"),
) -> tuple[np.ndarray, int]:
    """Convert one patient's ``mmai-clinical-extraction-v4`` record into a float32 factor vector."""
    biomarkers = vocab["biomarkers"]
    agents = vocab["agents"]
    bio_idx = {b: i for i, b in enumerate(biomarkers)}
    agent_idx = {a: i for i, a in enumerate(agents)}

    diags = items.get("diagnoses") or ()
    bios = items.get("biomarkers") or ()
    txs = items.get("treatments") or ()
    evs = items.get("events") or ()
    procs = items.get("procedures") or ()

    # Staging & diagnosis attributes
    stage_vals: list[float] = []
    stage_iv = 0.0
    high_grade = 0.0
    hist_blob_parts: list[str] = []
    for d in diags:
        st_raw = str(d.get("stage_group") or d.get("stage") or "").strip()
        m_st = _STAGE_NUM_RE.search(st_raw)
        if m_st:
            sv = _ROMAN_STAGE.get(m_st.group(1).lower())
            if sv is not None:
                stage_vals.append(sv)
                if sv >= 4.0:
                    stage_iv = 1.0
        ext = str(d.get("extent_at_diagnosis") or "").lower()
        m_cat = str(d.get("m_category") or "").lower()
        if "metast" in ext or "distant" in ext or m_cat.startswith("m1"):
            stage_iv = 1.0
        gr = f"{d.get('grade') or ''} {d.get('histology') or ''}"
        if _HIGH_GRADE_RE.search(gr):
            high_grade = 1.0
        hist_blob_parts.append(str(d.get("histology") or ""))

    stage_ord = max(stage_vals) if stage_vals else (4.0 if stage_iv > 0.5 else float("nan"))
    cancer_group = _classify_cancer_group(items)
    cg_id = CANCER_GROUP_TO_ID[cancer_group]
    hist_blob = " ".join(hist_blob_parts)

    # Biomarkers
    bio_vec = np.zeros(len(biomarkers), dtype=np.float32)
    n_bio_pos = 0
    has_ngs = 0.0
    has_ihc = 0.0
    for b in bios:
        meth = str(b.get("method") or "").lower()
        if "ngs" in meth or "seq" in meth:
            has_ngs = 1.0
        if "ihc" in meth or "immuno" in meth:
            has_ihc = 1.0
        res = str(b.get("result") or "").lower().strip()
        if res == "positive":
            n_bio_pos += 1
            bname = _clean_biomarker_name(b.get("biomarker"))
            idx = bio_idx.get(bname)
            if idx is not None:
                bio_vec[idx] = 1.0

    # Treatments
    max_line = 0.0
    had_sys = had_surg = had_rad = had_txp = 0.0
    tx_blob_parts: list[str] = []
    agent_vec = np.zeros(len(agents), dtype=np.float32)
    for t in txs:
        ttype = str(t.get("treatment_type") or "").lower()
        if "systemic" in ttype:
            had_sys = 1.0
        elif "surg" in ttype:
            had_surg = 1.0
        elif "rad" in ttype:
            had_rad = 1.0
        elif "transplant" in ttype:
            had_txp = 1.0
        try:
            ln = float(t.get("line_of_therapy") or 0)
            if 0.0 <= ln <= 15.0:
                max_line = max(max_line, ln)
        except Exception:
            pass
        tx_blob_parts.append(str(t.get("regimen") or ""))
        for ag in t.get("agents") or ():
            tx_blob_parts.append(str(ag))
            aname = _clean_agent_name(ag)
            idx = agent_idx.get(aname)
            if idx is not None:
                agent_vec[idx] = 1.0
    tx_blob = " ".join(tx_blob_parts)

    # Events
    n_prog = 0
    had_rem = had_nmet = had_rec = had_hosp = had_hospc = 0.0
    for ev in evs:
        et = str(ev.get("event_type") or "").lower()
        if et == "progression":
            n_prog += 1
        elif et == "remission":
            had_rem = 1.0
        elif et == "new_metastasis":
            had_nmet = 1.0
        elif et == "recurrence":
            had_rec = 1.0
        elif et == "hospitalization":
            had_hosp = 1.0
        elif et == "hospice_enrollment":
            had_hospc = 1.0

    # Procedures
    had_resect = had_bm = had_biop = had_cyto = 0.0
    for pr in procs:
        pg = str(pr.get("procedure_group") or "").lower()
        if "resection" in pg or "surg" in pg:
            had_resect = 1.0
        elif "bone_marrow" in pg:
            had_bm = 1.0
        elif "biopsy" in pg:
            had_biop = 1.0
        elif "cytolog" in pg or "aspirat" in pg:
            had_cyto = 1.0

    vals: list[float] = [
        float(age_years),
        float(sex_female),
        float(stage_ord),
        float(stage_iv),
        float(high_grade),
        float(len(diags)),
    ]
    for cg in CANCER_GROUPS[:-1]:
        vals.append(1.0 if cancer_group == cg else 0.0)
    for _, hpat in _HISTOLOGY_PATTERNS:
        vals.append(1.0 if hpat.search(hist_blob) else 0.0)
    vals.extend([float(len(bios)), float(n_bio_pos), has_ngs, has_ihc])
    vals.extend(bio_vec.tolist())
    vals.extend([float(len(txs)), max_line, had_sys, had_surg, had_rad, had_txp])
    for _, dpat in _DRUG_CLASS_PATTERNS:
        vals.append(1.0 if dpat.search(tx_blob) else 0.0)
    vals.extend(agent_vec.tolist())
    vals.extend(
        [
            1.0 if n_prog > 0 else 0.0,
            float(n_prog),
            had_rem,
            had_nmet,
            had_rec,
            had_hosp,
            had_hospc,
            float(len(procs)),
            had_resect,
            had_bm,
            had_biop,
            had_cyto,
        ]
    )
    return np.asarray(vals, dtype=np.float32), cg_id


# ==============================================================================
# 3. STREAMING FROZEN TINYONCBERT PRE-EMBEDDING CACHE BUILDER
# ==============================================================================
@torch.no_grad()
def _encode_sentence_list(
    sentences: list[str],
    backbone,
    tokenizer,
    device: torch.device,
    *,
    batch_size: int = 1024,
    max_length: int = 96,
) -> np.ndarray:
    """Encode unique sentences with frozen TinyOncBERT into L2-normalized fp16 numpy array."""
    hidden_size = int(backbone.config.hidden_size)
    if not sentences:
        return np.empty((0, hidden_size), dtype=np.float16)

    is_bidir = getattr(backbone.config, "model_type", "") in _BIDIRECTIONAL_MODELS
    accepts_use_cache = "use_cache" in inspect.signature(backbone.forward).parameters
    limit = int(getattr(backbone.config, "max_position_embeddings", max_length))
    eff_len = min(max_length, limit)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    order = sorted(range(len(sentences)), key=lambda i: (len(sentences[i]), i))
    out = np.empty((len(sentences), hidden_size), dtype=np.float16)

    for start in range(0, len(order), batch_size):
        b_idx = order[start : start + batch_size]
        b_txt = [sentences[i] for i in b_idx]
        enc = tokenizer(
            b_txt,
            padding=True,
            truncation=True,
            max_length=eff_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        kwargs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if accepts_use_cache:
            kwargs["use_cache"] = False
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            hidden = backbone(**kwargs).last_hidden_state
        hidden = hidden.float()
        mask = enc["attention_mask"]
        if is_bidir:
            w = mask.unsqueeze(-1).float()
            pooled = (hidden * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        else:
            last_pos = mask.sum(dim=1).clamp_min(1) - 1
            pooled = hidden[torch.arange(len(b_txt), device=device), last_pos]
        pooled = F.normalize(pooled, p=2, dim=-1).to(torch.float16).cpu().numpy()
        out[np.asarray(b_idx, dtype=np.int64)] = pooled
    return out


class _SplitWriter:
    """Streaming writer appending fp16 embeddings straight to disk for low peak host RAM."""

    def __init__(self, split_dir: Path, hidden_size: int):
        self.split_dir = split_dir
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_size = hidden_size
        self.bin_path = split_dir / "_sent_embs.fp16.bin"
        self._bin_fh = self.bin_path.open("wb")
        self.offsets: list[int] = [0]
        self.patient_ids: list[str] = []
        self.timeline_rows: list[np.ndarray] = []
        self.factor_rows: list[np.ndarray] = []
        self.cancer_groups: list[int] = []

    def append_patient(
        self,
        patient_id: str,
        embs_fp16: np.ndarray,
        timeline_meta: np.ndarray,
        causal_factors: np.ndarray,
        cancer_group: int,
    ) -> None:
        if embs_fp16.ndim != 2 or embs_fp16.shape[1] != self.hidden_size:
            raise ValueError(f"Expected (M, {self.hidden_size}) fp16 array, got {embs_fp16.shape}")
        if len(embs_fp16) > 0:
            self._bin_fh.write(np.ascontiguousarray(embs_fp16, dtype=np.float16).tobytes())
        self.offsets.append(self.offsets[-1] + int(len(embs_fp16)))
        self.patient_ids.append(patient_id)
        self.timeline_rows.append(np.asarray(timeline_meta, dtype=np.float32))
        self.factor_rows.append(np.asarray(causal_factors, dtype=np.float32))
        self.cancer_groups.append(int(cancer_group))

    def finalize(self) -> dict[str, Any]:
        self._bin_fh.close()
        total_sents = self.offsets[-1]
        n_pats = len(self.patient_ids)

        # Convert raw binary fp16 stream into a standard memmap-friendly .npy file cleanly
        npy_path = self.split_dir / "sent_embs.fp16.npy"
        mm = np.lib.format.open_memmap(
            npy_path, mode="w+", dtype=np.float16, shape=(total_sents, self.hidden_size)
        )
        if total_sents > 0:
            row_bytes = self.hidden_size * 2
            chunk_rows = 65536
            cursor = 0
            with self.bin_path.open("rb") as rfh:
                while cursor < total_sents:
                    take = min(chunk_rows, total_sents - cursor)
                    raw = rfh.read(take * row_bytes)
                    mm[cursor : cursor + take] = np.frombuffer(raw, dtype=np.float16).reshape(
                        take, self.hidden_size
                    )
                    cursor += take
            mm.flush()
        del mm
        if self.bin_path.is_file():
            self.bin_path.unlink()

        np.save(self.split_dir / "sent_offsets.int64.npy", np.asarray(self.offsets, dtype=np.int64))
        tl_mat = (
            np.stack(self.timeline_rows, axis=0).astype(np.float32)
            if self.timeline_rows
            else np.empty((0, TIMELINE_META_DIM), dtype=np.float32)
        )
        np.save(self.split_dir / "timeline_meta.fp32.npy", tl_mat)
        f_mat = (
            np.stack(self.factor_rows, axis=0).astype(np.float32)
            if self.factor_rows
            else np.empty((0, 0), dtype=np.float32)
        )
        np.save(self.split_dir / "causal_factors.fp32.npy", f_mat)
        np.save(
            self.split_dir / "cancer_group.int32.npy",
            np.asarray(self.cancer_groups, dtype=np.int32),
        )
        (self.split_dir / "patient_ids.json").write_text(
            json.dumps(self.patient_ids), encoding="utf-8"
        )
        return {
            "num_patients": n_pats,
            "total_sentences": total_sents,
            "mean_sentences_per_patient": round(total_sents / max(n_pats, 1), 2),
            "cancer_group_counts": {
                CANCER_GROUPS[g]: int(c) for g, c in sorted(Counter(self.cancer_groups).items())
            },
        }


def prepare_v23_cache(
    *,
    output_dir: str | Path,
    notes_parquet: str | Path = DEFAULT_V23_NOTES_PARQUET,
    meta_parquet: str | Path | None = DEFAULT_V23_META_PARQUET,
    extraction_source: str | Path = DEFAULT_V23_EXTRACTION_ROOT,
    backbone_path: str | Path = DEFAULT_BACKBONE_PATH,
    tinyoncbert_manifest: str | Path | None = DEFAULT_TINYONCBERT_MANIFEST,
    tinyoncbert_notes_cutoff: int | None = None,
    include_seen_train: bool = False,
    allow_partial_extractions: bool = False,
    max_sentences_per_patient: int = 320,
    min_sentence_chars: int = 12,
    sent_batch_size: int = 1024,
    sent_max_length: int = 96,
    flush_patients: int = 256,
    max_patients_per_split: int | None = None,
    device: str = "cuda",
    num_io_threads: int = 16,
) -> dict[str, Any]:
    """Build memory-mapped frozen TinyOncBERT + extracted-factor cache for MMAI v23.

    Streams ``all_synthetic_notes.parquet`` in a single sequential pass, flushing each
    patient immediately once all of their notes arrive so peak host RAM stays low (<2 GB).
    By default excludes every patient read by TinyOncBERT (``train_unseen`` only) while
    preserving ``validation`` and ``test`` splits cleanly for zero-shot evaluation.
    """
    t0 = time.monotonic()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    print(f"[mmai_v23] Resolving disjoint splits from {notes_parquet} ...", file=sys.stderr, flush=True)
    split_info = resolve_v23_splits(
        notes_parquet,
        meta_parquet,
        tinyoncbert_notes_cutoff=tinyoncbert_notes_cutoff,
        tinyoncbert_manifest=tinyoncbert_manifest,
    )
    splits_map: dict[str, set[str]] = split_info["splits"]
    note_counts: dict[str, int] = split_info["note_counts"]

    active_splits = ["train_unseen", "validation", "test"]
    if include_seen_train:
        active_splits.append("train_seen")

    # Apply optional cap per split (useful for fast smoke tests / dev subsets)
    if max_patients_per_split is not None and max_patients_per_split > 0:
        for sp in active_splits:
            splits_map[sp] = set(sorted(splits_map[sp])[:max_patients_per_split])

    pid_to_target_split: dict[str, str] = {}
    for sp in active_splits:
        for pid in splits_map[sp]:
            pid_to_target_split[pid] = sp

    print(
        f"[mmai_v23] Split counts: {split_info['counts']} | cutoff={split_info['tinyoncbert_notes_cutoff']:,} | "
        f"loading extraction records for {len(pid_to_target_split):,} patients ...",
        file=sys.stderr,
        flush=True,
    )
    records = load_v23_extraction_records(
        extraction_source,
        target_pids=set(pid_to_target_split.keys()),
        allow_partial=allow_partial_extractions,
        num_threads=num_io_threads,
    )
    # Keep only patients with valid extracted structured records
    pid_to_target_split = {pid: sp for pid, sp in pid_to_target_split.items() if pid in records}
    extracted_counts = Counter(pid_to_target_split.values())
    print(
        f"[mmai_v23] Loaded {len(records):,} complete extracted records: {dict(extracted_counts)}",
        file=sys.stderr,
        flush=True,
    )

    # Fit clinical factor vocabulary STRICTLY on train_unseen (or train_seen if train_unseen empty in unit test)
    vocab_pids = [pid for pid, sp in pid_to_target_split.items() if sp == "train_unseen"]
    if not vocab_pids:
        vocab_pids = [pid for pid, sp in pid_to_target_split.items() if sp.startswith("train")]
    if not vocab_pids:
        vocab_pids = list(pid_to_target_split.keys())
    vocab = fit_v23_factor_vocab({pid: records[pid] for pid in vocab_pids})
    print(
        f"[mmai_v23] Fitted clinical factor vocab strictly on {len(vocab_pids):,} train patients: "
        f"{len(vocab['factor_names'])} factors ({len(vocab['biomarkers'])} biomarkers, "
        f"{len(vocab['agents'])} drug agents)",
        file=sys.stderr,
        flush=True,
    )

    # Load frozen TinyOncBERT backbone
    tokenizer = AutoTokenizer.from_pretrained(backbone_path)
    backbone = AutoModel.from_pretrained(backbone_path).to(dev).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    hidden_size = int(backbone.config.hidden_size)

    writers = {sp: _SplitWriter(out_dir / sp, hidden_size) for sp in active_splits}

    # Stream all_synthetic_notes.parquet in a single pass, flushing completed patients immediately
    pf = pq.ParquetFile(notes_parquet)
    avail_cols = set(pf.schema_arrow.names)
    read_cols = ["patient_id", "synthetic_note"]
    if "event_type" in avail_cols:
        read_cols.append("event_type")
    if "date" in avail_cols:
        read_cols.append("date")
    if "event_number" in avail_cols:
        read_cols.append("event_number")

    buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ready_batch: list[tuple[str, str, list[dict[str, Any]]]] = []
    encoded_patients = 0
    total_target = len(pid_to_target_split)

    def _flush_ready():
        nonlocal encoded_patients
        if not ready_batch:
            return
        all_sents: list[str] = []
        per_pat_spans: list[tuple[int, int]] = []
        per_pat_meta: list[tuple[str, str, np.ndarray, np.ndarray, int]] = []

        for sp, pid, pat_notes in ready_batch:
            # Sort chronologically by (date, event_number)
            pat_notes.sort(key=lambda x: (x.get("date") or "", int(x.get("event_number") or 0)))
            sents = note_sentences_balanced(
                pat_notes,
                max_sentences=max_sentences_per_patient,
                min_chars=min_sentence_chars,
            )
            s0 = len(all_sents)
            all_sents.extend(sents)
            per_pat_spans.append((s0, len(all_sents)))

            tl_meta = extract_note_timeline_features(pat_notes)
            age_y, sex_f = parse_demographics_from_notes(pat_notes)
            factors, cg_id = featurize_v23_patient(
                records[pid], vocab, age_years=age_y, sex_female=sex_f
            )
            per_pat_meta.append((sp, pid, tl_meta, factors, cg_id))

        # Deduplicate sentences within this flush batch before TinyOncBERT forward pass!
        uniq_map: dict[str, int] = {}
        uniq_list: list[str] = []
        remap = np.empty(len(all_sents), dtype=np.int64)
        for idx, s in enumerate(all_sents):
            uid = uniq_map.get(s)
            if uid is None:
                uid = len(uniq_list)
                uniq_map[s] = uid
                uniq_list.append(s)
            remap[idx] = uid

        uniq_embs = _encode_sentence_list(
            uniq_list,
            backbone,
            tokenizer,
            dev,
            batch_size=sent_batch_size,
            max_length=sent_max_length,
        )
        for (sp, pid, tl_meta, factors, cg_id), (s0, s1) in zip(
            per_pat_meta, per_pat_spans, strict=True
        ):
            pat_embs = uniq_embs[remap[s0:s1]] if s1 > s0 else np.empty((0, hidden_size), dtype=np.float16)
            writers[sp].append_patient(pid, pat_embs, tl_meta, factors, cg_id)

        encoded_patients += len(ready_batch)
        ready_batch.clear()
        if encoded_patients % 2000 < flush_patients or encoded_patients == total_target:
            elapsed = time.monotonic() - t0
            rate = encoded_patients / max(elapsed, 1e-3)
            eta = (total_target - encoded_patients) / max(rate, 1e-3)
            print(
                f"[mmai_v23] Encoded {encoded_patients:,}/{total_target:,} patients "
                f"({rate:.1f} pat/s, elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)",
                file=sys.stderr,
                flush=True,
            )

    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg_idx, columns=read_cols)
        pids_c = tbl["patient_id"].to_pylist()
        notes_c = tbl["synthetic_note"].to_pylist()
        etypes_c = tbl["event_type"].to_pylist() if "event_type" in read_cols else [""] * len(pids_c)
        dates_c = tbl["date"].to_pylist() if "date" in read_cols else [""] * len(pids_c)
        enums_c = tbl["event_number"].to_pylist() if "event_number" in read_cols else [0] * len(pids_c)

        for pid, txt, et, dt, en in zip(pids_c, notes_c, etypes_c, dates_c, enums_c):
            pid_s = str(pid)
            sp = pid_to_target_split.get(pid_s)
            if sp is None:
                continue
            buf = buffers[pid_s]
            buf.append(
                {
                    "text": str(txt or ""),
                    "event_type": str(et or ""),
                    "date": str(dt or ""),
                    "event_number": int(en or 0),
                }
            )
            if len(buf) >= note_counts[pid_s]:
                ready_batch.append((sp, pid_s, buffers.pop(pid_s)))
                if len(ready_batch) >= flush_patients:
                    _flush_ready()

    # Flush any remaining buffered patients (defensive against rare count mismatch)
    for pid_s, pat_notes in list(buffers.items()):
        sp = pid_to_target_split.get(pid_s)
        if sp is not None:
            ready_batch.append((sp, pid_s, pat_notes))
    buffers.clear()
    _flush_ready()

    split_summaries = {sp: w.finalize() for sp, w in writers.items()}
    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "notes_parquet": str(notes_parquet),
        "meta_parquet": str(meta_parquet) if meta_parquet is not None else None,
        "extraction_source": str(extraction_source),
        "backbone_path": str(backbone_path),
        "hidden_size": hidden_size,
        "tinyoncbert_notes_cutoff": split_info["tinyoncbert_notes_cutoff"],
        "tinyoncbert_train_notes_before_cutoff": split_info["tinyoncbert_train_notes_before_cutoff"],
        "disjoint_train_only": not include_seen_train,
        "max_sentences_per_patient": max_sentences_per_patient,
        "min_sentence_chars": min_sentence_chars,
        "timeline_meta_dim": TIMELINE_META_DIM,
        "cancer_groups": list(CANCER_GROUPS),
        "vocab": vocab,
        "splits": split_summaries,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[mmai_v23] Cache complete in {(time.monotonic() - t0)/60:.1f}m -> {out_dir}",
        file=sys.stderr,
        flush=True,
    )
    return manifest


# ==============================================================================
# 4. FAST MEMORY-MAPPED EPISODE STORE & ON-THE-FLY CAUSAL RE-EQUATER
# ==============================================================================
@torch.no_grad()
def project_cached_sent_km(
    sent_embs_mm: np.ndarray,
    sent_offsets: np.ndarray,
    ctx_idx: np.ndarray,
    q_idx: np.ndarray,
    sent_km_k: int,
    device: torch.device,
    *,
    max_pool: int = 4096,
    n_iter: int = 10,
    chunk_patients: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-bounded, strictly context-only spherical k-means MaxSim projection on GPU.

    Never reads ``q_idx`` while fitting ``centroids`` (exact zero query leakage) and chunks
    projection over patients so peak VRAM stays ``< 50 MB`` even at ``N_ctx = 5,000``.
    """
    n_ctx = len(ctx_idx)
    n_q = len(q_idx)
    if sent_km_k <= 0 or n_ctx == 0:
        return (
            torch.zeros((n_ctx, 0), device=device, dtype=torch.float32),
            torch.zeros((n_q, 0), device=device, dtype=torch.float32),
        )

    def _gather_slice(indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        starts = sent_offsets[indices]
        ends = sent_offsets[indices + 1]
        lens = ends - starts
        total = int(lens.sum())
        if total == 0:
            return (
                torch.empty((0, sent_embs_mm.shape[1]), device=device, dtype=torch.float32),
                torch.empty((0,), device=device, dtype=torch.long),
            )
        chunks = [sent_embs_mm[int(s) : int(e)] for s, e in zip(starts, ends) if e > s]
        mat_np = np.concatenate(chunks, axis=0)
        pat_ids_np = np.repeat(np.arange(len(indices), dtype=np.int64), lens)
        embs_t = torch.from_numpy(mat_np).to(device=device, dtype=torch.float32, non_blocking=False)
        pids_t = torch.from_numpy(pat_ids_np).to(device=device, dtype=torch.long, non_blocking=False)
        return embs_t, pids_t

    # Pass 1 (context only): compute global mean direction & top-max_pool farthest sentences
    dim = int(sent_embs_mm.shape[1])
    sum_v = torch.zeros((1, dim), device=device, dtype=torch.float32)
    total_ctx_sents = 0
    for s in range(0, n_ctx, chunk_patients):
        embs_c, _ = _gather_slice(ctx_idx[s : s + chunk_patients])
        if len(embs_c) > 0:
            sum_v += embs_c.sum(dim=0, keepdim=True)
            total_ctx_sents += len(embs_c)

    if total_ctx_sents == 0:
        return (
            torch.zeros((n_ctx, sent_km_k), device=device, dtype=torch.float32),
            torch.zeros((n_q, sent_km_k), device=device, dtype=torch.float32),
        )

    mean_v = F.normalize(sum_v, p=2, dim=-1)

    # Maintain exact top-max_pool farthest vectors from mean_v across context chunks
    pool_embs: torch.Tensor | None = None
    pool_dists: torch.Tensor | None = None
    for s in range(0, n_ctx, chunk_patients):
        embs_c, _ = _gather_slice(ctx_idx[s : s + chunk_patients])
        if len(embs_c) == 0:
            continue
        dist_c = 1.0 - (embs_c @ mean_v.T).squeeze(-1)
        if len(embs_c) > max_pool:
            top = torch.topk(dist_c, max_pool, largest=True).indices
            embs_c = embs_c[top]
            dist_c = dist_c[top]
        if pool_embs is None:
            pool_embs = embs_c
            pool_dists = dist_c
        else:
            cand_e = torch.cat([pool_embs, embs_c], dim=0)
            cand_d = torch.cat([pool_dists, dist_c], dim=0)
            if len(cand_d) > max_pool:
                top = torch.topk(cand_d, max_pool, largest=True).indices
                pool_embs = cand_e[top]
                pool_dists = cand_d[top]
            else:
                pool_embs = cand_e
                pool_dists = cand_d

    assert pool_embs is not None
    # Deterministic farthest-point initialization + spherical Lloyd iterations
    k_eff = min(sent_km_k, len(pool_embs))
    p_mean = F.normalize(pool_embs.mean(dim=0, keepdim=True), p=2, dim=-1)
    first_idx = int((1.0 - (pool_embs @ p_mean.T).squeeze(-1)).argmax().item())
    centroids = torch.empty((k_eff, dim), device=device, dtype=torch.float32)
    centroids[0] = pool_embs[first_idx]
    max_sim = pool_embs @ centroids[0]
    for ci in range(1, k_eff):
        nxt = int(max_sim.argmin().item())
        centroids[ci] = pool_embs[nxt]
        max_sim = torch.maximum(max_sim, pool_embs @ centroids[ci])

    for _ in range(n_iter):
        sims = pool_embs @ centroids.T
        labels = sims.argmax(dim=1)
        new_c = torch.zeros_like(centroids)
        new_c.index_add_(0, labels, pool_embs)
        norms = new_c.norm(dim=1, keepdim=True)
        empty = norms[:, 0] < 1e-6
        if empty.any():
            n_empty = int(empty.sum().item())
            worst = torch.topk(sims.max(dim=1).values, n_empty, largest=False).indices
            new_c[empty] = pool_embs[worst]
            norms = new_c.norm(dim=1, keepdim=True)
        centroids = new_c / norms.clamp_min(1e-8)

    # Pass 2: chunked MaxSim projection for context and query independently
    def _project_indices(indices: np.ndarray) -> torch.Tensor:
        out = torch.zeros((len(indices), k_eff), device=device, dtype=torch.float32)
        for s in range(0, len(indices), chunk_patients):
            sub_idx = indices[s : s + chunk_patients]
            embs_c, pids_c = _gather_slice(sub_idx)
            if len(embs_c) == 0:
                continue
            vals = embs_c @ centroids.T
            sub_out = torch.full((len(sub_idx), k_eff), -1.0, device=device, dtype=torch.float32)
            sub_out.scatter_reduce_(
                0, pids_c.unsqueeze(-1).expand(-1, k_eff), vals, reduce="amax", include_self=False
            )
            has_s = torch.zeros(len(sub_idx), device=device, dtype=torch.bool)
            has_s[pids_c] = True
            out[s : s + len(sub_idx)] = torch.where(has_s.unsqueeze(-1), sub_out, torch.zeros_like(sub_out))
        return out

    s_ctx = _project_indices(ctx_idx)
    s_q = _project_indices(q_idx)
    if k_eff < sent_km_k:
        s_ctx = F.pad(s_ctx, (0, sent_km_k - k_eff))
        s_q = F.pad(s_q, (0, sent_km_k - k_eff))
    return s_ctx, s_q


class MMAIv23EpisodeStore:
    """Memory-mapped MMAI v23 long-note cache + on-the-fly causal study re-equater."""

    def __init__(self, cache_dir: str | Path, split: str = "train_unseen"):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        sp_dir = self.cache_dir / split
        if not sp_dir.is_dir():
            raise FileNotFoundError(f"Split {split!r} not found in {self.cache_dir}")

        self.sent_embs = np.load(sp_dir / "sent_embs.fp16.npy", mmap_mode="r")
        self.sent_offsets = np.load(sp_dir / "sent_offsets.int64.npy")
        self.timeline_meta = np.load(sp_dir / "timeline_meta.fp32.npy")
        self.causal_factors = np.load(sp_dir / "causal_factors.fp32.npy")
        self.cancer_group = np.load(sp_dir / "cancer_group.int32.npy")
        self.patient_ids: list[str] = json.loads(
            (sp_dir / "patient_ids.json").read_text(encoding="utf-8")
        )
        self.factor_names: list[str] = list(self.manifest["vocab"]["factor_names"])
        self.factor_is_binary = np.asarray(self.manifest["vocab"]["factor_is_binary"], dtype=bool)
        self.num_patients = len(self.patient_ids)

        # Pre-index patients by cancer group for fast realistic single-cancer cohort draws
        self.group_to_indices: dict[int, np.ndarray] = {}
        for g in np.unique(self.cancer_group):
            self.group_to_indices[int(g)] = np.where(self.cancer_group == g)[0]

        # Precompute split-level median fill values for continuous factors (e.g. age, stage)
        self._col_medians = np.zeros(self.causal_factors.shape[1], dtype=np.float32)
        for j in range(self.causal_factors.shape[1]):
            col = self.causal_factors[:, j]
            fin = col[np.isfinite(col)]
            self._col_medians[j] = float(np.median(fin)) if len(fin) > 0 else 0.0

    def sample_episode(
        self,
        seed: int,
        n_ctx: int,
        n_q: int,
        device: str | torch.device,
        *,
        sent_km_k: int = 256,
        within_cancer_prob: float = 0.65,
        notes_only_prob: float = 0.55,
        dgp_mode: str | None = None,
        prior: PriorConfig = PriorConfig(),
    ) -> dict[str, Any]:
        """Sample one long-note MMAI v23 causal study directly onto ``device``."""
        dev = torch.device(device)
        rng = np.random.default_rng(seed)
        n_tot = min(int(n_ctx + n_q), self.num_patients)
        eff_q = min(int(n_q), max(1, n_tot // 5))
        eff_ctx = n_tot - eff_q

        # 1. Sample cohort: 65% realistic single-cancer study (when group is large enough) vs 35% pan-cancer
        eligible_groups = [
            g for g, idxs in self.group_to_indices.items()
            if len(idxs) >= n_tot and g != CANCER_GROUP_TO_ID["other"]
        ]
        if eligible_groups and rng.random() < within_cancer_prob:
            chosen_g = int(rng.choice(eligible_groups))
            pool_idx = self.group_to_indices[chosen_g]
            sel = rng.choice(pool_idx, size=n_tot, replace=False)
            cohort_scope = f"cancer:{CANCER_GROUPS[chosen_g]}"
        else:
            sel = rng.choice(self.num_patients, size=n_tot, replace=False)
            cohort_scope = "pan_cancer"

        ctx_idx = sel[:eff_ctx]
        q_idx = sel[eff_ctx:]

        # 2. Extract cohort's real mmai-clinical-extraction-v4 latent factors & filter non-degenerate columns
        F_raw = self.causal_factors[sel].copy()
        F_fill = np.where(np.isfinite(F_raw), F_raw, self._col_medians[None, :]).astype(np.float32)
        col_std = F_fill.std(axis=0)
        col_mean = F_fill.mean(axis=0)
        valid_cols: list[int] = []
        for j in range(F_fill.shape[1]):
            if self.factor_is_binary[j]:
                if 0.03 <= col_mean[j] <= 0.97:
                    valid_cols.append(j)
            else:
                if col_std[j] > 0.08:
                    valid_cols.append(j)
        if len(valid_cols) < 4:
            # Fallback to highest-variance columns if cohort is unusually homogeneous
            valid_cols = list(np.argsort(-col_std)[: min(12, F_fill.shape[1])])

        valid_arr = np.asarray(valid_cols, dtype=np.int64)
        F_std = (F_fill[:, valid_arr] - col_mean[None, valid_arr]) / np.maximum(
            col_std[None, valid_arr], 1e-5
        )
        valid_bin = self.factor_is_binary[valid_arr]
        n_avail = len(valid_arr)

        # 3. Choose causal DGP regime & draw active real clinical factors
        if dgp_mode is None:
            mode = str(rng.choice(V23_DGP_MODES, p=[0.35, 0.22, 0.18, 0.25]))
        else:
            mode = str(dgp_mode)
        is_null_hte = mode == "null_hte"

        perm = rng.permutation(n_avail)
        if mode == "dense_nonlinear":
            n_conf = min(int(rng.integers(8, min(24, max(9, n_avail // 2)) + 1)), max(2, n_avail - 2))
            rem = n_avail - n_conf
            n_prog = min(int(rng.integers(2, min(10, max(3, rem // 2)) + 1)), max(0, rem - 1))
            rem -= n_prog
            n_mod = min(int(rng.integers(4, min(16, max(5, rem)) + 1)), max(1, rem)) if rem > 0 else 0
            n_dual = min(int(rng.integers(1, min(6, n_conf) + 1)), n_conf)
        else:
            n_conf = min(int(rng.integers(2, min(10, max(3, n_avail // 2)) + 1)), max(1, n_avail - 1))
            rem = n_avail - n_conf
            n_prog = min(int(rng.integers(1, min(6, max(2, rem // 2)) + 1)), max(0, rem))
            rem -= n_prog
            if is_null_hte:
                n_mod, n_dual = 0, 0
            elif mode == "epistatic_subgroup":
                n_mod = min(int(rng.integers(2, min(7, max(3, rem)) + 1)), max(0, rem))
                n_dual = min(int(rng.integers(0, min(3, n_conf) + 1)), n_conf)
                if n_mod + n_dual < 2 and n_conf >= 2:
                    n_dual = min(2, n_conf)
            else:
                n_mod = min(int(rng.integers(1, min(6, max(2, rem)) + 1)), max(0, rem))
                n_dual = min(int(rng.integers(0, min(3, n_conf) + 1)), n_conf)
                if n_mod + n_dual == 0:
                    n_dual = 1

        c_loc = perm[:n_conf]
        p_loc = perm[n_conf : n_conf + n_prog]
        m_loc = perm[n_conf + n_prog : n_conf + n_prog + n_mod]
        d_loc = c_loc[:n_dual] if n_dual > 0 else np.empty(0, dtype=np.int64)

        mech = str(rng.choice(["linear", "linear", "threshold", "quadratic", "hinge", "saturating"]))

        def _xf(loc_idx: np.ndarray) -> np.ndarray:
            if len(loc_idx) == 0:
                return np.empty((n_tot, 0), dtype=np.float32)
            cols_out = []
            for li in loc_idx:
                m_i = "linear" if valid_bin[li] else (mech if rng.random() < 0.65 else "linear")
                cols_out.append(_apply_nonlinear(F_std[:, li], m_i, rng))
            return np.stack(cols_out, axis=1)

        # Build treatment propensity & baseline outcome equations
        t_signal = np.zeros(n_tot, dtype=np.float32)
        y_base = np.zeros(n_tot, dtype=np.float32)
        ht_part = np.zeros(n_tot, dtype=np.float32)

        y_locs = np.unique(np.concatenate([c_loc, p_loc]))
        if len(m_loc) > 0 and mode != "epistatic_subgroup" and rng.random() < 0.45:
            y_locs = np.unique(np.concatenate([y_locs, m_loc]))

        if mode == "dense_nonlinear":
            if len(c_loc):
                t_lin = _standardize_vec(
                    F_std[:, c_loc] @ rng.normal(0.0, 1.0 / math.sqrt(len(c_loc)), size=len(c_loc)).astype(np.float32)
                )
                t_nl = _rbf_kernel_surface(F_std[:, c_loc], rng)
                wt = float(rng.uniform(0.35, 0.75))
                t_signal = _standardize_vec((1.0 - wt) * t_lin + wt * t_nl)
            if len(y_locs):
                y_lin = _standardize_vec(
                    F_std[:, y_locs] @ rng.normal(0.0, 1.0 / math.sqrt(len(y_locs)), size=len(y_locs)).astype(np.float32)
                )
                y_nl = _rbf_kernel_surface(F_std[:, y_locs], rng)
                wy = float(rng.uniform(0.35, 0.75))
                y_base = _standardize_vec((1.0 - wy) * y_lin + wy * y_nl)
        else:
            if len(c_loc):
                phi_t = _xf(c_loc)
                t_signal += phi_t @ rng.normal(0.0, 1.0 / math.sqrt(len(c_loc)), size=len(c_loc)).astype(np.float32)
                if len(c_loc) >= 2 and rng.random() < 0.45:
                    for _ in range(int(rng.integers(1, min(4, len(c_loc)) + 1))):
                        a, b = rng.choice(len(c_loc), size=2, replace=False)
                        t_signal += float(rng.normal(0.0, 0.50)) * _standardize_vec(phi_t[:, a] * phi_t[:, b])
            if len(y_locs):
                phi_y = _xf(y_locs)
                y_base += phi_y @ rng.normal(0.0, 1.0 / math.sqrt(len(y_locs)), size=len(y_locs)).astype(np.float32)
                if len(y_locs) >= 2 and rng.random() < 0.55:
                    for _ in range(int(rng.integers(1, min(5, len(y_locs)) + 1))):
                        a, b = rng.choice(len(y_locs), size=2, replace=False)
                        y_base += float(rng.normal(0.0, 0.50)) * _standardize_vec(phi_y[:, a] * phi_y[:, b])

        # Build HTE equation
        ht_locs = np.unique(np.concatenate([m_loc, d_loc])) if not is_null_hte else np.empty(0, dtype=np.int64)
        if not is_null_hte and len(ht_locs) > 0:
            if mode == "epistatic_subgroup":
                if len(ht_locs) >= 2 and rng.random() < 0.55:
                    # Strictly pure orthogonal M1 x M2 crossover epistasis (zero 1D slope!)
                    n_pairs = int(rng.integers(2, min(6, len(ht_locs) * (len(ht_locs) - 1) // 2 + 1) + 1))
                    for _ in range(n_pairs):
                        a, b = rng.choice(ht_locs, size=2, replace=False)
                        term = _orthogonalize_interaction(F_std[:, int(a)], F_std[:, int(b)])
                        if float(term.std()) > 1e-4:
                            ht_part += float(rng.normal(0.0, 1.0)) * term
                else:
                    # Depth 2-3 biomarker / clinical conjunction subgroup rules
                    ht_rules, _ = _eval_rule_ensemble(
                        F_std, ht_locs, int(rng.integers(2, 6)), rng, min_depth=2, max_depth=3
                    )
                    ht_part = ht_rules
                if float(ht_part.std()) < 1e-4:
                    ht_part = F_std[:, int(ht_locs[0])]
            elif mode == "dense_nonlinear":
                ht_lin = _standardize_vec(
                    F_std[:, ht_locs] @ rng.normal(0.0, 1.0 / math.sqrt(len(ht_locs)), size=len(ht_locs)).astype(np.float32)
                )
                ht_nl = _rbf_kernel_surface(F_std[:, ht_locs], rng)
                wh = float(rng.uniform(0.45, 0.85))
                ht_part = _standardize_vec((1.0 - wh) * ht_lin + wh * ht_nl)
            else:
                phi_h = _xf(ht_locs)
                ht_part += phi_h @ rng.normal(0.0, 1.0 / math.sqrt(len(ht_locs)), size=len(ht_locs)).astype(np.float32)
                if len(ht_locs) >= 2 and rng.random() < 0.45:
                    for _ in range(int(rng.integers(1, min(3, len(ht_locs)) + 1))):
                        a, b = rng.choice(len(ht_locs), size=2, replace=False)
                        ht_part += float(rng.normal(0.0, 0.65)) * _standardize_vec(phi_h[:, a] * phi_h[:, b])
                if len(c_loc) >= 1 and rng.random() < 0.45:
                    phi_c = _xf(c_loc)
                    for _ in range(int(rng.integers(1, min(3, len(c_loc)) + 1))):
                        ci = int(rng.integers(0, len(c_loc)))
                        mi = int(rng.integers(0, len(ht_locs)))
                        ht_part += float(rng.normal(0.0, 0.70)) * _standardize_vec(phi_c[:, ci] * phi_h[:, mi])

        # Optional unmeasured confounding (20%)
        if rng.random() < min(prior.unmeasured_conf_prob, 0.20):
            u_lat = rng.standard_normal(n_tot).astype(np.float32)
            u_str = float(rng.uniform(0.25, 0.65))
            t_signal = t_signal + u_str * u_lat
            y_base = y_base + u_str * u_lat

        t_obs, y_obs, true_prop, true_y0, true_y1, true_ite, scale_mode = _finalize_outcomes(
            t_signal, y_base, ht_part, not is_null_hte, prior, rng
        )

        # 4. Compute strictly context-only GPU spherical k-means MaxSim over real 14-note charts
        s_ctx, s_q = project_cached_sent_km(
            self.sent_embs, self.sent_offsets, ctx_idx, q_idx, sent_km_k, dev
        )
        tl_all = self.timeline_meta[sel]
        tl_ctx = torch.from_numpy(tl_all[:eff_ctx]).to(dev)
        tl_q = torch.from_numpy(tl_all[eff_ctx:]).to(dev)

        # 5. Assemble observed input columns: notes-only (55%) vs hybrid structured + long-note (45%)
        is_notes_only = bool(rng.random() < notes_only_prob)
        if is_notes_only:
            X_ctx = torch.cat([s_ctx, tl_ctx], dim=1)
            X_q = torch.cat([s_q, tl_q], dim=1)
            obs_mode = "notes_only"
        else:
            # Reveal a random 30-70% subset of active factors as structured columns while keeping
            # the remaining 30-70% strictly note-only, plus 20-80 inactive/distractor columns
            active_loc = np.unique(np.concatenate([c_loc, p_loc, ht_locs]))
            inactive_loc = np.setdiff1d(np.arange(n_avail), active_loc)
            reveal_frac = float(rng.uniform(0.30, 0.70))
            n_rev = max(1, int(round(len(active_loc) * reveal_frac)))
            rev_act = rng.choice(active_loc, size=min(n_rev, len(active_loc)), replace=False)
            n_inact_keep = min(len(inactive_loc), int(rng.integers(15, 65)))
            rev_inact = (
                rng.choice(inactive_loc, size=n_inact_keep, replace=False)
                if n_inact_keep > 0
                else np.empty(0, dtype=np.int64)
            )
            obs_orig = valid_arr[np.sort(np.concatenate([rev_act, rev_inact]))]
            X_struct_np = F_raw[:, obs_orig].copy()
            # Realistic EHR structured missingness (MCAR/MAR 0-30%) on half of hybrid episodes
            if rng.random() < 0.50 and X_struct_np.shape[1] > 0:
                miss_p = rng.uniform(0.0, 0.30, size=X_struct_np.shape[1]).astype(np.float32)
                miss_mask = rng.random(X_struct_np.shape) < miss_p[None, :]
                X_struct_np[miss_mask] = np.nan
            X_s_ctx = torch.from_numpy(X_struct_np[:eff_ctx]).to(dev)
            X_s_q = torch.from_numpy(X_struct_np[eff_ctx:]).to(dev)
            X_ctx = torch.cat([X_s_ctx, s_ctx, tl_ctx], dim=1)
            X_q = torch.cat([X_s_q, s_q, tl_q], dim=1)
            obs_mode = "hybrid"

        return {
            "X_ctx": X_ctx,
            "X_q": X_q,
            "t_ctx": torch.from_numpy(t_obs[:eff_ctx]).to(dev),
            "y_ctx": torch.from_numpy(y_obs[:eff_ctx]).to(dev),
            "true_y0_q": torch.from_numpy(true_y0[eff_ctx:]).to(dev),
            "true_y1_q": torch.from_numpy(true_y1[eff_ctx:]).to(dev),
            "true_prop_q": torch.from_numpy(true_prop[eff_ctx:]).to(dev),
            "true_ite_q": true_ite[eff_ctx:],
            "meta": {
                "split": self.split,
                "cohort_scope": cohort_scope,
                "dgp_mode": mode,
                "obs_mode": obs_mode,
                "scale": scale_mode,
                "n_ctx": eff_ctx,
                "n_q": eff_q,
                "p": int(X_ctx.shape[1]),
                "n_conf": int(len(c_loc)),
                "n_mod": int(len(ht_locs)),
            },
        }


# ==============================================================================
# 5. ZERO-SHOT HELD-OUT EVALUATION ON REALISTIC 14-NOTE MMAI V23 BENCHMARK
# ==============================================================================
@torch.inference_mode()
def evaluate_on_v23(
    tabular_head,
    store: MMAIv23EpisodeStore,
    *,
    studies: int = 32,
    context_size: int = 800,
    query_size: int = 128,
    sent_km_k: int = 256,
    seed: int = 777_001,
    notes_only_prob: float = 0.55,
    prior: PriorConfig = PriorConfig(),
) -> dict[str, Any]:
    """Evaluate a ``SparseTabularCausalPFN`` (or ``MultimodalTabularPFN.tabular_head``) on v23."""
    was_training = tabular_head.training
    tabular_head.eval()
    device = tabular_head.centers.device

    sq_cate, sq_zero, sq_arm, sq_risk, sq_prop = [], [], [], [], []
    ate_errs, within_rs = [], []
    all_pred_ite, all_true_ite = [], []
    by_mode: dict[str, dict[str, list]] = {}
    by_obs: dict[str, dict[str, list]] = {}

    for i in range(studies):
        # Odd seeds strictly reserved for held-out evaluation!
        ep_seed = 2 * (seed + i) + 1
        ep = store.sample_episode(
            ep_seed,
            context_size,
            query_size,
            device,
            sent_km_k=sent_km_k,
            notes_only_prob=notes_only_prob,
            prior=prior,
        )
        ol, pl, sc, _ = tabular_head(ep["X_ctx"], ep["t_ctx"], ep["y_ctx"], ep["X_q"])
        pred_m, _ = tabular_head.risk_stats(ol, sc)
        pred_np = pred_m.float().cpu().numpy()
        pred_ite = pred_np[:, 1] - pred_np[:, 0]
        true_y0 = ep["true_y0_q"].float().cpu().numpy()
        true_y1 = ep["true_y1_q"].float().cpu().numpy()
        true_ite = ep["true_ite_q"]
        true_prop = ep["true_prop_q"].float().cpu().numpy()
        pred_prop = pl.float().sigmoid().cpu().numpy()

        t_np = ep["t_ctx"].float().cpu().numpy()
        y_np = ep["y_ctx"].float().cpu().numpy()
        p1_bar = float(y_np[t_np > 0.5].mean()) if (t_np > 0.5).any() else 0.5
        p0_bar = float(y_np[t_np <= 0.5].mean()) if (t_np <= 0.5).any() else 0.5
        arm_ite = p1_bar - p0_bar

        err_c = (pred_ite - true_ite) ** 2
        sq_cate.append(err_c)
        sq_zero.append(true_ite ** 2)
        sq_arm.append((arm_ite - true_ite) ** 2)
        sq_risk.append(((pred_np[:, 0] - true_y0) ** 2 + (pred_np[:, 1] - true_y1) ** 2) * 0.5)
        sq_prop.append((pred_prop - true_prop) ** 2)
        ate_errs.append(float(pred_ite.mean() - true_ite.mean()) ** 2)
        all_pred_ite.append(pred_ite)
        all_true_ite.append(true_ite)

        w_r = None
        if float(true_ite.std()) > 1e-5 and float(pred_ite.std()) > 1e-5:
            r_val = float(np.corrcoef(pred_ite, true_ite)[0, 1])
            if math.isfinite(r_val):
                within_rs.append(r_val)
                w_r = r_val

        for bucket_map, key in ((by_mode, ep["meta"]["dgp_mode"]), (by_obs, ep["meta"]["obs_mode"])):
            b = bucket_map.setdefault(key, {"sq": [], "zero": [], "wr": [], "prop": []})
            b["sq"].append(err_c)
            b["zero"].append(true_ite ** 2)
            b["prop"].append((pred_prop - true_prop) ** 2)
            if w_r is not None:
                b["wr"].append(w_r)

    p_all = np.concatenate(all_pred_ite)
    t_all = np.concatenate(all_true_ite)
    pool_r = (
        float(np.corrcoef(p_all, t_all)[0, 1])
        if float(p_all.std()) > 1e-5 and float(t_all.std()) > 1e-5
        else 0.0
    )

    def _summarize_buckets(bmap):
        out = {}
        for k, v in sorted(bmap.items()):
            out[k] = {
                "studies": len(v["sq"]),
                "cate_rmse": float(np.sqrt(np.mean(np.concatenate(v["sq"])))),
                "zero_cate_rmse": float(np.sqrt(np.mean(np.concatenate(v["zero"])))),
                "within_r": float(np.mean(v["wr"])) if v["wr"] else 0.0,
                "prop_rmse": float(np.sqrt(np.mean(np.concatenate(v["prop"])))),
            }
        return out

    if was_training:
        tabular_head.train()

    return {
        "split": store.split,
        "studies": studies,
        "context_size": context_size,
        "cate_rmse": float(np.sqrt(np.mean(np.concatenate(sq_cate)))),
        "zero_cate_rmse": float(np.sqrt(np.mean(np.concatenate(sq_zero)))),
        "arm_mean_cate_rmse": float(np.sqrt(np.mean(np.concatenate(sq_arm)))),
        "within_r": float(np.mean(within_rs)) if within_rs else 0.0,
        "pool_r": pool_r,
        "risk_rmse": float(np.sqrt(np.mean(np.concatenate(sq_risk)))),
        "prop_rmse": float(np.sqrt(np.mean(np.concatenate(sq_prop)))),
        "ate_rmse": float(np.sqrt(np.mean(ate_errs))),
        "by_dgp_mode": _summarize_buckets(by_mode),
        "by_obs_mode": _summarize_buckets(by_obs),
    }


# ==============================================================================
# 6. CLI ENTRYPOINT (python -m oci.longitudinal.mmai_v23 prepare|evaluate)
# ==============================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MMAI v23 long-note pan-cancer cache builder & Option B evaluator"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("prepare", help="Pre-embed v23 notes & extract causal factors into cache")
    pr.add_argument("--output-dir", required=True, type=Path)
    pr.add_argument("--notes-parquet", default=DEFAULT_V23_NOTES_PARQUET, type=Path)
    pr.add_argument("--meta-parquet", default=DEFAULT_V23_META_PARQUET, type=Path)
    pr.add_argument("--extraction-source", default=DEFAULT_V23_EXTRACTION_ROOT, type=Path)
    pr.add_argument("--backbone-path", default=DEFAULT_BACKBONE_PATH, type=Path)
    pr.add_argument("--tinyoncbert-manifest", default=DEFAULT_TINYONCBERT_MANIFEST, type=Path)
    pr.add_argument("--tinyoncbert-notes-cutoff", default=None, type=int)
    pr.add_argument("--include-seen-train", action="store_true")
    pr.add_argument("--allow-partial-extractions", action="store_true")
    pr.add_argument("--max-sentences-per-patient", default=320, type=int)
    pr.add_argument("--min-sentence-chars", default=12, type=int)
    pr.add_argument("--sent-batch-size", default=1024, type=int)
    pr.add_argument("--sent-max-length", default=96, type=int)
    pr.add_argument("--flush-patients", default=256, type=int)
    pr.add_argument("--max-patients-per-split", default=None, type=int)
    pr.add_argument("--num-io-threads", default=16, type=int)
    pr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ev = sub.add_parser("evaluate", help="Evaluate a Tabular PFN or Option B model on v23 test/val")
    ev.add_argument("--cache-dir", required=True, type=Path)
    ev.add_argument("--split", default="test")
    ev.add_argument("--tabular-run-dir", default=None, type=Path)
    ev.add_argument("--option-b-run-dir", default=None, type=Path)
    ev.add_argument("--backbone-path", default=DEFAULT_BACKBONE_PATH, type=Path)
    ev.add_argument("--studies", default=64, type=int)
    ev.add_argument("--context-size", default=800, type=int)
    ev.add_argument("--query-size", default=128, type=int)
    ev.add_argument("--sent-km-k", default=256, type=int)
    ev.add_argument("--notes-only-prob", default=0.55, type=float)
    ev.add_argument("--seed", default=777_001, type=int)
    ev.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ev.add_argument("--output", default=None, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.cmd == "prepare":
        manifest = prepare_v23_cache(
            output_dir=args.output_dir,
            notes_parquet=args.notes_parquet,
            meta_parquet=args.meta_parquet,
            extraction_source=args.extraction_source,
            backbone_path=args.backbone_path,
            tinyoncbert_manifest=args.tinyoncbert_manifest,
            tinyoncbert_notes_cutoff=args.tinyoncbert_notes_cutoff,
            include_seen_train=args.include_seen_train,
            allow_partial_extractions=args.allow_partial_extractions,
            max_sentences_per_patient=args.max_sentences_per_patient,
            min_sentence_chars=args.min_sentence_chars,
            sent_batch_size=args.sent_batch_size,
            sent_max_length=args.sent_max_length,
            flush_patients=args.flush_patients,
            max_patients_per_split=args.max_patients_per_split,
            device=args.device,
            num_io_threads=args.num_io_threads,
        )
        print(json.dumps({"splits": manifest["splits"], "elapsed_s": manifest["elapsed_s"]}, indent=2))
    elif args.cmd == "evaluate":
        from oci.tabular_pfn.training import load_tabular_run
        from .option_b import load_option_b_run

        store = MMAIv23EpisodeStore(args.cache_dir, split=args.split)
        results: dict[str, Any] = {}
        if args.tabular_run_dir is not None:
            tab_m, tab_s = load_tabular_run(args.tabular_run_dir, device=args.device, prefer_best=True)
            res_t = evaluate_on_v23(
                tab_m,
                store,
                studies=args.studies,
                context_size=args.context_size,
                query_size=args.query_size,
                sent_km_k=args.sent_km_k,
                seed=args.seed,
                notes_only_prob=args.notes_only_prob,
            )
            res_t["checkpoint_step"] = int(tab_s.get("step", -1))
            results["tabular_pfn"] = res_t
        if args.option_b_run_dir is not None:
            ob_m, ob_s = load_option_b_run(
                args.option_b_run_dir,
                device=args.device,
                prefer_best=True,
                backbone_override=args.backbone_path,
            )
            res_b = evaluate_on_v23(
                ob_m.tabular_head,
                store,
                studies=args.studies,
                context_size=args.context_size,
                query_size=args.query_size,
                sent_km_k=args.sent_km_k,
                seed=args.seed,
                notes_only_prob=args.notes_only_prob,
            )
            res_b["checkpoint_step"] = int(ob_s.get("step", -1))
            results["option_b"] = res_b
        if not results:
            raise ValueError("Provide at least one of --tabular-run-dir or --option-b-run-dir")
        text = json.dumps(results, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
