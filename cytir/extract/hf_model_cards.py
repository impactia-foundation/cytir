"""Hugging Face model cards -> structured summary passage + chunked README (`ai_hf_model_cards`).

Moved out of `cytir.sources.hf_model_cards.ingest` (task 1101, S7) so the source family
(`cytir.sources.families.hf_model_cards`), the legacy CLI and the refresh lane share one card shape.
The functions are the driver's own, UNCHANGED; `records_for_model` is the body of the legacy
`parse` loop, lifted so the family can call it per model.
"""
from __future__ import annotations

import json
import os
import re

from cytir.chunk.generic import chunk_generic_records
from cytir.core.common import (Passage, extraction_junk, low_quality, passage_from_chunk,
                            passage_source_record)


MAJOR_ORGS = ["meta-llama", "mistralai", "google", "openai", "deepseek-ai",
              "Qwen", "microsoft", "anthropic", "stabilityai",
              "black-forest-labs", "sentence-transformers"]
TOP_DOWNLOADS = 3000
TOP_LIKES = 500
MIN_BODY_WORDS = 50
MAX_BODY_CHUNKS = 40           # guard a giant README flooding the index
MIN_INTERVAL = 0.28            # <=4 req/s politeness (per-process)

DOMAIN = "huggingface.co"
TIER = "C"
SOURCE_TYPE = "reference"
TRUST = 0.75
PROVENANCE = "ai_hf_model_cards"

# re-quant / duplicate-spam signals (name suffixes + tags). Original authors
# (major orgs) are NEVER treated as quant-spam.
_QUANT_RE = re.compile(
    r"(?i)(?:^|[-_.])(gguf|ggml|awq|gptq|exl2|exl3|eetq|mlx|onnx|"
    r"(?:[248]|w[248])bit|int[248]|q[2-8]_[0-9a-z]+|nf4|fp8|bnb|"
    r"quantized|smashed)(?:$|[-_.])")
_QUANT_ORGS = {"thebloke", "mradermacher", "bartowski", "lmstudio-community",
               "prunaai", "quantfactory", "second-state", "modelscope"}
_QUANT_TAGS = {"gguf", "awq", "gptq", "exl2", "4-bit", "8-bit", "2-bit",
               "3-bit", "5-bit", "6-bit"}
_LANGS_OK = {"en", "de", "fr", "it"}


def _safe(mid):
    return mid.replace("/", "__")



_FRONTMATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.S)
_BADGE_LINE = re.compile(r"^\s*(\[!\[|!\[|<img|<a\s|</a>|\|).*$", re.I)


def _human_params(total):
    if not total:
        return None
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if total >= div:
            return f"{total / div:.1f}{suf}".replace(".0", "")
    return str(total)


def _date(s):
    return (s or "")[:10] or None


def _pick_lang(card):
    langs = card.get("language") or card.get("languages")
    if isinstance(langs, str):
        langs = [langs]
    if isinstance(langs, list):
        lls = [str(l).lower()[:2] for l in langs]
        # card prose is almost always English; only label non-en when the
        # model does NOT declare en at all (multilingual tag lists are often
        # alphabetical — 'ar','de',… sort before 'en'; don't mislabel those)
        if "en" in lls:
            return "en", langs
        for ll in lls:
            if ll in _LANGS_OK:
                return ll, langs
        return "en", langs
    return "en", None


def _structured_text(mid, detail, row):
    card = detail.get("cardData") or {}
    org = mid.split("/")[0]
    st = detail.get("safetensors") or {}
    params = _human_params(st.get("total"))
    lic = card.get("license") or (
        ",".join(t.split(":", 1)[1] for t in detail.get("tags", [])
                 if t.startswith("license:")) or None)
    pipeline = detail.get("pipeline_tag") or card.get("pipeline_tag")
    lang, langs = _pick_lang(card)
    base = card.get("base_model")
    if isinstance(base, list):
        base = ", ".join(map(str, base[:3]))
    datasets = card.get("datasets")
    if isinstance(datasets, list):
        datasets = ", ".join(map(str, datasets[:3]))
    lib = detail.get("library_name") or card.get("library_name")
    dl = detail.get("downloads", row.get("downloads"))
    likes = detail.get("likes", row.get("likes"))
    tags = [t for t in (detail.get("tags") or [])
            if ":" not in t and t not in ("region:us", "region:eu")][:8]
    parts = [f"{mid} is a Hugging Face model card"]
    if pipeline:
        parts[0] += f" for the task {pipeline.replace('-', ' ')}"
    parts[0] += f", published by {org}."
    if params:
        parts.append(f"Parameters: {params}.")
    if lic:
        parts.append(f"License: {lic}.")
    if lib:
        parts.append(f"Library: {lib}.")
    if langs:
        parts.append(f"Languages: {', '.join(map(str, langs[:6]))}.")
    if base:
        parts.append(f"Base model: {base}.")
    if datasets:
        parts.append(f"Trained on: {datasets}.")
    if dl is not None:
        parts.append(f"Downloads (30d): {dl}.")
    if likes is not None:
        parts.append(f"Likes: {likes}.")
    cd, lm = _date(detail.get("createdAt")), _date(detail.get("lastModified"))
    if cd:
        parts.append(f"Created: {cd}.")
    if lm:
        parts.append(f"Last modified: {lm}.")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}.")
    parts.append(f"URL: https://huggingface.co/{mid}")
    return " ".join(parts), lang, lm


def _is_quant_spam(mid, detail, row, is_major, top500):
    if is_major or top500:
        return False
    org = mid.split("/")[0].lower()
    name = mid.split("/")[-1]
    tags = set(t.lower() for t in ((detail or {}).get("tags")
                                   or row.get("tags") or []))
    if org in _QUANT_ORGS:
        return True
    if _QUANT_RE.search(name):
        return True
    if tags & _QUANT_TAGS:
        return True
    return False


def _clean_body(md):
    md = _FRONTMATTER.sub("", md, count=1)
    md = re.sub(r"<!--.*?-->", " ", md, flags=re.S)
    lines = [l for l in md.splitlines() if not _BADGE_LINE.match(l)]
    return "\n".join(lines).strip()


def card_records(mid, detail, readme, *, row=None, gated=False, discovery=None,
                 seen_pid=None, stats=None):
    """The ONE task-120 card shape, shared by `cmd_parse` and the refresh
    lane (`direct_compose`, task 693): a structured summary
    passage (always) plus the README body chunked (skipped when gated, empty
    or junk). Title is the repo id (`org/model`), never the HTML page title.
    Yields `(record, kind)`, kind in {"structured", "body"}. `readme` is the
    raw README text or None."""
    row = row or {}
    seen_pid = set() if seen_pid is None else seen_pid
    stats = {} if stats is None else stats

    def bump(key, n=1):
        stats[key] = stats.get(key, 0) + n

    text, lang, lm = _structured_text(mid, detail, row)
    url = f"https://huggingface.co/{mid}"
    p = Passage(url=url, title=mid, snippet="", text=text, lang=lang,
                domain=DOMAIN, tier=TIER, source_type=SOURCE_TYPE,
                trust_tier=TRUST, anchor=None, last_seen=lm)
    p.snippet = text  # embed --keep-snippet-words 25 truncates
    if p.point_id not in seen_pid:
        seen_pid.add(p.point_id)
        rec = p.payload()
        rec["text"] = p.text
        rec["provenance"] = PROVENANCE
        # A recurring official-publisher delta retains the API response
        # identity that discovered this URL.  It is source lineage, not a
        # retrieval feature, and survives fetch/parse into the staged pack.
        if discovery:
            rec["source_discovery"] = discovery
        if gated:
            rec["gated"] = True
        bump("structured_passages")
        yield rec, "structured"
    if gated:
        bump("gated_metadata_only")
        return
    if readme is None:
        bump("empty_body_dropped")
        return
    body = _clean_body(readme)
    if len(body.split()) < MIN_BODY_WORDS:
        bump("empty_body_dropped")
        return
    nchunks = 0
    for unit in chunk_generic_records(body):
        ch = unit.text
        if nchunks >= MAX_BODY_CHUNKS:
            break
        if low_quality(ch) or extraction_junk(ch):
            bump("body_junk_chunks")
            continue
        from cytir.core.common import extractive_snippet
        bp = passage_from_chunk(
            unit, url=url, title=mid, snippet=extractive_snippet(ch),
            lang=lang, domain=DOMAIN, tier=TIER,
            source_type=SOURCE_TYPE, trust_tier=TRUST,
            anchor=None, last_seen=lm)
        if bp.point_id in seen_pid:
            continue
        seen_pid.add(bp.point_id)
        rec = passage_source_record(bp)
        rec["provenance"] = PROVENANCE
        if discovery:
            rec["source_discovery"] = discovery
        bump("body_passages")
        nchunks += 1
        yield rec, "body"


def base_model_repos(detail):
    """Repo ids a model card declares as its base: `cardData.base_model`
    (str or list) and the Hub's `base_model:<relation>:<repo>` tags."""
    out = []
    base = (detail.get("cardData") or {}).get("base_model")
    for b in ([base] if isinstance(base, str) else (base or [])):
        if isinstance(b, str) and "/" in b:
            out.append(b.strip())
    for t in detail.get("tags") or []:
        if t.startswith("base_model:"):
            repo = t.split(":", 2)[-1]
            if "/" in repo:
                out.append(repo)
    return sorted(set(out))


def derivative_base(mid, detail):
    """Task 693 (generic, no org special-casing): a card whose declared base
    model is owned by a DIFFERENT org than the card itself is a derivative
    (quantization, finetune or re-upload). Returns the foreign base repo id
    or None. Same-org bases (an org's own variants) are not derivatives."""
    owner = mid.split("/")[0].lower()
    for repo in base_model_repos(detail):
        if repo.split("/")[0].lower() != owner:
            return repo
    return None


def records_for_model(m, cache_dir, top500, seen_pid, stats):
    """The legacy `parse` loop body for ONE enumerated model (`m` = a `model_list.json` entry): the
    records it yields, in order, and the counters it bumps in `stats`. `parse` and the source
    family both call this, so the two cannot drift."""
    mid = m["id"]
    row = m.get("row") or {}
    base = os.path.join(cache_dir, _safe(mid))
    meta_p = base + ".meta.json"
    if not os.path.exists(meta_p):
        stats["no_detail"] += 1
        return []
    meta = json.load(open(meta_p))
    detail = None
    if os.path.exists(base + ".detail.json"):
        try:
            detail = json.load(open(base + ".detail.json"))
        except Exception:  # noqa: BLE001
            detail = None
    if detail is None:
        detail = {"tags": row.get("tags", []),
                  "pipeline_tag": row.get("pipeline_tag"),
                  "downloads": row.get("downloads"),
                  "likes": row.get("likes"),
                  "library_name": row.get("library_name"),
                  "createdAt": row.get("createdAt"),
                  "lastModified": row.get("lastModified"),
                  "cardData": {}, "safetensors": {}}
    if detail.get("private") or detail.get("disabled"):
        stats["private_disabled"] += 1
        return []
    is_major = any(r.startswith("org:") for r in m.get("reasons", []))
    if _is_quant_spam(mid, detail, row, is_major, mid in top500):
        stats["quant_spam_skipped"] += 1
        return []
    gated = bool(meta.get("gated") or detail.get("gated") or row.get("gated"))
    readme_p = base + ".README.md"
    readme = None
    if not gated and os.path.exists(readme_p):
        readme = open(readme_p, encoding="utf-8", errors="replace").read()
    recs = [rec for rec, _kind in card_records(mid, detail, readme, row=row, gated=gated,
                                                discovery=m.get("discovery"),
                                                seen_pid=seen_pid, stats=stats)]
    stats["kept_models"] += 1
    for r in m.get("reasons", []):
        key = "org" if r.startswith("org:") else r
        stats["reasons"][key] = stats["reasons"].get(key, 0) + 1
    return recs


def new_parse_stats(enumerated):
    return {"enumerated": enumerated, "no_detail": 0, "private_disabled": 0,
            "quant_spam_skipped": 0, "gated_metadata_only": 0,
            "empty_body_dropped": 0, "kept_models": 0,
            "structured_passages": 0, "body_passages": 0,
            "body_junk_chunks": 0, "reasons": {}}
