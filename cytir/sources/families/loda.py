#!/usr/bin/env python3
"""Task 633 — select the FULL in-force DILA LEGI/LODA population, ADR-105 shape.

The legacy driver, the retired `ingest_dila_loda` script (task 140; retired by task 1549), capped this family
by NATURE: it keeps only LOI/ORDONNANCE/DECRET-class articles and drops ~57% of the TNC_en_vigueur
branch -- the ARRETE (ministerial-order) articles -- as an undocumented engineering filter, not a
policy ceiling (see `sources/fr/legi.yaml`, the admission PR this script's output feeds). It also
predates ADR-105: it writes a capped `Passage` straight to JSONL with no persisted L1 record.

This module does neither. It:

  1. Walks every `LEGIARTI*.xml` under a `TNC_en_vigueur` branch of an
     extracted (or increment-applied "view") DILA LEGI dump.
  2. Applies the SAME quality floor task 140 already established (min body
     length, nature-sensitive: 450 chars for the short-form/administrative
     natures DECRET* and ARRETE*, 200 for the statutory natures LOI/
     ORDONNANCE; `aw_common.low_quality` / `extraction_junk` on top) --
     but over EVERY nature, not a fixed allow-list. Loosening the nature cap
     is the point of task 633; the quality floor is not a cap, it is the
     same "kill empty/placeholder bodies" gate every DILA family uses.
  3. Projects each kept article through `parse_legi.parse_article_xml_legal_markdown`
     (`article_url_family="loda"`) -- the SAME per-article Markdown splitter
     already serving this family's current 130,907 production rows
     (`cytir/extract/loda_current.py`) and its codified sibling
     `legifrance-codes`. One article = one `aw.l1-markdown.v1` L1 record =
     one `chunk.law` chunk while its embed text fits `ROW_CHARS`, else several chunks split on
     paragraph boundaries (task 706: `cytir.extract.legi.split_article_markdown`; never a cut).
     Nothing here re-implements the per-article splitter.
  4. Writes TWO aligned outputs from one pass:
       --out-l1     L1 records, directly consumable by
                    `cytir/corpus/persist_l1_records.py --from-jsonl` (ADR-105 precondition
                    to load -- ADR-105 guard #2).
       --out-shards `shard-NNNNN.texts.jsonl.gz` under --shard-dir, the exact
                    `{point_id, text}` contract task 413a's `embed_shards.py`
                    reads (the same producer that embedded the other sealed
                    1024-d fp16 packs -- reused, not reinvented), plus
                    provenance fields (`url`, `anchor`, `nature`, `title`,
                    `source_record_sha256`) a later assembly-binder pass needs
                    and this script otherwise would not owe anyone.

Sub-article chunking (task 706): an article whose embed text exceeds `ROW_CHARS` is split on its
paragraph boundaries into several rows (anchors `article:<id>:chunk-<n>`), never cut. That is the
behaviour `chunk.law.sub` (PROPOSED, not ratified) describes; the contract's `chunk_policy` for this
family is a founder ruling (task 706 question a), so the code is a DRAFT until it is answered.

Does NOT: touch
Qdrant, touch `data/expansion-pack-admission-status.v1.json`, or admit
anything. Task 633's policy PR (`sources/fr/legi.yaml`, `sources/PORTFOLIO.md`)
is the admission gate; this script only stages and seals the material so it is
ready the moment that PR merges.

  python extract_legi_loda_full.py --root <dila view/legi/.../TNC_en_vigueur's
      PARENT, i.e. the dir containing TNC_en_vigueur/> \\
      --out-l1 legi_loda_full.l1.jsonl \\
      --shard-dir out/shards --rows-per-shard 20000 \\
      --release aw-task633-dila-loda-full-v1
"""
from __future__ import annotations

import argparse
import functools
import gzip
import json
import os
import pathlib
import re
import time
from collections import Counter

from lxml import etree

from cytir.core.common import extraction_junk, low_quality
from cytir.core.common import mint_passage_point_id
from cytir.extract.legi import (MAX_ROW_TOKENS, ROW_CHARS, article_chunk_anchor,
                             parse_article_xml_legal_markdown, split_article_markdown)
from cytir.sources.family import CHUNK_NATIVE_CHUNKED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

# Same floor task 140 established for this family (ingest_dila_loda.py),
# extended uniformly to every nature now that the nature allow-list itself is
# lifted: DECRET's own short amendment articles were the documented reason
# for the higher floor, and ARRETE (ministerial orders) is the same shape --
# short, administrative, frequently a single cross-reference sentence.
STATUTORY_NATURES = {"LOI", "ORDONNANCE"}
SHORT_FORM_PREFIXES = ("DECRET", "DÉCRET", "ARRETE", "ARRÊTÉ")
MIN_BODY_STATUTORY = 200
MIN_BODY_SHORT_FORM = 450
# A row's embed text is at most ROW_CHARS (`cytir.extract.legi`, 8000; embed_shards.py max_seq=2560
# tok). Task 633 CUT every longer article there (13,948 rows lost text, dense AND sparse); since
# task 706 a longer article is split into several rows (`split_article_markdown`), never cut.
LIVE_STATES = ("VIGUEUR", "ABROGE_DIFF")  # task 322: deferred abrogation = still applicable


def _text_of(node) -> str:
    if node is None:
        return ""
    raw = etree.tostring(node, encoding="unicode", method="xml")
    return _WS.sub(" ", _TAG.sub(" ", raw)).strip()


def min_body_for(nature: str) -> int:
    if nature in STATUTORY_NATURES:
        return MIN_BODY_STATUTORY
    if nature.startswith(SHORT_FORM_PREFIXES):
        return MIN_BODY_SHORT_FORM
    # An unrecognized nature (the DILA nature vocabulary is not closed) gets
    # the stricter floor -- the same "unknown gets the conservative reading"
    # discipline the quality floor already applies to nature-less articles.
    return MIN_BODY_SHORT_FORM


class Reasons(Counter):
    pass


def evaluate_article(body: bytes, *, path_hint: str = "") -> tuple[dict | None, str]:
    """One XML article -> (kept-record-inputs, reason). record is None on reject.

    Kept path does NOT yet build the markdown record (parse_article_xml_legal_markdown
    is only called by the caller once accept is known, to avoid wasted work on
    rejects) -- this returns the (root, aid, nature, state, raw_body) tuple the
    caller needs, packed into a dict, or None + a reason string.
    """
    try:
        root = etree.fromstring(body)
    except Exception as exc:  # noqa: BLE001
        return None, f"xml_parse_error:{exc.__class__.__name__}"
    state = (root.findtext(".//META_ARTICLE/ETAT") or "").strip()
    if state not in LIVE_STATES:
        return None, f"not_in_force:{state or 'unknown'}"
    aid = (root.findtext(".//META_COMMUN/ID") or "").strip()
    if not aid:
        return None, "no_article_id"
    tnode = root.find(".//CONTEXTE//TEXTE")
    nature = (tnode.get("nature") if tnode is not None else "").strip().upper()
    raw_body = _text_of(root.find(".//BLOC_TEXTUEL/CONTENU"))
    floor = min_body_for(nature)
    if len(raw_body) < floor:
        return None, f"below_body_floor:{nature or 'unknown'}"
    if low_quality(raw_body):
        return None, "low_quality"
    junk = extraction_junk(raw_body)
    if junk:
        return None, f"extraction_junk:{junk}"
    return {"root": root, "aid": aid, "nature": nature, "state": state,
           "raw_body_len": len(raw_body)}, "kept"


_COUNTER = None


def embed_token_counter():
    """`text -> tokens` under the embedder's own tokenizer (the pinned arctic revision of
    `cytir.embed.admitted_stage`, the fast backend, special tokens counted): the counter an over-long
    article is split by. Resolved once. Raises `TokenCounterError` when it cannot be loaded: a
    stage that cannot measure the window must not guess it (tests replace this function)."""
    global _COUNTER
    if _COUNTER is None:
        from cytir.chunk import token_counter
        from cytir.embed import admitted_stage
        tk = token_counter.resolve_tokenizer(admitted_stage.MODEL, admitted_stage.REVISION)
        _COUNTER = lambda text: token_counter.count(tk, text)       # noqa: E731
    return _COUNTER


def build_records(accept: dict, *, body: bytes, archive_member: str,
                  archive_sha256: str, source_release: str,
                  count_tokens=None, split_over_window: bool = False
                  ) -> tuple[dict, list[dict], str]:
    """(accept) -> (l1_record, shard_rows, source_text), all from ONE markdown build.

    An article whose embed text fits `ROW_CHARS` is ONE row under the bare anchor
    `article:<LEGIARTI>`, byte-identical to the task-633 row. A longer one is several rows with
    anchors `article:<LEGIARTI>:chunk-<n>`, each with the same title and article header (task 706).
    `source_text` is the whole embed text before any splitting: the stage-time guard
    (`cytir.sources.staged_text_guard`) checks that the rows together still hold it.

    `split_over_window` (the restage, ruling 27 T6): a short article is also counted in embedder
    tokens and split when it is over the window (the 574 rows over 2,560 tokens). Off: a short
    article is never tokenised and stays byte-identical to the task-633 row."""
    l1 = parse_article_xml_legal_markdown(
        body, archive_member=archive_member, archive_sha256=archive_sha256,
        source_release=source_release, article_url_family="loda",
        source_id="dila-legi-loda-full-v1")
    if l1 is None:
        return None, [], ""
    short = len(l1["title"]) + 2 + len(l1["markdown"]) <= ROW_CHARS
    parts = split_article_markdown(
        l1["markdown"], title=l1["title"], row_chars=ROW_CHARS, max_tokens=MAX_ROW_TOKENS,
        count_tokens=(None if short and not split_over_window
                      else (count_tokens or embed_token_counter())),
        split_over_window=split_over_window)
    rows = []
    for index, part in enumerate(parts, 1):
        if len(parts) == 1:
            anchor, point_id = l1["anchor"], l1["point_id"]
        else:
            anchor = article_chunk_anchor(accept["aid"], index)
            # ID INPUT, exactly: uuid5(NAMESPACE_URL, f"{url}#{anchor}:{sha1(part)}") where `part` is
            # the chunk's markdown WITHOUT the `title\n\n` prefix that the row's `text` carries (the
            # one-row case keeps `l1["point_id"]`, minted over the whole markdown the same way).
            point_id = mint_passage_point_id(l1["url"], anchor, part)
        rows.append({
            "point_id": point_id, "text": f"{l1['title']}\n\n{part}",
            "url": l1["url"], "anchor": anchor, "title": l1["title"],
            "nature": accept["nature"], "article_id": accept["aid"],
            "source_record_sha256": l1["source_record_sha256"],
            "country": "FR", "provenance": "loda", "domain": "legifrance.gouv.fr",
            "lang": "fr", "tier": "A", "source_type": "official", "trust_tier": 1.0,
            "license": "Etalab-2.0", "consent_state": "dump",
            "consent_basis": "official-dump-license",
        })
    return l1, rows, f"{l1['title']}\n\n{l1['markdown']}"


_TRUE, _FALSE = ("1", "true", "yes"), ("0", "false", "no")


def only_split(options: dict) -> bool:
    """The `only_split` option: 1/true/yes = restage only the split articles; absent, empty or
    0/false/no = a full run. Any other value is a refusal: a misspelling must not silently mean a
    full run (task 706)."""
    raw = options.get("only_split")
    if raw is None or str(raw).strip() == "":
        return False
    value = str(raw).strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"loda_exact: only_split={raw!r} is not one of 1/true/yes/0/false/no")


def urls_from(options: dict, option: str = "urls_from") -> frozenset[str] | None:
    """The `urls_from` option (and, with `option="exclude_urls"`, the same file format for the
    exclusion list): a file (`.jsonl`, `.jsonl.gz`, `.txt`, `.gz`) that names URLs to
    stage whatever their row count, as `{"url": ...}` lines or one URL per line. The set is
    explicit and reproducible (its sha256 and count go in the manifest). Returns the set of
    canonical group keys (`supersede.group_key`: www/scheme/case folded, fragment kept), or None
    when the option is absent. A missing or empty file, a line that is not a URL, or `urls_from`
    without `only_split` is a refusal: the selector extends a restage, it is not a second kind of
    full run (ruling 27 T6)."""
    raw = options.get(option)
    if raw is None or str(raw).strip() == "":
        return None
    if not only_split(options):
        raise ValueError(f"loda_exact: {option} needs only_split=true (the selector adds URLs to "
                         "a restage; it never widens a full run)" if option == "urls_from" else
                         f"loda_exact: {option} needs only_split=true (the exclusion list narrows "
                         "a restage; a full run has no exclusion)")
    path = pathlib.Path(str(raw))
    if not path.is_file():
        raise ValueError(f"loda_exact: {option}={raw!r} is not a file")
    st = path.stat()
    return _load_url_keys(str(path), st.st_mtime_ns, st.st_size, option)


@functools.lru_cache(maxsize=4)
def _load_url_keys(file: str, _mtime_ns: int, _size: int,
                   option: str = "urls_from") -> frozenset[str]:
    """The parse of `urls_from`, once per file version (`extract` asks per unit)."""
    path = pathlib.Path(file)
    raw = file
    from cytir.index import supersede as US
    opener = gzip.open if path.suffix == ".gz" else open
    keys: set[str] = set()
    with opener(path, "rt", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            url = line
            if line.startswith("{"):
                try:
                    url = json.loads(line).get("url") or ""
                except json.JSONDecodeError as exc:
                    raise ValueError(f"loda_exact: {path} line {n} is not JSON ({exc})") from exc
            if not str(url).startswith(("http://", "https://")) or "/loda/article_lc/" not in url:
                raise ValueError(f"loda_exact: {path} line {n}: {url!r} is not a loda article URL")
            keys.add(US.group_key(url))
    if not keys:
        raise ValueError(f"loda_exact: {option}={raw!r} names no URL; an empty selector is a "
                         "wrong input")
    return frozenset(keys)


def _sha256_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


def iter_tnc_paths(root_dir: str):
    """Yield the path of every LEGIARTI*.xml under a TNC_en_vigueur branch (os.walk order)."""
    for dirpath, _, files in os.walk(root_dir):
        dp = dirpath + "/"
        if "/TNC_en_vigueur/" not in dp:
            continue
        for fn in files:
            if fn.startswith("LEGIARTI") and fn.endswith(".xml"):
                yield os.path.join(dirpath, fn)


def walk_tnc(root_dir: str):
    """Yield (path, xml_bytes) for every LEGIARTI*.xml under a TNC_en_vigueur branch."""
    for path in iter_tnc_paths(root_dir):
        with open(path, "rb") as fh:
            yield path, fh.read()


class ShardWriter:
    """Writes shard-NNNNN.texts.jsonl.gz at a fixed row cap (embed_shards.py contract)."""

    def __init__(self, out_dir: str, rows_per_shard: int):
        self.out_dir = out_dir
        self.rows_per_shard = rows_per_shard
        os.makedirs(out_dir, exist_ok=True)
        self._shard = 0
        self._buf: list[dict] = []
        self.shards_written = 0
        self.rows_written = 0

    def add(self, row: dict) -> None:
        self._buf.append(row)
        if len(self._buf) >= self.rows_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        name = f"shard-{self._shard:05d}.texts.jsonl.gz"
        path = os.path.join(self.out_dir, name)
        tmp = path + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            for row in self._buf:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        self.rows_written += len(self._buf)
        self.shards_written += 1
        self._shard += 1
        self._buf = []

    def close(self) -> None:
        self._flush()


def run(*, root: str, out_l1: str, shard_dir: str, rows_per_shard: int,
       release: str, limit: int = 0, log=print) -> dict:
    scanned = kept = 0
    reasons: Reasons = Reasons()
    nature_kept: Counter = Counter()
    writer = ShardWriter(shard_dir, rows_per_shard)
    t0 = time.time()
    seen_point_ids: set[str] = set()
    with open(out_l1, "w", encoding="utf-8") as l1_fh:
        for path, body in walk_tnc(root):
            scanned += 1
            accept, reason = evaluate_article(body, path_hint=path)
            if accept is None:
                # `reason` is only ever a REJECT reason here ("kept" short-
                # circuits below) -- reject_reasons must sum to scanned-kept,
                # so a "kept" key must never land in this counter.
                reasons[reason.split(":", 1)[0]] += 1
                if limit and scanned >= limit:
                    break
                continue
            archive_member = os.path.relpath(path, root)
            l1_rec, shard_recs, _source_text = build_records(
                accept, body=body, archive_member=archive_member,
                archive_sha256=_sha256_bytes(body), source_release=release)
            if l1_rec is None:
                reasons["markdown_build_failed"] += 1
                if limit and scanned >= limit:
                    break
                continue
            pid = l1_rec["point_id"]
            if pid in seen_point_ids:
                reasons["duplicate_point_id"] += 1
                if limit and scanned >= limit:
                    break
                continue
            seen_point_ids.add(pid)
            l1_fh.write(json.dumps(l1_rec, ensure_ascii=False) + "\n")
            for shard_rec in shard_recs:
                writer.add(shard_rec)
            kept += 1
            nature_kept[accept["nature"] or "unknown"] += 1
            if kept % 20000 == 0:
                elapsed = time.time() - t0
                log(f"  {kept:,} kept ({scanned:,} scanned, "
                   f"{kept/max(elapsed,1e-9):.0f} rows/s)", )
            if limit and scanned >= limit:
                break
    writer.close()
    elapsed = time.time() - t0
    summary = {
        "scanned": scanned, "kept": kept,
        "kept_ratio": round(kept / scanned, 4) if scanned else 0.0,
        "reject_reasons": dict(reasons.most_common()),
        "kept_by_nature": dict(nature_kept.most_common()),
        "shards_written": writer.shards_written, "rows_written": writer.rows_written,
        "elapsed_seconds": round(elapsed, 1),
        "rows_per_second": round(kept / max(elapsed, 1e-9), 1),
        "out_l1": out_l1, "shard_dir": shard_dir, "release": release,
    }
    log(f"DONE kept={kept:,} scanned={scanned:,} "
       f"({elapsed:.0f}s, {kept/max(elapsed,1e-9):.0f} rows/s) -> {out_l1}, {shard_dir}")
    return summary



# --- the SourceFamily adapter (task 683, S7 part 2) --------------------------------------------

@register_family
class LodaFamily:
    """DILA LEGI/LODA: every in-force non-codified article of an extracted LEGI dump, one
    `chunk.law` row per article (native chunks).

    Options: `root` (required; the dir that CONTAINS a `TNC_en_vigueur/` branch) and `release`
    (required; the `source_release` stamp, e.g. `aw-task633-dila-loda-full-v1`; keep the stamp of
    the served rows, it is part of the staged text), `only_split` (optional, `true`: stage only
    the articles that split into several rows, AND the short rows over the 2,560-token window
    (ruling 27 T6, tokens counted for every short article); the rest end
    `excluded(single_row_unchanged)`; for the restage of the task-633 cut, task 706), `urls_from`
    (optional, needs `only_split`: a file of loda article URLs that are staged as well, whatever
    their row count; an unchanged one keeps its id and opens its URL group for the url-group
    supersede; ruling 27 T6: the 120 URLs outside both earlier tombstone packs), `exclude_urls`
    (optional, needs `only_split`: a file in the same format; a listed article ends its unit as
    `excluded(restage_excluded_by_list)`, whatever else selected it. Its sha256 and count go in
    the manifest. This is the ONLY way to leave a URL out of the restage: `--exclude-ids-file` at
    compose does NOT work, it leaves the drain's rows for that URL in place and makes the
    drain's `docs_checked` differ from the number of groups). A unit is one
    `LEGIARTI*.xml` file, so every file scanned ends `admitted`, `excluded(<reason>)` (the
    same reasons `evaluate_article` gives: `not_in_force:ABROGE`, `below_body_floor:ARRETE`,
    ...) or `quarantined`; the legacy summary's `reject_reasons` is the same table. The pack
    holds the embed-shard rows (`{point_id, text, url, anchor, ...}`); the L1 record stream
    `run()` also writes is reproducible from the sealed XML (same `build_records`), it is not
    a second output of the framework yet (open item, S7 part 3). The dump is a dump-route
    source (ADR-076): no host is contacted, so no per-host consent is resolved."""

    family_id = "loda_exact"
    route = "source_owned_bulk_dump"
    direct_crawl = False
    chunk_policy_id = CHUNK_NATIVE_CHUNKED
    #: The family's stage-time coverage threshold (`cytir.sources.staged_text_guard`; the module's
    #: 0.98 is only the default for a family that declares none). Loda's split loses nothing
    #: (5-gram coverage 1.0 on all 13,948 split articles of the whole dump, task 706), so any loss
    #: is a defect: at 0.98 the guard let 325 of the 13,948 task-633 cuts pass. Not yet a contract
    #: field (legislative follow-up).
    staged_text_min_coverage = 1.0

    def preflight(self, ctx: RunContext):
        """Refuse at the start of the run, before a file is read (task 706). An over-long article
        is split by the embedder's own token count; without the tokenizer a run would admit the
        short articles and quarantine every long one (`extract_failed:TokenCounterError`): a
        stageable pack that silently lost all long articles. Also reads `only_split`, so a
        misspelt value is a refusal, not a full run."""
        only_split(ctx.options)
        urls_from(ctx.options)
        urls_from(ctx.options, "exclude_urls")
        try:
            embed_token_counter()
        except Exception as exc:                                        # noqa: BLE001
            from cytir.embed import admitted_stage
            from cytir.sources.run_family import RunError
            raise RunError(
                f"loda_exact: the embed tokenizer {admitted_stage.MODEL} at the pinned revision "
                f"{admitted_stage.REVISION} cannot be loaded ({type(exc).__name__}); without it "
                "every long article would be quarantined, so the run is refused. Fetch it once "
                "with `HF_HUB_OFFLINE=0 python3 -c \"from huggingface_hub import snapshot_download"
                f"; snapshot_download('{admitted_stage.MODEL}', "
                f"revision='{admitted_stage.REVISION}')\"` into the HF cache of this machine, "
                "then re-run (set HF_HUB_OFFLINE=1 for the run itself)") from exc

    def discover(self, ctx: RunContext):
        root, release = ctx.options.get("root"), ctx.options.get("release")
        if not root or not release:
            raise ValueError("loda_exact needs --opt root=<dir containing TNC_en_vigueur/> "
                             "--opt release=<source_release stamp>")
        if not os.path.isdir(root):
            raise ValueError(f"loda_exact: root {root!r} is not a directory")
        paths = sorted(iter_tnc_paths(root))
        if not paths:
            raise RuntimeError(f"loda_exact: no LEGIARTI*.xml under a TNC_en_vigueur branch of "
                               f"{root!r}; an empty walk is a wrong root, not an empty source")
        ctx.notes.update({"root": str(root), "release": release, "files": len(paths)})
        if only_split(ctx.options):
            ctx.notes["restage_selector"] = {
                "rule": "split (> 1 row) OR over the 2,560-token window OR url in urls_from",
                "split": 0, "over_window": 0, "urls_from_only": 0, "urls_from_also_split": 0}
        listed = urls_from(ctx.options)
        if listed is not None:
            ctx.notes["urls_from"] = {
                "path": str(ctx.options["urls_from"]), "urls": len(listed),
                "sha256": _sha256_bytes(pathlib.Path(str(ctx.options["urls_from"])).read_bytes())}
        barred = urls_from(ctx.options, "exclude_urls")
        if barred is not None:
            ctx.notes["exclude_urls"] = {
                "path": str(ctx.options["exclude_urls"]), "urls": len(barred),
                "sha256": _sha256_bytes(pathlib.Path(str(ctx.options["exclude_urls"])).read_bytes())}
        return [Unit(key=os.path.relpath(p, root), url=pathlib.Path(p).resolve().as_uri(),
                     meta={"path": p}) for p in paths]

    def acquire(self, unit: Unit, ctx: RunContext):
        with open(unit.meta["path"], "rb") as fh:
            body = fh.read()
        return [Captured(unit, body, {"member": unit.key, "archive_sha256": _sha256_bytes(body)})]

    def extract(self, cap: Captured, ctx: RunContext):
        accept, reason = evaluate_article(cap.body, path_hint=cap.unit.key)
        if accept is None:
            raise UnitExcluded(reason)
        restage = only_split(ctx.options)
        l1, rows, source_text = build_records(
            accept, body=cap.body, archive_member=cap.unit.key,
            archive_sha256=cap.meta["archive_sha256"], source_release=ctx.options["release"],
            split_over_window=restage)
        if l1 is None:
            raise UnitExcluded("markdown_build_failed")
        if restage:
            from cytir.index import supersede as US
            listed = urls_from(ctx.options)
            barred = urls_from(ctx.options, "exclude_urls")
            if barred is not None and US.group_key(l1["url"]) in barred:
                raise UnitExcluded("restage_excluded_by_list")
            hit = listed is not None and US.group_key(l1["url"]) in listed
            sel = ctx.notes.get("restage_selector")
            if len(rows) == 1 and not hit:
                # a RESTAGE of the task-633 cut: an article that fits one row, in characters AND
                # in embedder tokens, is byte-identical to the row already served, so it is not
                # staged again (counted, never silent)
                raise UnitExcluded("single_row_unchanged")
            if sel is not None:
                if len(rows) > 1:
                    key = ("over_window" if len(l1["title"]) + 2 + len(l1["markdown"]) <= ROW_CHARS
                           else "split")
                    sel[key] += 1
                    sel["urls_from_also_split"] += int(hit)
                else:
                    sel["urls_from_only"] += 1
        return [Doc(url=rows[0]["url"], text="", lang="fr", source_text=source_text,
                    extra={"rows": rows})]

    def chunk(self, doc: Doc, ctx: RunContext):
        return list(doc.extra["rows"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True,
                    help="dir CONTAINING a TNC_en_vigueur/ branch "
                        "(walked recursively)")
    ap.add_argument("--out-l1", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--rows-per-shard", type=int, default=20000)
    ap.add_argument("--release", required=True,
                    help="source_release stamp, e.g. aw-task633-dila-loda-full-v1")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after scanning this many source files (debug)")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()
    summary = run(root=args.root, out_l1=args.out_l1, shard_dir=args.shard_dir,
                 rows_per_shard=args.rows_per_shard, release=args.release,
                 limit=args.limit)
    print(json.dumps(summary, indent=1))
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
