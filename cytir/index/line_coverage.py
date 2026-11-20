#!/usr/bin/env python3
"""Per-id LINE COVERAGE of a url-group supersede (ruling 27 T6, ADR-114 am.2; task 706 restage).

WHY. The drain's snippet word-coverage guard (`promote_queue.tombstone_word_coverage`) compares the
25-word snippets of the rows, so it counts rows. On real loda URLs the old generation holds up to 11
rows against 2 to 6 new chunk rows, and the guard refused 1,248 of the 13,948 restage URLs although
the new rows hold the whole article. The index stores no body text, so the only place that can
compare text is a stage that still holds it. This module is that comparison, made once, offline,
from sealed inputs, and written beside the pack (by the compose step).

THE MEASURE (one definition; the #630 review's, `scratch-loda-stale-tomb/_scripts/measure.py`).
For one OLD row and the NEW rows at its URL:

  * BODY LINES of a text: split on "\\n"; every line is whitespace-normalised (`\\s+` -> one space)
    and stripped; the legal text itself, so these lines are NOT counted:
      - an empty line (it also ends a skipped block),
      - the metadata blocks `Article identity` and `Legal hierarchy` (the heading line and every
        line up to the next empty line): they differ between generations by design (archive path,
        SHA-256, release stamp),
      - a heading line (starts with `#`, or ends with an anchor `{#...}`),
      - a line shorter than 2 characters.
    The set of DISTINCT body lines is the measure's unit.
  * NEW TEXT at the URL = the texts of all pack rows at that URL (canonical group key,
    `supersede.group_key`), joined with "\\n" and whitespace-normalised.
  * old_lines = the distinct body lines of the old row; found_lines = how many of them occur as a
    substring of the new text; coverage = found_lines / old_lines (1.0 when the old row has no body
    line: nothing of the law text is at stake).

A row of the NEW format (its heading reads `## Article identity`, so the skipped-block test does
not fire; the #630 script behaves the same) contributes its identity lines as body lines; they are
in the new rows of the same release, so they are found.

It differs from the #630 number in one way only: #630 took the UNION of the body lines of all old
rows of a URL; here every old row gets its own number (the ruling asks per id). For a URL with one
old row the two are equal.

THE BLOCK. `{id: {url, coverage, old_lines, found_lines}}` in a file beside the pack
(`line-coverage.v1.json`; the manifest stores its plain file NAME, resolved in the pack's own dir),
bound two ways the drain checks: the manifest carries its sha256; the block carries the sha256 of the
sealed chunks file of the NEW rows (`PACK.json` `chunks_sha256`); every entry is re-checked for
internal consistency (`load`). The block also records the name, size and sha256 of every old-text
source, FOR THE RECORD: the drain does not re-read them (they are big shards on the box that
composed the pack). The TRUST BOUNDARY for the old texts is the manifest: the block was written by
the compose step and the manifest's sha256 binds the file as it was written. An id whose old
text cannot be found, or whose source row names another URL than the one the caller expects, gets NO
entry (never a guessed number): the drain falls back to the snippet guard for its URL.

WHAT IT DOES NOT DO. It decides nothing. The drain (`url_group_supersede.line_coverage_plan`) reads
the block, applies the floor `MIN_COVERAGE` (a constant here, not a manifest field: a manifest cannot
lower it) and skips the snippet guard per URL only when every superseded row of that URL is measured
at or above it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import pathlib
import re
from collections.abc import Iterable, Iterator, Mapping

from cytir.index import supersede as US

SCHEMA = "aw.url-group-line-coverage.v1"
FILE_NAME = "line-coverage.v1.json"
RULED_SKIP_FILE_NAME = "ruled-skip.v1.json"
# Tim's ruling (plan.md T24): the ONE waiver this file can carry. An entry names an exact
# (url, superseded id, measured coverage); it is not a floor change and not a pattern.
RULED_SKIP_RULING = "T24 keep with skip (Tim 2026-10-13)"
RULED_SKIP_TOLERANCE = 1e-6
#: Ruling 27 T6: a superseded row whose line coverage by the new rows is at or above this skips the
#: snippet guard (all superseded rows of its URL must). Below it, or unmeasured: snippet guard.
MIN_COVERAGE = 0.8

_WS = re.compile(r"\s+")
_SKIP_BLOCKS = frozenset({"Article identity", "Legal hierarchy"})
_HEADING = re.compile(r"^#|\{#[A-Za-z0-9_-]+\}\s*$")


class LineCoverageError(ValueError):
    """The block cannot be written, or a block cannot be trusted."""


def normalise(text: str) -> str:
    return _WS.sub(" ", text or "")


def body_lines(text: str) -> set[str]:
    """The distinct body lines of a text (see the module docstring for the definition)."""
    out: set[str] = set()
    skipping = False
    for raw in (text or "").split("\n"):
        line = _WS.sub(" ", raw).strip()
        if not line:
            skipping = False
            continue
        if line in _SKIP_BLOCKS:
            skipping = True
            continue
        if skipping or _HEADING.search(line) or len(line) < 2:
            continue
        out.add(line)
    return out


def line_coverage(old_text: str, new_norm: str) -> tuple[float, int, int]:
    """`(coverage, old_lines, found_lines)` of ONE old row against the whitespace-normalised
    joined text of the new rows at its URL (`normalise("\\n".join(texts))`)."""
    lines = body_lines(old_text)
    found = sum(1 for ln in lines if ln in new_norm)
    return ((found / len(lines)) if lines else 1.0), len(lines), found


def url_key(url: str) -> str:
    return US.group_key(url or "")


# ------------------------------------------------------------------ inputs
def _shards(path: pathlib.Path) -> list[pathlib.Path]:
    if path.is_dir():
        found = sorted(path.glob("shard-*.texts.jsonl.gz")) or sorted(path.glob("*.jsonl.gz"))
        if not found:
            raise LineCoverageError(f"{path}: no shard-*.texts.jsonl.gz file")
        return found
    if not path.is_file():
        raise LineCoverageError(f"{path}: not a file or directory")
    return [path]


def read_text_rows(sources: Iterable[str | pathlib.Path]) -> Iterator[dict]:
    """Every row `{point_id, url, text}` of the sources (files or stage dirs of
    `shard-*.texts.jsonl.gz`). The old texts of loda are the task-417 stage (old generation) and
    the task-633 stage (the cut rows of the new generation): both carry the twin's point ids."""
    for src in sources:
        for shard in _shards(pathlib.Path(src)):
            opener = gzip.open if shard.suffix == ".gz" else open
            with opener(shard, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        if "point_id" in row and "text" in row:
                            yield row


def read_wanted(paths) -> dict[str, str | None]:
    """`{id: url or None}` of the rows the twin may serve at the pack's URLs, the superseded ids
    the block should cover. `paths`: one path or several, united. Each is a `.jsonl[.gz]` of
    `{"id"|"point_id": ..., "url"?: ...}`, of export rows (`{"id": ..., "payload": {"url"}}`, the
    task-705 old-generation export) or of stage text rows (`{"point_id", "url", "text"}`), or a
    file with one id per line. `measure` keeps only the ids at a URL the pack writes."""
    if isinstance(paths, (str, pathlib.Path)):
        paths = [paths]
    out: dict[str, str | None] = {}
    files = [f for path in map(pathlib.Path, paths) for f in _shards(path)]   # a stage dir = its shards
    for path in files:
        opener = gzip.open if path.suffix == ".gz" else open
        before = len(out)
        with opener(path, "rt", encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("{"):
                    out[line] = None
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LineCoverageError(f"{path} line {n}: not JSON ({exc})") from exc
                pid = row.get("id") or row.get("point_id")
                if not pid:
                    raise LineCoverageError(f"{path} line {n}: no id")
                out[str(pid)] = row.get("url") or (row.get("payload") or {}).get("url")
        if len(out) == before:
            raise LineCoverageError(f"{path}: names no id")
    return out


def sha256_file(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ the measurement
def measure(old_rows: Iterable[Mapping], new_rows: Iterable[Mapping], *,
            pack_ids: Iterable[str] = (), wanted: Mapping[str, str | None] | None = None
            ) -> tuple[dict[str, dict], list[str]]:
    """`(entries, unmeasured)`.

    `new_rows`: the pack's rows (`url`, `text`). `old_rows`: candidates (`point_id`, `url`, `text`);
    a row is measured when its URL key is one the pack writes, it is not a pack id, and, with
    `wanted`, its id is wanted and (when `wanted` gives the URL) its URL key equals the wanted
    one. `unmeasured` = wanted ids that got no entry (sorted): text not found, or the source row
    names another URL. Without `wanted` it is empty (the universe of old ids is unknown here)."""
    texts: dict[str, list[str]] = {}
    for r in new_rows:
        texts.setdefault(url_key(r["url"]), []).append(r["text"])
    new_norm = {k: normalise("\n".join(v)) for k, v in texts.items()}
    own = {str(i) for i in pack_ids}
    entries: dict[str, dict] = {}
    for r in old_rows:
        pid = str(r["point_id"])
        if pid in own or (wanted is not None and pid not in wanted):
            continue
        key = url_key(r.get("url") or "")
        if key not in new_norm:
            continue
        if wanted is not None and wanted[pid] is not None and url_key(wanted[pid]) != key:
            continue                    # the source row is another page than the served row
        cov, old_lines, found = line_coverage(r["text"], new_norm[key])
        entries[pid] = {"url": r["url"], "coverage": round(cov, 6), "old_lines": old_lines,
                        "found_lines": found}
    unmeasured: list[str] = []
    if wanted is not None:
        scope = {i for i, u in wanted.items()
                 if i not in own and (u is None or url_key(u) in new_norm)}
        unmeasured = sorted(scope - set(entries))
    return entries, unmeasured


def build_block(*, new_rows: Iterable[Mapping], new_chunks_sha256: str,
                old_sources: list[str | pathlib.Path], wanted: Mapping[str, str | None] | None,
                pack_ids: Iterable[str] = ()) -> dict:
    """The block, from sealed inputs only: the pack's rows, the old-text sources and (optionally)
    the list of ids the drain will find superseded. Nothing here reads Qdrant. Raises
    `LineCoverageError` when no entry could be made at all (a block of nothing is a wrong input)."""
    if not new_chunks_sha256:
        raise LineCoverageError("the block is bound to the sealed chunks file: its sha256 is required")
    if not old_sources:
        raise LineCoverageError("no old-text source given")
    new_list = list(new_rows)
    pack_id_set = list(pack_ids) or [r["point_id"] for r in new_list if "point_id" in r]
    entries, unmeasured = measure(read_text_rows(old_sources), new_list, pack_ids=pack_id_set,
                                  wanted=wanted)
    if not entries:
        raise LineCoverageError("no superseded row has a text in the old-text sources: nothing to "
                                "write (every id would fall back to the snippet guard)")
    srcs = []
    for src in old_sources:
        for shard in _shards(pathlib.Path(src)):
            srcs.append({"name": shard.name, "dir": str(shard.parent), "size": shard.stat().st_size,
                         "sha256": sha256_file(shard)})
    return {"schema": SCHEMA, "definition": "cytir/index/line_coverage.py module docstring",
            "new_chunks_sha256": new_chunks_sha256, "old_text_sources": srcs,
            "wanted_ids_in_scope": None if wanted is None else len(entries) + len(unmeasured),
            "unmeasured_ids": len(unmeasured), "unmeasured_sample": unmeasured[:20],
            "entries": dict(sorted(entries.items()))}


def block_bytes(block: dict) -> bytes:
    """The bytes of the block file (canonical JSON). The writer is `compose.write_line_coverage_block`:
    this module is in the drain's import closure and writes nothing (`test_promote_queue_dry_run`)."""
    return json.dumps(block, sort_keys=True, separators=(",", ":")).encode()


def manifest_fields(block: dict, data: bytes, path: str | pathlib.Path) -> dict:
    """The manifest fields that bind the block file `path` (whose bytes are `data`). The file is
    named by its plain NAME: it lives in the pack dir, next to the manifest, and the drain resolves
    it there (a manifest that is moved with its pack dir stays valid)."""
    return {"line_coverage_file": pathlib.Path(path).name, "line_coverage_sha256": hashlib.sha256(data).hexdigest(),
            "line_coverage": summarize(block)}


def summarize(block: dict) -> dict:
    ent = block.get("entries") or {}
    vals = [e["coverage"] for e in ent.values()]
    return {"ids": len(ent), "urls": len({url_key(e["url"]) for e in ent.values()}),
            "at_or_above_min": sum(1 for v in vals if v >= MIN_COVERAGE),
            "below_min": sum(1 for v in vals if v < MIN_COVERAGE),
            "lowest": min(vals) if vals else None, "unmeasured_ids": block.get("unmeasured_ids"),
            "min_coverage_rule": MIN_COVERAGE}


# ------------------------------------------------------------------ the drain's side
def load(manifest: dict, *, sealed_chunks_sha256: str,
         pack_dir: str | pathlib.Path | None = None) -> dict[str, dict] | None:
    """The block of the manifest, verified, as `{id: entry}`; None when the manifest names none.

    Refuses (`LineCoverageError`): a block typed into the manifest (`line_coverage` carrying
    `entries`); a file whose sha256 is not the manifest's; a block bound to other new rows than the
    sealed pack's; an entry that is not internally consistent (`found_lines` <= `old_lines`,
    `coverage` = found/old, 1.0 when there is no old line) or names no URL. `line_coverage_file` must
    be a plain file name (no directory part); it is read from `pack_dir`, the pack's own dir."""
    typed = manifest.get("line_coverage")
    if isinstance(typed, dict) and "entries" in typed:
        raise LineCoverageError("the queue manifest carries a typed `line_coverage` block; only the "
                                "block file written by the compose step, bound by sha256, counts")
    named = manifest.get("line_coverage_file")
    if not named:
        if manifest.get("line_coverage_sha256"):
            raise LineCoverageError("`line_coverage_sha256` without `line_coverage_file`")
        return None
    want = manifest.get("line_coverage_sha256")
    if pathlib.PurePath(str(named)).name != str(named) or str(named) in ("", ".", ".."):
        raise LineCoverageError(f"line coverage: `line_coverage_file` {named!r} must be a plain file "
                                "name, read from the pack's own dir")
    if pack_dir is None:
        raise LineCoverageError("line coverage: the pack dir is unknown, so the block file "
                                f"{named!r} cannot be located")
    path = pathlib.Path(pack_dir) / str(named)
    if not want or not path.is_file():
        raise LineCoverageError(f"line coverage: {named} is missing or has no sha256 in the manifest")
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise LineCoverageError(f"line coverage: {named} sha256 {got[:12]} != manifest {want[:12]}")
    block = json.loads(data)
    if block.get("schema") != SCHEMA:
        raise LineCoverageError(f"line coverage: schema {block.get('schema')!r}, expected {SCHEMA!r}")
    if block.get("new_chunks_sha256") != sealed_chunks_sha256:
        raise LineCoverageError("line coverage: the block was measured against other new rows than "
                                "the sealed pack's chunks file (sha256 differs)")
    entries = block.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise LineCoverageError("line coverage: the block holds no entry")
    for pid, e in entries.items():
        ok = (isinstance(e, dict) and isinstance(e.get("url"), str) and e["url"]
              and all(isinstance(e.get(k), int) and not isinstance(e.get(k), bool)
                      for k in ("old_lines", "found_lines"))
              and isinstance(e.get("coverage"), (int, float)) and not isinstance(e["coverage"], bool)
              and 0 <= e["found_lines"] <= e["old_lines"]
              and abs(e["coverage"] - ((e["found_lines"] / e["old_lines"]) if e["old_lines"]
                                       else 1.0)) < 1e-5)
        if not ok:
            raise LineCoverageError(f"line coverage: entry {pid} is not a measurement ({e!r})")
    return entries


# ------------------------------------------------------------------ the ruled skip (T24)
_RULED_KEYS = {"url", "point_id", "measured_coverage", "ruling"}


def _is_unit_measure(v: object) -> bool:
    """A finite real number in [0, 1]; bool, NaN and inf are not measures."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and 0 <= v <= 1)


def parse_ruled_skip(data: bytes) -> list[dict]:
    """The entries of a ruled-skip file: a JSON list of `{url, point_id, measured_coverage,
    ruling}`, exactly those keys, `ruling` == `RULED_SKIP_RULING`, one entry per id, no wildcard.
    Raises `LineCoverageError`."""
    try:
        ruled = json.loads(data)
    except ValueError as exc:
        raise LineCoverageError(f"ruled skip: not JSON ({exc})") from exc
    if not isinstance(ruled, list) or not ruled:
        raise LineCoverageError("ruled skip: the file must be a non-empty JSON list of entries")
    seen: set[str] = set()
    for e in ruled:
        ok = (isinstance(e, dict) and set(e) == _RULED_KEYS
              and isinstance(e["url"], str) and e["url"].startswith(("http://", "https://"))
              and isinstance(e["point_id"], str) and e["point_id"] and "*" not in e["point_id"]
              and _is_unit_measure(e["measured_coverage"])
              and e["ruling"] == RULED_SKIP_RULING)
        if not ok:
            raise LineCoverageError(f"ruled skip: entry {e!r} is not {sorted(_RULED_KEYS)} with "
                                    f"ruling {RULED_SKIP_RULING!r}")
        if e["point_id"] in seen:
            raise LineCoverageError(f"ruled skip: id {e['point_id']} is listed twice")
        seen.add(e["point_id"])
    return ruled


def check_ruled_skip(ruled: list[dict], entries: dict[str, dict],
                     served_urls: dict[str, str] | None = None) -> dict[str, dict]:
    """Every ruled entry must (a) name an id that is measured in the block at that exact url (and,
    when `served_urls` = `{superseded id: served url}` is given, a superseded id of that url in
    the pack), (b) match the block's measured coverage within `RULED_SKIP_TOLERANCE` (a drifted
    measure refuses), and (c) sit below `MIN_COVERAGE` (else there is nothing to waive). Returns
    `{id: entry}`; raises `LineCoverageError`."""
    out: dict[str, dict] = {}
    for r in ruled:
        pid, key = r["point_id"], url_key(r["url"])
        e = entries.get(pid)
        if e is None:
            raise LineCoverageError(f"ruled skip: id {pid} has no entry in the line coverage block")
        if url_key(e["url"]) != key:
            raise LineCoverageError(f"ruled skip: id {pid} is measured at {e['url']}, not {r['url']}")
        if served_urls is not None and (pid not in served_urls or url_key(served_urls[pid]) != key):
            raise LineCoverageError(f"ruled skip: id {pid} is not a superseded id of {r['url']} "
                                    "in this pack")
        if abs(e["coverage"] - r["measured_coverage"]) > RULED_SKIP_TOLERANCE:
            raise LineCoverageError(
                f"ruled skip: id {pid} measures {e['coverage']!r} in the block, the ruling was "
                f"given at {r['measured_coverage']!r}; the measure drifted, ask for a new ruling")
        if e["coverage"] >= MIN_COVERAGE:
            raise LineCoverageError(f"ruled skip: id {pid} measures {e['coverage']!r}, at or "
                                    f"above {MIN_COVERAGE}: nothing to waive")
        out[pid] = r
    return out


def ruled_skip_fields(data: bytes, path: str | pathlib.Path, ruled: list[dict]) -> dict:
    """The manifest fields that bind the ruled-skip file (plain name beside the manifest)."""
    return {"ruled_skip_file": pathlib.Path(path).name,
            "ruled_skip_sha256": hashlib.sha256(data).hexdigest(),
            "ruled_skip": {"count": len(ruled), "entries": ruled}}


def load_ruled_skip(manifest: dict, *, pack_dir: str | pathlib.Path | None = None
                    ) -> list[dict] | None:
    """The verified entries of the manifest's ruled-skip file; None when it names none. Refuses a
    typed `ruled_skip` with no file, a sha256 mismatch, a name that is not a plain file name, and
    typed entries that differ from the file's."""
    named, want = manifest.get("ruled_skip_file"), manifest.get("ruled_skip_sha256")
    if not named:
        if want or manifest.get("ruled_skip"):
            raise LineCoverageError("ruled skip: typed in the manifest without `ruled_skip_file`; "
                                    "only the file bound by sha256 counts")
        return None
    if pathlib.PurePath(str(named)).name != str(named) or str(named) in ("", ".", ".."):
        raise LineCoverageError(f"ruled skip: `ruled_skip_file` {named!r} must be a plain file name")
    if pack_dir is None:
        raise LineCoverageError(f"ruled skip: the pack dir is unknown, {named!r} cannot be located")
    path = pathlib.Path(pack_dir) / str(named)
    if not want or not path.is_file():
        raise LineCoverageError(f"ruled skip: {named} is missing or has no sha256 in the manifest")
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise LineCoverageError(f"ruled skip: {named} sha256 {got[:12]} != manifest {want[:12]}")
    ruled = parse_ruled_skip(data)
    typed = manifest.get("ruled_skip")
    if typed is not None and typed != {"count": len(ruled), "entries": ruled}:
        raise LineCoverageError("ruled skip: the manifest's typed entries differ from the file's")
    return ruled
