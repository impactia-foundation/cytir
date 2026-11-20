"""Ratchet: no NEW direct read of a coded key in a module that reads stored Qdrant rows (task 855).

From the aw_e10 flip the twin is coded (enum codes, a 32-hex digest). A module that reads stored
rows (it scrolls or retrieves points) and then reads `content_digest` or an enum key straight off
the payload compares a code with a string or 32 hex with 64 -- silently. The one way to read such a
row is `cytir.index.row_decode` (`reader(collection)`, `canonical(row)`, `digest_key`).

The scan: every `cytir/` module that reads stored rows (`READ_MARKERS`), every line that reads one of
`CODED_KEYS` by `.get("key")` or `["key"]`. A line is fine when it ENDS with the comment
`# canonical row` (the row was decoded first: the author says so where the read is). Limits: a read
split over two lines (`.get(` then the key) is not seen, and the allowlist counts lines per file;
the reader tests (`test_aw_coded_twin_readers.py`) cover the routed paths themselves. Every other
line counts against `ALLOWED`, a SHRINK-ONLY list: a file not in it may have none; a file in it
may not have more, and when it has fewer the number here must come down (so it cannot grow back
unseen). Each entry says why its reads are not twin reads, or what must happen before its next
run against a coded collection.
"""
from __future__ import annotations

import pathlib
import re
import pytest

pytestmark = pytest.mark.guard

APP = pathlib.Path(__file__).resolve().parents[1]
PKG = APP / "cytir"

READ_MARKERS = re.compile(
    r"points/scroll|/points\"|/points'|\.retrieve\(|\.scroll\(|fetch_payload_fields|fetch_rows\b"
    r"|retrieve_payloads\(|fetch_by_id\(|served_group_members|served_payloads_by_url"
    r"|served_digests_by_url")
CODED_KEYS = ("content_digest", "provenance", "prov_id", "consent_state", "consent_basis",
              "consent_source", "rule_epochs", "build_id", "chunk_rule")
_K = "(%s)" % "|".join(CODED_KEYS)
DIRECT_READ = re.compile(r"""\.get\(\s*["']%s["']|(?<=[\w\)\]])\[\s*["']%s["']\s*\]""" % (_K, _K))
MARKER = "canonical row"
#: The marker counts only as a TRAILING comment of the line that reads the key.
MARKED = re.compile(r"#[^\n]*canonical row\s*$")

#: path (relative to the app) -> (unmarked direct reads, why). SHRINK ONLY.
ALLOWED: dict[str, tuple[int, str]] = {
    "cytir/corpus/store.py": (2, "one-shot `backfill` of the corpus store from a collection scroll: "
                              "route through row_decode before its next run against a coded "
                              "collection (the store's digest is the 64-hex one; decide which)"),
    "cytir/distribute/license_check.py": (1, "dump tooling: `provenance` is a report label only "
                                          "(an int reads as a label); route before the next dump"),
    "cytir/index/apply.py": (1, "the pre-queue news apply path, disabled since 2026-09-23 (task 640): "
                             "reads its own pack artifact, not stored rows; it writes RAW payloads, "
                             "so its writer must project before any reuse"),
    "cytir/index/apply_delta.py": (1, "reads content_digest from a raw dump_netnew row (not decoded); a "
                                   "coded source fails loudly, not silently: cytir/corpus/persisted.py "
                                   "counts the digest mismatch and returns FAIL"),
    "cytir/index/promote_sealed_pack.py": (1, "the pack's own payloads (ADR-105 persisted check), "
                                           "not stored rows"),
    "cytir/sources/family_stage.py": (1, "a family CONTRACT's `consent_basis`, not a stored row"),
    "cytir/ops/twin_patch_fields.py": (3, "task 845 (T17): a twin WRITER for the RAW twin before the "
                                       "aw_e10 T0; task 864: it REFUSES a collection declared with a "
                                       "coded profile (row_decode.profile_of), dry or not. Earlier note: "
                                       "it was a twin WRITER for the RAW twin before the "
                                       "aw_e10 T0; it writes raw `rule_epochs`/`stamps`. Before any "
                                       "run on a coded twin: project its writes and compare canonical "
                                       "rows (expect/readback), or refuse a coded profile"),
}


def scan() -> dict[str, list[tuple[int, str]]]:
    out: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(PKG.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not READ_MARKERS.search(text):
            continue
        hits = [(n, line.strip()) for n, line in enumerate(text.splitlines(), 1)
                if DIRECT_READ.search(line) and not MARKED.search(line)
                and not line.lstrip().startswith("#")]
        if hits:
            out[str(path.relative_to(APP))] = hits
    return out


def test_no_new_direct_coded_key_read_in_a_stored_row_reader():
    found = scan()
    new = {f: h for f, h in found.items() if f not in ALLOWED}
    assert not new, (
        "direct read of a coded key in a module that reads stored rows -- decode the row first "
        "(cytir.index.row_decode) and mark the line `# canonical row`:\n"
        + "\n".join(f"  {f}:{n}: {l}" for f, h in new.items() for n, l in h))
    grew = {f: (len(h), ALLOWED[f][0]) for f, h in found.items() if f in ALLOWED and len(h) > ALLOWED[f][0]}
    assert not grew, ("more unmarked coded-key reads than allowed (file: now, allowed): %s\n%s" % (
        grew, "\n".join(f"  {f}:{n}: {l}" for f in grew for n, l in found[f])))


def test_the_allowlist_only_shrinks():
    found = scan()
    stale = {f: (len(found.get(f, [])), n) for f, (n, _why) in ALLOWED.items() if len(found.get(f, [])) < n}
    assert not stale, ("reads were routed or removed: lower ALLOWED to the new count "
                       "(file: now, allowed): %s" % stale)


def test_the_scan_sees_a_raw_read_and_accepts_a_marked_one():
    assert DIRECT_READ.search('d = stored.get("content_digest")')
    assert DIRECT_READ.search("p = row['provenance']")
    assert not DIRECT_READ.search('body = {"with_payload": ["rule_epochs"]}')
    assert MARKED.search('d = stored.get("content_digest")  # canonical row')
    assert not MARKED.search('d = stored.get("content_digest")  # canonical row? no: raw')
    assert not MARKED.search('"""a docstring about the canonical row"""; x = r.get("provenance")')
    # the routed readers are still recognised as readers and every read in them is marked
    found = scan()
    for f in ("cytir/index/drain/content_checks.py", "cytir/consent/apply.py", "cytir/ops/german_parts_tripwire.py",
              "cytir/news/compose.py", "cytir/index/url_group_supersede.py"):
        text = (APP / f).read_text()
        assert READ_MARKERS.search(text) and MARKER in text and f not in found, f
