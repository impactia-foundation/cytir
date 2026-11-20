#!/usr/bin/env python3
"""[test] ADR-105 / task 581 -- persisted inputs are a precondition to load.

The record this suite is built around is real and measured, not invented:
``developers.openai.com/api/docs/guides/function-calling`` is held by
production as 41 rows written by the 48-host docs campaign applied LIVE on
2026-07-30 (provenance ``developers_openai_com_docs``, ``build_id aw_b1``,
``last_seen 2026-07-30``).  No L0 and no L1 was ever sealed for it; the L2
corpus holds it at 40 passages / 6,731 words, and the candidate holds ONE row
of 733 words from a different capture.  It is the instance; the class is task
579's 492,211 pages with no stored text at all.

Every fixture here is SEALED in this file or written into a tmp store tree --
nothing is fetched, nothing reads the build box, and the build-script test directory's
no-network fixture would fail the suite if anything tried.
"""
from __future__ import annotations

import sys as _sys_1822
from pathlib import Path as _Path_1822

_sys_1822.path.insert(0, str(_Path_1822(__file__).resolve().parents[1] / "tests"))  # task 1822: the surface-delta fixtures live in tests/

import gzip
import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cytir.corpus.persisted as PI  # noqa: E402
import cytir.corpus.persisted_build as BUILD  # noqa: E402  (the ONLY way to
# populate a sqlite index from scratch since task 664's 2026-09-27 fix --
# update_index_sqlite() refuses fast on an empty db)
import cytir.core.common as AC  # noqa: E402

# -- the real record and the sealed store tree (one home: tests/persisted_store_tree.py) ----------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from persisted_store_tree import (  # noqa: E402,F401  (store_tree is a re-exported fixture)
    REAL_URL, REAL_PROD_ROWS, REAL_L2_WORDS, PERSISTED_URL, PERSISTED_SHA, store_tree, _write_parquet,
)


@pytest.fixture()
def index(store_tree, tmp_path):
    out = tmp_path / "idx"
    PI.build_index(str(out), root=str(store_tree), log=lambda *a, **k: None)
    idx = PI.PersistedIndex.open(str(out), verify_sha=True)
    yield idx
    idx.close()


# -- 1. the resolver -------------------------------------------------------

def test_resolver_returns_the_store_file_that_holds_the_page(index):
    hit = PI.page_text_persisted(PERSISTED_URL, index=index)
    assert hit is not None
    assert hit.layer == "l1"
    assert hit.store_id == "l1-release-gz"
    assert hit.store_path.endswith("l1-markdown-00000.jsonl.gz")
    assert os.path.exists(hit.store_path)


def test_resolver_confirms_or_denies_the_capture_sha(index):
    assert PI.page_text_persisted(PERSISTED_URL, PERSISTED_SHA,
                                  index=index).sha_matched is True
    assert PI.page_text_persisted(PERSISTED_URL, "not-the-capture",
                                  index=index).sha_matched is False
    # no sha supplied -> the resolver says "not asked", never "matched"
    assert PI.page_text_persisted(PERSISTED_URL, index=index).sha_matched is None


def test_resolver_is_the_one_canonicalizer(index):
    """A `www.`/http variant is the SAME page. Mirror contract: aw_common."""
    variant = "http://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006219126"
    assert AC.canonical_url(variant) == AC.canonical_url(PERSISTED_URL)
    assert PI.page_text_persisted(variant, index=index) is not None
    assert PI.page_key(variant) == PI.page_key(PERSISTED_URL)


def test_unknown_page_resolves_to_nothing(index):
    assert PI.page_text_persisted("https://example.invalid/nope",
                                  index=index) is None


# -- 2. the refusal, on the real record ------------------------------------

def test_the_real_record_is_refused(index):
    """41 prod rows, 6,731 L2 words, no L1: not loadable under ADR-105."""
    rows = [{"url": REAL_URL, "payload": {"url": REAL_URL}}
            for _ in range(REAL_PROD_ROWS)]
    with pytest.raises(PI.UnpersistedInputs) as exc:
        PI.assert_pages_persisted(rows, index=index, label="openai-docs")
    msg = str(exc.value)
    assert "REFUSED" in msg and "openai-docs" in msg
    assert REAL_URL in msg                      # the refusal NAMES the page
    assert "ADR-093" in msg                     # and names the route it has
    v = exc.value.verdict
    assert v["verdict"] == "FAIL"
    assert v["pages"] == 1 and v["rows"] == REAL_PROD_ROWS
    assert v["l1b_reconstructible"] == 1 and v["l1b_not_accepted"] == 1
    assert v["missing"] == 0                    # it is not GONE; it is L2-only


def test_a_page_with_no_stored_text_at_all_is_refused_as_missing(index):
    with pytest.raises(PI.UnpersistedInputs) as exc:
        PI.assert_pages_persisted([{"url": "https://example.invalid/gone"}],
                                  index=index, label="gone")
    v = exc.value.verdict
    assert v["missing"] == 1 and v["l1b_reconstructible"] == 0
    assert "no persisted L1 record" in str(exc.value)


def test_the_adr093_route_is_a_declared_level_not_a_bypass(index):
    v = PI.assert_pages_persisted([{"url": REAL_URL}], index=index,
                                  require="l1b", label="chained")
    assert v["verdict"] == "PASS"
    # the verdict still SAYS the page came in on the weaker route
    assert v["l1b_reconstructible"] == 1 and v["persisted"] == 0
    assert v["require"] == "l1b"


# -- 3. a row with a persisted page passes, and changes nothing ------------

def test_a_persisted_page_passes_and_the_guard_is_a_no_op(index, store_tree):
    before = sorted(p.name for p in pathlib.Path(store_tree).rglob("*"))
    v = PI.assert_pages_persisted(
        [{"payload": {"url": PERSISTED_URL, "content_digest": "x"}}],
        index=index, label="legi")
    assert v["verdict"] == "PASS"
    assert v["persisted"] == 1 and v["missing"] == 0
    # A passing guard writes nothing and touches nothing.
    assert sorted(p.name for p in pathlib.Path(store_tree).rglob("*")) == before


def test_rows_collapse_to_pages_but_rows_are_still_counted(index):
    v = PI.audit_pages([{"url": PERSISTED_URL}] * 7, index=index)
    assert v["rows"] == 7 and v["pages"] == 1 and v["persisted"] == 1


def test_terminals_conserve(index):
    pages = ([{"url": PERSISTED_URL}] + [{"url": REAL_URL}]
             + [{"url": "https://example.invalid/a"}]
             + [{"url": "https://example.invalid/b"}])
    v = PI.audit_pages(pages, index=index, require="l1b")
    assert (v["persisted"] + v["l1b_reconstructible"] + v["missing"]
            == v["pages"] == 4)


# -- 3b. lazy index load (task 598) -----------------------------------------
# The shared corpus-wide index can be absent, or mid-rebuild-and-therefore-
# inconsistent (persisted_inputs.py's own build-index writes each layer's
# .bin files into --out-dir as it finishes and rewrites INDEX.json only at
# the very end -- there is a real window where the two disagree). A caller
# whose overlay covers every page it is checking must not be refused, or even
# probed, by that shared index at all.

def test_a_fully_overlay_covered_audit_never_touches_the_shared_index(
        monkeypatch):
    def boom():
        raise AssertionError("shared_index() must not be called")
    monkeypatch.setattr(PI, "shared_index", boom)
    hit = PI.Hit(url=PERSISTED_URL, layer="l1", store_id="fixture",
                store_path="/dev/null", sha_matched=None)
    v = PI.audit_pages([{"url": PERSISTED_URL}],
                       overlay={AC.canonical_url(PERSISTED_URL): hit})
    assert v["verdict"] == "PASS"
    assert v["persisted"] == 1 and v["persisted_in_this_run"] == 1
    assert v["index_not_consulted"] is True
    assert v["index_dir"] is None and v["index_built_at"] is None


def test_a_partially_covered_audit_still_consults_the_index_for_the_rest(
        index):
    hit = PI.Hit(url=REAL_URL, layer="l1", store_id="fixture",
                store_path="/dev/null", sha_matched=None)
    v = PI.audit_pages(
        [{"url": PERSISTED_URL}, {"url": REAL_URL}],
        index=index, overlay={AC.canonical_url(REAL_URL): hit})
    # PERSISTED_URL comes from the real index (l1); REAL_URL comes from the
    # overlay and is reported as l1b-if-checked-for-real, but here it is an
    # overlay hit routed the same as any l1 hit (layer != "l2").
    assert v["persisted"] == 2
    assert v["persisted_in_this_run"] == 1
    assert v["index_not_consulted"] is False
    assert v["index_dir"] == index.directory


def test_no_overlay_still_fails_closed_exactly_as_before(monkeypatch):
    """overlay=None (every existing caller) must still eagerly need a
    working index on the very first page -- this lazy load changes WHEN
    shared_index() is called, never WHETHER a caller with no overlay can
    skip it."""
    def boom():
        raise PI.PersistedInputsError("no index for this test")
    monkeypatch.setattr(PI, "shared_index", boom)
    with pytest.raises(PI.PersistedInputsError, match="no index for this test"):
        PI.audit_pages([{"url": PERSISTED_URL}])


# -- 4. fail closed --------------------------------------------------------

def test_a_missing_index_is_a_refusal_not_a_skip(tmp_path):
    with pytest.raises(PI.PersistedInputsError) as exc:
        PI.PersistedIndex.open(str(tmp_path / "nope"))
    assert "INDEX.json missing" in str(exc.value)


def test_a_truncated_index_is_a_refusal(index, tmp_path):
    d = pathlib.Path(index.directory)
    index.close()
    target = d / "l1.pages.bin"
    target.write_bytes(target.read_bytes()[:-1])
    with pytest.raises(PI.PersistedInputsError) as exc:
        PI.PersistedIndex.open(str(d))
    assert "truncated" in str(exc.value)


def test_a_corrupt_array_is_a_refusal_under_sha_verification(index):
    d = pathlib.Path(index.directory)
    index.close()
    target = d / "l1.pages.bin"
    raw = bytearray(target.read_bytes())
    raw[0] ^= 0xFF                       # same length, different content
    target.write_bytes(bytes(raw))
    PI.PersistedIndex.open(str(d))       # length alone still passes
    with pytest.raises(PI.PersistedInputsError) as exc:
        PI.PersistedIndex.open(str(d), verify_sha=True)
    assert "sha256 mismatch" in str(exc.value)


def test_a_wrong_schema_manifest_is_a_refusal(index):
    d = pathlib.Path(index.directory)
    index.close()
    man = json.loads((d / "INDEX.json").read_text())
    man["schema"] = "aw.something-else.v9"
    (d / "INDEX.json").write_text(json.dumps(man))
    with pytest.raises(PI.PersistedInputsError):
        PI.PersistedIndex.open(str(d))


def test_zero_pages_is_a_refusal_not_a_pass(index):
    with pytest.raises(PI.UnpersistedInputs) as exc:
        PI.assert_pages_persisted([], index=index, label="empty pack")
    assert "zero pages" in str(exc.value)


def test_a_missing_registry_is_a_refusal(tmp_path):
    with pytest.raises(PI.PersistedInputsError) as exc:
        PI.load_registry(str(tmp_path / "absent.json"))
    assert "missing" in str(exc.value)


def test_an_unreadable_store_file_is_recorded_and_travels_with_the_verdict(
        store_tree, tmp_path):
    """verification-doctrine 6: an error is not an empty.

    ADR-093 section 6 already found three `l0/wave=*` shards gzip-corrupt at
    seal. The index must not treat one as a store with no pages in it, and a
    refusal built on a holed index must say so -- otherwise "no persisted
    record" is being reported for a page the store may well hold.
    """
    bad = store_tree / "l1" / "release=test" / "source=legi" / \
        "l1-markdown-00001.jsonl.gz"
    bad.write_bytes(b"this is not gzip")
    man = PI.build_index(str(tmp_path / "idx2"), root=str(store_tree),
                         log=lambda *a, **k: None)
    assert man["unreadable_files"] == 1
    assert man["unreadable"][0]["path"].endswith("l1-markdown-00001.jsonl.gz")
    assert "BadGzipFile" in man["unreadable"][0]["error"]

    idx = PI.PersistedIndex.open(str(tmp_path / "idx2"))
    try:
        # the readable file's page is still indexed -- rows read are kept
        assert PI.page_text_persisted(PERSISTED_URL, index=idx) is not None
        with pytest.raises(PI.UnpersistedInputs) as exc:
            PI.assert_pages_persisted([{"url": "https://example.invalid/z"}],
                                      index=idx, label="holed")
        assert exc.value.verdict["index_unreadable_files"] == 1
        assert "unreadable" in str(exc.value)
        assert "not proof the page was never" in str(exc.value)
    finally:
        idx.close()


# -- 5. no in-band waiver --------------------------------------------------

def test_there_is_no_level_that_admits_an_unpersisted_page():
    """verification-doctrine 12: the standard is not softenable from inside."""
    assert set(PI.REQUIRE_LEVELS) == {"l1", "l1b"}
    for name, layers in PI.REQUIRE_LEVELS.items():
        assert layers, f"{name} accepts nothing"
        assert set(layers) <= set(PI.LAYERS)
    # every level requires at least one layer that actually holds text
    assert all("l1" in v or "l0" in v or "l2" in v
               for v in PI.REQUIRE_LEVELS.values())


def test_an_undeclared_level_is_an_error_not_a_default(index):
    with pytest.raises(PI.PersistedInputsError):
        PI.audit_pages([{"url": PERSISTED_URL}], index=index, require="off")
    with pytest.raises(PI.PersistedInputsError):
        PI.audit_pages([{"url": PERSISTED_URL}], index=index, require="")


def test_no_environment_variable_can_relax_the_bar(index, monkeypatch):
    for var in ("AW_PERSISTED_INPUTS", "AW_SKIP_PERSISTED", "AW_ALLOW_UNPERSISTED",
                "AW_PERSISTED_INPUTS_MODE"):
        monkeypatch.setenv(var, "report")
    with pytest.raises(PI.UnpersistedInputs):
        PI.assert_pages_persisted([{"url": "https://example.invalid/x"}],
                                  index=index, label="env")


def test_sha_checking_is_off_by_default_so_a_field_mismatch_is_not_a_verdict(
        index):
    """corpus-doctrine 3: a row's content_digest is not the store's sha."""
    v = PI.audit_pages([{"url": PERSISTED_URL, "content_digest": "unrelated"}],
                       index=index)
    assert v["sha_checked"] is False and v["verdict"] == "PASS"
    v2 = PI.audit_pages([{"url": PERSISTED_URL, "sha": "unrelated"}],
                        index=index, check_sha=True)
    assert v2["sha_checked"] is True and v2["verdict"] == "FAIL"
    assert v2["sha_mismatch"] == 1


# -- 6. the CLI carries the verdict in an exit code ------------------------

def test_cli_check_exits_nonzero_on_a_refused_page(index, tmp_path, capsys):
    pages = tmp_path / "pages.txt"
    pages.write_text(REAL_URL + "\n")
    rc = PI.main(["check", "--pages", str(pages), "--index-dir",
                  index.directory])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL"


def test_cli_check_exits_zero_on_a_persisted_page(index, tmp_path, capsys):
    pages = tmp_path / "pages.txt"
    pages.write_text(PERSISTED_URL + "\n")
    assert PI.main(["check", "--pages", str(pages), "--index-dir",
                    index.directory]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"


def test_cli_refuses_a_missing_index_with_exit_2(tmp_path, capsys):
    pages = tmp_path / "p.txt"
    pages.write_text(PERSISTED_URL + "\n")
    assert PI.main(["check", "--pages", str(pages), "--index-dir",
                    str(tmp_path / "absent")]) == 2


# -- 7. the registry is data, and it is globs -------------------------------

def test_the_tracked_registry_loads_and_declares_every_layer():
    doc = PI.load_registry()
    st = PI.stores(doc)
    assert {s.layer for s in st} == set(PI.LAYERS)
    for s in st:
        assert "*" in s.glob, f"{s.store_id} names files, not a glob"
        assert s.url_fields, f"{s.store_id} declares no url field"
        assert s.note, f"{s.store_id} carries no note saying what it is"


def test_every_registry_glob_is_a_pattern_in_its_own_file_component():
    """corpus-doctrine 8: the roster comes from the tree, not from a list.

    A store whose glob names one concrete file reports on one file and calls
    it the store -- which is how a first draft of this registry found EIGHT
    L1 files in a tree that holds thousands.
    """
    for s in PI.stores():
        assert "*" in os.path.basename(s.glob), \
            f"{s.store_id} names a file, not a pattern: {s.glob}"


def test_a_symlinked_leaf_file_is_refused_not_indexed(store_tree, tmp_path):
    """Task 641 (2026-09-23): `register_l1_release.py`'s first cut symlinked
    scratch-dir files (task634/, depth226/staged/) straight into l1/ --
    exactly the dependency-on-a-cleanable-scratch-dir ADR-105 exists to rule
    out. A store's leaf file must be a real, persisted copy; a symlinked
    RELEASE DIRECTORY stays fine (that's what followlinks=True is for --
    the news-daily incident this guard sits next to in the source)."""
    real_source = tmp_path / "outside-the-registry" / "scratch.jsonl"
    real_source.parent.mkdir(parents=True)
    real_source.write_text(json.dumps({
        "canonical_url": PERSISTED_URL, "url": PERSISTED_URL,
        "markdown": "# linked", "markdown_sha256": PERSISTED_SHA}) + "\n",
        encoding="utf-8")

    linked = store_tree / "l1" / "release=test" / "source=legi" / "linked-leaf.jsonl"
    linked.symlink_to(real_source)

    with pytest.raises(PI.PersistedInputsError, match="is a symlink"):
        PI.build_index(str(tmp_path / "idx2"), root=str(store_tree),
                        log=lambda *a, **k: None)


# -- 8. the guard is wired into the paths it guards -------------------------
# verification-doctrine 12: "a guard with a test and no caller" is decoration.
# These run an applier's own `main()` against its own fake, so what is asserted
# is the WIRING, not the module.

def test_apply_pack_delta_refuses_a_pack_whose_page_is_not_persisted(
        tmp_path, monkeypatch):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
    import test_aw_index_pack_delta as T576
    import cytir.index.surface_delta as T546_mod  # noqa: F401  (import order)
    import test_aw_index_surface_delta as T546
    import cytir.index.pack_delta as APD

    tgt, _pack = T576.build(tmp_path, monkeypatch)
    # `build` installs a store holding the fixture's page. Replace it with a
    # valid index that holds a DIFFERENT page, so the refusal comes from this
    # pack's page being absent and not from the guard being unable to run.
    T546.install_persisted_index(
        monkeypatch, tmp_path / "other",
        [{"url": PERSISTED_URL, "point_id": "x", "text": "body"}])
    PI.reset_index_cache()

    monkeypatch.setattr(sys, "argv", ["apply_pack_delta.py"] + T576.BASE)
    with pytest.raises(SystemExit) as exc:
        APD.main()
    assert "REFUSED" in str(exc.value)
    assert "https://example.org/a" in str(exc.value)
    assert tgt.upserted == []          # refused before a single write
    assert tgt.deleted == []


def test_apply_pack_delta_proceeds_when_the_page_is_persisted(
        tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
    import test_aw_index_pack_delta as T576
    import cytir.index.pack_delta as APD

    tgt, _pack = T576.build(tmp_path, monkeypatch)     # installs the store
    PI.reset_index_cache()
    # task 820: the card guard has its own tests (tests/test_card_guard_pipeline_writers.py)
    monkeypatch.setattr(APD, "guard_card_overwrites", lambda *a, **k: {})
    monkeypatch.setattr(APD, "client_url", lambda client: "http://fake")
    monkeypatch.setattr(sys, "argv", ["apply_pack_delta.py"] + T576.BASE)
    APD.main()
    assert "persisted-inputs OK: 1 page(s)" in capsys.readouterr().out
    # and the write really happened, unchanged: the pre-581 outcome -- 3 fresh
    # ids landed, 2 were overwritten, the stale one was superseded.
    assert set(tgt.points) == {"ov0", "ov1", "fr0", "fr1", "fr2"}


# -- 9. the two sort/dedup paths must agree -------------------------------

def test_numpy_and_stdlib_sort_dedup_agree(monkeypatch):
    """The index build uses numpy on the build box and the stdlib on a clean
    clone. One of them is what a stranger's rebuild runs, so they must produce
    the SAME bytes -- otherwise the guard's answer depends on which machine
    built its index (engineering-doctrine 6)."""
    import random
    import builtins
    rng = random.Random(581)
    chunks = []
    for store_ord in range(3):
        blob = bytearray()
        for file_ord in range(4):
            for _ in range(200):
                blob += PI._REC.pack(rng.getrandbits(64) % 500, store_ord,
                                     file_ord)
        chunks.append(bytes(blob))

    fast = PI._sort_dedup_pages(chunks)

    real_import = builtins.__import__

    def no_numpy(name, *a, **k):
        if name == "numpy":
            raise ImportError("numpy disabled for this test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    slow = PI._sort_dedup_pages(chunks)
    assert fast == slow
    # and the dedup really kept the FIRST occurrence: every surviving key
    # appears once, and its (store, file) is the lowest that carried it.
    seen = {}
    for c in chunks:
        for i in range(0, len(c), PI.RECORD_BYTES):
            k, s, f = PI._REC.unpack(c[i:i + PI.RECORD_BYTES])
            seen.setdefault(k, (s, f))
    got = {}
    for i in range(0, len(fast), PI.RECORD_BYTES):
        k, s, f = PI._REC.unpack(fast[i:i + PI.RECORD_BYTES])
        assert k not in got
        got[k] = (s, f)
    assert got == seen


def test_numpy_and_stdlib_sha_dedup_agree(monkeypatch):
    import random
    import builtins
    rng = random.Random(5812)
    chunks = [bytes(bytearray().join(
        PI._SHA_REC.pack(rng.getrandbits(64) % 1000) for _ in range(300)))
        for _ in range(3)]
    fast = PI._sort_dedup_shas(chunks)
    real_import = builtins.__import__

    def no_numpy(name, *a, **k):
        if name == "numpy":
            raise ImportError("numpy disabled for this test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    assert PI._sort_dedup_shas(chunks) == fast


# -- 10. the roster reaches hidden directories -----------------------------

def test_the_store_roster_descends_into_dot_directories(tmp_path):
    """The bug this test exists for, measured on the real corpus.

    `glob.glob` silently refuses to descend into or match any path component
    beginning with a dot. 156,522 of the L1 layer's 156,653 uncompressed
    documents live under `wikipedia-markdown-v3/lang=*/.multistream-fragments/
    range-*/`, so a glob-based roster found 131 of them -- and the guard built
    on that roster would have called every Wikipedia page in production
    un-persisted.
    """
    root = tmp_path / "corpus"
    hidden = root / "l1" / "release=r" / "source=wiki" / "lang=de" / \
        ".multistream-fragments" / "range-000001"
    hidden.mkdir(parents=True)
    (hidden / "shard-00.jsonl").write_text(
        json.dumps({"url": "https://de.wikipedia.org/wiki/Bern",
                    "markdown": "# Bern"}) + "\n", encoding="utf-8")

    store = next(s for s in PI.stores() if s.store_id == "l1-release-plain")
    found = store.files(str(root))
    assert found == [str(hidden / "shard-00.jsonl")], found


def test_a_single_star_does_not_cross_a_directory_separator(tmp_path):
    """`fnmatch`'s `*` matches `/`; a store glob's must not, or
    `l0/wave=*/shard-*.jsonl.gz` admits files nested arbitrarily deeper than
    the store reaches -- and an over-match declares a page persisted that is
    not."""
    root = tmp_path / "corpus"
    shallow = root / "l0" / "wave=w"
    deep = root / "l0" / "wave=w" / "nested"
    shallow.mkdir(parents=True)
    deep.mkdir(parents=True)
    for d in (shallow, deep):
        with gzip.open(d / "shard-0000.jsonl.gz", "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({"url": "https://x.test/a",
                                 "body_b64": "eA=="}) + "\n")

    store = next(s for s in PI.stores() if s.store_id == "l0-waves")
    assert store.files(str(root)) == [str(shallow / "shard-0000.jsonl.gz")]


@pytest.mark.parametrize("pattern,path,expected", [
    ("a/**/*.jsonl", "a/b/c/x.jsonl", True),
    ("a/**/*.jsonl", "a/x.jsonl", True),
    ("a/**/*.jsonl", "a/b/.hidden/x.jsonl", True),
    ("a/*/x.jsonl", "a/b/c/x.jsonl", False),
    ("a/*/x.jsonl", "a/b/x.jsonl", True),
    ("a/wave=*/s-*.gz", "a/wave=1/s-2.gz", True),
    ("a/wave=*/s-*.gz", "a/wave=1/deep/s-2.gz", False),
])
def test_glob_regex_semantics(pattern, path, expected):
    assert bool(PI._glob_regex(pattern).match(path)) is expected


# -- 11. address spellings ------------------------------------------------
# The real record: the dump-derived Wikipedia waves write a title's spaces as
# `%20`; production writes underscores. The first run of the 581 census read
# that as 6,215,311 en.wikipedia pages with no stored text.

WIKI_PROD = "https://en.wikipedia.org/wiki/Imperial_War_Museum_Duxford"
WIKI_STORE = "https://en.wikipedia.org/wiki/Imperial%20War%20Museum%20Duxford"


@pytest.fixture()
def wiki_index(tmp_path):
    """A store that spells the page the way the dump-derived wave does."""
    root = tmp_path / "corpus"
    d = root / "wave=l1_wiki_en_20260601"
    d.mkdir(parents=True)
    _write_parquet(d / "part-0000.parquet", [WIKI_STORE])
    out = tmp_path / "idx"
    PI.build_index(str(out), root=str(root), log=lambda *a, **k: None)
    idx = PI.PersistedIndex.open(str(out))
    yield idx
    idx.close()


def test_a_wikipedia_page_is_found_under_either_spelling(wiki_index):
    assert PI.page_text_persisted(WIKI_STORE, index=wiki_index) is not None
    hit = PI.page_text_persisted(WIKI_PROD, index=wiki_index)
    assert hit is not None and hit.layer == "l2"
    # and the canonicalizer did NOT move: the two are different canonical urls
    assert AC.canonical_url(WIKI_PROD) != AC.canonical_url(WIKI_STORE)


def test_the_spelling_rule_is_scoped_to_the_host_it_was_measured_on():
    """Elsewhere `_` and `%20` are different resources; folding them would
    manufacture a false PASS, which is the unsafe direction for this guard."""
    assert PI._wiki_title_spellings(
        "https://example.org/a_b") == ()
    assert PI._wiki_title_spellings(
        "https://example.org/a%20b") == ()
    assert PI.page_keys("https://example.org/a_b") == (
        PI.page_key("https://example.org/a_b"),)
    assert len(PI.page_keys(WIKI_PROD)) > 1


def test_a_single_word_title_matches_either_way_which_is_why_a_sample_missed_it():
    """`.../wiki/Rychvald` has no space, so it joined under both spellings and
    a head sample showed nothing wrong (corpus-doctrine 5)."""
    one = "https://en.wikipedia.org/wiki/Rychvald"
    assert PI.page_keys(one)[0] == PI.page_key(one)


def test_the_canonical_key_is_always_tried_first(wiki_index):
    assert PI.page_keys(WIKI_PROD)[0] == PI.page_key(WIKI_PROD)


# -- 12. incremental scan cache (task 641/662, 2026-09-25) ------------------
# The auto-refresh (`promote_queue.refresh_persisted_index`) used to rebuild
# the WHOLE index -- 157,032 L1 files, 41M+ rows -- on any new L1 file, 35-120
# minutes per drain. `build_index(incremental=True)` (the default) caches
# each store file's own scanned bytes, keyed by (path, size, mtime_ns), and
# only re-scans a file whose signature changed. These tests prove: (a) an
# incremental rebuild produces the byte-identical INDEX.json a full rebuild
# would, (b) unchanged files are genuinely skipped (not just "produce the
# same answer slowly"), (c) a changed or new file IS re-scanned, and (d) a
# registry edit forces a real full rescan.

def _layer_digest(man: dict) -> dict:
    """The part of a build manifest that must be IDENTICAL between a full
    and an incremental rebuild -- everything content-derived, none of the
    build's own bookkeeping (timings, incremental/cache_* counters)."""
    return {
        layer: {k: v for k, v in entry.items()
               if k in ("pages_bytes", "pages_sha256", "pages", "shas",
                        "shas_sha256", "files", "rows_read", "store_ids")}
        for layer, entry in man["layers"].items()
    }


def test_incremental_rebuild_is_byte_identical_to_a_full_rebuild(
        store_tree, tmp_path):
    full = PI.build_index(str(tmp_path / "idx-full"), root=str(store_tree),
                          log=lambda *a, **k: None, incremental=False)
    inc = PI.build_index(str(tmp_path / "idx-inc"), root=str(store_tree),
                         log=lambda *a, **k: None, incremental=True)
    assert _layer_digest(full) == _layer_digest(inc)
    assert full["unreadable_files"] == inc["unreadable_files"] == 0


def test_a_second_incremental_build_skips_every_unchanged_file(
        store_tree, tmp_path):
    out = tmp_path / "idx"
    first = PI.build_index(str(out), root=str(store_tree),
                           log=lambda *a, **k: None, incremental=True)
    assert first["scanned_files"] > 0          # nothing cached yet -- real work
    assert first["cache_hit_files"] == 0

    second = PI.build_index(str(out), root=str(store_tree),
                            log=lambda *a, **k: None, incremental=True)
    assert second["scanned_files"] == 0, (
        "an unchanged file must be served from the cache, never re-scanned")
    assert second["cache_hit_files"] == first["scanned_files"]
    assert second["cache_reads_used"] is True
    assert _layer_digest(first) == _layer_digest(second)


def test_a_new_file_is_scanned_but_old_files_are_not(store_tree, tmp_path):
    out = tmp_path / "idx"
    first = PI.build_index(str(out), root=str(store_tree),
                           log=lambda *a, **k: None, incremental=True)

    new_url = "https://www.legifrance.gouv.fr/codes/article_lc/NEWARTICLE0001"
    with gzip.open(store_tree / "l1" / "release=test" / "source=legi" /
                   "l1-markdown-00001.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "schema": "aw.l1-markdown.v1", "canonical_url": new_url,
            "url": new_url, "markdown": "# New\n",
            "markdown_sha256": "f" * 64}) + "\n")

    second = PI.build_index(str(out), root=str(store_tree),
                            log=lambda *a, **k: None, incremental=True)
    assert second["scanned_files"] == 1, "only the new file should be scanned"
    assert second["cache_hit_files"] == first["scanned_files"]

    idx = PI.PersistedIndex.open(str(out))
    try:
        assert PI.page_text_persisted(new_url, index=idx) is not None
        assert PI.page_text_persisted(PERSISTED_URL, index=idx) is not None
    finally:
        idx.close()


def test_a_changed_file_is_rescanned_not_served_stale(store_tree, tmp_path):
    out = tmp_path / "idx"
    PI.build_index(str(out), root=str(store_tree), log=lambda *a, **k: None,
                   incremental=True)

    target = (store_tree / "l1" / "release=test" / "source=legi" /
             "l1-markdown-00000.jsonl.gz")
    changed_url = "https://www.legifrance.gouv.fr/codes/article_lc/CHANGED0001"
    with gzip.open(target, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "schema": "aw.l1-markdown.v1", "canonical_url": changed_url,
            "url": changed_url, "markdown": "# Changed\n",
            "markdown_sha256": "e" * 64}) + "\n")
    # force a distinct mtime -- some filesystems have 1s mtime resolution and
    # this test's writes can land in the same tick as the original build
    new_mtime = os.stat(target).st_mtime + 5
    os.utime(target, (new_mtime, new_mtime))

    second = PI.build_index(str(out), root=str(store_tree),
                            log=lambda *a, **k: None, incremental=True)
    assert second["scanned_files"] == 1, "a changed file must be re-scanned"

    idx = PI.PersistedIndex.open(str(out))
    try:
        assert PI.page_text_persisted(changed_url, index=idx) is not None
        # the OLD content's url is gone -- this file no longer holds it
        assert PI.page_text_persisted(PERSISTED_URL, index=idx) is None
    finally:
        idx.close()


def test_full_equals_incremental_false_ignores_a_warm_cache(store_tree, tmp_path):
    out = tmp_path / "idx"
    PI.build_index(str(out), root=str(store_tree), log=lambda *a, **k: None,
                   incremental=True)
    forced_full = PI.build_index(str(out), root=str(store_tree),
                                 log=lambda *a, **k: None, incremental=False)
    assert forced_full["cache_reads_used"] is False
    assert forced_full["scanned_files"] > 0, (
        "incremental=False must ignore a warm cache and rescan everything")


def test_a_registry_edit_forces_a_full_rescan_even_with_a_warm_cache(
        store_tree, tmp_path):
    reg = json.loads(pathlib.Path(PI.DEFAULT_REGISTRY).read_text())
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(reg))

    out = tmp_path / "idx"
    first = PI.build_index(str(out), root=str(store_tree),
                           registry_path=str(reg_path),
                           log=lambda *a, **k: None, incremental=True)
    assert first["scanned_files"] > 0

    second = PI.build_index(str(out), root=str(store_tree),
                            registry_path=str(reg_path),
                            log=lambda *a, **k: None, incremental=True)
    assert second["scanned_files"] == 0          # warm cache, registry unchanged

    # Edit the registry (e.g. a note field -- content changed, not just touched)
    reg["note_edited_by_test"] = "task662"
    reg_path.write_text(json.dumps(reg))

    third = PI.build_index(str(out), root=str(store_tree),
                           registry_path=str(reg_path),
                           log=lambda *a, **k: None, incremental=True)
    assert third["cache_reads_used"] is False
    assert third["scanned_files"] == first["scanned_files"], (
        "a registry edit must force a full rescan even though no store "
        "file's own bytes changed")

    # ...and the cache is warm again for the NEXT run.
    fourth = PI.build_index(str(out), root=str(store_tree),
                            registry_path=str(reg_path),
                            log=lambda *a, **k: None, incremental=True)
    assert fourth["cache_reads_used"] is True
    assert fourth["scanned_files"] == 0


def test_scan_cache_lives_inside_the_index_dir(store_tree, tmp_path):
    """The cache travels WITH the index directory -- no separate location
    to lose track of, or to leave behind when an index dir is archived."""
    out = tmp_path / "idx"
    PI.build_index(str(out), root=str(store_tree), log=lambda *a, **k: None,
                   incremental=True)
    cache_dir = out / PI.CACHE_DIRNAME
    assert cache_dir.is_dir()
    assert (cache_dir / PI._REGISTRY_MARKER).exists()
    assert any(cache_dir.rglob("*.meta.json"))


# -- 13. anchored walk, never a decoy subtree (task 663, 2026-09-25) --------
# The real incident: `l2-legacy-corpus`'s glob is `wave=*/*.parquet` -- a
# WILDCARD as the very first segment after corpus_root. The old "strip
# trailing glob segments to find a literal head" approach can only anchor
# past corpus_root when every wildcard sits at the END of the pattern, so
# for this store it degenerated to a full os.walk(corpus_root) -- and
# py-spy caught a live drain 3h21m deep inside a decoy scratch subtree
# (`_preservation/scratch-landing-v1/.../legi/legi/global/code_et...`)
# millions of directories deep, that no store glob names at all.

@pytest.fixture()
def decoy_store_tree(tmp_path):
    """A real `wave=*` directory the `l2-legacy-corpus` store glob
    (`wave=*/*.parquet`) must find, sitting next to a large DECOY subtree
    (`_preservation/...`) that does not match `wave=*` at the corpus root
    and must never be entered at all -- not filtered out after a scan,
    NEVER SCANNED."""
    root = tmp_path / "corpus"
    real = root / "wave=cc_phase2"
    real.mkdir(parents=True)
    _write_parquet(real / "part-0000.parquet", [REAL_URL])

    # A modest decoy tree stands in for "millions of directories" -- the
    # point is that entering it AT ALL is the bug, so even a handful of
    # untouched decoy dirs proves the fix; the real incident's scale
    # (millions) would just make an un-anchored walk slower, not different
    # in kind.
    decoy_root = root / "_preservation" / "scratch-landing-v1"
    for i in range(5):
        d = decoy_root / f"legi{i}" / "global" / "code_etc"
        d.mkdir(parents=True)
        (d / f"unrelated-{i}.parquet").write_bytes(b"not a real store file")
    return root


def test_wildcard_first_segment_finds_its_files(decoy_store_tree):
    """The actual `l2-legacy-corpus` shape: `wave=*/*.parquet`, wildcard as
    the FIRST segment. Must still find the real file."""
    store = PI.Store(store_id="l2-legacy-corpus", layer="l2", kind="parquet",
                    glob="wave=*/*.parquet", url_fields=("url",), sha_fields=())
    found = store.files(str(decoy_store_tree))
    assert len(found) == 1
    assert found[0].endswith("wave=cc_phase2/part-0000.parquet")


def test_wildcard_first_segment_never_enters_the_decoy_subtree(
        decoy_store_tree, monkeypatch):
    """Not "the decoy files are filtered out of the result" (the old code
    already did that) -- PROVE the decoy directories are never even
    listed. Wraps `os.scandir` and `os.walk` (the only two primitives
    `Store.files` uses to look inside a directory) and asserts no call
    target's real path is anywhere under `_preservation`."""
    decoy_prefix = str((decoy_store_tree / "_preservation").resolve())
    visited: list[str] = []

    real_scandir = os.scandir
    real_walk = os.walk

    def spy_scandir(path="."):
        visited.append(os.path.realpath(path))
        return real_scandir(path)

    def spy_walk(top, *a, **k):
        visited.append(os.path.realpath(top))
        return real_walk(top, *a, **k)

    monkeypatch.setattr(PI.os, "scandir", spy_scandir)
    monkeypatch.setattr(PI.os, "walk", spy_walk)

    store = PI.Store(store_id="l2-legacy-corpus", layer="l2", kind="parquet",
                    glob="wave=*/*.parquet", url_fields=("url",), sha_fields=())
    found = store.files(str(decoy_store_tree))

    assert len(found) == 1, "the fix must not cost the real file"
    entered_decoy = [v for v in visited if v.startswith(decoy_prefix)]
    assert entered_decoy == [], (
        f"the decoy subtree was entered: {entered_decoy!r} -- an anchored "
        "walk must never list a directory that cannot match its glob "
        "segment at that depth")
    assert visited, "the spy must have seen at least the real directories"


def test_literal_prefix_before_a_wildcard_is_still_anchored(tmp_path,
                                                            monkeypatch):
    """`l0/wave=*/shard-*.jsonl.gz` -- literal `l0/`, then a wildcard.
    A decoy sibling of `l0` (e.g. a huge unrelated tree next to it) must
    never be entered either -- this was already correctly anchored by the
    old code (the bug was ONLY the wildcard-first case), this test pins
    that it stays correct under the new one."""
    root = tmp_path / "corpus"
    real = root / "l0" / "wave=x"
    real.mkdir(parents=True)
    (real / "shard-00.jsonl.gz").write_bytes(b"")
    decoy = root / "_work" / "huge" / "unrelated"
    decoy.mkdir(parents=True)
    (decoy / "shard-00.jsonl.gz").write_bytes(b"")

    visited: list[str] = []
    real_scandir = os.scandir

    def spy_scandir(path="."):
        visited.append(os.path.realpath(path))
        return real_scandir(path)

    monkeypatch.setattr(PI.os, "scandir", spy_scandir)

    store = PI.Store(store_id="l0-waves", layer="l0", kind="jsonl_gz",
                    glob="l0/wave=*/shard-*.jsonl.gz", url_fields=("url",), sha_fields=())
    found = store.files(str(root))
    assert len(found) == 1
    assert found[0].endswith("l0/wave=x/shard-00.jsonl.gz")
    decoy_prefix = str((root / "_work").resolve())
    assert not any(v.startswith(decoy_prefix) for v in visited)


def test_recursive_glob_still_finds_nested_dot_directories(tmp_path):
    """`**` must keep working exactly as before: the Wikipedia
    `.multistream-fragments` blind spot (this file's `wiki_index` fixture)
    is the real regression risk of touching this method at all."""
    root = tmp_path / "corpus"
    d = root / "l1" / "release=r" / "source=wiki" / "lang=de" / \
        ".multistream-fragments" / "range-000001"
    d.mkdir(parents=True)
    (d / "shard-00.jsonl").write_bytes(b"")

    store = PI.Store(store_id="l1-release-plain", layer="l1", kind="jsonl",
                    glob="l1/**/*.jsonl", url_fields=("url",), sha_fields=())
    found = store.files(str(root))
    assert len(found) == 1
    assert ".multistream-fragments" in found[0]


def test_a_real_file_set_is_identical_before_and_after_the_anchor_fix(
        tmp_path):
    """The acceptance test named at dispatch: build a fixture that
    exercises every real registered store shape (literal-then-wildcard,
    wildcard-first, and `**`) plus decoys at every level, and assert the
    file set the anchored walk finds is exactly the set a brute-force
    (unanchored) reference walk finds -- not merely "some files", the
    SAME files."""
    root = tmp_path / "corpus"
    wanted = []
    for rel in [
        "l0/wave=1/shard-00.jsonl.gz",
        "l0/wave=2/shard-01.jsonl.gz",
        "l0_direct/wave=9/page-00.raw.jsonl.gz",
        "wave=cc/part-0000.parquet",
        "l2/wave=oa/passages-00.jsonl.gz",
        "l1/release=a/source=x/shard-00.jsonl.gz",
        "l1/release=b/source=y/lang=de/.hidden/shard-01.jsonl.gz",
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        wanted.append(str(p))
    for decoy in [
        "_preservation/scratch/legi/global/code/unrelated.parquet",
        "_work/huge/unrelated/shard-00.jsonl.gz",
        "l0/not-a-wave-dir/shard-00.jsonl.gz",   # wrong segment 1 shape
        "wave=cc/subdir/part-0001.parquet",       # ONE level too deep
    ]:
        p = root / decoy
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")

    def brute_force(glob_pattern: str) -> list[str]:
        pattern = os.path.join(str(root), glob_pattern)
        rx = PI._glob_regex(pattern)
        out = []
        for dirpath, _dirnames, filenames in os.walk(str(root)):
            for name in filenames:
                p = os.path.join(dirpath, name)
                if rx.match(p):
                    out.append(p)
        return sorted(out)

    for store_id, glob_pattern in [
        ("l0-waves", "l0/wave=*/shard-*.jsonl.gz"),
        ("l0-direct", "l0_direct/wave=*/*.raw.jsonl.gz"),
        ("l2-legacy-corpus", "wave=*/*.parquet"),
        ("l2-openalex", "l2/wave=*/passages-*.jsonl.gz"),
        ("l1-release-gz", "l1/**/*.jsonl.gz"),
    ]:
        store = PI.Store(store_id=store_id, layer="l0", kind="parquet",
                         glob=glob_pattern, url_fields=("url",), sha_fields=())
        assert store.files(str(root)) == brute_force(glob_pattern), store_id


# --------------------------------------------------------------------------
# task 664, 2026-09-26: sqlite-backed index (answer equivalence + the
# incremental-merge behaviour it replaces the full in-memory rebuild with).
#
# WHY THIS EXISTS: measured live, the #358 scan cache made per-FILE
# scanning incremental (157,057 cached, 14 freshly scanned), but
# `build_index` still re-gathered and Python-sorted every cached record
# (88.8M) into one binary blob on EVERY drain -- ~1h of dead time reaching
# `_sort_dedup_pages`, however few files actually changed. `update_index_
# sqlite` merges ONLY a changed file's own rows into a persistent sqlite
# table; `SqlitePersistedIndex` answers the SAME lookup question the old
# mmap `PersistedIndex` does. These tests prove the two backends AGREE on
# the same real fixture, not just that the new one runs.

def test_sqlite_index_agrees_with_the_legacy_mmap_index(store_tree, tmp_path):
    legacy_dir = tmp_path / "idx-legacy"
    PI.build_index(str(legacy_dir), root=str(store_tree),
                   log=lambda *a, **k: None)
    legacy = PI.PersistedIndex.open(str(legacy_dir))

    sqlite_dir = tmp_path / "idx-sqlite"
    BUILD.build(str(sqlite_dir), root=str(store_tree), log=lambda *a, **k: None)
    fast = PI.SqlitePersistedIndex.open(str(sqlite_dir))

    for url, sha in [(PERSISTED_URL, PERSISTED_SHA),
                     (PERSISTED_URL, "wrong-sha-entirely"),
                     (PERSISTED_URL, None),
                     (REAL_URL, None),
                     ("https://nobody.example/never-stored", None)]:
        old_hit = legacy.lookup(url, sha)
        new_hit = fast.lookup(url, sha)
        if old_hit is None:
            assert new_hit is None, url
            continue
        assert new_hit is not None, url
        assert (old_hit.layer, old_hit.store_id, old_hit.sha_matched) == \
            (new_hit.layer, new_hit.store_id, new_hit.sha_matched), url
    legacy.close()
    fast.close()


def test_sqlite_index_audit_pages_agrees_with_the_legacy_index(
        store_tree, tmp_path, monkeypatch):
    """The real caller-facing surface, not just `.lookup()` directly:
    `audit_pages` must produce the identical verdict against either
    backend for the SAME pages."""
    legacy_dir = tmp_path / "idx-legacy"
    PI.build_index(str(legacy_dir), root=str(store_tree),
                   log=lambda *a, **k: None)
    sqlite_dir = tmp_path / "idx-sqlite"
    BUILD.build(str(sqlite_dir), root=str(store_tree), log=lambda *a, **k: None)

    pages = [{"url": PERSISTED_URL, "sha": PERSISTED_SHA},
            {"url": REAL_URL, "sha": None},
            {"url": "https://nobody.example/x", "sha": None}]

    legacy_idx = PI.PersistedIndex.open(str(legacy_dir))
    legacy_verdict = PI.audit_pages(pages, index=legacy_idx, require="l1")
    legacy_idx.close()

    fast_idx = PI.SqlitePersistedIndex.open(str(sqlite_dir))
    fast_verdict = PI.audit_pages(pages, index=fast_idx, require="l1")
    fast_idx.close()

    for key in ("pages", "persisted", "l1b_reconstructible", "missing",
               "verdict"):
        assert legacy_verdict[key] == fast_verdict[key], key


def test_update_index_sqlite_a_second_run_skips_unchanged_files(
        store_tree, tmp_path):
    out = tmp_path / "idx"
    first = BUILD.build(str(out), root=str(store_tree), log=lambda *a, **k: None)
    assert first["rows_indexed"] > 0

    second = PI.update_index_sqlite(str(out), root=str(store_tree),
                                    log=lambda *a, **k: None)
    assert second["files_changed"] == 0, (
        "an unchanged file must be skipped, never re-merged")
    assert second["files_scanned"] == first["files_scanned"]


def test_sqlite_writer_waits_for_another_writers_lock(store_tree, tmp_path,
                                                     monkeypatch):
    """Real incident, task 679, 2026-09-30: two ERIC seals refreshed the
    index together; the second died at sqlite's 5 s default with `database
    is locked` (tranche 070000, rc=1). A writer must WAIT for the holder."""
    import sqlite3
    import threading
    import time

    out = tmp_path / "idx"
    BUILD.build(str(out), root=str(store_tree), log=lambda *a, **k: None)
    monkeypatch.setenv("AW_PERSISTED_INDEX_BUSY_TIMEOUT_S", "30")
    assert PI.sqlite_busy_timeout_s() == 30.0

    holder = sqlite3.connect(PI.sqlite_index_path(str(out)),
                             check_same_thread=False)
    # the live index is WAL (open_sqlite_db switches it on first use)
    assert holder.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    holder.execute("BEGIN IMMEDIATE")          # takes the write lock
    release_after = 1.5
    threading.Timer(release_after, holder.rollback).start()

    t0 = time.time()
    writer = PI.open_sqlite_db(str(out))
    try:
        with writer:
            writer.execute("INSERT OR REPLACE INTO meta (key, value) "
                           "VALUES ('t676', '1')")
    finally:
        writer.close()
        holder.close()
    assert time.time() - t0 >= release_after * 0.8, "writer did not wait"


def test_sqlite_writer_gives_up_after_the_bound(store_tree, tmp_path,
                                               monkeypatch):
    """The wait is bounded and comes from the knob, not sqlite's default: a
    holder that never lets go surfaces as an error, not a hang."""
    import sqlite3

    out = tmp_path / "idx"
    BUILD.build(str(out), root=str(store_tree), log=lambda *a, **k: None)
    monkeypatch.setenv("AW_PERSISTED_INDEX_BUSY_TIMEOUT_S", "0.2")
    holder = sqlite3.connect(PI.sqlite_index_path(str(out)))
    assert holder.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    holder.execute("BEGIN IMMEDIATE")
    writer = PI.open_sqlite_db(str(out))
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            with writer:
                writer.execute("INSERT OR REPLACE INTO meta (key, value) "
                               "VALUES ('t676', '1')")
    finally:
        writer.close()
        holder.rollback()
        holder.close()


def test_sqlite_busy_timeout_defaults_to_a_bounded_wait(monkeypatch):
    monkeypatch.delenv("AW_PERSISTED_INDEX_BUSY_TIMEOUT_S", raising=False)
    assert 60 <= PI.sqlite_busy_timeout_s() <= 3600


def test_update_index_sqlite_rescans_a_changed_file_and_drops_stale_rows(
        store_tree, tmp_path):
    out = tmp_path / "idx"
    BUILD.build(str(out), root=str(store_tree), log=lambda *a, **k: None)
    idx = PI.SqlitePersistedIndex.open(str(out))
    assert idx.lookup(PERSISTED_URL) is not None
    idx.close()

    # Rewrite the L1 file so it no longer mentions PERSISTED_URL at all --
    # a real "page removed at seal-time correction" shape.
    l1_file = next((store_tree / "l1").rglob("*.jsonl.gz"))
    with gzip.open(l1_file, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "schema": "aw.l1-markdown.v1",
            "canonical_url": "https://other.example/page",
            "url": "https://other.example/page",
            "markdown": "# Something else\n",
            "markdown_sha256": "0" * 64}) + "\n")
    os.utime(l1_file, None)  # force a fresh mtime

    second = PI.update_index_sqlite(str(out), root=str(store_tree),
                                    log=lambda *a, **k: None)
    assert second["files_changed"] >= 1

    idx = PI.SqlitePersistedIndex.open(str(out))
    assert idx.lookup(PERSISTED_URL) is None, (
        "a page dropped from its source file must not still resolve -- "
        "delete-then-reinsert, not insert-only")
    assert idx.lookup("https://other.example/page") is not None
    idx.close()


def test_sqlite_persisted_index_open_refuses_when_no_db_exists(tmp_path):
    with pytest.raises(PI.PersistedInputsError, match="no sqlite"):
        PI.SqlitePersistedIndex.open(str(tmp_path / "nope"))


def test_shared_index_prefers_sqlite_when_both_exist(store_tree, tmp_path,
                                                      monkeypatch):
    out = tmp_path / "idx"
    PI.build_index(str(out), root=str(store_tree), log=lambda *a, **k: None)
    BUILD.build(str(out), root=str(store_tree), log=lambda *a, **k: None)
    PI.reset_index_cache()
    idx = PI.shared_index(str(out))
    try:
        assert isinstance(idx, PI.SqlitePersistedIndex)
    finally:
        PI.reset_index_cache()


def test_shared_index_falls_back_to_legacy_when_no_sqlite_db(store_tree,
                                                              tmp_path):
    out = tmp_path / "idx"
    PI.build_index(str(out), root=str(store_tree), log=lambda *a, **k: None)
    PI.reset_index_cache()
    idx = PI.shared_index(str(out))
    try:
        assert isinstance(idx, PI.PersistedIndex)
    finally:
        PI.reset_index_cache()
