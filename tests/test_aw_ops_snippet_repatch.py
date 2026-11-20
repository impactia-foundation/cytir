"""Task 2734: `cyt ops snippet-repatch` (producer) and `snippet` in the twin-patch-fields closed list.

Texts are BUILT in the sealed-pack shape (`Record identity\\n- ...` with no hashes, read from the real
packs on 2026-11-12) and the epoch-4 fedlex shape (`Art. 1 {#art_1}`); no pack or twin is read.
"""
from __future__ import annotations

import json

import pytest

import cytir.core.ids as ID
from cytir.index.b1_projection import projected_snippet
from cytir.index.snippet_surface import strip_provenance_header
from cytir.ops import snippet_repatch as SR
from cytir.ops import twin_patch_fields as TP

PUBMED = ("Record identity\n- PMID: `21`\n- Parent archive: `x.xml.gz`\n- Parent archive SHA-256: `ff`\n\n"
          "Abstract\nBisabolol has a primary antipeptic action depending on dosage.")
BFS = ("Record identity\n- Canonical asset: https://www.bfs.admin.ch/asset/en/1\n- Language: en\n\n"
       "Description\n- Subtitle: Weekly number of deaths")
FEDLEX = "Art. 336 c {#art_336_c}\nArt. 336 c\n\nparagraph {#art_336_c/para_1}\n1 Die Kuendigung ist missbraeuchlich."
PLAIN = "Just a normal passage about nothing in particular."


def key(text):
    return ID.content_digest(text)[:32]          # the twin stores the first 32 hex


def row(i, text, snippet, domain="x"):
    return {"id": f"id-{i}", "digest": key(text), "snippet": snippet}


def test_pack_shaped_header_is_stripped_and_both_forms_match():
    assert strip_provenance_header(PUBMED).startswith("Abstract\nBisabolol")
    assert strip_provenance_header("# T\n\n## Record identity\n\n- a: b\n\n## Abs\nx") == "# T\n\n## Abs\nx"
    assert strip_provenance_header("Foo\nRecord identity\n- a: b\n\nx").startswith("Foo\nRecord")


def test_build_patch_counts_every_class_and_leaves_no_text_alone():
    rows = {
        "pubmed.ncbi.nlm.nih.gov": [row(1, PUBMED, "Record identity - PMID: `21` - Parent archive"),
                                    row(2, PLAIN, projected_snippet("pubmed.ncbi.nlm.nih.gov", PLAIN)),
                                    {"id": "id-3", "digest": "f" * 32, "snippet": "old"}],
        "fedlex.admin.ch": [row(4, FEDLEX, "Art. 336 c {#art_336_c} Art. 336 c paragraph")],
    }
    patch, counts = SR.build_patch(rows, [iter([PUBMED, PLAIN]), iter([FEDLEX])], projected_snippet)
    assert counts["pubmed.ncbi.nlm.nih.gov"] == {"selected": 3, "recomputed": 2, "changed": 1,
                                                 "skipped_equal": 1, "no_text": 1}
    assert counts["fedlex.admin.ch"] == {"selected": 1, "recomputed": 1, "changed": 1}
    by = {p["id"]: p for p in patch}
    assert set(by) == {"id-1", "id-4"}                        # id-3 (no text) is never patched
    assert by["id-1"]["set"]["snippet"].startswith("Abstract Bisabolol")
    assert "{#" not in by["id-4"]["set"]["snippet"] and "Record identity" not in by["id-1"]["set"]["snippet"]
    assert by["id-4"]["expect"] == {"content_digest": key(FEDLEX),
                                    "snippet": "Art. 336 c {#art_336_c} Art. 336 c paragraph"}


def test_the_snippet_is_the_projection_functions_not_a_second_rule():
    patch, _ = SR.build_patch({"bfs.admin.ch": [row(1, BFS, "old")]}, [iter([BFS])],
                              lambda d, t: "SENTINEL")
    assert patch[0]["set"] == {"snippet": "SENTINEL"}


def test_patch_rows_load_in_twin_patch_fields_and_snippet_is_patchable(tmp_path):
    patch, _ = SR.build_patch({"bfs.admin.ch": [row(1, BFS, "old")]}, [iter([BFS])], projected_snippet)
    f = tmp_path / "p.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in patch))
    fields = TP.parse_fields("snippet")
    rows, sha = TP.load_rows(f, fields)
    assert rows[0]["set"]["snippet"].startswith("Description") and len(sha) == 64
    assert "snippet" in TP.PATCHABLE and "url" not in TP.PATCHABLE


def test_snippet_writes_are_coalesced_into_batches_of_at_most_batch_points():
    rows = [{"id": f"i{n}", "set": {"snippet": f"s{n}"}, "unset": [], "expect": {}} for n in range(2500)]
    calls = TP.write_groups(rows, 1000, coalesce=True)
    assert [op for op, _, _ in calls] == ["multi"] * 3
    assert [len(ids) for _, _, ids in calls] == [1000, 1000, 500]
    assert all(len(o["points"]) == 1 for _, what, _ in calls for o in what)
    plain = TP.write_groups(rows[:3], 1000)                       # default path unchanged
    assert [op for op, _, _ in plain] == ["set"] * 3


def test_the_producer_refuses_a_prod_target(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(SR.PG, "is_prod", lambda url, coll: True)
    code = SR.main(["--domain", "bfs.admin.ch", "--text-source", f"pack={tmp_path}", "--out",
                    str(tmp_path / "o"), "--twin-url", "http://127.0.0.1:1", "--twin-collection", "c"])
    assert code == 4 and "prod" in capsys.readouterr().out
    assert not (tmp_path / "o").exists()


def test_output_is_sha_pinned_and_never_overwritten(tmp_path):
    patch, counts = SR.build_patch({"bfs.admin.ch": [row(1, BFS, "old")]}, [iter([BFS])], projected_snippet)
    m = SR.write_outputs(tmp_path / "o", patch, counts, {})
    import hashlib
    assert m["sha256"] == hashlib.sha256((tmp_path / "o" / "snippet-patch.jsonl").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        SR.write_outputs(tmp_path / "o", patch, counts, {})


# ------------------------------------------------- apply through twin-patch-fields on a CODED twin
def _apply(tmp_path, monkeypatch, fields, patch_rows, extra_payload=None, profile="rebuild_e10"):
    import contextlib
    import datetime as dt
    import types
    from cytir.index import row_decode as RD
    monkeypatch.setattr(RD, "profile_of", lambda coll, **kw: profile)
    store = {r["id"]: {"content_digest": r["expect"]["content_digest"], "snippet": r["expect"]["snippet"],
                       "rule_epochs": 4, "url": "u", **(extra_payload or {})} for r in patch_rows}
    before = {k: dict(v) for k, v in store.items()}
    writes = []

    def fetch(url, coll, ids, keys, batch):
        return {i: {k: store[i][k] for k in keys if k in store[i]} for i in ids if i in store}

    def write(url, coll, op, what, ids):
        writes.append((op, len(ids)))
        for o in (what if op == "multi" else [{"payload": what, "points": ids}]):
            for i in o["points"]:
                store[i].update(o["payload"])

    @contextlib.contextmanager
    def locks(task):
        yield "lock"

    q = tmp_path / "q"
    (q / "_artifacts").mkdir(parents=True)
    f = tmp_path / "in.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in patch_rows))
    settings = types.SimpleNamespace(work_root=str(tmp_path / "w"), queue_dir=str(q))
    lines = []
    code = TP.run(["--twin-url", "http://twin.invalid:16457", "--twin-collection", "aw_e10_twin",
                   "--from", str(f), "--fields", fields, "--apply", "--export", str(tmp_path / "ex.jsonl"),
                   "--pause", "0.2", "--batch", "2"], settings=settings, fetch_fn=fetch, write_fn=write,
                  locks=locks, now=lambda: dt.datetime(2026, 11, 12, tzinfo=dt.timezone.utc),
                  out=lines.append, sleep_fn=lambda s: writes.append(("sleep", s)))
    return code, store, before, writes, lines


def test_snippet_applies_on_a_coded_twin_touching_no_other_key(tmp_path, monkeypatch):
    patch, _ = SR.build_patch({"bfs.admin.ch": [row(n, BFS, "old") for n in range(5)]}, [iter([BFS])],
                              projected_snippet)
    code, store, before, writes, _ = _apply(tmp_path, monkeypatch, "snippet", patch)
    assert code == 0
    for i, b in before.items():
        assert {k: v for k, v in store[i].items() if k != "snippet"} == {k: v for k, v in b.items() if k != "snippet"}
        assert store[i]["snippet"].startswith("Description")
    assert [w for w in writes if w[0] == "multi"] == [("multi", 2), ("multi", 2), ("multi", 1)]
    assert ("sleep", 0.2) in writes
    assert (tmp_path / "ex.jsonl").read_text().count("\n") == 5          # rollback = the old snippet


def test_a_coded_key_in_fields_still_refuses_on_a_coded_twin(tmp_path, monkeypatch):
    rows = [{"id": "a", "set": {"rule_epochs": 1}, "unset": [], "expect": {"content_digest": "d"}}]
    code, store, before, writes, lines = _apply(tmp_path, monkeypatch, "rule_epochs", [
        {**rows[0], "expect": {"content_digest": "d", "snippet": "s"}}])
    assert code == TP.EXIT_REFUSED and not writes and "CODED" in " ".join(lines)


# ------------------------------------------- the coded-twin refusal fires (review of the gate change)
def _row(sets, expect):
    return {"id": "a", "set": sets, "unset": [], "expect": {"content_digest": "d", **expect}}


@pytest.mark.parametrize("field", ["stamps"])   # `tier` is not in PATCHABLE: parse_fields refuses it first
def test_refuses_a_DROPPED_key_in_fields_on_a_coded_twin(tmp_path, monkeypatch, field):
    code, store, before, writes, lines = _apply(tmp_path, monkeypatch, field,
                                                [_row({field: 1}, {"snippet": "s"})])
    assert code == TP.EXIT_REFUSED and not writes and store == before and "CODED" in " ".join(lines)


@pytest.mark.parametrize("key", ["rule_epochs", "tier"])      # one coded key, one dropped key
def test_refuses_a_coded_key_only_in_expect_with_snippet_in_fields(tmp_path, monkeypatch, key):
    code, store, before, writes, lines = _apply(tmp_path, monkeypatch, "snippet",
                                                [_row({"snippet": "new"}, {"snippet": "s", key: 4})])
    assert code == TP.EXIT_REFUSED and not writes and store == before
    assert key in " ".join(lines)


def test_refuses_an_undeclared_profile_even_for_snippet(tmp_path, monkeypatch):
    code, store, before, writes, lines = _apply(
        tmp_path, monkeypatch, "snippet", [_row({"snippet": "new"}, {"snippet": "s"})],
        profile="not_a_declared_profile")
    assert code == TP.EXIT_REFUSED and not writes and store == before and "CODED" in " ".join(lines)
