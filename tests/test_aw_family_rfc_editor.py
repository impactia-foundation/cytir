"""`rfc_editor_net_new` and `rfc_exact` on the family Protocol (task 1395).

No network, no live fixture: the RFC tree is two tiny synthetic files written here. Pins: both
families are registered and conformant; discover / acquire / extract run on the tree; consent and
licence come from each row (`rfc_exact` declares none, so it refuses); the whole-tree seal runs and
is idempotent; an empty text is excluded.
"""
from __future__ import annotations

import pytest

from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.family import Doc, RunContext, SourceFamily, UnitExcluded
from cytir.sources.families.rfc_editor import RfcEditorNetNewFamily, RfcExactFamily
from cytir.sources.rfc_editor import preserved


def _ctx(tmp_path, row, **options) -> RunContext:
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
                      options=options, contract=row, log=lambda m: None)


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "run" / "input" / "rfcs"
    d.mkdir(parents=True)
    (d / "rfc1.txt").write_bytes(b"RFC one\n\nSee https://example.org/ref.\n")
    (d / "rfc2.txt").write_bytes(b"  \n")
    return d


def test_registered_and_conformant():
    assert {"rfc_editor_net_new", "rfc_exact"} <= set(registry.registered())
    adapter, row, spec = registry.admit("rfc_editor_net_new")
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == "rfc_editor_net_new"
    assert spec.declared == row["chunk_policy"]["policy_id"]
    assert adapter.PARTIAL == ()
    assert RfcExactFamily.PARTIAL == ()


def test_rfc_exact_is_admitted_for_its_chunk_policy_but_still_refuses_to_run_on_consent():
    """Task 2136 declared its chunk policy, so the registry admits it; consent and licence stay
    `unknown`, so `discover` refuses until a POLICY PR declares them (next test)."""
    adapter, row, spec = registry.admit("rfc_exact")
    assert isinstance(adapter, RfcExactFamily) and spec.declared == row["chunk_policy"]["policy_id"]


def test_net_new_discover_acquire_extract_from_its_row(raw, tmp_path):
    adapter, row, _ = registry.admit("rfc_editor_net_new")
    ctx = _ctx(tmp_path, row, raw_dir=str(raw))
    units = list(adapter.discover(ctx))
    assert [u.key for u in units] == ["rfc1.txt", "rfc2.txt"]
    assert all(u.url.startswith("file:") for u in units)
    terms = ctx.notes["contract_terms"]
    assert terms["search_ok"] is True and terms["consent_resolution"] == row["consent"]["resolution"]
    assert terms["licence_basis"] == row["licence"]["basis"]
    (cap,) = adapter.acquire(units[0], ctx)
    (doc,) = adapter.extract(cap, ctx)
    assert doc.url == "https://www.rfc-editor.org/rfc/rfc1.txt" and doc.title == "RFC 1"
    assert doc.lang == "en" and doc.text.startswith("RFC one")
    (empty,) = adapter.acquire(units[1], ctx)
    with pytest.raises(UnitExcluded, match="empty_source_text"):
        adapter.extract(empty, ctx)


SYNTH = ("RFC 9999  Synthetic Example  January 2000\n\n1.  Introduction\n\n"
         + "This synthetic paragraph exists only for the test. " * 30
         + "\n\n2.  Details\n\n" + "Another synthetic paragraph about nothing. " * 30 + "\n")


def test_net_new_chunk_rerenders_heading_markdown_and_mints_pack_ids(tmp_path, word_only_chunking_allowed):
    from cytir.core import common as C
    adapter, row, _ = registry.admit("rfc_editor_net_new")
    raw = tmp_path / "run" / "input" / "rfcs"
    raw.mkdir(parents=True)
    (raw / "rfc9999.txt").write_text(SYNTH)
    ctx = _ctx(tmp_path, row, raw_dir=str(raw))
    with pytest.raises(RuntimeError, match="preflight"):
        RfcEditorNetNewFamily().chunk(adapter.extract(adapter.acquire(adapter.discover(ctx)[0], ctx)[0], ctx)[0], ctx)
    adapter.preflight(ctx)
    (cap,) = adapter.acquire(adapter.discover(ctx)[0], ctx)
    (doc,) = adapter.extract(cap, ctx)
    assert doc.text == SYNTH                     # extract still returns the raw text
    recs = adapter.chunk(doc, ctx)
    assert recs and recs[0]["text"].startswith("---\naw_rfc_number: 9999\n")
    assert any("1.  Introduction" in r["text"] for r in recs)
    for n, r in enumerate(recs):
        assert r["anchor"] == f"chunk-{n}" and r["chunk_ordinal"] == n
        assert r["document_id"] == "rfc-editor:rfc:9999"
        assert r["point_id"] == C.mint_passage_point_id(doc.url, f"chunk-{n}", r["text"])


def test_rfc_exact_chunk_uses_the_sealed_anchor_shape_and_emits_no_invented_fields(tmp_path, word_only_chunking_allowed):
    from cytir.core import common as C
    row = contract_mod.by_id(contract_mod.load())["rfc_exact"]
    assert row["chunk_policy"]["sha256"].startswith("7dcb54c4") and row["chunk_policy"]["epoch"] == 4
    assert not any(u.startswith("chunk_policy") for u in row["unknowns"])
    ctx = _ctx(tmp_path, row)
    adapter = RfcExactFamily()
    raw = SYNTH.encode()
    mrow, _ = preserved.rfc_row(9999, raw, {"sha256": "0" * 64, "relative_path": "input/rfcs/rfc9999.txt",
                                           "bytes": len(raw)})
    doc = Doc(url=mrow["url"], text=mrow["markdown"], title=mrow["title"], lang=mrow["lang"],
              source_text=mrow["markdown"], extra={"row": mrow})
    with pytest.raises(RuntimeError, match="preflight"):
        adapter.chunk(doc, ctx)
    adapter.preflight(ctx)
    recs = adapter.chunk(doc, ctx)
    assert recs and recs[0]["text"].startswith("---\naw_rfc_number: 9999\n")
    for n, r in enumerate(recs):
        anchor = f"document:rfc-editor:rfc:9999:chunk-{n}"
        assert r["anchor"] == anchor and r["chunk_ordinal"] == n
        assert r["point_id"] == C.mint_passage_point_id(doc.url, anchor, r["text"])
        # the sealed stage rows carry none of these: they are absent, not invented
        for absent in ("tier", "source_type", "trust_tier", "license"):
            assert absent not in r


def test_rfc_exact_refuses_while_its_row_declares_no_consent(raw, tmp_path):
    row = contract_mod.by_id(contract_mod.load())["rfc_exact"]
    adapter = RfcExactFamily()
    assert row["consent"]["search_ok"] is None          # `unknown` stays unknown
    with pytest.raises(ValueError, match="search_ok"):
        adapter.discover(_ctx(tmp_path, row, raw_dir=str(raw)))


def test_wrong_root_is_an_error_not_an_empty_source(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    adapter, row, _ = registry.admit("rfc_editor_net_new")
    with pytest.raises(preserved.RfcAcquisitionError):
        adapter.discover(_ctx(tmp_path, row, raw_dir=str(empty)))
    with pytest.raises(ValueError, match="raw_dir"):
        adapter.discover(_ctx(tmp_path, row))


def test_seal_inventory_delegates_to_materialize(raw, tmp_path):
    root = raw.parent.parent
    before, after = root / "input/before.txt", root / "input/after.txt"
    before.write_text("rfc1.txt\nrfc2.txt\n")
    after.write_text("rfc1.txt\nrfc2.txt\n")
    adapter, row, _ = registry.admit("rfc_editor_net_new")
    opts = dict(raw_dir=str(raw), before_list=str(before), after_list=str(after), root=str(root),
                module="fixture-module::rfcs-text-only", started_at="2026-08-15T00:00:00Z",
                completed_at="2026-08-15T00:01:00Z", rsync_version="rsync version 3.2.7",
                fetch_vantage="vantage-a")
    run = adapter.seal_inventory(_ctx(tmp_path, row, **opts))
    assert run["counts"] == {"original_documents": 2, "markdown_documents": 1,
                             "outlink_records": 1, "quarantined_documents": 1}
    assert run == adapter.seal_inventory(_ctx(tmp_path, row, **opts))
    assert isinstance(adapter, RfcEditorNetNewFamily) and not isinstance(adapter, RfcExactFamily)


# Reference digests: produced by the ORIGINAL flat driver (the version before the move, task 1386's
# merge commit) on this exact tree, run twice with identical output. Nothing is excluded: every
# timestamp in RUN.json is an input of the call, so the whole file is deterministic.
REFERENCE = {
    "run_json": "d0871f2bc948e77e71922c2cab4125f2eec3d332c2bdff11499ddc5da48f5bc4",
    "original": "750d42ba2603b874c22f5701b0ccd1fe39d818e461495ffc32aab07d938c69ec",
    "markdown": "a01686a91a8dc606979200783dc6d1fdcd295cb63e1f573ddbd78726984b6421",
    "outlinks": "275d5e39b2fc9de2abf04e32d756682994bde32812027281111e885fa86ce972",
    "quarantine": "7bab58e600c8d9d70484b193a4d20cc9b8086272cd0c87ed89baf2ba4c048f29",
}


def test_seal_is_byte_identical_to_the_original_driver(tmp_path):
    import hashlib
    root = tmp_path / "run"
    raw = root / "input" / "rfcs"
    raw.mkdir(parents=True)
    (raw / "rfc1.txt").write_bytes(b"RFC one\n\nSee https://example.org/ref. And https://example.org/ref "
                                   b"again, https://[bad/x.\n")
    (raw / "rfc2.txt").write_bytes(b"\xef\xbb\xbfRFC two with BOM https://b.example/p;\n")
    (raw / "rfc3.txt").write_bytes("RFC three caf\xe9 https://c.example/\xe9x\n".encode("iso-8859-1"))
    (raw / "rfc4.txt").write_bytes(b" \n")
    before, after = root / "input" / "before.txt", root / "input" / "after.txt"
    before.write_text("x\n")
    after.write_text("x\n")
    adapter, row, _ = registry.admit("rfc_editor_net_new")
    run = adapter.seal_inventory(_ctx(
        tmp_path, row, raw_dir=str(raw), before_list=str(before), after_list=str(after), root=str(root),
        module="fixture-module::rfcs-text-only", started_at="2026-08-15T00:00:00Z",
        completed_at="2026-08-15T00:01:00Z", rsync_version="rsync version 3.2.7",
        fetch_vantage="vantage-a"))
    got = {"run_json": hashlib.sha256((root / "RUN.json").read_bytes()).hexdigest()}
    got.update({k: run["stages"][k]["logical_sha256"]
                for k in ("original", "markdown", "outlinks", "quarantine")})
    assert got == REFERENCE
