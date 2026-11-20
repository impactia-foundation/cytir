"""The aw_e10 writer lane (task 864; runbook P7, review 854 S5/S6/S7).

What is pinned here:
* (1) the I1 id of a CARD (ADR-067 am.1, `canon#card:<identity>`) is what every writer puts on an
  I1 collection: the drain mints it for a sealed pack (`promote_sealed_pack.project_rows(id_rule=)`),
  a content-pack composer keys its points with `id_rule.key_points_for` (the restore pack), and
  the build's map sends the REAL twin card's old url-only id to that same id as a `survivor`;
* (3) S6: every direct twin writer outside the drain refuses while an aw_e10 build holds the twin;
* (4) S7: `rebuild_e10 reverse-map` (deleted builder verb, task 1312): an aw_e10 id back to the old ids a rollback replay writes
  (the survivor's) and deletes (the losers').
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys
import types
import uuid

import pytest

from cytir.core import ids as ID
from cytir.core.canon import canonical_url
from cytir.core.ids import mint_point_id
from cytir.index import id_rule as IR
from cytir.index import payload_profile as PP
from cytir.index import prod_guard as PG
from cytir.index import promote_sealed_pack as PSP
from cytir.index import twin_write_hold as TWH
from cytir.ops import restore_pack as RP

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
CARDS = HERE / "fixtures" / "card-identity-shared-url-rows.json"


@pytest.fixture(autouse=True)
def _operator(monkeypatch):
    monkeypatch.setenv("AW_OPERATOR_URL", "https://websear.ch")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "bender@impactia.org")


def _live_copy(tmp_path: pathlib.Path, *names: str) -> pathlib.Path:
    """The declarations with `names` switched ON (state live), as the flip commit has them."""
    doc = json.loads(PP.PROFILES_PATH.read_text(encoding="utf-8"))
    for n in names:
        doc["collections"][n]["state"] = "live"
        doc["collections"][n]["sparse_rule"] = None      # the term set is not in CI
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return p


def _real_card(host: str = "batitec.ch", n: int = 0) -> dict:
    """A REAL twin card row (task 742's shared-url fixture), completed with the two raw keys the
    fixture leaves out: `tier` C and `point_id` = its stored id."""
    row = json.loads(CARDS.read_text())[host][n]
    return {"id": row["id"], "payload": dict(row["payload"], tier="C", point_id=row["id"])}


# ------------------------------------------------------------------------------- (1) cards


def _card_record(url="https://batitec.ch/", pid=None):
    rec = {"url": url, "text": "Batitec AG, a Swiss company in the commercial register.",
           "title": "Batitec AG", "lang": "de", "source_type": "entity_card", "trust_tier": 0.85,
           "license": "test-only-not-a-licence-finding", "register_id": "CHE-103.883.756",
           "register_scheme": "CHE-UID", "entity_kind": "org", "country": "CH"}
    rec["point_id"] = pid or mint_point_id(canonical_url(url), "")
    return rec


def _page_record(url="https://batitec.ch/about", text="About Batitec: we build things."):
    rec = {"url": url, "text": text, "title": "About", "lang": "en", "source_type": "reference",
           "trust_tier": 0.5, "license": "test-only-not-a-licence-finding"}
    rec["point_id"] = mint_point_id(canonical_url(url), ID.content_digest(text))
    return rec


def _project(rows, **kw):
    return PSP.project_rows(iter(rows), consent=None, attribution=None, license_version=None, **kw)


def test_the_drain_writes_the_am1_card_id_for_an_i1_target_and_keeps_the_pack_id_as_join_key():
    card, page = _card_record(), _page_record()
    out, skipped, modes, vector_key, _ = _project([card, page], id_rule="I1")
    want = IR.mint_i1(out[next(iter(out))][2])[0]
    assert skipped == 0 and set(out) == {want, page["point_id"]}
    assert want == mint_point_id(canonical_url(card["url"]), "card:reg:CHE:CHE103883756")
    assert vector_key[want] == card["point_id"] and vector_key[page["point_id"]] == page["point_id"]
    assert modes["id_rule"] == "I1" and modes["i1_reminted"] == 1 and modes["i1_merged"] == 0
    # every point the drain builds for the I1 target passes the I1 guard
    assert IR.check_points([{"id": k, "payload": v[2]} for k, v in out.items()]) == []


def test_without_an_id_rule_the_drain_writes_what_it_wrote_before():
    card, page = _card_record(), _page_record()
    out, _, modes, vector_key, _ = _project([card, page])
    assert set(out) == {card["point_id"], page["point_id"]} and "id_rule" not in modes
    legacy = dict(_page_record(), point_id=str(uuid.UUID(int=7)))
    out, _, modes, _, _ = _project([legacy], identity_check="pack-declared", identity_reason="test")
    assert set(out) == {legacy["point_id"]} and modes["pack_declared"] == 1


def test_a_pack_declared_legacy_id_is_reminted_to_i1_and_identical_text_merges():
    a = dict(_page_record(), point_id=str(uuid.UUID(int=11)))        # a generation-0 pack id
    b = dict(_page_record(), point_id=str(uuid.UUID(int=12)))        # same page, same text
    out, _, modes, vector_key, _ = _project([a, b], identity_check="pack-declared",
                                            identity_reason="test", id_rule="I1")
    i1 = mint_point_id(canonical_url(a["url"]), ID.content_digest(a["text"]))
    assert list(out) == [i1] and vector_key[i1] == a["point_id"]
    assert modes["i1_merged"] == 1 and modes["i1_merged_rows"][0]["point_id"] == b["point_id"]
    with pytest.raises(PSP.PromotionError, match="id_rule"):
        _project([a], id_rule="I2")




def test_the_drain_refuses_two_legs_with_different_id_rules_before_any_write(tmp_path, monkeypatch):
    from cytir.index import promote_queue as PQ
    p = _live_copy(tmp_path, "aw_e10")
    doc = json.loads(p.read_text())
    doc["collections"]["aw_e10_twin"] = dict(doc["collections"]["aw_e10"])
    p.write_text(json.dumps(doc))
    monkeypatch.setattr(PP, "PROFILES_PATH", p)
    with pytest.raises(PP.PayloadProfileError, match="id_rule"):
        PQ.payload_profile_preflight_both("aw_e8", "aw_e10", [])      # the F7 window: twin aw_e8
    with pytest.raises(PP.PayloadProfileError, match="id_rule"):
        PQ.payload_profile_preflight_both("aw_e10_twin", "aw_e9", [])
    assert set(PQ.payload_profile_preflight_both("aw_e10_twin", "aw_e10", [])) == {"twin", "prod"}
    assert set(PQ.payload_profile_preflight_both("aw_e8", "aw_e9", [])) == {"twin", "prod"}   # today


def test_a_content_pack_with_a_non_i1_id_is_refused_before_the_write_starts(tmp_path, monkeypatch):
    from cytir.index import promote_queue as PQ
    monkeypatch.setattr(PP, "PROFILES_PATH", _live_copy(tmp_path, "aw_e10"))
    calls = []
    monkeypatch.setattr(PQ.LAYOUT, "fingerprint", lambda *a, **k: calls.append(a) or {})
    monkeypatch.setattr(PSP, "upsert", lambda *a, **k: calls.append(a) or 0)
    wa = PQ.WriteAttempt()
    with pytest.raises(IR.IdRuleRefused):
        PQ.apply_content_pack_side("twin", "http://127.0.0.1:1", "aw_e10", [_real_card()], [],
                                   batch=10, pause_s=0, write_attempt=wa)
    assert wa.started is False and calls == []                         # REFUSED, not ABORTED


def test_key_points_for_keys_a_content_pack_by_the_targets_rule(tmp_path, monkeypatch):
    card = _real_card()
    page_pl = {"url": "https://batitec.ch/x", "tier": "B", "content_digest": "a" * 64}
    page = {"id": mint_point_id(canonical_url(page_pl["url"]), "a" * 64), "payload": page_pl}
    pts = [card, page]
    assert IR.key_points_for("aw_e8", pts) == (pts, {})                 # no id_rule: unchanged
    monkeypatch.setattr(PP, "PROFILES_PATH", _live_copy(tmp_path, "aw_e10"))
    out, moved = IR.key_points_for("aw_e10", pts)
    am1 = IR.mint_i1(card["payload"])[0]
    assert [p["id"] for p in out] == [am1, page["id"]] and moved == {card["id"]: am1}
    assert out[0]["payload"] is card["payload"]                         # payload untouched
    with pytest.raises(IR.IdRuleRefused, match="aw_e10"):
        IR.key_points_for("aw_e10", [page, dict(page, id=str(uuid.UUID(int=3)))])   # one I1 id twice
    with pytest.raises(IR.IdRuleRefused):
        IR.key_points_for("aw_e10", [{"id": "x", "payload": {"url": "https://a.example/", "tier": "B"}}])


def _vec():
    return {"dense": [0.1, 0.2], "bm25": {"indices": [3, 9], "values": [1.0, 2.0]}}


def test_a_restore_pack_for_an_i1_collection_carries_am1_ids_and_the_map(tmp_path, monkeypatch):
    card = dict(_real_card(), vector=_vec())
    man = RP.write_pack(str(tmp_path / "q8"), "p", [card], collection="aw_e8", family="general_web",
                        note="n", task=864)
    assert "id_map" not in man                                          # aw_e8: ids as exported
    monkeypatch.setattr(PP, "PROFILES_PATH", _live_copy(tmp_path, "aw_e10"))
    man = RP.write_pack(str(tmp_path / "q10"), "p", [card], collection="aw_e10", family="general_web",
                        note="n", task=864)
    am1 = IR.mint_i1(card["payload"])[0]
    assert man["id_rule"] == "I1" and man["id_map"] == {card["id"]: am1}
    row = json.loads(gzip.open(man["pack_points"], "rt").readline())
    assert row["id"] == am1 and row["payload"] == card["payload"] and row["vector"] == card["vector"]


# --------------------------------------------------------------------------- (3) S6: the hold
@pytest.fixture
def hold(tmp_path, monkeypatch):
    p = tmp_path / "hold.json"
    monkeypatch.setenv(TWH.ENV, str(p))
    TWH.write_hold(collection="aw_e8", source_qdrant="http://twin.invalid:6457", plan_dir="/plan", t0="T0")
    return p


def test_every_aw_twin_writer_calls_the_hold():
    import importlib
    for mod, what in TWH.HELD_WRITERS.items():
        src = pathlib.Path(importlib.import_module(mod).__file__).read_text()
        assert "refuse_if_held(" in src, (mod, what)


def test_twin_patch_fields_apply_is_refused_while_the_twin_is_held(hold, tmp_path):
    from cytir.ops import twin_patch_fields as TP
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"id": str(uuid.UUID(int=1)), "set": {"license": "x"},
                               "expect": {"content_digest": "d"}}) + "\n")
    calls = []
    settings = types.SimpleNamespace(work_root=str(tmp_path / "w"), queue_dir=str(tmp_path / "q"))
    out: list[str] = []
    code = TP.run(["--from", str(src), "--fields", "license", "--apply", "--export", str(tmp_path / "e"),
                   "--twin-url", "http://twin.invalid:16457", "--twin-collection", "aw_e8"],
                  settings=settings, fetch_fn=lambda *a: calls.append(a) or {},
                  write_fn=lambda *a: calls.append(a), locks=None, out=out.append)
    assert code == TP.EXIT_REFUSED and calls == [] and "aw_e10 build holds" in out[0]
    # the dry plan still runs (it writes nothing)
    code = TP.run(["--from", str(src), "--fields", "license", "--twin-url", "http://twin.invalid:16457",
                   "--twin-collection", "aw_e8"], settings=settings,
                  fetch_fn=lambda *a: {}, write_fn=lambda *a: calls.append(a), locks=None, out=out.append)
    assert code == TP.EXIT_OK and calls == []


@pytest.mark.parametrize("apply", [False, True])
def test_twin_patch_fields_refuses_a_coded_twin_dry_or_not(tmp_path, monkeypatch, apply):
    """Task 2734 rule: a coded twin refuses the keys its profile CODES or DROPS (in `--fields` or in
    `expect`); a key it stores verbatim (`snippet`, `license`) patches as on a raw twin."""
    from cytir.ops import twin_patch_fields as TP
    monkeypatch.setattr(PP, "PROFILES_PATH", _live_copy(tmp_path, "aw_e10"))
    settings = types.SimpleNamespace(work_root=str(tmp_path / "w"), queue_dir=str(tmp_path / "q"))

    def run(fields, setv, expect):
        src = tmp_path / f"in-{fields}-{sorted(expect)}.jsonl"
        src.write_text(json.dumps({"id": str(uuid.UUID(int=1)), "set": setv,
                                   "expect": {"content_digest": "d", **expect}}) + "\n")
        calls, out = [], []
        extra = ["--apply", "--export", str(tmp_path / f"ex-{len(list(tmp_path.iterdir()))}.jsonl")] if apply else []
        code = TP.run(["--from", str(src), "--fields", fields, "--twin-url", "http://twin.invalid:16457",
                       "--twin-collection", "aw_e10_twin", *extra], settings=settings,
                      fetch_fn=lambda *a: calls.append(a) or {}, write_fn=lambda *a: calls.append(a),
                      locks=None, out=out.append)
        return code, calls, out

    # aw_e10_twin is rebuild_e10 (aw_e10 now serves: refused as prod first)
    for fields, setv, expect in (("rule_epochs", {"rule_epochs": 1}, {}),      # CODED key in fields
                                 ("stamps", {"stamps": 1}, {}),                # DROPPED key in fields
                                 ("snippet", {"snippet": "n"}, {"tier": 1})):  # dropped key in expect
        code, calls, out = run(fields, setv, expect)
        assert code == TP.EXIT_REFUSED and calls == [] and "CODED" in out[-2], (fields, out)
    code, calls, out = run("snippet", {"snippet": "n"}, {"snippet": "o"})      # a verbatim key passes the gate
    assert "CODED" not in " ".join(out), out
    if not apply:
        assert code == TP.EXIT_OK                                              # dry: absent row, no refusal


def test_the_reconcile_cli_is_refused_on_a_held_twin_and_free_elsewhere(hold, monkeypatch):
    from cytir.index import supersede as US
    monkeypatch.setattr(sys, "argv", ["x", "--in", "/nonexistent.jsonl", "--qdrant-url",
                                      "http://127.0.0.1:1", "--collection", "aw_e8"])
    with pytest.raises(SystemExit, match="aw_e10 build holds"):
        US.main()
    TWH.refuse_if_held("aw_e10_replay_t", tool="t")                      # another collection: free


# ------------------------------------------------------------------------ (4) S7: reverse-map








def test_an_exclusion_naming_the_minted_i1_id_drops_the_row_whose_pack_id_differs():
    """Exclusion files name ids the TARGET holds (the minted I1 id). The card's pack id is the
    url-only id, so a declared-id test alone lets the row through (the drain refused a resealed
    pack for this). The dropped row carries the PACK id, which the vector join subtracts."""
    card, page = _card_record(), _page_record()
    minted = mint_point_id(canonical_url(card["url"]), "card:reg:CHE:CHE103883756")
    assert minted != card["point_id"]
    out, _s, modes, vector_key, _ = _project([card, page], id_rule="I1",
                                             exclude_ids={minted: "already landed"})
    assert set(out) == {page["point_id"]} and minted not in vector_key
    dropped = modes["out_of_scope_dropped_rows"]
    assert [(d["point_id"], d["minted_id"], d["reason"]) for d in dropped] == [
        (card["point_id"], minted, "already landed")]
    assert modes["excluded_out_of_scope_language"] == 1
    # the pack id of the dropped row is what leaves the shard side of the join
    assert {vector_key[k] for k in out} == {page["point_id"]}


def test_an_exclusion_naming_the_pack_id_still_works_under_an_id_rule():
    card, page = _card_record(), _page_record()
    out, _s, modes, _vk, _ = _project([card, page], id_rule="I1", exclude_ids={card["point_id"]: "r"})
    assert set(out) == {page["point_id"]} and modes["excluded_out_of_scope_language"] == 1


def test_an_excluded_minted_id_shared_by_a_merge_group_drops_the_whole_group_and_the_join_stays_total():
    """Review 1013: the excluded minted id is the I1 id of TWO pack rows (one page, two spellings).
    The survivor leaves as an exclusion (pack id + minted id + the file's reason), the other member
    stays an `i1_merged` row, and `plan()`'s vector-side subtraction still accounts every pack id."""
    text = "EEOC history page, the same page under two spellings."
    www = dict(_page_record(url="https://www.eeoc.gov/history", text=text), point_id=str(uuid.UUID(int=31)))
    canon = dict(_page_record(url="https://eeoc.gov/history", text=text), point_id=str(uuid.UUID(int=32)))
    other = _page_record()
    i1 = mint_point_id(canonical_url(canon["url"]), ID.content_digest(text))
    out, _s, modes, vector_key, undated = _project(
        [www, canon, other], identity_check="pack-declared", identity_reason="test", id_rule="I1",
        exclude_ids={i1: "already landed"})
    assert set(out) == {other["point_id"]} and i1 not in vector_key
    assert [(d["point_id"], d["minted_id"], d["reason"]) for d in modes["out_of_scope_dropped_rows"]] == [
        (canon["point_id"], i1, "already landed")]
    assert [m["point_id"] for m in modes["i1_merged_rows"]] == [www["point_id"]]
    assert modes["excluded_out_of_scope_language"] == 1 and modes["i1_merged"] == 1
    # the same subtraction `plan()` runs: every pack id is a written row, an exclusion or a merge
    vec_side = {r["point_id"] for r in (www, canon, other)}
    vec_side -= {str(r["point_id"]) for r in undated}
    vec_side -= {str(r["point_id"]) for r in modes["out_of_scope_dropped_rows"]}
    vec_side -= {str(r["point_id"]) for r in modes["i1_merged_rows"]}
    assert vec_side == {vector_key[k] for k in out}
