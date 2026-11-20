"""`canonical_duplicate` and `binary_content`: the two PURE, export-backed tombstone reasons of
task 767 (draft PR). Rows are REAL: five `ai.google.dev` `?authuser=N` variants and their
canonical page, and two real WEKO binary-PDF rows with two real prose rows of the same family,
read from the twin on 2026-10-08 (snippets are the stored ones; no vectors)."""
from __future__ import annotations

import json
import pathlib

import pytest

from cytir.distribute import pack
from cytir.index import point_delete as PD
from cytir.index import promote_queue as Q

CD, BC = Q.CANONICAL_DUPLICATE, Q.BINARY_CONTENT
PAGE = "https://ai.google.dev/gemini-api/docs/function-calling"
FIX = [json.loads(l) for l in (pathlib.Path(__file__).parent / "fixtures" / "prose_guard"
                               / "real_rows.jsonl").read_text(encoding="utf-8").splitlines()]
BIN = [r for r in FIX if r["kind"].startswith("bin:")][:2]
PROSE = [r for r in FIX if r["kind"] == "ctl:prose"][:2]


def row(i, url, prov="web", digest="dP", **kw):
    return {"id": f"00000000-0000-5000-8000-{i:012d}",
            "payload": {"url": url, "provenance": prov, "source_type": "official",
                        "content_digest": digest, **kw}}


CANON = row(1, PAGE)
VARS = [row(2, PAGE + "?authuser=1"), row(3, PAGE + "?authuser=2"), row(4, PAGE + "?authuser=0")]


class FakeTwin:
    def __init__(self, rows):
        self.rows = {r["id"]: dict(r["payload"]) for r in rows}

    def api(self, url, path, body=None, method=None):
        if path.endswith("/points/scroll"):
            vals = set(body["filter"]["must"][0]["match"]["any"])
            return {"points": [{"id": i, "payload": dict(p)} for i, p in self.rows.items()
                               if p.get("url") in vals], "next_page_offset": None}
        keep = body["with_payload"]
        return [{"id": i, "payload": (dict(self.rows[i]) if keep is True else
                                      {k: v for k, v in self.rows[i].items() if keep and k in keep})}
                for i in body["ids"] if i in self.rows]


@pytest.fixture
def twin(monkeypatch):
    t = FakeTwin([CANON] + VARS)
    monkeypatch.setattr(Q.PSP, "_api", t.api)
    monkeypatch.setattr(PD, "_api", t.api)
    return t


def tomb(r, reason):
    return {"id": r["id"], "url": r["payload"]["url"], "reason": reason}


def validate(tombs, export_ids=None):
    return Q.validate_tombstones(tombs, pack_urls=set(), target_url="http://twin",
                                 collection="aw_e8", export_ids=export_ids or {t["id"] for t in tombs})


def test_the_reasons_are_the_queues_and_never_a_takedown():
    # the PUBLIC vocabulary (cytir.distribute.pack) is untouched: an ADR amendment for Tim
    for r in (CD, BC):
        assert r in Q.TOMBSTONE_REASONS and r not in pack.TOMBSTONE_REASONS
        assert r not in pack.TAKEDOWN_REASONS
    assert set(pack.TOMBSTONE_REASONS) <= set(Q.TOMBSTONE_REASONS)


def test_variants_with_a_served_canonical_row_pass(twin):
    assert validate([tomb(v, CD) for v in VARS])["by_reason"] == {CD: 3}


def test_the_canonical_row_itself_is_refused(twin):
    with pytest.raises(Q.QueueError, match="already canonical"):
        validate([tomb(CANON, CD)])


def test_no_canonical_row_means_the_page_would_be_emptied(twin):
    twin.rows.pop(CANON["id"])
    with pytest.raises(Q.QueueError, match="no surviving row"):
        validate([tomb(v, CD) for v in VARS])
    # keeping one variant is not enough either: a variant is not canonical
    with pytest.raises(Q.QueueError, match="no surviving row"):
        validate([tomb(v, CD) for v in VARS[:2]])


def test_a_canonical_row_of_another_family_does_not_count(twin):
    twin.rows[CANON["id"]]["provenance"] = "other-family"
    with pytest.raises(Q.QueueError, match="no surviving row"):
        validate([tomb(v, CD) for v in VARS])


def test_a_canonical_row_in_the_tombstone_set_does_not_count(twin):
    with pytest.raises(Q.QueueError):
        validate([tomb(v, CD) for v in VARS] + [tomb(CANON, "takedown")])


def test_a_functional_param_variant_is_a_different_page_and_has_no_survivor(twin):
    ru = row(9, PAGE + "?authuser=1&hl=ru")
    twin.rows[ru["id"]] = dict(ru["payload"])
    with pytest.raises(Q.QueueError, match="no surviving row"):
        validate([tomb(ru, CD)])                    # PAGE?hl=ru is not served: only PAGE is


def test_binary_content_is_re_derived_from_the_stored_snippet(monkeypatch):
    rows = [row(20 + i, b["url"], "weko_docs", snippet=b["snippet"]) for i, b in enumerate(BIN)]
    rows += [row(30 + i, p["url"], "weko_docs", snippet=p["snippet"]) for i, p in enumerate(PROSE)]
    t = FakeTwin(rows)
    monkeypatch.setattr(Q.PSP, "_api", t.api)
    monkeypatch.setattr(PD, "_api", t.api)
    assert validate([tomb(r, BC) for r in rows[:2]])["by_reason"] == {BC: 2}
    with pytest.raises(Q.QueueError, match="reads as prose"):
        validate([tomb(rows[2], BC)])               # a prose row is never deleted as binary


def test_a_pack_holds_one_reason_and_no_points():
    Q.check_superseded_generation_pack({}, [], [tomb(VARS[0], CD)])
    Q.check_superseded_generation_pack({}, [], [tomb(VARS[0], BC)])
    with pytest.raises(Q.QueueError, match="only that reason"):
        Q.check_superseded_generation_pack({}, [], [tomb(VARS[0], CD), tomb(VARS[1], BC)])
    with pytest.raises(Q.QueueError, match="carries no points"):
        Q.check_superseded_generation_pack({}, [{"id": "p"}], [tomb(VARS[0], BC)])


def test_composer_classify_applies_the_drains_rule_and_counts_what_it_drops(twin):
    from cytir.ops import index_text_tombstones as ITT
    twin.rows[CANON["id"]]["domain"] = "ai.google.dev"
    for v in VARS:
        twin.rows[v["id"]]["domain"] = "ai.google.dev"
    cands = [{"id": v["id"]} for v in VARS] + [{"id": CANON["id"]},
                                                {"id": "00000000-0000-5000-8000-0000000000ff"}]
    res = ITT.classify(CD, cands, "http://twin", "aw_e8")
    assert [t["id"] for t in res["pass"]] == [v["id"] for v in VARS]
    assert res["absent"] == 1 and list(res["failed"]) == [CANON["id"]]
    s = ITT.summary(CD, res, len(cands))
    assert s["tombstones"] == 3 and s["twin_count_delta"] == -3 and s["refused_by_the_drains_rule"] == 1
    with pytest.raises(SystemExit):
        ITT.classify("takedown", cands, "http://twin", "aw_e8")


def test_write_pack_carries_reason_and_export_and_zero_points(tmp_path):
    from cytir.ops import spam_tombstones as ST
    tombs = [{"id": v["id"], "url": v["payload"]["url"], "reason": CD, "domain": "d", "detail": "x"}
             for v in VARS]
    m = ST.write_pack(str(tmp_path), "p-001", tombs, family="f", collection="c", note="n", reason=CD,
                      task=767, extra={"export": {"path": "/x", "sha256": "0" * 64, "rows": 3}})
    assert m["tombstones_by_reason"] == {CD: 3} and m["rows"] == 0 and m["task"] == 767
    assert m["export"]["rows"] == 3
    # the default stays the spam pack of task 701
    m2 = ST.write_pack(str(tmp_path), "p-002", [dict(tombs[0], reason="takedown")], family="f",
                       collection="c", note="n")
    assert m2["tombstones_by_reason"] == {"takedown": 1} and m2["task"] == 701


def test_a_page_only_ever_captured_as_variants_keeps_one_spelling(twin):
    from cytir.ops import index_text_tombstones as ITT
    hl = PAGE + "?hl=ko"
    a = [row(40, hl + "&authuser=1", digest="dA"), row(41, hl + "&authuser=1", digest="dB")]  # two chunks
    b = [row(42, hl + "&authuser=2", digest="dA"), row(43, hl + "&authuser=3", digest="dB")]
    for r in a + b:
        twin.rows[r["id"]] = dict(r["payload"], domain="ai.google.dev")
    res = ITT.classify(CD, [{"id": r["id"]} for r in a + b], "http://twin", "aw_e8")
    assert sorted(t["id"] for t in res["pass"]) == sorted(r["id"] for r in b)   # authuser=1 stays
    assert {t["survivor_url"] for t in res["pass"]} == {hl + "&authuser=1"}
    assert res["failed"] == {}


def test_a_declared_survivor_that_does_not_serve_is_refused(twin):
    hl = PAGE + "?hl=ko"
    victim = row(50, hl + "&authuser=2")
    twin.rows[victim["id"]] = dict(victim["payload"])
    t = dict(tomb(victim, CD), survivor_url=hl + "&authuser=1")                 # nothing serves it
    with pytest.raises(Q.QueueError, match="declared survivor_url"):
        validate([t])
    keeper = row(51, hl + "&authuser=1")
    twin.rows[keeper["id"]] = dict(keeper["payload"])
    assert validate([t])["by_reason"] == {CD: 1}
    # the survivor must be the declared spelling: another variant does not count
    other = row(52, hl + "&authuser=9")
    twin.rows.pop(keeper["id"])
    twin.rows[other["id"]] = dict(other["payload"])
    with pytest.raises(Q.QueueError, match="declared survivor_url"):
        validate([t])


def test_a_row_whose_text_the_survivor_does_not_hold_stays(twin):
    """Task 773: 631 of the first 11,441 rows had a survivor without their text."""
    from cytir.ops import index_text_tombstones as ITT
    extra = row(60, PAGE + "?authuser=7", digest="dOther")           # a chunk the canonical row lacks
    twin.rows[extra["id"]] = dict(extra["payload"], domain="ai.google.dev")
    for v in VARS:
        twin.rows[v["id"]]["domain"] = "ai.google.dev"
    res = ITT.classify(CD, [{"id": r["id"]} for r in VARS + [extra]], "http://twin", "aw_e8")
    assert sorted(t["id"] for t in res["pass"]) == sorted(v["id"] for v in VARS)
    assert "text not on the canonical row" in res["failed"][extra["id"]]
    # and the DRAIN enforces it, whatever a pack says
    with pytest.raises(Q.QueueError, match="same content_digest"):
        validate([tomb(extra, CD)])


def test_a_row_without_a_content_digest_is_refused(twin):
    twin.rows[VARS[0]["id"]].pop("content_digest")
    with pytest.raises(Q.QueueError, match="no `content_digest`"):
        validate([tomb(VARS[0], CD)])


def test_the_kept_spelling_is_the_one_that_covers_the_others(twin):
    from cytir.ops import index_text_tombstones as ITT
    hl = PAGE + "?hl=ja"
    full = [row(70 + i, hl + "&authuser=1", digest=f"d{i}") for i in range(3)]
    part = [row(80 + i, hl + "&authuser=0", digest=f"d{i}") for i in range(2)]   # fewer rows, subset
    for r in full + part:
        twin.rows[r["id"]] = dict(r["payload"], domain="x")
    res = ITT.classify(CD, [{"id": r["id"]} for r in full + part], "http://twin", "aw_e8")
    assert sorted(t["id"] for t in res["pass"]) == sorted(r["id"] for r in part)
    assert {t["survivor_url"] for t in res["pass"]} == {hl + "&authuser=1"}


def test_binary_content_only_on_a_binary_file_url():
    assert Q.is_binary_file_url("https://www.weko.admin.ch/dam/de/sd-web/x/a%20b.pdf")
    assert Q.is_binary_file_url("https://www.formazioneprof.ch/it/documenti/x")
    assert Q.is_binary_file_url("https://x.ch/a.PDF?download=1")
    for u in ("https://math.stackexchange.com/a/3564980", "https://www.rieden.ch/index.php/x",
              "https://de.wikipedia.org/wiki/Liste_von_Kartoffelsorten", "https://x.ch/pdf/info", None):
        assert not Q.is_binary_file_url(u)


def test_binary_content_refuses_a_row_on_a_page_that_is_not_a_file(monkeypatch):
    b = BIN[0]
    r = row(90, "https://www.rieden.ch/index.php/x", "web", snippet=b["snippet"])
    t = FakeTwin([r])
    monkeypatch.setattr(Q.PSP, "_api", t.api)
    monkeypatch.setattr(PD, "_api", t.api)
    with pytest.raises(Q.QueueError, match="not a binary file"):
        validate([tomb(r, BC)])
