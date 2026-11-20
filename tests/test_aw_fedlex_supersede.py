"""The fedlex url-group supersede (task 1038, RULING 52 + 53; mechanism of ADR-067 am.1 point 4, task 876).

A fedlex refresh replaces a whole act in one language: the door writes windows (`#art_14-c1`,
`-c2`) for an article the twin serves at the bare `#art_14`, and the served rows carry no
`provenance`. So the family is told by its URL scope (host `fedlex.admin.ch`, WHOLE path
`/eli/<eli>/<lang>`) and the group is the act-language.

`fixtures/fedlex_2022_491_scope.json` holds REAL records read on 2026-10-22 (the build host, twin
`aw_e10_twin`, keyed url filter, read-only): the 322 urls the door writes for `cc/2022/491`
(de fr it en) and the 308 twin rows (id + url) served at that act's anchors. The simulated twin
below serves exactly those, plus real rows of other acts from `fedlex_pack_v3_twin_ids.json`.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.core.ids import mint_point_id  # noqa: E402
from cytir.index import compose  # noqa: E402
from cytir.index import promote_queue as Q  # noqa: E402
from cytir.index import supersede as US  # noqa: E402
from cytir.index import url_group_supersede as UGS  # noqa: E402
from cytir.sources import staged_text_guard as STG  # noqa: E402
from cytir.sources.families import fedlex as FX  # noqa: E402
from cytir.sources.families import fedlex_contract as FC  # noqa: E402

ACT = json.loads((HERE / "fixtures" / "fedlex_2022_491_scope.json").read_text(encoding="utf-8"))
OTHER_ACTS = json.loads((HERE / "fixtures" / "fedlex_pack_v3_twin_ids.json").read_text(encoding="utf-8"))["rows"]
BASE = "https://www.fedlex.admin.ch/eli/cc/2022/491"


def _payload(url: str) -> dict:
    # a served fedlex row as read on the twin: `domain`, `source_type`, `prov_id`, NO `provenance`
    return {"url": url, "domain": "fedlex.admin.ch", "source_type": "official", "prov_id": 8}


def _pack(urls) -> dict:
    """Pack rows from urls (synthetic id per url) or from REAL `(url, point_id)` pairs: since RULING 55
    (task 1040) the windows of one article share its bare url, so only real ids keep them apart."""
    out = {}
    for u in urls:
        url, pid = (u, mint_point_id(u, "pack")) if isinstance(u, str) else (u[0], u[1])
        out[pid] = {"url": url, "domain": "fedlex.admin.ch", "source_type": "official",
                    "license": "public-domain"}
    return out


def _served() -> list:
    """The simulated twin: the 308 real rows of the act, real rows of other acts, a distractor act
    whose number starts with the act's (`cc/2022/4910`), and an entity card at an act url."""
    rows = [(i, _payload(u)) for u, i in ACT["twin"]]
    rows += [(r["twin_id"], _payload(r["url"])) for r in OTHER_ACTS if "cc/2022/491/" not in r["url"]]
    rows += [("00000000-0000-5000-8000-0000000049a0", _payload(f"{BASE[:-3]}4910/fr#art_1")),
             ("00000000-0000-5000-8000-0000000049a1", _payload(f"{BASE}/fr#art_1"))]
    rows[-1][1].update({"source_type": "entity_card", "entity_kind": "org"})
    return rows


def _install_twin(monkeypatch, served, *, leaky=False):
    """`served_group_members` against the simulated twin. Exact: only rows at an asked url (and its
    spellings), as the real keyword index answers. Leaky: every row that shares the act path
    TEXT with an asked url (what a prefix/text index would return), to prove the rule's own
    scope test is what keeps another act or language out."""
    def members(_twin, _collection, urls):
        asked = set()
        for u in urls:
            asked |= US.url_variants(u)
        prefixes = {u.split("#")[0].rsplit("/", 1)[0] for u in asked}
        out: dict = {}
        for pid, pl in served:
            hit = (any(pl["url"].startswith(p) for p in prefixes) if leaky else pl["url"] in asked)
            if hit:
                out.setdefault(US.payload_group_key(pl), []).append((pid, pl))
        return out
    monkeypatch.setattr(Q, "served_group_members", members)


def _victims(pack: dict) -> dict:
    rule = UGS.family_provenance({"family": "fedlex_exact"})
    groups, cards = UGS.pack_groups(pack, rule)
    return UGS.compute(pack, groups, cards, rule, twin_url="http://twin.invalid", collection="aw_e10_twin")


# ------------------------------------------------------------------ the scope
@pytest.mark.parametrize("url,want", [
    ("https://www.fedlex.admin.ch/eli/cc/2022/491/fr#art_24", ("cc/2022/491", "fr")),
    ("https://fedlex.admin.ch/eli/cc/2022/491/fr", ("cc/2022/491", "fr")),
    ("http://www.fedlex.admin.ch/eli/cc/27/317_321_377/en#art_341-c2", ("cc/27/317_321_377", "en")),
    ("https://www.fedlex.admin.ch/eli/cc/2022/4910/fr#art_1", ("cc/2022/4910", "fr")),
    ("https://www.fedlex.admin.ch/eli/cc/2022/491/rm#art_1", None),          # not a language we run
    ("https://www.fedlex.admin.ch/eli/cc/2022/491#art_1", None),             # no language segment
    ("https://fedlex.data.admin.ch/eli/cc/2022/491/fr#art_1", None),         # another host
    ("https://www.fedlex.admin.ch/eli/cc/2022/491/fr/extra#art_1", None),    # the path is matched whole
    ("https://example.org/eli/cc/2022/491/fr#art_1", None),
    ("", None), (None, None)])
def test_the_act_scope_is_host_plus_the_whole_eli_path_plus_language(url, want):
    assert FC.act_scope(url) == want


def test_anchors_split_into_base_and_window_and_coverage_compares_the_base():
    f = "https://www.fedlex.admin.ch/eli/cc/2022/491/fr"
    assert FC.split_anchor(f + "#art_14-c2") == (f, "art_14", 2)
    assert FC.split_anchor(f + "#art_14") == (f, "art_14", None)
    assert FC.split_anchor(f + "#art_656_b-c1") == (f, "art_656_b", 1)       # a lettered article
    assert FC.split_anchor(f) == (f, None, None)
    assert FC.coverage_url(f + "#art_14-c1") == FC.coverage_url(f + "#art_14") == f + "#art_14"


# ------------------------------------------------------------------ the victim set, real records
def test_the_fixture_is_the_dry_run_of_the_act():
    """The 322 rows of the task 1040 door run (RULING 55: bare urls) and the 308 twin rows."""
    assert len(ACT["pack"]) == 322 and len(ACT["twin"]) == 308
    assert len({i for _u, i in ACT["pack"]}) == 322                         # 322 distinct ids
    assert not any(FC.split_anchor(u)[2] for u, _i in ACT["pack"])          # no `-cK` url any more
    per_url: dict = {}
    for u, _i in ACT["pack"]:
        per_url[u] = per_url.get(u, 0) + 1
    assert len(per_url) == 308                                  # one url per article
    assert sorted(n for n in per_url.values() if n > 1) == [2] * 14   # 14 articles as 28 windows
    assert not any(FC.split_anchor(u)[2] for u, _i in ACT["twin"])         # the twin holds bare urls
    assert len({i for _u, i in ACT["twin"]}) == 308


def test_the_victims_for_cc_2022_491_are_exactly_the_308_twin_rows(monkeypatch):
    """Every one of the 308 twin urls is a pack url now (294 one-row articles + 14 windowed articles
    at their bare url, RULING 55). Nothing of another act, another language of another act, or a card."""
    _install_twin(monkeypatch, _served())
    pack = _pack(ACT["pack"])
    plan = _victims(pack)
    assert plan["ids"] == sorted(i for _u, i in ACT["twin"])
    assert len(plan["ids"]) == 308 and plan["groups"] == 4 and plan["superseded_groups"] == 4
    assert plan["pack_rows"] == 322 and plan["kept_other_family"] == 0
    pack_urls = {u for u, _i in ACT["pack"]}
    assert sum(1 for u, _i in ACT["twin"] if u in pack_urls) == 308
    others = {r["twin_id"] for r in OTHER_ACTS if "cc/2022/491/" not in r["url"]}
    assert not others & set(plan["ids"])
    assert "00000000-0000-5000-8000-0000000049a1" not in plan["ids"]        # the card stays


def test_a_rerun_of_the_same_pack_deletes_nothing(monkeypatch):
    """The twin already holds the pack's own ids (a second refresh of an unchanged act): no victim."""
    pack = _pack(ACT["pack"])
    served = [(pid, _payload(pl["url"])) for pid, pl in pack.items()]
    _install_twin(monkeypatch, served)
    assert _victims(pack)["ids"] == []


def test_the_victim_rule_survives_a_twin_that_answers_by_text_prefix(monkeypatch):
    """A French-only refresh against a twin that returns every row sharing the act path text: only
    the 77 French rows of cc/2022/491 leave; German, Italian, English, the act `cc/2022/4910` and the
    other acts stay."""
    fr = [(u, i) for u, i in ACT["pack"] if u.startswith(BASE + "/fr")]
    _install_twin(monkeypatch, _served(), leaky=True)
    plan = _victims(_pack(fr))
    want = sorted(i for u, i in ACT["twin"] if u.startswith(BASE + "/fr"))
    assert len(want) == 77 and plan["ids"] == want
    assert plan["kept_other_family"] > 0                       # the leaked rows were seen and kept


@pytest.mark.parametrize("name", ["drops_the_language", "matches_the_act_number_as_a_text_prefix"])
def test_a_mutant_that_widens_the_scope_is_caught(monkeypatch, name):
    """Mutation: the scope test loses its `<lang>` (or compares the act number as text). The same
    French-only refresh then deletes rows of other languages / of `cc/2022/4910`: the exact-set
    assertion of the previous test fails."""
    fr = [(u, i) for u, i in ACT["pack"] if u.startswith(BASE + "/fr")]
    want = sorted(i for u, i in ACT["twin"] if u.startswith(BASE + "/fr"))
    real = FC.act_scope

    def drops_the_language(url):
        sc = real(url)
        return None if sc is None else (sc[0], "*")

    def matches_the_act_number_as_a_text_prefix(url):
        return ("cc/2022/491", "*") if str(url).startswith(BASE) else None
    monkeypatch.setattr(FC, "act_scope", locals()[name])
    _install_twin(monkeypatch, _served(), leaky=True)
    got = _victims(_pack(fr))["ids"]
    assert got != want, "the widened scope must be visible to the exact-set check"
    assert set(got) > set(want)


# ------------------------------------------------------------------ windows
def test_a_stale_tail_window_is_found_by_chasing_the_contiguous_windows(monkeypatch):
    f = BASE + "/fr"
    served = [("a1", _payload(f + "#art_9")),                                      # was one row
              ("b1", _payload(f + "#art_14-c1")), ("b2", _payload(f + "#art_14-c2")),
              ("b3", _payload(f + "#art_14-c3")),                                  # the new text has two windows
              ("c1", _payload(f + "#art_30"))]                                     # not in the pack
    _install_twin(monkeypatch, served)
    pack = _pack([f + "#art_9-c1", f + "#art_9-c2", f + "#art_14-c1", f + "#art_14-c2"])
    plan = _victims(pack)
    assert plan["ids"] == ["a1", "b1", "b2", "b3"]          # bare row, and the tail window c3
    assert "c1" not in plan["ids"], "an article the pack does not write is not this pack's to delete"


def test_the_candidates_are_the_pack_url_its_base_anchor_and_the_first_window():
    rule = UGS.FEDLEX_SCOPE
    f = BASE + "/fr"
    assert rule.candidates([f + "#art_14-c2"]) == sorted([f + "#art_14-c2", f + "#art_14", f + "#art_14-c1"])
    assert rule.chase([f + "#art_14-c2", f + "#art_9"], {f + "#art_14-c3"}) == []
    assert rule.chase([f + "#art_14-c2", f + "#art_9"], set()) == [f + "#art_14-c3"]


def test_a_pack_row_outside_the_family_scope_refuses_the_pack():
    pack = _pack([BASE + "/fr#art_1"]) | {"x": {"url": "https://www.admin.ch/eli/cc/2022/491/fr#art_1"}}
    with pytest.raises(Q.QueueError, match="outside the family scope"):
        UGS.pack_groups(pack, UGS.FEDLEX_SCOPE)


def test_the_word_coverage_guard_compares_windows_with_the_bare_row_they_replace():
    """Skipped urls are keyed on the BASE anchor: both windows are held out together with the bare
    row they would have replaced (a window beside the old row is the doubled url the supersede
    exists to prevent)."""
    f = BASE + "/fr"
    rows = _pack([f + "#art_14-c1", f + "#art_14-c2", f + "#art_15"])
    comp = {"ids": ["old14", "old15"], "urls": {"old14": f + "#art_14", "old15": f + "#art_15"}}
    key = UGS.document_url_key(f + "#art_14")
    cov = {"skipped": [{"doc_key": key, "url": f + "#art_14", "ratio": 0.1}]}
    out = UGS.apply_skips(cov, rows, comp, True, FC.coverage_url)
    assert out["skipped_old_rows"] == 1 and comp["ids"] == ["old15"]
    held = {rows[p]["url"] for p in out["skipped_pack_ids"]}
    assert held == {f + "#art_14-c1", f + "#art_14-c2"}


# ------------------------------------------------------------------ registry, compose, door guards
def test_fedlex_is_on_the_url_group_allowlist_with_a_scope_rule_not_a_provenance():
    assert UGS.URL_GROUP_FAMILIES["fedlex_exact"] is UGS.FEDLEX_SCOPE
    assert UGS.family_provenance({"family": "fedlex_exact"}) is UGS.FEDLEX_SCOPE
    assert compose.check_supersede_options("fedlex_exact", supersede=UGS.URL_GROUP,
                                           allow_supersede="refresh", copy_sealed_guard=True) is True
    with pytest.raises(compose.ComposeError, match="refused for this family"):
        compose.check_supersede_options("fedlex_not_a_family", supersede=UGS.URL_GROUP,
                                        allow_supersede="x", copy_sealed_guard=True)


def test_the_door_gives_the_stage_guard_its_source_text_and_a_threshold_of_one():
    assert FX.FedlexFamily.staged_text_min_coverage == 1.0
    body = " ".join(f"w{i}" for i in range(700))                 # a long article: 3 windows
    windows = [t for _a, t in FX.window(body, "art_1")]
    assert len(windows) == 3
    assert STG.coverage(body, windows) == 1.0                    # a split costs nothing
    assert STG.coverage(body, windows[:-1]) < 1.0                # a lost tail window is caught
    assert STG.coverage(body, [w[: len(w) // 2] for w in windows]) < 1.0      # a cut inside a row too


def test_the_door_stamps_the_licence_the_contract_declares():
    assert FX.LICENSE == "public-domain"
    row = json.loads((REPO / "data" / "source-family-contracts.v1.json").read_text(encoding="utf-8"))
    crow = next(r for r in row["families"] if r["family_id"] == "fedlex_exact")
    assert crow["licence"]["spdx"] == FX.LICENSE
    assert crow["licence"]["basis"] == "statute" and crow["licence"]["inferred_from_hosting"] is False
    assert "mint_point_id" in crow["identity"]["minted_by"]
