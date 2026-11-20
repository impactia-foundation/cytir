"""ADR-076 dump-basis admission in `build_aw_b1.ConsentGate` (task 432).

ADR-076 (ratified Tim 2026-07-25) rules that an explicit open license on an
official bulk distribution is the admission basis (`search_ok`) for records
built from that distribution, *independent of the canonical host's robots/WAF
state*: "A robots-based decline of the host does NOT evict dump-based records;
conversely the dump basis extends only to what the license actually covers --
`train_ok` follows the license terms, not the admission ruling."

The gate did not implement it: `verdict()` never read `consent_state` in any
decision path, so a cached legifrance decline (ADR-073 robots-4xx) evicted the
DILA dump records exactly as the 318 build did before ADR-076 existed.

These tests also pin the task-318/319 absence-of-data ruling UNCHANGED. That
ruling -- eviction only on a resolved DECLINE or grace expiry, never on
absence-of-data -- is the more expensive invariant (the 318 gate evicted
1,793,172 points on absence-of-data), and a dump exemption placed carelessly
is exactly the kind of edit that re-opens it.

Run: python3 -m pytest tests/test_consent_dump_basis.py
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

from cytir.consent import build_gate as BG


ROOT = pathlib.Path(__file__).resolve().parent.parent
PIPELINE = ROOT / ("pipe" + "line")   # absent in the public cut; harmless on sys.path
sys.path.insert(0, str(PIPELINE))


from cytir.index import b1_rebuild as B1  # noqa: E402  (BASE_COLL, the builder's base collection name)
from cytir.index import b1_projection as BP  # noqa: E402

HOST = "www.legifrance.gouv.fr"

# The evidence shape the two live dump packs actually carry, read from the
# staging qdrant on 2026-08-19: aw_dfrlaw_cov (DILA / Etalab-2.0) and
# aw_dfedlex2_cov (Fedlex Akoma Ntoso / SwissGov-Fedlex-Terms). Both carry all
# four fields; the fixture is a copy of a real aw_dfrlaw_cov payload.
DUMP_PAYLOAD = {
    "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000051962161",
    "domain": "legifrance.gouv.fr",
    "license": "Etalab-2.0",
    "consent_state": "dump",
    "consent_basis": "official-dump-license",
    "consent_source": "dila-opendata echanges.dila.gouv.fr (Licence Ouverte v2.0)",
    "consent_checked_at": "2026-07-25",
}

# A resolved DECLINE per ADR-073: robots readable-and-disallowing, or a
# robots-4xx (legifrance WAF-403s robots from every vantage).
DECLINED_REC = {"status": "ok", "search_ok": False, "train_ok": False,
                "reason": "declined"}

STALE = "2026-06-01T00:00:00Z"     # older than the 30-day default grace
FRESH = "2026-08-18T00:00:00Z"     # inside it


def _gate(mode="cache", grace_days=30, cache=None, post_census=False):
    gate = BG.ConsentGate(mode=mode, grace_days=grace_days, cache_path="")
    gate.host_cache = dict(cache or {})
    gate.post_census = post_census
    return gate


def _corpus(search_ok=True, checked_at=FRESH):
    return {"consent_search_ok": search_ok, "consent_train_ok": True,
            "consent_checked_at": checked_at}


# ---------------------------------------------------------------------------
# 1. the defect: a dump-basis record must survive a canonical-host decline
# ---------------------------------------------------------------------------
def test_dump_basis_admits_with_fresh_stamp():
    """Baseline: a fresh stamp already admitted at HEAD. Unchanged."""
    gate = _gate(cache={HOST: DECLINED_REC})
    admit, reason = gate.verdict(HOST, _corpus(checked_at=FRESH),
                                 payload=DUMP_PAYLOAD)
    assert admit is True, reason


def test_dump_basis_admits_with_stale_stamp_and_cached_host_decline():
    """THE BUG (cache/live mode).

    Stale positive -> `verdict` re-resolved against the canonical HOST and
    returned an unconditional decline, evicting dump-route records. ADR-076
    says the host's robots state does not govern them.
    """
    gate = _gate(mode="cache", cache={HOST: DECLINED_REC})
    admit, reason = gate.verdict(HOST, _corpus(checked_at=STALE),
                                 payload=DUMP_PAYLOAD)
    assert admit is True, f"dump-basis record evicted by host decline: {reason}"
    assert "declined" not in reason


def test_dump_basis_admits_in_pipelined_mode_against_cached_decline():
    """THE BUG (pipelined mode) -- the route a full rebuild actually takes.

    `verdict` returns from the pipelined branch BEFORE it ever reaches the
    corpus-row block, and `_corpus_row_for` supplies no consent fields at all
    in a full rebuild, so here a cached decline evicts on its own -- stamp
    staleness is not even required. An exemption placed only ahead of the
    stale-positive branch would never fire on this path.
    """
    gate = _gate(mode="pipelined", cache={HOST: DECLINED_REC}, post_census=True)
    admit, reason = gate.verdict(HOST, None, payload=DUMP_PAYLOAD)
    assert admit is True, f"dump-basis record evicted by host decline: {reason}"
    assert "declined" not in reason


def test_dump_basis_survives_a_live_resolved_decline_in_corpus_branch():
    """Corpus row itself carrying search_ok=False must not evict a dump record."""
    gate = _gate(mode="cache", cache={HOST: DECLINED_REC})
    admit, reason = gate.verdict(HOST, _corpus(search_ok=False),
                                 payload=DUMP_PAYLOAD)
    assert admit is True, reason


# ---------------------------------------------------------------------------
# 2. no laundering: the stamp is not a magic word
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mutation,label", [
    ({"consent_state": None}, "no consent_state"),
    ({"consent_state": "grace"}, "grace, not dump"),
    ({"consent_basis": None}, "no consent_basis"),
    ({"consent_basis": "trust-me"}, "unrecognized consent_basis"),
    ({"consent_source": None}, "no consent_source"),
    ({"consent_source": "  "}, "blank consent_source"),
    ({"license": None}, "no explicit license"),
])
def test_incomplete_dump_evidence_still_declines(mutation, label):
    """ADR-076 admits on a LICENSE GRANT of an official bulk distribution, so
    the gate requires the evidence of that grant. A bare `consent_state=dump`
    must never launder an ordinary robots decline."""
    payload = {**DUMP_PAYLOAD, **mutation}
    payload = {k: v for k, v in payload.items() if v is not None}
    gate = _gate(mode="cache", cache={HOST: DECLINED_REC})
    admit, reason = gate.verdict(HOST, _corpus(checked_at=STALE),
                                 payload=payload)
    assert admit is False, f"{label} was laundered into an admit: {reason}"
    assert "declined" in reason


def test_ordinary_record_with_stale_stamp_and_host_decline_still_declines():
    gate = _gate(mode="cache", cache={HOST: DECLINED_REC})
    admit, reason = gate.verdict(HOST, _corpus(checked_at=STALE), payload={})
    assert admit is False
    assert "declined" in reason
    assert gate.stats["declined"] == 1


def test_ordinary_record_declines_in_pipelined_mode():
    gate = _gate(mode="pipelined", cache={HOST: DECLINED_REC}, post_census=True)
    admit, reason = gate.verdict(HOST, None, payload={"url": "x"})
    assert admit is False
    assert "declined" in reason


def test_missing_payload_argument_keeps_prior_behavior():
    """`payload` is optional; omitting it must reproduce HEAD exactly."""
    gate = _gate(mode="cache", cache={HOST: DECLINED_REC})
    assert gate.verdict(HOST, _corpus(checked_at=STALE))[0] is False


# ---------------------------------------------------------------------------
# 3. ADR-076's bound: train_ok follows the licence, not the admission ruling
# ---------------------------------------------------------------------------
def test_admission_never_grants_train_ok():
    """The gate decides `search_ok` only. It must not write, infer or upgrade
    `train_ok` -- ADR-076 bounds the dump basis to what the licence covers."""
    payload = {**DUMP_PAYLOAD, "train_ok": False, "search_ok": False}
    before = dict(payload)
    gate = _gate(mode="cache", cache={HOST: DECLINED_REC})
    admit, _ = gate.verdict(HOST, _corpus(checked_at=STALE), payload=payload)
    assert admit is True
    assert payload == before, "gate mutated the record payload"
    assert payload["train_ok"] is False


def test_dump_admission_does_not_touch_a_train_ok_false_licence():
    """A dump whose licence forbids training keeps train_ok False while still
    being admitted for search (e.g. the ADR-082 VKF/SIA class)."""
    payload = {**DUMP_PAYLOAD, "license": "SwissGov-Fedlex-Terms",
               "train_ok": False}
    gate = _gate(mode="pipelined", cache={HOST: DECLINED_REC}, post_census=True)
    admit, reason = gate.verdict(HOST, None, payload=payload)
    assert admit is True
    assert payload["train_ok"] is False
    assert "train" not in reason


# ---------------------------------------------------------------------------
# 4. task-318/319 absence-of-data ruling: UNCHANGED
#    (the regression that matters most -- the 318 gate evicted 1,793,172
#     points on absence-of-data; 319 ruled eviction happens only on a resolved
#     DECLINE or grace expiry)
#
# These call `verdict(host, row)` POSITIONALLY, with no payload argument, so
# they exercise byte-identical code on HEAD and on the fixed builder and PASS
# ON BOTH. That is what "unchanged" has to mean: a regression guard that only
# failed on HEAD because the signature grew would prove nothing about the
# ruling it claims to protect. The dump exemption's discrimination is carried
# by sections 1-3, which do pass a payload.
# ---------------------------------------------------------------------------
# (host_cache, host, expected_admit, expected_reason_prefix, expected_stat)
ABSENCE_OF_DATA_TABLE = [
    # 319 fix 2: unreachable != declined (bfs.admin.ch 160,186 -> 0 in 318)
    ({"bfs.admin.ch": {"status": "unreachable", "search_ok": True,
                       "reason": "unreachable", "canary_ok": True}},
     "bfs.admin.ch", True, "grace:unreachable", "unreachable_grace"),
    # resolver error is absence-of-data, not a decline
    ({"broken.example": {"status": "unknown", "search_ok": None,
                         "reason": "error"}},
     "broken.example", True, "grace:unresolved", "unresolved_grace"),
    # `resolve_host` leaves search_ok=True on an unreachable host: admitting
    # that as a real `ok:` verdict would fail OPEN. Grace only.
    ({"h.example": {"status": "unreachable", "search_ok": True,
                    "reason": "unreachable"}},
     "h.example", True, "grace:", "unresolved_grace"),
    # a real robots fetch is the ONLY live_ok admit
    ({"good.example": {"status": "ok", "search_ok": True, "reason": "ok"}},
     "good.example", True, "ok:", "live_ok"),
    # ADR-073 resolved decline still evicts
    ({"bad.example": {"status": "ok", "search_ok": False,
                      "reason": "declined"}},
     "bad.example", False, "declined", "declined"),
]


@pytest.mark.parametrize("cache,host,admit_exp,prefix,stat",
                         ABSENCE_OF_DATA_TABLE)
def test_absence_of_data_ruling_unchanged(cache, host, admit_exp, prefix, stat):
    gate = _gate(mode="cache", cache=cache)
    admit, reason = gate.verdict(host, None)
    assert admit is admit_exp, reason
    assert reason.startswith(prefix) or prefix in reason, reason
    assert gate.stats[stat] == 1
    if admit_exp and prefix.startswith("grace"):
        assert host in gate.retry_hosts


def test_post_census_cache_miss_grace_retains():
    """319 fix 1: census-absent host -> grace-retain, never exclude.
    (The 318 build excluded 100% of 929,826 uncensused points.)"""
    gate = _gate(mode="pipelined", cache={}, post_census=True)
    admit, reason = gate.verdict("nowhere.example", None)
    assert admit is True
    assert reason == "grace:uncensused"
    assert gate.stats["uncensused_grace"] == 1
    assert "nowhere.example" in gate.retry_hosts


def test_pre_census_cache_miss_defers_not_excludes():
    """A pipelined main-pass miss is a DEFERRAL, not an exclusion."""
    gate = _gate(mode="pipelined", cache={}, post_census=False)
    admit, reason = gate.verdict("nowhere.example", None)
    assert admit is False
    assert reason.startswith("defer:")
    assert gate.stats["deferred"] == 1


def test_trust_mode_still_allows_with_flag():
    gate = _gate(mode="trust", cache={})
    admit, reason = gate.verdict("unknown.example", None)
    assert admit is True
    assert gate.stats["unresolved_allowed"] == 1


def test_declined_counter_is_truthful_and_admits_never_say_declined():
    """318's counters lied (STATE said consent_excluded: 0 while 1.79M were
    excluded). No admitted class may increment `declined` or carry the
    substring in its reason -- including the new dump class."""
    gate = _gate(mode="cache", cache={HOST: DECLINED_REC})
    admits = [
        gate.verdict(HOST, _corpus(checked_at=STALE), payload=DUMP_PAYLOAD),
        gate.verdict("good.example", _corpus(checked_at=FRESH)),
    ]
    for admit, reason in admits:
        assert admit is True
        assert "declined" not in reason
    assert gate.stats["declined"] == 0
    admit, reason = gate.verdict(HOST, _corpus(checked_at=STALE))
    assert admit is False and "declined" in reason
    assert gate.stats["declined"] == 1


# ---------------------------------------------------------------------------
# 5. a NON-dump payload must not perturb any pre-existing verdict
#    (the exemption is admit-only and evidence-gated; every other class must
#     land exactly where it landed before the payload argument existed)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cache,host,_a,_p,_s", ABSENCE_OF_DATA_TABLE)
@pytest.mark.parametrize("payload", [
    None, {}, {"url": "u"},
    {"consent_state": "grace"},                       # a grace stamp
    {"consent_state": "dump"},                        # bare word, no evidence
    {"consent_state": "dump", "consent_basis": "official-dump-license"},
])
def test_non_dump_payload_does_not_change_any_verdict(cache, host, _a, _p, _s,
                                                      payload):
    baseline = _gate(mode="cache", cache=cache).verdict(host, None)
    withpl = _gate(mode="cache", cache=cache).verdict(host, None,
                                                      payload=payload)
    assert withpl == baseline


# ---------------------------------------------------------------------------
# 6. task 468 / task 433 WI-2 — the projection must EMIT what the gate read
#
# The gate admitting a dump-route record is only half the exemption. The full
# rebuild re-projects its own previous output, so evidence the projection
# drops is evidence the NEXT rebuild cannot see: it re-resolves the canonical
# host, finds the cached decline, and evicts. `project_record` dropped exactly
# `consent_state` / `consent_basis` / `consent_source`, so the fuse re-armed
# one generation later. Two cohorts ride on the round trip: legifrance /
# `fr_dila_deep` (499,302 points, WI-1) and rii_courts (38,159, task 465).
# ---------------------------------------------------------------------------
from collections import defaultdict                                # noqa: E402

TEXT = ("Le présent article fixe les conditions d'application du chapitre "
        "relatif aux obligations du preneur.")


def _source(payload_extra=None, payload=None):
    """A twin base point as `project_record` receives it."""
    base = dict(DUMP_PAYLOAD if payload is None else payload)
    base.update(payload_extra or {})
    return SimpleNamespace(id="p1", vector={"dense": [0.1] * 8}, payload=base)


def _project(source, gate=None, stats=None):
    gate = gate or _gate(mode="cache", cache={HOST: DECLINED_REC})
    point, meta = BP.project_record(
        source, B1.BASE_COLL, {"texts": [TEXT]}, {}, {},
        gate, stats if stats is not None else defaultdict(int))
    return point, meta


def test_projection_emits_the_dump_stamp_it_was_admitted_on():
    """THE WI-2 DEFECT. All four fields dump_basis_evidence needs survive."""
    point, meta = _project(_source())
    assert point is not None, meta
    payload = point["payload"]
    assert payload["consent_state"] == "dump"
    assert payload["consent_basis"] == "official-dump-license"
    assert payload["consent_source"] == DUMP_PAYLOAD["consent_source"]
    assert payload["consent_checked_at"] == DUMP_PAYLOAD["consent_checked_at"]
    assert payload["license"] == "Etalab-2.0"


def test_projected_payload_survives_the_next_rebuilds_gate():
    """The round trip, end to end: feed this build's OUTPUT back to the gate
    against the same cached host decline. This is the property that was
    missing -- the pack collection stayed admissible while the base
    collection, one generation on, did not."""
    point, _meta = _project(_source())
    assert BG.dump_basis_evidence(point["payload"]) is not None
    regate = _gate(mode="cache", cache={HOST: DECLINED_REC})
    admit, reason = regate.verdict(HOST, _corpus(checked_at=STALE),
                                   payload=point["payload"])
    assert admit is True, reason
    # ...and again, so the projection is a fixed point rather than decaying.
    again, _ = _project(SimpleNamespace(id="p1", vector={"dense": [0.1] * 8},
                                        payload=dict(point["payload"])))
    assert again["payload"]["consent_state"] == "dump"
    assert BG.dump_basis_evidence(again["payload"]) is not None


def test_train_ok_false_rides_through_untouched():
    """rii carries `tdm-reservation: 1` -> train_ok=False is mandatory and the
    dump basis decides search_ok ONLY (ADR-076's explicit bound)."""
    point, _meta = _project(_source({"search_ok": True, "train_ok": False}))
    assert point["payload"]["train_ok"] is False
    assert point["payload"]["search_ok"] is True


@pytest.mark.parametrize("mutation,label", [
    ({"consent_state": None}, "no state"),
    ({"consent_basis": None}, "no basis"),
    ({"consent_basis": "we-said-so"}, "unrecognized basis"),
    ({"consent_source": None}, "no named distribution"),
    ({"license": None}, "no license"),
])
def test_incomplete_evidence_is_not_emitted_as_a_dump_stamp(mutation, label):
    """The laundering vector, closed on the way OUT as well as the way in: a
    payload the gate would not admit on must not leave the projection wearing
    a stamp that a later rebuild WOULD admit on."""
    payload = {k: v for k, v in {**DUMP_PAYLOAD, **mutation}.items()
               if v is not None}
    gate = SimpleNamespace(verdict=lambda *a, **k: (True, "ok"),
                           grace_until="2026-09-30")
    point, meta = _project(_source(payload=payload), gate=gate)
    assert point is not None, (label, meta)
    assert point["payload"].get("consent_state") != "dump", label
    assert BG.dump_basis_evidence(point["payload"]) is None, label


def test_a_stale_grace_stamp_is_not_resurrected():
    """Why the fix reads `dump_basis_evidence` instead of adding the three
    names to the passthrough tuple: a blanket passthrough would carry a
    previous build's `consent_state=grace` onto a record THIS build admitted
    normally -- without the `consent_grace_until` that bounds it."""
    stale_grace = {k: v for k, v in DUMP_PAYLOAD.items()
                   if not k.startswith("consent_")}
    stale_grace.update({"consent_state": "grace",
                        "consent_reason": "uncensused",
                        "consent_grace_until": "2026-08-24"})
    gate = SimpleNamespace(verdict=lambda *a, **k: (True, "corpus: fresh"),
                           grace_until="2026-09-30")
    point, _meta = _project(_source(payload=stale_grace), gate=gate)
    payload = point["payload"]
    assert "consent_state" not in payload
    assert "consent_grace_until" not in payload
    assert "consent_reason" not in payload


def test_this_builds_grace_stamp_is_still_written():
    """Task 319's grace retention is untouched by the new branch."""
    gate = SimpleNamespace(verdict=lambda *a, **k: (True, "grace:uncensused"),
                           grace_until="2026-09-30")
    ordinary = {k: v for k, v in DUMP_PAYLOAD.items()
                if not k.startswith("consent_")}
    point, _meta = _project(_source(payload=ordinary), gate=gate)
    assert point["payload"]["consent_state"] == "grace"
    assert point["payload"]["consent_grace_until"] == "2026-09-30"


def test_two_consent_states_on_one_payload_fail_loud():
    """Unreachable from ConsentGate (branch 0 admits on exactly the dump
    predicate, so a dump record never reaches the grace branch). If that ever
    stops being true, the build stops rather than emitting a payload whose two
    states disagree about which rule holds the record."""
    gate = SimpleNamespace(verdict=lambda *a, **k: (True, "grace:uncensused"),
                           grace_until="2026-09-30")
    with pytest.raises(ValueError, match="contradictory consent state"):
        _project(_source(), gate=gate)


def test_unknown_fields_are_still_dropped():
    """The default-exclude design is the point of the whitelist: this fix adds
    three names on validated evidence, it does not open the payload."""
    noise = {"internal_note": "ops scratch", "raw_html": "<p>x</p>",
             "consent_receipt_blob": {"robots": "Disallow: /"},
             "operator_email": "someone@example.test", "crawl_id": "CC-2026-30"}
    point, _meta = _project(_source(noise))
    assert not (set(noise) & set(point["payload"])), point["payload"]


# --- the build gate that keeps it from dropping out again -----------------
def _pt(dump_basis, payload):
    return {"dump_basis": dump_basis, "payload": payload}


def test_dump_basis_emission_check_passes_on_a_stamped_build():
    point, _meta = _project(_source())
    ok, detail = BP.evaluate_dump_basis_emission([point, _pt(False, {"url": "u"})])
    assert ok and detail["population"] == 1 and detail["unstamped"] == 0


def test_dump_basis_emission_check_catches_the_strip():
    """The regression this exists for: the gate admitted on the evidence, the
    payload went out without it. One point is enough to fail -- it is one
    point the next rebuild evicts."""
    stripped = {k: v for k, v in DUMP_PAYLOAD.items()
                if k not in ("consent_state", "consent_basis", "consent_source")}
    ok, detail = BP.evaluate_dump_basis_emission([_pt(True, stripped)])
    assert not ok and detail["unstamped"] == 1


def test_dump_basis_emission_check_catches_an_invented_stamp():
    ok, detail = BP.evaluate_dump_basis_emission(
        [_pt(False, {"url": "u", "consent_state": "dump"})])
    assert not ok and detail["stamped_without_source_evidence"] == 1
    assert detail["first_unsourced"] == ["u"]
