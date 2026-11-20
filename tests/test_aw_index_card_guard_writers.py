"""Task 820: the by-id writers outside the drain run the drain's card check before they write.

The L1-backfill failure overwrote six entity cards in place with shop text (same Tier C id). The
sealed-pack drain refuses that (`promote_queue.check_card_overwrites`); these writers wrote by id
without it. This module drives `cytir.index.apply` and `cytir.index.upsert` through their own entry points
with fake Qdrant clients. The two build-tree scripts have the same cases in their own module.

The ONE seam is `PSP.retrieve_payloads` (the indexed id retrieve the drain's check reads through),
answered per ENDPOINT url from the fake's stored rows, so a guard that read the wrong endpoint finds
no card and the refusal test fails. The check itself is the real `check_card_overwrites`.

Four cases per writer: stored card + non-card row at one id -> refused, nothing written, the text
names the writer, the id and the stored url; card over a card of overlapping identity -> written;
stored non-card -> written; sealed waiver -> written and logged.

`cytir.index.apply` is a twin-first transaction (review 823 M2): the guard runs in the PREFLIGHT, for
both targets, before the twin is written. Those tests drive the real `twin_first` with a preflight
and apply_target built from the real `guard_cards_before_writes` / `apply_elided_points`.

MUTATION: deleting the guard call in a writer fails that writer's refusal test (the
`guard_card_overwrites_for_input(` call in `upsert._run`; the `guard_cards_before_writes(` calls in
`apply.main`, pinned by the wiring test). Recorded in the 820 task doc.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import pathlib
import sys
import urllib.request
from types import SimpleNamespace

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.index import apply as AP  # noqa: E402
from cytir.index import card_waivers as CW  # noqa: E402
from cytir.index import prod_guard as PG  # noqa: E402
from cytir.index import promote_sealed_pack as PSP  # noqa: E402
from cytir.index import supersede as US  # noqa: E402
from cytir.index import upsert as UP  # noqa: E402
from cytir.index.twin_first import ApplyAbort, twin_first  # noqa: E402

PID = "11111111-0000-5000-8000-00000000000a"
STORED_CARD = {"url": "https://a.ch/", "source_type": "entity_card", "entity_kind": "company",
               "register_id": "CHE-100.000.001", "register_scheme": "CHE-UID", "title": "A AG"}
SHOP = {"url": "https://a.ch/", "source_type": "official", "title": "Shop"}
REFRESHED_CARD = {**STORED_CARD, "title": "A AG (refreshed)"}
STORED_PAGE = {"url": "https://a.ch/", "source_type": "official", "title": "old page"}
WAIVER_PR = "#820"
WAIVER_SHA = "c" * 64
TWIN, PROD = "http://twin:6333", "http://prod:6334"      # 6334 = the prod tunnel port (prod_guard)


def _waiver_file(tmp_path):
    row = {"id": PID, "url": "https://a.ch/", "wrong_card_identity": sorted(US.card_identity(STORED_CARD)),
           "reason": "task 820 test", "evidence_sha256": WAIVER_SHA, "approved_by": "tim",
           "approved_at": "2026-10-09", "pr": WAIVER_PR}
    body = (json.dumps({"schema": CW.WAIVER_SCHEMA}) + "\n" + json.dumps(row) + "\n").encode()
    path = tmp_path / "waivers.jsonl"
    path.write_bytes(body + CW.seal_line(body, 1).encode())
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def stored(monkeypatch):
    """`PSP.retrieve_payloads` answered per endpoint url: `stored.by_url[url][id] = payload`."""
    holder = SimpleNamespace(by_url={}, calls=[])

    def fake(url, collection, ids, fields, batch=500, workers=1):
        holder.calls.append((url, collection))
        rows = holder.by_url.get(url, {})
        return {i: dict(rows[i]) for i in ids if i in rows}

    monkeypatch.setattr(PSP, "retrieve_payloads", fake)
    return holder


class Fake:
    """The Qdrant client surface the writers use."""

    def __init__(self, url, points):
        self._client = SimpleNamespace(rest_uri=url)
        self.points = points
        self.writes: list[str] = []

    def upsert(self, collection, points, wait=False):
        for p in points:
            self.writes.append(str(p.id))
            self.points[str(p.id)] = p.payload

    def count(self, *a, **k):
        return SimpleNamespace(count=len(self.points))

    def retrieve(self, collection, ids, with_payload=True, with_vectors=True):
        return [SimpleNamespace(id=i, payload=self.points[i], vector={"dense": [0.1]})
                for i in ids if i in self.points]


def _assert_refusal_names(text: str, writer: str):
    assert writer in text
    assert PID in text
    assert "https://a.ch/" in text              # the stored card's url
    assert "would be overwritten" in text       # the drain's own reason text


def _ledger_rows(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------------------------
# cytir.index.apply: the real twin_first, the real guard + apply_elided_points
# ---------------------------------------------------------------------------------------------
class _Sup:
    def observe_payload(self, *a):
        pass

    def run(self):
        return {"superseded": 0}


class _Lock:
    def acquire(self):
        pass

    def release(self):
        pass


def run_apply(tmp_path, stored, *, twin_stored, prod_stored, new, armed=True, waivers=None):
    """Returns (error or None, twin client, prod client, ledger rows)."""
    stored.by_url[TWIN] = {PID: twin_stored} if twin_stored is not None else {}
    stored.by_url[PROD] = {PID: prod_stored} if prod_stored is not None else {}
    art = tmp_path / "points.jsonl"
    art.write_text(json.dumps({"id": PID, "vector": {"dense": [0.1]}, "payload": new}) + "\n")
    ledger = tmp_path / "ledger.jsonl"
    clients = {"twin": Fake(TWIN, dict(stored.by_url[TWIN])), "prod": Fake(PROD, dict(stored.by_url[PROD]))}
    collections = {"twin": "aw_e8", "prod": "aw_e9"}
    comparison = SimpleNamespace(write_ids={PID})
    guards: dict = {}

    def preflight():          # what apply.main's preflight() does with the guard
        guards.update(AP.guard_cards_before_writes(
            clients, collections, art, {"twin": comparison, "prod": comparison},
            waivers=waivers, ledger=ledger, pack="P"))
        return {}

    def apply_target(name, _state):     # what apply.main's apply_target() does
        AP.apply_elided_points(clients[name], collections[name], art, 10, comparison, _Sup(),
                               card_guard=guards.get(name))

    err = None
    try:
        twin_first(ledger, _Lock(), preflight=preflight, apply_target=apply_target,
                   verify_target=lambda n, s: {"ok": n}, armed=armed,
                   post_twin_gate=lambda s, t: {"status": "PASS"},
                   preservation_verifier=lambda: {"status": "PASS"})
    except (ApplyAbort, Exception) as exc:      # noqa: BLE001
        err = exc
    return err, clients["twin"], clients["prod"], _ledger_rows(ledger) if ledger.exists() else []


def test_apply_refuses_a_card_overwrite_on_both_targets_and_writes_nothing(tmp_path, stored):
    err, twin, prod, rows = run_apply(tmp_path, stored, twin_stored=STORED_CARD,
                                      prod_stored=STORED_CARD, new=SHOP)
    assert err is not None
    _assert_refusal_names(str(err), "cytir.index.apply")
    assert twin.writes == [] and prod.writes == []
    refused = [r for r in rows if r["status"] == "CARD-GUARD-REFUSED"]
    assert refused and refused[0]["writer"] == "cytir.index.apply" and refused[0]["target"] == "twin"
    assert PID in refused[0]["reason"] and "https://a.ch/" in refused[0]["reason"]


def test_apply_a_card_only_on_PROD_refuses_before_the_twin_is_written(tmp_path, stored):
    """Review 823 M2: the mirrors differ; the prod refusal must come before ANY write."""
    err, twin, prod, rows = run_apply(tmp_path, stored, twin_stored=STORED_PAGE,
                                      prod_stored=STORED_CARD, new=SHOP)
    assert err is not None
    _assert_refusal_names(str(err), "cytir.index.apply")
    assert twin.writes == [] and prod.writes == []
    refused = [r for r in rows if r["status"] == "CARD-GUARD-REFUSED"]
    assert [r["target"] for r in refused] == ["prod"]


def test_apply_dry_run_says_the_refusal_not_preflight_ok(tmp_path, stored):
    err, twin, prod, rows = run_apply(tmp_path, stored, twin_stored=STORED_CARD,
                                      prod_stored=STORED_CARD, new=SHOP, armed=False)
    assert err is not None and "would be overwritten" in str(err)
    assert "PREFLIGHT-OK" not in [r["status"] for r in rows]


def test_apply_under_replay_makes_zero_prod_calls_and_refuses_before_any_write(
        tmp_path, stored, monkeypatch):
    http = []

    def no_http(req, timeout=None):
        http.append(getattr(req, "full_url", req))
        raise AssertionError("no HTTP in this test")

    monkeypatch.setattr(urllib.request, "urlopen", no_http)
    PG.reset()
    monkeypatch.setenv("AW_REPLAY", "1")
    real = PSP.retrieve_payloads        # the stub of the `stored` fixture
    stub = real

    def routed(url, collection, ids, fields, batch=500, workers=1):
        if url.endswith(":6334"):        # prod: the REAL read path (prod_guard first)
            return _REAL_RETRIEVE(url, collection, ids, fields, batch, workers)
        return stub(url, collection, ids, fields, batch, workers)

    monkeypatch.setattr(PSP, "retrieve_payloads", routed)
    err, twin, prod, rows = run_apply(tmp_path, stored, twin_stored=STORED_PAGE,
                                      prod_stored=STORED_PAGE, new=SHOP)
    assert err is not None
    assert twin.writes == [] and prod.writes == []
    assert http == []                                   # zero prod HTTP requests
    assert "CARD-GUARD-ERROR" in [r["status"] for r in rows]
    PG.reset()


_REAL_RETRIEVE = PSP.retrieve_payloads


def test_apply_a_card_over_a_card_is_written_on_both_targets(tmp_path, stored):
    err, twin, prod, rows = run_apply(tmp_path, stored, twin_stored=STORED_CARD,
                                      prod_stored=STORED_CARD, new=REFRESHED_CARD)
    assert err is None and twin.writes == [PID] and prod.writes == [PID]
    assert [r["target"] for r in rows if r["status"] == "CARD-GUARD-OK"] == ["twin", "prod"]


def test_apply_a_stored_non_card_is_not_the_guards_business(tmp_path, stored):
    err, twin, prod, _ = run_apply(tmp_path, stored, twin_stored=STORED_PAGE,
                                   prod_stored=STORED_PAGE, new=SHOP)
    assert err is None and twin.writes == [PID] and prod.writes == [PID]


def test_apply_a_sealed_waiver_admits_the_overwrite_and_is_recorded(tmp_path, stored, capsys):
    path, sha = _waiver_file(tmp_path)
    err, twin, prod, rows = run_apply(tmp_path, stored, twin_stored=STORED_CARD,
                                      prod_stored=STORED_CARD, new=SHOP,
                                      waivers=US.load_card_waivers(path, sha))
    assert err is None and twin.writes == [PID] and prod.writes == [PID]
    assert f"cytir.index.apply: waiver admitted id {PID} pr={WAIVER_PR} evidence_sha256={WAIVER_SHA}" \
        in capsys.readouterr().out
    ok = [r for r in rows if r["status"] == "CARD-GUARD-OK"]
    assert all(r["card_guard"]["waived"][0]["pr"] == WAIVER_PR for r in ok)


def test_apply_elided_points_refuses_without_the_preflights_proof(tmp_path):
    art = tmp_path / "p.jsonl"
    art.write_text(json.dumps({"id": PID, "vector": {"dense": [0.1]}, "payload": SHOP}) + "\n")
    client = Fake(TWIN, {})
    for proof in (None, {}, {"collection": "another"}):
        with pytest.raises(ApplyAbort, match="no card-overwrite check ran"):
            AP.apply_elided_points(client, "aw_e8", art, 10, SimpleNamespace(write_ids={PID}), _Sup(),
                                   card_guard=proof)
    assert client.writes == []


def test_apply_main_runs_the_guard_in_preflight_and_in_recovery_preflight_not_in_apply_target():
    """Wiring (the house style for `main`'s closures): deleting either call fails here."""
    src = inspect.getsource(AP.main)
    assert src.count("guard_cards_before_writes(") == 2
    pre = src.split("def preflight():")[1].split("def apply_target(")[0]
    rec = src.split("def recovery_preflight():")[1].split("def recovery_gate(")[0]
    app = src.split("def apply_target(")[1].split("def verify_target(")[0]
    assert "guard_cards_before_writes(" in pre
    assert "guard_cards_before_writes(" in rec and '{"prod": clients["prod"]}' in rec
    assert "guard_cards_before_writes(" not in app and "card_guard=card_guards.get(name)" in app


# ---------------------------------------------------------------------------------------------
# cytir.index.upsert: _run, twin + prod
# ---------------------------------------------------------------------------------------------
def run_upsert(stored, tmp_path, monkeypatch, *, twin_stored, new, use_waiver=False):
    stored.by_url[TWIN] = {PID: twin_stored}
    stored.by_url[PROD] = {}              # a guard that read prod would find no card
    inp = tmp_path / "wave.jsonl"
    inp.write_text(json.dumps({"id": PID, "vector": {"dense": [0.1]}, "payload": new}) + "\n")
    twin, prod = Fake(TWIN, {PID: twin_stored}), Fake(PROD, {PID: twin_stored})
    monkeypatch.setattr(UP, "qdrant_connect", lambda *a, **k: prod)
    monkeypatch.setattr(UP, "LOCK", str(tmp_path / "lock"))
    monkeypatch.setattr(UP, "ram_guard", lambda *_: None)
    wpath, wsha = _waiver_file(tmp_path) if use_waiver else (None, None)
    args = SimpleNamespace(inp=str(inp), collection="aw_prod", qdrant_url=PROD, batch=10,
                           min_avail_gib=0.0, health_url="", health_every=20, purpose="t",
                           skip_counts=True, break_stale_lock=False, no_supersede=True,
                           create_url_index=False, card_waivers=wpath, card_waivers_sha256=wsha)
    ledger_lines: list[str] = []
    try:
        UP._run(args, ledger=lambda line, **k: ledger_lines.append(line), twin_client=twin,
                twin_coll="aw_e8")
    except SystemExit as exc:
        assert not twin.writes and not prod.writes
        return exc, [], ledger_lines
    assert twin.writes == prod.writes
    return None, twin.writes, ledger_lines


def test_upsert_refuses_a_card_overwrite_writes_nothing_and_ledgers_it(stored, tmp_path, monkeypatch):
    exc, writes, ledger = run_upsert(stored, tmp_path, monkeypatch, twin_stored=STORED_CARD, new=SHOP)
    assert exc is not None and writes == []
    _assert_refusal_names(str(exc), "cytir.index.upsert")
    assert any(line.startswith("CARD-GUARD-REFUSED") and PID in line and "https://a.ch/" in line
               for line in ledger)


def test_upsert_card_over_card_is_written(stored, tmp_path, monkeypatch):
    exc, writes, _ = run_upsert(stored, tmp_path, monkeypatch, twin_stored=STORED_CARD, new=REFRESHED_CARD)
    assert exc is None and writes == [PID]


def test_upsert_stored_non_card_is_not_the_guards_business(stored, tmp_path, monkeypatch):
    exc, writes, _ = run_upsert(stored, tmp_path, monkeypatch, twin_stored=STORED_PAGE, new=SHOP)
    assert exc is None and writes == [PID]


def test_upsert_waiver_admits_it_and_the_ledger_lists_the_waived_id(stored, tmp_path, monkeypatch, capsys):
    exc, writes, ledger = run_upsert(stored, tmp_path, monkeypatch, twin_stored=STORED_CARD, new=SHOP,
                                     use_waiver=True)
    assert exc is None and writes == [PID]
    assert f"cytir.index.upsert: waiver admitted id {PID} pr={WAIVER_PR}" in capsys.readouterr().out
    ok = [line for line in ledger if line.startswith("CARD-GUARD-OK")]
    assert ok and PID in ok[0] and "waived=" in ok[0]


def test_upsert_reads_the_twin_not_prod_when_there_is_a_twin(stored, tmp_path, monkeypatch):
    """No extra prod read: the twin mirrors prod and is written first."""
    run_upsert(stored, tmp_path, monkeypatch, twin_stored=STORED_PAGE, new=SHOP)
    assert stored.calls == [(TWIN, "aw_e8")]


# ---------------------------------------------------------------------------------------------
# the shared wrapper
# ---------------------------------------------------------------------------------------------
def test_a_failed_read_is_not_a_pass(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise PSP.PromotionError("twin down")
    monkeypatch.setattr(PSP, "retrieve_payloads", boom)
    with pytest.raises(PSP.PromotionError):
        US.guard_card_overwrites([{"id": PID, "payload": SHOP}], TWIN, "aw_e8", writer="w")


def test_a_client_with_no_known_endpoint_is_refused_not_skipped():
    with pytest.raises(US.CardOverwriteRefused, match="cannot tell which endpoint"):
        US.client_url(object())


def test_waiver_path_and_sha_go_together():
    with pytest.raises(US.CardOverwriteRefused, match="go together"):
        US.load_card_waivers("w.jsonl", None)
    assert US.load_card_waivers(None, None) == {}
