"""`cytir.gates.prod_legs` — the incumbent's leg arms read on prod (task 865, P12).

Every bound the module promises is executed here, offline: httpx requests are
answered by a MockTransport (nothing leaves the process), and the guard is the
real `cytir.index.prod_guard`.
"""
from __future__ import annotations

import datetime as dt
import json

import httpx
import pytest

from cytir.gates import prod_legs as PL
from cytir.index import prod_guard

PROD = "http://prod-tunnel.invalid:6334"   # the tunnel PORT makes it prod; the host never resolves


@pytest.fixture(autouse=True)
def _fresh_guard(monkeypatch):
    prod_guard.reset()
    monkeypatch.delenv(prod_guard.REPLAY_ENV, raising=False)
    monkeypatch.delenv(prod_guard.BUDGET_ENV, raising=False)
    yield
    prod_guard.reset()


def _client(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"result": {"points": []}, "status": "ok"})
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=PROD)


@pytest.mark.parametrize("minute,forbidden", [(14, False), (15, True), (27, True), (39, True), (40, False), (0, False)])
def test_the_drains_window_is_forbidden(minute, forbidden):
    t = dt.datetime(2026, 10, 10, 7, minute, tzinfo=dt.timezone.utc)
    assert PL.in_forbidden_window(t) is forbidden


def test_the_guard_counts_prod_query_posts_and_lets_gets_through():
    seen: list = []
    g = PL.Guard(rate_per_s=None)
    with g.installed(), _client(seen) as c:
        c.get("/collections/aw_prod")
        c.post("/collections/aw_prod/points/query/batch", json={})
        c.post("/collections/aw_prod/points/query", json={})
    assert g.prod_posts == 2 and not g.refusals
    assert len(seen) == 3


def test_a_scroll_is_refused_before_it_leaves_the_process():
    seen: list = []
    g = PL.Guard(rate_per_s=None)
    with g.installed(), _client(seen) as c:
        with pytest.raises(prod_guard.ProdReadRefused, match="REFUSED scroll on prod"):
            c.post("/collections/aw_prod/points/scroll", json={})
    assert seen == [], "the scroll never reached the transport"
    assert g.refusals and g.prod_posts == 0


def test_the_budget_refuses_the_call_after_the_last_allowed_one(monkeypatch):
    monkeypatch.setenv(prod_guard.BUDGET_ENV, "2")
    seen: list = []
    g = PL.Guard(rate_per_s=None)
    with g.installed(), _client(seen) as c:
        c.post("/collections/aw_prod/points/query", json={})
        c.post("/collections/aw_prod/points/query", json={})
        with pytest.raises(prod_guard.ProdReadRefused, match="budget 2"):
            c.post("/collections/aw_prod/points/query", json={})
    assert len(seen) == 2


def test_the_guard_is_removed_after_the_block():
    orig = httpx.Client.send
    with PL.Guard(rate_per_s=None).installed():
        assert httpx.Client.send is not orig
    assert httpx.Client.send is orig


def test_qdrant_clients_rest_transport_goes_through_the_guard():
    """The serving code's QdrantClient sends through `httpx.Client.send`: the
    class-level wrap covers it without touching the serving code. Driven through
    qdrant_client's own ApiClient (its kwargs reach httpx.Client, so a
    MockTransport keeps it offline)."""
    from qdrant_client.http.api_client import ApiClient
    seen: list = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"result": None, "status": "ok"})
    api = ApiClient(host=PROD, transport=httpx.MockTransport(handler))
    g = PL.Guard(rate_per_s=None)
    with g.installed():
        api.send_inner(httpx.Request("POST", PROD + "/collections/aw_prod/points/query"))
        # qdrant_client re-raises it as its own ResponseHandlingException: the
        # caller sees a generic client error, which is why the Guard RECORDS it
        with pytest.raises(Exception) as exc:
            api.send_inner(httpx.Request("POST", PROD + "/collections/aw_prod/points/scroll"))
    assert "REFUSED scroll on prod" in str(exc.value)
    assert seen == ["/collections/aw_prod/points/query"]
    assert g.prod_posts == 1 and len(g.refusals) == 1


def _search_with(client: httpx.Client, posts: int, scroll: bool = False, swallow: bool = False):
    """A stand-in for `retrieval.search`: `posts` query POSTs per query, an
    optional scroll whose refusal the 'serving code' swallows."""
    def search(q, k, lang=None, country=None):
        if scroll:
            try:
                client.post("/collections/aw_prod/points/scroll", json={})
            except prod_guard.ProdReadRefused:
                if not swallow:
                    raise
        for _ in range(posts):
            client.post("/collections/aw_prod/points/query", json={})
        return [{"url": f"https://example.org/{q}/{i}"} for i in range(k)]
    return search


def _quiet_clock():
    return dt.datetime(2026, 10, 10, 7, 5, tzinfo=dt.timezone.utc)


def test_read_legs_counts_posts_per_query_and_scores_ok_rows(monkeypatch):
    monkeypatch.setenv(prod_guard.BUDGET_ENV, "10")
    g = PL.Guard(rate_per_s=None)
    paced = []
    with g.installed(), _client([]) as c:
        rows = PL.read_legs([{"id": "a", "query": "x"}, {"id": "b", "query": "y",
                             "gold_urls": ["https://example.org/y/2"]}],
                            search=_search_with(c, 2), k=5, guard=g, pace=lambda: paced.append(1),
                            clock=_quiet_clock)
    assert [r["status"] for r in rows] == ["OK", "OK"]
    assert [r["prod_posts"] for r in rows] == [2, 2]
    assert rows[1]["gold_rank"] == 3
    assert paced == [1], "paced between queries, not before the first"


def test_a_swallowed_refusal_makes_the_query_refused_not_a_quiet_ranking(monkeypatch):
    """The identifier fast path catches every exception and falls through to the
    normal search: a refused scroll must not become a different ranking."""
    monkeypatch.setenv(prod_guard.BUDGET_ENV, "10")
    g = PL.Guard(rate_per_s=None)
    with g.installed(), _client([]) as c:
        rows = PL.read_legs([{"id": "uid", "query": "CHE-123.456.789"}, {"id": "b", "query": "y"}],
                            search=_search_with(c, 1, scroll=True, swallow=True), k=3, guard=g,
                            pace=lambda: None, clock=_quiet_clock)
    assert rows[0]["status"] == "REFUSED" and "scroll" in rows[0]["reason"]
    assert "urls" not in rows[0]
    assert rows[1]["status"] == "NOT-RUN", "any refusal stops the whole read (review 866 S4)"


def test_the_read_stops_at_the_first_budget_refusal(monkeypatch):
    monkeypatch.setenv(prod_guard.BUDGET_ENV, "5")
    g = PL.Guard(rate_per_s=None)
    queries = [{"id": f"q{i}", "query": f"q{i}"} for i in range(5)]
    with g.installed(), _client([]) as c:
        rows = PL.read_legs(queries, search=_search_with(c, 2), k=3, guard=g,
                            pace=lambda: None, clock=_quiet_clock)
    assert [r["status"] for r in rows] == ["OK", "OK", "REFUSED", "NOT-RUN", "NOT-RUN"]
    assert g.prod_posts == 5, "never more prod POSTs than the budget"


def test_the_read_stops_before_a_query_inside_the_window():
    g = PL.Guard(rate_per_s=None)
    times = iter([dt.datetime(2026, 10, 10, 7, 14, tzinfo=dt.timezone.utc),
                  dt.datetime(2026, 10, 10, 7, 15, tzinfo=dt.timezone.utc),
                  dt.datetime(2026, 10, 10, 7, 16, tzinfo=dt.timezone.utc)])
    with g.installed(), _client([]) as c:
        rows = PL.read_legs([{"query": "a"}, {"query": "b"}, {"query": "c"}],
                            search=_search_with(c, 1), k=1, guard=g, pace=lambda: None,
                            clock=lambda: next(times))
    assert [r["status"] for r in rows] == ["OK", "NOT-RUN", "NOT-RUN"]


def test_an_error_is_recorded_not_scored():
    g = PL.Guard(rate_per_s=None)

    def boom(q, **kw):
        raise TimeoutError("qdrant timed out")
    rows = PL.read_legs([{"query": "a"}], search=boom, k=1, guard=g, pace=lambda: None,
                        clock=_quiet_clock)
    assert rows[0]["status"] == "ERROR" and "TimeoutError" in rows[0]["reason"]


def test_only_aw_keys_of_the_env_file_are_applied(tmp_path):
    f = tmp_path / "api.env"
    f.write_text("PATH=/usr/local/bin\nHOME=/root\nAW_RANKHEAD=1\nAW_TEI_URL=http://tei:80\n# x\n\n")
    assert PL.load_env_file(str(f)) == {"AW_RANKHEAD": "1", "AW_TEI_URL": "http://tei:80"}


def test_read_target_uses_two_metadata_gets_only():
    calls = []

    class R:
        def __init__(self, body):
            self.body = json.dumps(body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.body

    def opener(url, timeout):
        calls.append(url)
        if url.endswith("/aliases"):
            return R({"result": {"aliases": [{"alias_name": "aw_prod", "collection_name": "aw_e9"}]}})
        return R({"result": {"points_count": 62505339, "status": "green"}})
    t = PL.read_target(PROD, "aw_prod", opener=opener)
    assert t == {"qdrant": PROD, "alias": "aw_prod", "collection": "aw_e9",
                 "points_count": 62505339, "status": "green"}
    assert calls == [PROD + "/aliases", PROD + "/collections/aw_prod"]


@pytest.mark.parametrize("argv,needle", [
    (["--qdrant", PROD, "--max-prod-calls", "401"], "must be 1..400"),
    (["--qdrant", PROD, "--max-prod-calls", "0"], "must be 1..400"),
])
def test_main_refuses_a_budget_above_the_guards(tmp_path, argv, needle):
    w = tmp_path / "w.json"
    w.write_text("{}")
    e = tmp_path / "e.env"
    e.write_text("")
    base = ["--leg", "dense-only", "--weights", str(w), "--env-file", str(e), "--tei", "http://tei.invalid:1",
            "--query", "x", "--out", str(tmp_path / "o.json")]
    with pytest.raises(PL.ProdLegRefused, match=needle):
        PL.main(argv + base)


def test_main_refuses_a_target_that_is_not_prod(tmp_path):
    prod_guard.mark_throwaway("http://scratch.invalid:6999")
    w = tmp_path / "w.json"
    w.write_text("{}")
    e = tmp_path / "e.env"
    e.write_text("")
    with pytest.raises(PL.ProdLegRefused, match="not prod"):
        PL.main(["--qdrant", "http://scratch.invalid:6999", "--alias", "scratch_coll", "--leg", "sparse-only",
                 "--weights", str(w), "--env-file", str(e), "--tei", "http://tei.invalid:1",
                 "--query", "x", "--max-prod-calls", "5", "--out", str(tmp_path / "o.json")])


def test_image_paths_map_to_this_checkout_and_a_missing_one_is_refused():
    env = {"AW_HUBFLOOR_TABLE": "/app/cytir/serve/hub_hosts.json", "AW_RANKHEAD": "1"}
    out = PL.map_image_paths(env)
    assert out["AW_HUBFLOOR_TABLE"].endswith("cytir/serve/hub_hosts.json")
    assert out["AW_RANKHEAD"] == "1"
    with pytest.raises(PL.ProdLegRefused, match="does not exist"):
        PL.map_image_paths({"AW_X": "/app/cytir/serve/no-such-file.json"})


def test_a_live_env_dump_maps_under_the_old_and_the_renamed_image_to_this_trees_own_dirs():
    """The image on the box was built before the rename (`/app/cytir/serve`); the first one after it holds
    `/app/cytir/serve`. A live `docker inspect` dump shows whichever runs; both reach THIS tree's serve dir."""
    own_serve = PL.ROOT / PL.SERVER_REL
    for pkg in ("aw", "cytir"):  # rename-codemod: keep (the old name is the live image)
        out = PL.map_image_paths({"AW_HUBFLOOR_TABLE": f"/app/{pkg}/serve/hub_hosts.aw_b1.json"})
        assert out["AW_HUBFLOOR_TABLE"] == str(own_serve / "hub_hosts.aw_b1.json"), pkg
        out = PL.map_image_paths({"AW_X": f"/app/{pkg}/serve/hub_hosts.json"})
        assert out["AW_X"] == str(own_serve / "hub_hosts.json"), pkg
    own_pkg = PL.ROOT / PL.SERVER_REL.split("/")[0]
    for pkg in ("aw", "cytir"):  # rename-codemod: keep (the old name is the live image)
        out = PL.map_image_paths({"AW_Y": f"/app/{pkg}/core/serve_layout.py"})
        assert out["AW_Y"] == str(own_pkg / "core" / "serve_layout.py"), pkg


def test_an_image_path_under_no_known_prefix_is_refused_by_name_never_passed_through():
    with pytest.raises(PL.ProdLegRefused) as e:
        PL.map_image_paths({"AW_HUBFLOOR_TABLE": "/app/other/x.json", "AW_RANKHEAD": "1"})
    assert "AW_HUBFLOOR_TABLE" in str(e.value) and "/app/other/x.json" in str(e.value)
    # the nearest misspelling of a real prefix is refused too (a package name no image holds)
    with pytest.raises(PL.ProdLegRefused, match="AW_X=/app/awx/serve/hub_hosts.json"):
        PL.map_image_paths({"AW_X": "/app/awx/serve/hub_hosts.json"})



# ---------------------------------------------------------------------------
# Review 866 M1: the leg is the gate's arm (one definition for both sides)
# ---------------------------------------------------------------------------
from cytir.gates import leg_definition as LD  # noqa: E402


def test_dense_only_turns_the_14_floor_head_flags_off_and_takes_the_contracts_weights():
    env = LD.leg_env("dense-only", "/w/dense.json", LD.DENSE_ONLY_WEIGHTS_SHA256)
    assert env["AW_FUSION_WEIGHTS"] == "/w/dense.json"
    assert {k: v for k, v in env.items() if k != "AW_FUSION_WEIGHTS"} == LD.FLOOR_HEAD_OFF
    assert len(LD.FLOOR_HEAD_OFF) == 14
    assert all(v in ("0", "0.0") for v in LD.FLOOR_HEAD_OFF.values())


def test_the_contracts_dense_only_weights_file_is_the_pinned_one():
    import pathlib
    p = pathlib.Path(PL.ROOT) / "data" / "flip-gate-contract.v1.json"
    legs = json.loads(p.read_text())["legs"]
    assert "floor/head flags off" in legs["dense-only"]["weights"]
    assert "fusion_weights.dense_only.json" in legs["dense-only"]["weights"]


@pytest.mark.parametrize("leg,weights,sha,needle", [
    ("dense-only", "/w.json", "0" * 64, "is not the contract's"),
    ("dense-only", None, None, "needs the contract's dense-only weights"),
    ("sparse-only", "/w.json", "0" * 64, "is not the contract's"),
    ("sparse-only", None, None, "needs the contract's sparse-only weights"),
    ("full-stack", "/w.json", "0" * 64, "pass no --weights"),
    ("nope", None, None, "unknown leg"),
])
def test_an_undefined_leg_is_refused(leg, weights, sha, needle):
    with pytest.raises(LD.LegUndefined, match=needle):
        LD.leg_env(leg, weights, sha)


def test_the_gate_call_is_k10_with_each_rows_lang_and_country():
    seen = []

    def search(q, **kw):
        seen.append(kw)
        return []
    PL.read_legs([{"query": "a", "lang": "de", "country": "CH"}, {"query": "b"}], search=search,
                 k=LD.GATE_K, guard=PL.Guard(rate_per_s=None), pace=lambda: None, clock=_quiet_clock)
    assert seen == [{"k": 10, "lang": "de", "country": "CH"}, {"k": 10, "lang": None, "country": None}]


# ---------------------------------------------------------------------------
# Review 866 M2: the TEI pin is checked and recorded
# ---------------------------------------------------------------------------
from cytir.core import model_pin  # noqa: E402


def test_the_tei_pin_passes_on_the_pinned_revision_and_is_returned_for_the_artifact():
    info = {"model_id": model_pin.EMBED_MODEL_ID, "model_sha": "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"}
    got = PL.check_tei_pin("http://tei.invalid:8902", get=lambda url: info)
    assert got == {"tei": "http://tei.invalid:8902", **info}


@pytest.mark.parametrize("info", [
    {"model_id": model_pin.EMBED_MODEL_ID, "model_sha": None},
    {"model_id": model_pin.EMBED_MODEL_ID, "model_sha": "f" * 40},
    {"model_id": "BAAI/bge-m3", "model_sha": "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"},
])
def test_an_unpinned_or_wrong_tei_is_refused(info):
    with pytest.raises(PL.ProdLegRefused, match="refused before any prod request"):
        PL.check_tei_pin("http://tei.invalid:8902", get=lambda url: info)


# ---------------------------------------------------------------------------
# Review 866 section 2.5: the bounds a budget ruling depends on
# ---------------------------------------------------------------------------

def _transport(status=200, exc=None, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request.url.path)
        if exc is not None:
            raise exc
        return httpx.Response(status, json={"status": "ok"})
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=PROD)


def _q(n):
    return [{"id": f"q{i}", "query": f"q{i}"} for i in range(n)]


def test_a_5xx_from_prod_aborts_the_whole_read():
    g = PL.Guard(rate_per_s=None)
    with g.installed(), _transport(503) as c:
        rows = PL.read_legs(_q(3), search=_search_with(c, 1), k=1, guard=g, pace=lambda: None,
                            clock=_quiet_clock)
    assert [r["status"] for r in rows] == ["ABORTED", "NOT-RUN", "NOT-RUN"]
    assert "503" in g.abort


@pytest.mark.parametrize("exc", [httpx.ConnectTimeout("t"), httpx.ReadTimeout("t"), httpx.ConnectError("c")])
def test_a_transport_error_to_prod_aborts_the_whole_read(exc):
    g = PL.Guard(rate_per_s=None)

    with g.installed(), _transport(exc=exc) as c:
        rows = PL.read_legs(_q(2), search=_search_with(c, 1), k=1, guard=g, pace=lambda: None,
                            clock=_quiet_clock)
    assert rows[0]["status"] == "ABORTED" and rows[1]["status"] == "NOT-RUN"
    assert "transport error to prod" in g.abort


def test_the_prod_round_trip_p95_aborts_and_is_timed_inside_the_guard():
    t = [0.0]

    def clock():
        t[0] += 1.5          # every reading advances: one send = 1.5 s of prod round trip
        return t[0]
    g = PL.Guard(rate_per_s=None, p95_max_s=1.0, clock=clock)
    with g.installed(), _transport() as c:
        rows = PL.read_legs(_q(30), search=_search_with(c, 1), k=1, guard=g, pace=lambda: None,
                            clock=_quiet_clock)
    statuses = [r["status"] for r in rows]
    assert statuses[:PL.P95_MIN_SAMPLES - 1] == ["OK"] * (PL.P95_MIN_SAMPLES - 1)
    assert statuses[PL.P95_MIN_SAMPLES - 1] == "ABORTED" and "p95" in g.abort
    assert set(statuses[PL.P95_MIN_SAMPLES:]) == {"NOT-RUN"}


def test_at_most_two_prod_posts_per_second():
    t, slept = [0.0], []

    def sleep(s):
        slept.append(s)
        t[0] += s
    g = PL.Guard(rate_per_s=2.0, clock=lambda: t[0], sleep=sleep)
    with g.installed(), _transport() as c:
        for _ in range(4):
            c.post("/collections/aw_prod/points/query", json={})
    assert slept == [0.5, 0.5, 0.5], "a 0.5 s gap between prod POSTs, per POST"


def test_a_second_prod_request_in_flight_is_refused():
    g = PL.Guard(rate_per_s=None)
    with g.installed(), _transport() as c:
        g._inflight.acquire()
        try:
            with pytest.raises(prod_guard.ProdReadRefused, match="in flight"):
                c.post("/collections/aw_prod/points/query", json={})
        finally:
            g._inflight.release()
    assert "sequential" in g.abort


def test_the_ruled_cap_holds_across_processes(tmp_path):
    a = PL.Ledger(tmp_path, "RULING-31 gate budget", cap=3)
    assert [a.reserve() for _ in range(3)] == [1, 2, 3]
    with pytest.raises(PL.ProdLegAbort, match="3 of 3"):
        a.reserve()
    b = PL.Ledger(tmp_path, "RULING-31 gate budget", cap=3)          # a re-run after an abort
    assert b.used() == 3
    with pytest.raises(PL.ProdLegAbort):
        b.reserve()
    other = PL.Ledger(tmp_path, "RULING-32", cap=3)
    assert other.used() == 0, "keyed by the ruling"


def test_the_guard_spends_the_ledger_and_stops_the_read_at_the_cap(tmp_path):
    led = PL.Ledger(tmp_path, "R", cap=3)
    g = PL.Guard(rate_per_s=None, ledger=led)
    with g.installed(), _transport() as c:
        rows = PL.read_legs(_q(3), search=_search_with(c, 2), k=1, guard=g, pace=lambda: None,
                            clock=_quiet_clock)
    assert [r["status"] for r in rows] == ["OK", "REFUSED", "NOT-RUN"]
    assert led.used() == 3 and g.prod_posts == 3


def test_before_query_objections_stop_the_read():
    calls = []

    def before(i):
        calls.append(i)
        return "aw-promotion-nightly.service is active" if i == 1 else None
    rows = PL.read_legs(_q(3), search=lambda q, **kw: [], k=1, guard=PL.Guard(rate_per_s=None),
                        pace=lambda: None, clock=_quiet_clock, before_query=before)
    assert [r["status"] for r in rows] == ["OK", "NOT-RUN", "NOT-RUN"]
    assert "active" in rows[1]["reason"]


def test_the_precommit_measurement_stops_above_the_mean():
    g = PL.Guard(rate_per_s=None)
    with g.installed(), _transport() as c:
        rows = PL.read_legs(_q(4), search=_search_with(c, 3), k=1, guard=g, pace=lambda: None,
                            clock=_quiet_clock, precommit=(2, 2.4))
    assert [r["status"] for r in rows] == ["OK", "OK", "NOT-RUN", "NOT-RUN"]
    assert "pre-commit: 3.00" in rows[2]["reason"]


def _main_base(tmp_path, *extra):
    w = tmp_path / "w.json"
    w.write_text("{}")
    e = tmp_path / "e.env"
    e.write_text("")
    return ["--qdrant", PROD, "--leg", "dense-only", "--weights", str(w), "--env-file", str(e),
            "--tei", "http://tei.invalid:1", "--query", "x", "--out", str(tmp_path / "o.json"), *extra]


def test_a_ruled_budget_needs_the_ruling_written_in_the_plan(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("nothing ruled here\n")
    with pytest.raises(PL.ProdLegRefused, match="is not written in"):
        PL.main(_main_base(tmp_path, "--max-prod-calls", "20000", "--budget-ruling", "RULING-31",
                           "--plan-file", str(plan), "--ledger-dir", str(tmp_path / "l")))


def test_a_ruled_budget_needs_a_stop_file_and_the_env_cmd(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("RULING-31: gate budget 20,000\n")
    base = _main_base(tmp_path, "--max-prod-calls", "20000", "--budget-ruling", "RULING-31",
                      "--plan-file", str(plan), "--ledger-dir", str(tmp_path / "l"))
    with pytest.raises(PL.ProdLegRefused, match="--stop-file"):
        PL.main(base)
    with pytest.raises(PL.ProdLegRefused, match="--prod-env-cmd"):
        PL.main(base + ["--stop-file", str(tmp_path / "stop")])


def test_a_ruled_budget_is_capped_at_20000_and_a_spent_ruling_is_refused(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("RULING-31\n")
    common = ["--budget-ruling", "RULING-31", "--plan-file", str(plan), "--ledger-dir", str(tmp_path / "l"),
              "--stop-file", str(tmp_path / "stop"), "--prod-env-cmd", "true"]
    with pytest.raises(PL.ProdLegRefused, match="1..20000"):
        PL.main(_main_base(tmp_path, "--max-prod-calls", "20001", *common))
    led = PL.Ledger(tmp_path / "l", "RULING-31", cap=2)
    led.reserve()
    led.reserve()
    with pytest.raises(PL.ProdLegRefused, match="is spent"):
        PL.main(_main_base(tmp_path, "--max-prod-calls", "2", *common))


def test_metadata_gets_are_counted():
    class R:
        def __init__(self, body):
            self.body = json.dumps(body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.body

    def opener(url, timeout):
        if url.endswith("/aliases"):
            return R({"result": {"aliases": []}})
        return R({"result": {"points_count": 1, "status": "green"}})
    m = PL.MetaReader(opener)
    m.read_target(PROD, "aw_prod")
    m.read_target(PROD, "aw_prod")
    assert m.gets == 4


def test_a_ruled_read_refuses_any_k(tmp_path):
    """Review 866 round 2: a ruled read can only be the gate's arm (k = 10)."""
    plan = tmp_path / "plan.md"
    plan.write_text("RULING-31\n")
    common = ["--budget-ruling", "RULING-31", "--plan-file", str(plan), "--ledger-dir", str(tmp_path / "l"),
              "--stop-file", str(tmp_path / "stop"), "--prod-env-cmd", "true", "--max-prod-calls", "100"]
    for k in ("10", "100"):
        with pytest.raises(PL.ProdLegRefused, match="no arm-definition override"):
            PL.main(_main_base(tmp_path, "--k", k, *common))
    assert not (tmp_path / "l").exists() or not any((tmp_path / "l").iterdir()), "refused before the ledger"


def test_an_unruled_read_defaults_to_the_gates_k():
    a = PL.parse_args(["--qdrant", PROD, "--leg", "dense-only", "--env-file", "e", "--tei", "t",
                       "--query", "x", "--max-prod-calls", "5", "--out", "o"])
    assert a.k is None and LD.GATE_K == 10


def test_sparse_only_turns_the_floor_head_flags_off_but_keeps_the_bm25_legs_own_flag():
    """Task 863 (POLICY: the sparse-only arm definition): the dense-only flags minus AW_BM25_PER_TOKEN
    (it changes which rows the BM25 leg returns), the contract's weights file, k = 10."""
    env = LD.leg_env("sparse-only", "/w/sparse.json", LD.SPARSE_ONLY_WEIGHTS_SHA256)
    assert env["AW_FUSION_WEIGHTS"] == "/w/sparse.json"
    assert "AW_BM25_PER_TOKEN" not in env and LD.SPARSE_LEG_FLAGS == ("AW_BM25_PER_TOKEN",)
    assert {k: v for k, v in env.items() if k != "AW_FUSION_WEIGHTS"} == {
        k: v for k, v in LD.FLOOR_HEAD_OFF.items() if k != "AW_BM25_PER_TOKEN"}
    import pathlib
    leg = json.loads((pathlib.Path(PL.ROOT) / "data" / "flip-gate-contract.v1.json").read_text())["legs"]["sparse-only"]
    assert "SPARSE_ONLY_WEIGHTS_SHA256" in leg["weights_sha256_source"] and leg["k"] == LD.GATE_K == 10


def test_the_sparse_only_weights_file_is_the_mirror_of_the_dense_only_file():
    """The file the contract names hashes to the pinned sha; dense 0 and bm25 1, every other weight 0,
    and the non-weight knobs of the dense-only file. (Not in the public cut, like the dense-only file.)"""
    import hashlib
    import pathlib
    root = pathlib.Path(PL.ROOT)
    leg = json.loads((root / "data" / "flip-gate-contract.v1.json").read_text())["legs"]["sparse-only"]
    f = root / leg["weights_file"]
    if not f.exists():
        pytest.skip("the sparse-only weights file is not in this tree (the public cut ships neither leg file)")
    assert hashlib.sha256(f.read_bytes()).hexdigest() == LD.SPARSE_ONLY_WEIGHTS_SHA256
    w = json.loads(f.read_text())
    knobs = {"legifrance_bulk_relgate": 1.0, "entity_card_beta": 0.5, "entity_card_name_thr": 0.6,
             "entity_card_cap": 1, "commercial_gov_scale": 1.0}
    assert (w["dense"], w["bm25"]) == (0.0, 1.0)
    assert {k: v for k, v in w.items() if k in knobs} == knobs
    assert all(v == 0.0 for k, v in w.items() if k not in knobs and k != "bm25")


def test_qdrant_client_version_check_thread_sends_nothing_under_the_guard():
    """916: the client's background `GET /` raced the first POST and tripped the sequential guard."""
    import threading
    pytest.importorskip("qdrant_client")
    from cytir.core.qdrant import qdrant_client
    g = PL.Guard(rate_per_s=None)
    with g.installed():
        qdrant_client(PROD, timeout=1)
        for t in threading.enumerate():
            if t is not threading.current_thread() and t.daemon:
                t.join(5)
    assert g.requests_total == 0
    assert g.abort is None


def test_the_qdrant_timeout_override_reaches_the_client_and_none_keeps_the_env_value(monkeypatch):
    import sys
    sys.path[:0] = [str(PL.ROOT / "server")]
    from cytir.serve import retrieval
    built = []
    import qdrant_client as _qc
    monkeypatch.setattr(_qc, "QdrantClient", lambda *a, **kw: built.append({"args": a, **kw}) or object())
    io = retrieval.qdrant_io
    monkeypatch.setattr(io, "_QDRANT_TIMEOUT", 8.0)
    monkeypatch.setattr(io, "_client", None)
    assert LD.apply_qdrant_timeout(retrieval, None) == 8.0
    io.client()
    assert built[-1]["timeout"] == 8.0
    assert LD.apply_qdrant_timeout(retrieval, 30) == 30.0
    io.client()
    assert built[-1]["timeout"] == 30.0
    with pytest.raises(ValueError):
        LD.apply_qdrant_timeout(retrieval, 0)


@pytest.mark.parametrize("src", ["prod_legs", "scratch_legs"])
def test_both_tools_take_the_timeout_flag_apply_it_and_record_it_in_the_receipt(src):
    text = (PL.ROOT / "cytir" / "gates" / f"{src}.py").read_text()
    assert '"--qdrant-timeout-s"' in text
    assert "LD.apply_qdrant_timeout(retrieval," in text
    assert '"qdrant_timeout_s_override"' in text
