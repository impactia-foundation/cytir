"""Task 1556: the runner owns the family HTTP client. A family declares `http_spec`; `run_family`
builds the client from it, hands it over as `ctx.http`, and closes it. An injected client is used
untouched; a `kind="curl"` spec builds the curl client."""
from __future__ import annotations

import pathlib
import sys

import httpx

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.sources import registry, run_family  # noqa: E402
from cytir.sources.family import CHUNK_NATIVE, Captured, Doc, HttpSpec, Unit  # noqa: E402
import test_aw_source_family_protocol as P  # noqa: E402

SEEN: list = []


class Toy:
    family_id = "toy_http_spec_for_tests"
    route = "documented_source_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_NATIVE
    http_spec = HttpSpec(timeout=7.0, headers={"x-toy": "1"}, follow_redirects=False)

    def discover(self, ctx):
        return [Unit(key="u0", url="file:///held/a.txt")]

    def acquire(self, unit, ctx):
        SEEN.append(ctx.http)
        return [Captured(unit, b"some text", {"status": 200})]

    def extract(self, cap, ctx):
        return [Doc(url=cap.unit.url, text=cap.body.decode("utf-8"))]

    def chunk(self, doc, ctx):
        from cytir.core.ids import record_point_id
        return [{"url": doc.url, "text": doc.text,
                 "point_id": record_point_id({"point_id": doc.url})}]


class ToyCurl(Toy):
    family_id = "toy_http_curl_for_tests"
    http_spec = HttpSpec(kind="curl", timeout=11)


class Bare(Toy):
    family_id = "toy_http_bare_for_tests"
    http_spec = None


def _run(monkeypatch, tmp_path, cls, **kw):
    monkeypatch.setenv("AW_ALLOW_UNIDENTIFIED_CRAWL", "1")
    monkeypatch.setitem(registry._FACTORIES, cls.family_id, cls)

    def mutate(row):
        row["chunk_policy"].update(policy_id="unknown", sha256="unknown", epoch="unknown",
                                   policy_family=None)
        row["status"] = "backfilled"
        row["consent"].update(basis="declared: test", search_ok=True,
                              resolution="live-per-origin")
        from cytir.sources import contract
        row["unknowns"] = sorted(contract.actual_unknowns(row))
    c = P._contracts(tmp_path, cls.family_id, mutate=mutate)
    SEEN.clear()
    return run_family.run_family(cls.family_id, out_dir=tmp_path / "out", today="2026-10-30",
                                 consent=P._consent_ok, contracts_path=c,
                                 log=lambda m: None, **kw)


def test_runner_builds_the_client_from_the_spec_and_closes_it(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, Toy)
    (client,) = SEEN
    assert isinstance(client, httpx.Client)
    assert client.timeout == httpx.Timeout(7.0)
    assert client.headers["x-toy"] == "1" and client.headers["user-agent"]
    assert client.follow_redirects is False
    assert client.is_closed


def test_an_injected_client_is_used_untouched_and_not_closed(monkeypatch, tmp_path):
    mine = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    _run(monkeypatch, tmp_path, Toy, http=mine)
    assert SEEN == [mine] and not mine.is_closed
    mine.close()


def test_a_curl_spec_builds_the_curl_client(monkeypatch, tmp_path):
    from cytir.sources.ctgov_psy import harvest
    _run(monkeypatch, tmp_path, ToyCurl)
    (client,) = SEEN
    assert isinstance(client, harvest.CurlClient) and client.timeout == 11


def test_a_family_without_a_spec_gets_no_client(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, Bare)
    assert SEEN == [None]


class Boom(Toy):
    family_id = "toy_http_boom_for_tests"

    def discover(self, ctx):
        SEEN.append(ctx.http)
        raise RuntimeError("discover broke")


class Windowed(Toy):
    family_id = "toy_http_windowed_for_tests"

    def plan_windows(self, ctx, cap=0):
        SEEN.append(ctx.http)
        return [{"key": "w1", "from": "2026-01-01", "to": "2026-01-31"}]


def test_the_client_is_closed_when_discover_raises(monkeypatch, tmp_path):
    import pytest
    with pytest.raises(RuntimeError, match="discover broke"):
        _run(monkeypatch, tmp_path, Boom)
    (client,) = SEEN
    assert isinstance(client, httpx.Client) and client.is_closed


def test_run_windows_plan_builds_the_client_from_the_spec_and_closes_it(monkeypatch, tmp_path):
    from cytir.sources import run_windows
    monkeypatch.setenv("AW_ALLOW_UNIDENTIFIED_CRAWL", "1")
    monkeypatch.setattr(registry, "admit", lambda fid, strict=False: (Windowed(), {}, None))
    SEEN.clear()
    wins = run_windows.plan(Windowed.family_id, tmp_path, {}, "2026-10-30", 10, http=None)
    (client,) = SEEN
    assert wins[0]["key"] == "w1"
    assert isinstance(client, httpx.Client) and client.timeout == httpx.Timeout(7.0)
    assert client.is_closed
