#!/usr/bin/env python3
"""Offline unit tests for the shared pack fetch layer (task 195).

Covers: robots parsing (specific-UA precedence, crawl-delay, wildcard/$,
longest-match allow-vs-disallow, JS-shell), the L0 store (exact-bytes
roundtrip, shard rolling, resume append, truncated-tail tolerance,
complete-vs-partial meta markers), canon/page_links helpers, and an
end-to-end run_pack crawl against an httpx.MockTransport (no network):
robots enforcement, L0 retention, outlink sidecar, passages schema,
resume, and --limit.

Run: python3 -m pytest tests/test_pack_fetch.py
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import types

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cytir.acquire import crawl as pack_fetch  # noqa: E402
from cytir.acquire.crawl import (  # noqa: E402
    L0Store, MarkdownStore, Robots, iter_l0, iter_l1, make_canon, page_links,
    run_pack)
from cytir.core import common as aw_common  # noqa: E402


# --- robots -------------------------------------------------------------------
def test_robots_specific_ua_wins_over_star():
    rb = Robots("User-agent: *\nDisallow: /\n\n"
                "User-agent: AuthoritativeWebBot\nDisallow: /private/\n"
                "Crawl-delay: 7\n")
    assert rb.allowed("/anything")
    assert not rb.allowed("/private/x")
    assert rb.delay == 7.0


def test_robots_longest_match_allow_beats_shorter_disallow():
    rb = Robots("User-agent: *\nDisallow: /a/\nAllow: /a/public/\n")
    assert not rb.allowed("/a/x")
    assert rb.allowed("/a/public/x")


def test_robots_wildcard_and_dollar():
    rb = Robots("User-agent: *\nDisallow: /*.json$\nDisallow: /tmp*\n")
    assert not rb.allowed("/api/data.json")
    assert rb.allowed("/api/data.jsonl")
    assert not rb.allowed("/tmp/x")


def test_robots_js_shell_is_broken():
    assert Robots("<html><body>app</body></html>").broken
    assert not Robots("User-agent: *\nDisallow:\n").broken


# --- L0 store -----------------------------------------------------------------
def test_l0_roundtrip_exact_bytes(tmp_path):
    st = L0Store(str(tmp_path), "testpack")
    body = "höhere Gewalt & <em>exact</em> bytes\n".encode("utf-8")
    sha = st.put("https://x.ch/a", body, content_type="text/html",
                 encoding="utf-8")
    st.put("https://x.ch/api", b'{"k": 1}', kind="api",
           content_type="application/json")
    st.close(complete=True)

    rows = list(iter_l0(str(tmp_path / "wave=testpack")))
    assert len(rows) == 2
    assert rows[0]["body"] == body
    assert rows[0]["sha256"] == sha == hashlib.sha256(body).hexdigest()
    assert rows[1]["kind"] == "api"
    meta = json.load(open(tmp_path / "wave=testpack" / "_meta.json"))
    assert meta["complete"] is True and meta["records"] == 2
    assert meta["layer"] == "L0"
    # complete meta carries per-shard sha256
    assert set(meta["sha256"]) == set(meta["files"])
    assert not (tmp_path / "wave=testpack" / "_meta.partial.json").exists()


def test_l0_shard_rolling_and_resume_append(tmp_path):
    st = L0Store(str(tmp_path), "p", shard_max_bytes=200)  # tiny: roll fast
    for i in range(5):
        st.put(f"https://x.ch/{i}", (f"body-{i}" * 30).encode())
    st.close(complete=False)                      # crawl paused: partial meta
    wdir = tmp_path / "wave=p"
    shards = sorted(wdir.glob("shard-*.jsonl.gz"))
    assert len(shards) > 1                         # rolled
    assert (wdir / "_meta.partial.json").exists()
    assert not (wdir / "_meta.json").exists()

    st2 = L0Store(str(tmp_path), "p", shard_max_bytes=200)   # resume
    assert st2.records == 5                        # carried over
    st2.put("https://x.ch/5", b"tail")
    st2.close(complete=True)
    rows = list(iter_l0(str(wdir)))
    assert [r["url"] for r in rows] == [f"https://x.ch/{i}" for i in range(6)]


def test_l0_truncated_tail_tolerated(tmp_path):
    st = L0Store(str(tmp_path), "t")
    st.put("https://x.ch/1", b"one")
    st.put("https://x.ch/2", b"two")
    st.close(complete=False)
    shard = sorted((tmp_path / "wave=t").glob("shard-*.jsonl.gz"))[0]
    with open(shard, "ab") as f:                  # killed-run torn gzip member
        f.write(b"\x1f\x8b\x08\x00GARBAGE-TRUNCATED")
    rows = list(iter_l0(str(tmp_path / "wave=t")))
    assert [r["body"] for r in rows] == [b"one", b"two"]


def test_l1_retains_exact_markdown_lineage_and_quarantine(tmp_path):
    store = MarkdownStore(str(tmp_path), "p")
    markdown = "# Heading\n\nExact retained body."
    digest = store.put(
        url="https://x.ch/a", original_sha256="a" * 64,
        markdown=markdown, extraction_mode="precision", rejection=None,
        chunk_count=2, chunk_rejections={}, source_profile="web_html")
    store.put(
        url="https://x.ch/b", original_sha256="b" * 64,
        markdown=None, extraction_mode=None, rejection="extract_empty_or_low_quality",
        chunk_count=0, chunk_rejections={}, source_profile="web_html")
    store.close(complete=True)

    wave = tmp_path / "wave=p"
    rows = list(iter_l1(str(wave)))
    assert rows[0]["markdown"] == markdown
    assert rows[0]["markdown_sha256"] == digest == \
        hashlib.sha256(markdown.encode()).hexdigest()
    assert rows[0]["parent_original_sha256"] == "a" * 64
    assert rows[1]["status"] == "quarantined"
    meta = json.load(open(wave / "_meta.json"))
    assert meta["complete"] is True
    assert (meta["records"], meta["emitted"], meta["quarantined"]) == (2, 1, 1)
    assert set(meta["sha256"]) == set(meta["files"])


# --- URL helpers ---------------------------------------------------------------
def test_make_canon_alias_and_query():
    canon = make_canon({"x.ch": "www.x.ch"})
    assert canon("http://x.ch/a?utm=1") == "https://www.x.ch/a"
    assert canon("https://www.x.ch/a?p=2", keep_query=True) == \
        "https://www.x.ch/a?p=2"


def test_page_links_same_host_only_no_binaries():
    canon = make_canon({})
    html = ('<a href="/keep">k</a> <a href="https://other.ch/x">o</a> '
            '<a href="/f.pdf">b</a> <a href="/page#frag">f</a>')
    out = page_links("https://www.x.ch/base", html, "www.x.ch", canon)
    assert out == ["https://www.x.ch/keep", "https://www.x.ch/page"]


# --- run_pack end-to-end (MockTransport, no network) ----------------------------
PAGE = """<html><head><title>Testseite | Amt</title></head><body><main>
<h1>Familienzulagen im Kanton</h1>
<p>Die Familienzulagen betragen mindestens 200 Franken pro Kind und Monat.
Anspruch haben Arbeitnehmende sowie Selbststaendige. Die Ansaetze werden vom
Kanton festgelegt und regelmaessig ueberprueft. Der Anspruch entsteht mit dem
Arbeitsverhaeltnis und erlischt mit dessen Ende. Massgebend ist das
Bundesgesetz ueber die Familienzulagen, das die Mindestansaetze festlegt.</p>
<p>Zustaendig ist die kantonale Ausgleichskasse. Gesuche sind beim Arbeitgeber
einzureichen. Die Kasse prueft den Anspruch und zahlt die Zulagen mit dem Lohn
aus. Gegen Verfuegungen kann innert 30 Tagen Einsprache erhoben werden. Ein
Merkblatt erklaert das Verfahren im Detail und nennt die Fristen.</p>
<a href="https://www.example.org/extern">Bundesrecht</a>
<a href="/sub">Unterseite</a>
</main></body></html>"""


def _mock_app(request: httpx.Request) -> httpx.Response:
    p = request.url.path
    host = request.url.host
    if host != "www.t.ch":
        return httpx.Response(404)
    if p == "/robots.txt":
        return httpx.Response(200, text="User-agent: *\nDisallow: /geheim/\n")
    if p in ("/", "/sub"):
        return httpx.Response(200, text=PAGE,
                              headers={"content-type": "text/html"})
    if p == "/geheim/x":
        return httpx.Response(200, text=PAGE,
                              headers={"content-type": "text/html"})
    return httpx.Response(404)


def _args(tmp_path, **over):
    ns = argparse.Namespace(
        out_dir=str(tmp_path / "work"), max_chunks=50, limit=0, only="",
        budget=100, l0_root=str(tmp_path / "l0"),
        l1_root=str(tmp_path / "l1"), no_l0=False, hb_secs=0.0)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


HOSTS = {"t": dict(host="www.t.ch", mode="bfs",
                   seeds=["https://www.t.ch/", "https://www.t.ch/geheim/x"],
                   allow=lambda u: True,
                   lang=lambda u: "de", delay=0.0, cap=10,
                   source_type="official", tier="B", trust=0.9)}


@pytest.fixture()
def mock_client(monkeypatch):
    # Keep this offline unit test independent of the optional crawl-box venv.
    # The production dependency and version are pinned in requirements.txt;
    # here the fake makes the exact L1 -> chunk hand-off observable.
    markdown = """# Familienzulagen im Kanton

Die Familienzulagen betragen mindestens 200 Franken pro Kind und Monat.
Anspruch haben Arbeitnehmende sowie Selbststaendige. Die kantonale
Ausgleichskasse prueft Gesuche und Verfuegungen koennen angefochten werden.

Das Bundesgesetz legt die Mindestansaetze und die massgebenden Fristen fest.
Ein Merkblatt beschreibt das Verfahren und die erforderlichen Unterlagen.
"""
    fake = types.SimpleNamespace(
        extract=lambda *_args, **_kwargs: markdown,
        extract_metadata=lambda _html: types.SimpleNamespace(title="Testseite"))
    monkeypatch.setitem(sys.modules, "trafilatura", fake)
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    c = httpx.Client(transport=httpx.MockTransport(_mock_app),
                     headers=pack_fetch.HEADERS, follow_redirects=True)
    yield c
    c.close()
    aw_common.disable_outlink_capture()


def test_run_pack_end_to_end(tmp_path, mock_client):
    canon = make_canon({})
    state = run_pack(pack="test_pack", hosts=HOSTS, args=_args(tmp_path),
                     canon=canon, client=mock_client)
    work = tmp_path / "work"

    # robots: /geheim/x skipped, never fetched
    assert state["hosts"]["t"]["robots_skipped"] == 1
    fetched_urls = {json.loads(l)["url"]
                    for l in open(work / "fetch_log.jsonl") if "url" in l}
    assert "https://www.t.ch/geheim/x" not in fetched_urls
    assert state["hosts"]["t"]["pages_ok"] == 2          # / and /sub

    # passages: pack dump schema (asdict minus content_digest + provenance)
    rows = [json.loads(l) for l in open(work / "passages.jsonl")]
    assert rows and all(r["provenance"] == "test_pack" for r in rows)
    assert all("content_digest" not in r for r in rows)
    for k in ("url", "title", "snippet", "text", "lang", "domain", "tier",
              "source_type", "trust_tier", "point_id"):
        assert k in rows[0], k
    assert rows[0]["title"] == "Familienzulagen im Kanton"   # H1 wins
    assert all(r["parent_original_sha256"] for r in rows)
    assert all(r["parent_markdown_sha256"] for r in rows)
    by_parent = {}
    for row in rows:
        by_parent.setdefault((row["url"], row["parent_original_sha256"]), []).append(row)
    for siblings in by_parent.values():
        assert [r["chunk_ordinal"] for r in siblings] == list(range(len(siblings)))
        assert all(r["chunk_count"] == len(siblings) for r in siblings)

    # L1 is durable, and the exact bytes hashed here are the same object passed
    # to process_html's chunk dispatcher (no second extraction).
    l1_rows = list(iter_l1(str(tmp_path / "l1" / "wave=test_pack")))
    assert len(l1_rows) == 2 and all(r["status"] == "emitted" for r in l1_rows)
    assert all(hashlib.sha256(r["markdown"].encode()).hexdigest() ==
               r["markdown_sha256"] for r in l1_rows)
    assert json.load(open(tmp_path / "l1" / "wave=test_pack" /
                          "_meta.json"))["complete"] is True
    l2_meta = json.load(open(work / "_chunks_meta.json"))
    assert l2_meta["complete"] is True
    assert l2_meta["records"] == len(rows)
    assert l2_meta["sha256"] == hashlib.sha256(
        (work / "passages.jsonl").read_bytes()).hexdigest()

    # L0: both fetched pages retained, exact bytes, complete marker
    wdir = tmp_path / "l0" / "wave=test_pack"
    l0_rows = list(iter_l0(str(wdir)))
    assert {r["url"] for r in l0_rows} == \
        {"https://www.t.ch/", "https://www.t.ch/sub"}
    assert all(hashlib.sha256(r["body"]).hexdigest() == r["sha256"]
               for r in l0_rows)
    assert json.load(open(wdir / "_meta.json"))["complete"] is True

    # outlinks: 194 sink armed by run_pack, external link captured
    olrows = [json.loads(l) for l in open(work / "outlinks.jsonl")]
    assert {r["url"] for r in olrows} == \
        {"https://www.t.ch/", "https://www.t.ch/sub"}
    assert any("https://www.example.org/extern" in r["links"] for r in olrows)

    # resume: the 175-pattern re-seeds an empty BFS queue (seeds only —
    # "BFS cannot silently stall"); non-seed pages are NOT refetched, and
    # every refetch lands in L0 again (append-only capture history).
    aw_common.disable_outlink_capture()
    n_log = sum(1 for _ in open(work / "fetch_log.jsonl"))
    c2 = httpx.Client(transport=httpx.MockTransport(_mock_app),
                      headers=pack_fetch.HEADERS)
    run_pack(pack="test_pack", hosts=HOSTS, args=_args(tmp_path),
             canon=canon, client=c2)
    c2.close()
    second = [json.loads(l)
              for l in open(work / "fetch_log.jsonl")][n_log:]
    assert {r["url"] for r in second} <= {"https://www.t.ch/"}  # seeds only
    assert "https://www.t.ch/sub" not in {r["url"] for r in second}
    assert len(list(iter_l0(str(wdir)))) == len(l0_rows) + len(second)


def test_run_pack_limit_stops_early_and_is_resumable(tmp_path, mock_client):
    canon = make_canon({})
    state = run_pack(pack="lim_pack", hosts=HOSTS,
                     args=_args(tmp_path, limit=1), canon=canon,
                     client=mock_client)
    assert state["fetches"] == 1
    # frontier persisted for the next run
    st = json.load(open(tmp_path / "work" / "state.json"))
    assert st["queued"].get("t")


def test_run_pack_no_l0_writes_nothing(tmp_path, mock_client):
    canon = make_canon({})
    run_pack(pack="nol0", hosts=HOSTS, args=_args(tmp_path, no_l0=True),
             canon=canon, client=mock_client)
    assert not (tmp_path / "l0" / "wave=nol0").exists()
