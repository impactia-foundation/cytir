"""`cyt gate` (slice 8 step 5): the scorer is the authority gate's, the pass rule is one named
function, and a gate that cannot score REFUSES (an error is never a miss).

No Qdrant and no model: a fake HTTP server answers `/health` and `/search` from a table, so every
test controls exactly which URLs come back in which order.
"""
from __future__ import annotations

import http.server
import json
import pathlib
import threading
import urllib.parse

import pytest

pytest.importorskip("yaml")

from cytir import cli  # noqa: E402
from cytir.gates import authority as AUTH  # noqa: E402
from cytir.gates import slice_gate as G  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
SLICE = REPO / "data" / "gate-slices" / "ci-fixture.v1.json"
A, B, C = "https://x.example/law/a#art_1", "https://x.example/law/a#art_2", "https://y.example/other"


def _slice(tmp_path, golds=None, name="s.json"):
    doc = {"schema": G.SLICE_SCHEMA, "name": "t", "golds": golds or [
        {"id": "g1", "class": "law", "query": "alpha", "target_url": A},
        {"id": "g2", "class": "law", "query": "beta", "target_url": B}]}
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


class _Fake:
    """`answers[query] -> list of urls` (or an int: that HTTP status)."""

    def __init__(self, answers, health="ok", points=7):
        answers_ = answers
        health_ = health
        points_ = points

        class H(http.server.BaseHTTPRequestHandler):
            def _json(self, code, doc):
                body = json.dumps(doc).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                u = urllib.parse.urlparse(self.path)
                if u.path == "/health":
                    return self._json(200, {"status": health_, "points": points_,
                                            "config": {"collection": "c", "mrl_dim": 256}})
                q = urllib.parse.parse_qs(u.query)["q"][0]
                got = answers_[q]
                if isinstance(got, int):
                    return self._json(got, {"error": "boom", "status": "degraded"})
                if isinstance(got, dict):                  # a raw body, however malformed
                    return self._json(200, got)
                return self._json(200, {"query": q, "count": len(got),
                                        "results": [{"url": x} for x in got]})

            def log_message(self, *a):
                pass

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def fake():
    made = []

    def make(answers, **kw):
        f = _Fake(answers, **kw)
        made.append(f)
        return f
    yield make
    for f in made:
        f.close()


def _run(argv, capsys):
    rc = cli.main(["gate", *argv])
    out = capsys.readouterr()
    return rc, out.out, out.err


# ------------------------------------------------------------------ the scorer is the authority's
def test_the_per_gold_rank_is_the_authority_scorers_not_a_copy():
    gold = {"id": "g", "query": "q", "target_url": A, "class": "law"}
    urls = [C, "https://x.example/law/a#art_9", A]
    row = G.score_gold(gold, urls, 10)
    gu, gd = AUTH.gold_of(gold)
    assert row["rank"] == AUTH.rank_of(urls, gu, gd) == 2          # page-level: the sibling article counts
    assert row["rank_exact"] == 3                                  # the exact URL, fragment included
    assert row["hit_at_k"] is True and row["top"] == urls
    assert G.score_gold(gold, [C], 10)["rank"] is None
    # the module has no scorer of its own
    assert G.AUTH is AUTH and not hasattr(G, "rank_of") and not hasattr(G, "norm_url")


def test_the_adr043_rule_lives_in_the_authority_module_and_compare_uses_it():
    assert AUTH.rank_regressed(1, 2) and AUTH.rank_regressed(3, None) and AUTH.rank_regressed(1, None)
    assert not AUTH.rank_regressed(2, 2) and not AUTH.rank_regressed(2, 1)
    assert not AUTH.rank_regressed(None, None) and not AUTH.rank_regressed(None, 5)
    src = (REPO / "cytir" / "gates" / "authority.py").read_text()
    assert "rank_regressed(ra, rb)" in src and "rb > ra" not in src.split("def compare")[1]


# ------------------------------------------------------------------ THE pass rule
def _row(i, rank, exact=None):
    return {"id": i, "rank": rank, "rank_exact": rank if exact is None else exact}


def test_the_pass_rule_no_gold_worse_than_the_baseline():
    base = [_row("a", 1), _row("b", 3), _row("c", None)]
    ok, d = G.passes_against_baseline([_row("a", 1), _row("b", 2), _row("c", 4)], base)
    assert ok and d["regressions"] == []
    assert {(x["id"], x["measure"]) for x in d["improved"]} >= {("b", "rank"), ("c", "rank")}
    ok, d = G.passes_against_baseline([_row("a", 2), _row("b", 3), _row("c", None)], base)
    assert not ok and d["regressions"][0] == {"id": "a", "measure": "rank", "baseline": 1, "now": 2}
    ok, d = G.passes_against_baseline([_row("a", None), _row("b", 3), _row("c", None)], base)
    assert not ok                                                  # found before, gone now
    ok, d = G.passes_against_baseline([_row("a", 1, exact=4), _row("b", 3), _row("c", None)],
                                      [_row("a", 1), _row("b", 3), _row("c", None)])
    assert not ok and d["regressions"][0]["measure"] == "rank_exact"
    ok, d = G.passes_against_baseline([_row("new", 1)], base)
    assert ok and d["not_in_baseline"] == ["new"]


# ------------------------------------------------------------------ the verb
def test_without_a_baseline_it_writes_the_verdict_and_exits_0_with_pass_null(tmp_path, fake, capsys):
    f = fake({"alpha": [A, C], "beta": [C, B]})
    out = tmp_path / "v.json"
    rc, stdout, _ = _run(["--api", f.url, "--slice", str(_slice(tmp_path)), "--verdict-out", str(out),
                          "--label", "vectors=stand-in"], capsys)
    assert rc == 0 and "no --baseline" in stdout
    v = json.loads(out.read_text())
    assert v["schema"] == G.SCHEMA and v["pass"] is None and v["labels"] == {"vectors": "stand-in"}
    assert [(g["id"], g["rank"], g["rank_exact"], g["hit_at_k"]) for g in v["golds"]] == [
        ("g1", 1, 1, True), ("g2", 2, 2, True)]
    assert v["classes"]["law"] == {"n": 2, "h@1": 1, "h@10": 2, "mrr": 0.75}
    assert v["slice"]["sha256"] == G.sha256_file(_slice(tmp_path)) and v["server"]["points"] == 7


def test_pass_and_fail_against_a_baseline_by_exit_code(tmp_path, fake, capsys):
    s = _slice(tmp_path)
    good = fake({"alpha": [A, C], "beta": [B, C]})
    base = tmp_path / "base.json"
    assert _run(["--api", good.url, "--slice", str(s), "--verdict-out", str(base)], capsys)[0] == 0
    rc, stdout, _ = _run(["--api", good.url, "--slice", str(s), "--baseline", str(base)], capsys)
    assert rc == 0 and '"pass": true' in stdout
    worse = fake({"alpha": [C, A], "beta": [B, C]})
    out = tmp_path / "w.json"
    rc, _, err = _run(["--api", worse.url, "--slice", str(s), "--baseline", str(base),
                       "--verdict-out", str(out)], capsys)
    assert rc == 1 and "ranks worse" in err
    v = json.loads(out.read_text())
    assert v["pass"] is False and v["baseline"]["regressions"][0]["id"] == "g1"
    gone = fake({"alpha": [A], "beta": [C]})                       # a gold's page left the answers
    assert _run(["--api", gone.url, "--slice", str(s), "--baseline", str(base)], capsys)[0] == 1


def test_a_baseline_of_another_slice_is_refused_never_compared(tmp_path, fake, capsys):
    f = fake({"alpha": [A], "beta": [B], "gamma": [C]})
    base = tmp_path / "base.json"
    _run(["--api", f.url, "--slice", str(_slice(tmp_path)), "--verdict-out", str(base)], capsys)
    other = _slice(tmp_path, golds=[{"id": "g1", "class": "law", "query": "gamma", "target_url": C}],
                   name="other.json")
    rc, _, err = _run(["--api", f.url, "--slice", str(other), "--baseline", str(base)], capsys)
    assert rc == 3 and "changed slice needs a new baseline" in err


def test_an_error_is_never_a_miss(tmp_path, fake, capsys):
    s = str(_slice(tmp_path))
    erroring = fake({"alpha": [A], "beta": 503})
    rc, _, err = _run(["--api", erroring.url, "--slice", s], capsys)
    assert rc == 3 and "HTTP 503" in err
    sick = fake({"alpha": [A], "beta": [B]}, health="degraded")
    rc, _, err = _run(["--api", sick.url, "--slice", s], capsys)
    assert rc == 3 and "not ok" in err
    rc, _, err = _run(["--api", "http://127.0.0.1:1", "--slice", s, "--timeout", "2"], capsys)
    assert rc == 3 and "REFUSED" in err


def test_a_bad_slice_or_label_is_refused(tmp_path, fake, capsys):
    f = fake({})
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "x", "golds": []}))
    assert _run(["--api", f.url, "--slice", str(bad)], capsys)[0] == 3
    dup = _slice(tmp_path, golds=[{"id": "g", "query": "q", "target_url": A},
                                  {"id": "g", "query": "r", "target_url": B}], name="dup.json")
    assert _run(["--api", f.url, "--slice", str(dup)], capsys)[0] == 3
    assert _run(["--api", f.url, "--slice", str(_slice(tmp_path)), "--label", "novalue"], capsys)[0] == 3


# ------------------------------------------------------------------ the shipped slice
def test_the_shipped_slice_is_new_golds_with_a_source_and_a_licence_each():
    doc = json.loads(SLICE.read_text(encoding="utf-8"))
    assert doc["schema"] == G.SLICE_SCHEMA and len(doc["golds"]) == 24
    fams = doc["families"]
    # no court decision: its texts stay held until the lawyer's opinion (founder, "keep them held")
    assert set(fams) == {"fedlex_exact", "loda_exact", "dila_kali", "service_public_fiche",
                         "stackexchange_nonso_qa_p0", "openalex_academic_dump", "stackoverflow_qa"}
    assert all(f["source"] and f["licence"] and f["fixture"] for f in fams.values())
    assert {g["family"] for g in doc["golds"]} == set(fams)
    assert all(g["query"] and g["target_url"].startswith("https://") for g in doc["golds"])
    # a gold's target is a row of the fixture stage it names
    import gzip
    for fam in fams:
        with gzip.open(REPO / "data" / "index-fixture" / fam / "chunks" / "shard-0000.texts.jsonl.gz",
                       "rt", encoding="utf-8") as fh:
            urls = {json.loads(x)["url"] for x in fh}
        for g in (g for g in doc["golds"] if g["family"] == fam):
            assert g["target_url"] in urls, g["id"]
    # one query per gold
    queries = {g["query"] for g in doc["golds"]}
    assert len(queries) == 24


# ------------------------------------------------------------------ review of #606: refusals, not verdicts
def _receipt(tmp_path, name, profile="p1", idf="i1", info=None, mode="tei"):
    doc = {"schema": "aw.serve-receipt.v1", "collection": "c", "points_count": 7,
           "profile": {"digest": profile}, "idf_table": {"sha256": idf},
           "query_embedding": {"mode": mode, "info": info}}
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


def test_an_unexpected_exception_is_a_refusal_exit_3_never_exit_1(tmp_path, fake, capsys, monkeypatch):
    s = str(_slice(tmp_path))
    f = fake({"alpha": [A], "beta": [B]})

    def boom(**kw):
        raise RuntimeError("a bug in the gate")
    monkeypatch.setattr(G, "run", boom)
    rc, _, err = _run(["--api", f.url, "--slice", s], capsys)
    assert rc == 3 and "unexpected RuntimeError" in err         # exit 1 means ONLY "a gold regressed"
    monkeypatch.undo()
    # the malformed answers a stranger's server can give: each is a refusal, not a crash
    for bad in ({"results": [{"nourl": 1}]}, {"results": None}, {"results": ["x"]}, {"count": 0}):
        g = fake({"alpha": bad, "beta": [B]})
        rc, _, err = _run(["--api", g.url, "--slice", s], capsys)
        assert rc == 3 and "REFUSED" in err, bad
    # a truncated body is an HTTPException, not an OSError: still a refusal
    import http.client
    import urllib.request

    def cut(*a, **kw):
        raise http.client.IncompleteRead(b"x")
    monkeypatch.setattr(urllib.request, "urlopen", cut)
    with pytest.raises(G.GateRefused):
        G._get_json("http://127.0.0.1:1/x", 1)
    # a baseline without golds does not crash either
    monkeypatch.undo()
    bad_base = tmp_path / "bb.json"
    bad_base.write_text(json.dumps({"schema": G.SCHEMA, "slice": {"sha256": G.sha256_file(s)}}))
    rc, _, err = _run(["--api", f.url, "--slice", s, "--baseline", str(bad_base)], capsys)
    assert rc == 3 and "REFUSED" in err


def test_an_empty_index_is_a_refusal_and_writes_no_verdict(tmp_path, fake, capsys):
    s = str(_slice(tmp_path))
    out = tmp_path / "v.json"
    zero = fake({"alpha": [A], "beta": [B]}, points=0)
    rc, _, err = _run(["--api", zero.url, "--slice", s, "--verdict-out", str(out)], capsys)
    assert rc == 3 and "0 points" in err and not out.exists()
    empty = fake({"alpha": [], "beta": []})                    # a server on an empty collection
    rc, _, err = _run(["--api", empty.url, "--slice", s, "--verdict-out", str(out)], capsys)
    assert rc == 3 and "no result" in err and not out.exists()    # no baseline that looks valid
    good = fake({"alpha": [A], "beta": [B]})
    base = tmp_path / "base.json"
    assert _run(["--api", good.url, "--slice", s, "--verdict-out", str(base)], capsys)[0] == 0
    rc, _, err = _run(["--api", empty.url, "--slice", s, "--baseline", str(base)], capsys)
    assert rc == 3                                             # refused, not a FAIL (exit 1)
    # ONE empty answer among non-empty ones stays a miss
    one = fake({"alpha": [A], "beta": []})
    rc, _, _ = _run(["--api", one.url, "--slice", s, "--baseline", str(base)], capsys)
    assert rc == 1


def test_a_baseline_of_another_identity_is_refused_unless_allowed_and_then_recorded(
        tmp_path, fake, capsys):
    s = str(_slice(tmp_path))
    f = fake({"alpha": [A], "beta": [B]})
    base = tmp_path / "base.json"
    r1 = _receipt(tmp_path, "r1.json", profile="p1", idf="i1")
    assert _run(["--api", f.url, "--slice", s, "--serve-receipt", r1, "--verdict-out", str(base)],
                capsys)[0] == 0
    # the same identity compares
    assert _run(["--api", f.url, "--slice", s, "--serve-receipt", r1, "--baseline", str(base)],
                capsys)[0] == 0
    r2 = _receipt(tmp_path, "r2.json", profile="p1", idf="i2")          # another IDF table
    rc, _, err = _run(["--api", f.url, "--slice", s, "--serve-receipt", r2, "--baseline", str(base)],
                      capsys)
    assert rc == 3 and "'i1'" in err and "'i2'" in err and "idf_table.sha256" in err
    r3 = _receipt(tmp_path, "r3.json", profile="p9", idf="i1")          # another profile digest
    rc, _, err = _run(["--api", f.url, "--slice", s, "--serve-receipt", r3, "--baseline", str(base)],
                      capsys)
    assert rc == 3 and "'p1'" in err and "'p9'" in err and "profile.digest" in err
    out = tmp_path / "v.json"
    rc, _, _ = _run(["--api", f.url, "--slice", s, "--serve-receipt", r2, "--baseline", str(base),
                     "--allow-identity-mismatch", "--verdict-out", str(out)], capsys)
    assert rc == 0
    b = json.loads(out.read_text())["baseline"]
    assert b["identity_mismatch_allowed"] is True
    assert b["identity_mismatches"] == [{"field": "serve_receipt.idf_table.sha256",
                                         "baseline": "i1", "now": "i2"}]
    # a run with no receipt against a baseline with one is a mismatch too
    assert _run(["--api", f.url, "--slice", s, "--baseline", str(base)], capsys)[0] == 3


def test_the_verdict_says_what_it_proves_from_the_embedder_identity(tmp_path, fake, capsys):
    from cytir.core import model_pin
    s = str(_slice(tmp_path))
    f = fake({"alpha": [A], "beta": [B]})

    def proves(receipt):
        out = tmp_path / "v.json"
        args = ["--api", f.url, "--slice", s, "--verdict-out", str(out)]
        rc, stdout, _ = _run(args + (["--serve-receipt", receipt] if receipt else []), capsys)
        assert rc == 0
        return json.loads(out.read_text())["proves"], stdout

    standin = _receipt(tmp_path, "a.json", info={"model_id": model_pin.EMBED_MODEL_ID,
                                                 "model_sha": model_pin.embed_model_revision(),
                                                 "stand_in": "hashed-bow-v1"})
    got, stdout = proves(standin)
    assert got == G.PROVES_PLUMBING and "plumbing-only" in stdout     # a stand-in announces itself
    real = _receipt(tmp_path, "b.json", info={"model_id": model_pin.EMBED_MODEL_ID,
                                              "model_sha": model_pin.embed_model_revision()})
    got, stdout = proves(real)
    assert got == "ranking" and "proves:" not in stdout
    assert proves(_receipt(tmp_path, "c.json", info={"model_id": "other/model", "model_sha": "x"})
                  )[0].startswith("unknown (")
    assert proves(_receipt(tmp_path, "d.json", info=None, mode="in-process")
                  )[0].startswith("unknown (")
    assert proves(None)[0].startswith("unknown (no --serve-receipt")


def test_in_process_embedder_identity_is_proved_and_printed(tmp_path, capsys, fake):
    """D5: an in-process embedder's identity (as `cyt serve` records it) lets the gate prove ranking,
    and the `embedder:` line shows model, revision, device, dtype and library versions."""
    from cytir.core import model_pin
    info = {"model_id": model_pin.EMBED_MODEL_ID, "model_sha": model_pin.embed_model_revision(),
            "device": "cpu", "dtype": "float32", "sentence_transformers": "9.9", "torch": "8.8",
            "source": "pin"}
    rcpt = _receipt(tmp_path, "ip.json", info=info, mode="in-process")
    assert G.embedder_proves(G.receipt_identity(rcpt)) == G.PROVES_RANKING
    s = str(_slice(tmp_path))
    f = fake({"alpha": [A], "beta": [B]})
    rc, stdout, _ = _run(["--api", f.url, "--slice", s, "--serve-receipt", rcpt], capsys)
    assert rc == 0 and "embedder: " in stdout and "cpu/float32" in stdout
    assert "sentence-transformers 9.9 torch 8.8" in stdout and "proves:" not in stdout
    wrong = dict(info, model_sha="deadbeef")
    assert G.embedder_proves(G.receipt_identity(
        _receipt(tmp_path, "ip2.json", info=wrong, mode="in-process"))).startswith("unknown (")
