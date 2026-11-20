"""Task 766 -- probe tooling: identity-matching scorer, runner outcome accounting,
identity-seeded (date-independent) parameter draw. Fixtures / a fake local HTTP
server only: no prod, no twin, no network beyond 127.0.0.1."""
import http.server
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.parse

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from cytir.probe import run as probe_run  # noqa: E402
from cytir.probe import score_sliceb as sc  # noqa: E402

RUN = os.path.join(REPO, "probes", "runs", "2026-10-08")
POOL = os.path.join(REPO, "probes", "slice_b_pool.jsonl")


# ------------------------------------------------------------------ scorer --
def _rule_url(rule_kind, value, url):
    return sc.rule_matches({"kind": rule_kind, "value": value}, url)


def test_rule_kinds_and_normalisation():
    assert _rule_url("url", "https://www.gesetze-im-internet.de/bgb/__573.html",
                     "http://gesetze-im-internet.de/bgb/__573.html")
    assert not _rule_url("url", "https://www.gesetze-im-internet.de/bgb/__573.html",
                         "https://www.gesetze-im-internet.de/bgb/__573a.html")
    assert _rule_url("prefix", "https://eur-lex.europa.eu/eli/reg/2024/1689/",
                     "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng#art_53")
    assert not _rule_url("prefix", "https://eur-lex.europa.eu/eli/reg/2024/1689/",
                         "https://eur-lex.europa.eu/eli/reg/2016/679/oj")
    rx = r"^gesetze-im-internet\.de/aufenthg(_2004)?/__9\.html$"
    assert _rule_url("regex", rx, "https://www.gesetze-im-internet.de/aufenthg_2004/__9.html")
    assert not _rule_url("regex", rx, "https://www.gesetze-im-internet.de/aufenthg_2004/__91k.html")
    assert not _rule_url("regex", rx, "https://www.gesetze-im-internet.de/nutzev/__9.html")
    with pytest.raises(ValueError):
        _rule_url("glob", "*", "https://a.b/c")


def _rec(urls, gold, rules=None, k=10, in_scope="yes", **kw):
    return {"qid": "q", "probe_id": "p", "gold_domains": gold, "gold_match": rules,
            "in_scope": in_scope, "params": {"k": k}, "n_results": len(urls),
            "results": [{"rank": i + 1, "url": u} for i, u in enumerate(urls)], **kw}


def test_host_only_is_not_a_hit():
    rule = [{"kind": "regex", "value": r"/__9\.html$"}]
    r = sc.score_record(_rec(["https://x.org/a", "https://gesetze-im-internet.de/aufenthg_2004/__91k.html"],
                             ["gesetze-im-internet.de"], rule))
    assert r["verdict"] == "host-only" and r["hit_rank"] is None and r["host_only_rank"] == 2
    r = sc.score_record(_rec(["https://gesetze-im-internet.de/aufenthg_2004/__91k.html",
                              "https://www.gesetze-im-internet.de/aufenthg_2004/__9.html"],
                             ["gesetze-im-internet.de"], rule))
    assert r["verdict"] == "hit" and r["hit_rank"] == 2 and r["match_rule"] == "identity"


def test_no_rule_keeps_host_semantics_and_borderline_host_only_is_exempt():
    r = sc.score_record(_rec(["https://www.migros.ch/en"], ["migros.ch"]))
    assert r["verdict"] == "hit" and r["match_rule"] == "host(default)"
    r = sc.score_record(_rec(["https://iso.org/x"], ["iso.org"], [{"kind": "url", "value": "iso.org/9001"}],
                             in_scope="borderline"))
    assert r["verdict"] == "miss-exempt"


def test_deep_identity_hit_and_no_data_verdict():
    rule = [{"kind": "url", "value": "https://a.org/x"}]
    urls = [f"https://z.org/{i}" for i in range(11)] + ["https://a.org/x"]
    assert sc.score_record(_rec(urls, ["a.org"], rule, k=10))["verdict"] == "deep-hit"
    r = sc.score_record(_rec([], ["a.org"], rule, outcome="timeout", error="timed out"))
    assert r["verdict"] == "no-data"


def test_pool_declares_rules_without_touching_golds():
    if not os.path.isfile(POOL):
        pytest.skip("repo-only artefact probes/slice_b_pool.jsonl not present (public cut)")
    rows = [json.loads(l) for l in open(POOL)]
    ruled = {r["qid"]: r for r in rows if r.get("gold_match")}
    assert {"sb-de-law-aufenthg9", "sb-ch-law-335c", "sb-sw-spec-jwt"} <= set(ruled)
    for r in ruled.values():
        for rule in r["gold_match"]:
            assert rule["kind"] in ("url", "prefix", "regex") and rule["value"]
    assert ruled["sb-de-law-aufenthg9"]["gold_domains"] == ["gesetze-im-internet.de"]


def test_rescore_recorded_2026_10_08():
    """The recorded run: b021 (AufenthG §9) flips hit r3 -> host-only; the only
    other changes are the identity-rule consequences listed in the task doc."""
    if not (os.path.isfile(POOL) and os.path.isdir(RUN)):
        pytest.skip("repo-only artefact probes/runs/2026-10-08 not present (public cut)")
    rules = sc.load_pool_rules(POOL)
    old = {json.loads(l)["qid"]: json.loads(l) for l in open(os.path.join(RUN, "sliceb_scores.jsonl"))}
    new = {}
    for l in open(os.path.join(RUN, "results_sliceb.jsonl")):
        s = sc.score_record(json.loads(l), rules)
        new[s["qid"]] = s
    changed = {q: (old[q]["verdict"], old[q]["hit_rank"], new[q]["verdict"], new[q]["hit_rank"])
               for q in old if (old[q]["verdict"], old[q]["hit_rank"]) != (new[q]["verdict"], new[q]["hit_rank"])}
    # re-pinned for the task-2001 gold rules (#1510): sb-ch-energy-nuclear
    assert changed == {
        "sb-de-law-aufenthg9": ("hit", 3, "host-only", None),
        "sb-ch-energy-nuclear": ("hit", 4, "host-only", None),  # bfe.admin.ch / ensi.ch rule
        "sb-sw-spec-jwt": ("hit", 1, "host-only", None),     # r1 is RFC 9068, not RFC 7519
        "sb-ch-law-335c": ("hit", 1, "hit", 3),               # r1 is Art. 335b
        "sb-ch-law-257-de": ("hit", 1, "hit", 6),             # r1 is Art. 257e
    }


# ------------------------------------------------------------------- draw ---
def test_profile_is_fixed_per_identity_not_per_day():
    a = {"slice": "B", "qid": "sb-x", "probe_id": "2026-10-08-b003", "query": "q", "lang": "de",
         "country": "CH"}
    b = dict(a, probe_id="2026-10-09-b017")            # next day, different index
    assert probe_run.draw_profile(a) == probe_run.draw_profile(b)
    assert probe_run.params_seed(a) == probe_run.params_seed(b)
    assert "qid:sb-x" in probe_run.params_seed(a)
    # Slice A: identity is the normalized query text, not the per-day index
    x = {"slice": "A", "probe_id": "2026-10-08-001", "query": "  Wie Einen  Obstbaum "}
    y = {"slice": "A", "probe_id": "2026-10-09-007", "query": "wie einen obstbaum"}
    assert probe_run.draw_profile(x) == probe_run.draw_profile(y)
    # different identities still spread over the profile space
    ks = {probe_run.draw_profile(dict(a, qid=f"sb-{i}"))["k"] for i in range(60)}
    assert len(ks) >= 4


# ----------------------------------------------------------------- runner ---
class _H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("q", [""])[0]
        if q.startswith("slow"):
            time.sleep(self.server.slow_s)
        if q.startswith("boom"):
            self.send_response(500); self.end_headers(); self.wfile.write(b"nope"); return
        body = json.dumps({"results": [] if q.startswith("none") else
                           [{"url": "https://a.org/1", "snippet": "s"}], "authorities": []}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    srv.handle_error = lambda *a: None      # clients that timed out leave broken pipes
    srv.slow_s = 3
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def _sample(tmp_path, queries):
    d = tmp_path / "2026-10-08"
    d.mkdir()
    with open(d / "sample.jsonl", "w") as f:
        for i, q in enumerate(queries):
            f.write(json.dumps({"query": q, "probe_id": f"2026-10-08-{i:03d}", "slice": "A"}) + "\n")
    return d


def _run(tmp_path, srv, *extra):
    cmd = [sys.executable, "-m", "cytir.probe.run", "--date", "2026-10-08",
           "--runs-dir", str(tmp_path), "--api", f"http://127.0.0.1:{srv.server_port}",
           "--sleep", "0", "--timeout", "0.4", "--retries", "1", "--backoff", "0", *extra]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO)


def test_runner_records_every_outcome_with_a_timing_out_server(tmp_path, server):
    qs = ["fine one", "slow one", "none here", "boom", "fine two", "slow two"]
    d = _sample(tmp_path, qs)
    p = _run(tmp_path, server)
    out, err = p.communicate(timeout=60)
    recs = [json.loads(l) for l in open(d / "results.jsonl")]
    assert [r["outcome"] for r in recs] == ["ok", "timeout", "empty", "error", "ok", "timeout"]
    assert all(r["probe_id"] for r in recs) and recs[1]["error"] and recs[3]["error"]
    assert recs[1]["params_seed"].startswith("aw-probe-params:v")
    s = json.load(open(d / "run_summary.json"))
    assert s["state"] == "complete" and s["expected"] == 6 and s["recorded"] == 6
    assert s["outcomes"] == {"ok": 2, "timeout": 2, "empty": 1, "error": 1}
    assert s["complete"] is False and s["gaps"] == 3         # 2 timeouts + 1 error say so
    assert "RUN HAS GAPS" in err
    # a timeout is not a latency sample
    from cytir.probe.latency_check import percentile50  # noqa: F401  (import must keep working)


def test_clean_run_says_complete(tmp_path, server):
    d = _sample(tmp_path, ["a", "b", "c"])
    _run(tmp_path, server).communicate(timeout=60)
    s = json.load(open(d / "run_summary.json"))
    assert s["complete"] is True and s["gaps"] == 0 and s["state"] == "complete"


def test_sigterm_mid_run_accounts_for_unreached_queries(tmp_path, server):
    server.slow_s = 0.0
    qs = ["a", "b", "slow c", "d", "e", "f"]
    d = _sample(tmp_path, qs)
    server.slow_s = 5                       # the 3rd query hangs -> kill the runner there
    p = _run(tmp_path, server, "--timeout", "30")
    for _ in range(100):
        if os.path.exists(d / "results.jsonl") and len(open(d / "results.jsonl").readlines()) >= 2:
            break
        time.sleep(0.1)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=30)
    recs = [json.loads(l) for l in open(d / "results.jsonl")]
    assert len(recs) == 6, "every sampled query has a record, none silently dropped"
    assert [r["outcome"] for r in recs][:2] == ["ok", "ok"]
    assert {r["outcome"] for r in recs[2:]} == {"not-run"}
    assert "SIGTERM" in recs[-1]["error"]
    s = json.load(open(d / "run_summary.json"))
    assert s["state"] == "interrupted" and s["complete"] is False and s["gaps"] == 4


def test_check_detects_a_sigkilled_run_from_the_files_alone(tmp_path, server):
    """The 2026-10-08 shape: 24 sampled, 4 recorded, nobody said anything."""
    d = _sample(tmp_path, [f"q{i}" for i in range(24)])
    with open(d / "results.jsonl", "w") as f:
        for i in range(4):
            f.write(json.dumps({"probe_id": f"2026-10-08-{i:03d}", "n_results": 3, "error": None}) + "\n")
    p = subprocess.run([sys.executable, "-m", "cytir.probe.run", "--check",
                        "--date", "2026-10-08", "--runs-dir", str(tmp_path)],
                       capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 4 and "4/24 recorded" in p.stderr and "RUN HAS GAPS" in p.stderr
    # ... and the scorer emits an explicit no-data line per missing sampled query
    d2 = tmp_path / "2026-10-08"
    with open(d2 / "sample_sliceb.jsonl", "w") as f:
        f.write(json.dumps({"probe_id": "p0", "qid": "sb-x", "query": "q", "gold_domains": ["a.org"]}) + "\n")
    open(d2 / "results_sliceb.jsonl", "w").close()
    subprocess.run([sys.executable, "-m", "cytir.probe.score_sliceb", "--date",
                    "2026-10-08", "--runs-dir", str(tmp_path), "--pool", str(tmp_path / "none")],
                   capture_output=True, text=True, check=True, cwd=REPO)
    lines = [json.loads(l) for l in open(d2 / "sliceb_scores.jsonl")]
    assert lines[0]["verdict"] == "no-data" and lines[0]["outcome"] == "missing"


def test_second_same_day_run_is_refused_and_force_preserves_the_old_file(tmp_path, server):
    d = _sample(tmp_path, ["a", "b"])
    _run(tmp_path, server).communicate(timeout=60)
    first = open(d / "results.jsonl").read()
    p = _run(tmp_path, server)
    out, err = p.communicate(timeout=60)
    assert p.returncode == 3
    assert "REFUSED" in err and "2 records" in err and "started 20" in err
    assert open(d / "results.jsonl").read() == first          # untouched
    p = _run(tmp_path, server, "--force")
    p.communicate(timeout=60)
    assert p.returncode == 0
    kept = [f for f in os.listdir(d) if f.startswith("results.jsonl.superseded-")]
    assert len(kept) == 1 and kept[0].endswith("Z")
    assert open(d / kept[0]).read() == first                   # nothing lost
    assert len(open(d / "results.jsonl").readlines()) == 2


def test_zero_sampled_queries_is_refused_not_complete(tmp_path):
    # task 1058: Slice A sampled 0 for two days and was accepted as complete:true.
    d = tmp_path / "2026-10-08"
    d.mkdir()
    (d / "sample.jsonl").write_text("")
    cov = probe_run.coverage([], [])
    assert cov["complete"] is False and "reason" in cov and cov["gaps"] >= 1
    p = subprocess.run([sys.executable, "-m", "cytir.probe.run", "--date",
                        "2026-10-08", "--runs-dir", str(tmp_path), "--check"],
                       capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 4
    p = subprocess.run([sys.executable, "-m", "cytir.probe.run", "--date",
                        "2026-10-08", "--runs-dir", str(tmp_path)],
                       capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "0 queries" in p.stderr
    assert json.load(open(d / "run_summary.json"))["complete"] is False


def test_sampler_exits_nonzero_when_pools_exhausted(tmp_path, monkeypatch):
    from cytir.probe import sample as probe_sample
    monkeypatch.setattr(probe_sample, "PROBES", str(tmp_path))
    monkeypatch.setattr(probe_sample, "load_datasets", lambda: [])
    monkeypatch.setattr(probe_sample, "load_seen", lambda: set())
    monkeypatch.setattr(probe_sample, "sample_adapted", lambda n, rng, seen: [])
    monkeypatch.setattr(sys, "argv", ["probe_sample", "--date", "2026-10-08"])
    with pytest.raises(SystemExit) as e:
        probe_sample.main()
    assert e.value.code == 4
    s = json.load(open(tmp_path / "runs" / "2026-10-08" / "run_summary.json"))
    assert s["complete"] is False and "exhausted" in s["reason"]



def _write_sample(tmp_path, day, queries):
    d = tmp_path / "runs" / day
    d.mkdir(parents=True)
    with open(d / "sample.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps({"query": q, "origin": "verbatim", "dataset": "x", "lang": "en"}) + "\n")


def test_reprobe_tops_up_oldest_first_when_pools_exhausted(tmp_path, monkeypatch):
    """task 1482: fresh pools empty -> re-probe queries last probed >= 30 d ago, oldest first."""
    from cytir.probe import sample as probe_sample
    _write_sample(tmp_path, "2026-08-01", ["old a", "old b"])
    _write_sample(tmp_path, "2026-09-01", ["Old A", "mid c"])   # old a re-probed 09-01
    _write_sample(tmp_path, "2026-10-20", ["recent d"])         # inside the 30 d window
    monkeypatch.setattr(probe_sample, "PROBES", str(tmp_path))
    monkeypatch.setattr(probe_sample, "SEEN", str(tmp_path / "seen.txt"))
    monkeypatch.setattr(probe_sample, "load_datasets", lambda: [])
    monkeypatch.setattr(probe_sample, "load_seen", lambda: {"old a", "old b", "mid c", "recent d"})
    monkeypatch.setattr(probe_sample, "sample_adapted", lambda n, rng, seen: [])
    monkeypatch.setattr(sys, "argv", ["probe_sample", "--date", "2026-10-30", "--n", "2", "--reprobe-min-age-days", "30"])
    probe_sample.main()
    rows = [json.loads(x) for x in open(tmp_path / "runs" / "2026-10-30" / "sample.jsonl")]
    assert [r["query"] for r in rows] == ["old b", "mid c"]
    assert all(r["origin"] == "reprobe" for r in rows)
    assert rows[0]["reprobe_of"] == "2026-08-01"
    assert open(tmp_path / "seen.txt").read() == ""  # ledger unchanged by re-probes


def test_reprobe_disabled_still_refuses(tmp_path, monkeypatch):
    from cytir.probe import sample as probe_sample
    _write_sample(tmp_path, "2026-08-01", ["old a"])
    monkeypatch.setattr(probe_sample, "PROBES", str(tmp_path))
    monkeypatch.setattr(probe_sample, "load_datasets", lambda: [])
    monkeypatch.setattr(probe_sample, "load_seen", lambda: set())
    monkeypatch.setattr(probe_sample, "sample_adapted", lambda n, rng, seen: [])
    monkeypatch.setattr(sys, "argv", ["probe_sample", "--date", "2026-10-30"])
    with pytest.raises(SystemExit) as e:
        probe_sample.main()
    assert e.value.code == 4
