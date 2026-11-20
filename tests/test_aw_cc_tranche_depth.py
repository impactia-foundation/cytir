"""`cytir.ops.cc_tranche_depth` (task 2194): the depth-continuation band, roster, resolver narrowing and gate."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cytir.ops import cc_tranche_depth as D  # noqa: E402


def _recs(n):
    # shallowest first, as the resolver returns them; distinct depth/length so the order is total
    return [{"url": f"https://h.example/{'a/' * (i % 4)}p{i:03d}", "offset": i, "length": 1,
             "warc_filename": "f", "digest": f"d{i}"} for i in range(n)]


def test_band_is_new_selection_minus_old_and_never_refetches():
    recs = _recs(50)
    old = {r["url"] for r in D.select(recs, 10)}
    new = D.band(recs, 10, 25)
    assert len(new) == 15 and not ({r["url"] for r in new} & old)
    assert {r["url"] for r in new} | old == {r["url"] for r in D.select(recs, 25)}


def test_band_empty_when_host_holds_nothing_beyond_old_depth():
    assert D.band(_recs(8), 10, 25) == []


def _tranche(root: Path, t: int, hosts: dict, declined=(), unknown=(), gate=True):
    out = root / f"t{t}_out"
    (out / "gated").mkdir(parents=True)
    (out / "state.json").write_text(json.dumps(
        {"done": {h: {"status": "ok", "pages": p, "rows": 9} for h, p in hosts.items()}}))
    (root / f"tranche{t}_never_queued.json").write_text(json.dumps(
        [{"domain": h, "class": "edu", "class_basis": "endorsed"} for h in hosts]))
    if gate:
        (out / "gated" / "gate_summary.json").write_text(json.dumps(
            {"declined_hosts": [{"host": h} for h in declined], "unknown_hosts": [{"host": h} for h in unknown]}))


def test_roster_is_seeded_skips_consent_and_uncapped_and_counts(tmp_path):
    _tranche(tmp_path, 7, {f"a{i}.example": 60 for i in range(30)} | {"short.example": 40},
             declined=["a1.example"], unknown=["a2.example"])
    _tranche(tmp_path, 8, {"nogate.example": 60}, gate=False)
    rows, rep = D.build_roster(tmp_path, n=10, seed=1)
    again, _ = D.build_roster(tmp_path, n=10, seed=1)
    assert rows == again and len(rows) == 10
    names = {r["domain"] for r in rows}
    assert not names & {"a1.example", "a2.example", "short.example", "nogate.example"}
    assert rep["skipped"] == {"not_cap60_capped": 1, "consent_declined": 1, "consent_unknown": 1,
                              "no_gate_summary": 1, "tranche_missing": 12}
    assert all(r["band_from"] == 60 and r["band_to"] == 120 and r["max_pages"] == 60 for r in rows)


def test_roster_exclude_hosts_never_appear_and_are_counted(tmp_path):
    _tranche(tmp_path, 7, {f"a{i}.example": 60 for i in range(30)})
    base_rows, base_rep = D.build_roster(tmp_path, n=10, seed=3)
    banned = {r["domain"] for r in base_rows[:5]}
    rows, rep = D.build_roster(tmp_path, n=10, seed=3, exclude_hosts=banned)
    assert len(rows) == 10 and not {r["domain"] for r in rows} & banned
    assert rep["excluded"] == 5 and rep["excluded_hosts_given"] == 5
    assert rows == D.build_roster(tmp_path, n=10, seed=3, exclude_hosts=banned)[0]


def test_exclude_hosts_are_normalised_with_the_pool_host_key(tmp_path):
    _tranche(tmp_path, 7, {"example.com": 60, **{f"a{i}.example": 60 for i in range(20)}})
    rows, rep = D.build_roster(tmp_path, n=30, seed=4, exclude_hosts={"Example.COM."})
    assert "example.com" not in {r["domain"] for r in rows} and rep["excluded"] == 1


def test_cli_report_names_each_exclude_roster_with_its_sha256(tmp_path):
    tr = tmp_path / "tr"
    tr.mkdir()
    _tranche(tr, 7, {"example.com": 60, "b.example": 60})
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps([{"domain": "Example.COM."}]))
    out, rep = tmp_path / "roster.json", tmp_path / "rep.json"
    assert D.main(["roster", "--tranches-dir", str(tr), "--out", str(out), "--report", str(rep),
                   "--n", "10", "--seed", "1", "--exclude-roster", str(ex)]) == 0
    r = json.loads(rep.read_text())
    assert r["exclude_rosters"] == [{"path": str(ex), "sha256": hashlib.sha256(ex.read_bytes()).hexdigest()}]
    assert [x["domain"] for x in json.loads(out.read_text())] == ["b.example"]


def test_roster_without_exclude_is_unchanged_and_report_has_no_new_keys(tmp_path):
    _tranche(tmp_path, 7, {f"a{i}.example": 60 for i in range(30)})
    rows, rep = D.build_roster(tmp_path, n=10, seed=3)
    rows_none, rep_none = D.build_roster(tmp_path, n=10, seed=3, exclude_hosts=None)
    rows_empty, rep_empty = D.build_roster(tmp_path, n=10, seed=3, exclude_hosts=set())
    assert rows == rows_none == rows_empty and rep == rep_none == rep_empty
    assert "excluded" not in rep


def test_band_resolver_narrows_what_fetch_asks_for_and_restores(monkeypatch):
    from cytir.acquire import cc_columnar as ccx
    recs = _recs(50)
    monkeypatch.setattr(ccx, "resolve_domains",
                        lambda crawl, doms, con=None, per_domain=None, **kw: {d: recs for d in doms})
    orig = ccx.resolve_domains
    q = [{"domain": "h.example", "band_from": 10, "band_to": 25, "max_pages": 15}]
    with D.band_resolver(q):
        got = ccx.resolve_domains("C", ["h.example", "other.example"], per_domain=30)
    assert len(got["h.example"]) == 15 and len(got["other.example"]) == 50
    assert ccx.resolve_domains is orig


def test_fetch_refuses_a_queue_without_a_band(tmp_path):
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"domain": "x.example", "max_pages": 60}]))
    assert D.run_fetch(["--queue", str(q)]) == 2


def test_limit_domains_run_writes_the_probe_marker_before_fetching(tmp_path, monkeypatch):
    from cytir.acquire import discovery_ingest as di
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"domain": "a.example", "band_from": 0, "band_to": 5},
                             {"domain": "b.example", "band_from": 0, "band_to": 5}]))
    d = tmp_path / "t22_out"
    seen = {}
    monkeypatch.setattr(di, "main", lambda: seen.update(marker=(d / ".probe").read_text()))
    argv = ["--queue", str(q), "--out", str(d / "passages.jsonl"), "--state", str(d / "state.json"),
            "--limit-domains", "1"]
    assert D.run_fetch(argv) == 0
    assert seen["marker"].startswith("limit=1 roster=2 time=")
    assert not (d / "passages.jsonl").is_dir()


def test_full_run_removes_a_stale_probe_only_when_resuming_over_state(tmp_path, monkeypatch):
    from cytir.acquire import discovery_ingest as di
    monkeypatch.setattr(di, "main", lambda: None)
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"domain": "a.example", "band_from": 0, "band_to": 5}]))
    d = tmp_path / "t22_out"
    d.mkdir()
    (d / ".probe").write_text("limit=1 roster=1 time=x\n")
    argv = ["--queue", str(q), "--out", str(d / "passages.jsonl"), "--state", str(d / "state.json")]
    D.run_fetch(argv)   # no state file: not a resume, marker stays
    assert (d / ".probe").exists()
    (d / "state.json").write_text("{}")
    D.run_fetch(argv)   # resume over the state file: marker goes
    assert not (d / ".probe").exists()
    assert not (d / "passages.jsonl").is_dir()


def _gate_inputs(tmp_path, rows_per_page, dup=0):
    queue = [{"domain": f"h{i}.example"} for i in range(4)]
    state = {"done": {q["domain"]: {"pages": 10, "rows": int(10 * rows_per_page)} for q in queue}}
    p = tmp_path / "passages.jsonl"
    with p.open("w") as f:
        for i, q in enumerate(queue):
            for j in range(int(10 * rows_per_page)):
                f.write(json.dumps({"domain": q["domain"], "url": f"https://{q['domain']}/{j // 3}",
                                    "digest": f"{i}-{j}" if j >= dup else "same"}) + "\n")
    return queue, state, p


def test_gate_go_and_stop_on_each_floor(tmp_path):
    q, s, p = _gate_inputs(tmp_path, 4)
    assert D.gate(q, s, p, first=4, served_urls=lambda urls: set())["verdict"] == "GO"
    q, s, p = _gate_inputs(tmp_path, 1.5)
    assert D.gate(q, s, p, first=4)["verdict"] == "STOP"
    q, s, p = _gate_inputs(tmp_path, 4, dup=20)
    r = D.gate(q, s, p, first=4)
    assert r["landed_share"] < 0.8 and r["verdict"] == "STOP"
    q, s, p = _gate_inputs(tmp_path, 4)
    r = D.gate(q, s, p, first=4, served_urls=lambda urls: set(urls))
    assert r["served_rows"] == r["rows_read"] and r["verdict"] == "STOP"


def test_gate_without_twin_stops_with_reason_and_exit_3(tmp_path, capsys):
    q, s, _ = _gate_inputs(tmp_path, 4)
    (tmp_path / "queue.json").write_text(json.dumps(q))
    (tmp_path / "state.json").write_text(json.dumps(s))
    rc = D.main(["gate", "--queue", str(tmp_path / "queue.json"), "--out-dir", str(tmp_path), "--first", "4"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 3
    assert out["verdict"] == "STOP" and out["reason"] == "served-url check not run (pass --twin)"
    assert out["served_checked"] is False


def test_gate_waits_until_enough_hosts_are_done(tmp_path):
    q, s, p = _gate_inputs(tmp_path, 4)
    s["done"].pop("h3.example")
    assert D.gate(q, s, p, first=4)["verdict"] == "STOP"
