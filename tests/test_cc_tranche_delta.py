"""`cytir.ops.cc_tranche_delta` (task 698): the delta of a sealed CC tranche is filtered by served URL,
staged, filed to L1, verified, and made visible to the compose timer only by the last rename."""
from __future__ import annotations

import gzip
import json
import pathlib
import re
import sys
from types import SimpleNamespace

import pytest

from tests.helpers.repo_files import repo_file_or_skip

HERE = pathlib.Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.ops import cc_tranche_delta as D  # noqa: E402

# the delta's spam screen reads this list; the public cut does not ship it
repo_file_or_skip(pathlib.Path("data") / "spam-host-lists.v1.json")


def _landed(queue, pack, status="LANDED", urls=4, **extra):
    """A receipt shaped like the drain's: `mechanical_plan.rows.distinct_urls` is the pack's URL count
    (the real t5 receipt: 17,417). `urls=None` leaves the plan out."""
    (queue / pack).mkdir(parents=True, exist_ok=True)
    rec = {"status": status, **extra}
    if urls is not None:
        rec["mechanical_plan"] = {"rows": {"distinct_urls": urls}}
    (queue / pack / "receipt.json").write_text(json.dumps(rec))


def _tree(tmp_path, n=5):
    t = tmp_path / "tranches"
    (t / f"t{n}_out" / "gated").mkdir(parents=True)
    (t / f"t{n}_gpu").mkdir(parents=True)
    (t / f"t{n}_gpu" / ".sealed").write_text("2026-09-26T11:30:18Z\n")
    rows = [{"url": f"https://a.example/{i}", "point_id": f"id{i}", "text": "x"} for i in range(6)]
    (t / f"t{n}_out" / "gated" / "passages_ok.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    _landed(tmp_path / "queue", D.first_pack_id(n))          # the first pack is served
    return tmp_path


class Runner:
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def __call__(self, argv, env=None):
        name = argv[2].rsplit(".", 1)[-1] + ".py"   # `python -m cytir.corpus.<name>`
        self.calls.append(name)
        if name == self.fail_on:
            return SimpleNamespace(returncode=1)
        if name == "stage_cc_tranche.py":
            out = pathlib.Path(argv[argv.index("--out") + 1])
            (out / "all" / "chunks").mkdir(parents=True)
            with gzip.open(out / "all" / "chunks" / "shard-0000.texts.jsonl.gz", "wt") as fh:
                fh.write('{"point_id": "x"}\n')
            (out / "stage_report.json").write_text("{}")
        return SimpleNamespace(returncode=0)


SERVED = {f"https://a.example/{i}" for i in range(4)}


def _run(root, runner, served=lambda urls: {u for u in urls if u in SERVED}, **kw):
    return D.run_delta(5, data_root=root, python_bin="py", served_urls=served, runner=runner,
                       log=lambda m: None, now=lambda: 0, queue_dir=root / "queue", **kw)


def test_the_delta_excludes_served_urls_and_is_sealed_last(tmp_path):
    root = _tree(tmp_path)
    r = Runner()
    out = _run(root, r)
    assert out["delta_rows"] == 2 and out["rows_served_excluded"] == 4 and out["urls_served"] == 4
    assert r.calls == ["stage_cc_tranche.py", "register_l1_release.py", "verify_l1_coverage.py"]
    part = root / "tranches" / "t5_gpu" / "delta"
    assert (part / "chunks" / "shard-0000.texts.jsonl.gz").is_file() and (part / ".sealed").is_file()
    assert not (root / "tranches" / "t5_gpu" / "_delta_stage").exists()
    kept = (root / "tranches" / "t5_out" / "delta" / "passages_ok.jsonl").read_text().splitlines()
    assert [json.loads(x)["point_id"] for x in kept] == ["id4", "id5"]
    assert out["pack_id"] == "cc-authority-frontier-w7-tranche5-delta"
    assert _run(root, Runner())["status"] == "already_sealed"          # resumable no-op


def test_a_failed_step_leaves_no_visible_part(tmp_path):
    root = _tree(tmp_path)
    with pytest.raises(D.DeltaError, match="verify_l1"):
        _run(root, Runner(fail_on="verify_l1_coverage.py"))
    assert not (root / "tranches" / "t5_gpu" / "delta").exists()       # the timer never sees it
    assert _run(root, Runner())["status"] == "sealed"                  # and a rerun completes


def test_an_empty_delta_and_an_unsealed_tranche_refuse(tmp_path):
    root = _tree(tmp_path)
    with pytest.raises(D.DeltaError, match="0 rows"):
        _run(root, Runner(), served=lambda urls: set(urls))
    (root / "tranches" / "t5_gpu" / ".sealed").unlink()
    with pytest.raises(D.DeltaError, match="first seal"):
        _run(root, Runner())


def _grow(root, extra=3, n=5):
    f = root / "tranches" / f"t{n}_out" / "gated" / "passages_ok.jsonl"
    with open(f, "a") as fh:
        for i in range(100, 100 + extra):
            fh.write(json.dumps({"url": f"https://b.example/{i}", "point_id": f"id{i}", "text": "x"}) + "\n")


def test_a_later_part_is_sealed_only_when_the_file_grew_and_the_earlier_part_landed(tmp_path):
    root, queue = _tree(tmp_path), tmp_path / "queue"
    _run(root, Runner())                                               # part 1
    assert _run(root, Runner())["status"] == "already_sealed"          # not grown: no-op
    _grow(root)
    with pytest.raises(D.DeltaError, match="--queue-dir"):             # growth needs the queue check
        D.run_delta(5, data_root=root, python_bin="py", served_urls=lambda u: set(), runner=Runner(),
                    log=lambda m: None, now=lambda: 0)
    _landed(queue, "cc-authority-frontier-w7-tranche5-delta", "READY")
    r = D.run_delta(5, data_root=root, python_bin="py", served_urls=lambda u: set(),
                    runner=Runner(), log=lambda m: None, now=lambda: 0, queue_dir=queue)
    assert r == {"tranche": 5, "status": "waiting_previous_delta",
                 "pack_id": "cc-authority-frontier-w7-tranche5-delta"}
    assert not (root / "tranches" / "t5_gpu" / "delta2").exists()      # rows in flight: not sealed twice
    _landed(queue, "cc-authority-frontier-w7-tranche5-delta")
    r = D.run_delta(5, data_root=root, python_bin="py", served_urls=lambda u: {x for x in u if x in SERVED | {"https://a.example/4", "https://a.example/5"}},
                    runner=Runner(), log=lambda m: None, now=lambda: 0, queue_dir=queue)
    assert r["status"] == "sealed" and r["pack_id"] == "cc-authority-frontier-w7-tranche5-delta2"
    assert (root / "tranches" / "t5_gpu" / "delta2" / ".sealed").is_file()
    assert (root / "tranches" / "t5_gpu" / "delta" / ".sealed").is_file()   # part 1 untouched
    assert r["l1_release"] == "task698-cc-tranche5-w7-delta2"


def test_main_refuses_without_the_nvme_persisted_index_env(tmp_path, monkeypatch):
    monkeypatch.delenv("AW_PERSISTED_INDEX_SQLITE_DIR", raising=False)
    assert D.main(["--data-root", str(tmp_path), "--tranche", "5"]) == 2


def test_screen_delta_replaces_the_delta_file_and_keeps_the_original(tmp_path):
    import json as _json
    import pathlib as _pl
    from cytir.ops import cc_tranche_delta as D
    here = _pl.Path(__file__).resolve().parent.parent / "tests"
    rows = _json.loads((here / "fixtures" / "spam_rows" / "real_host_rows.json").read_text())
    rows = [r for r in rows if r["domain"] in ("sunlitcentrekenya.co.ke", "kungrad.com")]
    d = tmp_path / "delta"
    d.mkdir()
    (d / "passages_ok.jsonl").write_text("".join(_json.dumps(r) + "\n" for r in rows))
    res = D.screen_delta(d, here.parent / "data" / "spam-host-lists.v1.json", log=lambda m: None)
    kept = [_json.loads(x) for x in (d / "passages_ok.jsonl").read_text().splitlines()]
    assert {r["domain"] for r in kept} == {"kungrad.com"} and res["rows_screened_out"] == 25
    assert (d / "passages_ok.prescreen.jsonl").is_file() and (d / "spam_screen.jsonl").is_file()


# ---- task 703 + follow-up: part 1 waits for the first pack to be SERVED; the twin must serve 95 % of
# its URLs; the wait and the refused first pack each have their own dated line

FIRST = "cc-authority-frontier-w7-tranche5"
Q = "queue"


def _first(tmp_path, status, **extra):
    _landed(tmp_path / Q, FIRST, status, **extra)


def _state(tmp_path):
    return D.first_pack_state(tmp_path / Q, 5)


def _waits(root):
    f = root / "tranches" / "t5_gpu" / ".delta-waits"
    return int(f.read_text()) if f.exists() else 0


@pytest.mark.parametrize("status,extra", [
    ("LANDED", {}),
    ("DONE", {"finalized_from": "CARRIED-BY-FLIP"}),
])
def test_a_served_first_pack_opens_part_1(status, extra, tmp_path):
    root = _tree(tmp_path)
    _first(tmp_path, status, **extra)
    assert _state(tmp_path) == {"state": "served", "packs": [FIRST], "expected_urls": 4}
    assert _run(root, Runner())["status"] == "sealed"


@pytest.mark.parametrize("status,extra", [
    (None, {}), ("READY", {}), ("DONE", {}), ("CARRIED-BY-FLIP", {}), ("PARKED", {}),
    ("SUPERSEDED", {}),                                   # no superseded_by list: a human names the packs
])
def test_part_1_waits_while_the_first_pack_is_not_served(status, extra, tmp_path):
    """Tranche 7, 2026-10-04: part 1 ran in the firing that wrote the first `.sealed`, when no row was
    served yet. A `DONE` without the carried-by-flip finalization and a bare `CARRIED-BY-FLIP` are not
    proof of served rows either: they wait."""
    root = _tree(tmp_path)
    if status is None:
        (tmp_path / Q / FIRST / "receipt.json").unlink()
    else:
        _first(tmp_path, status, **extra)
    r, lines = Runner(), []
    out = D.run_delta(5, data_root=root, python_bin="py", served_urls=lambda u: set(), runner=r,
                      log=lines.append, now=lambda: 0, queue_dir=tmp_path / Q)
    assert out["status"] == "waiting_first_pack" and out["pack_id"] == FIRST and not out["loud"]
    assert re.match(r"1970-01-01T00:00:00Z \[cc-tranche-delta\] tranche 5: waiting for the first pack "
                    r"to be served: .*cc-authority-frontier-w7-tranche5.*\(wait 1\)$", lines[0]), lines
    assert r.calls == [] and not (root / "tranches" / "t5_out" / "delta").exists()   # nothing was read
    assert not (root / "tranches" / "t5_gpu" / "delta").exists()


def _tranche2_queue(tmp_path):
    """Tranche 2 as the box holds it (read 2026-10-04): the first pack is SUPERSEDED with prose only (no
    `superseded_by`); `-b` and `-c` are mechanical LANDED receipts with `distinct_urls`; `-a` and `-d` are
    hand-reconciled LANDED receipts with a `note` and NO `mechanical_plan`; `-delta` is READY."""
    q = tmp_path / Q
    _landed(q, FIRST, "SUPERSEDED", urls=None, reason="restage duplicate of tranche2-a..d: 791,041 of "
            "791,305 rows already on twin and prod; 259 new rows to land as a separate pack", task="673")
    for sfx in "ad":
        _landed(q, f"{FIRST}-{sfx}", urls=None, note="task 664, 2026-09-26 manual reconciliation")
    _landed(q, f"{FIRST}-b", urls=49591)
    _landed(q, f"{FIRST}-c", urls=47736)
    _landed(q, f"{FIRST}-delta", "READY")


def test_a_superseded_first_pack_without_a_superseded_by_list_waits_and_asks_a_human(tmp_path):
    """Real tranche 2 shape: siblings are NOT inferred from names; the line says who must act."""
    root = _tree(tmp_path)
    _tranche2_queue(tmp_path)
    st = _state(tmp_path)
    assert st["state"] == "waiting" and "superseded_by" in st["why"] and "a human must name" in st["why"]
    assert "--allow-unserved-first-pack" in st["why"]
    lines = []
    out = D.run_delta(5, data_root=root, python_bin="py", served_urls=lambda u: set(), runner=Runner(),
                      log=lines.append, now=lambda: 0, queue_dir=tmp_path / Q)
    assert out["status"] == "waiting_first_pack" and "a human must name" in lines[0]


def test_an_unrelated_later_pack_is_never_taken_for_a_superseder(tmp_path):
    """`-e` ("259 new rows as a separate pack"), landed, does not open the gate by being a letter."""
    _tranche2_queue(tmp_path)
    _landed(tmp_path / Q, f"{FIRST}-e", urls=259)
    assert _state(tmp_path)["state"] == "waiting"


def test_superseded_by_list_is_served_when_all_named_packs_are_and_waits_naming_one_that_is_not(tmp_path):
    _tranche2_queue(tmp_path)
    named = [f"{FIRST}-{x}" for x in "abcd"]
    _first(tmp_path, "SUPERSEDED", urls=None, superseded_by=named)
    # real receipts: -a and -d have no distinct_urls -> served, but not measurable
    assert _state(tmp_path) == {"state": "served", "packs": named, "expected_urls": None}
    _first(tmp_path, "SUPERSEDED", urls=None, superseded_by=[f"{FIRST}-b", f"{FIRST}-c"])
    assert _state(tmp_path) == {"state": "served", "packs": [f"{FIRST}-b", f"{FIRST}-c"],
                                "expected_urls": 49591 + 47736}
    _landed(tmp_path / Q, f"{FIRST}-c", "READY")
    st = _state(tmp_path)
    assert st["state"] == "waiting" and st["pack"] == f"{FIRST}-c" and "SUPERSEDED" in st["why"]


def test_tranche_2_real_shape_with_the_real_names_is_unmeasurable_not_served(tmp_path):
    """Named `-a`..`-d` (a human wrote the list): `-a`/`-d` carry no distinct_urls, so part 1 is blocked
    as unmeasurable BEFORE the gated file is read, with no twin request."""
    root = _tree(tmp_path)
    _tranche2_queue(tmp_path)
    _first(tmp_path, "SUPERSEDED", urls=None, superseded_by=[f"{FIRST}-{x}" for x in "abcd"])
    calls, lines = [], []
    out = D.run_delta(5, data_root=root, python_bin="py", served_urls=lambda u: calls.append(u) or set(),
                      runner=Runner(), log=lines.append, now=lambda: 0, queue_dir=tmp_path / Q)
    assert out["status"] == "first_pack_unmeasurable" and calls == []
    assert not (root / "tranches" / "t5_out" / "delta").exists()


def test_a_refused_first_pack_stages_nothing_and_names_pack_and_stage_each_firing(tmp_path):
    root = _tree(tmp_path)
    _first(tmp_path, "REFUSED", stage="live-apply-preflight", reason="143290 ids on neither side")
    for _ in range(2):                                    # each firing says it again
        r, lines = Runner(), []
        out = D.run_delta(5, data_root=root, python_bin="py", served_urls=lambda u: set(), runner=r,
                          log=lines.append, now=lambda: 0, queue_dir=tmp_path / Q)
        assert out["status"] == "first_pack_refused" and out["stage"] == "live-apply-preflight"
        assert re.match(r"1970-01-01T00:00:00Z \[cc-tranche-delta\] tranche 5: FIRST PACK REFUSED, no delta "
                        r"is staged: cc-authority-frontier-w7-tranche5 was REFUSED at stage "
                        r"live-apply-preflight", lines[0]), lines
        assert "waiting" not in lines[0] and r.calls == []
    assert not (root / "tranches" / "t5_gpu" / "delta").exists() and _waits(root) == 0


def test_the_wait_turns_loud_at_the_12th_consecutive_firing_and_resets_when_served(tmp_path):
    root = _tree(tmp_path)
    _first(tmp_path, "READY")
    seen = []
    for i in range(1, 14):
        lines = []
        out = D.run_delta(5, data_root=root, python_bin="py", served_urls=lambda u: set(),
                          runner=Runner(), log=lines.append, now=lambda: 0, queue_dir=tmp_path / Q)
        seen.append((i, out["loud"], "LOUD" in lines[0]))
    assert seen[10] == (11, False, False) and seen[11] == (12, True, True) and seen[12] == (13, True, True)
    assert "after 12 firings" in lines[0] or "after 13 firings" in lines[0]
    assert _waits(root) == 13
    _first(tmp_path, "LANDED")                            # served: the count is gone
    assert _run(root, Runner())["status"] == "sealed" and _waits(root) == 0


def test_main_maps_each_blocked_state_to_its_own_unstamped_exit_code(tmp_path, monkeypatch, capsys):
    """3 wait, 5 loud wait, 4 refused first pack, 6/7 below the floor (loud), 8/9 no distinct_urls (loud)."""
    root = _tree(tmp_path)
    monkeypatch.setenv("AW_PERSISTED_INDEX_SQLITE_DIR", str(tmp_path / "pi"))
    monkeypatch.setattr(D, "_twin_served_urls", lambda *a: (lambda urls: set()))
    argv = ["--data-root", str(root), "--tranche", "5", "--queue-dir", str(tmp_path / Q)]
    gpu = root / "tranches" / "t5_gpu"
    _first(tmp_path, "READY")
    assert D.main(argv) == 3
    (gpu / ".delta-waits").write_text(f"{D.LOUD_AFTER_WAITS - 1}\n")
    assert D.main(argv) == 5
    _first(tmp_path, "REFUSED", stage="intake")
    assert D.main(argv) == 4
    _first(tmp_path, "LANDED", urls=1000)
    assert D.main(argv) == 6
    (gpu / ".delta-below-floor").write_text(f"{D.LOUD_AFTER_WAITS - 1}\n")
    assert D.main(argv) == 7
    _first(tmp_path, "LANDED", urls=None)
    assert D.main(argv) == 8
    (gpu / ".delta-unmeasurable").write_text(f"{D.LOUD_AFTER_WAITS - 1}\n")
    assert D.main(argv) == 9
    assert "LOUD" in capsys.readouterr().out


def test_part_1_without_a_queue_dir_is_refused(tmp_path):
    with pytest.raises(D.DeltaError, match="--queue-dir"):
        D.run_delta(5, data_root=_tree(tmp_path), python_bin="py", served_urls=lambda u: set(),
                    runner=Runner(), log=lambda m: None, now=lambda: 0)


def _first_pack_urls(n_pack, n_served):
    """A twin that serves `n_served` of the gated URLs (the first pack holds `n_pack` of them)."""
    return lambda urls: {u for u in urls if int(u.rsplit("/", 1)[1]) < n_served}


def test_the_95_percent_rule_passes_at_the_floor_and_refuses_one_below(tmp_path):
    """Floor = ceil(0.95 x distinct_urls). 100 URLs in the first pack: 95 served passes, 94 refuses."""
    for served, ok in ((95, True), (94, False)):
        root = _tree(tmp_path / str(served))
        gated = root / "tranches" / "t5_out" / "gated" / "passages_ok.jsonl"
        gated.write_text("".join(json.dumps({"url": f"https://a.example/{i}", "point_id": f"id{i}",
                                             "text": "x"}) + "\n" for i in range(150)))
        _landed(root / Q, FIRST, urls=100)
        if ok:
            out = _run(root, Runner(), served=_first_pack_urls(100, served))
            assert out["first_pack_check"] == {"packs": [FIRST], "expected_urls": 100,
                                               "urls_served": 95, "floor": 95}
        else:
            r, lines = Runner(), []
            out = D.run_delta(5, data_root=root, python_bin="py", runner=r, log=lines.append,
                              now=lambda: 0, queue_dir=root / Q,
                              served_urls=_first_pack_urls(100, served))
            msg = lines[-1]
            assert out["status"] == "first_pack_below_floor" and not out["loud"]
            assert re.match(r"1970-01-01T00:00:00Z \[cc-tranche-delta\] tranche 5: ", msg)
            assert "serves 94 gated URLs" in msg and "holds 100" in msg and "floor 95" in msg
            assert "--allow-unserved-first-pack" in msg and r.calls == [] and "(wait 1)" in msg
            assert not (root / "tranches" / "t5_out" / "delta" / "passages_ok.jsonl").exists()
            assert not (root / "tranches" / "t5_gpu" / "delta").exists()


def test_the_tranche_7_incident_is_still_waiting_when_the_first_pack_has_not_landed(tmp_path):
    """t7 as it really was: the first pack was NOT landed (READY) in the firing that wrote the first
    seal and the twin served 1 URL of its 156,560 (scaled: 1 of 1,000). The receipt gate stops it."""
    root = _tree(tmp_path / "t7")
    _landed(root / Q, FIRST, "READY", urls=1000)
    seen, r, lines = [], Runner(), []
    out = D.run_delta(5, data_root=root, python_bin="py", runner=r, log=lines.append, now=lambda: 0,
                      queue_dir=root / Q, served_urls=lambda u: seen.append(u) or {"https://a.example/0"})
    assert out["status"] == "waiting_first_pack" and r.calls == [] and seen == []


def test_the_two_gates_are_an_AND_not_an_OR(tmp_path):
    """(1) first pack READY, the twin serves EVERY gated URL: the receipt gate alone must still WAIT
    (the twin is never asked). (2) first pack LANDED, the twin serves under 95 %: the floor alone
    must still block. A change of the AND to an OR fails one of these."""
    root = _tree(tmp_path / "a")
    _landed(root / Q, FIRST, "READY", urls=6)
    seen, r = [], Runner()
    out = D.run_delta(5, data_root=root, python_bin="py", runner=r, log=lambda m: None, now=lambda: 0,
                      queue_dir=root / Q, served_urls=lambda u: seen.append(u) or set(u))
    assert out["status"] == "waiting_first_pack" and r.calls == [] and seen == []
    root = _tree(tmp_path / "b")
    _landed(root / Q, FIRST, "LANDED", urls=1000)
    out = _run(root, Runner(), served=lambda urls: {"https://a.example/0"})
    assert out["status"] == "first_pack_below_floor"
    assert not (root / "tranches" / "t5_gpu" / "delta").exists()


def test_a_legitimate_big_delta_with_the_first_pack_served_passes(tmp_path):
    """t3/t5: a delta of 86-99 % of the gated rows, with the first pack landed and 100 % served."""
    root = _tree(tmp_path / "t3")                          # 6 rows, 4 URLs served = the whole first pack
    out = _run(root, Runner(), served=lambda urls: {u for u in urls if u.endswith(("/0", "/1", "/2", "/3"))})
    assert out["status"] == "sealed" and out["delta_rows"] == 2
    root = _tree(tmp_path / "big")                         # 83 % of the rows are new, the pack is served
    gated = root / "tranches" / "t5_out" / "gated" / "passages_ok.jsonl"
    gated.write_text("".join(json.dumps({"url": f"https://a.example/{i}", "point_id": f"id{i}",
                                         "text": "x"}) + "\n" for i in range(24)))
    out = _run(root, Runner(), served=lambda urls: {u for u in urls if u.endswith(("/0", "/1", "/2", "/3"))})
    assert out["status"] == "sealed" and out["delta_rows"] == 20


def test_the_floor_block_is_loud_from_the_12th_firing_and_resets_when_it_clears(tmp_path):
    root = _tree(tmp_path)
    _landed(root / Q, FIRST, urls=1000)
    f = root / "tranches" / "t5_gpu" / ".delta-below-floor"
    for i in range(1, 14):
        lines = []
        out = D.run_delta(5, data_root=root, python_bin="py", served_urls=lambda u: set(),
                          runner=Runner(), log=lines.append, now=lambda: 0, queue_dir=root / Q)
        assert out["status"] == "first_pack_below_floor" and out["loud"] == (i >= 12)
        assert ("LOUD" in lines[-1]) == (i >= 12) and f"(wait {i})" in lines[-1]
    assert f.read_text().strip() == "13" and _waits(root) == 0
    _landed(root / Q, FIRST, urls=4)                       # the receipt now matches what the twin serves
    out = _run(root, Runner())
    assert out["status"] == "sealed" and not f.exists()


def test_the_unmeasurable_block_is_loud_from_the_12th_firing_and_changes_of_state_reset_counters(tmp_path):
    root = _tree(tmp_path)
    _landed(root / Q, FIRST, urls=None)
    f = root / "tranches" / "t5_gpu" / ".delta-unmeasurable"
    for i in range(1, 13):
        lines = []
        out = D.run_delta(5, data_root=root, python_bin="py", served_urls=lambda u: set(),
                          runner=Runner(), log=lines.append, now=lambda: 0, queue_dir=root / Q)
    assert out["status"] == "first_pack_unmeasurable" and out["loud"] and "LOUD" in lines[0]
    assert "distinct_urls" in lines[0] and "tranche 5" in lines[0] and "--allow-unserved-first-pack" in lines[0]
    _landed(root / Q, FIRST, "READY")                      # another state: the counter starts again
    D.run_delta(5, data_root=root, python_bin="py", served_urls=lambda u: set(), runner=Runner(),
                log=lambda m: None, now=lambda: 0, queue_dir=root / Q)
    assert not f.exists() and _waits(root) == 1


def test_the_human_override_stages_part_1_below_the_floor(tmp_path):
    root = _tree(tmp_path)
    _landed(tmp_path / Q, FIRST, urls=1000)
    out = _run(root, Runner(), served=lambda urls: set(), allow_unserved_first_pack=True)
    assert out["status"] == "sealed" and out["delta_rows"] == 6 and "first_pack_check" not in out


def test_a_receipt_without_distinct_urls_is_blocked_before_the_gated_file_is_read(tmp_path):
    root = _tree(tmp_path)
    _landed(tmp_path / Q, FIRST, urls=None)               # a hand-reconciled LANDED receipt
    seen, r = [], Runner()
    out = D.run_delta(5, data_root=root, python_bin="py", runner=r, log=lambda m: None, now=lambda: 0,
                      queue_dir=tmp_path / Q, served_urls=lambda u: seen.append(u) or set())
    assert out["status"] == "first_pack_unmeasurable" and seen == [] and r.calls == []
    assert not (root / "tranches" / "t5_out" / "delta").exists()   # no filtered file was written
    out = _run(root, Runner(), served=lambda urls: set(), allow_unserved_first_pack=True)
    assert out["status"] == "sealed"                       # the human override still works


def test_a_tranche_with_a_sealed_part_never_reads_the_first_pack_receipt(tmp_path):
    """Already-sealed tranches (t2-t7 on the box) are untouched: no receipt, no wait count, no read."""
    root = _tree(tmp_path)
    _run(root, Runner())                                   # part 1
    (tmp_path / Q / FIRST / "receipt.json").write_text(json.dumps({"status": "REFUSED", "stage": "x"}))
    before = sorted(p.name for p in (root / "tranches" / "t5_gpu").iterdir())
    assert _run(root, Runner())["status"] == "already_sealed"
    assert sorted(p.name for p in (root / "tranches" / "t5_gpu").iterdir()) == before


def test_main_waits_with_rc_3_for_the_first_pack(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    (tmp_path / Q / FIRST / "receipt.json").unlink()
    monkeypatch.setenv("AW_PERSISTED_INDEX_SQLITE_DIR", str(tmp_path / "pi"))
    argv = ["--data-root", str(root), "--tranche", "5", "--queue-dir", str(tmp_path / Q)]
    assert D.main(argv) == 3
