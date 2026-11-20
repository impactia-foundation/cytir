"""Task 2880: `python3 -m cytir.index.promote_queue` must run the drain in the ONE canonical copy of
the module. Run as `__main__`, the file was a second copy: `QueueError` was two classes and the
drain's per-pack `except QueueError` missed what `url_group_supersede` raised."""
from __future__ import annotations

import runpy
import sys

import pytest

import test_aw_url_group_supersede as T
from cytir.index import promote_queue as Q
from cytir.index import url_group_supersede as UGS

settings = T.settings

MOD = "cytir.index.promote_queue"


def _run_as_dash_m(monkeypatch):
    """Run the module with `-m` semantics. The canonical `main` is replaced by a probe, and argv is
    `--help`, so an entry that does NOT dispatch into the canonical module still stops (argparse
    exits 0) before any queue, Qdrant or network access."""
    seen = []

    def probe(argv=None):
        seen.append(sys.modules[MOD] is Q)
        return 0

    monkeypatch.setattr(Q, "main", probe)
    monkeypatch.setattr(sys, "argv", ["promote_queue", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module(MOD, run_name="__main__", alter_sys=True)
    return seen


def test_dash_m_entry_dispatches_into_the_canonical_module(monkeypatch):
    seen = _run_as_dash_m(monkeypatch)
    assert seen == [True], "the -m entry did not call the canonical module's main()"


def test_the_queue_error_url_group_supersede_raises_is_the_one_the_drain_handler_names(monkeypatch):
    holder = []

    def probe(argv=None):
        canon = sys.modules[MOD]
        # the drain that runs, and the class its `except QueueError` names
        holder.append((canon.drain.__globals__["QueueError"], UGS._q().QueueError))
        return 0

    monkeypatch.setattr(Q, "main", probe)
    monkeypatch.setattr(sys, "argv", ["promote_queue", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module(MOD, run_name="__main__", alter_sys=True)
    assert holder, "the -m entry did not reach the canonical main()"
    handler_cls, raised_cls = holder[0]
    assert handler_cls is raised_cls
    with pytest.raises(handler_cls):
        UGS.declared({"supersede": "bogus"})


def test_the_globals_that_run_main_under_dash_m_hold_the_class_url_group_supersede_raises(monkeypatch):
    """The frame that executes `main` (it builds the parser first) must hold the SAME `QueueError`
    class `url_group_supersede` raises. Run as the old `__main__` copy, that frame's globals held a
    second class; this reads the globals of the frame that really runs, not of the canonical module."""
    import argparse

    seen = []

    def parser_probe(*a, **k):
        seen.append(sys._getframe(1).f_globals)
        raise SystemExit(0)

    monkeypatch.setattr(argparse, "ArgumentParser", parser_probe)
    monkeypatch.setattr(sys, "argv", ["promote_queue", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module(MOD, run_name="__main__", alter_sys=True)
    assert seen, "main() did not build its parser under -m"
    g = seen[0]
    assert g["QueueError"] is UGS._q().QueueError


@T.needs_pack
def test_drain_refuses_a_bad_mode_supersede_pack_per_pack(tmp_path, monkeypatch, settings):
    urls = T.real_pack_urls(monkeypatch)
    T.make_cluster(monkeypatch, [T.old_row_at(sorted(urls.values())[0])])
    T.give_the_pack_a_family(monkeypatch)
    queue, contract = T.drain_fixture(tmp_path, monkeypatch, settings, supersede="bogus")
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda *a, **k: "aw_e8")
    monkeypatch.setattr(Q, "payload_profile_preflight_both", lambda *a, **k: {})
    monkeypatch.setattr(Q, "live_apply", lambda **k: pytest.fail("must not write"))
    res = T.run_drain(queue, contract, tmp_path)
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "supersede_check"
    assert "QueueError" in res[0]["reason"]


# ---- task 2938: the process exit codes of the `-m` entry (a real subprocess: the exit status is the subject) ----

import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _dash_m(*argv):
    return subprocess.run([sys.executable, "-m", MOD, *argv], capture_output=True, text=True, cwd=ROOT, timeout=120)


def _queue_with(tmp_path):
    queue = tmp_path / "queue"
    (queue / "p1").mkdir(parents=True)
    (queue / "p1" / "manifest.json").write_text('{"pack_id": "p1"}')
    return queue


def test_2938_the_dash_m_entry_exits_1_for_a_refusal_that_is_a_system_exit(tmp_path):
    queue = _queue_with(tmp_path)
    r = _dash_m("--queue-dir", str(queue), "--supersede", "nope", "--reason", "r")
    assert r.returncode == 1 and "[REFUSED] no such pack dir" in r.stderr, (r.returncode, r.stderr[-300:])
    r = _dash_m("--queue-dir", str(queue), "--supersede", "p1")
    assert r.returncode == 1 and "--supersede requires --reason" in r.stderr
    assert not (queue / "p1" / "SUPERSEDED").exists()


def test_2938_the_dash_m_entry_exits_0_after_a_verb_and_the_marker_is_on_disk(tmp_path):
    queue = _queue_with(tmp_path)
    r = _dash_m("--queue-dir", str(queue), "--supersede", "p1", "--reason", "dup")
    assert r.returncode == 0, r.stderr[-300:]
    assert (queue / "p1" / "SUPERSEDED").exists()
    assert json.loads((queue / "p1" / "receipt.json").read_text())["status"] == "SUPERSEDED"
