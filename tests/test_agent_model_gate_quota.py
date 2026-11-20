"""Quota tiers in the agent gate and the pace sensor parser (task 979)."""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.helpers.repo_files import repo_file_or_skip, repo_glob_or_skip

ROOT = Path(__file__).resolve().parents[1]
GATE = repo_file_or_skip("scripts/hooks/agent_model_gate.py")
PACE = repo_glob_or_skip("deploy/*/quota-pace.py")[0]

METER = """Current session: 2% used · resets Oct 17, 1:10pm (UTC)
Current week (all models): 51% used · resets Oct 19, 11am (UTC)
Current week (Fable): 33% used · resets Oct 19, 11am (UTC)
"""


def state(tmp_path, tier, pace=1.0, relaxed=False, age_h=0.0):
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - age_h * 3600))
    f = tmp_path / "state.json"
    f.write_text(json.dumps({"at": at, "week_pct": 50, "fable_pct": 30, "session_pct": 1,
                             "resets_at": at, "elapsed_pct": 50, "pace": pace, "tier": tier,
                             "relaxed": relaxed}))
    return f


def run_gate(f, subagent, prompt, model=None):
    ti = {"subagent_type": subagent, "prompt": prompt}
    if model:
        ti["model"] = model
    env = dict(os.environ, AW_QUOTA_STATE=str(f))
    r = subprocess.run([sys.executable, "-I", str(GATE)], input=json.dumps({"tool_name": "Agent", "tool_input": ti}),
                       capture_output=True, text=True, env=env)
    return r.returncode, r.stderr


def test_amber_opus(tmp_path):
    f = state(tmp_path, "amber", 1.0)
    base = "OPUS-REASON: novel gate\nCONTEXT BUDGET: 80 calls\n"
    rc, err = run_gate(f, "opus-deep", base)
    assert rc == 2 and err.count("[quota amber pace=1.00 week=50% elapsed=50%]") == 1
    assert run_gate(f, "opus-deep", base + "QUOTA-OVERRIDE: needed\n")[0] == 0


def test_amber_budget(tmp_path):
    f = state(tmp_path, "amber", 1.0)
    rc, err = run_gate(f, "sonnet-exec", "WHY-NOT-HAIKU: x\nCONTEXT BUDGET: 150 calls\n")
    assert rc == 2 and "amber" in err


def test_red(tmp_path):
    f = state(tmp_path, "red", 1.3)
    sonnet = "WHY-NOT-HAIKU: x\nCONTEXT BUDGET: 40 calls\n"
    assert run_gate_effort(f, "sonnet-exec", sonnet, "medium")[0] == 2
    assert run_gate_effort(f, "sonnet-exec", sonnet + "QUOTA-OVERRIDE: y\n", "medium")[0] == 0
    # red: a launch without an effort field is refused (task 981); `effort: low` passes (see below)
    rc, err = run_gate(f, "haiku-exec", "CONTEXT BUDGET: 40 calls\n")
    assert rc == 2 and "missing effort field" in err
    assert run_gate(f, "haiku-exec", "CONTEXT BUDGET: 150 calls\n")[0] == 2
    opus = "OPUS-REASON: z\nQUOTA-OVERRIDE: y\nCONTEXT BUDGET: 40 calls\n"
    assert run_gate(f, "opus-deep", opus)[0] == 2


def test_relaxed_green_skips_why_not_haiku(tmp_path):
    p = "CONTEXT BUDGET: 150 calls\n"
    assert run_gate(state(tmp_path, "green", 0.4, relaxed=True), "sonnet-exec", p)[0] == 0
    assert run_gate(state(tmp_path, "green", 0.7), "sonnet-exec", p)[0] == 2


def test_stale_state_is_green(tmp_path):
    f = state(tmp_path, "red", 1.5, age_h=4)
    assert run_gate(f, "sonnet-exec", "WHY-NOT-HAIKU: x\nCONTEXT BUDGET: 150 calls\n")[0] == 0


def test_missing_and_garbage_state_is_green(tmp_path):
    p = "WHY-NOT-HAIKU: x\nCONTEXT BUDGET: 150 calls\n"
    assert run_gate(tmp_path / "nope.json", "sonnet-exec", p)[0] == 0
    g = tmp_path / "g.json"
    g.write_text("{not json")
    assert run_gate(g, "sonnet-exec", p)[0] == 0


def test_pace_parse(tmp_path):
    spec = importlib.util.spec_from_file_location("quota_pace", PACE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from datetime import datetime, timezone
    now = datetime(2026, 10, 17, 8, 30, tzinfo=timezone.utc)
    s = m.parse(METER, now)
    assert abs(s["elapsed_pct"] - 70) < 1 and abs(s["pace"] - 0.73) < 0.02
    assert s["tier"] == "green" and s["relaxed"] is False and s["fable_pct"] == 33
    f = tmp_path / "m.txt"
    f.write_text(METER)
    r = subprocess.run([sys.executable, "-I", str(PACE), "--parse", str(f)], capture_output=True, text=True,
                       env=dict(os.environ, AW_QUOTA_NOW="2026-10-17T08:30:00Z"))
    assert r.returncode == 0 and json.loads(r.stdout)["tier"] == "green"


def run_gate_effort(f, subagent, prompt, effort, model=None):
    ti = {"subagent_type": subagent, "prompt": prompt, "effort": effort}
    if model:
        ti["model"] = model
    env = dict(os.environ, AW_QUOTA_STATE=str(f))
    r = subprocess.run([sys.executable, "-I", str(GATE)], input=json.dumps({"tool_name": "Agent", "tool_input": ti}),
                       capture_output=True, text=True, env=env)
    return r.returncode, r.stderr


BRIEF = "CONTEXT BUDGET: 40 calls\n"


def test_effort_red_medium_refused(tmp_path):
    rc, err = run_gate_effort(state(tmp_path, "red", 1.2), "haiku-exec", BRIEF, "medium")
    assert rc == 2 and "effort: low" in err


def test_effort_red_low_allowed(tmp_path):
    rc, _ = run_gate_effort(state(tmp_path, "red", 1.2), "haiku-exec", BRIEF, "low")
    assert rc == 0


def test_effort_amber_sonnet_high_refused(tmp_path):
    rc, err = run_gate_effort(state(tmp_path, "amber", 1.0), "sonnet-exec",
                              "WHY-NOT-HAIKU: judgment\nEFFORT-REASON: x\n" + BRIEF, "high")
    assert rc == 2 and "at most the default" in err


def test_effort_green_sonnet_high_needs_reason(tmp_path):
    f = state(tmp_path, "green", 0.8)
    rc, err = run_gate_effort(f, "sonnet-exec", "WHY-NOT-HAIKU: judgment\n" + BRIEF, "high")
    assert rc == 2 and "EFFORT-REASON" in err
    rc, _ = run_gate_effort(f, "sonnet-exec",
                            "WHY-NOT-HAIKU: judgment\nEFFORT-REASON: cross-file root cause\n" + BRIEF, "high")
    assert rc == 0


@pytest.mark.parametrize("tier", ["green", "amber", "red"])
def test_effort_xhigh_refused_any_tier(tmp_path, tier):
    rc, err = run_gate_effort(state(tmp_path, tier, 0.5), "haiku-exec", BRIEF, "xhigh")
    assert rc == 2 and "agent-model-gate" in err


def test_effort_pairs_only(tmp_path):
    # RULING 42 (Tim 2026-10-17): haiku low/medium/high, sonnet medium/high, opus medium/high
    g = state(tmp_path, "green", 0.7)
    sonnet = "WHY-NOT-HAIKU: judgment\n" + BRIEF
    rc, err = run_gate_effort(g, "sonnet-exec", sonnet, "low")
    assert rc == 2 and "sonnet runs at medium/high only" in err
    rc, err = run_gate_effort(g, "opus-deep", "OPUS-REASON: design\n" + BRIEF, "low")
    assert rc == 2 and "opus runs at medium/high only" in err
    assert run_gate_effort(g, "opus-deep", "OPUS-REASON: design\n" + BRIEF, "medium")[0] == 0
    assert run_gate_effort(g, "haiku-exec", BRIEF, "low")[0] == 0
    rc, err = run_gate_effort(g, "haiku-exec", BRIEF, "high")
    assert rc == 2 and "EFFORT-REASON" in err
    assert run_gate_effort(g, "haiku-exec", "EFFORT-REASON: tricky recipe\n" + BRIEF, "high")[0] == 0


def test_effort_red_sonnet_lowest_is_medium(tmp_path):
    r = state(tmp_path, "red", 1.3)
    sonnet = "WHY-NOT-HAIKU: x\nQUOTA-OVERRIDE: y\n" + BRIEF
    rc, err = run_gate_effort(r, "sonnet-exec", sonnet, "high")
    assert rc == 2 and "effort: medium" in err
    assert run_gate_effort(r, "sonnet-exec", sonnet, "medium")[0] == 0
