"""Task 2254 (S10 step C): `cyt gate serve-parity run` base-vs-head on the `index-serve-gate` fixture.

The cheap per-PR tripwire: the fixture collection `aw_gate_fixture` (built by the job's own world,
tests/test_aw_index_serve_gate.py) is queried with the fixture's gold slice through the BASE tree and
the HEAD tree; the report must be a PASS (served bodies byte-equal, deep records equal, clean null
pair) and `serve-parity check` must accept it. The twin run (S10 D3) stays the
authoritative gate; this proves the plumbing on stand-in vectors, not the real embedder's ranking.

The query vectors come from `tests/standin_tei_server.py`, which shares the handler (and KIND
`hashed-bow-v1`) with the job's other tests, so the fixture index and the parity vectors agree.
`$SERVE_PARITY_HEAD_TREE` (default: this checkout) exists to prove the step is red on a deliberate
ranking change in a scratch copy. The base tree is `$SERVE_PARITY_BASE_TREE` (the CI job checks the PR base out there) and
`$SERVE_PARITY_BASE_SHA`; with neither set, base is this checkout (head vs itself: a null run).
The guard sees an explicit operator overlay naming the job's throwaway Qdrant as the twin.
"""
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys

import pytest

pytest.importorskip("yaml")
pytest.importorskip("qdrant_client")

from test_aw_index_serve_gate import COLLECTION, REPO, _free_port, world  # noqa: E402,F401  (the job's world)
from cytir.core.serve_layout import package_name  # noqa: E402


def serve_dir(tree: pathlib.Path) -> pathlib.Path:
    """The serving package of ONE tree: the PR base may still carry `cytir/` where the head carries `cytir/`."""
    return tree / package_name(tree) / "serve"


def _sha(tree: pathlib.Path) -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tree, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "0" * 40


def _cli(args, env, **kw):
    return subprocess.run([sys.executable, "-m", "cytir.cli", "gate", "serve-parity", *args], cwd=REPO,
                          env=env, capture_output=True, text=True, timeout=900, **kw)


def test_serve_parity_base_vs_head_passes_on_the_fixture_collection(world, tmp_path):
    url = world["url"]
    head = pathlib.Path(os.environ.get("SERVE_PARITY_HEAD_TREE") or REPO).resolve()
    base = pathlib.Path(os.environ.get("SERVE_PARITY_BASE_TREE") or head).resolve()
    base_sha = os.environ.get("SERVE_PARITY_BASE_SHA") or _sha(base)
    head_sha = _sha(head)
    overlay = tmp_path / "operator.json"
    overlay.write_text(json.dumps({"schema": "aw.operator-overlay.v1", "settings": {"twin_url": url}}))
    env = dict(world["stack"].env, AW_OPERATOR_OVERLAY=str(overlay))
    out = tmp_path / "parity"
    port = _free_port()
    tei = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).with_name("standin_tei_server.py")),
                            "--port", str(port)], stdout=subprocess.PIPE, text=True, cwd=REPO,
                           env=dict(os.environ, PYTHONPATH=str(REPO)))
    try:
        assert tei.stdout.readline().strip() == "ready"
        prep = _cli(["prepare", "--base-tree", str(base), "--tei-url", f"http://127.0.0.1:{port}",
                     "--out-dir", str(out), "--sets", "fixture"], env)
        assert prep.returncode == 0, prep.stdout[-1500:] + prep.stderr[-1500:]
    finally:
        tei.send_signal(signal.SIGTERM)
        tei.wait(timeout=20)
    run = _cli(["run", "--base-tree", str(base), "--head-tree", str(head), "--collection", COLLECTION,
                "--qdrant-url", url, "--vec-cache", str(out / "vectors.json"), "--out-dir", str(out),
                "--sets", "fixture", "--shards", "2", "--base-sha", base_sha, "--head-sha", head_sha], env)
    print(run.stdout[-3000:])
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-1500:]
    chk = _cli(["check", "--report", str(out / "parity.json"), "--base-sha", base_sha, "--head-sha", head_sha,
                "--base-server-dir", str(serve_dir(base)), "--head-server-dir", str(serve_dir(head))],
               env)
    # `check` is the release door: it refuses a report whose two trees hold the SAME server code ("proves
    # nothing about a serving-code release"). Most PRs move no serving code, so that reason is the
    # expected answer then. A value-change PR (task 2832) adds "under the BASE env", and only together
    # with the same-code reason: a base-env report is never run with cytir/serve moved (the run refuses
    # first), so a refusal when the server code differs, or ANY other reason, is red.
    reasons = [x.strip()[2:] for x in (chk.stdout + chk.stderr).splitlines() if x.strip().startswith("- ")]
    only_same_code = (bool(reasons) and any("compares a tree with itself" in x for x in reasons)
                      and all("compares a tree with itself" in x or "under the BASE env" in x
                              for x in reasons))
    assert chk.returncode == 0 or only_same_code, chk.stdout[-2000:] + chk.stderr[-1500:]
    assert json.loads((out / "parity.json").read_text())["verdict"] == "PASS"
