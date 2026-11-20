"""task 1158: compose.sh refuses `up <svc> --force-recreate` without --no-deps.

The REAL compose.sh runs against a temp "box": ssh runs the command locally, scp copies locally,
and `docker`, `curl` and `sleep` are stubs on PATH that record their argv. Pattern as in
test_aw_release_serving_gate.py (ssh/curl stubbed, script real).
"""
from __future__ import annotations

import pathlib
import subprocess

import pytest

from tests.helpers.repo_files import repo_file_or_skip

APP = pathlib.Path(__file__).resolve().parents[1]
# deploy/scripts/* is operator tooling and is not in the public cut: a tree without the script skips.
SCRIPT = repo_file_or_skip("deploy/scripts/compose.sh")

STUBS = {
    "docker": 'echo "$*" >> "$STUB_LOG"\nexit 0\n',
    "sleep": "exit 0\n",
    "scp": 'cp "$2" "${3#*:}"\n',
    "ssh": (
        'cmd="$2"\n'
        'case "$cmd" in\n'
        '  *"docker compose"*) bash -c "$cmd" ;;\n'
        '  *"docker inspect"*) echo "true 0" ;;\n'
        '  *"-X POST"*) printf \'{"result":{}}\\n200\\n\' ;;\n'
        '  *curl*) printf 200 ;;\n'
        '  *) bash -c "$cmd" ;;\n'
        'esac\n'
    ),
}


def _run(tmp_path, *args):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    for name, body in STUBS.items():
        p = bindir / name
        p.write_text("#!/usr/bin/env bash\n" + body)
        p.chmod(0o755)
    box = tmp_path / "box"
    (box / "deploy").mkdir(parents=True, exist_ok=True)
    log = tmp_path / "docker.log"
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "STUB_LOG": str(log),
           "AW_DEPLOY_HOST": "x", "AW_REMOTE_ROOT": str(box)}
    r = subprocess.run(["bash", str(SCRIPT), "up", *args], env=env, capture_output=True, text=True,
                       timeout=120)
    return r, (log.read_text() if log.exists() else "")


def test_force_recreate_without_no_deps_is_refused(tmp_path):
    r, log = _run(tmp_path, "aw-mcp", "--", "--force-recreate")
    assert r.returncode == 2
    assert "without --no-deps" in r.stderr
    assert log == ""  # nothing reached docker


def test_force_recreate_with_no_deps_reaches_docker(tmp_path):
    r, log = _run(tmp_path, "aw-mcp", "--", "--force-recreate", "--no-deps")
    assert r.returncode == 0, r.stderr
    assert "--no-deps" in log and "--force-recreate" in log


def test_with_deps_is_stripped_before_docker(tmp_path):
    r, log = _run(tmp_path, "aw-mcp", "--", "--force-recreate", "--with-deps")
    assert r.returncode == 0, r.stderr
    assert "--force-recreate" in log
    assert "--with-deps" not in log
