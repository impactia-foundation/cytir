#!/usr/bin/env python3
"""Run any command -- a gate script, or (the build box's live use) `uvicorn
cytir.serve.app:api` itself -- under PROD'S OWN serving environment.

MOVED HERE from task 516's `prod_env_run.py` (task 646, weekly goal
2026-09-25: "nothing in production runs from tasks/"). This is the
ExecStart of two build-box units (aw-test-e8, aw-test-twin; retired 2026-11-04, task 2328) --
confirmed by a live inventory
(systemctl + ps) the day this moved, not by reading the repo alone.

cytir.ops.prod_env reads deploy/docker-compose.yaml's aw-api environment block
and rewrites its CONTAINER paths to this host's own copy, so a caller
cannot measure/serve a configuration nobody runs. Every gate script in the
battery needs the same thing, and `gate_authority.py` needs it most:
started with a bare environment it prints

    AW_AUTHORITY=0 ... auth_table=0 hosts hub_table=0 HUBFLOOR=0

and then measures a ranker production does not have -- authority off,
hubfloor off, no authority table loaded. Task 393 §559 records the same
trap from the other side.

    python3 cytir/ops/prod_env_run.py --collection aw_e6 \\
        --qdrant-url http://localhost:16457 --mrl-dim 256 [--extra K=V ...] \\
        -- python -m cytir.gates.authority ...

Four values are overridden and they are the only four (plus explicit
--extra): the collection, its qdrant, the embedder, and the width the
collection was BUILT at. Prints the effective config, because a caller that
does not echo its own configuration cannot be compared to another one.

SHIM (task 696 slice 8 step 4): the env comes from `cytir.ops.prod_env.resolve_env`, the one
function `cyt serve` also uses. Without `--overlays` it is the base compose block alone, so the
two live units run exactly the env they ran before. `--overlays` adds the compose overlays
(the flip values), after which the hand-written `--extra AW_DENSE_RESCORE=1 ...` flags are
redundant; repointing the units is a separate operator change.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

# Import `aw` (this file's own sibling package, ../../cytir/) by absolute path
# FIRST, so a stale/foreign `aw` editable install elsewhere on sys.path
# (e.g. another git worktree's `pip install -e .` on a shared dev venv) can
# never shadow this checkout's own package -- see cytir/core/common.py's
# comment for the full story. This matters here specifically: systemd
# invokes this file as a direct script run (`python3 cytir/ops/
# prod_env_run.py`, not `-m`), whose sys.path[0] is this file's OWN
# directory (cytir/ops/), never the repo root.
REPO = pathlib.Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = str(REPO)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cytir.core import envnames as _envnames  # noqa: E402
from cytir.ops.prod_env import prod_env, resolve_env, to_repo_paths  # noqa: E402,F401  # re-exported: read as shim.prod_env / shim.to_repo_paths by tests/operator/test_prod_env_run_shim.py:43


def main(argv=None, prog=None) -> int:
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--qdrant-url", required=True)
    ap.add_argument("--tei-url", default="http://localhost:8088")
    ap.add_argument("--mrl-dim", required=True)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--overlays", action="store_true",
                    help="also apply deploy/compose-overlays.manifest (what `cyt serve` does); "
                         "off by default so the live units keep the env they always had")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    cmd = [c for c in args.cmd if c != "--"]
    if not cmd:
        ap.error("no command given after --")

    env = dict(os.environ)
    env.update(resolve_env(overlays=args.overlays).env)
    env.update({"AW_COLLECTION": args.collection,
                "AW_QDRANT_URL": args.qdrant_url,
                "AW_TEI_URL": args.tei_url,
                "AW_MRL_DIM": str(args.mrl_dim),
                "PYTHONPATH": str(REPO)})
    for item in args.extra:
        k, _, v = item.partition("=")
        env[k] = v
    shown = {k: v for k, v in sorted(env.items()) if _envnames.is_knob(k) and k != _envnames.MARKER}
    print(f"=== prod env, pointed at {args.collection} ({len(shown)} AW_* flags) ===")
    for k in ("AW_COLLECTION", "AW_QDRANT_URL", "AW_MRL_DIM", "AW_AUTHORITY",
              "AW_HUBFLOOR", "AW_TWO_STAGE", "AW_RANKHEAD", "AW_TRUSTFLOOR"):
        print(f"  {k}={shown.get(k, '(unset)')}")
    print("  " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=env, cwd=str(REPO))


if __name__ == "__main__":
    sys.exit(main())
