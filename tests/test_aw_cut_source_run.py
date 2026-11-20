"""The family door, end to end, as the cut rehearsal runs it (task 725, slice 7 step 1).

ONE declared family, `dila_kali`, from the committed sample to a stage that `cyt index build`
would accept: `cyt source run dila_kali --from SAMPLE --out STAGE` (the CLI, a fresh process, no
`AW_*` env) -> the stage is byte for byte the committed fixture stage (the stage door check) ->
stand-in vectors -> the build verb's own input checks (`load_stage_pack`, `check_vectors`, the
pinned embedder). This is the index dry check: nothing is written to a Qdrant, none is needed.

The job "cut rehearsal" runs this file by name before the full shipped suite, so a broken door
is named in its own step. Sealed input only: no network, no model.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

FAMILY = "dila_kali"
SAMPLE = REPO / "data" / "samples" / "dila-kali"
COMMITTED = REPO / "data" / "index-fixture" / "dila_kali"
TODAY = "2026-10-03"


def _tree(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_source_run_makes_a_stage_that_the_index_dry_check_accepts(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AW_", "CYTIR_"))}
    stage = tmp_path / "stage"
    res = subprocess.run(
        [sys.executable, "-m", "cytir.cli", "source", "run", FAMILY, "--from", str(SAMPLE),
         "--out", str(stage), "--today", TODAY],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout[-1500:] + res.stderr[-1500:]
    # the stage door check: the door's bytes are the committed fixture's bytes
    assert _tree(stage) == _tree(COMMITTED)

    # the index dry check: the build verb's own input checks, no Qdrant
    import standin_embedder as SE
    from cytir.index import build
    assert SE.write_vectors(stage) >= 1
    pack = build.load_stage_pack(str(stage))
    assert pack.family == FAMILY
    build.check_vectors(pack)
