"""The live unit's wrapper puts the `LIVE-QUEUE` marker in the queue it drains (round 6 of #559).

`cyt index promote` and the Single transaction refuse a queue that carries the marker, whatever the
caller's environment holds (`tests/test_aw_index_live_queue.py`). The wrapper is the one place that
creates it: an idempotent `touch` before the `exec`. It must name the SAME queue the drain is given
and must never stop the drain.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

from tests.helpers.repo_files import repo_glob_or_skip

from cytir.index import promote_queue as Q

# the operator tree under `deploy/` is withheld from the public cut: the test skips there
WRAPPER = repo_glob_or_skip("deploy/systemd/*/promote_nightly.sh")[0]


def _lines():
    return WRAPPER.read_text().splitlines()


def _queue() -> str:
    return re.search(r"--queue-dir (\S+)", WRAPPER.read_text()).group(1)


def test_the_wrapper_touches_the_marker_in_the_queue_it_drains():
    text = WRAPPER.read_text()
    touch = [ln for ln in _lines() if ln.startswith("touch ")]
    assert len(touch) == 1 and touch[0].startswith(f"touch {_queue()}/{Q.LIVE_QUEUE_MARKER} ")
    # before the exec, and it cannot stop the drain (`set -e` is on: a failed touch only warns)
    assert text.index(touch[0]) < text.index("exec python3 -m cytir.index.promote_queue")
    assert "|| say " in touch[0]


def test_the_touch_line_creates_the_marker_idempotently_and_a_failure_does_not_stop_the_wrapper(
        tmp_path):
    line = next(ln for ln in _lines() if ln.startswith("touch "))
    queue = tmp_path / "queue"
    queue.mkdir()
    run = line.replace(_queue(), str(queue))
    script = f'set -euo pipefail\nsay() {{ echo "said: $*"; }}\n{run}\n{run}\necho survived\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "survived" and (queue / Q.LIVE_QUEUE_MARKER).is_file()
    (queue / Q.LIVE_QUEUE_MARKER).unlink()
    queue.rmdir()                                      # the queue is gone: the touch fails
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert "said: WARNING" in out and out.strip().endswith("survived")
