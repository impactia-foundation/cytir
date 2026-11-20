"""The one home for lesson-ledger ids, file shape, and pointer verification.

Task 559. The ledger's ids were minted `L%d % (len(lessons) + 1)`
(the task-546 and task-553 `add_lesson.py` scripts): a
counter over the CURRENT length, computed independently by whoever's branch
appends next. Two branches cut from the same base (task 552, task 553, both
2026-09-08) both computed `len==182` and both minted `L183`. The merge kept
one and silently dropped the other; the orchestrator noticed only because the
count looked wrong, and hand-renumbered the survivor to `L185` — an id nobody
had derived from anything, chosen only because `L184` was already spoken for
by a THIRD branch (task 555) still sitting unmerged in PR #123. Separately,
`data/lessons-ledger.v1.json` was one JSON document (schema + `why` +
`provenance` + derived `counts` + all 184 rows nested inside one array), so
two branches appending different rows produced two top-level objects
concatenated in one file, not a mergeable diff — the identical shape of
defect task 555 (roster merge safety) fixed for the test-suite roster.

Fix, mirroring that precedent:

  * **Ids.** Going forward, a new lesson's id is `L<task>-<n>` — the task
    number that produced it, plus a 1-based counter WITHIN that task. Two
    branches for two different tasks cannot collide because task numbers are
    the thing the orchestrator already serializes by hand before dispatch;
    two lessons from the SAME task collide only if the same task doc appends
    twice, which is a single-writer situation with no merge involved. This
    is not a content hash: a hash would also be collision-free, but it is
    opaque (an id tells you nothing) and it still has to be minted from
    something — the task number already IS the coordination-free key this
    project uses for ADRs, task docs and PRs, so lesson ids reuse it rather
    than forking a second numbering scheme. The `L001`..`L185`
    ids that already shipped are history and do not change (`LEGACY_IDS`,
    frozen below, verbatim) — `is_valid_id` accepts them by exact membership,
    never by pattern, so a FUTURE bare `L186`-style id is refused rather than
    grandfathered in by a numeric range check.
  * **File shape.** `data/lessons-ledger.v2.jsonl`: a header line (schema,
    `why`, `provenance` — never `counts`, which drifted from its own rows
    four times across `add_lesson.py` scripts that hand-incremented a
    counter after every append) followed by one compact, key-sorted lesson
    object per line, sorted by id. `counts` is recomputed from the rows by
    `compute_counts()` wherever it is needed (currently
    the task-529 ledger-table renderer) — nothing stores it, so
    nothing can drift from it.

  * **Pointers.** A `test` field cites `path.py::func`; both must exist. A
    test in a DELETED file is pinned as `git-blob:<sha40>:<original path>::<func>`
    (task 1914): the blob must be in the object store and define `<func>`.

Every script that reads or writes the ledger imports THIS module rather than
re-deriving id minting, sort order, or the line format locally
(engineering-doctrine 1/6: one home, mirror contract).
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
from typing import Iterable

REPO = pathlib.Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO / "data" / "lessons-ledger.v2.jsonl"

SCHEMA = "aw.lessons-ledger.v2"

#: The ids that shipped under the old `L%03d` scheme before this task. Frozen
#: VERBATIM (not "any L### <= 185") so a new sequential-looking id is never
#: silently accepted just because it falls in the old numeric range.
LEGACY_IDS: frozenset[str] = frozenset(
    [f"L{i:03d}" for i in range(1, 186)]
)

#: The new, coordination-free scheme: `L<task>-<n>`.
NEW_ID_RE = re.compile(r"^L(\d+)-(\d+)$")

#: `.py` and `.sh` test/guard pointers this project's `test` field cites.
_POINTER = re.compile(
    r"(?P<file>[\w./-]+\.(?:py|sh))(?:::(?P<func>[\w*]+))?|::(?P<func2>[\w*]+)"
)

#: A `test` pointer may pin a test in a DELETED file by its git blob:
#: `git-blob:<sha40>:<original path>::<func>` (mirrors `GIT_BLOB_RE` in
#: the experiments ledger's canonical module; the pin records a test that existed, not
#: a claim that it still runs). It resolves iff the blob is in the object store
#: and its text has `def <func>`. A shallow clone holds no old blobs, so there
#: only the form is checked.
GIT_BLOB_PIN = re.compile(
    r"git-blob:(?P<sha>[0-9a-f]{40}):(?P<path>[\w./-]+\.py)::(?P<func>\w+)")

ENCODINGS = ("contract", "guard", "runbook-step", "doc-only", "memory-only")


# --------------------------------------------------------------------- ids
def is_valid_id(lesson_id: str) -> bool:
    """True iff `lesson_id` is a shipped legacy id or a well-formed new one."""
    if lesson_id in LEGACY_IDS:
        return True
    return bool(NEW_ID_RE.match(lesson_id))


def id_task(lesson_id: str) -> str | None:
    """Task number encoded in a new-scheme id, else None (legacy or invalid)."""
    m = NEW_ID_RE.match(lesson_id)
    return m.group(1) if m else None


def next_id(existing_ids: Iterable[str], task_no: int | str) -> str:
    """Next collision-free id for `task_no`, given the ids already on file.

    Only collides with a concurrent branch for the SAME task number appending
    twice — a single-writer situation, not a merge race.
    """
    task_no = str(int(task_no))
    existing = set(existing_ids)
    n = 1
    while f"L{task_no}-{n}" in existing:
        n += 1
    return f"L{task_no}-{n}"


def sort_key(lesson_id: str) -> tuple:
    """Total order: legacy ids first (numeric), then new ids by (task, n)."""
    if lesson_id in LEGACY_IDS:
        return (0, int(re.sub(r"\D", "", lesson_id) or 0), 0, lesson_id)
    m = NEW_ID_RE.match(lesson_id)
    if m:
        return (1, int(m.group(1)), int(m.group(2)), lesson_id)
    return (2, 0, 0, lesson_id)  # unknown scheme — sorts last, caught by validate()


# ------------------------------------------------------------- file I/O
def read_ledger(path: pathlib.Path = LEDGER_PATH) -> tuple[dict, list[dict]]:
    """Return (header, lessons) from the JSONL ledger. Line 1 is the header."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty file, no header line")
    header = json.loads(lines[0])
    lessons = [json.loads(line) for line in lines[1:] if line.strip()]
    return header, lessons


def write_ledger(header: dict, lessons: list[dict],
                  path: pathlib.Path = LEDGER_PATH) -> None:
    """Canonical formatter every writer must call.

    One compact, key-sorted object per line, sorted by id, header first.
    Deterministic byte output: a no-op re-write reproduces the same bytes,
    so two branches editing DIFFERENT rows merge as a clean line-wise diff
    and two branches editing the SAME row conflict for real instead of
    silently unioning or truncating.
    """
    rows = sorted(lessons, key=lambda l: sort_key(l["id"]))
    out_lines = [json.dumps(header, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"))]
    out_lines += [json.dumps(row, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")) for row in rows]
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------- counts
def compute_counts(lessons: list[dict]) -> dict:
    """Derived stats, computed fresh every time — never stored (see module
    docstring: a stored `counts` is what drifted four times)."""
    import collections
    by_encoding = collections.Counter(l["encoding"] for l in lessons)
    by_stage = collections.Counter(l["stage"] for l in lessons)
    by_group = collections.Counter(l["group"] for l in lessons)
    unencoded = [l for l in lessons if l["encoding"] in ("doc-only", "memory-only")]
    not_class_wide = [l for l in lessons
                      if l["encoding"] in ("contract", "guard")
                      and l.get("class_wide") is False]
    no_test = [l for l in lessons
              if l["encoding"] == "contract" and not l.get("test")]
    return {
        "lessons": len(lessons),
        "by_encoding": {k: by_encoding.get(k, 0) for k in ENCODINGS},
        "by_stage": dict(sorted(by_stage.items())),
        "by_group": dict(sorted(by_group.items())),
        "unencoded": len(unencoded),
        "contract_but_not_class_wide": len(not_class_wide),
        "contract_without_a_test": len(no_test),
    }


# --------------------------------------------------- pointer verification
def _git(repo: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


def _pin_problems(pin: re.Match, repo: pathlib.Path) -> list[str]:
    sha, path, func = pin.group("sha"), pin.group("path"), pin.group("func")
    shallow = _git(repo, "rev-parse", "--is-shallow-repository")
    if shallow.returncode != 0 or shallow.stdout.strip() == "true":
        return []  # old blobs are not fetched here; the form is all we can check
    blob = _git(repo, "cat-file", "-p", f"{sha}^{{blob}}")
    if blob.returncode != 0:
        return [f"{path}::{func} pinned to blob {sha}, which is not in the object store"]
    if not re.search(r"^\s*(?:async\s+)?def\s+" + re.escape(func) + r"\s*\(",
                     blob.stdout, re.M):
        return [f"{path}::{func} not found in pinned blob {sha}"]
    return []


def resolve_pointers(test_field: str, repo: pathlib.Path = REPO) -> list[str]:
    """Verify every file/function the `test` field cites actually exists.

    Returns a list of problem strings (empty = every pointer resolves).
    Handles the field's several separators (`+`, `,`, `;`) uniformly by
    scanning in order and tracking the last-seen file, so `a.py::f, ::g`
    resolves `g` against `a.py` without re-stating the filename. A function
    name ending in `*` is a deliberate glob over several tests
    (`test_no_fallback_*`) and resolves if ANY `def` in the file matches the
    prefix. A `git-blob:<sha40>:<path>::<func>` pin stands for a test in a
    deleted file (see `GIT_BLOB_PIN`).
    """
    problems: list[str] = []
    for pin in GIT_BLOB_PIN.finditer(test_field or ""):
        problems += _pin_problems(pin, repo)
    test_field = GIT_BLOB_PIN.sub("", test_field or "")
    cur_file: str | None = None
    for m in _POINTER.finditer(test_field):
        if m.group("file"):
            cur_file = m.group("file")
            path = repo / cur_file
            if not path.exists():
                problems.append(f"file does not exist: {cur_file}")
            func = m.group("func")
        else:
            func = m.group("func2")
        if not func:
            continue
        if cur_file is None:
            problems.append(f"::{func} with no preceding file")
            continue
        path = repo / cur_file
        if not path.exists():
            continue  # already reported the missing file above
        if not cur_file.endswith(".py"):
            continue  # shell script — no `def` to check
        src = path.read_text(encoding="utf-8", errors="ignore")
        if func.endswith("*"):
            prefix = func[:-1]
            pat = re.compile(r"^\s*(?:async\s+)?def\s+" + re.escape(prefix),
                             re.M)
            if not pat.search(src):
                problems.append(f"{cur_file}::{func} matches no def")
        else:
            pat = re.compile(r"^\s*(?:async\s+)?def\s+" + re.escape(func)
                             + r"\s*\(", re.M)
            if not pat.search(src):
                problems.append(f"{cur_file}::{func} not found")
    return problems


# --------------------------------------------------------------- validate
def validate(header: dict, lessons: list[dict],
             repo: pathlib.Path = REPO) -> list[str]:
    """Full guard: schema, duplicate ids, id scheme, pointer resolution."""
    problems: list[str] = []
    if header.get("schema") != SCHEMA:
        problems.append(f"header schema {header.get('schema')!r} != {SCHEMA!r}")
    if "counts" in header:
        problems.append("header carries a stored `counts` — derive it instead "
                        "(compute_counts), never store it")

    ids = [l["id"] for l in lessons]
    seen: dict[str, int] = {}
    for i in ids:
        seen[i] = seen.get(i, 0) + 1
    dupes = sorted(i for i, n in seen.items() if n > 1)
    if dupes:
        problems.append(f"duplicate id(s): {dupes}")

    for l in lessons:
        lid = l.get("id", "<missing>")
        if not is_valid_id(lid):
            problems.append(
                f"{lid}: not a legacy id and does not match L<task>-<n> — "
                "mint it with lessons_ledger.next_id(existing_ids, task_no)")
        test_field = l.get("test")
        if test_field:
            for p in resolve_pointers(test_field, repo):
                problems.append(f"{lid}: {p}")
    return problems


# ------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None, prog: str = "cyt ops lessons") -> int:
    """`cyt ops lessons [--ledger FILE]`: validate the ledger, print its counts.

    Exit 0 when every id and every `test` pointer resolves, 1 otherwise.
    """
    import argparse
    ap = argparse.ArgumentParser(
        prog=prog, description="Validate data/lessons-ledger.v2.jsonl (ids, "
        "shape, test pointers) and print counts derived from its rows.")
    ap.add_argument("--ledger", type=pathlib.Path, default=LEDGER_PATH,
                    help="ledger file (default: data/lessons-ledger.v2.jsonl)")
    args = ap.parse_args(argv)
    if args.ledger == LEDGER_PATH and not LEDGER_PATH.is_file():   # a stranger's cut ships no operator ledger
        print(f"ledger not shipped in this tree: {LEDGER_PATH}")
        return 0
    header, lessons = read_ledger(args.ledger)
    problems = validate(header, lessons, REPO)
    print(json.dumps(compute_counts(lessons), sort_keys=True, indent=2))
    for prob in problems:
        print(f"[PROBLEM] {prob}")
    return 1 if problems else 0
