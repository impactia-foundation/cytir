#!/usr/bin/env python3
"""The one home for experiment-ledger ids, file shape, and pointer verification.

Task 569. The lessons ledger (`lessons_ledger`) records what we
LEARNED. Nothing recorded what we TRIED: the hypothesis, the prediction that
would have falsified it, the arms, and whether the prediction cashed. That
absence has a measurable cost in this project's own record --

  * Task 562 attributed 61% of the held-out sparse-only gap to chunk width.
    Task 565 re-cut the whole density class and the sparse leg went the OTHER
    way (215 -> 209 on the held-out set). The attribution was falsified by the
    next lane, and nothing in the repo connected the two.
  * ADR-090 amendment 1 DID write its success condition before the run ("if the
    held-out dense-only leg does not recover ... not a licence to widen the
    selector"). That is the exception, not the rule: most measurements in
    tasks 533-565 were designed and reported in the same document, after the
    numbers arrived.

Verification-doctrine 3 ("the experiment is the flag") and 9 ("good news is
guilty until confirmed") both depend on a prediction that existed BEFORE the
run. `prediction.stated_before_run` is therefore the ledger's load-bearing
field, and it is recorded honestly for the backfill -- a `false` is a finding
about the method, not a defect to tidy away.

FILE SHAPE mirrors `lessons_ledger` exactly and for the same reason (task 559):
a header line (schema, `why`, `provenance` -- never a stored `counts`, which
drifted four times in the lessons ledger) followed by one compact, key-sorted
experiment per line, sorted by id. Two branches appending different experiments
merge as a clean line-wise diff; two branches editing the SAME row conflict for
real.

IDS are `E<task>-<n>` -- the task number that ran the experiment plus a 1-based
counter within that task. Coordination-free for the same reason lesson ids are
(`lessons_ledger` docstring): task numbers are the key the orchestrator already
serializes by hand before dispatch. There is no legacy id set: this ledger
starts under the new scheme, so `is_valid_id` is a pure pattern check and a
bare `E1` is refused.

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
LEDGER_PATH = REPO / "data" / "experiments-ledger.v1.jsonl"

SCHEMA = "aw.experiments-ledger.v1"

#: `E<task>-<n>`. No legacy set: this ledger has no pre-scheme history.
ID_RE = re.compile(r"^E(\d+)-(\d+)$")

#: A verdict is one of three. "supported" and "falsified" are symmetric on
#: purpose: a falsified experiment is a RESULT, and the ledger exists partly to
#: stop those being quietly dropped (verification-doctrine 7).
#: "inconclusive" means the instrument could not decide -- underpowered,
#: confounded, or arms that were not comparable. It is NOT a soft "falsified".
VERDICTS = ("supported", "falsified", "inconclusive")

#: Keys every row must carry. Absent is a validation failure, not a default:
#: an experiment with no `prediction` block is an anecdote (verification-
#: doctrine 3) and the ledger must not be able to hold one silently.
REQUIRED = (
    "id", "title", "tasks", "date", "hypothesis", "prediction", "method",
    "result", "verdict", "prediction_cashed", "cashed_note", "what_changed",
    "artifacts", "scripts", "offrepo",
)

#: Optional keys. `moved` maps an ORIGINAL pointer (kept verbatim in
#: `artifacts`/`scripts`, because a pointer is a record of fact) to the repo
#: path(s) where that file's content lives now. The old pointer then resolves
#: iff every new path exists; it is never rewritten or dropped (Tim 2026-09-23).
OPTIONAL = ("moved",)

PREDICTION_KEYS = ("text", "falsifiable", "stated_before_run", "where")
METHOD_KEYS = ("arms", "instruments", "legs", "sample", "one_variable",
               "isolation")

#: An off-repo pointer NAMES THE HOST the evidence is sealed on, so a stranger
#: with a clean clone can tell "not in the repo" from "lost" (corpus-doctrine
#: 6). The host token is all this can check: what follows is a path, a work
#: dir, or a live store ("the GPU-box twin qdrant"), and all three are real
#: shapes in this project's record. What it refuses is a bare path with no
#: host -- the pointer that reads as evidence and resolves nowhere.
OFFREPO_RE = re.compile(r"^[a-z0-9_.-]+[: ]\S")


# --------------------------------------------------------------------- ids
def is_valid_id(experiment_id: str) -> bool:
    return bool(ID_RE.match(experiment_id or ""))


def id_task(experiment_id: str) -> str | None:
    m = ID_RE.match(experiment_id or "")
    return m.group(1) if m else None


def next_id(existing_ids: Iterable[str], task_no: int | str) -> str:
    """Next collision-free id for `task_no`, given the ids already on file."""
    task_no = str(int(task_no))
    existing = set(existing_ids)
    n = 1
    while f"E{task_no}-{n}" in existing:
        n += 1
    return f"E{task_no}-{n}"


def sort_key(experiment_id: str) -> tuple:
    m = ID_RE.match(experiment_id or "")
    if m:
        return (0, int(m.group(1)), int(m.group(2)), experiment_id)
    return (1, 0, 0, experiment_id)  # unknown scheme -- caught by validate()


# ------------------------------------------------------------- file I/O
def read_ledger(path: pathlib.Path = LEDGER_PATH) -> tuple[dict, list[dict]]:
    """Return (header, experiments). Line 1 is the header."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty file, no header line")
    header = json.loads(lines[0])
    rows = [json.loads(line) for line in lines[1:] if line.strip()]
    return header, rows


def write_ledger(header: dict, experiments: list[dict],
                 path: pathlib.Path = LEDGER_PATH) -> None:
    """Canonical formatter every writer must call.

    One compact, key-sorted object per line, sorted by id, header first.
    Deterministic byte output: a no-op re-write reproduces the same bytes.
    """
    rows = sorted(experiments, key=lambda e: sort_key(e["id"]))
    out = [json.dumps(header, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))]
    out += [json.dumps(row, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")) for row in rows]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# --------------------------------------------------------------- counts
def compute_counts(experiments: list[dict]) -> dict:
    """Derived stats, computed fresh every time -- never stored."""
    import collections
    by_verdict = collections.Counter(e["verdict"] for e in experiments)
    by_task = collections.Counter(t for e in experiments for t in e["tasks"])
    pre = [e for e in experiments if e["prediction"]["stated_before_run"]]
    cashed = [e for e in experiments if e.get("prediction_cashed") is True]
    not_cashed = [e for e in experiments if e.get("prediction_cashed") is False]
    return {
        "experiments": len(experiments),
        "by_verdict": {v: by_verdict.get(v, 0) for v in VERDICTS},
        "by_task": dict(sorted(by_task.items(), key=lambda kv: int(kv[0]))),
        "prediction_stated_before_run": len(pre),
        "prediction_cashed": len(cashed),
        "prediction_did_not_cash": len(not_cashed),
        "no_pre_run_prediction": len(experiments) - len(pre),
    }


# --------------------------------------------------- pointer verification
#: A `moved` target may also pin a DELETED file by its git blob:
#: `git-blob:<sha40>:<original path>` (the form lessons-ledger L096/L161 use in
#: prose, task 1709). It resolves iff the blob is in the object store. A
#: shallow clone cannot hold old blobs, so there only the form is checked; a
#: full clone proves the blob exists.
GIT_BLOB_RE = re.compile(r"^git-blob:([0-9a-f]{40}):(\S+)$")


def _git(repo: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


def _blob_problem(pointer: str, target: str, repo: pathlib.Path) -> str | None:
    m = GIT_BLOB_RE.match(target)
    if not m:
        return f"moved target of {pointer} is not a git-blob pin: {target!r}"
    sha = m.group(1)
    shallow = _git(repo, "rev-parse", "--is-shallow-repository")
    if shallow.returncode != 0 or shallow.stdout.strip() == "true":
        return None  # old blobs are not fetched here; the form is all we can check
    if _git(repo, "cat-file", "-e", f"{sha}^{{blob}}").returncode != 0:
        return f"pointer {pointer} pinned to blob {sha}, which is not in the object store"
    return None


def resolve_pointers(paths: Iterable[str],
                     repo: pathlib.Path = REPO,
                     moved: dict | None = None) -> list[str]:
    """Every repo-relative artifact/script pointer must exist on disk.

    An experiment whose evidence path does not resolve is a summary, not
    evidence (verification-doctrine 5).
    """
    problems: list[str] = []
    for p in paths or ():
        if not isinstance(p, str) or not p:
            problems.append(f"empty pointer {p!r}")
            continue
        if p.startswith("/") or ".." in pathlib.PurePosixPath(p).parts:
            problems.append(
                f"pointer must be repo-relative and contain no '..': {p}")
            continue
        if (repo / p).exists():
            continue
        targets = (moved or {}).get(p)
        if targets is None:
            problems.append(f"pointer does not exist: {p}")
            continue
        targets = [targets] if isinstance(targets, str) else list(targets)
        if not targets:
            problems.append(f"pointer moved to nothing: {p}")
        for t in targets:
            if isinstance(t, str) and t.startswith("git-blob:"):
                bp = _blob_problem(p, t, repo)
                if bp:
                    problems.append(bp)
            elif (not isinstance(t, str) or not t or t.startswith("/")
                    or ".." in pathlib.PurePosixPath(t).parts):
                problems.append(f"moved target of {p} must be repo-relative: {t!r}")
            elif not (repo / t).exists():
                problems.append(f"pointer {p} moved to {t}, which does not exist")
    return problems


# --------------------------------------------------------------- validate
def validate(header: dict, experiments: list[dict],
             repo: pathlib.Path = REPO) -> list[str]:
    """Full guard: schema, duplicate ids, id scheme, required keys, verdict
    vocabulary, prediction/cashed consistency, pointer resolution."""
    problems: list[str] = []
    if header.get("schema") != SCHEMA:
        problems.append(f"header schema {header.get('schema')!r} != {SCHEMA!r}")
    if "counts" in header:
        problems.append("header carries a stored `counts` -- derive it instead "
                        "(compute_counts), never store it")

    seen: dict[str, int] = {}
    for e in experiments:
        i = e.get("id", "<missing>")
        seen[i] = seen.get(i, 0) + 1
    dupes = sorted(i for i, n in seen.items() if n > 1)
    if dupes:
        problems.append(f"duplicate id(s): {dupes}")

    for e in experiments:
        eid = e.get("id", "<missing>")
        missing = [k for k in REQUIRED if k not in e]
        if missing:
            problems.append(f"{eid}: missing required key(s) {missing}")
            continue
        extra = sorted(set(e) - set(REQUIRED) - set(OPTIONAL))
        if extra:
            problems.append(f"{eid}: unknown key(s) {extra} -- extend REQUIRED "
                            "deliberately rather than letting rows drift apart")
        if not is_valid_id(eid):
            problems.append(f"{eid}: does not match E<task>-<n> -- mint it with "
                            "experiments_ledger.next_id(existing_ids, task_no)")
        else:
            t = id_task(eid)
            if t not in (e.get("tasks") or []):
                problems.append(f"{eid}: id names task {t} but `tasks` is "
                                f"{e.get('tasks')}")
        if e["verdict"] not in VERDICTS:
            problems.append(f"{eid}: verdict {e['verdict']!r} not in {VERDICTS}")

        pred = e.get("prediction")
        if not isinstance(pred, dict):
            problems.append(f"{eid}: `prediction` is not an object")
        else:
            pmissing = [k for k in PREDICTION_KEYS if k not in pred]
            if pmissing:
                problems.append(f"{eid}: prediction missing {pmissing}")
            else:
                if not isinstance(pred["stated_before_run"], bool):
                    problems.append(f"{eid}: prediction.stated_before_run must "
                                    "be a bool")
                if not isinstance(pred["falsifiable"], bool):
                    problems.append(f"{eid}: prediction.falsifiable must be a "
                                    "bool")
                if not str(pred.get("where") or "").strip():
                    problems.append(
                        f"{eid}: prediction.where is empty -- state WHERE the "
                        "pre-run prediction lives, or say plainly that the "
                        "record carries none")
                # The consistency rule the ledger exists for: a prediction that
                # was never stated before the run cannot have cashed, and one
                # that was must say whether it did.
                cashed = e.get("prediction_cashed")
                if cashed not in (True, False, None):
                    problems.append(f"{eid}: prediction_cashed must be true, "
                                    "false or null")
                elif not pred["stated_before_run"] and cashed is not None:
                    problems.append(
                        f"{eid}: prediction_cashed={cashed!r} but "
                        "stated_before_run is false -- a prediction written "
                        "after the numbers arrived cannot cash")
                elif pred["stated_before_run"] and cashed is None:
                    problems.append(
                        f"{eid}: stated_before_run is true but "
                        "prediction_cashed is null -- say whether it cashed")

        meth = e.get("method")
        if not isinstance(meth, dict):
            problems.append(f"{eid}: `method` is not an object")
        else:
            mmissing = [k for k in METHOD_KEYS if k not in meth]
            if mmissing:
                problems.append(f"{eid}: method missing {mmissing}")
            for key in ("arms", "instruments", "legs"):
                if key in meth and not isinstance(meth[key], list):
                    problems.append(f"{eid}: method.{key} must be a list")
            if isinstance(meth.get("arms"), list) and len(meth["arms"]) < 1:
                problems.append(f"{eid}: method.arms is empty")

        res = e.get("result")
        if not isinstance(res, dict) or not str(res.get("numbers") or "").strip():
            problems.append(f"{eid}: result.numbers is empty -- a verdict with "
                            "no numbers is an opinion")

        if not str(e.get("cashed_note") or "").strip():
            problems.append(f"{eid}: cashed_note is empty")
        for key in ("tasks", "what_changed", "artifacts", "scripts", "offrepo"):
            if not isinstance(e.get(key), list):
                problems.append(f"{eid}: `{key}` must be a list")
        if not (e.get("what_changed") or []):
            problems.append(f"{eid}: what_changed is empty -- write "
                            "[\"nothing\"] if the experiment moved nothing")

        moved = e.get("moved") or {}
        listed = set(e.get("artifacts") or []) | set(e.get("scripts") or [])
        for old in sorted(set(moved) - listed):
            problems.append(f"{eid}: moved names {old!r}, which no "
                            "artifact/script pointer carries")
        for p in resolve_pointers(list(e.get("artifacts") or [])
                                  + list(e.get("scripts") or []), repo, moved):
            problems.append(f"{eid}: {p}")
        for p in (e.get("offrepo") or []):
            if not OFFREPO_RE.match(str(p)):
                problems.append(
                    f"{eid}: offrepo pointer {p!r} does not name a host "
                    "(expected `<host>:/abs/path`)")
    return problems


# ------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None, prog: str = "cyt ops experiments-ledger") -> int:
    """`cyt ops experiments-ledger [--ledger FILE]`: validate the ledger, print its counts.

    Exit 0 when every id, shape rule and pointer resolves, 1 otherwise.
    """
    import argparse
    ap = argparse.ArgumentParser(
        prog=prog, description="Validate data/experiments-ledger.v1.jsonl (ids, shape, "
        "prediction/verdict consistency, evidence pointers) and print counts derived from its rows.")
    ap.add_argument("--ledger", type=pathlib.Path, default=LEDGER_PATH,
                    help="ledger file (default: data/experiments-ledger.v1.jsonl)")
    args = ap.parse_args(argv)
    if args.ledger == LEDGER_PATH and not LEDGER_PATH.is_file():   # a stranger's cut ships no operator ledger
        print(f"ledger not shipped in this tree: {LEDGER_PATH}")
        return 0
    header, experiments = read_ledger(args.ledger)
    problems = validate(header, experiments, REPO)
    print(json.dumps(compute_counts(experiments), sort_keys=True, indent=2))
    for prob in problems:
        print(f"[PROBLEM] {prob}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
