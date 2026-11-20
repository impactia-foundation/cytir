"""cytir.index.e10_ship -- what the FLIP reads of an `aw_e10` build (task 863, review 870 S4).

`deploy/scripts/flip-relocate.sh` step 2b runs `python -m cytir.index.e10_ship --manifest ... --target ...
--expect-count ...` on the operator host: the builder's manifest must say `shippable`, with a PASSING
data-checks verdict (the deleted `e10_checks verdict` (git history, task 1312)) bound to the manifest's plan, its LAST catch-up round
and the flip's expected count. Stdlib only (the operator host runs it with no build dependencies); the two
schema names live here and the deleted `rebuild_e10` builder imported them (git history, task 1312).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys

SCHEMA_MANIFEST = "aw.rebuild-e10-manifest.v1"
#: The data checks' verdict (the deleted `e10_checks verdict` (git history, task 1312)), read back by `finalize` and by the flip.
SCHEMA_DATA_CHECKS = "aw.rebuild-e10-data-checks.v1"


class ShipRefused(RuntimeError):
    pass


#: The verdict's three checks: the CLI name -> the key in the verdict's `checks`.
ACCEPTABLE = {"dense-guard": "dense_guard", "census": "posting_census", "german-parts": "german_parts"}
DEFAULT_PLAN_FILE = pathlib.Path(__file__).resolve().parents[2] / "plan.md"


def ruling_in_plan(plan_file, ruling: str) -> bool:
    """True when `plan_file` writes `RULING <ruling>` (case-insensitive, whole token). Fail-closed: a
    missing file or an empty id is False."""
    p = pathlib.Path(plan_file)
    if not ruling or not re.fullmatch(r"[A-Za-z0-9._-]+", ruling) or not p.is_file():
        return False
    return re.search(rf"\bRULING\s+{re.escape(ruling)}(?![A-Za-z0-9._-])", p.read_text(encoding="utf-8"),
                     re.I) is not None


def check_report_sha256(checks_dir, check: str) -> str:
    """The sha256 of the report a check left in `checks-<target>/`: `german-parts.json`, or, for the
    slice-summed checks, a digest over every `dense-*` / `census-*` slice file (name + sha256, sorted).
    '' when the report does not exist."""
    d = pathlib.Path(checks_dir)
    if check == "german-parts":
        f = d / "german-parts.json"
        return _sha256(f) if f.is_file() else ""
    prefix = {"dense-guard": "dense", "census": "census"}[check]
    files = sorted(d.glob(f"{prefix}-*-of-*.json"))
    if not files:
        return ""
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.name}:{_sha256(f)}\n".encode())
    return h.hexdigest()


def check_accepted_entries(doc: dict, *, plan_dir: pathlib.Path, target: str, plan_file) -> list[str]:
    """Problems with a verdict's `accepted` entries (empty = sound): each ruling is written in
    `plan_file`, each report hashes to the entry's sha on disk, each accepted check really failed, and
    EVERY failing check is accepted (a verdict `pass` never hides an unaccepted failure)."""
    problems: list[str] = []
    acc = doc.get("accepted") or []
    names = {v: k for k, v in ACCEPTABLE.items()}
    cdir = plan_dir / f"checks-{target}"
    accepted_keys = set()
    for e in acc:
        chk, ruling = e.get("check"), e.get("ruling")
        if chk not in ACCEPTABLE:
            problems.append(f"accepted check {chk!r} is not one of {sorted(ACCEPTABLE)}")
            continue
        accepted_keys.add(ACCEPTABLE[chk])
        if (doc.get("checks") or {}).get(ACCEPTABLE[chk], {}).get("pass"):
            problems.append(f"{chk} is accepted but it passed")
        if not ruling_in_plan(plan_file, ruling):
            problems.append(f"the ruling {ruling!r} accepting {chk} is not written as `RULING {ruling}` in {plan_file}")
        have = check_report_sha256(cdir, chk)
        if not have or have != e.get("report_sha256"):
            problems.append(f"the {chk} report on disk does not hash to the accepted report_sha256")
    for k, v in (doc.get("checks") or {}).items():
        if not v.get("pass") and k not in accepted_keys:
            problems.append(f"{names.get(k, k)} failed and is not accepted")
    return problems


def _read(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def round_dirs(plan_dir: pathlib.Path, target: str) -> list[str]:
    """Every `catchup-<target>-rNN/` directory in the plan directory, applied or not, sorted."""
    return sorted(d.name for d in plan_dir.glob(f"catchup-{target}-r*") if d.is_dir())


def assert_shippable(manifest: str | os.PathLike, *, target: str, expect_count: int,
                     plan_file: str | os.PathLike | None = None) -> dict:
    """The flip's refusal (task 863, review 870 S4): the build manifest of `target` must say
    `shippable`, with a PASSING data-checks verdict bound to the state that ships: the manifest's own
    plan.json, its LAST catch-up round, its exact count == `expect_count` (F2's count). A verdict of an
    earlier round (the last catch-up round, Δ2, drops it) or another count refuses: the checks re-run
    inside the freeze. Reads files only; no Qdrant call.

    THE MANIFEST IS READ INSIDE ITS PLAN DIRECTORY (review 870 verification, M4): `manifest` must be
    `<plan-dir>/manifest-<target>.json`, beside the `plan.json` it names (sha256) and EVERY
    `catchup-<target>-rNN/` directory the build has. The rounds on disk must be exactly the manifest's
    rounds: a round applied (or started) after `finalize` wrote the manifest -- Δ2 without a re-run of
    `finalize` and the checks -- refuses, and so does a copy of the manifest without its round
    directories. Copy the whole plan directory to the operator host, never the manifest alone."""
    mp = pathlib.Path(manifest)
    problems: list[str] = []
    if not mp.is_file():
        raise ShipRefused(f"REFUSED: the build manifest {mp} does not exist")
    man = _read(mp)
    pd = mp.parent
    if mp.name != f"manifest-{target}.json":
        problems.append(f"{mp.name} is not manifest-{target}.json: the flip reads the builder's manifest "
                        "inside its plan directory")
    pj = pd / "plan.json"
    if not pj.is_file():
        problems.append(f"{pd} holds no plan.json: the flip reads the manifest inside its plan directory "
                        "(copy the whole plan directory, never the manifest alone)")
    elif _sha256(pj) != man.get("plan_sha256"):
        problems.append(f"{pj} does not hash to the manifest's plan_sha256: another plan's directory")
    rounds_listed = [r.get("round") for r in man.get("catchup_rounds") or []]
    want_dirs = [f"catchup-{target}-r{k:02d}" for k in range(1, len(rounds_listed) + 1)]
    have_dirs = round_dirs(pd, target)
    if rounds_listed != list(range(1, len(rounds_listed) + 1)):
        problems.append(f"the manifest's catch-up rounds are not 1..N: {rounds_listed}")
    if have_dirs != want_dirs:
        extra = sorted(set(have_dirs) - set(want_dirs))
        missing = sorted(set(want_dirs) - set(have_dirs))
        problems.append(
            f"the plan directory's catch-up rounds are not the manifest's (on disk beyond it: {extra}; "
            f"listed but absent: {missing}): a round ran after `finalize` (re-run `finalize`, the three "
            f"checks and `verdict`), or the round directories were not copied with the manifest")
    if man.get("schema") != SCHEMA_MANIFEST:
        problems.append(f"{mp.name} is not a {SCHEMA_MANIFEST}")
    if man.get("target") != target:
        problems.append(f"the manifest is of {man.get('target')!r}, the flip ships {target!r}")
    if man.get("points_exact") != expect_count:
        problems.append(f"the manifest counts {man.get('points_exact')} points, the flip expects {expect_count}")
    if man.get("shippable") is not True or man.get("shippable_blockers"):
        problems.append(f"not shippable: {man.get('shippable_blockers')}")
    dc = man.get("data_checks") or {}
    doc = None
    if not dc.get("file"):
        problems.append("no data-checks verdict recorded in the manifest")
    else:
        dp = mp.parent / dc["file"]
        doc = _read(dp) if dp.is_file() else None
        if doc is None:
            problems.append(f"the data-checks verdict {dp} does not exist")
    if doc is not None:
        b = doc.get("binding") or {}
        rounds = len(man.get("catchup_rounds") or [])
        want = {"target": target, "plan_sha256": man.get("plan_sha256"), "round": rounds,
                "points_exact": expect_count}
        bad = {k: (b.get(k), v) for k, v in want.items() if b.get(k) != v}
        if doc.get("schema") != SCHEMA_DATA_CHECKS:
            problems.append(f"{dc['file']} is not a {SCHEMA_DATA_CHECKS}")
        acc = doc.get("accepted") or []
        if not doc.get("pass"):
            problems.append("the data checks did not PASS: " + ", ".join(
                k for k, v in (doc.get("checks") or {}).items() if not v.get("pass")))
        elif acc or any(not v.get("pass") for v in (doc.get("checks") or {}).values()):
            problems += check_accepted_entries(doc, plan_dir=pd, target=target,
                                               plan_file=plan_file or DEFAULT_PLAN_FILE)
        if bad:
            problems.append(f"the data checks were read on another state (got, want): {bad}")
    if problems:
        raise ShipRefused("REFUSED: the build is not proven shippable: " + "; ".join(problems))
    out = {"target": target, "points_exact": expect_count, "round": len(man.get("catchup_rounds") or []),
           "data_checks_at": doc.get("at"), "plan_sha256": man.get("plan_sha256")}
    if doc.get("accepted"):
        out["accepted"] = doc["accepted"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cytir.index.e10_ship", description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, help="<plan-dir>/manifest-<target>.json")
    ap.add_argument("--target", required=True)
    ap.add_argument("--expect-count", type=int, required=True)
    ap.add_argument("--plan-file", default=str(DEFAULT_PLAN_FILE),
                    help="plan.md: where a verdict's accepted rulings must be written (default the repo's)")
    a = ap.parse_args(argv)
    try:
        print(json.dumps(assert_shippable(a.manifest, target=a.target, expect_count=a.expect_count,
                                          plan_file=a.plan_file)))
    except ShipRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
