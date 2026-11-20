"""Static drift test for the consent request choke point (task 1623, spec section 4, test 7).

Scans `cytir/**/*.py` for the forbidden HTTP names (raw `httpx.Client(` and friends). Origin hosts go
through `cytir.sources.http.make_client`; operator services through `make_internal_client`. Excluded:
`cytir/consent/`, `cytir/sources/http.py`, `cytir/acquire/tape.py` (and this tool, which names them).

`data/http-drift-allowlist.v1.json` = {"modules": {path: reason}, "max": N}: the modules that still
offend. The set may only shrink: `--check` fails for an offender not listed AND for a listed module
that no longer offends (fix: `--rebuild`).

    python3 -m cytir.ops.http_drift            # print the offending modules
    python3 -m cytir.ops.http_drift --check
    python3 -m cytir.ops.http_drift --rebuild  # rewrite the file, keeping reasons

Landing rule: step-4 PRs shrink the file; the orchestrator takes `--theirs` on conflict, then runs
`python3 -m cytir.ops.http_drift --rebuild` in the batch (like the shim ratchet).
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

from cytir.core.paths import data_path

APP = pathlib.Path(__file__).resolve().parents[2]
ALLOWLIST = data_path("http-drift-allowlist.v1.json")
CENSUS = APP / "tasks" / "1592-consent-bypass-census.md"
FORBIDDEN = ("httpx.Client(", "httpx.AsyncClient(", "httpx.get(", "httpx.post(", "CurlClient(",
             "requests.get", "requests.Session", "urllib.request.urlopen")
EXCLUDED = ("cytir/consent/", "cytir/sources/http.py", "cytir/acquire/tape.py", "cytir/ops/http_drift.py")


def offenders(app: pathlib.Path = APP) -> list[str]:
    out = []
    for p in sorted((app / "cytir").rglob("*.py")):
        rel = p.relative_to(app).as_posix()
        if any(rel.startswith(e) for e in EXCLUDED):
            continue
        if any(n in p.read_text(errors="replace") for n in FORBIDDEN):
            out.append(rel)
    return out


def census_class(rel: str, census: str) -> str:
    """The census 1592 class of a module ('?' when the census does not list it)."""
    stem = rel.removeprefix("cytir/").removesuffix(".py")
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    pat = re.compile(r"(?<![\w/])" + re.escape(stem) + r"(?![\w/])")
    for line in census.splitlines():
        if not pat.search(line):
            continue
        if re.match(r"\|\s*\S+:\d", line):
            m = re.search(r"\|\s*([A-F])\s*\|\s*$", line)
            return f"EXTERNAL-WEB lane {m.group(1)}" if m else "EXTERNAL-WEB"
        for key in ("INTERNAL-SERVICE", "VERIFY", "THIRD-PARTY-API", "CONSENT-PROBE", "TEST-ONLY"):
            if line.startswith(f"| {key}"):
                return key
    return "?"


def rebuild(app: pathlib.Path = APP, allowlist: pathlib.Path = ALLOWLIST,
            census: pathlib.Path = CENSUS) -> dict:
    old = json.loads(allowlist.read_text())["modules"] if allowlist.exists() else {}
    text = census.read_text() if census.exists() else ""
    mods = {}
    for rel in offenders(app):
        mods[rel] = old.get(rel) or f"pre-existing at task 1623 (census 1592: {census_class(rel, text)})"
    doc = {"modules": mods, "max": len(mods)}
    allowlist.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return doc


def check(app: pathlib.Path = APP, allowlist: pathlib.Path = ALLOWLIST) -> list[str]:
    allowed = json.loads(allowlist.read_text())["modules"]
    found = set(offenders(app))
    errs = [f"{m}: raw HTTP client not in the allow-list; use make_client / make_internal_client"
            for m in sorted(found - set(allowed))]
    errs += [f"{m}: allow-listed but no longer offends (the set may only shrink); "
             "run: python3 -m cytir.ops.http_drift --rebuild" for m in sorted(set(allowed) - found)]
    return errs


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail on a new or stale allow-list entry")
    mode.add_argument("--rebuild", action="store_true", help="rewrite the allow-list, keeping reasons")
    args = ap.parse_args(argv)
    if args.rebuild:
        print(len(rebuild()["modules"]))
        return 0
    if args.check:
        errs = check()
        print("\n".join(errs) if errs else f"http-drift: OK ({len(offenders())} allow-listed)")
        return 1 if errs else 0
    print("\n".join(offenders()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
