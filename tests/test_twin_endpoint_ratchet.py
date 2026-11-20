"""Ratchet (task 832): the twin endpoint is named in ONE place, `cytir/core/paths.py` (settings
`twin_url` / `twin_collection`, read through `cytir.core.twin`). A new raw literal of the twin's port
or collection name in code, a shell driver or a unit fails here.

Shrink-only: ALLOWED maps a file to the exact number of raw hits it may keep. More hits fail. Fewer
also fail, with the new number to write, so the list can only get shorter. Receipts, rulings,
fixtures and sealed data (`data/`, `testdata/`, `fixtures/`) name the collection as HISTORY (a ruling
for `aw_e8` is not a ruling for anything else) and are never rewritten, so they are not scanned.
Test modules are not scanned either (a fixture may name a collection).

What a hit is: `:6457`, or the collection name as a quoted string, after `=`/`:`/`:-`, or as the
value of a collection flag.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
HIT = re.compile(r':6457\b|["\']aw_e8["\']|[=:]-?aw_e8\b|(?:--collection|--twin-collection|--source) aw_e8\b'
                 r'|/collections/aw_e8\b')
#: shell-ish files (units, drivers, extension-less scripts) also count a bare port number
BARE_PORT = re.compile(r'(?<![\d.:])6457(?!\d)')
SHELLISH = {".sh", ".conf", ".service", ".timer", ""}
SUFFIXES = {".py", ".sh", ".conf", ".service", ".json", ".env", ".yaml", ".yml", ".timer", ""}
SKIP_DIRS = {"data", "testdata", "fixtures", "tasks", "tests", "__pycache__", "node_modules", ".git"}
SCAN = ("cytir", "pipeline", "deploy", "server", "scripts", "webapp")

#: file (relative to the app root) -> how many raw hits it may keep, and why.
ALLOWED = {
    # THE one home: the defaults of the two settings.
    "cytir/core/paths.py": 2,
    # Historical one-shot flip / relocation drivers: each is a receipt of what ran on its day against
    # that day's collections. Rewriting them would rewrite the record (the flip is over).
    # task 865: now a thin wrapper keeping the aw_e9 flip's rollback (aw_e8 from task 668's backup).
    # The generalized deploy/scripts/flip-relocate.sh takes every name as an argument (0 hits).
    "deploy/scripts/flip-aw_e9-methodA.sh": 2,
    "deploy/scripts/attach-aw_e8.sh": 1,
    "deploy/scripts/relocate-e8-local-e3-volume.sh": 5,
    # A descriptive snapshot of the ERIC chain's environment (the settings probe reads it); the chain's
    # driver no longer needs the twin line (embed_compose_enqueue.sh reads the setting).
    "deploy/operator/probe-entrypoints.json": 1,
}


from tests.helpers.repo_files import repo_file_or_skip

repo_file_or_skip("deploy/operator/probe-entrypoints.json")  # the public cut withholds this file; skip there (task 2382)


def _hits():
    out = {}
    for top in SCAN:
        base = ROOT / top
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            rel = p.relative_to(ROOT)
            if not p.is_file() or p.suffix not in SUFFIXES or SKIP_DIRS & set(rel.parts):
                continue
            if p.name.startswith("test_"):
                continue
            try:
                text = p.read_text(encoding="utf-8")
                n = len(HIT.findall(text))
                if p.suffix in SHELLISH:
                    n += len(BARE_PORT.findall(HIT.sub("", text)))
            except (OSError, UnicodeDecodeError):
                continue
            if n:
                out[rel.as_posix()] = n
    return out


def test_no_new_raw_twin_literal():
    found = _hits()
    grew = {f: (n, ALLOWED.get(f, 0)) for f, n in found.items() if n > ALLOWED.get(f, 0)}
    assert not grew, (
        "raw twin endpoint literal(s) outside the allowlist (file: found, allowed). Take the twin from "
        "`cytir.core.twin` (Python) or `aw_setting_in \"$APP\" twin_url` (shell), never a literal: "
        f"{grew}")


def test_the_allowlist_only_shrinks():
    found = _hits()
    stale = {f: (found.get(f, 0), n) for f, n in ALLOWED.items() if found.get(f, 0) < n}
    assert not stale, (
        f"the allowlist is longer than the tree (file: found, allowed). Lower it to the found count, "
        f"or delete the row at 0: {stale}")


#: Only the settings loader and the accessor may read the twin's environment names: another reader
#: with another default is how one name came to mean three twins (review of #764).
ENV_NAMES = re.compile(r"AW_TWIN_URL|AW_TWIN_COLLECTION")
ENV_HOMES = {"cytir/core/paths.py", "cytir/core/twin.py"}


def test_only_the_loader_and_the_accessor_read_the_twin_env_names():
    bad = {}
    for top in SCAN:
        base = ROOT / top
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            rel = p.relative_to(ROOT)
            if (not p.is_file() or p.suffix not in SUFFIXES or SKIP_DIRS & set(rel.parts)
                    or p.name.startswith("test_") or rel.as_posix() in ENV_HOMES):
                continue
            try:
                n = len(ENV_NAMES.findall(p.read_text(encoding="utf-8")))
            except (OSError, UnicodeDecodeError):
                continue
            if n:
                bad[rel.as_posix()] = n
    assert not bad, f"these read the twin env names themselves; use cytir.core.twin / aw_setting_in: {bad}"
