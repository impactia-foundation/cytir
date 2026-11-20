"""Serve parity gate (S10, task 752): a serving-code move must leave every ranking identical.

    python -m cytir.gates.serve_parity prepare --base-tree A --tei-url URL --seed-cache C.json \
        --out-dir OUT [sets]                            # one vector cache for every query, once
    cyt gate serve-parity run --base-tree A --head-tree B --collection NAME \
        --qdrant-url URL --vec-cache OUT/vectors.json --out-dir OUT [sets] [--shards N]

What it compares. Two source trees (`--base-tree`, `--head-tree`; each holds `cytir/serve`
and `cytir/`) answer the same requests through the server's own FastAPI app, in process, against the
same read-only twin collection. Per request it runs, in this order:

    base/served  base/deep  head/served  head/deep  base/served again  base/deep again (the nulls)

`served` is prod's serving environment exactly (`cytir.ops.prod_env.resolve_env`, compose plus every
overlay; the BASE tree's, and the head tree's must match it outside the timing knobs). A
config-only change (task 2816: the two trees declare different ranking envs but hold identical
serving inputs, `serving_inputs_digest`) has no code to compare: every arm then runs the HEAD
tree's env, the report records `config_change` {env_source: head, changed: {KEY: {base, head}}},
and the verdict is the normal one (identical inputs under one env must PASS). Different env AND
different serving inputs: refused, unless the ONLY moved serving inputs are under `cytir/` outside
`cytir/serve/` (a profile VALUE lives in cytir/core; task 2832). Then the head code is served under the
BASE env, `config_change` is {env_source: base, head_code_under_base_env, changed, moved_inputs,
receipt}, and only a PASS (every request EQUAL) is accepted; else refused with the DIFF count:
split the PR. A moved path under `cytir/serve/`, `data/` or `deploy/operator/` is refused before any
run. A config-only report (same inputs) never licenses a release: `check` refuses a report whose
trees hold the same server code; it also refuses every base-env report.
`deep` adds the two read-only instruments the server already has, `AW_DEBUG_SCORES=1`
and `AW_QHEAD_COUNTRY_TRACE=1`: every stage-2 row's components, the leg points with their raw
scores, and the whole deduplicated ranking with point ids. The served body must be byte-equal
(canonical JSON); the deep record must have identical ids in identical order and every number
within `--score-tol`. The real resolution is set by the server's own rounding, not by the
tolerance: the served `score` and the debug `fused`/`dense_raw` carry 4 decimals, the debug
components 6, the trace's final scores 8. So the gate sees every ORDER change in the whole
deduplicated ranking, and score changes of about 1e-6 and larger. The second base calls bracket the head calls in time: when either differs
from the first one the twin moved (a write shifts BM25 statistics) or the code is not
deterministic, and the request is UNSTABLE, not a diff. Unstable requests are retried; a verdict
counts only stable ones.

Why in process and per request. The server reads its flags at import, so each arm is its own
process (one per tree and mode), started once and fed requests over a pipe. Both trees query
the twin within milliseconds of each other, so a write by another lane between them shows up
in the null, not in the A/B. Query vectors come from one cache file both arms read
(`prepare`): the embedder is not a variable. `PYTHONHASHSEED` is pinned in every arm, because
`search()` sorts sets of string point ids and Python's string hash is salted per process: equal
scores would otherwise reorder between two processes of the same code. `--seed-probe` adds an
arm with a different seed and reports how many requests are sensitive to that (information:
prod restarts with a random seed).

Verdicts (exit codes): PASS 0 - every stable request equal, null pair clean, both arms report
the same configuration; FAIL 1 - a request differs, or the two arms load a different
configuration; INCONCLUSIVE 3 - no diff, but a request stayed unstable after the retries;
ERROR 4 - an arm answered non-200, missed a cached vector, imported the wrong tree, or the deep
instrument changed the served answer. REFUSED 2 - not the twin (`cytir.retune.guard`).

`check` is the release door's half (task 792, fix round 799): `release-serving.sh` calls
`cyt gate serve-parity check --report parity.json --base-sha PROD --head-sha NEW
--base-server-dir LIVE --head-server-dir NEW_TREE` before it renders a serving-code change. Exit 0
only when that report is a PASS for exactly that (prod sha, new sha) pair; BOTH trees are bound by
content digest (the base arm ran on the code the box serves NOW, the head arm on the code being
released, and the two differ); and there is no unstable, differing or erroring request and a clean
null pair. Exit 1 names every reason it is not; exit 2 = unreadable report or usage.
`run --base-sha/--head-sha` records the pair in the report (a report without it never satisfies
`check`).

`check --envelope-change REF --envelope-fields a.b,c.d` (task 2679) is the declared mode for a release
that changes the response ENVELOPE (a new `boosts` key, a `filters.note`) and not the ranking. Every
request records `ranked_equal` per tier (served: the body's `results` ids, order and scores exactly;
deep: the whole deep record equal within the tolerance, any trace difference counts) and `envelope_paths` (the differing
JSON paths outside the ranked list, key-set differences expanded one level, list indexes folded to
`[]`; deep has no envelope paths and a field starting `deep.` is refused). Fields match EXACT
paths: a bare parent (`filters`) does not cover its children (`filters.note`). A FAIL report is accepted only when ERROR = 0,
UNSTABLE = 0, no configuration difference, a clean null pair, every DIFF request is `ranked_equal` at
both tiers with `envelope_paths` inside the declared fields, and REF appears in plan.md or
decisions.md AT the release sha (same rule as --ranking-change). A report made before this schema
(no `ranked_equal`) is refused. It cannot be combined with --ranking-change.

Per-result fields (task 2712): `--envelope-fields` also takes `results[].<field>` (one level, a plain
field name) for an additive field on each served result (`read_url`). A request whose results differ
records `ranked_equal.served_stripped` (equal once every differing per-result field is removed from
BOTH arms: count, order, urls, scores and every other field byte-exact), the paths `results[].<f>`
in `envelope_paths`, and `result_field_counts` {f: {base: n, head: n}} (how many results carried
it). Refused at argument parsing: `results[].*`, nested `results[].a.b`, a bare `results`, and any
ranking key (`RANKING_RESULT_KEYS`: every key `respond.py` emits today). A change to a
ranking key is never stripped, so the request is not ranked-equal and is refused.

`check --ranking-change REF --max-diff N --null-report R2` is the declared mode for a release that
changes ranking ON PURPOSE (a tie-break, task 758). It accepts a FAIL report only when ALL hold:
ERROR = 0, UNSTABLE = 0, no configuration differences, a clean null pair, DIFF <= N, both trees
bound by digest, R2 is a head-vs-head PASS (release vs itself, `--seed-probe`, 0 seed-sensitive
requests), and REF appears in plan.md or decisions.md AT THE RELEASE SHA (the ruling is part of the
tree being released). The differing requests are printed for the release doc. A change to a served
DATA FILE (a weights promotion) is recorded in the report (`data_file_diffs`) but is a
configuration difference, so this mode does not cover it yet.

Configuration equality of a served JSON table ignores its `_comment` keys at every depth (they are
documentation); any other key or value counts, and non-JSON files are byte-compared. The tree digest
that binds `check` to the two trees still covers the raw bytes: tree binding = bytes, config
equality = semantics (task 841).

The harness holds no scoring. It never writes to Qdrant. Runtime and the numbers of its first
runs: specs/s10-serve-split.md section 4.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import importlib
import json
import math
import os
# Loaded by path against FOREIGN trees (a base tree may predate `cytir.core.envnames`): no package import here.
_KNOB_PREFIXES = ("AW_", "CYTIR_")
_MIRROR_MARKER = "_CYTIR_ENV_MIRRORED"   # same literal as cytir.core.envnames.MARKER (a test pins it)


def _is_knob(name: str) -> bool:
    return name.startswith(_KNOB_PREFIXES) or name == _MIRROR_MARKER
import pathlib
import re
import subprocess
import sys
import threading
import time

SCHEMA = "aw.serve-parity.v1"
CACHE_SCHEMA = "aw.serve-parity-vectors.v1"
MARK = "@@serve-parity@@ "
#: A request whose base arm disagrees with itself is retried this many times.
RETRIES = 2
#: Timing only. The twin is shared with other lanes; a slow call is not a ranking.
#: AW_AUTHLEG_FILTERED_TIMEOUT (task 2665) bounds one leg by wall clock, so a filtered request on a
#: busy twin could drop its authority admits in one arm and not the other: both arms run it at 0
#: (the single-batch dispatch), whatever each tree's compose declares (review 2671 F2).
TIMING_ENV = {"AW_QDRANT_TIMEOUT": "60", "AW_AUTHLEG_FILTERED_TIMEOUT": "0",
              "AW_STAGE1_TIMEOUT": "0"}
#: The two read-only instruments the deep arms switch on. Neither changes a score (the harness
#: checks that, per request: deep body == served body of the same tree).
DEEP_ENV = {"AW_DEBUG_SCORES": "1", "AW_QHEAD_COUNTRY_TRACE": "1"}
#: Trace keys that measure time, not ranking.
VOLATILE_KEYS = frozenset({"stage1_request_ms"})
SEED = "0"
PROBE_SEED = "1"
EXIT = {"PASS": 0, "FAIL": 1, "REFUSED": 2, "INCONCLUSIVE": 3, "ERROR": 4}

APP = pathlib.Path(__file__).resolve().parents[2]
# Not imported from `cytir.serve`: this file is loaded by path inside an arm (see BOOT) and must not
# import `aw` at module level. tests/test_aw_serve_parity.py holds the two equal.
SERVER_REL = "cytir/serve"

# The package a tree carries is asked of THAT tree (RULING 82-85, the `aw` -> `cytir` rename): the base
# tree of the first release after the rename is an old `aw` tree, the head a `cytir` tree (plus the
# thin `aw` import shim). An inline copy of `cytir.core.serve_layout` (same reason: no import of the
# package at module level, and this file is loaded by path against old trees that lack the new
# module); tests/test_aw_package_name_resolver.py holds the two equal.
PACKAGE_NAMES = ("cytir", "aw")  # rename-codemod: keep
SHIM_MARKER = "rename-codemod: shim"
SHIM_HEAD_BYTES = 400


def _is_shim(init: pathlib.Path) -> bool:
    try:
        return SHIM_MARKER in init.read_bytes()[:SHIM_HEAD_BYTES].decode("utf-8", "replace")
    except OSError:
        return False


def package_name(tree) -> str:
    """`cytir` when `<tree>/cytir/__init__.py` exists (it wins over the shim; beside a real
    old-package dir that is not the shim it is refused), else `aw` when
    `<tree>/aw/__init__.py` exists and is not the shim; anything else is refused. The one resolver
    behind every `<tree>/<package>/...` path of a tree other than this one. `serve_parity` holds an
    inline copy (it is loaded by path inside an arm); tests/test_aw_package_name_resolver.py pins them."""
    tree = pathlib.Path(tree)
    new, old = PACKAGE_NAMES
    init = tree / old / "__init__.py"
    if (tree / new / "__init__.py").is_file():
        if init.is_file() and not _is_shim(init):
            # a half-renamed tree (a copy without delete): never pick one of two real packages silently
            raise SystemExit(f"{tree} carries two real packages, {tree / new} and {tree / old} (its "
                             f"__init__.py is not the import shim): a stale directory, refusing to guess")
        return new
    if init.is_file() and not _is_shim(init):
        return old
    raise SystemExit(f"{tree} carries no serving package ({new}/ or {old}/ with its own __init__.py; "
                     f"a {old}/ that is only the import shim does not count): refusing to guess")


#: Where a serving data file sits inside the api image (`/app/<package>/serve/`): the compose files
#: of a tree name it under THAT tree's package name, so both spellings map into the arm's own tree.
IMAGE_SERVE_PREFIXES = tuple(f"/app/{p}/serve/" for p in PACKAGE_NAMES)


# --------------------------------------------------------------------------------------------- #
# requests
# --------------------------------------------------------------------------------------------- #

def _req(set_name: str, rid: str, q: str, k: int = 10, **extra) -> dict:
    params = {"q": q, "k": int(k)}
    for key in ("lang", "country", "authorities"):
        v = extra.get(key)
        if v not in (None, "", 0, False):
            params[key] = v
    for key in ("site", "exclude_site", "source"):          # the named operator parameters (task 2640)
        v = extra.get(key)
        if v:
            params[key] = list(v)
    req = {"id": f"{set_name}:{rid}", "set": set_name, "params": params}
    if extra.get("expect_status"):
        req["expect_status"] = int(extra["expect_status"])
    if extra.get("embed_q"):
        req["embed_q"] = extra["embed_q"]
    return req


# --------------------------------------------------------------------------------------------- #
# the operator slice (task 2652)
# --------------------------------------------------------------------------------------------- #
# One request per operator path of `/search` (task 2640). The hosts are the stable ones of the
# index (`site:` expands through a host table built from a live facet: a host that comes or goes
# would move the answer of BOTH arms alike, but a flaky host would show as UNSTABLE; these five are
# in the index for good). `source:` values are the four biggest source_type values. The served body
# is compared whole, so `filters` (the echo of what was applied: hosts, sources, lang, country,
# note) is part of every comparison: a change that drops a filter but keeps results still DIFFs.
# `embed_q` = the text the server embeds once the operators are cut (`parse_operators(q)[0]`); the
# vector cache is keyed on it (a test pins it). `expect_status` = a 400 is the right answer.
# `site:` on a host the index does not hold: the request falls back to the exact host, which matches
# nothing: 0 results, `filters.hosts` echoes it (no 400, no 500).
OPERATOR_SLICE = (
    ("site-exact", "Art. 336 OR Kündigung Missbrauch site:fedlex.admin.ch", {}),
    ("site-parent", "Bundesgesetz Datenschutz site:admin.ch", {}),
    ("site-multi", "python requests session timeout site:github.com site:stackoverflow.com", {}),
    ("minus-site", "Schweizer Geschichte Reformation -site:wikipedia.org", {}),
    ("site-wikipedia", "Reformation Zwingli site:wikipedia.org", {}),
    ("source", "Mehrwertsteuer Satz Schweiz source:official", {}),
    ("source-multi", "python asyncio gather source:docs source:community", {}),
    ("lang", "Mietrecht Kündigungsfrist lang:de", {}),
    ("country", "data protection act country:CH", {}),
    ("combo", "droit du travail congé site:admin.ch -site:wikipedia.org lang:fr country:CH", {}),
    ("param-site", "Kündigungsfrist Arbeitsvertrag", {"site": ["fedlex.admin.ch"]}),
    ("param-exclude-site", "Reformation Zwingli", {"exclude_site": ["wikipedia.org"]}),
    ("param-source", "Mehrwertsteuer Satz Schweiz", {"source": ["official"]}),
    ("param-and-operator", "Datenschutz site:admin.ch", {"site": ["github.com"]}),
    ("override-note", "Mietrecht lang:fr", {"lang": "de"}),
    ("authorities", "Kündigungsfrist site:admin.ch", {"authorities": 1}),
    ("unknown-source", "import os source:zzznotasource", {}),
    ("unknown-host", "Kündigungsfrist site:no-such-host.invalid", {}),
    ("only-operator", "site:admin.ch", {"expect_status": 400}),
    ("only-operators", "site:admin.ch -site:github.com lang:de", {"expect_status": 400}),
)
OPERATOR_EMBED = {   # id -> what the server embeds, when it differs from the raw query
    "site-exact": "Art. 336 OR Kündigung Missbrauch",
    "site-parent": "Bundesgesetz Datenschutz",
    "site-multi": "python requests session timeout",
    "minus-site": "Schweizer Geschichte Reformation",
    "site-wikipedia": "Reformation Zwingli",
    "source": "Mehrwertsteuer Satz Schweiz",
    "source-multi": "python asyncio gather",
    "lang": "Mietrecht Kündigungsfrist",
    "country": "data protection act",
    "combo": "droit du travail congé",
    "param-and-operator": "Datenschutz",
    "override-note": "Mietrecht",
    "authorities": "Kündigungsfrist",
    "unknown-host": "Kündigungsfrist",
    "unknown-source": "import os source:zzznotasource",     # an unknown value stays query text
}


def operator_requests() -> list[dict]:
    out = []
    for rid, q, extra in OPERATOR_SLICE:
        extra = dict(extra)
        if rid in OPERATOR_EMBED:
            extra["embed_q"] = OPERATOR_EMBED[rid]
        out.append(_req("operators", rid, q, **extra))
    return out


def chooser_requests(path: pathlib.Path) -> list[dict]:
    """The 629 chooser, as `cytir.retune.arm` sends it: q and k only."""
    import yaml
    if not path.is_file():
        raise SystemExit(f"chooser set missing at {path} (pass --chooser, or --sets without chooser)")
    rows = yaml.safe_load(path.read_text())["eval"]
    return [_req("chooser", str(r["id"]), r["query"]) for r in rows]


def heldout_requests() -> list[dict]:
    """The held-out battery, as `cytir.gates.authority.run` sends it: q, k=10, the row's lang and
    country. Every set the gate scores, including the slice-B sample."""
    from cytir.gates import authority as gate
    out = []
    for name, path in gate.SETS:
        if not os.path.exists(path):
            raise SystemExit(f"held-out set {name} missing at {path}: a moved yardstick is not "
                             f"a parity run (pass --sets without heldout to skip it on purpose)")
        for j, r in enumerate(gate.load_rows(path)):
            rid = str(r.get("id") or r.get("qid") or j)
            out.append(_req(f"heldout.{name}", rid, r.get("query") or r.get("q"),
                            lang=r.get("lang"), country=r.get("country")))
    if os.path.exists(gate.SLICEB):
        for j, r in enumerate(gate.load_sliceb(gate.SLICEB)):
            if r.get("query"):
                out.append(_req("heldout.slice_b", str(r.get("qid") or j), r["query"],
                                lang=r.get("lang"), country=r.get("country")))
    return out


def release_requests(paths: list[pathlib.Path]) -> list[dict]:
    """The task-665 release slices (new + control): the requests that arm the AW_VERSION_MATCH
    path (W=1.0 in prod, plan ruling 12). Neither the chooser nor the battery holds many."""
    from cytir.gates import authority as gate
    out = []
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"release set missing at {path}")
        for j, r in enumerate(gate.load_rows(str(path))):
            out.append(_req(f"release.{path.stem}", str(r.get("id") or j),
                            r.get("query") or r.get("q"), lang=r.get("lang"),
                            country=r.get("country")))
    return out


def probe_requests(runs_dir: pathlib.Path, days: int) -> list[dict]:
    """The probe's own requests of its last `days` run days, with the parameters it sent."""
    dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir() and p.name[:4].isdigit())[-days:]
    if not dirs:
        raise SystemExit(f"no probe run days under {runs_dir}")
    out = []
    for d in dirs:
        for fname, sliceb in (("sample.jsonl", False), ("sample_sliceb.jsonl", True)):
            f = d / fname
            if not f.is_file():
                continue
            for j, line in enumerate(f.read_text().splitlines()):
                if not line.strip():
                    continue
                r = json.loads(line)
                q = r.get("query")
                if not q:
                    continue
                if sliceb:
                    out.append(_req(f"probe.{d.name}.b", str(r.get("qid") or j), q,
                                    lang=r.get("lang"), country=r.get("country")))
                else:
                    p = r.get("params") or {}
                    out.append(_req(f"probe.{d.name}", str(r.get("probe_id") or j), q,
                                    k=p.get("k", r.get("k", 10)), lang=p.get("lang"),
                                    country=p.get("country"), authorities=p.get("authorities")))
    return out


def fixture_requests(path: pathlib.Path) -> list[dict]:
    """The CI fixture's own gold slice (`cyt gate`'s `data/gate-slices/ci-fixture.v1.json`), as
    `cytir.gates.slice_gate` sends it: q, k=10, the gold's lang and country (task 2254)."""
    if not path.is_file():
        raise SystemExit(f"fixture gold slice missing at {path} (pass --fixture)")
    golds = json.loads(path.read_text())["golds"]
    return [_req("fixture", str(g["id"]), g["query"], lang=g.get("lang"), country=g.get("country"))
            for g in golds]


def dedupe(requests: list[dict]) -> list[dict]:
    """One request per distinct parameter set; the first id wins, the others are recorded."""
    seen: dict[str, dict] = {}
    for r in requests:
        key = json.dumps(r["params"], sort_keys=True, ensure_ascii=False)
        if key in seen:
            seen[key].setdefault("also", []).append(r["id"])
        else:
            seen[key] = dict(r)
    return list(seen.values())


def collect(args) -> list[dict]:
    want = [s.strip() for s in args.sets.split(",") if s.strip()]
    reqs: list[dict] = []
    for s in want:
        if s == "chooser":
            reqs += chooser_requests(pathlib.Path(args.chooser))
        elif s == "heldout":
            reqs += heldout_requests()
        elif s == "fixture":
            reqs += fixture_requests(pathlib.Path(args.fixture))
        elif s == "release":
            reqs += release_requests([pathlib.Path(x) for x in args.release.split(",")])
        elif s == "operators":
            reqs += operator_requests()
        elif s == "probe":
            if not args.probe_runs:
                raise SystemExit("--sets probe needs --probe-runs DIR (the probe's runs/ folder)")
            reqs += probe_requests(pathlib.Path(args.probe_runs), args.probe_days)
        else:
            raise SystemExit(f"unknown set {s!r} (chooser, heldout, release, probe, fixture, operators)")
    reqs = dedupe(reqs)
    if args.limit:
        reqs = reqs[: args.limit]
    return reqs


# --------------------------------------------------------------------------------------------- #
# environment of an arm
# --------------------------------------------------------------------------------------------- #

def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_comments(o):
    """A JSON value without its `_comment` keys, at every depth."""
    if isinstance(o, dict):
        return {k: _strip_comments(v) for k, v in o.items() if k != "_comment"}
    if isinstance(o, list):
        return [_strip_comments(x) for x in o]
    return o


def _config_sha(path: pathlib.Path) -> str:
    """Hash of a served file for CONFIGURATION equality (not tree binding): a `.json` file by its
    canonical JSON without `_comment` keys (documentation, not configuration; any other key or
    value still counts); every other file, and a `.json` that does not parse, by its bytes. The
    tree digest (`_source_digest`) still covers the raw bytes: tree binding = bytes, config
    equality = semantics."""
    if path.suffix == ".json":
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return _sha256(path)
        return hashlib.sha256(json.dumps(_strip_comments(doc), sort_keys=True,
                                         ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return _sha256(path)


def serving_env(tree: pathlib.Path) -> dict:
    """Prod's aw-api environment as the tree's own deploy files declare it (compose + every
    overlay; the shipped default when the tree has no `deploy/`). Container paths stay
    container paths here; `arm_env` maps them into each tree."""
    from cytir.ops import prod_env
    compose = tree / "deploy" / "docker-compose.yaml"
    if compose.is_file():
        merged = {}
        for f in [compose] + prod_env.overlay_files(tree / "deploy" / "compose-overlays.manifest"):
            merged.update(prod_env._service_env(f))
        return merged
    # the TREE's shipped default (not the harness's own copy): a public cut has no deploy/
    default = tree / package_name(tree) / "ops" / prod_env.DEFAULT_ENV.name
    doc = json.loads((default if default.is_file() else prod_env.DEFAULT_ENV).read_text())
    return dict(doc["env"])


#: What a serving process reads from a tree besides its env: the whole package (the server imports
#: its core, its sources..., lazily too, so an import closure would under-cover), the repo's
#: `data/` contracts (`core.paths.data_path`) and the operator overlay. Every file, by relative
#: path and bytes; `__pycache__` skipped. The package directory is the one the TREE carries
#: (`package_name`), so an old `aw` tree and a `cytir` tree each digest their own.
SERVING_INPUT_OTHER_ROOTS = ("data", "deploy/operator")
#: Derived from deploy/ (`prod_env --write-default`): it IS the env, which `serving_env` compares
#: on its own; a config-only PR regenerates it with the compose change.
SERVING_INPUT_EXCLUDE_REL = "ops/serving_env.default.json"
#: In `moved_inputs` and its callers a path under the package is spelled `~/...`, whatever the
#: tree calls the package: the same file of an old and a new tree is then the same key, and a
#: different byte under it is still a moved input. The bytes are never normalised.
PACKAGE_KEY = "~"


def serving_input_roots(tree: pathlib.Path) -> tuple[str, ...]:
    return (package_name(tree),) + SERVING_INPUT_OTHER_ROOTS


def _input_files(tree: pathlib.Path, root: str) -> list[pathlib.Path]:
    """The files under `tree/root`, sorted. In a git checkout: tracked plus untracked-not-ignored
    (working-copy bytes), so what a job leaves in ignored paths (state files, caches) is not a
    serving input; elsewhere every file. An EMPTY git listing is not trusted (a tree whose files an
    unrelated enclosing repo ignores lists nothing for a populated dir): every file then."""
    try:
        out = subprocess.run(["git", "-C", str(tree), "ls-files", "-z", "-co", "--exclude-standard",
                              "--", root], capture_output=True, check=True, timeout=120).stdout
        files = [tree / x for x in out.decode().split("\0") if x]
    except (OSError, subprocess.SubprocessError):
        files = []
    if not files:
        files = list((tree / root).rglob("*"))
    return sorted(f for f in files if f.is_file() and "__pycache__" not in f.parts)


def _input_key(tree: pathlib.Path, path: pathlib.Path, pkg: str) -> str:
    rel = path.relative_to(tree).as_posix()
    return PACKAGE_KEY + rel[len(pkg):] if rel.startswith(pkg + "/") else rel


def _serving_input_files(tree: pathlib.Path) -> dict[str, pathlib.Path]:
    """{key: file} of every serving input of `tree` (exclusions applied; see PACKAGE_KEY)."""
    pkg = package_name(tree)
    out = {}
    for root in serving_input_roots(tree):
        if not (tree / root).is_dir():
            continue
        for p in _input_files(tree, root):
            key = _input_key(tree, p, pkg)
            if key == f"{PACKAGE_KEY}/{SERVING_INPUT_EXCLUDE_REL}" or p.suffix == ".pyc":
                continue
            out[key] = p
    return out


def serving_inputs_digest(tree: pathlib.Path) -> str:
    """sha256 over every serving input of `tree` other than its env. Two trees with the same
    digest run the same code on the same data: a difference in their declared env is then a pure
    configuration change (task 2816). Each file enters by its relative path and its bytes; the
    package directory keeps its name in the hashed path, and a file that merely says `aw` where its
    twin says `cytir` has other bytes: the rename moves this digest, by design (a content
    difference must always show), and the rename release is judged on the served answers."""
    h = hashlib.sha256()
    for root in serving_input_roots(tree):
        base = tree / root
        if not base.is_dir():
            h.update(f"missing:{root}".encode() + b"\0")
            continue
        n = 0
        pkg = package_name(tree)
        for p in _input_files(tree, root):
            rel = p.relative_to(tree).as_posix()
            if rel == f"{pkg}/{SERVING_INPUT_EXCLUDE_REL}" or p.suffix == ".pyc":
                continue
            h.update(rel.encode() + b"\0")
            h.update(_sha256(p).encode())
            n += 1
        if root == pkg and not n:
            # two empty digests are equal: a code+env PR would pass as config-only
            raise SystemExit(f"{tree}/{pkg} lists no files: refusing an empty serving-code digest")
    return h.hexdigest()


def moved_inputs(base_tree: pathlib.Path, head_tree: pathlib.Path) -> list[str]:
    """The serving-input paths (same exclusions) whose bytes differ, or that only one tree has. A
    file under the package is the same file in an old and a new tree whatever the package is called
    (PACKAGE_KEY); it is NAMED with the head tree's package directory."""
    a = {k: _sha256(p) for k, p in _serving_input_files(base_tree).items()}
    b = {k: _sha256(p) for k, p in _serving_input_files(head_tree).items()}
    head_pkg = package_name(head_tree)
    return sorted((head_pkg + k[len(PACKAGE_KEY):] if k.startswith(PACKAGE_KEY + "/") else k)
                  for k in set(a) | set(b) if a.get(k) != b.get(k))


#: With a declared-env change, only these may move as well (task 2832): the profile value lives
#: in `<package>/core`; `<package>/serve` (what release-serving ships), data/ and deploy/operator/
#: may not. `path` is named with the head tree's package directory `pkg` (see `moved_inputs`).
def _value_change_blocked(path: str, pkg: str) -> bool:
    return not path.startswith(pkg + "/") or path.startswith(pkg + "/serve/")


def _canon_env_value(v, pkg: str | None):
    """A declared value that names a file in the api image's serving package reads `<image-serve>/x`,
    but ONLY with the prefix `/app/<pkg>/serve/` of the package `pkg` of the tree that declared it:
    a compose of an old tree and one of a renamed tree spell the same file with their own package
    names. A value carrying the OTHER package's image prefix is not canonicalised (a stale spelling is
    a real difference: prod would open a missing file and switch the feature off). The file's BYTES
    are compared separately (`data_files`), so no content difference hides behind this."""
    if isinstance(v, str) and pkg:
        pre = f"/app/{pkg}/serve/"
        if v.startswith(pre):
            return "<image-serve>/" + v[len(pre):]
    return v


def env_changes(base_env: dict, head_env: dict, base_pkg: str | None = None,
                head_pkg: str | None = None) -> dict:
    """{KEY: {base, head}} for every ranking key (outside TIMING_ENV: those are forced per arm)
    whose value differs; an absent key is None. `base_pkg` / `head_pkg` = the package each tree
    carries: each side's image-serve prefix is read with ITS OWN package only. With none given, the
    values are compared as written."""
    keys = (set(base_env) | set(head_env)) - set(TIMING_ENV)
    return {k: {"base": base_env.get(k), "head": head_env.get(k)} for k in sorted(keys)
            if _canon_env_value(base_env.get(k), base_pkg) != _canon_env_value(head_env.get(k), head_pkg)}


def refuse_foreign_image_prefix(env: dict, pkg: str) -> None:
    """Refuse a declared value that names a serving file under the image prefix of ANOTHER package
    than `pkg`, the package of the tree that declared `env` (a stale spelling: prod would open a
    missing file and switch the feature off silently)."""
    own = f"/app/{pkg}/serve/"
    for k, v in env.items():
        prefix = next((x for x in IMAGE_SERVE_PREFIXES if isinstance(v, str) and v.startswith(x)), None)
        if prefix and prefix != own:
            raise SystemExit(f"{k}={v} carries the image prefix {prefix}, but the tree that declares it "
                             f"carries the package {pkg}/ (expects {own}): a stale package spelling is "
                             "refused, not mapped")


def arm_env(declared: dict, tree: pathlib.Path, *, collection: str, qdrant_url: str,
            deep: bool, seed: str = SEED, base: dict | None = None,
            declared_pkg: str | None = None) -> tuple[dict, dict]:
    """(env, data_files) of one arm. No inherited AW_* reaches the arm; container paths of the
    server's own data files map into this tree, and a mapped file that is missing refuses (the
    server would otherwise disable the feature it configures and the arm would rank differently
    from prod without saying so). `declared_pkg` = the package of the tree whose compose declared
    `declared` (default: this tree's own): a value with `/app/<declared_pkg>/serve/` maps into this
    tree's serving package; a value with the image prefix of ANOTHER package refuses (a stale
    spelling is never mapped silently)."""
    # PYTHONPATH is inherited on purpose (a dependency outside the venv, e.g. the pinned
    # CharSplit); the worker's boot puts the arm's tree AHEAD of it and checks where `aw` and
    # `cytir.serve` were imported from.
    env = {k: v for k, v in (os.environ if base is None else base).items()
           if not _is_knob(k) and k != "PYTHONHASHSEED"}
    files = {}
    refuse_foreign_image_prefix(declared, declared_pkg or package_name(tree))
    for k, v in declared.items():
        prefix = next((x for x in IMAGE_SERVE_PREFIXES if v.startswith(x)), None)
        if prefix:
            local = serve_dir(tree) / v[len(prefix):]
            if not local.is_file():
                raise SystemExit(f"{k}={v} maps to {local}, which does not exist")
            v = str(local)
            files[k] = _config_sha(local)
        env[k] = v
    for k in ("AW_MODELS_DIR",):          # a container path with no meaning outside one
        if env.get(k, "").startswith("/models/"):
            env.pop(k, None)
            env.pop("CYTIR_" + k[3:], None)   # the arm env holds the AW_ spelling; both go (no envnames import)
    env.update({"AW_COLLECTION": collection, "AW_QDRANT_URL": qdrant_url, "AW_TEI_URL": "",
                "AW_WEBAPP": str(tree / "webapp" / "index.html"), **TIMING_ENV})
    if deep:
        env.update(DEEP_ENV)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env, files


# --------------------------------------------------------------------------------------------- #
# the worker: runs INSIDE an arm process, with the arm's tree first on sys.path. It must not
# import `aw` at module level: this file is loaded by path, so `aw` resolves to the arm's tree.
# --------------------------------------------------------------------------------------------- #

BOOT = (
    "import sys, importlib.util\n"
    "tree, harness = sys.argv[1], sys.argv[2]\n"
    "sys.path[:0] = [tree]\n"
    "spec = importlib.util.spec_from_file_location('aw_serve_parity_worker', harness)\n"
    "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
    "raise SystemExit(mod.worker_main(sys.argv[3:]))\n"
)


def _emit(doc: dict) -> None:
    # The leading newline ends any partial line a print() in the server left in the buffer.
    sys.stdout.write("\n" + MARK + json.dumps(doc, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def serve_dir(tree: pathlib.Path) -> pathlib.Path:
    """The serving package of a source tree: `<package>/serve`, the package being the one THAT tree
    carries (an old `aw` base and a `cytir` head are served from their own directories)."""
    return tree / package_name(tree) / "serve"


def serve_package(tree: pathlib.Path) -> str:
    """The module name of that tree's serving package: `<package>.serve`."""
    return package_name(tree) + ".serve"


def _source_digest(srv_dir: pathlib.Path) -> str:
    """sha256 over every `*.py` and `*.json` under `srv_dir` (code AND the served tables: weights,
    qhead, host lists), by relative path and bytes. `__pycache__` is skipped."""
    h = hashlib.sha256()
    for p in sorted(q for q in srv_dir.rglob("*") if q.is_file() and q.suffix in (".py", ".json")
                    and "__pycache__" not in q.parts):
        h.update(str(p.relative_to(srv_dir)).encode() + b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def _content_sha(obj) -> str:
    """sha256 of a loaded object's CONTENT (dicts by key, sets sorted, arrays as lists): two
    files of the same size but different content must not fingerprint alike (task 757 S6)."""
    def norm(o):
        if isinstance(o, dict):
            return {str(k): norm(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
        if isinstance(o, (set, frozenset)):
            return sorted((norm(x) for x in o), key=repr)
        if isinstance(o, (list, tuple)):
            return [norm(x) for x in o]
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "pattern"):            # a compiled regex
            return o.pattern
        return o
    raw = json.dumps(norm(obj), sort_keys=True, default=repr).encode()
    return hashlib.sha256(raw).hexdigest()


#: Module attributes holding a loaded table or head. A moved module that resolves its default path
#: to another file disables or swaps the table without an error (spec risk R5): the CONTENT is
#: what the gate compares.
LOADED = ("WEIGHTS", "_RANKHEAD_MODEL", "_RANKHEAD_V2_MODEL", "_QHEAD_MODEL", "_SCOPE_HEAD",
          "_QUERY_POLICY_HEADS", "_AUTH_HOSTS", "_HUB_HOSTS", "_PARASITE_HOSTS", "_PARASITE_RES",
          "_GAZETTEER", "_COUNTRY_MAP")
#: Module attributes naming the file a table was read from (the file's bytes are hashed too).
RESOLVED_PATHS = ("_RANKHEAD_PATH", "_RANKHEAD_V2_PATH", "_QHEAD_PATH", "_HUBFLOOR_TABLE",
                  "_AUTHORITY_TABLE", "_GAZ_PATH", "_COUNTRY_MAP_PATH")


def _loaded_state(retrieval) -> dict:
    """What the process loaded, read off the module the way the facade promises to keep it
    (a name that vanished is reported as `<missing>`, and that is a diff): per table whether it
    loaded, its size and the sha256 of its content; per resolved path the sha256 of the file."""
    miss = "<missing>"
    out: dict = {}
    for name in LOADED:
        v = getattr(retrieval, name, miss)
        if v is miss:
            out[name] = miss
        elif v is None:
            out[name] = {"loaded": False}
        else:
            out[name] = {"loaded": True, "size": len(v) if hasattr(v, "__len__") else None,
                         "sha256": _content_sha(v)}
    for name in RESOLVED_PATHS:
        v = getattr(retrieval, name, miss)
        if v is miss or not v:
            out[name] = v if v is miss else None
            continue
        f = pathlib.Path(v)
        out[name] = {"file": f.name, "sha256": _config_sha(f) if f.is_file() else None}
    try:
        out["weights_file"] = retrieval.weights_fingerprint().get("sha256")
    except Exception as e:  # noqa: BLE001 -- recorded, compared, never silently equal
        out["weights_file"] = f"<error {type(e).__name__}>"
    return out


def _install_cached_vectors(models, vecs: dict) -> list:
    """`models.embed_query` answers from the cache, exactly as the TEI path post-processes a
    vector (slice to MRL_DIM, renormalize; the full vector on the head side-channel). The same
    contract as `cytir.gates.authority.install_cached_embedder`; a miss raises (an error, not a
    silently different embedding)."""
    import numpy as np
    misses = []
    mrl = models.MRL_DIM

    def _emb(q):
        full = vecs.get(q)
        if full is None:
            misses.append(q)
            raise KeyError(f"no cached vector for {q!r}")
        full = np.asarray(full, dtype="float32")
        models._tls.full_qvec = full
        v = full[:mrl]
        nrm = float(np.linalg.norm(v)) or 1.0
        return (v / nrm).tolist(), None

    models.embed_query = _emb
    return misses


def _deep_capture(retrieval) -> dict:
    dbg = getattr(retrieval, "_LAST_SCORE_DEBUG", None)
    trace = getattr(retrieval, "_LAST_QUERY_PLAN_TRACE", None)
    v2 = getattr(retrieval, "_LAST_V2_DUMP", None)     # filled only with AW_RANKHEAD_V2_DUMP=1
    return {"score_debug": json.loads(json.dumps(list(dbg or []), default=str)),
            "trace": json.loads(json.dumps({k: v for k, v in dict(trace or {}).items()
                                            if k not in VOLATILE_KEYS}, default=str)),
            "rankhead_v2_dump": json.loads(json.dumps(list(v2 or []), default=str))}


def worker_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="serve_parity worker")
    ap.add_argument("--mode", choices=("served", "deep", "embed"), required=True)
    ap.add_argument("--vec-cache")
    ap.add_argument("--tree", required=True)
    a = ap.parse_args(argv)
    tree = pathlib.Path(a.tree).resolve()
    # the tree's OWN package, by name: an old tree has `aw`, a new one `cytir`, and the interpreter
    # may hold the other one installed (the head's editable install): `import <that>` would not be this tree
    root_pkg = importlib.import_module(package_name(tree))
    serve_pkg = importlib.import_module(serve_package(tree))
    for mod in (serve_pkg, root_pkg):
        if tree not in pathlib.Path(mod.__file__).resolve().parents:
            _emit({"fatal": f"{mod.__name__} imported from {mod.__file__}, not from {tree}"})
            return 4
    models = importlib.import_module(serve_package(tree) + ".models")
    if a.mode == "embed":
        _emit({"ready": True, "tei": models.TEI_URL})
        for line in sys.stdin:
            req = json.loads(line)
            try:
                models._embed_query_tei(req["q"])   # raises on failure: no silent fallback
                full = models.last_full_query_vec()
                _emit({"q": req["q"], "vec": [float(x) for x in full]})
            except Exception as e:  # noqa: BLE001 -- reported, never swallowed
                _emit({"q": req["q"], "error": f"{type(e).__name__}: {e}"})
        _emit({"tei_revision": models.tei_revision_status()})
        return 0
    vecs = json.loads(pathlib.Path(a.vec_cache).read_text())
    vecs = vecs.get("vectors", vecs)
    misses = _install_cached_vectors(models, vecs)
    srv = importlib.import_module(serve_package(tree) + ".app")
    retrieval = importlib.import_module(serve_package(tree) + ".retrieval")
    from fastapi.testclient import TestClient
    with TestClient(srv.api) as client:        # runs the lifespan, as the container does
        fp = {"effective_config": retrieval.effective_config(),
              "loaded": _loaded_state(retrieval),
              "retrieval_file": str(pathlib.Path(retrieval.__file__).resolve().relative_to(tree)),
              "server_source_sha256": _source_digest(serve_dir(tree)),
              # the server imports cytir.core (decompound, model_pin): in prod that is the IMAGE's
              # copy, here each tree's. Reported, so a slice that changes cytir/core is visible.
              "aw_core_sha256": _source_digest(tree / package_name(tree) / "core"),
              "hashseed": os.environ.get("PYTHONHASHSEED")}
        # Paths the server reports (weights file, head files) are inside the arm's own tree;
        # the trees differ by location, not by content, so the location is masked (the serving
        # package's own directory first: it sits at `<package>/serve`).
        fp = json.loads(json.dumps(fp, default=str).replace(str(serve_dir(tree)), "<serve>")
                        .replace(str(tree), "<tree>"))
        _emit({"ready": True, "fingerprint": fp})
        for line in sys.stdin:
            req = json.loads(line)
            before = len(misses)
            try:
                r = client.get("/search", params=req["params"])
                ctype = r.headers.get("content-type", "")
                out = {"status": r.status_code,
                       "body": r.json() if ctype.startswith("application/json") else r.text}
            except Exception as e:  # noqa: BLE001 -- an error row, never an empty
                out = {"status": None, "body": f"{type(e).__name__}: {e}"}
            if len(misses) > before:
                out["cache_miss"] = True
            if a.mode == "deep":
                out["deep"] = _deep_capture(retrieval)
            _emit(out)
    return 0


# --------------------------------------------------------------------------------------------- #
# the orchestrator side of an arm
# --------------------------------------------------------------------------------------------- #

class Arm:
    """One worker process. `ask(params)` returns its parsed answer; non-marker stdout (a print
    in the server) is ignored, stderr goes to `<log>`."""

    def __init__(self, name: str, tree: pathlib.Path, env: dict, mode: str, log: pathlib.Path,
                 python: str = sys.executable, vec_cache: str | None = None,
                 nice: bool = False, ready_timeout: float = 600.0):
        self.name, self.tree = name, tree
        cmd = [python, "-c", BOOT, str(tree), str(pathlib.Path(__file__).resolve()),
               "--mode", mode, "--tree", str(tree)]
        if vec_cache:
            cmd += ["--vec-cache", str(vec_cache)]
        if nice:
            cmd = ["nice", "-n", "19", "ionice", "-c3"] + cmd
        self._log = open(log, "a")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._log, env=env, text=True, bufsize=1,
                                     cwd=str(tree))
        self._lock = threading.Lock()
        hello = self._read(ready_timeout)
        if "fatal" in hello or not hello.get("ready"):
            raise ArmError(f"{name}: {hello.get('fatal') or hello}")
        self.fingerprint = hello.get("fingerprint", {})
        self.hello = hello

    def _read(self, timeout: float | None = None) -> dict:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise ArmError(f"{self.name}: worker exited (rc={self.proc.poll()}); "
                               f"see {self._log.name}")
            if line.startswith(MARK):
                return json.loads(line[len(MARK):])
            if deadline is not None and time.monotonic() > deadline:
                raise ArmError(f"{self.name}: not ready after {timeout:.0f} s")

    def ask(self, payload: dict) -> dict:
        with self._lock:
            self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            return self._read()

    def close(self) -> dict | None:
        tail = None
        try:
            self.proc.stdin.close()
            out = self.proc.stdout.read()
            for line in out.splitlines():
                if line.startswith(MARK):
                    tail = json.loads(line[len(MARK):])
            self.proc.wait(timeout=60)
        finally:
            self._log.close()
        return tail


class ArmError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------- #
# comparison (pure)
# --------------------------------------------------------------------------------------------- #

def json_key(body) -> str:
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _num_diff(a, b, tol: float, path: str, out: list, limit: int = 20) -> None:
    """Structural compare: dict keys and list lengths/orders exactly, numbers within `tol`,
    everything else equal. Appends up to `limit` human-readable differences."""
    if len(out) >= limit:
        return
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            out.append(f"{path}: {a!r} != {b!r}")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return
        if not (abs(a - b) <= tol):
            out.append(f"{path}: {a!r} != {b!r} (|d|={abs(a - b):.3g})")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{path}: keys {sorted(set(a) ^ set(b))[:6]} differ")
            return
        for k in sorted(a):
            _num_diff(a[k], b[k], tol, f"{path}.{k}", out, limit)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _num_diff(x, y, tol, f"{path}[{i}]", out, limit)
        return
    if a != b:
        out.append(f"{path}: {str(a)[:80]!r} != {str(b)[:80]!r}")


def max_abs_delta(a, b) -> float:
    m = 0.0
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) & set(b):
            m = max(m, max_abs_delta(a[k], b[k]))
    elif isinstance(a, list) and isinstance(b, list):
        for x, y in zip(a, b):
            m = max(m, max_abs_delta(x, y))
    elif (isinstance(a, (int, float)) and isinstance(b, (int, float))
          and not isinstance(a, bool) and not isinstance(b, bool)):
        if not (math.isnan(a) and math.isnan(b)):
            m = abs(a - b)
    return m


def served_error(ans: dict, expect: int = 200) -> str | None:
    if ans.get("cache_miss"):
        return "cache miss"
    if ans.get("status") != expect:
        return f"status {ans.get('status')}: {str(ans.get('body'))[:160]}"
    return None


def _diff_paths(a, b, tol: float, path: str, out: dict) -> None:
    """Every differing JSON path (list indexes folded to `[]`), key-set differences expanded one
    level (`boosts.country`). `out` maps path -> 'keys' (present on one side only) | 'value'."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) ^ set(b)):
            out.setdefault(f"{path}.{k}" if path else str(k), "keys")
        for k in sorted(set(a) & set(b)):
            _diff_paths(a[k], b[k], tol, f"{path}.{k}" if path else str(k), out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.setdefault(f"{path}[]", "value")
            return
        for x, y in zip(a, b):
            _diff_paths(x, y, tol, f"{path}[]", out)
        return
    probe: list = []
    _num_diff(a, b, tol, path, probe, limit=1)
    if probe:
        out.setdefault(path, "value")


def _body_ranked(a, b) -> tuple[bool, dict]:
    """(ranked list equal, envelope diff paths) of two served bodies: `results` (ids, order,
    scores, exactly) against everything else."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return (json_key(a) == json_key(b)), ({} if json_key(a) == json_key(b) else {"body": "value"})
    ranked = ("results" in a) == ("results" in b)
    if ranked and "results" in a:
        ranked = json_key(a["results"]) == json_key(b["results"])
    env: dict = {}
    _diff_paths({k: v for k, v in a.items() if k != "results"},
                {k: v for k, v in b.items() if k != "results"}, 0.0, "", env)
    return ranked, env


#: Keys of one served result that carry the ranking or identify the hit (task 2712; the real names
#: are the result row's in cytir/serve). A declared per-result envelope field may never be one of them.
#: Every key the result row carries TODAY (cytir/serve/respond.py, the `item` dict and the keys added
#: to it: country, entity-card fields, confidence). Only a NEW key can be declared additive.
RANKING_RESULT_KEYS = frozenset({
    "url", "anchor", "title", "snippet", "source_type", "lang", "trust_tier", "last_seen",
    "published", "score", "country", "confidence",
    "entity_kind", "register_id", "register_scheme", "qid", "canton", "reachable"})
_PER_RESULT = "results[]."
_FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_envelope_fields(fields: list[str]) -> str | None:
    """None when every declared field is acceptable; else the refusal message (task 2712). Beside the
    top-level paths, `results[].<field>` is accepted: one level, a plain field name, not a ranking key."""
    for f in fields:
        if f.startswith("deep."):
            return "--envelope-fields may not start with 'deep.' (the deep record is ranking evidence)"
        if f.startswith(_PER_RESULT):
            name = f[len(_PER_RESULT):]
            if not _FIELD_NAME.fullmatch(name):
                return (f"--envelope-fields {f!r}: a per-result path is exactly 'results[].<field>' "
                        "(one level, a plain field name; no '*', no nested path)")
            if name in RANKING_RESULT_KEYS:
                return (f"--envelope-fields {f!r}: {name!r} is a ranking-bearing result key "
                        f"({', '.join(sorted(RANKING_RESULT_KEYS))}) and cannot be declared")
        elif f.startswith("results"):
            return f"--envelope-fields {f!r}: the ranked list is declared only as 'results[].<field>'"
    return None


def _result_fields(a, b) -> tuple[bool, dict]:
    """(results equal once every per-result field that differs is stripped, {field: {base, head}} of
    those fields: how many results carry it). Only same-length lists of objects can be equal."""
    ra, rb = a.get("results"), b.get("results")
    if not (isinstance(ra, list) and isinstance(rb, list) and len(ra) == len(rb)
            and all(isinstance(x, dict) for x in ra + rb)):
        return False, {}
    d: set[str] = set()
    for x, y in zip(ra, rb):
        d |= {k for k in set(x) | set(y) if (k in x) != (k in y) or json_key(x.get(k)) != json_key(y.get(k))}
    d -= RANKING_RESULT_KEYS  # a ranking key is never stripped: its change leaves the lists unequal

    def strip(r):
        return {k: v for k, v in r.items() if k not in d}

    same = json_key([strip(x) for x in ra]) == json_key([strip(y) for y in rb])
    counts = {k: {"base": sum(k in x for x in ra), "head": sum(k in y for y in rb)} for k in sorted(d)}
    return same, counts


def _deep_ranked(a, b, tol: float) -> tuple[bool, dict]:
    """The deep record is ALL ranking evidence (score debug, v2 dump, and the trace's candidates and
    final ranking): ANY difference, including a key present in only one trace, makes it not
    ranked-equal (review 2680 F1). It has no envelope paths."""
    probe: list = []
    _num_diff(a, b, tol, "deep", probe, limit=1)
    return (not probe), {}


def classify(seq: dict, tol: float, expect: int = 200) -> dict:
    """One attempt of one request: {'state': EQUAL | DIFF | UNSTABLE | ERROR, ...}. `expect` is the
    status the request is MEANT to get (400 for an operator-only query); any other is an ERROR."""
    for k, ans in seq.items():
        err = served_error(ans, expect)
        if err:
            return {"state": "ERROR", "arm": k, "error": err}
    if json_key(seq["base"]["body"]) != json_key(seq["null"]["body"]):
        return {"state": "UNSTABLE", "why": "base served != base served again"}
    if "base_deep" in seq and "null_deep" in seq:
        drift: list[str] = []
        _num_diff(seq["base_deep"]["deep"], seq["null_deep"]["deep"], tol, "deep", drift, limit=1)
        if drift:
            # the deep record moved under the SAME code: the twin changed (a write shifts BM25
            # statistics in the 5th digit while the rounded served answer stays put). Task 752's
            # first slice-1 run counted two such shifts as diffs before this bracket existed.
            return {"state": "UNSTABLE", "why": f"base deep != base deep again ({drift[0]})"}
    for tree in ("base", "head"):
        dk = f"{tree}_deep"
        if dk in seq and json_key(seq[dk]["body"]) != json_key(seq[tree]["body"]):
            return {"state": "ERROR", "arm": dk,
                    "error": "the deep instrument changed the served answer"}
    # task 2679: the ranked list and the envelope are compared on their own, at both tiers, and
    # recorded for every request. The verdict below is the one it always was.
    rs, env = _body_ranked(seq["base"]["body"], seq["head"]["body"])
    ranked_equal = {"served": rs, "deep": None}
    envelope = dict(env)
    result_fields: dict = {}
    if not rs and isinstance(seq["base"]["body"], dict) and isinstance(seq["head"]["body"], dict):
        # task 2712: the ranked list with its per-result fields that differ stripped, apart.
        ranked_equal["served_stripped"], result_fields = _result_fields(seq["base"]["body"],
                                                                        seq["head"]["body"])
        if ranked_equal["served_stripped"]:
            envelope.update({f"{_PER_RESULT}{k}": "value" for k in result_fields})
    deep_pair = "base_deep" in seq and "head_deep" in seq
    if deep_pair:
        rd, denv = _deep_ranked(seq["base_deep"]["deep"], seq["head_deep"]["deep"], tol)
        ranked_equal["deep"] = rd
        envelope.update(denv)
    extra = {"ranked_equal": ranked_equal, "envelope_paths": sorted(envelope)}
    if result_fields:
        extra["result_field_counts"] = result_fields
    diffs: list[str] = []
    if json_key(seq["base"]["body"]) != json_key(seq["head"]["body"]):
        _num_diff(seq["base"]["body"], seq["head"]["body"], 0.0, "body", diffs)
        diffs = diffs or ["body: canonical JSON differs"]
        return {"state": "DIFF", "tier": "served", "diffs": diffs,
                "max_abs_delta": max_abs_delta(seq["base"]["body"], seq["head"]["body"]), **extra}
    if deep_pair:
        a, b = seq["base_deep"]["deep"], seq["head_deep"]["deep"]
        _num_diff(a, b, tol, "deep", diffs)
        if diffs:
            return {"state": "DIFF", "tier": "deep", "diffs": diffs, "max_abs_delta": max_abs_delta(a, b),
                    **extra}
        return {"state": "EQUAL", "deep_max_abs_delta": max_abs_delta(a, b), **extra}
    return {"state": "EQUAL", **extra}


def verdict(fp_diffs: list, results: list[dict]) -> str:
    states = [r["final"]["state"] for r in results]
    if "ERROR" in states:
        return "ERROR"
    if fp_diffs or "DIFF" in states:
        return "FAIL"
    if "UNSTABLE" in states:
        return "INCONCLUSIVE"
    return "PASS"


# --------------------------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------------------------- #

def _twin_points(qdrant_url: str, collection: str) -> int | None:
    from cytir.core import internal_http as IH  # lazy: a parity worker may run from a tree without cytir.sources
    try:
        with IH.urlopen(f"{qdrant_url.rstrip('/')}/collections/{collection}", timeout=30) as r:
            return json.loads(r.read())["result"]["points_count"]
    except Exception:  # noqa: BLE001 -- recorded as unknown, never as zero
        return None


def _fp_diffs(a: dict, b: dict) -> list[str]:
    out: list[str] = []
    for key in ("effective_config", "loaded"):
        _num_diff(a.get(key), b.get(key), 0.0, key, out)
    return out


def run_parity(requests: list[dict], *, base_tree: pathlib.Path, head_tree: pathlib.Path,
               collection: str, qdrant_url: str, vec_cache: pathlib.Path,
               out_dir: pathlib.Path, shards: int = 1, deep: bool = True,
               score_tol: float = 1e-9, seed_probe: bool = False, python: str = sys.executable,
               nice: bool = False, guard=None, declared_env: dict | None = None,
               arm_extra: dict | None = None, base_sha: str | None = None,
               head_sha: str | None = None, say=print) -> dict:
    """Run the gate. `guard(qdrant_url, collection)` must raise when the target is not the twin
    (the CLI passes `cytir.retune.guard.refuse_non_twin`; tests pass a stub)."""
    if guard is None:
        raise SystemExit("run_parity needs a twin guard")
    guard(qdrant_url, collection)
    out_dir.mkdir(parents=True, exist_ok=True)
    declared = serving_env(base_tree) if declared_env is None else declared_env
    head_declared = serving_env(head_tree) if declared_env is None else declared_env
    # The gate compares CODE under ONE env. When the envs differ (outside the timing knobs) the
    # PR is a config change: with identical serving inputs there is no code to compare and the
    # one env that matters is the new one, so every arm runs the HEAD env (task 2816).
    base_pkg, head_pkg = package_name(base_tree), package_name(head_tree)
    refuse_foreign_image_prefix(declared, base_pkg)
    refuse_foreign_image_prefix(head_declared, head_pkg)
    changed = env_changes(declared, head_declared, base_pkg, head_pkg)
    config_change = None
    declared_pkg = base_pkg                 # the package spelling of the tree `declared` came from
    if changed:
        digests = {"base": serving_inputs_digest(base_tree), "head": serving_inputs_digest(head_tree)}
        keys = ", ".join(changed)
        if digests["base"] != digests["head"]:
            moved = moved_inputs(base_tree, head_tree)
            blocked = [p for p in moved if _value_change_blocked(p, head_pkg)]
            if blocked:
                raise SystemExit(
                    f"the two trees declare different serving environments ({keys}) AND move "
                    f"serving code, data or operator inputs ({', '.join(blocked[:8])}): this gate "
                    "compares code under ONE env. Split the PR: the code move and the config "
                    "change are two PRs.")
            # task 2832: not refused at once. A serving-profile VALUE lives under cytir/
            # (cytir/core/serving_config.py) and the compose overlay is generated from it, so every
            # value change differs in both. The head CODE is served under the BASE env (one env,
            # one variable: the code); only an EQUAL result passes (checked after the run).
            config_change = {"env_source": "base", "head_code_under_base_env": True,
                             "changed": changed,
                             "moved_inputs": moved,
                             "serving_inputs_sha256": {"base": digests["base"],
                                                       "head": digests["head"]}}
            say(f"config change ({keys}) AND different serving inputs: serving the head code "
                "under the BASE env; PASS needs every request EQUAL")
        else:
            say(f"config-only change ({keys}): identical serving inputs, every arm runs the head env")
            config_change = {"env_source": "head", "changed": changed,
                             "serving_inputs_sha256": digests["head"]}
            declared, declared_pkg = head_declared, head_pkg
    # A lever prod does not run (a dark path a slice moves) is exercised by arming it in EVERY
    # arm: still one env, still one variable (the code).
    declared = {**declared, **(arm_extra or {})}
    plans = {"base": (base_tree, False, SEED), "head": (head_tree, False, SEED),
             "null": (base_tree, False, SEED)}
    if deep:
        plans.update({"base_deep": (base_tree, True, SEED), "head_deep": (head_tree, True, SEED),
                      "null_deep": (base_tree, True, SEED)})
    if seed_probe:
        plans["seed_probe"] = (base_tree, False, PROBE_SEED)
    data_files = {}
    envs = {}
    for name, (tree, is_deep, seed) in plans.items():
        envs[name], files = arm_env(declared, tree, collection=collection,
                                    qdrant_url=qdrant_url, deep=is_deep, seed=seed,
                                    declared_pkg=declared_pkg)
        data_files[name] = files
    # A served data file that differs is RECORDED (a ranking release needs a report to show), and
    # it is a configuration difference: the verdict is FAIL and `check` refuses it.
    data_file_diffs = {k: {"base": data_files["base"].get(k), "head": data_files["head"].get(k)}
                       for k in sorted(set(data_files["base"]) | set(data_files["head"]))
                       if data_files["base"].get(k) != data_files["head"].get(k)}
    points_before = _twin_points(qdrant_url, collection)
    started = time.time()
    shard_reqs = [requests[i::shards] for i in range(shards)]
    # Each base arm is asked again after its head arm: the repeat brackets the head call in time,
    # so a twin that moves during the request shows up as base != base again, never as a diff.
    order = ["base", "base_deep", "head", "head_deep", "null", "null_deep"]
    order = [o for o in order if o in plans]

    def run_shard(si: int) -> tuple[list[dict], dict]:
        arms = {}
        try:
            for name in plans:
                tree, is_deep, _seed = plans[name]
                arms[name] = Arm(f"{name}#{si}", tree, envs[name],
                                 "deep" if is_deep else "served",
                                 out_dir / f"worker-{name}-{si}.log", python=python,
                                 vec_cache=str(vec_cache), nice=nice)
            fps = {n: a.fingerprint for n, a in arms.items()}
            res = []
            for i, req in enumerate(shard_reqs[si]):
                attempts = []
                final = None
                for _attempt in range(1 + RETRIES):
                    seq = {name: arms[name].ask({"params": req["params"]}) for name in order}
                    c = classify(seq, score_tol, req.get("expect_status", 200))
                    attempts.append(c)
                    if c["state"] in ("EQUAL", "ERROR"):
                        final = c
                        break
                    if c["state"] == "DIFF":
                        final = c               # stable base, different head: a real diff
                        break
                final = final or attempts[-1]
                row = {"id": req["id"], "params": req["params"], "final": final,
                       "attempts": len(attempts),
                       "first_attempt_unstable": attempts[0]["state"] == "UNSTABLE"}
                if seed_probe and final["state"] == "EQUAL":
                    sp = arms["seed_probe"].ask({"params": req["params"]})
                    row["seed_sensitive"] = (served_error(sp, req.get("expect_status", 200)) is None and json_key(sp["body"])
                                             != json_key(seq["base"]["body"]))
                if final["state"] != "EQUAL":
                    row["base_urls"] = [x.get("url") for x in
                                        (seq["base"].get("body") or {}).get("results", [])] \
                        if isinstance(seq["base"].get("body"), dict) else None
                res.append(row)
                if (i + 1) % 100 == 0:
                    say(f"  shard {si}: {i + 1}/{len(shard_reqs[si])}")
            return res, fps
        finally:
            for a in arms.values():
                try:
                    a.close()
                except Exception:  # noqa: BLE001
                    pass

    results: list[dict] = []
    fingerprints: dict = {}
    shard_fp_diffs: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=shards) as ex:
        for si, (res, fps) in enumerate(ex.map(run_shard, range(shards))):
            results += res
            if not fingerprints:
                fingerprints = fps
                continue
            for arm in fps:             # every shard's arm must load what shard 0's loaded
                for d in _fp_diffs(fingerprints[arm], fps[arm]):
                    shard_fp_diffs.append(f"shard {si} {arm}: {d}")
    elapsed = time.time() - started
    points_after = _twin_points(qdrant_url, collection)
    fp_diffs = (_fp_diffs(fingerprints["base"], fingerprints["head"]) + shard_fp_diffs
                + [f"served data file {k} differs between the trees" for k in data_file_diffs])
    # S1 (task 757): the per-request repeat already turns a write into UNSTABLE; this flag makes a
    # PASS on a moving (or unreadable) twin visible without reading the two numbers.
    twin_moved = (points_before is None or points_after is None or points_before != points_after)
    same_code = (fingerprints["base"].get("server_source_sha256")
                 == fingerprints["head"].get("server_source_sha256"))
    counts = {s: sum(1 for r in results if r["final"]["state"] == s)
              for s in ("EQUAL", "DIFF", "UNSTABLE", "ERROR")}
    v = verdict(fp_diffs, results)
    if config_change and config_change["env_source"] == "base":
        pairs = "; ".join(f"{k}: {c['base']!r} -> {c['head']!r}" for k, c in changed.items())
        config_change["receipt"] = (
            f"envs differ ({pairs}); serving inputs differ ({len(config_change['moved_inputs'])} moved: "
            f"{', '.join(config_change['moved_inputs'][:8])}); head code under base env = "
            f"{'EQUAL' if v == 'PASS' else 'NOT EQUAL'} on {len(results)} queries")
    report = {
        "schema": SCHEMA,
        "verdict": v,
        "created_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_s": round(elapsed, 1),
        "collection": collection,
        "twin_points": {"before": points_before, "after": points_after},
        "twin_moved": twin_moved,
        "release": {"base_sha": base_sha, "head_sha": head_sha},
        "trees": {"base": str(base_tree), "head": str(head_tree), "same_server_code": same_code,
                  "same_aw_core": (fingerprints["base"].get("aw_core_sha256")
                                   == fingerprints["head"].get("aw_core_sha256"))},
        "requests": len(results),
        "counts": counts,
        "null_pair": {"first_attempt_unstable": sum(1 for r in results
                                                    if r["first_attempt_unstable"]),
                      "unstable_after_retries": counts["UNSTABLE"]},
        "arm_extra_env": dict(arm_extra or {}),
        "config_change": config_change,
        "deep": deep, "score_tol": score_tol,
        "deep_max_abs_delta": max((r["final"].get("deep_max_abs_delta", 0.0) for r in results),
                                  default=0.0),
        "diff_max_abs_delta": max((r["final"].get("max_abs_delta", 0.0) for r in results
                                   if r["final"]["state"] == "DIFF"), default=0.0),
        "seed_probe": ({"seed": PROBE_SEED,
                        "sensitive": sum(1 for r in results if r.get("seed_sensitive"))}
                       if seed_probe else None),
        "fingerprint_diffs": fp_diffs,
        "fingerprints": {"base": fingerprints["base"], "head": fingerprints["head"]},
        "data_files_sha256": data_files["base"],
        "data_file_diffs": data_file_diffs,
        "vec_cache_sha256": _sha256(vec_cache),
        "sets": sorted({r["id"].split(":")[0].split(".")[0] for r in results}),
        "not_equal": [r for r in results if r["final"]["state"] != "EQUAL"],
        "seed_sensitive_ids": [r["id"] for r in results if r.get("seed_sensitive")],
    }
    (out_dir / "parity.json").write_text(json.dumps(report, indent=1, ensure_ascii=False,
                                                    default=str) + "\n")
    if config_change and config_change["env_source"] == "base" and v != "PASS":
        raise SystemExit(
            f"the two trees declare different serving environments ({', '.join(changed)}) AND hold "
            "different serving inputs (cytir/, data/, deploy/operator/), and the head code under the "
            f"BASE env is not equal to the base ({counts['DIFF']} DIFF, {counts['UNSTABLE']} UNSTABLE, "
            f"{counts['ERROR']} ERROR of {len(results)} queries; verdict {v}; report "
            f"{out_dir / 'parity.json'}): this gate compares code under ONE env. Split the PR: the "
            "code move and the config change are two PRs.")
    return report


def _digest_of(report: dict, arm: str):
    return ((report.get("fingerprints") or {}).get(arm) or {}).get("server_source_sha256")


def _ref_at_release(ref: str, sha: str, repo_dir: pathlib.Path) -> str | None:
    """Which of plan.md / decisions.md, read AT `sha` (not in the checkout), names `ref`."""
    pat = re.compile(r"(?<![\w.-])" + re.escape(ref) + r"(?![\w-])")
    for name in ("plan.md", "decisions.md"):
        r = subprocess.run(["git", "-C", str(repo_dir), "show", f"{sha}:./{name}"],
                           capture_output=True, text=True)
        if r.returncode == 0 and pat.search(r.stdout):
            return name
    return None


def _null_report_problems(r2, *, head_sha: str, head_digest: str, collection) -> list[str]:
    """R2 of a ranking change: the release compared with ITSELF (head vs head), run with the seed
    probe, a PASS with no seed-sensitive request. It shows that the release is deterministic, so
    the DIFF count of the main report is the change and not noise."""
    if not isinstance(r2, dict) or r2.get("schema") != SCHEMA:
        return [f"the null report is not a {SCHEMA} report"]
    bad: list[str] = []
    if r2.get("verdict") != "PASS":
        bad.append(f"the null report verdict is {r2.get('verdict')!r}, not 'PASS'")
    rel = r2.get("release") or {}
    if rel.get("base_sha") != head_sha or rel.get("head_sha") != head_sha:
        bad.append(f"the null report is not head-vs-head for the release (pair "
                   f"{rel.get('base_sha')!r} -> {rel.get('head_sha')!r})")
    if not (_digest_of(r2, "base") == _digest_of(r2, "head") == head_digest):
        bad.append("the null report's trees are not the release's serving package")
    if (r2.get("trees") or {}).get("same_server_code") is not True:
        bad.append("the null report's two arms did not run the same server code")
    if not r2.get("requests") or (r2.get("counts") or {}).get("EQUAL") != r2.get("requests"):
        bad.append("the null report has a request that is not EQUAL")
    sprobe = r2.get("seed_probe")
    if not isinstance(sprobe, dict) or sprobe.get("sensitive") != 0:
        bad.append(f"the null report needs a seed probe with 0 sensitive requests (has {sprobe!r})")
    if collection is not None and r2.get("collection") != collection:
        bad.append(f"the null report ran on {r2.get('collection')!r}, the report on {collection!r}")
    return bad


def check_report(report: dict, *, base_sha: str, head_sha: str, head_server_dir: pathlib.Path,
                 base_server_dir: pathlib.Path, ranking: dict | None = None,
                 envelope: dict | None = None) -> list[str]:
    """Why `report` does not license releasing `head_sha` over a box serving `base_sha`; [] = it does.
    The ONE function behind `check` (and so behind the two release scripts' refusal).

    `base_server_dir` is the serving package the box serves NOW and `head_server_dir` the
    release's: the report's two arms must have run on exactly those two trees (the sha strings are
    a record, the digests are the binding). `ranking` = {ref, max_diff, null_report, repo_dir}
    selects the declared ranking-change mode, `envelope` = {ref, fields, repo_dir} the declared
    envelope-change mode (module docstring)."""
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        return [f"not a {SCHEMA} report"]
    bad: list[str] = []
    rel = report.get("release") or {}
    if rel.get("base_sha") != base_sha:
        bad.append(f"report base sha {rel.get('base_sha')!r} is not the sha the box serves {base_sha!r}")
    if rel.get("head_sha") != head_sha:
        bad.append(f"report head sha {rel.get('head_sha')!r} is not the release {head_sha!r}")
    head_digest = _source_digest(head_server_dir)
    base_digest = _source_digest(base_server_dir)
    if _digest_of(report, "head") != head_digest:
        bad.append(f"the report's head tree is not the release's serving package "
                   f"(report {_digest_of(report, 'head')}, release {head_digest})")
    if _digest_of(report, "base") != base_digest:
        bad.append(f"the report's base tree is not the serving package the box serves now "
                   f"(report {_digest_of(report, 'base')}, box {base_digest})")
    if (report.get("trees") or {}).get("same_server_code") is not False \
            or _digest_of(report, "base") == _digest_of(report, "head"):
        bad.append("the report compares a tree with itself (same server code): it proves nothing "
                   "about a serving-code release")
    if (report.get("config_change") or {}).get("env_source") == "base":
        bad.append("the report served the head code under the BASE env (a config change plus a "
                   "serving-input change, task 2832): it never licenses a serving-code release")
    counts = report.get("counts") or {}
    requests_n = report.get("requests") or 0
    null_dirty = (report.get("null_pair") or {}).get("unstable_after_retries")
    if ranking is None and envelope is None:
        if report.get("verdict") != "PASS":
            bad.append(f"verdict is {report.get('verdict')!r}, not 'PASS'")
        if not requests_n or counts.get("EQUAL") != requests_n:
            bad.append(f"not every request is EQUAL (requests {report.get('requests')}, counts {counts})")
        if null_dirty:
            bad.append("the null pair is not clean")
        if report.get("fingerprint_diffs"):
            bad.append("the two arms loaded different configuration")
        return bad
    if envelope is not None:
        return bad + _envelope_problems(report, envelope, head_sha)
    # --- declared ranking change
    if report.get("verdict") not in ("PASS", "FAIL"):
        bad.append(f"verdict is {report.get('verdict')!r}: a ranking change needs PASS or FAIL")
    if counts.get("ERROR"):
        bad.append(f"{counts.get('ERROR')} request(s) ERROR")
    if counts.get("UNSTABLE"):
        bad.append(f"{counts.get('UNSTABLE')} request(s) UNSTABLE")
    if not requests_n or sum(counts.get(k, 0) for k in ("EQUAL", "DIFF")) != requests_n:
        bad.append(f"requests are not all EQUAL or DIFF (requests {report.get('requests')}, counts {counts})")
    if report.get("fingerprint_diffs") or report.get("data_file_diffs"):
        bad.append("the two arms loaded different configuration (a changed served data file is a "
                   "configuration difference: this mode covers code only)")
    if null_dirty:
        bad.append("the null pair is not clean")
    if (counts.get("DIFF") or 0) > ranking["max_diff"]:
        bad.append(f"{counts.get('DIFF')} DIFF requests, more than --max-diff {ranking['max_diff']}")
    bad += _null_report_problems(ranking.get("null_report"), head_sha=head_sha,
                                 head_digest=head_digest, collection=report.get("collection"))
    where = _ref_at_release(ranking["ref"], head_sha, ranking["repo_dir"]) if ranking["ref"] else None
    if where is None:
        bad.append(f"ruling {ranking['ref']!r} is in neither plan.md nor decisions.md AT the release "
                   f"{head_sha[:12]}")
    return bad


def _envelope_problems(report: dict, envelope: dict, head_sha: str) -> list[str]:
    """The declared envelope-change conditions (module docstring); [] = met. The caller already
    holds the pair/digest/same-code reasons."""
    bad: list[str] = []
    counts = report.get("counts") or {}
    requests_n = report.get("requests") or 0
    if report.get("verdict") not in ("PASS", "FAIL"):
        bad.append(f"verdict is {report.get('verdict')!r}: an envelope change needs PASS or FAIL")
    if counts.get("ERROR"):
        bad.append(f"{counts.get('ERROR')} request(s) ERROR")
    if counts.get("UNSTABLE"):
        bad.append(f"{counts.get('UNSTABLE')} request(s) UNSTABLE")
    if not requests_n or sum(counts.get(k, 0) for k in ("EQUAL", "DIFF")) != requests_n:
        bad.append(f"requests are not all EQUAL or DIFF (requests {report.get('requests')}, counts {counts})")
    if report.get("fingerprint_diffs") or report.get("data_file_diffs"):
        bad.append("the two arms loaded different configuration")
    if (report.get("null_pair") or {}).get("unstable_after_retries"):
        bad.append("the null pair is not clean")
    declared = set(envelope["fields"])
    if any(f.startswith("deep.") for f in declared):
        bad.append("a declared field starting 'deep.' is refused: the deep record is ranking evidence")
    rows = [r for r in report.get("not_equal") or [] if (r.get("final") or {}).get("state") == "DIFF"]
    if len(rows) != counts.get("DIFF", 0):
        bad.append(f"the report lists {len(rows)} DIFF rows for a DIFF count of {counts.get('DIFF')}")
    for r in rows:
        f = r.get("final") or {}
        re_ = f.get("ranked_equal")
        if not isinstance(re_, dict) or "envelope_paths" not in f:
            bad.append("the report predates the ranked_equal schema (task 2679): re-run the pair "
                       "with the amended harness")
            break
        if re_.get("served") is not True and re_.get("served_stripped") is not True:
            hint = "" if "served_stripped" in re_ else (
                " (report predates per-result envelope support: re-run `run` with this harness)")
            bad.append(f"{r.get('id')}: the served ranked list differs{hint}")
        for fld, c in sorted((f.get("result_field_counts") or {}).items()):
            if f"{_PER_RESULT}{fld}" in declared and c.get("head", 0) < c.get("base", 0):
                bad.append(f"{r.get('id')}: declared per-result field {fld!r} is additive but "
                           f"{c.get('base')} base result(s) carry it and only {c.get('head')} head")
        if re_.get("deep") is not True:
            bad.append(f"{r.get('id')}: the deep ranked record differs or was not compared "
                       f"({re_.get('deep')!r})")
        extra = sorted(set(f.get("envelope_paths") or []) - declared)
        if extra:
            bad.append(f"{r.get('id')}: undeclared envelope path(s) {extra}")
    where = _ref_at_release(envelope["ref"], head_sha, envelope["repo_dir"]) if envelope["ref"] else None
    if where is None:
        bad.append(f"ruling {envelope['ref']!r} is in neither plan.md nor decisions.md AT the release "
                   f"{head_sha[:12]}")
    return bad


def _envelope_table(report: dict) -> str:
    lines = ["id | ranked_equal served/deep | envelope_paths"]
    for r in report.get("not_equal") or []:
        f = r.get("final") or {}
        if f.get("state") != "DIFF":
            continue
        re_ = f.get("ranked_equal") or {}
        lines.append(f"{r.get('id')} | {re_.get('served')}/{re_.get('deep')} | "
                     f"{', '.join(f.get('envelope_paths') or [])}")
    return "\n".join(lines)


def _load(path, what):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError) as e:
        print(f"serve-parity check: cannot read {what} {path}: {e}", file=sys.stderr)
        return None


def check_main(a) -> int:
    report = _load(a.report, "report")
    if report is None:
        return 2
    ranking = None
    envelope = None
    if a.envelope_change is not None or a.envelope_fields is not None:
        if a.ranking_change is not None or a.max_diff is not None or a.null_report:
            print("serve-parity check: --envelope-change and --ranking-change are two modes; "
                  "declare one", file=sys.stderr)
            return 2
        fields = [x.strip() for x in (a.envelope_fields or "").split(",") if x.strip()]
        if not (a.envelope_change and a.envelope_change.strip() and fields):
            print("serve-parity check: --envelope-change REF needs --envelope-fields a.b,c.d",
                  file=sys.stderr)
            return 2
        refusal = parse_envelope_fields(fields)
        if refusal:
            print(f"serve-parity check: {refusal}", file=sys.stderr)
            return 2
        envelope = {"ref": a.envelope_change.strip(), "fields": fields,
                    "repo_dir": pathlib.Path(a.repo_dir)}
    if a.ranking_change is not None or a.max_diff is not None or a.null_report:
        if not (a.ranking_change and a.max_diff is not None and a.null_report and a.max_diff >= 0):
            print("serve-parity check: --ranking-change REF needs --max-diff N (>= 0) and "
                  "--null-report R2, all three", file=sys.stderr)
            return 2
        r2 = _load(a.null_report, "null report")
        if r2 is None:
            return 2
        ranking = {"ref": a.ranking_change.strip(), "max_diff": a.max_diff, "null_report": r2,
                   "repo_dir": pathlib.Path(a.repo_dir)}
    bad = check_report(report, base_sha=a.base_sha, head_sha=a.head_sha,
                       head_server_dir=pathlib.Path(a.head_server_dir),
                       base_server_dir=pathlib.Path(a.base_server_dir), ranking=ranking, envelope=envelope)
    if envelope is not None:
        print(_envelope_table(report))
    if bad:
        print(f"serve-parity check: NOT satisfied for {a.base_sha[:12]} -> {a.head_sha[:12]}",
              file=sys.stderr)
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        return 1
    if envelope is not None:
        print(f"serve-parity check: ENVELOPE CHANGE {envelope['ref']} accepted for "
              f"{a.base_sha[:12]} -> {a.head_sha[:12]}: {report['counts']['DIFF']} DIFF of "
              f"{report['requests']} requests, every ranked list equal at both tiers, paths within "
              f"{','.join(envelope['fields'])}; ruling found in "
              f"{_ref_at_release(envelope['ref'], a.head_sha, envelope['repo_dir'])} at the release")
        return 0
    if ranking is None:
        print(f"serve-parity check: PASS report covers {a.base_sha[:12]} -> {a.head_sha[:12]} "
              f"({report['requests']} requests, collection {report.get('collection')})")
        return 0
    diffs = [r["id"] for r in report.get("not_equal") or []]
    print(f"serve-parity check: RANKING CHANGE {ranking['ref']} accepted for "
          f"{a.base_sha[:12]} -> {a.head_sha[:12]}: {report['counts']['DIFF']} DIFF of {report['requests']} "
          f"requests (--max-diff {ranking['max_diff']}), ruling found in "
          f"{_ref_at_release(ranking['ref'], a.head_sha, ranking['repo_dir'])} at the release; "
          f"null report PASS, 0 seed-sensitive")
    print("serve-parity check: differing requests (for the release doc): " + ", ".join(diffs))
    return 0


def summary(report: dict) -> str:
    c = report["counts"]
    lines = [f"serve parity: {report['verdict']}  ({report['requests']} requests, "
             f"{report['elapsed_s']} s, collection {report['collection']}, points "
             f"{report['twin_points']['before']} -> {report['twin_points']['after']}"
             f"{', TWIN MOVED during the run' if report['twin_moved'] else ''})",
             f"  equal {c['EQUAL']}  diff {c['DIFF']}  unstable {c['UNSTABLE']}  "
             f"error {c['ERROR']}",
             f"  null pair: first-attempt unstable {report['null_pair']['first_attempt_unstable']}"
             f", unstable after retries {report['null_pair']['unstable_after_retries']}",
             f"  same server code in both trees: {report['trees']['same_server_code']}",
             f"  same cytir/core in both trees: {report['trees']['same_aw_core']}",
             f"  deep tier: {report['deep']} (max |d| equal rows {report['deep_max_abs_delta']:.3g}, "
             f"diff rows {report['diff_max_abs_delta']:.3g}; tol {report['score_tol']}, resolution "
             f"~1e-6: the server rounds debug scores)"]
    if (report.get("config_change") or {}).get("env_source") == "base":
        lines.append(f"  CONFIG + INPUT change: {report['config_change']['receipt']}")
    elif report.get("config_change"):
        lines.append(f"  CONFIG-ONLY change: every arm ran the head env; changed "
                     f"{', '.join(report['config_change']['changed'])}")
    if report.get("seed_probe"):
        lines.append(f"  seed probe (PYTHONHASHSEED {report['seed_probe']['seed']} vs {SEED}): "
                     f"{report['seed_probe']['sensitive']} requests change")
    for d in report["fingerprint_diffs"][:10]:
        lines.append(f"  CONFIG DIFF {d}")
    for r in report["not_equal"][:15]:
        lines.append(f"  {r['final']['state']:9s} {r['id']}  "
                     f"{(r['final'].get('diffs') or [r['final'].get('error') or r['final'].get('why')])[0]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- #
# prepare: one vector per query, from the pinned seed cache or the embedder (once)
# --------------------------------------------------------------------------------------------- #

def prepare_cache(requests: list[dict], *, base_tree: pathlib.Path, tei_url: str,
                  seed_cache: pathlib.Path | None, out: pathlib.Path,
                  python: str = sys.executable, say=print) -> dict:
    from urllib.parse import urlparse
    u = urlparse(tei_url or "")
    if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password:
        raise SystemExit(f"--tei-url {tei_url!r} is not an http(s) URL with a host (and no "
                         f"credentials): the cache records it, so it must be a plain endpoint")
    vecs: dict = {}
    seed_sha = None
    if seed_cache:
        raw = json.loads(seed_cache.read_text())
        vecs = dict(raw.get("vectors", raw))
        seed_sha = _sha256(seed_cache)
    # a request meant to be refused (400) is never embedded; an operator request embeds its cleaned text
    qs = sorted({r.get("embed_q") or r["params"]["q"] for r in requests
                 if r.get("expect_status", 200) == 200})
    missing = [q for q in qs if q not in vecs]
    say(f"{len(qs)} distinct queries: {len(qs) - len(missing)} from the seed cache, "
        f"{len(missing)} to embed")
    tei_status = None
    if missing:
        env = {k: v for k, v in os.environ.items() if not _is_knob(k)}
        env.update({"AW_TEI_URL": tei_url, "AW_EMBEDDER": "arctic", "PYTHONHASHSEED": SEED})
        arm = Arm("embed", base_tree, env, "embed", out.with_suffix(".embed.log"), python=python)
        try:
            errors = []
            for i, q in enumerate(missing):
                a = arm.ask({"q": q})
                if "error" in a:
                    errors.append((q, a["error"]))
                else:
                    vecs[q] = a["vec"]
                if (i + 1) % 200 == 0:
                    say(f"  embedded {i + 1}/{len(missing)}")
        finally:
            tail = arm.close()
        tei_status = (tail or {}).get("tei_revision")
        if errors:
            raise SystemExit(f"{len(errors)} queries failed to embed (first: {errors[0]}); "
                             f"no cache written")
    keep = {q: vecs[q] for q in qs}
    doc = {"schema": CACHE_SCHEMA, "seed_cache_sha256": seed_sha, "tei_url": tei_url,
           "tei_revision": tei_status, "n": len(keep), "embedded": len(missing),
           "vectors": keep}
    out.write_text(json.dumps(doc))
    return {k: v for k, v in doc.items() if k != "vectors"}


# --------------------------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------------------------- #

def _kv(items: list[str]) -> dict:
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not _is_knob(k):
            raise SystemExit(f"--arm-env wants CYTIR_NAME=VALUE (or AW_NAME=VALUE), got {it!r}")
        out[k] = v
    return out


def _add_sets(ap):
    ap.add_argument("--sets", default="chooser,heldout,release,probe,operators",
                    help="comma list of chooser, heldout, release, probe, fixture, operators")
    ap.add_argument("--chooser", default=str(APP / "data" / "eval_tasks.v2.yaml"))
    ap.add_argument("--release", default=",".join(
        str(APP / "data" / f) for f in ("eval_release_new.yaml", "eval_release_control.yaml")))
    ap.add_argument("--fixture", default=str(APP / "data" / "gate-slices" / "ci-fixture.v1.json"),
                    help="the gold slice of --sets fixture (the index-serve-gate CI job's own)")
    ap.add_argument("--probe-runs", help="the probe's runs/ folder (one subfolder per day)")
    ap.add_argument("--probe-days", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="first N requests only (a smoke run)")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["worker"]:
        return worker_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="build the one vector cache every arm reads")
    p.add_argument("--base-tree", required=True)
    p.add_argument("--tei-url", required=True)
    p.add_argument("--seed-cache", help="a pinned query-vector cache to reuse (q -> 1024 floats)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--python", default=sys.executable)
    _add_sets(p)
    r = sub.add_parser("run", help="compare two trees on the twin")
    r.add_argument("--base-tree", required=True)
    r.add_argument("--head-tree", required=True)
    r.add_argument("--collection", required=True)
    r.add_argument("--qdrant-url", required=True)
    r.add_argument("--vec-cache", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--shards", type=int, default=4)
    r.add_argument("--no-deep", action="store_true")
    r.add_argument("--score-tol", type=float, default=1e-9)
    r.add_argument("--seed-probe", action="store_true")
    r.add_argument("--python", default=sys.executable,
                   help="interpreter of the arms (the serving venv)")
    r.add_argument("--nice", action="store_true", help="arms at nice 19 / ionice idle")
    r.add_argument("--arm-env", action="append", default=[], metavar="KEY=VALUE",
                   help="an CYTIR_* value set in EVERY arm, to exercise a lever prod does not run")
    r.add_argument("--base-sha", help="commit the base tree was archived from (recorded; `check` needs it)")
    r.add_argument("--head-sha", help="commit the head tree was archived from (recorded; `check` needs it)")
    _add_sets(r)
    c = sub.add_parser("check", help="exit 0 only if a parity.json is a PASS for (base sha, head sha)")
    c.add_argument("--report", required=True)
    c.add_argument("--base-sha", required=True)
    c.add_argument("--head-sha", required=True)
    c.add_argument("--head-server-dir", required=True,
                   help="the release's serving package (its source digest must equal the report's "
                        "head arm)")
    c.add_argument("--base-server-dir", required=True,
                   help="the serving package the box serves NOW (its source digest must equal the "
                        "report's base arm)")
    c.add_argument("--ranking-change", metavar="REF",
                   help="declare a ranking change: accept a FAIL report under the conditions of the "
                        "module docstring; REF must be in plan.md/decisions.md at --head-sha")
    c.add_argument("--max-diff", type=int, help="most DIFF requests a ranking change may show")
    c.add_argument("--null-report", help="R2: a head-vs-head PASS parity.json made with --seed-probe")
    c.add_argument("--envelope-change", metavar="REF",
                   help="declare an envelope-only change (task 2679): accept a FAIL report whose "
                        "DIFF requests are ranked-equal at both tiers; REF in plan.md/decisions.md "
                        "at --head-sha")
    c.add_argument("--envelope-fields", help="comma list of the envelope paths the release may change")
    c.add_argument("--repo-dir", default=".", help="a directory of the repo that holds plan.md "
                                                   "(git show <head-sha>:./plan.md); default cwd")
    d = sub.add_parser("digest", help="print the source digest of a serving-package directory")
    d.add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "digest":
        print(_source_digest(pathlib.Path(a.dir)))
        return 0
    if a.cmd == "check":
        return check_main(a)
    from cytir.retune import guard
    try:
        return _main(a, guard)
    except ArmError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT["ERROR"]
    except SystemExit as e:              # every refusal above, the twin guard's included
        if isinstance(e.code, int) and not isinstance(e, guard.Refused):
            raise
        print(e.args[0] if e.args else e.code, file=sys.stderr)
        return EXIT["REFUSED"]


def _main(a, guard) -> int:
    out_dir = pathlib.Path(a.out_dir).resolve()
    guard.refuse_serving_path(str(out_dir), "--out-dir")
    reqs = collect(a)
    if a.cmd == "prepare":
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = prepare_cache(reqs, base_tree=pathlib.Path(a.base_tree).resolve(),
                             tei_url=a.tei_url,
                             seed_cache=pathlib.Path(a.seed_cache) if a.seed_cache else None,
                             out=out_dir / "vectors.json", python=a.python)
        print(json.dumps(meta, indent=1))
        return 0
    report = run_parity(reqs, base_tree=pathlib.Path(a.base_tree).resolve(),
                        head_tree=pathlib.Path(a.head_tree).resolve(),
                        collection=a.collection, qdrant_url=a.qdrant_url,
                        vec_cache=pathlib.Path(a.vec_cache).resolve(), out_dir=out_dir,
                        shards=a.shards, deep=not a.no_deep, score_tol=a.score_tol,
                        seed_probe=a.seed_probe, python=a.python, nice=a.nice,
                        guard=guard.refuse_non_twin, arm_extra=_kv(a.arm_env),
                        base_sha=a.base_sha, head_sha=a.head_sha)
    print(summary(report))
    print(f"wrote {out_dir / 'parity.json'}")
    return EXIT[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
