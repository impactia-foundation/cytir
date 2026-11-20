#!/usr/bin/env python3
# Moved from the retired news_daily_gate script (task 677, task 643 S5); the old path was an alias shim, deleted in S9.
"""ADR-043 amendment 4 -- the daily gate for routine news packs.

Tim, 2026-09-17 14:25Z: *"I don't think you need to recheck the held-out set
every time we add news articles"*.  This module is that ruling in code, and
it is deliberately narrow.

**What it replaces.**  Every content pack today cuts two 1,418-query arms and
compares them across the whole held-out set; a refusal then costs a rollback
and, under am.3, a third arm before the number may be called the pack's.  For
a 14,819-row pack of Swiss news that is the held-out battery run twice a day.
The daily gate scores the **gold core plus the flood set and the three probes
-- 667 queries** -- and applies **all four ADR-043 conditions** to them: net
h@1/h@10 non-negative, flood non-increasing, no class broken, no LEGAL or
NAMES loss.  The population is the one ADR-043 is written over, so the daily
number IS the ADR-043 number and not a proxy for it.

**What it does not replace.**  The weekly battery -- the other 755 held-out
queries (slice_b, the dom_* sets, jurisdiction, freshness, commercial,
general_flood, entity cards, zero_signal, vocab robustness) and am.3's null
arm.  That is what Tim's ruling moved off the daily path.  The cadence ledger
here is what makes it unskippable: a routine pack is refused once seven packs,
seven days or 100,000 rows have accumulated since the last passing battery, or
if the release moved under it.

**What it may never do.**  Lower a tripwire, waive a refusal, or adjudicate
its own failure.  A refusal aborts the ADR-098 transaction before production
is written, returns the lane to handoff-only and escalates to Tim with the
number (memory: gate-waivers-escalate-to-tim).  The standard it applies lives
in `data/news-daily-gate-contract.v1.json` and the queries in
`data/news-daily-gate-slice.v1.json`; neither can be edited from inside a run.

**The measurement that justifies the size.**  On the 09-16 pack -- the one
the full battery refused at h@1 219->214 -- this gate reads **h@1 -5 and h@10
-1**, the battery's own deltas, with a null arm that moves nothing
(the task 605 gate-arms test fixture, projected from task 604's three
sealed arms).  It refuses what the battery refused.  That is the property a
cheap gate has to have before it is allowed to stand in for an expensive one.

**On the size, honestly.**  Task 605 first cut a 329-query slice of the
classes task 603's mechanism predicts.  It worked -- same verdict, 23 s -- and
was still the wrong design, because it left 361 golds unmeasured between
batteries and made the daily number a detector rather than ADR-043's own.  The
whole core costs 45 s.  Twenty-two seconds a day is not a trade worth making
against the bar.
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
from cytir.core.paths import data_path  # noqa: E402
from cytir.core.serve_layout import package_name  # noqa: E402
if str(ROOT) not in sys.path:   # run as a script (`python3 cytir/gates/news_daily.py`), sys.path[0] is cytir/gates
    sys.path.insert(0, str(ROOT))
from cytir.index import layout as layout  # noqa: E402
from cytir.gates import content_gate_null_arm as null  # noqa: E402
from cytir.gates import authority as gate  # noqa: E402
from cytir.core import serving_config as serving_config  # noqa: E402

ARM_SCHEMA = "aw.news-daily-gate-arm.v1"
LEDGER_SCHEMA = "aw.news-gate-cadence.v1"
CONTRACT_SCHEMA = "aw.news-daily-gate-contract.v1"
SLICE_SCHEMA = "aw.news-daily-gate-slice.v1"

CONTRACT = data_path("news-daily-gate-contract.v1.json")
LEDGER_EVENTS = ("pack", "battery")


class DailyGateError(RuntimeError):
    """The daily gate cannot be run or read as a measurement."""


# ---------------------------------------------------------------------------
# contract + slice


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_contract(path: Path = CONTRACT) -> dict:
    if not path.is_file():
        raise DailyGateError(f"daily-gate contract absent: {path}")
    doc = json.loads(path.read_text())
    if doc.get("schema") != CONTRACT_SCHEMA:
        raise DailyGateError(f"{path} is not a {CONTRACT_SCHEMA}")
    return doc


def load_slice(contract: Mapping, release_dir: Path = ROOT) -> dict:
    """The slice, bound to the contract by sha256.

    Two files, one standard: the contract says which queries the gate is
    judged on and the slice file carries them.  A slice the contract does not
    name by hash is not this gate's slice -- that is the whole point of
    keeping the digest here rather than trusting the path.
    """
    spec = contract["slice"]
    path = release_dir / spec["path"]
    if not path.is_file():
        raise DailyGateError(f"daily-gate slice absent: {path}")
    actual = digest(path)
    if actual != spec["sha256"]:
        raise DailyGateError(
            f"slice {path} is {actual[:12]}..., the contract binds "
            f"{spec['sha256'][:12]}... -- regenerate one or the other in a PR")
    doc = json.loads(path.read_text())
    if doc.get("schema") != SLICE_SCHEMA:
        raise DailyGateError(f"{path} is not a {SLICE_SCHEMA}")
    return doc


def assert_slice_matches_gold_sets(slice_doc: Mapping,
                                   release_dir: Path = ROOT) -> None:
    """The slice must be the gold sets THIS release has, not last month's.

    A count or a roster that does not track the set it describes is the defect
    this project keeps paying for: `compare_corpus_arms.py` still demands
    `total == 663` while ADR-107 am.3 (PR #183, merged 2026-09-17) removed 24
    contaminated queries, so the next battery arm cut from main reports 639
    and the ratified comparator calls it a measurement error.

    Here the slice records each gold file's sha256 at generation time, and a
    file that changed refuses the run: a gold added to an eval set must reach
    the daily bar by a PR that regenerates the slice, and a gold removed must
    leave it the same way. Never silently.
    """
    for source in slice_doc["sources"]:
        path = release_dir / source["path"]
        if not path.is_file():
            raise DailyGateError(f"gold set absent from this release: {path}")
        actual = digest(path)
        if actual != source["sha256"]:
            raise DailyGateError(
                f"{source['path']} is {actual[:12]}..., the slice was "
                f"generated from {source['sha256'][:12]}... -- regenerate "
                f"data/news-daily-gate-slice.v1.json in a PR "
                f"(cyt gate news-daily-slice)")


# ---------------------------------------------------------------------------
# cutting an arm


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DailyGateError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module   # dataclasses resolve their module here
    spec.loader.exec_module(module)
    return module


def pinned_env(release_dir: Path) -> dict[str, str]:
    """Task 369's pinned gate environment, from its one home.

    Loaded rather than copied.  A second literal copy of these 39 knobs is
    exactly how a gate arm ends up measuring a ranker production does not run
    -- which is what task 604 found had been happening for every content-pack
    verdict since the flip.
    """
    module = _load_module(release_dir / package_name(release_dir) / "gates" / "run_arm.py",
                          "news_daily_gate_run369")
    expected = dict(module.PINNED_ENV)
    expected.update({"AW_COUNTRY_SCOPE_MODE": "policy_only",
                     "AW_QHEAD_COUNTRY_LEGS": "0",
                     "AW_QHEAD_COUNTRY_TRACE": "1"})
    return expected


def assert_env(release_dir: Path) -> dict[str, str]:
    expected = pinned_env(release_dir)
    bad = {key: (os.environ.get(key), value)
           for key, value in expected.items() if os.environ.get(key) != value}
    if bad:
        raise DailyGateError(f"daily-gate environment mismatch: {bad}")
    for key in ("AW_QHEAD_PATH", "AW_HUBFLOOR_TABLE"):
        value = os.environ.get(key)
        if not value or not Path(value).is_file():
            raise DailyGateError(f"{key} must name a readable artifact: {value!r}")
    return expected


def target_state(url: str, collection: str) -> dict:
    """The arm's record of WHAT it measured — counts and, since task 627, the
    LAYOUT FINGERPRINT that decides what the BM25 leg scores.

    `status: green` / `optimizer_status: ok` stay the liveness check they always
    were. They are NOT a settle predicate: both are true of a collection
    carrying 2.6% stale postings (task 627 §3.3), and task 606 recorded both
    true of one carrying a plain segment. The fingerprint is what a later reader
    needs in order to know whether two arms were comparable at all, and it is
    recorded unconditionally — an arm that cannot read it is not refused here,
    it just says so, because a MISSING fingerprint must never be mistaken for a
    small gap (`judge` declines to adjudicate rather than assuming).
    """
    endpoint = f"{url.rstrip('/')}/collections/{collection}"
    with IH.urlopen(endpoint, timeout=30) as response:
        result = json.load(response)["result"]
    queue = int((result.get("update_queue") or {}).get("length", 0))
    state = {"status": result.get("status"),
             "optimizer_status": result.get("optimizer_status"),
             "points_count": int(result["points_count"]),
             "update_queue_length": queue}
    if (state["status"] != "green" or state["optimizer_status"] != "ok"
            or queue):
        raise DailyGateError(f"target is not quiet-green: {state}")
    try:
        fingerprint = layout.fingerprint(url, collection)
    except Exception as exc:      # noqa: BLE001 — never fail an arm on telemetry
        state["layout"] = None
        state["layout_error"] = f"{type(exc).__name__}: {exc}"
    else:
        state["layout"] = fingerprint
        state["layout_settled"] = layout.settled(fingerprint)
    return state


def cut_arm(*, release_dir: Path, url: str, collection: str, identity: str,
            cache: str, out_dir: Path, arm_id: str,
            contract_path: Path = CONTRACT) -> Path:
    """Score the slice against one collection state, read-only."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", arm_id):
        raise DailyGateError("arm-id must be lowercase [a-z0-9_-]")
    contract = load_contract(contract_path)
    slice_doc = load_slice(contract, release_dir)
    expected_env = assert_env(release_dir)
    match = re.fullmatch(r"([0-9a-f]{64})=(.+)", cache)
    if not match:
        raise DailyGateError("--cache must be SHA256=PATH")
    cache_path = Path(match.group(2))
    if not cache_path.is_file():
        raise DailyGateError(f"query-vector cache absent: {cache_path}")
    cache_sha = digest(cache_path)
    if cache_sha != match.group(1):
        raise DailyGateError(f"cache sha256 mismatch: {cache_sha}")
    state = target_state(url, collection)

    miss = gate.install_cached_embedder(str(cache_path))
    sys.path[:0] = [str(release_dir)]
    # the RELEASE tree's own package (an old release carries `cytir/`, a new one `cytir/`)
    retrieval = importlib.import_module(package_name(release_dir) + ".serve.retrieval")

    runner = _load_module(
        release_dir / package_name(release_dir) / "gates" / "run_corpus_arm.py",
        "news_daily_gate_run370")
    binding = runner.serving_binding(retrieval)
    problems = serving_config.gate_serving_binding_problems(binding)
    if problems:
        raise DailyGateError(
            "refusing an arm that did not measure what production serves: "
            + "; ".join(problems))

    paths = dict(gate.SETS)
    assert_slice_matches_gold_sets(slice_doc, release_dir)
    wanted = {name: set(qids) for name, qids in slice_doc["sets"].items()}
    sets: dict[str, dict] = {}
    for name, qids in slice_doc["sets"].items():
        rows = gate.load_rows(paths[name])
        per_q, h1, h10, n_gold = {}, 0, 0, 0
        for index, row in enumerate(rows):
            qid = str(row.get("id") or row.get("qid") or f"{name}-{index}")
            if qid not in wanted[name]:
                continue
            gold_urls, gold_domains = gate.gold_of(row)
            retrieval._LAST_QUERY_PLAN_TRACE.clear()
            urls = retrieval.search(row.get("query") or row.get("q"), k=10,
                                    lang=row.get("lang"),
                                    country=row.get("country"))
            urls = [hit.get("url") or "" for hit in urls]
            rank = (gate.rank_of(urls, gold_urls, gold_domains)
                    if (gold_urls or gold_domains) else None)
            # task626 (Tim, 2026-09-21): the same trace shape Task370 records
            # (`runner.trace`, one home -- not re-derived here), so the
            # content guard (every lost gold's dense leg identical between
            # baseline and candidate) can run off the daily gate's OWN arms
            # instead of a second, separate arm-cutting pass.
            per_q[qid] = {"rank": rank, "urls": urls,
                          "trace": runner.trace(retrieval)}
            if gold_urls or gold_domains:
                n_gold += 1
                h1 += rank == 1
                h10 += bool(rank and rank <= 10)
        missing = wanted[name] - set(per_q)
        if missing:
            raise DailyGateError(
                f"{name}: the slice names {len(missing)} queries this release's "
                f"gold set does not have (e.g. {sorted(missing)[:3]}) -- the "
                f"slice is stale, regenerate it in a PR")
        sets[name] = {"n": len(per_q), "n_gold": n_gold, "h1": h1, "h10": h10,
                      "per_q": per_q}

    # The probe fixture is pinned by the slice, exactly as the battery pins it
    # in every arm: a probe whose spec changed underneath two arms would make
    # them disagree about a contract neither of them stated.
    probe_fixture = release_dir / slice_doc["probe_fixture"]["path"]
    probe_fixture_sha256 = digest(probe_fixture)
    if probe_fixture_sha256 != slice_doc["probe_fixture"]["sha256"]:
        raise DailyGateError(
            f"probe fixture {probe_fixture} is {probe_fixture_sha256[:12]}..., "
            f"the slice pins {slice_doc['probe_fixture']['sha256'][:12]}...")
    probe_specs = {item["id"]: item for item in json.loads(
        probe_fixture.read_text())}
    probes = {}
    for probe_id in slice_doc["probes"]:
        spec = probe_specs[probe_id]
        urls = [hit.get("url") or "" for hit in
                retrieval.search(spec["query"], k=10, lang=spec.get("lang"),
                                 country=spec.get("country"))]
        # EXACT url, not `gate.rank_of`. A probe's contract is that the
        # anchored article itself is returned -- `rank_of` also accepts the
        # page the anchor is on (it compares before the `#`), which on
        # `art257-de` is the difference between rank 10 and rank 1. Cutting
        # both arms on one twin state and comparing them query by query is
        # what found this: every gold agreed and two probes did not
        # (task 605's `compare_daily_and_battery_arm.py`). A daily
        # gate whose probes are laxer than the battery's would sign off a
        # regression the battery is there to catch.
        rank = next((index for index, url in enumerate(urls, 1)
                     if gate.norm_url(url) == gate.norm_url(spec["target_url"])),
                    None)
        probes[probe_id] = {"rank": rank, "urls": urls, "spec": spec}

    arm = {
        "schema": ARM_SCHEMA,
        "identity": identity,
        "arm_id": arm_id,
        "cache_miss": miss[0],
        "cache_sha256": cache_sha,
        "slice_sha256": digest(release_dir / contract["slice"]["path"]),
        "contract_sha256": digest(contract_path),
        "probe_fixture_sha256": probe_fixture_sha256,
        "target": {"qdrant_url": url, "collection": collection, **state},
        "serving_binding": binding,
        "pinned_env": expected_env,
        "effective_env": {key: os.environ.get(key) for key in expected_env},
        "retrieval_sha256": digest(release_dir / package_name(release_dir) / "serve" /
                                   "retrieval.py"),
        "qhead_sha256": digest(Path(os.environ["AW_QHEAD_PATH"])),
        "sets": sets,
        "probes": probes,
    }
    refuse_error_arm(arm)
    payload = json.dumps(arm, indent=1, sort_keys=True) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (f"news-daily-gate-arm.{arm_id}."
                      f"{hashlib.sha256(payload.encode()).hexdigest()}.json")
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def refuse_error_arm(arm: Mapping) -> None:
    """An arm made of errors is not a measurement (verification-doctrine 6).

    Task 600's 1,418 queries all came back `__ERROR__:UnexpectedResponse` --
    the gate was querying a 256-dim collection at 128 -- and the arm was
    written and reported as a result.  A daily gate that swallowed the same
    outage would read every gold as "absent from the top 10" and refuse the
    pack for the wrong reason, or, on the baseline side, accept anything.
    """
    bad = []
    for name, result in sorted((arm.get("sets") or {}).items()):
        for qid, row in sorted((result.get("per_q") or {}).items()):
            urls = row.get("urls")
            if (not isinstance(urls, list) or not urls
                    or str(urls[0]).startswith("__ERROR__")):
                bad.append(f"{name}/{qid}")
    for probe_id, row in sorted((arm.get("probes") or {}).items()):
        urls = (row or {}).get("urls")
        if (not isinstance(urls, list) or not urls
                or str(urls[0]).startswith("__ERROR__")):
            bad.append(f"probe {probe_id}")
    if bad:
        raise DailyGateError(
            f"refusing an arm whose queries errored -- this is an outage, "
            f"not a result: {len(bad)} of "
            f"{sum(len(r.get('per_q') or {}) for r in arm['sets'].values()) + len(arm['probes'])}"
            f" (e.g. {bad[:3]})")
    if arm.get("cache_miss") != 0:
        raise DailyGateError(f"cache_miss={arm.get('cache_miss')}")


# ---------------------------------------------------------------------------
# the cadence ledger


def read_ledger(path: Path) -> list[dict]:
    """Every row, or a refusal. A counter that cannot be read is not zero."""
    if not path.is_file():
        raise DailyGateError(
            f"cadence ledger absent: {path} -- the routine path is refused "
            f"until a passing battery is recorded (ADR-043 am.4)")
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise DailyGateError(f"{path}:{number} is not JSON: {exc}") from None
        if row.get("schema") != LEDGER_SCHEMA:
            raise DailyGateError(f"{path}:{number} is not a {LEDGER_SCHEMA}")
        if row.get("event") not in LEDGER_EVENTS:
            raise DailyGateError(f"{path}:{number} unknown event "
                                 f"{row.get('event')!r}")
        rows.append(row)
    return rows


def parse_time(value: str, label: str) -> dt.datetime:
    try:
        stamp = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise DailyGateError(f"{label} is not an ISO-8601 time: {value!r}") from None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.timezone.utc)


def cadence_state(rows: Sequence[Mapping], *, now: dt.datetime) -> dict:
    """Where this lane stands against the battery interval."""
    batteries = [row for row in rows if row["event"] == "battery"]
    if not batteries:
        raise DailyGateError(
            "the cadence ledger records no passing battery -- a routine pack "
            "may not be the first thing this lane does (ADR-043 am.4)")
    last = batteries[-1]
    at = parse_time(last["at"], "battery.at")
    since = [row for row in rows if parse_time(row["at"], "row.at") > at
             and row["event"] == "pack"]
    return {"battery_at": at.isoformat(),
            "battery_release_sha": last.get("release_sha"),
            "packs_since": len(since),
            "rows_since": sum(int(row.get("rows") or 0) for row in since),
            "days_since": round((now - at).total_seconds() / 86400, 3)}


def cadence_problems(state: Mapping, contract: Mapping, *,
                     release_sha: str | None, pack_rows: int) -> list[str]:
    cadence = contract["cadence"]
    problems = []
    if state["packs_since"] + 1 > cadence["max_packs_between_batteries"]:
        problems.append(
            f"battery due: this would be routine pack "
            f"{state['packs_since'] + 1} since the last battery, the contract "
            f"allows {cadence['max_packs_between_batteries']}")
    if state["days_since"] > cadence["max_days_between_batteries"]:
        problems.append(
            f"battery due: {state['days_since']} days since the last battery, "
            f"the contract allows {cadence['max_days_between_batteries']}")
    if (state["rows_since"] + pack_rows >
            cadence["max_rows_between_batteries"]):
        problems.append(
            f"battery due: {state['rows_since'] + pack_rows} rows would have "
            f"accumulated since the last battery, the contract allows "
            f"{cadence['max_rows_between_batteries']}")
    if release_sha and state["battery_release_sha"] != release_sha:
        problems.append(
            f"battery due: this release is {str(release_sha)[:12]}, the last "
            f"battery ran on {str(state['battery_release_sha'])[:12]} -- arms "
            f"cut under different releases do not measure one drift")
    return problems


def append_ledger(path: Path, row: dict) -> dict:
    row = {"schema": LEDGER_SCHEMA, **row}
    missing = [key for key in ("event", "at", "release_sha") if not row.get(key)]
    if missing:
        raise DailyGateError(f"cadence row missing {missing}")
    if row["event"] not in LEDGER_EVENTS:
        raise DailyGateError(f"unknown cadence event {row['event']!r}")
    parse_time(row["at"], "at")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return row


# ---------------------------------------------------------------------------
# scope


def routine_scope_problems(intent: Mapping, contract: Mapping, *,
                           points_count: int) -> list[str]:
    """Is this pack one the daily gate is allowed to judge at all?

    Read from the SEALED preservation intent, never from the operator's
    claim about it: `pack.rows`, `scope` and each source's profile are what
    the ADR-098 transaction is about to write, and they are hashed into the
    intent the transaction verifies.
    """
    routine = contract["routine_pack"]
    problems = []
    scope = intent.get("scope")
    if scope != routine["scope"]:
        problems.append(f"intent scope {scope!r} is not {routine['scope']!r}")
    rows = intent.get("pack", {}).get("rows")
    if not isinstance(rows, int) or rows <= 0:
        problems.append(f"intent names no positive pack.rows ({rows!r})")
    else:
        if rows > routine["max_rows"]:
            problems.append(f"pack is {rows} rows, the routine bound is "
                            f"{routine['max_rows']}")
        if points_count <= 0:
            problems.append("collection points_count is not positive")
        elif rows / points_count > routine["max_fraction_of_collection"]:
            problems.append(
                f"pack is {rows / points_count:.4%} of the collection, the "
                f"routine bound is "
                f"{routine['max_fraction_of_collection']:.2%}")
    profiles = sorted({str((source or {}).get("profile"))
                       for source in intent.get("sources") or []})
    if not profiles:
        problems.append("intent names no sources")
    unknown = [p for p in profiles if p not in routine["preservation_profiles"]]
    if unknown:
        problems.append(
            f"source profiles {unknown} are not news-lane families "
            f"{routine['preservation_profiles']} -- this takes the full "
            f"ADR-043 gate")
    return problems


# ---------------------------------------------------------------------------
# judging


def rank_key(rank) -> int:
    return null.rank_key(rank)


def _without_path_fields(value):
    """Strip every key literally named `path` (or ending `_path`), anywhere
    in a nested serving_binding structure. Task 626 (2026-09-22, 09-21
    under r17): a release cut alone moves every release-dir-prefixed
    filesystem path a serving_binding carries (`fusion_weights_path`,
    `effective_config.fusion_weights.path`,
    `effective_config.rankhead_v2.path`) even when the referenced file's
    CONTENT is byte-identical (same `sha256` sitting right next to each
    path) -- a path is WHERE a file was read from, never a property of
    what it did to scoring, and every one of these already carries its
    own content hash for the comparison that actually matters. Applies
    regardless of `same_target`: a same-target comparison across two
    releases hits the exact same false difference.
    """
    if isinstance(value, Mapping):
        return {k: _without_path_fields(v) for k, v in value.items()
                if k != "path" and not k.endswith("_path")}
    if isinstance(value, list):
        return [_without_path_fields(v) for v in value]
    return value


def _serving_binding_for_comparison(binding: Mapping | None,
                                    *, same_target: bool) -> object:
    """`serving_binding`, minus the fields the target is allowed to move.

    `effective_config.collection` is a property of WHICH corpus answered the
    queries (task 626: found comparing a real prod-baseline arm against a
    real twin candidate -- `aw_prod` vs `aw_e8`, otherwise byte-identical:
    same rescore/oversampling/hnsw_ef/mrl_dim/fusion_weights/rankhead). Under
    `same_target=False` that is exactly the difference am.7 exists to allow,
    so it is excluded from the comparison rather than failing every
    cross-target reading on the one field the whole control is FOR. Path
    fields (see `_without_path_fields`) are stripped unconditionally.
    """
    if not isinstance(binding, Mapping):
        return binding
    binding = _without_path_fields(binding)
    if same_target:
        return binding
    trimmed = dict(binding)
    effective = trimmed.get("effective_config")
    if isinstance(effective, Mapping):
        trimmed["effective_config"] = {k: v for k, v in effective.items()
                                       if k != "collection"}
    return trimmed


def arm_binding_problems(baseline: Mapping, candidate: Mapping,
                         same_target: bool = True) -> list[str]:
    """Two arms are comparable only if one thing differed: the pack.

    `same_target=False` is ADR-043 am.7's cross-target reading: a concurrent
    PRODUCTION baseline against a TWIN candidate is deliberately two
    different `qdrant_url`/`collection` pairs -- that check is the one thing
    this comparison is FOR, so it is skipped, not relaxed by accident.
    Everything else (slice, cache, contract, gate-set/query/probe
    composition, pinned env, serving binding) must still match: the only
    thing allowed to differ is which corpus answered the same queries.
    """
    problems = []
    for key in ("schema", "slice_sha256", "contract_sha256", "cache_sha256",
                "probe_fixture_sha256", "retrieval_sha256", "qhead_sha256",
                "pinned_env"):
        if baseline.get(key) != candidate.get(key):
            problems.append(f"arms differ in {key}")
    if (_serving_binding_for_comparison(baseline.get("serving_binding"),
                                        same_target=same_target) !=
            _serving_binding_for_comparison(candidate.get("serving_binding"),
                                            same_target=same_target)):
        problems.append("arms differ in serving_binding")
    for arm, label in ((baseline, "baseline"), (candidate, "candidate")):
        if arm.get("pinned_env") != arm.get("effective_env"):
            problems.append(f"{label}: pinned environment drift")
        problems.extend(
            f"{label}: {problem}" for problem in
            serving_config.gate_serving_binding_problems(
                arm.get("serving_binding")))
    if same_target:
        base, cand = baseline.get("target") or {}, candidate.get("target") or {}
        for key in ("qdrant_url", "collection"):
            if base.get(key) != cand.get(key):
                problems.append(f"arms are not the same target: {key}")
    if baseline.get("identity") == candidate.get("identity"):
        problems.append("arms do not have separate identities")
    if set(baseline.get("sets") or {}) != set(candidate.get("sets") or {}):
        problems.append("gate-set composition differs")
    else:
        for name, result in (baseline.get("sets") or {}).items():
            if (set(result.get("per_q") or {}) !=
                    set((candidate["sets"][name].get("per_q") or {}))):
                problems.append(f"{name}: query composition differs")
    if set(baseline.get("probes") or {}) != set(candidate.get("probes") or {}):
        problems.append("probe composition differs")
    return problems


def metrics_of(arm: Mapping, slice_doc: Mapping) -> tuple[int, int]:
    h1 = sum(arm["sets"][name]["h1"] for name in slice_doc["gold_sets"])
    h10 = sum(arm["sets"][name]["h10"] for name in slice_doc["gold_sets"])
    return h1, h10


def loss_rows(baseline: Mapping, candidate: Mapping,
              names: Iterable[str]) -> list[tuple[str, object, object]]:
    out = []
    for name in names:
        before = baseline["sets"][name]["per_q"]
        after = candidate["sets"][name]["per_q"]
        for qid, row in before.items():
            if rank_key(after[qid].get("rank")) > rank_key(row.get("rank")):
                out.append((f"{name}/{qid}", row.get("rank"),
                            after[qid].get("rank")))
    return out


def probe_problems(baseline: Mapping, candidate: Mapping,
                   slice_doc: Mapping) -> list[str]:
    problems = []
    for probe_id in slice_doc["probes"]:
        before, after = baseline["probes"][probe_id], candidate["probes"][probe_id]
        if before.get("spec") != after.get("spec"):
            problems.append(f"probe {probe_id}: fixture contract drift")
            continue
        if rank_key(after.get("rank")) > rank_key(before.get("rank")):
            problems.append(f"probe {probe_id}: exact anchor regressed "
                            f"{before.get('rank')}->{after.get('rank')}")
    return problems


def _fp(arm: Mapping) -> Mapping | None:
    return ((arm.get("target") or {}).get("layout")) or None


def layout_comparability(baseline: Mapping, candidate: Mapping,
                         prod_baseline: Mapping | None) -> dict:
    """Is this pair of arms a MEASUREMENT of the pack, or of the instrument?

    Task 627 established that the BM25 leg's ranking of rows nobody touched is a
    function of the segment layout, and that two mirrors at a byte-equal
    membership can be 32 golds and 5 h@1 apart because of it. So the gate asks
    two questions before it believes a number, and both are answered from
    fingerprints the arms already carry:

    1. **The cross-target gap** (ADR-043 am.7's `baseline_disagreement`, with its
       cause attached). The twin pre-apply baseline and the concurrent
       production baseline are read at the SAME membership, so any difference in
       their stale-posting ratio is pure instrument. That gap is the quantity to
       judge, not the raw `n` difference: `n` legitimately rises when a pack is
       written, so comparing a pre-apply baseline against a post-apply candidate
       would always look broken.

    2. **A straddled optimizer pass.** Baseline and candidate are read on the
       same collection, but a merge or vacuum between them rebuilds segments
       from live points and moves every term's IDF with no write at all. Task
       626c's own twin arms straddled one (baseline 08:07Z, merge
       08:32:00Z→08:40:46Z, candidate 08:42Z on 2026-09-21).

    A failure of either makes the reading UNADJUDICATED, never FAIL: the pack
    has not been shown to cost anything, and it has not been shown to be safe.
    That is the same shape as am.6's unadjudicated tripwire — it buys a
    measurement, never a tolerance, and it widens no band.
    """
    out = {"adjudicable": True, "checked": True, "problems": [],
           "cross_target_gap": None, "straddled_optimizer": [], "settled": {}}
    fps = {"twin_preapply": _fp(baseline), "candidate": _fp(candidate)}
    if prod_baseline is not None:
        fps["prod_baseline"] = _fp(prod_baseline)
    for name, fp in fps.items():
        out["settled"][name] = (None if fp is None else layout.settled(fp))

    missing = sorted(name for name, fp in fps.items() if fp is None)
    if len(missing) == len(fps):
        # Every arm predates task 627. The data to check simply does not exist,
        # and refusing retroactively would make the gate unable to judge any
        # historical pair -- including the sealed 2026-09-17 arms am.7 rests on.
        # So the pair is adjudicated exactly as it was before, and the verdict
        # is STAMPED `checked: false` so nobody can later cite it as
        # layout-clean. "We could not check" is recorded, never rounded to
        # "we checked and it was fine".
        out["checked"] = False
        out["problems_informational"] = [
            "no arm carries a layout fingerprint (all pre-task-627) — this "
            "verdict is NOT layout-checked and must not be cited as evidence "
            "that the arms were comparable"]
        return out
    if missing:
        # Mixed is the suspicious case: a release that records fingerprints cut
        # one of these arms and not the others, so one of them is not what it
        # claims to be. A missing fingerprint is not a small gap.
        out["adjudicable"] = False
        out["problems"].append(
            "no layout fingerprint on " + ", ".join(missing)
            + " while the other arm(s) carry one — the arms were not cut by the "
              "same release; re-cut them together (task 627)")
        return out

    if prod_baseline is not None:
        gap = layout.gap(fps["prod_baseline"], fps["twin_preapply"])
        out["cross_target_gap"] = gap
        if not gap["same_membership"]:
            out["adjudicable"] = False
            out["problems"].append(
                "the production baseline and the twin pre-apply baseline are at "
                f"different memberships ({gap['a']['points_count']:,} vs "
                f"{gap['b']['points_count']:,}) — their disagreement cannot be "
                "attributed to layout")
        elif not gap["within_measured_safe"]:
            out["adjudicable"] = False
            out["problems"].append(
                f"cross-target layout gap {gap['stale_ratio_gap']:.3e} exceeds "
                f"the measured-safe {gap['measured_safe_threshold']:.3e} "
                f"(stale postings {gap['a']['stale_postings']:+,.0f} vs "
                f"{gap['b']['stale_postings']:+,.0f}) — at this distance the "
                "two mirrors have been measured to disagree by 5 h@1 and 32 "
                "golds on identical content, so a verdict here would be about "
                "the instrument")

    straddled = layout.optimizer_ran_between(fps["twin_preapply"],
                                             fps["candidate"])
    out["straddled_optimizer"] = straddled
    if straddled:
        out["adjudicable"] = False
        out["problems"].append(
            f"{len(straddled)} optimizer pass(es) ran between the baseline and "
            "candidate arms ("
            + "; ".join(f"{e.get('name')} {e.get('start_at')}->{e.get('end_at')}"
                        for e in straddled[:3])
            + ") — a rebuild moves every term's IDF with no write, so the pair "
              "is not a converged reading")
    return out


def judge(*, baseline: Mapping, candidate: Mapping, contract: Mapping,
          slice_doc: Mapping, intent: Mapping, cadence: Mapping,
          null_arm: Mapping | None = None,
          null_basis: Mapping | None = None,
          baseline_arm_sha256: str | None = None,
          prod_baseline: Mapping | None = None) -> dict:
    """The verdict. Every condition ADR-043 am.4 names, on the slice.

    ADR-043 am.7 (task 625, 2026-09-20): `baseline` is the twin's pre-apply
    reading, cut before the pack writes it -- and the twin can disagree with
    production on the SAME slice at the SAME logical membership, because
    approximate search over two different physical segment layouts is not
    guaranteed to agree on a near-tied rank (task 600/601/604/609/613/616/
    617/622's `de-law-013` bistability; task 625's 09-17 pack: twin baseline
    206, a converged concurrent prod reading 201, the candidate 202 -- 2
    ranks from prod, 32 from the twin baseline that a plain reading charged
    to the pack). `prod_baseline`, when supplied, is what the candidate is
    actually SCORED against (h@1/h@10/losses/tripwire/probes/broken-classes
    all move to it) -- concurrent production is ground truth for "did this
    pack cost anything", not the twin's own pre-write reading. `baseline`
    stays the null-arm's identity anchor regardless (am.6's control is about
    twin write-noise, orthogonal to which reading the candidate is judged
    against), and a twin/prod disagreement beyond the tripwire band is
    reported in the verdict (`baseline_disagreement`), not silently traded
    for whichever number is convenient.
    """
    problems = arm_binding_problems(baseline, candidate)
    for arm in (baseline, candidate):
        refuse_error_arm(arm)
    if baseline.get("slice_sha256") != contract["slice"]["sha256"]:
        problems.append("arms were cut against a slice the contract does not "
                        "bind")
    baseline_disagreement = None
    if prod_baseline is not None:
        prod_problems = arm_binding_problems(prod_baseline, candidate,
                                             same_target=False)
        if prod_problems:
            raise DailyGateError(
                "prod baseline: " + "; ".join(prod_problems))
        refuse_error_arm(prod_baseline)
        disagreement_moved = null.moved(baseline["sets"], prod_baseline["sets"])
        h1_twin, _ = metrics_of(baseline, slice_doc)
        h1_prod, _ = metrics_of(prod_baseline, slice_doc)
        baseline_disagreement = {
            "moved": len(disagreement_moved),
            "h@1": {"twin_preapply": h1_twin, "prod_concurrent": h1_prod},
            "representative": len(disagreement_moved) == 0,
            "note": ("twin pre-apply baseline agrees with a concurrent "
                    "production reading" if len(disagreement_moved) == 0 else
                    f"baseline disagreement: twin {h1_twin} vs prod "
                    f"{h1_prod} -- twin baseline not representative, "
                    "adjudicating against production"),
        }
    if problems:
        raise DailyGateError("; ".join(problems))

    scope = routine_scope_problems(
        intent, contract,
        points_count=int((baseline.get("target") or {}).get("points_count", 0)))
    effective_baseline = prod_baseline if prod_baseline is not None else baseline
    h1a, h10a = metrics_of(effective_baseline, slice_doc)
    h1b, h10b = metrics_of(candidate, slice_doc)
    null_h1 = null_h10 = None
    null_moved: set[str] = set()
    if null_arm is not None:
        if null_basis is None:
            raise DailyGateError("a null arm needs its basis receipt "
                                 "(ADR-043 am.3 rule 1)")
        for problem in arm_binding_problems(baseline, null_arm):
            raise DailyGateError(f"null arm: {problem}")
        if baseline_arm_sha256 is None:
            raise DailyGateError(
                "a null arm needs the baseline arm's sha256 to check the "
                "basis's `adjudicates` citation (ADR-043 am.6 rule 3)")
        null.validate_basis(
            null_basis,
            collection=(baseline["target"] or {}).get("collection"),
            points_count=int((baseline["target"] or {}).get("points_count", 0)),
            baseline_identity=baseline.get("identity"),
            baseline_arm_sha256=baseline_arm_sha256)
        h1n, h10n = metrics_of(null_arm, slice_doc)
        null_h1, null_h10 = h1n - h1a, h10n - h10a
        null_moved = null.moved(baseline["sets"], null_arm["sets"])
    metrics = {"h@1": null.metric(h1b - h1a, null_h1),
               "h@10": null.metric(h10b - h10a, null_h10)}

    raw_losses = loss_rows(effective_baseline, candidate, slice_doc["sets"])
    attributable, instrument = null.split_losses(raw_losses, null_moved)
    tripwire = [row for row in raw_losses
                if row[0].split("/", 1)[0] in slice_doc["tripwire_sets"]]
    # ADR-043 am.6, on the same terms as the battery door: the ranks stay raw,
    # zero tolerance stays, and the null arm decides whose loss it is. No null
    # arm means unadjudicated, which still refuses.
    trip = null.tripwire_verdict(tripwire, null_moved,
                                 adjudicated=null_arm is not None)
    probes = probe_problems(effective_baseline, candidate, slice_doc)
    flood_name = slice_doc["flood_set"]
    flood_ok = (candidate["sets"][flood_name]["h1"] >=
                effective_baseline["sets"][flood_name]["h1"]
                and candidate["sets"][flood_name]["h10"] >=
                effective_baseline["sets"][flood_name]["h10"])
    # ADR-043 condition 3, computed exactly as the comparator computes it --
    # possible here only because the daily gate scores the whole gold core.
    broken = [name for name in slice_doc["gold_sets"]
              if effective_baseline["sets"][name]["h10"] > 0
              and candidate["sets"][name]["h10"] <
              effective_baseline["sets"][name]["h10"] / 2]

    # TASK 627 / ADR-043 am.8 (proposed): is this pair a measurement of the
    # pack, or of the instrument? Computed before the refusal list so the
    # verdict can say UNADJUDICATED instead of PASS or REFUSE — a reading taken
    # across a layout gap is neither evidence for the pack nor against it.
    comparability = layout_comparability(baseline, candidate, prod_baseline)

    refusals = []
    refusals += [f"out of routine scope: {problem}" for problem in scope]
    refusals += [f"cadence: {problem}" for problem in cadence.get("problems", [])]
    refusals += [f"tripwire loss {key} {before}->{after}"
                 + (" (unadjudicated: no null arm, am.6)"
                    if trip["unadjudicated"] else
                    f" (attributable; the null arm read "
                    f"{null.null_ranks(null_arm['sets']).get(key)})")
                 for key, before, after in trip["attributable"]]
    refusals += [f"probe: {problem}" for problem in probes]
    if not flood_ok:
        refusals.append(
            f"flood increased: rel_vs_trust "
            f"{effective_baseline['sets'][flood_name]['h1']}/"
            f"{effective_baseline['sets'][flood_name]['h10']} -> "
            f"{candidate['sets'][flood_name]['h1']}/"
            f"{candidate['sets'][flood_name]['h10']}")
    refusals += [f"class broken: {name} h@10 "
                 f"{effective_baseline['sets'][name]['h10']} -> "
                 f"{candidate['sets'][name]['h10']}" for name in broken]
    # baseline_disagreement (am.7) is reported in the verdict below, not
    # added to refusals: production is already the effective baseline above,
    # so a disagreement here is information about the twin, not a reason to
    # refuse the candidate a second time.
    for name, value in metrics.items():
        if not value["ok"]:
            refusals.append(
                f"net {name} {value['attributable']:+d} against a band "
                f"of +-{value['band']}")

    # UNADJUDICATED outranks PASS and REFUSE both. It is not a soft refusal and
    # it is not a waiver: the numbers below were computed across a layout gap
    # this gate cannot see through, so they are reported in full and no verdict
    # is drawn from them. The lane does not ship on it.
    if not comparability["adjudicable"]:
        verdict_word = "UNADJUDICATED"
    elif refusals:
        verdict_word = "REFUSE"
    else:
        verdict_word = "PASS"

    return {
        "schema": "aw.news-daily-gate-verdict.v1",
        "verdict": verdict_word,
        "layout": comparability,
        "slice_sha256": contract["slice"]["sha256"],
        "slice_queries": slice_doc["counts"]["queries"],
        "baseline": effective_baseline.get("identity"),
        "twin_preapply_baseline": baseline.get("identity"),
        "candidate": candidate.get("identity"),
        "baseline_disagreement": baseline_disagreement,
        "metrics": {"h@1": {"before": h1a, "after": h1b, **metrics["h@1"]},
                    "h@10": {"before": h10a, "after": h10b, **metrics["h@10"]}},
        "broken_classes": broken,
        "flood": {"ok": flood_ok,
                  "before": [effective_baseline["sets"][flood_name]["h1"],
                             effective_baseline["sets"][flood_name]["h10"]],
                  "after": [candidate["sets"][flood_name]["h1"],
                            candidate["sets"][flood_name]["h10"]]},
        "losses": {"raw": [f"{key} {before}->{after}"
                           for key, before, after in raw_losses],
                   "attributable": len(attributable),
                   "instrument": len(instrument)},
        "tripwire_losses": [f"{key} {before}->{after}"
                            for key, before, after in tripwire],
        "tripwire": {"raw": len(tripwire),
                     "attributable": [f"{key} {before}->{after}" for
                                      key, before, after
                                      in trip["attributable"]],
                     "instrument": [f"{key} {before}->{after}" for
                                    key, before, after in trip["instrument"]],
                     "unadjudicated": trip["unadjudicated"]},
        "probes": {probe_id: [effective_baseline["probes"][probe_id].get("rank"),
                              candidate["probes"][probe_id].get("rank")]
                   for probe_id in slice_doc["probes"]},
        "cadence": cadence,
        "scope_problems": scope,
        "refusals": refusals,
        "adjudicated": null_arm is not None,
    }


def render(verdict: Mapping) -> list[str]:
    metrics = verdict["metrics"]
    lines = [
        f"DAILY-GATE slice {verdict['slice_sha256'][:8]} "
        f"({verdict['slice_queries']} queries) "
        f"{verdict['baseline']} -> {verdict['candidate']}"
        + (f"  [am.7: twin pre-apply reading was {verdict['twin_preapply_baseline']}]"
           if verdict.get("twin_preapply_baseline") != verdict["baseline"]
           else ""),
        f"  h@1 {metrics['h@1']['before']}->{metrics['h@1']['after']} "
        f"({metrics['h@1']['attributable']:+d}; band "
        f"+-{metrics['h@1']['band']}); "
        f"h@10 {metrics['h@10']['before']}->{metrics['h@10']['after']} "
        f"({metrics['h@10']['attributable']:+d}; band "
        f"+-{metrics['h@10']['band']})",
        f"  per-gold losses={len(verdict['losses']['raw'])} "
        f"(attributable={verdict['losses']['attributable']} "
        f"instrument={verdict['losses']['instrument']}); "
        f"legal/names={len(verdict['tripwire_losses'])}"
        + (f" (attributable={len(verdict['tripwire']['attributable'])} "
           f"instrument={len(verdict['tripwire']['instrument'])})"
           if verdict["tripwire"]["raw"] and not
           verdict["tripwire"]["unadjudicated"] else "") + "; "
        f"flood ok={verdict['flood']['ok']}; "
        f"probes {verdict['probes']}",
        f"  cadence: pack {verdict['cadence'].get('packs_since', 0) + 1} of "
        f"{verdict['cadence'].get('max_packs')} since the battery at "
        f"{verdict['cadence'].get('battery_at')} "
        f"({verdict['cadence'].get('days_since')} d, "
        f"{verdict['cadence'].get('rows_since')} rows)",
    ]
    if verdict.get("baseline_disagreement") is not None:
        lines.append(f"  BASELINE (am.7): {verdict['baseline_disagreement']['note']}")
    lay = verdict.get("layout") or {}
    for name, reasons in sorted((lay.get("settled") or {}).items()):
        if reasons:
            lines.append(f"  LAYOUT {name}: NOT settled -- " + "; ".join(reasons))
    if lay.get("cross_target_gap"):
        lines.append("  LAYOUT " + layout.render_gap(lay["cross_target_gap"]).strip())
    for problem in lay.get("problems") or ():
        lines.append(f"  NOT A MEASUREMENT {problem}")
    for note in lay.get("problems_informational") or ():
        lines.append(f"  LAYOUT UNCHECKED {note}")
    for loss in verdict["losses"]["raw"]:
        lines.append(f"  RANK-DRIFT {loss}")
    for refusal in verdict["refusals"]:
        lines.append(f"  REFUSED {refusal}")
    lines.append(
        f"VERDICT: {verdict['verdict']}"
        + (" (ADR-043 am.8, proposed) -- the arms were not comparable, so the "
           "numbers above measure the instrument as much as the pack; nothing "
           "ships on an unadjudicated reading and no band is widened to cover "
           "it. Settle the layouts and re-read."
           if verdict["verdict"] == "UNADJUDICATED" else
           "" if verdict["verdict"] == "PASS" else
           " (ADR-043 am.4 daily gate) -- the transaction aborts before the "
           "production write; the lane returns to handoff-only and escalates"))
    return lines


# ---------------------------------------------------------------------------
# CLI


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    sub = parser.add_subparsers(dest="command", required=True)

    arm = sub.add_parser("arm", help="score the slice against one state")
    arm.add_argument("--release-dir", type=Path, required=True)
    arm.add_argument("--qdrant-url", required=True)
    arm.add_argument("--collection", required=True)
    arm.add_argument("--identity", required=True)
    arm.add_argument("--cache", required=True, help="SHA256=PATH")
    arm.add_argument("--out-dir", type=Path, required=True)
    arm.add_argument("--arm-id", required=True)

    judge_parser = sub.add_parser("judge", help="the daily-gate verdict")
    judge_parser.add_argument("--baseline", type=Path, required=True)
    judge_parser.add_argument("--candidate", type=Path, required=True)
    judge_parser.add_argument("--intent", type=Path, required=True)
    judge_parser.add_argument("--ledger", type=Path, required=True)
    judge_parser.add_argument("--release-sha")
    judge_parser.add_argument("--null-arm", type=Path)
    judge_parser.add_argument("--null-basis", type=Path)
    judge_parser.add_argument(
        "--prod-baseline", type=Path,
        help="ADR-043 am.7: a concurrent production reading on the same "
             "slice, scored against instead of --baseline (the twin's "
             "pre-apply reading, which can disagree with production at the "
             "same logical membership -- task 625's 2026-09-17 incident)")
    judge_parser.add_argument("--now")
    judge_parser.add_argument("--out", type=Path)

    record = sub.add_parser("record", help="append a cadence row")
    record.add_argument("--ledger", type=Path, required=True)
    record.add_argument("--event", choices=LEDGER_EVENTS, required=True)
    record.add_argument("--release-sha", required=True)
    record.add_argument("--pack")
    record.add_argument("--rows", type=int, default=0)
    record.add_argument("--evidence")
    record.add_argument("--now")

    status = sub.add_parser("status", help="how far the lane is from a battery")
    status.add_argument("--ledger", type=Path, required=True)
    status.add_argument("--now")

    args = parser.parse_args(argv)
    now = (parse_time(args.now, "--now") if getattr(args, "now", None)
           else dt.datetime.now(dt.timezone.utc))
    try:
        contract = load_contract(args.contract)
        if args.command == "arm":
            path = cut_arm(release_dir=args.release_dir, url=args.qdrant_url,
                           collection=args.collection, identity=args.identity,
                           cache=args.cache, out_dir=args.out_dir,
                           arm_id=args.arm_id, contract_path=args.contract)
            print(f"ARM_PATH={path}")
            return 0
        if args.command == "record":
            row = append_ledger(args.ledger, {
                "event": args.event, "at": now.isoformat(),
                "release_sha": args.release_sha, "pack": args.pack,
                "rows": args.rows, "evidence": args.evidence})
            print(json.dumps(row, sort_keys=True))
            return 0
        if args.command == "status":
            state = cadence_state(read_ledger(args.ledger), now=now)
            problems = cadence_problems(state, contract, release_sha=None,
                                        pack_rows=0)
            print(json.dumps({**state, "battery_due": bool(problems),
                              "problems": problems}, indent=1, sort_keys=True))
            return 0

        slice_doc = load_slice(contract)
        state = cadence_state(read_ledger(args.ledger), now=now)
        intent = _load(args.intent)
        pack_rows = int((intent.get("pack") or {}).get("rows") or 0)
        cadence = {**state,
                   "max_packs": contract["cadence"]["max_packs_between_batteries"],
                   "problems": cadence_problems(state, contract,
                                                release_sha=args.release_sha,
                                                pack_rows=pack_rows)}
        verdict = judge(
            baseline=_load(args.baseline), candidate=_load(args.candidate),
            contract=contract, slice_doc=slice_doc, intent=intent,
            cadence=cadence,
            null_arm=_load(args.null_arm) if args.null_arm else None,
            null_basis=_load(args.null_basis) if args.null_basis else None,
            baseline_arm_sha256=digest(args.baseline),
            prod_baseline=(_load(args.prod_baseline)
                          if args.prod_baseline else None))
    except DailyGateError as exc:
        print(f"MEASUREMENT ERROR: {exc}")
        return 3
    except null.NullArmError as exc:
        print(f"MEASUREMENT ERROR: null arm: {exc}")
        return 3
    for line in render(verdict):
        print(line)
    if getattr(args, "out", None):
        args.out.write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n")
    # 0 PASS / 2 REFUSE as before; 3 UNADJUDICATED, distinct on purpose — a
    # caller that treats "not 0" as "refused" is still safe (nothing ships),
    # but one that wants to tell "the pack cost something" from "we did not
    # measure the pack" now can, without parsing the text.
    return {"PASS": 0, "UNADJUDICATED": 3}.get(verdict["verdict"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
