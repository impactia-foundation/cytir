"""The live news refresh chain's two shell steps, in the package (task 1989; design task 1987).

  `cyt news stage`   fetch -> live consent gate -> aw_b1-native staging       (was news_refresh_stage.sh)
  `cyt news delta`   staged arms -> promote delta + aged-out deletes + plan   (was news_refresh_delta.sh)

Same inputs (the `AW_*` environment names), same command lines for every child process, same
receipt bytes: `stage_input_manifest.json`, `stage_manifest.json`, `baseline.json`,
`source_status.json`, `promote_plan.json`, `CHANGED_COUNT`. The six heredocs of the two scripts run
in process; the one that needs `qdrant_client` (`manifest`) is started under the `qpy` interpreter
as `python -m cytir.news.refresh_chain manifest ...`. The stage-manifest hash recipe lives ONCE, in
`manifest_payload_sha256`; the writer and the delta's verifier both call it.

Never writes the canonical twin or production: the stage writes staging collections, the delta
reads the twin and writes files under DELTA_DIR.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import glob
import hashlib
import json
import os
import pathlib
import subprocess
import sys

from cytir.core.paths import SettingRefused, SettingsError, load_settings
from cytir.news import source_queue

REPO_DEFAULT = pathlib.Path(__file__).resolve().parents[2]

STAGE_SCHEMA = "aw.news-stage.v2"
STAGE_INPUT_SCHEMA = "aw.news-stage-input.v2"
BASELINE_SCHEMA = "aw.news-delta-baseline.v1"
REPRESENTATION = "aw_b1-native"
#: the arms, in manifest order.
ARM_ORDER = ("chmedia", "swissinfo", "hn")
#: the delta has always read the local staging store at this address (the stage's own default).
DELTA_STAGING_URL = "http://localhost:6398"
#: consent literals: they land in the payload of every staged point (index data -- do not edit).
CONSENT_BASIS = "robots-live-resolve"
CONSENT_SOURCE = "consent_gate.py direct-crawl receipt"


class _Abort(Exception):
    """A step ended the script: `.rc` is the exit status `set -e` / `exit N` gave."""

    def __init__(self, rc: int, message: str = ""):
        super().__init__(message)
        self.rc = rc
        self.message = message


def _need(env, name: str, why: str) -> str:
    value = env.get(name)
    if not value:
        raise _Abort(1, f"{name}: {why}")
    return value


def _today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _setting(env, name: str) -> str:
    try:
        return str(getattr(load_settings(env), name))
    except (SettingRefused, SettingsError) as exc:
        raise _Abort(2, f"[REFUSED] {name}: {' '.join(str(exc).split())}") from exc


def _child_env(env, **extra) -> dict:
    merged = dict(env)
    merged["PYTHONDONTWRITEBYTECODE"] = "1"
    merged.update(extra)
    return merged


def _run(argv, env, *, capture: bool = False) -> str:
    """Run one child; a non-zero status ends the script with that status (`set -e`)."""
    sys.stdout.flush()
    result = subprocess.run([str(a) for a in argv], env=env,
                            stdout=subprocess.PIPE if capture else None, text=True)
    if result.returncode:
        raise _Abort(result.returncode)
    return result.stdout if capture else ""


# --------------------------------------------------------------------------- the one hash recipe

def _canonical_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def manifest_payload_sha256(manifest: dict) -> str:
    """sha256 of the manifest JSON WITHOUT its `manifest_payload_sha256` key."""
    body = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def write_stage_manifest(spool: pathlib.Path, inputs: dict, arms: dict, *, day: str, mode: str,
                         mrl_dim: int) -> dict:
    """Write `stage_manifest.json` (the hash key last) and return the manifest."""
    manifest = {"schema": STAGE_SCHEMA, "day": day, "mode": mode,
                "representation": REPRESENTATION, "arms": arms,
                "source_status": inputs.get("source_status", {}), "mrl_dim": mrl_dim}
    manifest["manifest_payload_sha256"] = manifest_payload_sha256(manifest)
    (spool / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_stage_manifest(path) -> list[str]:
    """Check the stage manifest (schema, representation, hash, green non-empty arms); return the
    arm collections in file order. Refuses by `SystemExit(message)`."""
    value = json.load(open(path))
    if value.get("schema") != STAGE_SCHEMA or value.get("representation") != REPRESENTATION:
        raise SystemExit("invalid stage manifest")
    expected = value.get("manifest_payload_sha256")
    if expected != manifest_payload_sha256(value):
        raise SystemExit("stage manifest hash mismatch")
    arms = list(value.get("arms", {}).values())
    for arm in arms:
        if arm.get("status") != "green" or arm.get("points_count", 0) < 1:
            raise SystemExit("stage arm is not green/nonempty")
    return [arm["collection"] for arm in arms]


# --------------------------------------------------------------------------- stage

def concat_enriched(spool: pathlib.Path) -> None:
    with (spool / "all.enriched.jsonl").open("w") as out:
        for name in ("news_rows.enriched.jsonl", "hn_rows.enriched.jsonl"):
            with (spool / name).open() as source:
                for line in source:
                    if line.strip():
                        out.write(line)


def write_stage_input(passages_ok: pathlib.Path, spool: pathlib.Path, day: str, routes: dict,
                      status_hosts: list[str]) -> dict:
    """Bucket the consent-gated passages by arm; write `gated_<arm>.jsonl` and
    `stage_input_manifest.json`. sitemap_ingest's resumable state distinguishes an unreachable
    source from an empty feed: that evidence rides in `source_status`."""
    buckets = collections.defaultdict(list)
    for line in passages_ok.open():
        row = json.loads(line)
        label = next((candidate for suffix, candidate in routes.items()
                      if suffix in row.get("domain", "")), "hn")
        buckets[label].append(line)
    try:
        state = json.loads((spool / "state.json").read_text()).get("hosts", {})
    except (OSError, json.JSONDecodeError):
        state = {}
    source_status = {
        domain: {"status": state.get(domain, {}).get("status", "missing"),
                 "note": state.get(domain, {}).get("note", "")}
        for domain in status_hosts
    }
    try:
        hn_receipt = json.loads((spool / "hn_source_receipt.json").read_text())
    except (OSError, json.JSONDecodeError):
        hn_receipt = None
    hn_completeness = hn_receipt.get("completeness") if hn_receipt else None
    source_status["hn"] = {
        "status": ("complete" if hn_completeness == "time-window-complete"
                   else "partial" if hn_receipt else "missing-receipt"),
        "receipt": hn_receipt,
    }
    manifest = {"schema": STAGE_INPUT_SCHEMA, "day": day, "arms": {},
                "source_status": source_status}
    for label in ARM_ORDER:
        path = spool / f"gated_{label}.jsonl"
        path.write_text("".join(buckets[label]))
        manifest["arms"][label] = {"source": str(path), "input_rows": len(buckets[label])}
    (spool / "stage_input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_stage_manifest(spool: pathlib.Path, url: str, day: str, mode: str, mrl_dim: int,
                         client=None) -> dict:
    """Stage step 11: read each non-empty arm's collection from staging, write the manifest."""
    if client is None:
        from cytir.core.qdrant import qdrant_connect  # noqa: PLC0415 -- the one client factory
        client = qdrant_connect(url, timeout=600, check_compatibility=False)
    from cytir.news import stage_contract as NSC
    inputs = json.loads((spool / "stage_input_manifest.json").read_text())
    arms = {}
    for label, item in inputs["arms"].items():
        collection = NSC.news_stage_collection_name(day, label, mrl_dim)
        source = pathlib.Path(item["source"])
        if not source.stat().st_size:
            continue
        info = client.get_collection(collection)
        arms[label] = {**item,
                       "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                       "collection": collection, "points_count": info.points_count,
                       "status": str(info.status).split(".")[-1].lower()}
    return write_stage_manifest(spool, inputs, arms, day=day, mode=mode, mrl_dim=mrl_dim)


def run_stage(env=None) -> int:
    env = dict(os.environ if env is None else env)
    try:
        return _stage(env)
    except _Abort as stop:
        if stop.message:
            print(stop.message, file=sys.stderr)
        return stop.rc


def _stage(env) -> int:
    now = env.get("AW_NOW") or _today()
    mode = env.get("AW_MODE") or "daily"
    months = env.get("AW_NEWS_FRESH_MONTHS") or "12"
    repo = pathlib.Path(env.get("AW_RELEASE_DIR") or REPO_DEFAULT)
    work = _setting(env, "news_work")
    lg = _setting(env, "langgate_py")
    app_pp = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    release_manifest = _need(env, "AW_NEWS_RELEASE_MANIFEST", "set immutable release manifest path")
    release_manifest_sha = _need(env, "AW_NEWS_RELEASE_MANIFEST_SHA256",
                                 "set immutable release manifest SHA-256")
    release_sha = _need(env, "AW_NEWS_RELEASE_SHA", "set immutable release git SHA")
    langgate_req_sha = _need(env, "AW_NEWS_LANGGATE_REQUIREMENTS_SHA256",
                             "set langgate requirements SHA-256")
    qpy = _setting(env, "qpy")
    staging_url = env.get("AW_STAGING_URL") or "http://localhost:6398"
    mrl_dim = env.get("AW_NEWS_MRL_DIM") or "128"
    recent_cap = env.get("AW_RECENT_CAP") or "1500"
    host_rate = env.get("AW_HOST_RATE") or "1.5"
    global_rate = env.get("AW_GLOBAL_RATE") or "6"
    workers = env.get("AW_WORKERS") or "3"
    max_child_sm = env.get("AW_MAX_CHILD_SM") or "1000"
    spool = pathlib.Path(work) / f"stage-{now}"
    queue = env.get("AW_NEWS_QUEUE") or str(source_queue.queue_path(repo))
    host_meta = env.get("AW_NEWS_HOST_META") or str(repo / source_queue.host_meta_relpath())
    consent_cache = env.get("AW_CONSENT_CACHE") or str(pathlib.Path(work) / "consent-cache")
    reuse_retained = env.get("AW_REUSE_RETAINED_SOURCE") or "0"
    resume = env.get("AW_NEWS_RESUME") or "0"

    if env.get("AW_LANGGATE_OPTIONAL") == "1":
        raise _Abort(2, "ABORT: AW_LANGGATE_OPTIONAL=1 is forbidden for news indexing")
    mod = _child_env(env, PYTHONPATH=app_pp)
    _run(["python3", "-m", "cytir.news.release_preflight", "check",
          "--release-dir", repo, "--release-manifest", release_manifest,
          "--expected-git-sha", release_sha,
          "--expected-release-manifest-sha256", release_manifest_sha,
          "--langgate-python", lg,
          "--expected-langgate-requirements-sha256", langgate_req_sha], mod)

    for sub in ("raw", "gated", "pages"):
        (spool / sub).mkdir(parents=True, exist_ok=True)
    pathlib.Path(consent_cache).mkdir(parents=True, exist_ok=True)

    if reuse_retained == "1":
        for retained in ("news_rows.enriched.jsonl", "hn_rows.enriched.jsonl"):
            path = spool / retained
            if not (path.exists() and path.stat().st_size > 0):
                raise _Abort(2, f"ABORT: retained source missing/nonempty: {path}")
        print(f"REUSE retained source rows from {spool}; consent is re-resolved below")
    else:
        resume_args: list = []
        if resume == "1":
            resume_args = ["--resume"]
            print(f"resuming prior same-day sitemap state in {spool}")
        recapture_args: list = []
        if env.get("AW_NEWS_TWIN_URL") and env.get("AW_NEWS_TWIN_COLLECTION"):
            recapture_args = ["--recapture-twin-url", env["AW_NEWS_TWIN_URL"],
                              "--recapture-collection", env["AW_NEWS_TWIN_COLLECTION"]]
        else:
            print("recapture-overhang: no AW_NEWS_TWIN_URL/_COLLECTION -> fetch-time skip OFF")
        _run([lg, "-m", "cytir.acquire.sitemap_ingest", "fetch",
              "--queue", queue, "--wave", "news",
              "--out", spool / "news_rows.jsonl", "--links-out", spool / "news_links.jsonl",
              "--state", spool / "state.json", "--raw-dir", spool / "raw",
              "--preserve-dir", spool / "pages", "--docs-out", spool / "news_documents.jsonl",
              "--freshness", "--window-months", months, "--now", now,
              "--max-per-domain", recent_cap, "--max-child-sitemaps", max_child_sm,
              "--workers", workers, "--host-rate", host_rate, "--rate", global_rate,
              "--direct-crawl", *recapture_args, *resume_args],
             _child_env(env, AW_INPROC_EXTRACT="1", PYTHONPATH=app_pp))
        _run([lg, "-m", "cytir.sources.hn.freshness_ingest",
              "--min-points", env.get("AW_HN_MIN_POINTS") or "40",
              "--since-days", env.get("AW_HN_SINCE_DAYS") or "7", "--now", now,
              "--out", spool / "hn_rows.jsonl", "--receipt", spool / "hn_source_receipt.json",
              "--host-meta", host_meta, "--preserve-dir", spool / "hn_linked"],
             _child_env(env, PYTHONPATH=app_pp))
        _run([lg, "-m", "cytir.news.enrich_tier1", spool / "news_rows.jsonl", host_meta,
              spool / "news_rows.enriched.jsonl"], _child_env(env, PYTHONPATH=app_pp))
        _run([lg, "-m", "cytir.news.enrich_tier1", spool / "hn_rows.jsonl", host_meta,
              spool / "hn_rows.enriched.jsonl"], _child_env(env, PYTHONPATH=app_pp))

    # A publisher transport failure is evidence of an incomplete source boundary, not an empty
    # feed: refuse to mint a partial stage manifest. AW_NEWS_EXPECTED_HOSTS narrows the boundary
    # for a run that deliberately covers one publisher (coverage backfill, task 676).
    expected_args = ["--expected-hosts", env["AW_NEWS_EXPECTED_HOSTS"]] \
        if env.get("AW_NEWS_EXPECTED_HOSTS") else []
    _run([lg, "-m", "cytir.news.stage_contract", "source-boundary",
          "--state", spool / "state.json", "--hn-receipt", spool / "hn_source_receipt.json",
          *expected_args], mod)
    # The staging database is a managed dependency: fail before projection/GPU work.
    _run(["python3", "-m", "cytir.news.stage_contract", "staging-ready", "--url", staging_url], mod)

    concat_enriched(spool)

    # Re-resolve at merge time: only passages_ok is eligible for projection and embedding.
    _run([lg, "-m", "cytir.consent.gate", "gate", "--passages", spool / "all.enriched.jsonl",
          "--out-dir", spool / "gated", "--cache-dir", consent_cache,
          "--direct-crawl", "--search-indexing"], mod)

    routes_json = _run(["python3", "-m", "cytir.news.stage_contract", "routes"], mod, capture=True)
    status_hosts = _run(["python3", "-m", "cytir.news.stage_contract", "status-hosts",
                         *expected_args], mod, capture=True)
    write_stage_input(spool / "gated" / "passages_ok.jsonl", spool, now,
                      json.loads(routes_json), status_hosts.split())

    for label in ARM_ORDER:
        source = spool / f"gated_{label}.jsonl"
        if not (source.exists() and source.stat().st_size > 0):
            continue
        coll = _run(["python3", "-m", "cytir.news.stage_contract", "collection-name",
                     "--day", now, "--label", label, "--mrl-dim", mrl_dim],
                    mod, capture=True).rstrip("\n")
        _run([qpy, "-m", "cytir.index.stage_pack_b1", "--in", source, "--collection", coll,
              "--qdrant-url", staging_url, "--mrl-dim", mrl_dim,
              "--consent-state", "resolved", "--consent-basis", CONSENT_BASIS,
              "--consent-source", CONSENT_SOURCE, "--consent-checked-at", now],
             _child_env(env, AW_FOLD_ACCENTS="1", PYTHONPATH=app_pp))

    _run([qpy, "-m", "cytir.news.refresh_chain", "manifest", spool, staging_url, now, mode, mrl_dim],
         mod)
    print(f"STAGE DONE mode={mode} day={now} mrl_dim={mrl_dim}; canonical twin/prod untouched")
    return 0


# --------------------------------------------------------------------------- delta

def write_baseline(twin_url: str, collection: str, out: pathlib.Path) -> None:
    """Refuse to build a handoff from an absent or unhealthy baseline: a stale one makes
    already-live rows look net-new and corrupts aged-out deletes."""
    try:
        from cytir.core.qdrant import qdrant_connect  # noqa: PLC0415 -- the one client factory
        info = qdrant_connect(twin_url, timeout=30, check_compatibility=False).get_collection(
            collection)
        status = getattr(info.status, "value", info.status)
        points = info.points_count
    except Exception as exc:
        raise SystemExit(f"canonical twin unavailable: {exc}")
    if status != "green":
        raise SystemExit("canonical twin is not green")
    if not isinstance(points, int) or isinstance(points, bool) or points < 1:
        raise SystemExit("canonical twin has no valid point count")
    receipt = {"schema": BASELINE_SCHEMA, "url": twin_url, "collection": collection,
               "status": status, "points_count": points}
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def write_source_status(stage_manifest: str, out: pathlib.Path) -> None:
    manifest = json.load(open(stage_manifest))
    out.write_text(json.dumps(
        {"stage_manifest": stage_manifest, "source_status": manifest.get("source_status", {})},
        indent=2, sort_keys=True) + "\n")


def write_promote_plan(delta_dir: str, add_modify: int, delete: int, now: str,
                       months: int) -> None:
    plan = {"delta_dir": delta_dir, "add_modify": add_modify, "delete": delete,
            "now": now, "retain_months": months,
            "dumps": sorted(os.path.basename(x) for x in glob.glob(delta_dir + "/*.netnew.jsonl.gz")),
            "delete_ids": "aged_out.delete-ids.txt",
            "news_point_dump_receipt": "twin_news_points.receipt.json",
            "successor_news_point_candidates": "news_point_candidates.next.jsonl",
            "source_status": "source_status.json",
            "baseline": "baseline.json",
            "tombstone_semantics": "explicit point ids only; retained in delete_ids"}
    (pathlib.Path(delta_dir) / "promote_plan.json").write_text(json.dumps(plan, indent=2) + "\n")


def run_delta(env=None) -> int:
    env = dict(os.environ if env is None else env)
    try:
        return _delta(env)
    except _Abort as stop:
        if stop.message:
            print(stop.message, file=sys.stderr)
        return stop.rc


def _delta(env) -> int:
    now = env.get("AW_NOW") or _today()
    months = env.get("AW_NEWS_RETAIN_MONTHS") or "36"
    delta_dir = _need(env, "DELTA_DIR", "set DELTA_DIR")
    repo = pathlib.Path(env.get("AW_RELEASE_DIR") or REPO_DEFAULT)
    pipe = repo / "pipeline"
    newsfresh = env.get("AW_NEWS_WORK") or _setting(env, "news_work")
    qpy = _setting(env, "qpy")
    twin = _need(env, "AW_NEWS_TWIN_URL", "set canonical twin URL")
    base = _need(env, "AW_NEWS_TWIN_COLLECTION", "set canonical twin collection")
    candidates = _need(env, "AW_NEWS_POINT_CANDIDATES", "set hash-pinned NEWS candidate registry")
    candidates_sha = _need(env, "AW_NEWS_POINT_CANDIDATES_SHA256",
                           "set candidate-registry SHA-256")
    stage_manifest = env.get("AW_STAGE_MANIFEST") or f"{newsfresh}/stage-{now}/stage_manifest.json"
    out_dir = pathlib.Path(delta_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    child = _child_env(env, AW_PIPE=str(pipe))
    mod = _child_env(env, AW_PIPE=str(pipe),
                     PYTHONPATH=str(repo) + (os.pathsep + env["PYTHONPATH"]
                                             if env.get("PYTHONPATH") else ""))

    write_baseline(twin, base, out_dir / "baseline.json")
    staging_colls = verify_stage_manifest(stage_manifest)
    if not staging_colls:
        raise _Abort(2, "no manifested staging arms")
    write_source_status(stage_manifest, out_dir / "source_status.json")
    for coll in staging_colls:
        print(f"[delta] dump_netnew {coll} vs {base}")
        _run([qpy, "-m", "cytir.index.dump_netnew",
              "--staging-url", DELTA_STAGING_URL, "--staging-collection", coll,
              "--base-url", twin, "--base-collection", base, "--digest-field", "content_digest",
              "--out", f"{delta_dir}/{coll}.netnew.jsonl.gz"], mod)

    counts = sorted(glob.glob(delta_dir + "/*.netnew.jsonl.gz.count"))
    counts = [c for c in counts if os.path.isfile(c)]
    if len(counts) != len(staging_colls):
        raise _Abort(2, f"expected {len(staging_colls)} delta receipts, found {len(counts)}")

    print(f"[delta] dump twin news points + window-filter (now={now} months={months})")
    dump_args = ["--qdrant-url", twin, "--collection", base, "--staging-url", DELTA_STAGING_URL,
                 "--candidate-registry", candidates,
                 "--expected-candidate-registry-sha256", candidates_sha,
                 "--out", f"{delta_dir}/twin_news_points.jsonl",
                 "--successor-candidates-out", f"{delta_dir}/news_point_candidates.next.jsonl",
                 "--receipt-out", f"{delta_dir}/twin_news_points.receipt.json"]
    for coll in staging_colls:
        dump_args += ["--staging-collection", coll]
    _run([qpy, "-m", "cytir.news.dump_points", *dump_args], mod)
    _run([qpy, "-m", "cytir.index.news_window_filter",
          "--in", f"{delta_dir}/twin_news_points.jsonl", "--now", now, "--months", months,
          "--delete-ids-out", f"{delta_dir}/aged_out.delete-ids.txt"], mod)

    add_mod = sum(int(json.load(open(f))["changed"]) for f in counts)
    try:
        deleted = (out_dir / "aged_out.delete-ids.txt").read_bytes().count(b"\n")
    except OSError:
        deleted = 0
    changed = add_mod + deleted
    (out_dir / "CHANGED_COUNT").write_text(f"{changed}\n")
    write_promote_plan(delta_dir, add_mod, deleted, now, int(months))
    print(f"[delta] add_modify={add_mod} delete={deleted} changed={changed} -> {delta_dir}")
    return 0


# --------------------------------------------------------------------------- entry

def _manifest_main(argv: list[str]) -> int:
    spool, url, day, mode, mrl_dim = argv
    manifest = build_stage_manifest(pathlib.Path(spool), url, day, mode, int(mrl_dim))
    print(json.dumps(manifest, sort_keys=True))
    return 0


def main(argv: list[str] | None = None, prog: str = "cyt news", env=None) -> int:
    """`cyt news stage` / `cyt news delta`; `manifest` is the step that needs qdrant_client."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["manifest"] and len(argv) == 6:
        return _manifest_main(argv[1:])
    parser = argparse.ArgumentParser(
        prog=prog, description="The live news refresh chain's steps (settings and inputs come "
                               "from the AW_* environment, as the scripts they replace read them).",
        epilog="other doors: `cyt news routine-apply [--dry-run]` (the armed daily apply, task 2379), "
               "`cyt news enrich-tier1`, `cyt news seal-source-evidence`, `cyt news truncated-cc-recovery`.")
    parser.add_argument("step", choices=("stage", "delta"),
                        help="stage: fetch -> consent gate -> staging collections + "
                             "stage_manifest.json;  delta: staged arms vs the canonical twin -> "
                             "DELTA_DIR (promote plan, aged-out deletes)")
    args = parser.parse_args(argv)
    return run_stage(env) if args.step == "stage" else run_delta(env)


if __name__ == "__main__":
    raise SystemExit(main(prog="python -m cytir.news.refresh_chain"))
