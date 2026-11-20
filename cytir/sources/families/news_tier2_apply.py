#!/usr/bin/env python3
"""Drive one daily-news pack from a sealed spool to an armed tier-2 (ADR-098)
twin-first apply, one named step at a time.

Moved from the task 598 script `news_tier2_apply.sh` (task 646 S4 / task 671:
"nothing in production runs from tasks/"). A thin shim at the old path execs
this module's CLI unchanged, passing argv and env through untouched -- every
caller still invoking the old path (the daily timer's own
`news_daily_pack_and_queue.py::run_pack`, and any operator running
task598's runbook by hand) keeps working. See
`task 671` for the step-by-step characterization
this port was built and tested against, and its own evidence that every
step's LOGIC is unchanged -- this file is a faithful line-for-line
translation of the bash `case` statement into a Python subcommand
dispatcher, not a redesign. Every step is read-only or additive-immutable
except `apply-armed`, which is the only form that writes to a mirror. Run on
the build box.

Usage: python3 news_tier2_apply.py <step>
Steps: delta staged-ids seal-stages pack durability source-evidence
       baseline-twin baseline-prod baseline-refs gate-baseline preflight-dry
       mutation-plan intent runtime verify apply-armed

Two groups of inputs are supplied by the operator rather than pinned here,
for the same reason the bash version left them out (a second copy of a
value that lives somewhere else is how the two go out of step):
  AW_NEWS_CONFIG   the deployed news-refresh `current.env`, needed by
                   `delta`, `pack` and `source-evidence`
  AW_QVEC_CACHE + AW_QVEC_QUERY_SOURCE_MANIFEST
                   the ADR-043 gate query-vector cache and the task370
                   manifest that binds it, needed by `gate-baseline`,
                   `preflight-dry` and `apply-armed`
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Started by path (the sealed release runs this file as a script): the repo root first.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cytir.core.paths import operator_settings  # noqa: E402
from cytir.core import internal_http as IH  # noqa: E402 -- the one door to the operator's own services (consent step 5, task 1673)
from cytir.index import prod_guard as PG  # noqa: E402 -- no prod scroll, no prod replay (2026-10-07)

#: The lane's stage-environment defaults: every name `lane_value` is asked
#: for, with the `${NAME:-default}` value the lane's cron script
#: (news_refresh_cron.sh, lines 71-72, 79, 91) carries for it.
#: Copied byte-for-byte (task 2121); this table replaces the old text-read of
#: that script so the script can be retired.
LANE_DEFAULTS = {
    "AW_NEWS_FRESH_MONTHS": "12",
    "AW_NEWS_RETAIN_MONTHS": "36",
    "AW_RECENT_CAP": "1500",
    "AW_NEWS_RESUME": "0",
}

#: Publisher-only, per ADR-098 Ruling 2: the HN arm cannot satisfy
#: `news_article.original`, so it is excluded by regeneration rather than by
#: a waiver. Unchanged from the bash `ARMS=(chmedia swissinfo)`.
ARMS = ("chmedia", "swissinfo")


class ApplyDriverError(RuntimeError):
    """Raised for a refusal this driver itself detects (not a called
    subprocess's own failure, which surfaces as CalledProcessError)."""


def _required(env: dict, name: str, message: str | None = None) -> str:
    value = env.get(name)
    if not value:
        raise SystemExit(message or f"set {name}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Env:
    """The bash script's own top-of-file variable resolution, unchanged."""

    def __init__(self, environ: dict | None = None):
        e = dict(environ if environ is not None else os.environ)
        self.environ = e
        self.day = e.get("AW_NOW", "2026-08-28")
        self.work = Path(e.get("AW_NEWS_WORK") or operator_settings().news_work)
        self.release_dir = Path(_required(
            e, "AW_RELEASE_DIR", "set AW_RELEASE_DIR to the immutable release"))
        self.release_manifest_sha256 = _required(e, "AW_NEWS_RELEASE_MANIFEST_SHA256")
        self.spool = self.work / f"stage-{self.day}"
        self.ev = Path(e.get("AW_TASK598_EV", str(self.work / "task598-evidence")))
        self.pack_dir = self.work / f"pack-{self.day.replace('-', '')}-task598"
        self.delta_dir = Path(e.get("DELTA_DIR", str(self.work / "delta-task598")))
        self.qpy = e.get("AW_QPY") or operator_settings().qpy
        self.langgate_py = e.get("AW_NEWS_LANGGATE_PY") or operator_settings().langgate_py
        self.staging_url = e.get("AW_STAGING_URL", "http://localhost:6398")
        self.mrl_dim = e.get("AW_NEWS_MRL_DIM", "128")
        self.twin_url = e.get("AW_NEWS_TWIN_URL", "http://127.0.0.1:6444")
        self.twin_collection = e.get("AW_NEWS_TWIN_COLLECTION", "aw_e3")
        self.prod_url = e.get("AW_PROD_URL", "http://127.0.0.1:6334")
        self.prod_collection = e.get("AW_PROD_COLLECTION", "aw_prod")
        self.prod_lock_host = e.get("AW_PROD_LOCK_HOST") or operator_settings().prod_lock_host
        self.pack_name = e.get("AW_PACK_NAME", f"news598-{self.day.replace('-', '')}")
        self.news_config = e.get("AW_NEWS_CONFIG", "")
        self.arms = self._resolve_arms()

    def subprocess_env(self, **extra: str) -> dict:
        """Every python this script starts inherits the bytecode ban -- the
        one place the bash version built an env dict of its own re-stated
        it there rather than trusting inheritance (task 501 §6)."""
        merged = dict(self.environ)
        merged["PYTHONDONTWRITEBYTECODE"] = "1"
        merged.update(extra)
        return merged

    def module_env(self, **extra: str) -> dict:
        """`subprocess_env` plus THIS release on PYTHONPATH, for `python3 -m cytir.<module>` starts
        (task 739): the module resolves from the pinned release's own tree, never from a foreign
        editable install, with the working directory left exactly as the by-path call had it."""
        merged = self.subprocess_env(**extra)
        old = merged.get("PYTHONPATH")
        merged["PYTHONPATH"] = str(self.release_dir) + (os.pathsep + old if old else "")
        return merged

    def _resolve_arms(self) -> tuple:
        """`ARMS` unless AW_NEWS_EXPECTED_HOSTS narrows the run to some publishers
        (the coverage backfill): then only the arms those hosts route to, asked of
        `news_stage_contract.py arms` -- the stage only creates a collection for an
        arm that has rows, so scrolling an arm nobody fetched would be a 404.
        Unset (the daily run) keeps both arms, and a missing one still refuses."""
        hosts = self.environ.get("AW_NEWS_EXPECTED_HOSTS", "").strip()
        if not hosts:
            return ARMS
        result = subprocess.run(
            ["python3", "-m", "cytir.news.stage_contract",
             "arms", "--expected-hosts", hosts],
            capture_output=True, text=True, env=self.module_env())
        arms = tuple(result.stdout.split())
        if result.returncode or not arms:
            raise SystemExit(f"CYTIR_NEWS_EXPECTED_HOSTS={hosts!r} resolves to no arm: "
                             f"{(result.stderr or result.stdout).strip()}")
        return arms

    def collection_of(self, arm: str) -> str:
        result = subprocess.run(
            ["python3", "-m", "cytir.news.stage_contract",
             "collection-name", "--day", self.day, "--label", arm,
             "--mrl-dim", self.mrl_dim],
            capture_output=True, text=True, check=True, env=self.module_env())
        return result.stdout.strip()

    # -----------------------------------------------------------------
    # The lane's own resolved stage environment -- never this driver's
    # guess at it. Resolution order is the lane's own:
    #   1. an explicit override in this driver's environment,
    #   2. the deployed `current.env` the scheduler sources,
    #   3. otherwise the default in LANE_DEFAULTS (copied from the lane script).
    # -----------------------------------------------------------------
    def lane_value(self, name: str) -> str:
        value = self.environ.get(name)
        if value:
            return value
        config = self.news_config
        if not config or not os.path.isfile(config):
            raise SystemExit(
                f"{name}: cannot read the lane config {config!r}. Set "
                "AW_NEWS_CONFIG to the deployed news-refresh current.env "
                "(the scheduler's own copy lives on its host), or set "
                f"{name} explicitly. The declared stage environment is "
                "evidence; it is not guessed.")
        found = None
        with open(config, encoding="utf-8") as handle:
            for line in handle:  # last assignment wins, as `source` does
                line = line.strip()
                if line.startswith(f"{name}="):
                    found = line.split("=", 1)[1]
        if found is not None:
            return found
        if name not in LANE_DEFAULTS:
            raise SystemExit(
                f"{name}: no default in LANE_DEFAULTS and not in the lane "
                f"config {config!r}. Set {name} explicitly.")
        return LANE_DEFAULTS[name]

    # -----------------------------------------------------------------
    # The ADR-043 gate inputs are supplied, not pinned here (see module
    # docstring). Returns (cache_path, qvec_path, cache_sha, qvec_sha).
    # -----------------------------------------------------------------
    def resolve_gate_inputs(self) -> tuple[str, str, str, str]:
        cache = _required(self.environ, "AW_QVEC_CACHE",
                          "set CYTIR_QVEC_CACHE to the gate query-vector cache")
        qvec = _required(self.environ, "AW_QVEC_QUERY_SOURCE_MANIFEST",
                         "set CYTIR_QVEC_QUERY_SOURCE_MANIFEST to the task370 "
                         "query-source manifest that binds it")
        cache_sha = sha256_file(Path(cache))
        qvec_sha = sha256_file(Path(qvec))
        manifest = json.loads(Path(qvec).read_text(encoding="utf-8"))
        declared = manifest.get("cache_sha256")
        if declared != cache_sha:
            raise SystemExit(
                f"qvec cache is not the one this manifest binds: "
                f"{cache_sha} != {declared}")
        gate_path = self.release_dir / "cytir/gates/authority.py"
        served = hashlib.sha256(gate_path.read_bytes()).hexdigest()
        bound = (manifest.get("gate_authority") or {}).get("sha256")
        if served != bound:
            raise SystemExit(
                "qvec query-source manifest was derived from a different "
                f"cytir/gates/authority.py than this release ships ({bound} != "
                f"{served}); its query set cannot match the gate's, and "
                "the arm would report the gap as cache_miss")
        print(f"gate inputs OK: {manifest['query_count']} queries, "
             f"cache {cache_sha[:12]}")
        return cache, qvec, cache_sha, qvec_sha


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_delta(env: Env) -> None:
    retain_months = env.lane_value("AW_NEWS_RETAIN_MONTHS")
    run_env = env.subprocess_env(
        AW_RELEASE_DIR=str(env.release_dir), AW_NEWS_WORK=str(env.work),
        AW_NOW=env.day, AW_NEWS_RETAIN_MONTHS=retain_months,
        DELTA_DIR=str(env.delta_dir), AW_NEWS_TWIN_URL=env.twin_url,
        AW_NEWS_TWIN_COLLECTION=env.twin_collection,
        AW_NEWS_POINT_CANDIDATES=_required(env.environ, "AW_NEWS_POINT_CANDIDATES"),
        AW_NEWS_POINT_CANDIDATES_SHA256=_required(
            env.environ, "AW_NEWS_POINT_CANDIDATES_SHA256"))
    subprocess.run([sys.executable, "-m", "cytir.cli", "news", "delta"],
                   env={**run_env, "AW_QPY": env.qpy, "PYTHONPATH": os.pathsep.join(
                       p for p in (str(env.release_dir), run_env.get("PYTHONPATH")) if p)},
                   check=True)


def step_staged_ids(env: Env) -> None:
    env.ev.mkdir(parents=True, exist_ok=True)
    for arm in env.arms:
        collection = env.collection_of(arm)
        ids = list(_scroll_ids(env.staging_url, collection))
        out_path = env.ev / f"staged-ids.{arm}.txt"
        out_path.write_text("".join(f"{pid}\n" for pid in ids))
        print(f"{collection}: {len(ids)}", file=sys.stderr)
        print(len(ids))


def _scroll_ids(url: str, collection: str):
    offset = None
    while True:
        body = {"limit": 20000, "with_payload": False, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        PG.check(url, "POST", f"/collections/{collection}/points/scroll")
        request = urllib.request.Request(
            f"{url.rstrip('/')}/collections/{collection}/points/scroll",
            method="POST", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with IH.urlopen(request, timeout=300) as response:
            result = json.loads(response.read())["result"]
        for point in result["points"]:
            yield point["id"]
        offset = result.get("next_page_offset")
        if offset is None:
            break


def step_seal_stages(env: Env) -> None:
    for arm in env.arms:
        subprocess.run(
            ["python3", "-m", "cytir.news.seal_lane_stages",
             "--spool", str(env.spool), "--artifact-root", str(env.work),
             "--arm", arm, "--out", str(env.spool / "preservation" / arm),
             "--day", env.day, "--producer-version", "task452.v1",
             "--chunk-policy", str(env.release_dir /
                                   "data/chunk-policy-legacy-180w-o30.v1.json"),
             "--staged-point-ids", str(env.ev / f"staged-ids.{arm}.txt"),
             "--gated-rows", str(env.spool / f"gated_{arm}.jsonl")],
            env=env.module_env(), check=True)


def _news_retention_cutoff(day: str, months: str) -> str:
    d = datetime.date.fromisoformat(day)
    year, month = divmod((d.year * 12 + d.month - 1) - int(months), 12)
    return d.replace(year=year, month=month + 1).isoformat()


def step_pack(env: Env) -> None:
    # The aged-out cutoff is a deletion horizon: a value this driver invented
    # would tombstone the wrong points. Read from the lane, like everything
    # else.
    retain_months = env.lane_value("AW_NEWS_RETAIN_MONTHS")
    args = []
    for arm in env.arms:
        args += ["--staging-collection", env.collection_of(arm)]
    cutoff = _news_retention_cutoff(env.day, retain_months)
    result = subprocess.run(
        [env.qpy, str(env.release_dir / "cytir/sources/families/news.py"),
         "--staging-url", env.staging_url, *args,
         "--twin-url", env.twin_url, "--collection", env.twin_collection,
         "--aged-ids", str(env.delta_dir / "aged_out.delete-ids.txt"),
         "--cutoff", cutoff, "--out-dir", str(env.pack_dir)],
        capture_output=True, text=True, env=env.subprocess_env())
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    env.ev.mkdir(parents=True, exist_ok=True)
    (env.ev / "pack.json").write_text(result.stdout)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args)


def step_durability(env: Env) -> None:
    box_rsh = env.environ.get("AW_BOX_RSH") or operator_settings().box_rsh
    run_env = env.subprocess_env(AW_BOX_RSH=box_rsh)
    for arm in env.arms:
        subprocess.run(["bash", str(env.release_dir / "deploy/scripts/offbox_push_tree.sh"),
                        f"stage-{env.day}/pages"], env=run_env, check=True)
        subprocess.run(["bash", str(env.release_dir / "deploy/scripts/offbox_push_tree.sh"),
                        f"stage-{env.day}/preservation/{arm}"], env=run_env, check=True)
        subprocess.run(["bash", str(env.release_dir / "deploy/scripts/offbox_push_tree.sh"),
                        env.pack_dir.name], env=run_env, check=True)
        subprocess.run(
            ["python3", "-m", "cytir.acquire.offbox_durability",
             "--artifact-root", str(env.work),
             "--receipt", str(env.ev / f"durability.{arm}.json"),
             "--source-id", f"news-{env.day}-{arm}",
             "--box", env.environ.get("AW_BOX") or operator_settings().storagebox,
             "--rsh", box_rsh,
             "--remote-root", env.environ.get("AW_STORAGEBOX_ROOT", "aw-news-evidence"),
             "--tree", f"stage-{env.day}/pages=stage-{env.day}/pages",
             "--tree", f"stage-{env.day}/preservation/{arm}="
                      f"stage-{env.day}/preservation/{arm}",
             "--tree", f"{env.pack_dir.name}={env.pack_dir.name}"],
            env=env.module_env(AW_BOX_RSH=box_rsh), check=True)


def step_source_evidence(env: Env) -> None:
    fresh_months = env.lane_value("AW_NEWS_FRESH_MONTHS")
    recent_cap = env.lane_value("AW_RECENT_CAP")
    news_resume = env.lane_value("AW_NEWS_RESUME")
    print(f"lane env: fresh_months={fresh_months} recent_cap={recent_cap} "
         f"resume={news_resume}", file=sys.stderr)
    langgate_requirements_sha256 = _required(
        env.environ, "AW_NEWS_LANGGATE_REQUIREMENTS_SHA256")
    for arm in env.arms:
        manifest = json.loads((env.spool / "stage_manifest.json").read_text())
        arm_manifest = manifest["arms"][arm]
        collection = env.collection_of(arm)
        if arm_manifest["collection"] != collection:
            raise ApplyDriverError(
                f"stage manifest collection mismatch: {arm_manifest['collection']}")
        if arm_manifest["status"] != "green":
            raise ApplyDriverError(f"stage manifest status: {arm_manifest['status']}")
        staged = arm_manifest["points_count"]
        subprocess.run(
            [env.langgate_py, "-m", "cytir.news.seal_source_evidence",
             "--artifact-root", str(env.work),
             "--run", str(env.spool / "preservation" / arm / "RUN.json"),
             "--out", str(env.ev / "sources" / arm), "--release-dir", str(env.release_dir),
             "--release-manifest-sha256", env.release_manifest_sha256,
             "--spool", str(env.spool), "--source-id", f"news-{env.day}-{arm}",
             "--staged-collection", collection, "--staging-url", env.staging_url,
             "--durability-receipt", str(env.ev / f"durability.{arm}.json"),
             "--expect-count", f"vector_records={staged}",
             "--env", f"AW_NOW={env.day}", "--env", f"AW_NEWS_FRESH_MONTHS={fresh_months}",
             "--env", f"AW_RECENT_CAP={recent_cap}",
             "--env", f"AW_NEWS_RESUME={news_resume}",
             "--env", f"AW_NEWS_LANGGATE_PY={env.langgate_py}",
             "--env", f"AW_NEWS_LANGGATE_REQUIREMENTS_SHA256={langgate_requirements_sha256}"],
            env=env.module_env(), check=True)


def _baseline(env: Env, *, role: str, url: str, collection: str,
             expect_points: str, receipt: str, extra: tuple[str, ...] = ()) -> None:
    subprocess.run(
        ["python3", "-m", "cytir.index.seal_collection_membership", *extra,
         "--url", url, "--collection", collection, "--role", role,
         "--expect-points", expect_points, "--artifact-root", str(env.work),
         "--receipt", str(env.ev / receipt),
         "--evidence", str(env.ev / f"{receipt.rsplit('.', 1)[0]}.evidence.json"),
         "--batch", "20000", "--pace-seconds", "0.02"],
        env=env.module_env(), check=True)


def step_baseline_twin(env: Env) -> None:
    expect_points = _required(env.environ, "AW_EXPECT_POINTS")
    _baseline(env, role="twin", url=env.twin_url, collection=env.twin_collection,
             expect_points=expect_points, receipt="baseline-twin.json")


def step_baseline_prod(env: Env) -> None:
    expect_points = env.environ.get("AW_EXPECT_POINTS_PROD") or _required(
        env.environ, "AW_EXPECT_POINTS")
    # A full id scroll of prod scans the serving box (incident 2026-10-07): the seal tool refuses
    # it unless the operator names a reason in AW_PROD_BASELINE_SCROLL_REASON.
    reason = (env.environ.get("AW_PROD_BASELINE_SCROLL_REASON") or "").strip()
    _baseline(env, role="prod", url=env.prod_url, collection=env.prod_collection,
             expect_points=expect_points, receipt="baseline-prod.json",
             extra=("--allow-prod-scroll", reason) if reason else ())


def step_baseline_refs(env: Env) -> None:
    for role in ("twin", "prod"):
        receipt = env.ev / f"baseline-{role}.json"
        out = env.ev / f"baseline-{role}.ref.json"
        body = receipt.read_bytes()
        value = json.loads(body)
        ref = {
            "uri": f"artifact://{env.ev.name}/{receipt.name}",
            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
            "rows": value["points"], "points": value["points"],
            "logical_membership_sha256": value["logical_membership_sha256"],
        }
        out.write_text(json.dumps(ref, indent=2, sort_keys=True))
        print(out.read_text())


def step_mutation_plan(env: Env) -> None:
    ledger = env.ev / "preflight-dry-ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    plans = [row for row in rows if row.get("status") == "CONTENT-PACK-PLAN"]
    if len(plans) != 1:
        raise SystemExit(f"expected exactly one CONTENT-PACK-PLAN row, got {len(plans)}")
    proof = plans[-1]["proof"]
    twin = {k: v for k, v in proof["twin"].items() if k != "target"}
    prod = {k: v for k, v in proof["prod"].items() if k != "target"}
    if twin != prod:
        raise SystemExit("mirror plans differ; refusing to seal an intent")
    (env.ev / "mutation-plan.json").write_text(json.dumps(twin, indent=2, sort_keys=True))
    print(json.dumps({k: twin[k] for k in
                     ("rows", "existing_ids", "new_ids", "point_changed",
                      "superseded", "aged_tombstones", "expected_delta")},
                     indent=2, sort_keys=True))


def step_runtime(env: Env) -> None:
    sys.path.insert(0, str(env.release_dir))
    import cytir.index.apply as mod  # noqa: E402  -- task 646 S4: cytir/, not tasks-scripts

    def ref(uri: str, path: str) -> dict:
        body = Path(path).read_bytes()
        return {"uri": uri, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}

    rows = int(_required(env.environ, "ROWS"))
    art = _required(env.environ, "ART")
    sv = _required(env.environ, "SV")
    tp = _required(env.environ, "TP")
    runtime = {
        **mod.runtime_release_identity(env.release_dir),
        "pack": {"name": env.pack_name, "rows": rows,
                "artifact": ref("runtime://artifact", art),
                "staged_verification": ref("runtime://staged-verification", sv),
                "tombstone_artifact": ref("runtime://tombstone-pointers", tp)},
        "targets": {
            "twin": mod.target_binding("twin", env.twin_url, env.twin_collection),
            "prod": mod.target_binding("prod", env.prod_url, env.prod_collection)},
        "mutation_plan": json.loads((env.ev / "mutation-plan.json").read_text()),
    }
    text = json.dumps(runtime, indent=2, sort_keys=True)
    (env.ev / "runtime.json").write_text(text)
    print("\n".join(text.splitlines()[:5]))


def step_gate_baseline(env: Env) -> None:
    _cache, _qvec, cache_sha, qvec_sha = env.resolve_gate_inputs()
    (env.ev / "gate").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(env.release_dir))
    import cytir.index.apply as mod  # noqa: E402

    run_env = mod.gate_environment(env.release_dir, env.twin_url, env.twin_collection)
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
    # cytir/gates/content_pack_arm.py: moved from
    # task 351 run_content_pack_arm.py (task 677); shim left there.
    subprocess.run(
        [sys.executable,
         str(env.release_dir / "cytir/gates/content_pack_arm.py"),
         "--release-dir", str(env.release_dir), "--qdrant-url", env.twin_url,
         "--collection", env.twin_collection, "--representation", "current-b1",
         "--identity", f"current-b1:{env.pack_name}-preapply",
         "--source-sha256", _required(env.environ, "SV_SHA"),
         "--supersede-sha256", _required(env.environ, "TP_SHA"),
         "--cache", f"{cache_sha}={_cache}",
         "--qvec-query-source-manifest", f"{qvec_sha}={_qvec}",
         "--out-dir", str(env.ev / "gate/preapply-baseline"),
         "--arm-id", f"{env.pack_name}-preapply"],
        cwd=env.release_dir, env=run_env, check=True)


def step_preflight_or_apply(env: Env, step: str) -> None:
    _cache, _qvec, cache_sha, qvec_sha = env.resolve_gate_inputs()
    art = _required(env.environ, "ART")
    sv = _required(env.environ, "SV")
    tp = _required(env.environ, "TP")
    baseline_arm = _required(env.environ, "BASELINE_ARM")
    rows = _required(env.environ, "ROWS")
    staged_collection = _required(env.environ, "STAGED_COLLECTION")
    batch = env.environ.get("BATCH", "256")
    settle_timeout = env.environ.get("SETTLE_TIMEOUT", "21600")
    persisted_overlay = env.environ.get("AW_PERSISTED_OVERLAY")

    extra: list[str] = []
    run_env = env.module_env()
    if step == "apply-armed":
        extra = ["--arm", "APPLY-CONTENT-PACK"]
        run_env["AW_CONTENT_PACK_PRESERVATION_INTENT"] = str(env.ev / "intent.json")
        run_env["AW_CONTENT_PACK_PRESERVATION_PROFILE_REGISTRY"] = str(
            env.ev / "intent-evidence/source-preservation-profiles.v1.json")
        run_env["AW_ARTIFACT_ROOT"] = str(env.work)
        run_env["AW_CONTENT_PACK_PRESERVATION_DEEP"] = "1"

    (env.ev / "gate" / step).mkdir(parents=True, exist_ok=True)
    args = [env.qpy, "-m", "cytir.index.apply",
           "--pack", env.pack_name,
           "--artifact", art, "--artifact-sha256", sha256_file(Path(art)),
           "--rows", rows,
           "--staged-verification", sv, "--staged-verification-sha256", sha256_file(Path(sv)),
           "--staged-collection", staged_collection,
           "--tombstone-pointers", tp, "--tombstone-pointers-sha256", sha256_file(Path(tp)),
           "--twin-url", env.twin_url, "--prod-url", env.prod_url,
           "--twin-collection", env.twin_collection, "--prod-collection", env.prod_collection,
           "--prod-lock-host", env.prod_lock_host, "--batch", batch,
           "--settle-timeout", settle_timeout,
           "--ledger", str(env.ev / f"{step}-ledger.jsonl"), "--release-dir", str(env.release_dir),
           "--gate-dir", str(env.ev / "gate" / step),
           "--baseline-arm", baseline_arm,
           "--baseline-arm-sha256", sha256_file(Path(baseline_arm)),
           "--cache", f"{cache_sha}={_cache}",
           "--qvec-query-source-manifest", f"{qvec_sha}={_qvec}"]
    if persisted_overlay:
        args += ["--persisted-overlay", persisted_overlay]
    args += extra
    subprocess.run(args, env=run_env, check=True)


def step_intent(env: Env) -> None:
    sources = []
    for arm in env.arms:
        sources += ["--source", str(env.ev / "sources" / arm / "SOURCE.json")]
    subprocess.run(
        ["python3", "-m", "cytir.index.seal_content_pack_intent",
         "--artifact-root", str(env.work), "--out", str(env.ev / "intent-evidence"),
         "--intent", str(env.ev / "intent.json"),
         "--intent-id", f"task598-{env.pack_name}", "--release-dir", str(env.release_dir),
         "--profile-registry", str(env.release_dir /
                                   "data/source-preservation-profiles.v1.json"),
         "--pack-name", env.pack_name,
         "--pack-artifact", _required(env.environ, "ART"),
         "--staged-verification", _required(env.environ, "SV"),
         "--tombstone-pointers", _required(env.environ, "TP"),
         "--mutation-plan", str(env.ev / "mutation-plan.json"),
         "--target", f"twin={env.twin_url},{env.twin_collection},"
                     f"{env.ev / 'baseline-twin.ref.json'}",
         "--target", f"prod={env.prod_url},{env.prod_collection},"
                     f"{env.ev / 'baseline-prod.ref.json'}",
         *sources],
        env=env.module_env(), check=True)


def step_verify(env: Env) -> None:
    subprocess.run(
        ["python3", "-m", "cytir.index.content_pack_preservation",
         "--intent", str(env.ev / "intent.json"), "--runtime", str(env.ev / "runtime.json"),
         "--registry", str(env.release_dir / "data/source-preservation-profiles.v1.json"),
         "--artifact-root", str(env.work), "--deep"],
        env=env.module_env(), check=True)


STEPS = {
    "delta": step_delta,
    "staged-ids": step_staged_ids,
    "seal-stages": step_seal_stages,
    "pack": step_pack,
    "durability": step_durability,
    "source-evidence": step_source_evidence,
    "baseline-twin": step_baseline_twin,
    "baseline-prod": step_baseline_prod,
    "baseline-refs": step_baseline_refs,
    "mutation-plan": step_mutation_plan,
    "runtime": step_runtime,
    "gate-baseline": step_gate_baseline,
    "intent": step_intent,
    "verify": step_verify,
}


def run_step(step: str, env: Env) -> None:
    env.ev.mkdir(parents=True, exist_ok=True)
    if step in ("preflight-dry", "apply-armed"):
        step_preflight_or_apply(env, step)
        return
    handler = STEPS.get(step)
    if handler is None:
        print(f"unknown step: {step}", file=sys.stderr)
        raise SystemExit(2)
    handler(env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", help="usage: news_tier2_apply.py <step>")
    args = parser.parse_args(argv)
    env = Env()
    try:
        run_step(args.step, env)
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
