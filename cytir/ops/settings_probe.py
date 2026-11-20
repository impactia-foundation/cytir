"""Settings probe: record every effective value a live entry point reads, and diff two trees.

Task 696 slice 4. The review of #563/#567 proved "no value changed" by running one probe per live
entry point (a unit's own interpreter, `Environment=` lines, wrapper exports and working
directory) against `main`, step A and step C. This is that probe, kept in the repo so the
orchestrator can re-run "13 entry points x before/after" on the box before and after installing
`/etc/aw/operator.json`.

    python3 cytir/ops/settings_probe.py probe <tree-app-root> <cwd> <out.json>
    python3 cytir/ops/settings_probe.py compare <out-dir> <baseline-label> <label>...

`probe` imports the modules of `<tree-app-root>` (never a live checkout: pass a copy), changes to
`<cwd>` first, and writes `<out.json>`; it creates nothing else, and records every directory
that appeared in `<cwd>`. A value that cannot be read is recorded as `!! <ExceptionName>: ...`
(so a refusal by name is a finding, not a crash). The file is a script on purpose: it must run
against an OLD tree that has no `cytir/ops/settings_probe.py` of its own, and it takes nothing
from that tree but the values it reads.

`compare` reads `<out-dir>/<entry>.<label>.json` for every entry point and prints, per label, the
number of values that differ from the baseline label (tree root paths normalised away).
"""
from __future__ import annotations

import glob
import importlib
import json
import os
import sys


def _probe(tree: str, cwd: str, out: str) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [tree] + [p for p in sys.path if os.path.abspath(p or ".") != here]
    os.chdir(cwd)
    before = sorted(os.listdir(cwd))
    result = {"_python": sys.executable, "_cwd": os.getcwd(), "_tree": tree,
              "_env_AW": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("AW_", "CYTIR_"))}}

    def rec(key, fn):
        try:
            v = fn()
            if isinstance(v, (set, tuple)):
                v = list(v)
            json.dumps(v, default=str)
            result[key] = v
        except BaseException as exc:  # noqa: BLE001  (a refusal is a finding)
            result[key] = f"!! {type(exc).__name__}: {str(exc)[:300]}"

    def mod(name):
        return importlib.import_module(name)

    rec("paths.CORPUS_ROOT", lambda: str(mod("cytir.core.paths").CORPUS_ROOT))
    rec("promote_queue.DEFAULT_QUEUE_DIR", lambda: str(mod("cytir.index.promote_queue").DEFAULT_QUEUE_DIR))
    rec("promote_queue.PROD_LOCK_HOST", lambda: mod("cytir.index.promote_queue").PROD_LOCK_HOST)
    rec("promote_queue.PROD_LOCK_PATH", lambda: mod("cytir.index.promote_queue").PROD_LOCK_PATH)
    rec("compose.DEFAULT_QUEUE_DIR", lambda: str(mod("cytir.index.compose").DEFAULT_QUEUE_DIR))
    rec("write_lock.lock_location", lambda: list(mod("cytir.ops.write_lock").lock_location()))
    rec("stage.DEFAULT_LEGACY_LOCK_DIR", lambda: mod("cytir.embed.stage").DEFAULT_LEGACY_LOCK_DIR)
    rec("supersede.twin_config.ssh_host", lambda: mod("cytir.index.supersede").twin_config()["ssh_host"])
    rec("model_assets.lid.candidates", lambda: [
        str(p) for p in next(v for v in vars(mod("cytir.embed.model_assets")).values()
                             if type(v).__name__ == "ModelAsset").search_paths()])
    rec("crawl.UA", lambda: mod("cytir.acquire.crawl").UA)
    rec("crawl.HEADERS", lambda: mod("cytir.acquire.crawl").HEADERS)
    rec("crawl.DEFAULT_L0_ROOT", lambda: mod("cytir.acquire.crawl").DEFAULT_L0_ROOT)
    rec("crawl.DEFAULT_L1_ROOT", lambda: mod("cytir.acquire.crawl").DEFAULT_L1_ROOT)
    rec("l0_store.DEFAULT_ROOT", lambda: mod("cytir.acquire.l0_store").DEFAULT_ROOT)
    rec("corpus.store.DEFAULT_DEST", lambda: mod("cytir.corpus.store").DEFAULT_DEST)
    rec("daily.W", lambda: str(mod("cytir.news.daily").W))
    rec("daily.R", lambda: str(mod("cytir.news.daily").R))
    rec("daily.RELEASES_ROOT", lambda: str(mod("cytir.news.daily").RELEASES_ROOT))
    rec("daily.QUEUE_DIR", lambda: str(mod("cytir.news.daily").QUEUE_DIR))
    rec("daily._STABLE_ENV", lambda: dict(mod("cytir.news.daily")._STABLE_ENV))
    rec("daily.day_env-minus-os", lambda: {
        k: v for k, v in mod("cytir.news.daily").day_env("2026-10-03").items() if os.environ.get(k) != v})
    rec("hn_daily.W", lambda: str(mod("cytir.news.hn_daily").W))
    rec("hn_daily.day_env-minus-os", lambda: {
        k: v for k, v in mod("cytir.news.hn_daily").day_env("2026-10-03").items() if os.environ.get(k) != v})
    rec("coverage_backfill.STATE_DIR", lambda: str(mod("cytir.news.coverage_backfill").STATE_DIR))
    rec("published_page_date.DEFAULT_WORK", lambda: mod("cytir.news.published_page_date").DEFAULT_WORK)
    rec("growth_pending_compose.queue_dir", lambda: (
        sys.path.insert(1, os.path.join(tree, "pipeline")),
        str(getattr(mod("cytir.ops.growth_pending_compose"), "DEFAULT_QUEUE_DIR", None)
            or mod("cytir.index.compose").DEFAULT_QUEUE_DIR))[1])

    def tier2():
        t2 = mod("cytir.sources.families.news_tier2_apply")
        daily = mod("cytir.news.daily")
        env = t2.Env(daily.day_env("2026-10-03"))
        calls = []

        class _Done:
            returncode, stdout, stderr = 0, "", ""

        real = t2.subprocess.run
        t2.subprocess.run = lambda *a, **k: (calls.append([str(x) for x in a[0]]), _Done())[1]
        try:
            t2.step_durability(env)
        except BaseException as exc:  # noqa: BLE001
            calls.append(f"!! {type(exc).__name__}: {exc}")
        finally:
            t2.subprocess.run = real
        box = rsh = None
        for c in calls:
            if isinstance(c, list) and "--box" in c:
                box, rsh = c[c.index("--box") + 1], c[c.index("--rsh") + 1]
        return {"work": str(env.work), "qpy": env.qpy, "langgate_py": env.langgate_py,
                "prod_lock_host": env.prod_lock_host, "box": box, "rsh": rsh}

    rec("news_tier2_apply.Env", tier2)
    def direct_compose():
        # task 2116: the driver moved into the package; a tree cut before that still has the script
        try:
            return mod("cytir.sources.refresh_direct.direct_compose")
        except ModuleNotFoundError:
            sys.path.insert(1, os.path.join(tree, "pipeline"))
            return mod("refresh_lane_direct_compose")

    rec("refresh_lane_direct_compose", lambda: {
        "QUEUE_DIR": str(direct_compose().QUEUE_DIR),
        "QPY": mod("cytir.news.daily")._STABLE_ENV["AW_QPY"]})
    result["_cwd_created"] = sorted(set(os.listdir(cwd)) - set(before))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, sort_keys=True, default=str)
    print("ok", out)


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k != "_env_AW":
            out.update(_flat(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def _normalised(doc: dict) -> dict:
    """Flattened values with the probed tree's own root path replaced by `<TREE>`."""
    tree = doc.get("_tree", "").rstrip("/")
    text = json.dumps({k: v for k, v in doc.items() if not k.startswith("_")})
    if tree:
        text = text.replace(tree, "<TREE>")
    return _flat(json.loads(text))


def compare(out_dir: str, baseline: str, labels: list[str]) -> dict:
    """{entry: {label: [(key, baseline_value, value), ...]}} for every entry point in `out_dir`."""
    entries = sorted({os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(out_dir, "*.json"))})
    report: dict = {}
    for entry in entries:
        base_path = os.path.join(out_dir, f"{entry}.{baseline}.json")
        if not os.path.exists(base_path):
            continue
        base = _normalised(json.load(open(base_path)))
        report[entry] = {}
        for label in labels:
            path = os.path.join(out_dir, f"{entry}.{label}.json")
            if not os.path.exists(path):
                report[entry][label] = None
                continue
            doc = json.load(open(path))
            cur = _normalised(doc)
            diffs = [(k, base.get(k), cur.get(k)) for k in sorted(set(base) | set(cur))
                     if base.get(k) != cur.get(k)]
            report[entry][label] = {"created": doc.get("_cwd_created", []), "diffs": diffs}
    return report


def _print_report(report: dict, labels: list[str], verbose: bool) -> None:
    for entry, per in report.items():
        print(f"\n######## {entry}")
        for label in labels:
            r = per.get(label)
            if r is None:
                print(f"  [{label}] missing")
                continue
            refused = sum(1 for _, _, v in r["diffs"] if isinstance(v, str) and v.startswith("!! SettingRefused"))
            print(f"  [{label}] differences={len(r['diffs'])} refused_by_name={refused} created_in_cwd={r['created']}")
            if verbose:
                for k, a, b in r["diffs"]:
                    print(f"     {k}\n        baseline: {str(a)[:160]}\n        {label}: {str(b)[:160]}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 4 and argv[0] == "probe":
        _probe(argv[1], argv[2], argv[3])
        return 0
    if len(argv) >= 4 and argv[0] == "compare":
        verbose = "-v" in argv
        rest = [a for a in argv[1:] if a != "-v"]
        _print_report(compare(rest[0], rest[1], rest[2:]), rest[2:], verbose)
        return 0
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
