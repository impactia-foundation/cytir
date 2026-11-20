#!/usr/bin/env python3
"""flip_config_verify — task 522: after the flip, confirm the RUNNING api
reports the ratified serving config. Editing a compose file correctly is not
evidence it was applied to the right container and picked up (deployment-
doctrine "No Invisible State": "the running system can echo its effective
config so drift is a `curl | diff`, not a memory exercise").

`cytir/serve/app.py`'s `/health` now echoes `retrieval.effective_config()`
(task 522). This script is the one place that reads it back and diffs it
against `cytir/core/serving_config.py`'s ratified "256-pq8" dense policy and the
task-519 fusion-weights promotion (decisions.md's ADR-099 amendment note).
stdlib only (urllib), so it runs on any box without the serving deps —
matches the house style of the other flip-day tools.
(Task 2152: it needs numpy + qdrant_client only because `cytir.core.serving_config`
does -- the same dependency set it had before it moved into the package.)

Usage (flip-day, the flip-day runbook, step 3 — normally reached
through `deploy/scripts/flip-release.sh`, which passes `--gate-collection`
for you):
    python3 -m cytir.gates.flip_config_verify --url https://websear.ch \
        --gate-collection aw_e8

And the mirror, after a rollback (`deploy/scripts/flip-rollback.sh`):
    python3 -m cytir.gates.flip_config_verify --url https://websear.ch \
        --rollback

Two things task 590 added, both because the flip-day path exercised them:

  * `--gate-collection`. `/health` reports the ALIAS (`aw_prod`); a sealed
    flip-gate verdict is bound to the PHYSICAL collection a battery scored
    (`aw_e8`). Comparing the two refuses every flip, and the refusal reads
    like tampering. The release script proves `aw_prod -> aw_e8` against
    Qdrant and passes the physical name here.
  * `--rollback`. Reverting needs its own readback, for the same reason the
    forward direction does: an api left at `AW_MRL_DIM=256` over a 128-dim
    collection is the same outage, in the other direction (runbook §6).

Exit 0 only if every knob agrees with the ratified source; nonzero and a
labelled diff otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

from cytir.core import internal_http as IH
from cytir.core import serving_config as sc
from cytir.gates import flip_gate as fg

#: What production serves BEFORE the flip, and what a rollback returns it to.
#: Named here rather than typed into the rollback path, so the two directions
#: read their expectations out of the same module (`cytir/core/serving_config.py`)
#: -- engineering-doctrine 2, one home.
INCUMBENT_SERVING_CONFIG = "prod-128-int8"


def fetch_health(url: str, timeout: float = 10.0) -> dict:
    with IH.public_self_urlopen(f"{url.rstrip('/')}/health", timeout=timeout,
                                caller="flip-config-verify") as r:
        return json.loads(r.read().decode())


def check(health: dict, config_name: str = sc.FLIP_SERVING_CONFIG) -> list[str]:
    """Return a list of human-readable problems; empty means the running API
    reports exactly the ratified flip config."""
    problems: list[str] = []
    if health.get("status") != "ok":
        problems.append(f"status={health.get('status')!r}, not 'ok' — full "
                        f"response: {json.dumps(health)}")
        return problems  # a degraded /health carries no config to check

    cfg = health.get("config")
    if cfg is None:
        problems.append(
            "response has no 'config' key — is this an older aw-api build "
            "that predates task 522's /health change?")
        return problems

    want_env = sc.flip_deploy_env(config_name)
    want = {
        "mrl_dim": int(want_env["AW_MRL_DIM"]),
        "dense_rescore": want_env["AW_DENSE_RESCORE"] == "1",
        "dense_oversampling": float(want_env["AW_DENSE_OVERSAMPLING"]),
        "dense_ignore_quant": want_env["AW_DENSE_IGNORE_QUANT"] == "1",
    }
    for key, value in want.items():
        got = cfg.get(key)
        if got != value:
            problems.append(f"config.{key} = {got!r}, ratified value is "
                            f"{value!r} (cytir/core/serving_config.py, "
                            f"{config_name!r})")

    got_sha = (cfg.get("fusion_weights") or {}).get("sha256")
    if got_sha != sc.FLIP_FUSION_WEIGHTS_SHA256:
        problems.append(
            f"config.fusion_weights.sha256 = {got_sha!r}, the promoted "
            f"candidate is {sc.FLIP_FUSION_WEIGHTS_SHA256!r} "
            "(decisions.md ADR-099 amendment note; "
            "task 519 fusion_weights candidate)")

    return problems


def check_incumbent(health: dict) -> list[str]:
    """The ROLLBACK readback: the running api is back on what prod served.

    `check()` above verifies the flip target. Rolling back needs the mirror
    assertion, and it needs it for the same reason: dropping the compose
    overlay correctly is not evidence the container picked it up, and an api
    still at `AW_MRL_DIM=256` over a 128-dim collection is the same outage as
    the forward one (runbook §6).

    Every expected value is derived from tracked files -- the incumbent
    `ServingConfig` and `cytir/serve/fusion_weights.json`, the file prod
    loads when `AW_FUSION_WEIGHTS` is unset -- never from an operator-supplied
    number. A rollback check whose expectations can be typed in is a rollback
    check that passes.
    """
    problems: list[str] = []
    if health.get("status") != "ok":
        return [f"status={health.get('status')!r}, not 'ok' — full "
                f"response: {json.dumps(health)}"]
    cfg = health.get("config")
    if cfg is None:
        return ["response has no 'config' key"]

    incumbent = sc.get(INCUMBENT_SERVING_CONFIG)
    if cfg.get("mrl_dim") != incumbent.dense_dim:
        problems.append(
            f"config.mrl_dim = {cfg.get('mrl_dim')!r}, the incumbent "
            f"{INCUMBENT_SERVING_CONFIG!r} is {incumbent.dense_dim} — the "
            "flip overlay is still applied")
    if cfg.get("dense_rescore") is not False:
        problems.append(
            f"config.dense_rescore = {cfg.get('dense_rescore')!r}; the "
            "incumbent int8 config keeps rescore off")

    want = _incumbent_weights_sha256()
    got = (cfg.get("fusion_weights") or {}).get("sha256")
    if got != want:
        problems.append(
            f"config.fusion_weights.sha256 = {got!r}, the incumbent "
            f"cytir/serve/fusion_weights.json is {want!r}")
    return problems


def _incumbent_weights_sha256() -> str:
    path = pathlib.Path(__file__).resolve().parents[1] / "serve" / "fusion_weights.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate_problems(verdict_path=None, *, collection: str | None = None,
                  config_name: str = sc.FLIP_SERVING_CONFIG) -> list[str]:
    """Problems with the SEALED FLIP-GATE VERDICT, or [] if it permits a flip.

    Task 529. Echoing the right config back is evidence the deploy landed; it
    is not evidence the build earned the flip. Tim's 2026-09-06 ruling made
    the second question a gate: no class loses vs prod on any leg without a
    named, ruled reason (`data/flip-gate-contract.v1.json`). Without a sealed
    PASS from that battery this script refuses, because the failure it would
    otherwise miss is exactly the one task 524 measured — a full-stack dead
    heat at 276/629 over a dense leg at 141 against 189.
    """
    try:
        fg.assert_sealed_pass(verdict_path, collection=collection,
                              serving_config=config_name)
    except fg.FlipGateError as exc:
        return [f"flip gate: {exc}"]
    return []


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__)
    ap.add_argument("--url", required=True, help="e.g. https://websear.ch")
    ap.add_argument("--config", default=sc.FLIP_SERVING_CONFIG,
                    help="serving_config.py name to verify against "
                         f"(default {sc.FLIP_SERVING_CONFIG!r})")
    ap.add_argument("--gate-verdict", default=None,
                    help="the sealed flip-gate verdict "
                         "(default: the path in "
                         "data/flip-gate-contract.v1.json). There is no flag "
                         "to skip it: a flip without a sealed PASS is refused.")
    ap.add_argument("--gate-collection", default=None,
                    help="the PHYSICAL collection the sealed verdict is bound "
                         "to, e.g. aw_e8. /health reports the ALIAS name "
                         "(aw_prod), which is never what a battery scored, so "
                         "without this the gate check compares the alias to "
                         "the candidate and always refuses. "
                         "deploy/scripts/flip-release.sh passes the collection "
                         "it just proved the alias resolves to.")
    ap.add_argument("--rollback", action="store_true",
                    help="verify the INCUMBENT config instead: the api is back "
                         "at 128-dim with the incumbent fusion weights. Skips "
                         "the flip-gate check -- a rollback to what production "
                         "already served is always permitted.")
    args = ap.parse_args(argv)

    health = fetch_health(args.url)
    if args.rollback:
        problems = check_incumbent(health)
        ok_msg = (f"OK: {args.url}/health is back on the incumbent "
                  f"{INCUMBENT_SERVING_CONFIG!r} config and the incumbent "
                  "fusion weights.")
    else:
        collection = args.gate_collection or (
            (health.get("config") or {}).get("collection")
            if isinstance(health.get("config"), dict) else None)
        problems = gate_problems(args.gate_verdict, collection=collection,
                                 config_name=args.config)
        problems += check(health, args.config)
        ok_msg = (f"OK: {args.url}/health reports the ratified {args.config!r} "
                  "dense policy and the promoted fusion weights, and a sealed "
                  "flip-gate PASS covers the collection it is serving.")
    if problems:
        label = "ROLLBACK INCOMPLETE" if args.rollback else "FLIP REFUSED"
        print(f"{label} — {args.url}/health:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(f"\nfull /health response:\n{json.dumps(health, indent=2)}",
              file=sys.stderr)
        return 1
    print(ok_msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
