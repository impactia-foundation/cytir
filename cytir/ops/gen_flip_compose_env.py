#!/usr/bin/env python3
"""Generate/validate the flip's compose overlay from ONE source.

`cytir/core/serving_config.py`'s `flip_deploy_env()` is the ratified truth for
the env `deploy/docker-compose.flip.yaml`'s `aw-api` service
must carry. Task 519 found the compose file silently missing
`AW_DENSE_RESCORE`/`AW_DENSE_OVERSAMPLING` for the whole life of the "256-pq8"
proposal (decisions.md's ADR-011 amendment) -- nothing failed loudly, a gate
arm just quietly measured a configuration nobody had ratified. This script is
the mechanism that makes that unrepeatable:

  --check   read the tracked overlay back and refuse (nonzero exit, a labelled
            diff) if ANY service's environment disagrees with
            `flip_deploy_env()`, or if the promoted fusion-weights artifact
            (`cytir/serve/fusion_weights.candidate519.json`) has drifted
            from its sealed sha256. Used by its test module (test_flip_compose_env)
            and may be run standalone in CI before a flip.
  --write   regenerate the overlay file from the ratified config.

Today's `deploy/docker-compose.yaml` is UNTOUCHED by this script -- it serves
the live 128-int8 collection and stays byte-identical. The overlay is applied
on top of it on flip day only:

    docker compose -f deploy/docker-compose.yaml \\
                    -f deploy/docker-compose.flip.yaml up -d

Task 596 added a SECOND thing this overlay carries, besides the aw-api/
aw-api env: the `qdrant` service's bind mount that attaches an
off-local-disk collection directory into the serving qdrant --
`serving_config.py`'s `flip_qdrant_volumes()`. It has to live in THIS file
(not a separate overlay) because `flip-release.sh` step 6 applies exactly
`-f docker-compose.yaml -f docker-compose.flip.yaml` -- any mount that is
not in one of those two files is invisible to that command and would be
DROPPED the moment the flip recreates the qdrant service.

At task 596 that mount attached the flip CANDIDATE (`aw_e8`, restored by
task 589 onto the scratch volume) ahead of the flip, because local disk had
no room for a second 135 GB collection. Since the flip
(2026-09-17), task 597 re-pointed it at the UNSERVED ROLLBACK collection
(`aw_e3`) instead: the served collection (now `aw_e8`) belongs on local
NVMe like every other served collection (random-4KB read latency, not
disk space, is the reason -- see `serving_config.py`'s comment), and only
the rollback collection, which takes no query traffic, could tolerate the
slower volume. Task 618 (2026-09-19) retired `aw_e3` from the serving box
entirely (Tim's ruling, ADR-098 am.1 addendum), so `flip_qdrant_volumes()`
currently returns no mounts and this overlay's `qdrant.volumes` is empty.
The mechanism -- one bind mount, generated from one pointer, applied with a
qdrant-only `up -d qdrant` -- is not removed, only unused today; a future
re-point moves it by PR, the same way `FLIP_FUSION_WEIGHTS_FILE` moves when
the promoted candidate changes.

Run:
    python3 -m cytir.ops.gen_flip_compose_env --check
    python3 -m cytir.ops.gen_flip_compose_env --write
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import yaml

from cytir.core import paths as PATHS
from cytir.core import serving_config as sc

REPO_ROOT = PATHS.REPO_ROOT
OVERLAY_PATH = REPO_ROOT / "deploy" / "docker-compose.flip.yaml"
SERVICES = ("aw-api",)

# Any key touching the dense/MRL/weights knobs this module owns. Used to catch
# an overlay that carries a STALE key this module no longer declares (e.g. a
# typo'd rename left both the old and new key in the file).
_OWNED_PREFIXES = ("AW_DENSE_", "AW_MRL_DIM", "AW_FUSION_WEIGHTS")

HEADER = """# deploy/docker-compose.flip.yaml — GENERATED, do not hand-edit.
#
# Flip overlay for the ratified "256-pq8" serving config
# (cytir/core/serving_config.py; decisions.md's ADR-011 amendment) plus the
# flip's promoted fusion weights (cytir/core/serving_config.py's
# FLIP_FUSION_WEIGHTS_FILE/SHA256 -- the ONE home for that pointer, per task
# 552; task 519's candidate519 was promoted by decisions.md's ADR-099
# amendment / PR #91, superseded by task 533's candidate533), PLUS the
# qdrant bind mount slot that CAN attach an off-local-disk collection into
# the serving qdrant (serving_config.py's flip_qdrant_volumes()). It
# attached the flip candidate `aw_e8` ahead of the flip (task 596, restored
# by task 589), then the UNSERVED rollback collection `aw_e3` once `aw_e8`
# was relocated to local disk (task 597, 2026-09-17). Task 618 (2026-09-19)
# retired `aw_e3` from the serving box entirely (ADR-098 am.1 addendum,
# preserved elsewhere -- see data/production-flip-ruling-aw_e8.v1.json), so
# `flip_qdrant_volumes()` currently returns no mounts: `aw_e8` (served)
# lives on local NVMe in docker-compose.yaml's own `aw-qdrant` volume, and
# there is no second collection to attach. Task 668 (2026-09-28) tried
# re-pointing this at `aw_e9` and reverted it the same day: the measured
# local-disk footprints (aw_e8 149 GB + aw_e9 113 GB against 89 GB free)
# mean serving aw_e9 off this attach would mean serving it from the
# network volume -- the cold-latency antipattern task 597 exists to avoid.
# aw_e9 reaches local disk by relocation instead
# (deploy/scripts/flip-aw_e9-methodA.sh), not an attach.
#
# Apply ON TOP of docker-compose.yaml on flip day, in the SAME release as the
# alias swap (task 508, the flip-day runbook — a dim change without this is an
# outage, not a degradation):
#
#   docker compose -f deploy/docker-compose.yaml \\
#                   -f deploy/docker-compose.flip.yaml up -d
#
# The qdrant mount below can be applied to just that service, scoped
# (as task 596 did ahead of the flip, and task 597 did for the re-point):
#
#   docker compose -f deploy/docker-compose.yaml \\
#                   -f deploy/docker-compose.flip.yaml up -d qdrant
#
# docker-compose.yaml itself stays BYTE-IDENTICAL to what prod serves today
# (the 128-int8 collection) — this file exists precisely so the flip's env
# never has to be hand-typed into it. Regenerate with:
#
#   python3 -m cytir.ops.gen_flip_compose_env --write
#
# and its test module fails the build if this file ever
# disagrees with cytir/core/serving_config.py — the exact gap task 519 found
# (the compose file silently ran an unratified dense policy for months).
"""


def render() -> str:
    env = sc.flip_deploy_env()
    body = {"services": {
        "qdrant": {"volumes": list(sc.flip_qdrant_volumes()),
                   "environment": dict(sc.flip_qdrant_environment())},
        **{svc: {"environment": dict(env)} for svc in SERVICES},
    }}
    return HEADER + "\n" + yaml.safe_dump(
        body, sort_keys=False, default_flow_style=False)


def _fusion_weights_artifact_problems() -> list[str]:
    path = REPO_ROOT / "cytir" / "serve" / sc.FLIP_FUSION_WEIGHTS_FILE
    if not path.exists():
        return [f"{path} does not exist -- AW_FUSION_WEIGHTS in the flip "
                f"overlay would point at nothing inside the container"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sc.FLIP_FUSION_WEIGHTS_SHA256:
        return [f"{path} sha256 {digest} != the sealed candidate "
                f"{sc.FLIP_FUSION_WEIGHTS_SHA256} (serving_config.py / "
                f"task 519's fusion_weights.candidate519.json) -- "
                f"the promoted artifact has drifted from what it was gated"]
    return []


def check() -> list[str]:
    """Return a list of human-readable problems; empty means everything the
    flip's compose-facing config depends on agrees with the ratified source.
    """
    problems: list[str] = []
    want = sc.flip_deploy_env()
    if not OVERLAY_PATH.exists():
        return [f"{OVERLAY_PATH} does not exist -- run "
                f"`python3 -m cytir.ops.gen_flip_compose_env --write`"]
    doc = yaml.safe_load(OVERLAY_PATH.read_text()) or {}
    services = doc.get("services") or {}

    want_qenv = sc.flip_qdrant_environment()
    got_qenv = (services.get("qdrant") or {}).get("environment") or {}
    for k, v in want_qenv.items():
        if str(got_qenv.get(k)) != str(v):
            problems.append(
                f"qdrant.environment {k}={got_qenv.get(k)!r} != the ratified "
                f"{v!r} (serving_config.flip_qdrant_environment) -- P18: a "
                f"snapshot stages under <storage>/tmp unless this is set, and "
                f"the serving disk has no room for a 113 GB tar")
    want_volumes = sc.flip_qdrant_volumes()
    got_volumes = (services.get("qdrant") or {}).get("volumes") or []
    for v in want_volumes:
        if v not in got_volumes:
            problems.append(
                f"qdrant.volumes is missing {v!r} -- the flip's aw_e8 "
                f"attach mount (cytir/core/serving_config.py:"
                f"flip_qdrant_volumes) is not in the tracked overlay")
    for v in got_volumes:
        if v not in want_volumes:
            problems.append(
                f"qdrant.volumes has {v!r}, which flip_qdrant_volumes() no "
                "longer declares -- stale mount from a prior generation, "
                "remove it and re-run --write")

    for svc in SERVICES:
        got = (services.get(svc) or {}).get("environment") or {}
        for key, value in want.items():
            if str(got.get(key)) != str(value):
                problems.append(
                    f"{svc}.environment[{key!r}] = {got.get(key)!r}, "
                    f"ratified value is {value!r} "
                    f"(cytir/core/serving_config.py:flip_deploy_env)")
        stale = {k for k in got if k.startswith(_OWNED_PREFIXES)} - set(want)
        for key in sorted(stale):
            problems.append(
                f"{svc}.environment has {key!r}, which "
                "flip_deploy_env() no longer declares -- stale key from a "
                "prior generation, remove it and re-run --write")
    problems.extend(_fusion_weights_artifact_problems())
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true",
                   help="fail if the tracked overlay disagrees (default CI use)")
    g.add_argument("--write", action="store_true",
                   help="regenerate deploy/docker-compose.flip.yaml")
    args = ap.parse_args()

    if args.write:
        OVERLAY_PATH.write_text(render())
        print(f"wrote {OVERLAY_PATH}")
        return 0

    problems = check()
    if problems:
        print(f"{OVERLAY_PATH} disagrees with the ratified serving config:",
              file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"OK: {OVERLAY_PATH} matches cytir/core/serving_config.py "
          f"({sc.FLIP_SERVING_CONFIG}) and the promoted fusion weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
