"""cytir.ops.check_qdrant_temp_path -- `cyt ops check-qdrant-temp-path`: the P18 monitor — is qdrant's snapshot STAGING on the roomy volume? (task 637 §16)

WHY A BEHAVIOURAL CHECK AND NOT A CONFIG READ
---------------------------------------------
Qdrant builds a snapshot tar under `<storage>/tmp/` and only copies it to the
snapshots directory afterwards. Snapshotting a 113 GB collection therefore needs
~113 GB free **on the storage volume**. On 2026-09-22 that staged a 140.7 GB file
onto `/media/disk2` — the volume the live twin sits on — and took it to 65 GB
free (task 637 §16).

`QDRANT__STORAGE__TEMP_PATH` moves the staging. The problem is that **the running
process does not echo it**: `GET /telemetry?details_level=3` reports no temp path
at any level, so "is it set?" cannot be answered from the API, and a monitor that
checked the compose file would be checking a declaration rather than a process
(deployment-doctrine: the running process is the source of truth).

So this checks the **effect**, which is stronger. Verified empirically on
qdrant 1.18.3:

* with `QDRANT__STORAGE__TEMP_PATH=/tmpvol` set, a snapshot creates
  `/tmpvol/tmp/` and **never** creates `<storage>/tmp/`;
* without it, `<storage>/tmp/` appears.

The check is therefore: **the container's env carries the variable, the directory
it names is a mount with room, and `<storage>/tmp` does not exist.** All three,
because any one alone can pass while the setting is ineffective — the env var can
name a path inside the container image (no mount, so it fills the container's own
layer), and an absent `<storage>/tmp` only proves no snapshot has run yet.

    cyt ops check-qdrant-temp-path --container deploy-qdrant-1 --min-free-gb 150

Exit 0 = staging is on a roomy volume · 2 = it is not · 3 = cannot tell.

Moved from the pipeline tree in task 1234, behaviour unchanged. The container default is not a
box: it is the name `docker compose` gives the `qdrant` service of the shipped
`deploy/docker-compose.yaml` (project `deploy`); any other stack passes `--container`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

#: `docker compose`'s name for the shipped compose file's qdrant service (project `deploy`).
COMPOSE_QDRANT_CONTAINER = "deploy-qdrant-1"


def docker(*args, timeout=30):
    out = subprocess.run(["docker", *args], capture_output=True, text=True,
                         timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError("docker %s -> %s" % (" ".join(args), out.stderr.strip()))
    return out.stdout.strip()


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--container", default=COMPOSE_QDRANT_CONTAINER,
                    help="the qdrant container to inspect (default: %(default)s)")
    ap.add_argument("--min-free-gb", type=float, default=150.0,
                    help="the staging volume must hold a full snapshot tar; "
                         "default is aw_e9's ~113 GB plus headroom")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if shutil.which("docker") is None:
        print("REFUSE: no docker CLI here; run this on the serving host")
        return 3

    rep = {"container": a.container}
    try:
        env = json.loads(docker("inspect", "-f", "{{json .Config.Env}}", a.container))
        mounts = json.loads(docker("inspect", "-f", "{{json .Mounts}}", a.container))
    except Exception as exc:
        print("REFUSE: cannot inspect %s: %s" % (a.container, exc))
        return 3

    temp = None
    for e in env or []:
        if e.startswith("QDRANT__STORAGE__TEMP_PATH="):
            temp = e.split("=", 1)[1]
    rep["temp_path_env"] = temp
    problems = []
    if not temp:
        problems.append(
            "QDRANT__STORAGE__TEMP_PATH is NOT set on %s, so a snapshot will "
            "stage under <storage>/tmp on the serving volume (task 637 section 16)"
            % a.container)

    # 2. is the temp path backed by a mount, and does that mount have room?
    backing = None
    if temp:
        for m in mounts or []:
            dest = (m.get("Destination") or "").rstrip("/")
            if dest and (temp.rstrip("/") == dest or temp.startswith(dest + "/")):
                backing = m
                break
        if backing is None:
            problems.append(
                "%s is not backed by any mount, so qdrant would stage the tar "
                "inside the container's own filesystem" % temp)
        else:
            src = backing.get("Source")
            rep["temp_path_host"] = src
            try:
                free = shutil.disk_usage(src).free / 1e9
            except OSError as exc:
                free = None
                problems.append("cannot stat the staging volume %s: %s" % (src, exc))
            rep["temp_path_free_gb"] = round(free, 1) if free is not None else None
            if free is not None and free < a.min_free_gb:
                problems.append(
                    "the staging volume %s has %.1f GB free, below the %.1f GB a "
                    "snapshot tar needs" % (src, free, a.min_free_gb))

    # 3. has qdrant created <storage>/tmp anyway? (the effect, not the setting)
    storage_host = None
    for m in mounts or []:
        if (m.get("Destination") or "").rstrip("/") == "/qdrant/storage":
            storage_host = m.get("Source")
    rep["storage_host"] = storage_host
    if storage_host:
        try:
            listing = docker("exec", a.container, "sh", "-c",
                             "test -d /qdrant/storage/tmp && echo PRESENT || echo absent")
        except Exception as exc:
            listing = "unknown(%s)" % exc
        rep["storage_tmp"] = listing
        if listing == "PRESENT":
            problems.append(
                "/qdrant/storage/tmp EXISTS -- qdrant has staged a snapshot on "
                "the serving volume, which is the effect this check exists to "
                "catch (the setting may have been added after the fact)")
    else:
        problems.append("no /qdrant/storage mount found; cannot check the effect")

    rep["problems"] = problems
    rep["ok"] = not problems
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        print("qdrant temp path: %s (host %s, %s GB free); storage/tmp: %s"
              % (rep.get("temp_path_env"), rep.get("temp_path_host"),
                 rep.get("temp_path_free_gb"), rep.get("storage_tmp")))
        for pr in problems:
            print("  PROBLEM: %s" % pr)
        print("  OK" if rep["ok"] else "  FAIL")
    return 0 if rep["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
