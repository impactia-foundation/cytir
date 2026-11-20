"""Acquire the INSEE SIRENE legal-units stock and hand it to the official-bulk adapter.

    AW_SIRENE_ROOT=<dir> AW_FETCH_VANTAGE=<name> python3 -m cytir.sources.official_bulk.sirene_legal_units

The `fr-insee-sirene` pack's declared extractor (ported from the bash driver by task 2176; same
environment, same files, same exit codes). Two steps:

  1. Fetch the stock parquet ONCE, with `curl`, and write its immutable HTTP evidence set
     (body, headers, transport metadata) under `$AW_SIRENE_ROOT/input/`. A partial evidence set
     is never completed or overwritten: exit 2.
  2. Run `cytir.sources.official_bulk.bulk` on that evidence set, and return its exit code.

Environment: `AW_SIRENE_ROOT` (required, exit 1 without it), `AW_FETCH_VANTAGE` (required, exit 1
without it; the name of the fetching box, passed to the adapter as `--fetch-vantage`),
`AW_PYTHON_BIN` (default `python3`).
The crawler identity comes from `cytir.core.user_agent`, which fails closed when no operator
identity is configured.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cytir.core import user_agent

REPO_ROOT = Path(__file__).resolve().parents[3]
URL = "https://www.data.gouv.fr/api/1/datasets/r/350182c9-148a-46e0-8389-76c2ec1374a3"
CONTENT_SCOPE = ("registry identity spine; includes active and ceased legal units, including "
                 "protected-person rows requiring projection-time privacy/status filtering")
NEXT_PROJECTION = ("filter active public legal-person records; join validated website evidence; "
                   "compose evidence-backed company cards; never index raw person/address rows")


def _fetch_evidence(*, ua: str, python: str, body: Path, headers: Path, metadata: Path,
                    env: dict) -> int:
    present = [p.exists() for p in (body, headers, metadata)]
    if all(present):
        return 0
    if any(present):
        print("refusing partial immutable HTTP evidence set", file=sys.stderr)
        return 2
    pid = os.getpid()
    tmp_body = Path(f"{body}.transport.{pid}")
    tmp_headers = Path(f"{headers}.transport.{pid}")
    metrics = Path(f"{metadata}.curl.{pid}")
    tmp_meta = Path(f"{metadata}.transport.{pid}")
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        return _fetch_and_record(ua=ua, python=python, body=body, headers=headers, metadata=metadata,
                                 env=env, tmp_body=tmp_body, tmp_headers=tmp_headers,
                                 metrics=metrics, tmp_meta=tmp_meta, fetched_at=fetched_at)
    finally:
        # a failed fetch must leave no raw (unredacted) transport temp file behind
        for leftover in (tmp_body, tmp_headers, metrics, tmp_meta):
            leftover.unlink(missing_ok=True)


def _fetch_and_record(*, ua, python, body, headers, metadata, env, tmp_body, tmp_headers,
                      metrics, tmp_meta, fetched_at) -> int:
    with open(metrics, "wb") as out:
        rc = subprocess.run(
            ["curl", "--fail-with-body", "--location", "--silent", "--show-error",
             "--retry", "8", "--retry-all-errors", "--retry-delay", "3",
             "--connect-timeout", "30", "--max-time", "3600", "--user-agent", ua,
             "--dump-header", str(tmp_headers), "--write-out", "%{json}",
             URL, "--output", str(tmp_body)], stdout=out, env=env).returncode
    if rc:
        return rc
    with open(os.devnull, "wb") as null:
        rc = subprocess.run(
            [python, "-m", "cytir.cli", "source", "http-evidence",
             "--body", str(tmp_body), "--headers", str(tmp_headers),
             "--curl-metrics", str(metrics), "--requested-url", URL,
             "--fetched-at", fetched_at, "--out", str(tmp_meta)],
            stdout=null, env=env).returncode
    if rc:
        return rc
    os.sync()
    os.replace(tmp_body, body)
    os.replace(tmp_headers, headers)
    os.replace(tmp_meta, metadata)
    metrics.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None, env: dict | None = None) -> int:
    env = dict(os.environ if env is None else env)
    ua = user_agent.user_agent()
    root = env.get("AW_SIRENE_ROOT")
    if not root:
        print("CYTIR_SIRENE_ROOT: set CYTIR_SIRENE_ROOT", file=sys.stderr)
        return 1
    vantage = env.get("AW_FETCH_VANTAGE")
    if not vantage:
        print("CYTIR_FETCH_VANTAGE: set CYTIR_FETCH_VANTAGE", file=sys.stderr)
        return 1
    python = env.get("AW_PYTHON_BIN") or "python3"
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    inputs = Path(root) / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    body = inputs / "stock-unite-legale.parquet"
    headers = Path(f"{body}.headers")
    metadata = Path(f"{body}.transport.json")
    rc = _fetch_evidence(ua=ua, python=python, body=body, headers=headers, metadata=metadata, env=env)
    if rc:
        return rc
    return subprocess.run(
        [python, "-m", "cytir.sources.official_bulk.bulk",
         "--body", str(body), "--headers", str(headers), "--metadata", str(metadata),
         "--root", root, "--source-id", "fr-sirene-legal-units",
         "--source-name", "INSEE SIRENE stockUniteLegale", "--country", "FR",
         "--requested-url", URL, "--content-scope", CONTENT_SCOPE,
         "--next-projection", NEXT_PROJECTION, "--fetch-vantage", vantage],
        env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
