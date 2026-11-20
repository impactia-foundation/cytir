#!/usr/bin/env python3
"""Bind one executable source route to ``cytir.sources.cov.engine``.

Task 351 already has the proven multi-mode engine and its configuration in
``data/cov351_routes.yaml``.  This adapter does not copy either.  It reads
the exact ``config_ref`` carried by a source-runner plan, checks that the
referenced source still names the requested host, and runs only that source.

The source runner supplies ``aw.source-extractor-env.v1`` variables.  Explicit
CLI flags are also accepted so the adapter remains independently runnable.
All artifacts land below the requested output directory; nothing stages or
promotes a collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]

from cytir.sources.cov import engine as COV  # noqa: E402


class AdapterError(RuntimeError):
    """The source plan and its extractor-owned configuration disagree."""


def _env_config_ref() -> dict:
    raw = os.environ.get("AW_SOURCE_CONFIG_REF", "")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"AW_SOURCE_CONFIG_REF is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError("AW_SOURCE_CONFIG_REF must encode an object")
    return value


def resolve_binding(*, pack_id: str, route_id: str, config_ref: dict) -> dict:
    """Resolve and validate one cov351 source without duplicating its config."""
    required = {"path", "pack", "source"}
    missing = required - set(config_ref)
    if missing:
        raise AdapterError(f"config_ref missing {sorted(missing)}")
    config_path = (REPO_ROOT / config_ref["path"]).resolve()
    try:
        config_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise AdapterError("config_ref.path escapes the repository") from exc
    if not config_path.is_file():
        raise AdapterError(f"config_ref.path does not exist: {config_path}")

    legacy_pack = COV.load_routes(str(config_path), config_ref["pack"])
    source_key = config_ref["source"]
    source = legacy_pack.get("sources", {}).get(source_key)
    if source is None:
        raise AdapterError(
            f"{config_ref['pack']}: no source {source_key!r} in {config_path}")
    if source["host"] != route_id:
        raise AdapterError(
            f"{pack_id}/{route_id}: config_ref resolves host "
            f"{source['host']!r}, refusing cross-route acquisition")
    return {
        "pack_id": pack_id,
        "route_id": route_id,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config_pack": config_ref["pack"],
        "source_key": source_key,
        "source": source,
    }


def _safe_route(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "route"


def _frontier_drained(out_dir: Path, source_key: str) -> bool:
    inv = out_dir / "inv" / f"{source_key}.jsonl"
    state_path = out_dir / "fetch_state.json"
    if not inv.is_file() or not state_path.is_file():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    done = set(state.get("done") or [])
    for line in inv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("fetch_url") or row.get("url")) not in done:
            return False
    return not any(state.get("pdf_queue", {}).values())


def run(binding: dict, *, out_root: Path, stage: str, limit: int, budget: int,
        no_l0: bool, max_chunks: int) -> dict:
    route_dir = out_root / binding["pack_id"] / _safe_route(binding["route_id"])
    route_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        routes=binding["config_path"], pack=binding["config_pack"],
        out_dir=str(route_dir), only=binding["source_key"], l0_root="",
        no_l0=no_l0)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    completed: list[str] = []

    if stage in ("enumerate", "all"):
        COV.stage_enumerate(SimpleNamespace(**common, re_enumerate=False))
        completed.append("enumerate")
    if stage in ("fetch", "all"):
        COV.stage_fetch(SimpleNamespace(
            **common, limit=limit, budget=budget, hb_secs=240.0))
        completed.append("fetch")

    drained = _frontier_drained(route_dir, binding["source_key"])
    if stage == "gate" or (stage == "all" and drained):
        COV.stage_gate(SimpleNamespace(
            routes=binding["config_path"], pack=binding["config_pack"],
            out_dir=str(route_dir), max_chunks=max_chunks))
        completed.append("gate")

    receipt = {
        "schema": "aw.cov-source-adapter-run.v1",
        "source_runner_contract": "aw.source-extractor-env.v1",
        "pack_id": binding["pack_id"],
        "route_id": binding["route_id"],
        "config_ref": {
            "path": str(Path(binding["config_path"]).relative_to(REPO_ROOT)),
            "pack": binding["config_pack"],
            "source": binding["source_key"],
        },
        "config_sha256": binding["config_sha256"],
        "out_dir": str(route_dir),
        "requested_stage": stage,
        "completed_stages": completed,
        "frontier_drained": drained,
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt_path = route_dir / "ADAPTER_RUN.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    env_ref = _env_config_ref()
    ap = argparse.ArgumentParser(prog=prog, description=__doc__)
    ap.add_argument("--pack-id", default=os.environ.get("AW_SOURCE_PACK_ID", ""))
    ap.add_argument("--route-id", default=os.environ.get("AW_SOURCE_ROUTE_ID", ""))
    ap.add_argument("--config-path", default=env_ref.get("path", ""))
    ap.add_argument("--config-pack", default=env_ref.get("pack", ""))
    ap.add_argument("--config-source", default=env_ref.get("source", ""))
    ap.add_argument("--out-dir", default=os.environ.get(
        "AW_SOURCE_OUT_DIR", "build/run_source"))
    ap.add_argument("--stage", choices=("plan", "enumerate", "fetch", "gate", "all"),
                    default="all")
    ap.add_argument("--limit", type=int, default=int(os.environ.get(
        "AW_SOURCE_LIMIT", "0")))
    ap.add_argument("--budget", type=int, default=int(os.environ.get(
        "AW_SOURCE_BUDGET", "60000")))
    ap.add_argument("--no-l0", action="store_true",
                    default=os.environ.get("AW_SOURCE_NO_L0") == "1")
    ap.add_argument("--max-chunks", type=int, default=50)
    return ap


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    if not args.pack_id or not args.route_id:
        raise AdapterError("pack-id and route-id are required")
    binding = resolve_binding(
        pack_id=args.pack_id, route_id=args.route_id,
        config_ref={"path": args.config_path, "pack": args.config_pack,
                    "source": args.config_source})
    summary = {
        key: binding[key] for key in (
            "pack_id", "route_id", "config_path", "config_sha256",
            "config_pack", "source_key")
    }
    summary["mode"] = binding["source"]["mode"]
    if args.stage == "plan":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    receipt = run(
        binding, out_root=Path(args.out_dir), stage=args.stage,
        limit=args.limit, budget=args.budget, no_l0=args.no_l0,
        max_chunks=args.max_chunks)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
