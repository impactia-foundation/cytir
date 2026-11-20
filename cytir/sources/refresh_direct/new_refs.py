"""new_refs — DAILY new-external-refs discovery lane (task 632,
mission item 1).

For every already-admitted, `eligible_now` source pack with a declared
source-owned discovery route (feed/dump/sitemap/API/registry — never scraping
where a feed exists, per data/acquisition-policy.v1.json route precedence),
report whether a discovery pass is due today and what route it would use.

Scope, stated honestly (see specs/refresh-cadence.md §5): this script always
computes real due/overdue state from the real cadence table and a real state
file, and never invents counts. What it does NOT yet do is fetch anything —
per-family discovery adapters (the code that actually walks a specific
source's feed/sitemap/dump and lists candidate URLs) are follow-up work,
registered one at a time in ADAPTER_REGISTRY below. A due source with no
registered adapter is reported as `status: not_implemented` and is never
silently skipped or counted as "checked".

Never runs against prod. This script only ever WRITES a report/queue file and
(when --mark-complete is passed by an operator, never by cron in dry-run) the
local due-state file. It performs no fetch, no embed, no qdrant write.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from cytir.core.paths import REPO_ROOT
from cytir.sources.refresh_direct import lane_lib as lib
from cytir.sources.refresh_direct import github_releases as _github_releases
import cytir.sources.refresh_direct.hf_org_models as _hf_org_models
import cytir.sources.refresh_direct.vendor_blog_rss as _vendor_blog_rss
import cytir.sources.refresh_direct.docs_sitemap as _docs_sitemap
from cytir.sources.refresh_direct import github_repo_docs as _github_repo_docs

# adapter family (cc/dump/sitemap/registry/api/cards) OR a specific pack
# `id` -> callable(source_row) -> dict (candidate discovery result); see
# the `id` fallback note where this registry is looked up, below.
# task 661 F5, 2026-09-26: entries below are keyed by pack id, not the
# shared `api` family, because they need pack-specific endpoints/parsing
# -- they await their pack's admission PR to land as `ingested`:
# `sources/us/github-releases-tracked.yaml` (id
# us-github-releases-tracked) / `sources/us/huggingface-orgs-tracked.yaml`
# (id us-huggingface-orgs-tracked) via #383; `sources/us/vendor-ai-blogs-
# tracked.yaml` (id us-vendor-ai-blogs-tracked) via a follow-up POLICY PR
# -- all `status: planned` today. The code path is ready the moment any
# of them does. The next family to onboard adds one entry here and
# nothing else in this file changes.
ADAPTER_REGISTRY: dict[str, object] = {
    "us-github-releases-tracked": _github_releases.discover,
    "us-huggingface-orgs-tracked": _hf_org_models.discover,
    "us-vendor-ai-blogs-tracked": _vendor_blog_rss.discover,
    # task 972: the docs delta adapters, keyed by pack id like the three above
    "us-opencode-ai-docs": _docs_sitemap.discover,
    "us-github-repo-docs-tracked": _github_repo_docs.discover,
}


def load_cadence_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["sources"]


def run(
    cadence_path: Path,
    state_path: Path,
    now: datetime,
    armed: bool,
) -> dict:
    rows = load_cadence_rows(cadence_path)
    state = lib.load_state(state_path)
    due = lib.due_report(rows, state, "new_refs", now=now)
    rows_by_id = {r["id"]: r for r in rows}

    entries = []
    for d in due:
        row = rows_by_id[d.source_id]
        entry = {
            "id": d.source_id,
            "layer": row["layer"],
            "country": row["country"],
            "adapter": row["adapter"],
            "due": d.due,
            "alarm": d.alarm,
            "last_run": d.to_json_dict()["last_run"],
            "days_since_last_run": d.to_json_dict()["days_since_last_run"],
        }
        if not d.due:
            entry["status"] = "not_due"
            entries.append(entry)
            continue
        if not armed:
            entry["status"] = "would_check (dry-run)"
            entries.append(entry)
            continue
        # `row["adapter"]` is the pack schema's coarse 6-value family enum
        # (cc/dump/sitemap/registry/api/cards) -- one shared handler per
        # family is the base design (see test_a_fully_wired_adapter_
        # marks_state_complete's own "dump" monkeypatch). A source whose
        # discovery needs pack-specific logic within a shared family (task
        # 661 F5: github_releases and hf_org_models are both `adapter: api`
        # but need different endpoints/parsing) registers under its own
        # pack `id` instead -- checked SECOND, never overriding a real
        # family-wide handler if one exists.
        impl = ADAPTER_REGISTRY.get(row["adapter"]) or ADAPTER_REGISTRY.get(d.source_id)
        if impl is None:
            entry["status"] = "not_implemented"
            entry["reason"] = (
                f"no discovery adapter registered for adapter='{row['adapter']}' "
                f"(id={d.source_id}) — add one to ADAPTER_REGISTRY before arming this family"
            )
        else:
            result = impl(row)
            entry["status"] = "checked"
            entry.update(result)
            state = lib.mark_run_complete(state, d.source_id, "new_refs", lib.utcnow_iso())
        entries.append(entry)

    if armed:
        lib.save_state(state_path, state)

    return {
        "schema": "aw.refresh-lane-new-refs-report.v1",
        "lane": "new_refs",
        "armed": armed,
        "generated_at": lib.utcnow_iso(),
        "as_of": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "due_count": sum(1 for e in entries if e["due"]),
        "alarm_count": sum(1 for e in entries if e["alarm"]),
        "not_implemented_count": sum(1 for e in entries if e.get("status") == "not_implemented"),
        "sources": entries,
    }


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--cadence-table", default=str(REPO_ROOT / "data" / "refresh-cadence.v1.json"))
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", default="-")
    ap.add_argument("--now", default=None, help="override 'now' as ISO UTC (tests only)")
    ap.add_argument("--armed", action="store_true", default=False)
    args = ap.parse_args(argv)

    now = (
        datetime.strptime(args.now, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if args.now
        else datetime.now(timezone.utc)
    )
    report = run(Path(args.cadence_table), Path(args.state), now, args.armed)
    text = json.dumps(report, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], "python -m cytir.sources.refresh_direct.new_refs"))
