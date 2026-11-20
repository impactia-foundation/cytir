"""Prepare, but never embed or apply, entity-card content enrichments.

The input is an already-admitted join JSONL.  Website body text is optional;
when present it must carry the receipt validated by ``card_recipe_adr060``.
This deliberately does not fetch. A source-neutral upstream snapshot stage
must prefer a recent Common Crawl index/WARC range source, re-resolve consent
live under CC provenance, and use an own-UA/path-robots direct crawl only as a
fallback. The output is ordinary card records for the normal
stage/embed/projection path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cytir.corpus import facet_bundle as FB
from cytir.corpus.card_recipe import build_two_channel, norm_uid


def main(argv: list[str] | None = None, *, prog: str = "cyt corpus card-content-dry-run") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--events-out", required=True, type=Path,
                    help="append-only source-wave facet event JSONL")
    current = ap.add_mutually_exclusive_group(required=True)
    current.add_argument("--current-parquet", type=Path,
                         help="manifested compacted current-state facet table")
    current.add_argument("--bootstrap-empty-current", action="store_true",
                         help="explicit first-wave bootstrap; no existing facet truth")
    ap.add_argument("--events-root", type=Path,
                    help="canonical manifested event root backing --current-parquet")
    ap.add_argument("--phase", default="website-content-dry-run")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    event_manifest_path = args.events_out.with_suffix(
        args.events_out.suffix + ".manifest.json")
    if args.out.exists() or args.events_out.exists() or event_manifest_path.exists():
        raise SystemExit("ABORT refusing to overwrite dry-run or event-wave artifact")

    if args.current_parquet and not args.events_root:
        raise SystemExit("ABORT --current-parquet requires its canonical --events-root")
    if args.bootstrap_empty_current and args.events_root:
        raise SystemExit("ABORT empty bootstrap cannot name an existing event root")
    canonical_events = (FB.load_canonical_events(args.events_root)
                        if args.current_parquet else [])
    current_rows = (FB.read_parquet(
        args.current_parquet, source_events=canonical_events,
        require_event_lineage=True) if args.current_parquet else [])
    current_by_entity = {}
    for event in current_rows:
        current_by_entity.setdefault(event["entity"], []).append(event)
    rows = []
    seen_entities = set()
    for line_no, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            seed = row["seed"]
            entity_key = norm_uid(seed.get("register_id")) or row.get("qid") or seed["domain"]
            if entity_key in seen_entities:
                raise ValueError(f"duplicate entity in one source wave: {entity_key}")
            seen_entities.add(entity_key)
            rec, audit = build_two_channel(
                seed, row.get("existing"), ent=row.get("entity"),
                qid=row.get("qid"), zweck=row.get("zweck"), phase=args.phase,
                current_bundle=current_by_entity.get(entity_key, []))
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"ABORT input line {line_no}: {exc}") from exc
        # Output is a normal card record, consumable unchanged by stage_pack_b1;
        # audit/event JSONL is deliberately separate master-side evidence.
        rows.append({"record": rec, "audit": audit, "input_line": line_no})
        if args.limit and len(rows) >= args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row["record"], ensure_ascii=False, sort_keys=True) + "\n")
    events = [event for row in rows for event in row["audit"]["facet_events"]]
    event_manifest = FB.publish_event_wave(events, args.events_out, task=args.phase)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps({"status": "DRY_RUN_ONLY", "rows": len(rows),
                      "output": str(args.out), "sha256": digest,
                      "event_manifest": event_manifest}, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(prog="python -m cytir.corpus.card_content_dry_run"))
