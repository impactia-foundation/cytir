#!/usr/bin/env python3
"""Verify a url-group supersede BY ID on every target you name (task 1050; review 1049 REQUIRED 7).

One receipt says what an apply wrote and what it removed. This tool reads each target with the
Qdrant RETRIEVE endpoint only (`POST /collections/<c>/points`, ids in the body). It NEVER scrolls:
a scroll on the serving box is refused by `cytir.index.prod_guard` and took prod down once
(incident 2026-10-07). It uses the drain's own reader (`cytir.index.point_delete`), so every call
passes the one prod gate and the pacing.

Per target it checks:
  * every WRITTEN id is present, and its `license` equals the receipt's (when the receipt names one);
  * exactly the receipt's expected number of the written rows carry a `window` field
    (`{"index", "count"}`, RULING 55), and each field is well formed;
  * every REMOVED id is absent.
With two or more targets it then compares the written rows between each target and the first one:
the payload digest and the `dense` vector digest of every id must be equal (the twin-mirror rule:
counts are not content).

Output: one line per target (`PASS`/`FAIL`) and one line per pair, then the exit code (0 only when
every line is PASS).

A target's collection must be a declared live name (`cytir.index.live_target.live_collection_names`)
or the prod alias. Anything else is refused unless `--allow-scratch` is given, so a mistyped name
cannot make a scratch copy look like the verified target.

Receipt (JSON): {"schema": "aw.supersede-receipt.v1", "pack_id", "consolidation", "license",
"window_rows", "written": [ids], "removed": [ids]}, and optionally "distinct_urls" (the written rows
sit at this many urls) and "window_urls" (the windowed rows sit at this many urls).

`--pack-records records.jsonl` checks the INPUT instead: the door's output must carry exactly the
receipt's written ids (and the receipt's consolidation date as `source_version`), so a re-run on
another day (rows carry `last_seen = today`, the records file hash changes) is judged by what
matters: the ids are url + text digest and do not change with the date.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from cytir.index import live_target as LT
from cytir.index import point_delete as PD

SCHEMA = "aw.supersede-receipt.v1"
BATCH = 100


class VerifyRefused(RuntimeError):
    """The inputs are not acceptable (bad receipt, undeclared collection): nothing was read."""


def ids_sha256(ids) -> str:
    """sha256 of the sorted ids joined by one newline (no trailing newline): the digest the
    runbook writes down for an id set."""
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode()).hexdigest()


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def load_receipt(path: str | Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise VerifyRefused(f"{path}: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    written, removed = doc.get("written"), doc.get("removed")
    if not isinstance(written, list) or not isinstance(removed, list) or not written:
        raise VerifyRefused(f"{path}: `written` and `removed` must be lists, `written` not empty")
    if len(set(written)) != len(written) or len(set(removed)) != len(removed):
        raise VerifyRefused(f"{path}: an id is listed twice")
    if set(written) & set(removed):
        raise VerifyRefused(f"{path}: an id is both written and removed")
    if not isinstance(doc.get("window_rows"), int) or doc["window_rows"] < 0:
        raise VerifyRefused(f"{path}: `window_rows` must be a non-negative integer")
    return doc


def parse_target(spec: str) -> tuple[str, str, str]:
    """`LABEL=URL,COLLECTION` -> (label, normalised url, collection)."""
    label, sep, rest = spec.partition("=")
    url, sep2, collection = rest.rpartition(",")
    if not (sep and sep2 and label and url and collection):
        raise VerifyRefused(f"--target {spec!r}: expected LABEL=URL,COLLECTION")
    return label, LT.normalize_endpoint(url), collection


def refuse_undeclared(collection: str, *, allow_scratch: bool) -> None:
    declared = LT.live_collection_names() | {LT.DEFAULT_PROD_ALIAS}
    if collection not in declared and not allow_scratch:
        raise VerifyRefused(
            f"collection {collection!r} is not a declared live name ({sorted(declared)}); "
            "pass --allow-scratch to verify a scratch copy on purpose")


def read_target(url: str, collection: str, receipt: dict, *, vectors: bool = True) -> dict:
    """Read the receipt's ids from one target, by id. Returns the facts the checks need."""
    written = sorted(receipt["written"])
    rows: dict[str, dict] = {}
    for i in range(0, len(written), BATCH):
        rows.update(PD.fetch_rows(url, collection, written[i:i + BATCH],
                                  vectors=["dense"] if vectors else [], batch=BATCH))
    removed = sorted(receipt["removed"])
    still = PD.fetch_payload_fields(url, collection, removed, False, batch=BATCH) if removed else {}
    return {"rows": rows, "removed_present": sorted(still)}


def check_target(receipt: dict, got: dict) -> list[str]:
    """The failures of one target (empty = PASS)."""
    problems: list[str] = []
    rows = got["rows"]
    missing = sorted(set(receipt["written"]) - set(rows))
    if missing:
        problems.append(f"{len(missing)} written id(s) absent, e.g. {missing[:3]}")
    lic = receipt.get("license")
    if lic:
        wrong = sorted(i for i, r in rows.items() if r["payload"].get("license") != lic)
        if wrong:
            problems.append(f"{len(wrong)} row(s) without license {lic!r}, e.g. {wrong[:3]}")
    windowed = {i: r["payload"]["window"] for i, r in rows.items() if "window" in r["payload"]}
    bad = sorted(i for i, w in windowed.items()
                 if not (isinstance(w, dict) and isinstance(w.get("index"), int)
                         and isinstance(w.get("count"), int) and 1 <= w["index"] <= w["count"]
                         and w["count"] > 1))
    if bad:
        problems.append(f"{len(bad)} malformed `window` field(s), e.g. {bad[:3]}")
    if len(windowed) != receipt["window_rows"]:
        problems.append(f"window rows {len(windowed)}, expected {receipt['window_rows']}")
    # optional receipt facts: the written rows sit at this many distinct urls, the windowed rows at this many
    urls = {r["payload"].get("url") for r in rows.values()}
    if "distinct_urls" in receipt and len(urls) != receipt["distinct_urls"]:
        problems.append(f"distinct urls {len(urls)}, expected {receipt['distinct_urls']}")
    wurls = {rows[i]["payload"].get("url") for i in windowed}
    if "window_urls" in receipt and len(wurls) != receipt["window_urls"]:
        problems.append(f"windowed urls {len(wurls)}, expected {receipt['window_urls']}")
    if got["removed_present"]:
        problems.append(f"{len(got['removed_present'])} removed id(s) still present, "
                        f"e.g. {got['removed_present'][:3]}")
    return problems


def row_digests(rows: dict[str, dict]) -> dict[str, tuple[str, str]]:
    return {i: (digest(r["payload"]), digest(r.get("vector") or {})) for i, r in rows.items()}


def compare_pair(a: dict, b: dict) -> list[str]:
    """Differences between two targets' digests of the same ids (empty = equal)."""
    da, db = row_digests(a["rows"]), row_digests(b["rows"])
    problems: list[str] = []
    only = sorted(set(da) ^ set(db))
    if only:
        problems.append(f"{len(only)} id(s) on one side only, e.g. {only[:3]}")
    pay = sorted(i for i in set(da) & set(db) if da[i][0] != db[i][0])
    vec = sorted(i for i in set(da) & set(db) if da[i][1] != db[i][1])
    if pay:
        pa, pb = a["rows"][pay[0]]["payload"], b["rows"][pay[0]]["payload"]
        keys = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
        problems.append(f"{len(pay)} payload digest(s) differ, e.g. {pay[:3]} "
                        f"(fields differing on the first: {keys})")
    if vec:
        problems.append(f"{len(vec)} vector digest(s) differ, e.g. {vec[:3]}")
    return problems


def set_digest(rows: dict[str, dict]) -> str:
    """One sha256 over every `id payload_digest vector_digest` line, sorted by id."""
    return hashlib.sha256("\n".join(f"{i}\t{p}\t{v}" for i, (p, v) in
                                    sorted(row_digests(rows).items())).encode()).hexdigest()


def check_pack_records(receipt: dict, records_path: str | Path) -> list[str]:
    """The door output against the receipt: the id set and the consolidation date."""
    rows = [json.loads(x) for x in Path(records_path).read_text(encoding="utf-8").splitlines()
            if x.strip()]
    ids = [r.get("point_id") for r in rows]
    problems: list[str] = []
    if len(set(ids)) != len(ids):
        problems.append("the pack carries a repeated point_id")
    extra, lost = sorted(set(ids) - set(receipt["written"])), sorted(set(receipt["written"]) - set(ids))
    if extra or lost:
        problems.append(f"pack ids differ from the receipt: {len(extra)} new {extra[:3]}, "
                        f"{len(lost)} missing {lost[:3]}")
    want = receipt.get("consolidation")
    if want:
        got = sorted({str(r.get("source_version")) for r in rows})
        if got != [want]:
            problems.append(f"consolidation {got}, expected [{want!r}]")
    return problems


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--receipt", required=True, help="supersede receipt JSON (see the module doc)")
    ap.add_argument("--target", action="append", default=[], metavar="LABEL=URL,COLLECTION",
                    help="a target to read by id; repeat for the twin and for prod")
    ap.add_argument("--allow-scratch", action="store_true",
                    help="accept a collection that is not a declared live name")
    ap.add_argument("--no-vectors", action="store_true", help="payload digests only")
    ap.add_argument("--pack-records", metavar="RECORDS_JSONL", default=None,
                    help="check the door output against the receipt instead of (or before) the targets")
    args = ap.parse_args(argv)
    try:
        receipt = load_receipt(args.receipt)
        targets = [parse_target(t) for t in args.target]
        for _l, _u, coll in targets:
            refuse_undeclared(coll, allow_scratch=args.allow_scratch)
        if not targets and not args.pack_records:
            raise VerifyRefused("name at least one --target or --pack-records")
    except (VerifyRefused, LT.LiveTargetRefused, OSError, ValueError) as exc:
        print(f"REFUSED {exc}")
        return 2
    ok = True
    if args.pack_records:
        problems = check_pack_records(receipt, args.pack_records)
        ok &= not problems
        print(f"{'PASS' if not problems else 'FAIL'} pack-records {args.pack_records} "
              f"ids={len(receipt['written'])} ids_sha256={ids_sha256(receipt['written'])}"
              + ("" if not problems else " :: " + "; ".join(problems)))
    got: dict[str, dict] = {}
    for label, url, coll in targets:
        try:
            got[label] = read_target(url, coll, receipt, vectors=not args.no_vectors)
            problems = check_target(receipt, got[label])
        except Exception as exc:   # noqa: BLE001 -- an unreadable target is a FAIL, never a pass
            problems = [f"unreadable: {type(exc).__name__}: {exc}"]
        ok &= not problems
        line = f"{'PASS' if not problems else 'FAIL'} {label} {url} {coll}"
        if label in got and not problems:
            line += (f" written={len(got[label]['rows'])}/{len(receipt['written'])}"
                     f" removed_absent={len(receipt['removed'])}/{len(receipt['removed'])}"
                     f" window={receipt['window_rows']}/{receipt['window_rows']}"
                     f" urls={len({r['payload'].get('url') for r in got[label]['rows'].values()})}"
                     f" rows_sha256={set_digest(got[label]['rows'])}")
        elif problems:
            line += " :: " + "; ".join(problems)
        print(line)
    labels = [t[0] for t in targets]
    for other in labels[1:]:
        if labels[0] in got and other in got:
            problems = compare_pair(got[labels[0]], got[other])
        else:
            problems = ["a side was unreadable"]
        ok &= not problems
        print(f"{'PASS' if not problems else 'FAIL'} pair {labels[0]}=={other}"
              + (f" payload_and_vector_equal={len(receipt['written'])}/{len(receipt['written'])}"
                 if not problems else " :: " + "; ".join(problems)))
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
