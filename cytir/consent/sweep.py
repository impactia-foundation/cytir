"""Run the approved Task 427 consent sweep in an isolated evidence namespace.

The input is the sealed per-origin/per-mode frontier.  Every batch resolves in
a throw-away cache between two liveness controls; only a batch with both
controls green is appended to the private evidence set.  Results are never
copied to the shared consent cache and never applied to corpus or production.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from cytir.consent import resolver as cr


SCHEMA = "aw.task427-consent-reresolution.v1"
CONTROL_ORIGIN = "https://www.admin.ch"
# --- liveness control pacing (task 485) --------------------------------------
# The control decides whether a batch may be recorded, and only a real HTTP 200
# from CONTROL_ORIGIN is green -- that rule is the evidence discipline and is
# unchanged here. What changed is that the control used to be a single shot
# with no retry and no pacing, and the discard path `continue`d with no pause.
# When the control origin began refusing our egress IP the driver answered by
# re-probing it as fast as the loop could turn: measured on the task 464 sweep
# at ~1.5 probes/s, ~550 discarded batches per pass, 6,880 discards across
# eleven passes, with the green rate falling 21% -> 1.5% -> 0% as the block
# hardened. The spin was feeding the block that was starving the sweep.
CONTROL_ATTEMPTS = 3               # one RST is an instrument failure, not evidence
CONTROL_RETRY_SLEEP = (1.0, 3.0)   # between attempts within one control
# Backoff after N consecutive failed controls. The first two are free so an
# ordinary flap costs nothing; a sustained block ramps to a minute apart.
CONTROL_BACKOFF = (0.0, 0.0, 5.0, 10.0, 20.0, 40.0, 60.0)
CONTROL_FAIL_STREAK = 12           # then end the pass and let the origin recover
# ``held_invalid_evidence`` is READ-ONLY history: it is the terminal the
# retracted evidence gate wrote (task 466, Tim 2026-08-28).  Sealed artifacts
# still carry it and must stay loadable; ``classify`` never emits it again.
DISPOSITIONS = {
    "admitted", "admitted_search_only", "declined",
    "held_invalid_evidence", "held_unknown",
}
# Task 721: the shared consent cache is the resolver's own CACHE_DIR (env
# AW_CONSENT_CACHE_DIR, else ~/aw-data/consent) -- the old literal root-home
# spelling is gone; the default resolves to the same directory when run as root.
FORBIDDEN_CACHE_ROOTS = tuple({
    Path(cr.CACHE_DIR).resolve(),
    Path(os.path.expanduser("~/aw-data/consent")).resolve(),
})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_private(path: Path) -> None:
    real = path.resolve()
    for root in FORBIDDEN_CACHE_ROOTS:
        if real == root or root in real.parents:
            raise ValueError(f"private evidence path resolves inside shared cache: {real}")


def control_ok(timeout: int = 25, attempts: int = CONTROL_ATTEMPTS) -> bool:
    """Is the control origin answering us right now?

    Green still means one thing only: an HTTP 200 from CONTROL_ORIGIN. The
    retries do not widen that -- they stop a single refused connection from
    deciding it. A control is an instrument, and an instrument that is being
    rate-limited says nothing about whether the batch it brackets was answered
    honestly. Attempts are bounded and spaced so the probe cannot itself become
    the flood.
    """
    for attempt in range(max(1, attempts)):
        try:
            result = subprocess.run(
                ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
                 "--max-time", str(timeout), "-A", cr.UA_STR,
                 CONTROL_ORIGIN + "/robots.txt"],
                capture_output=True, text=True, timeout=timeout + 5, check=False)
            if result.stdout.strip() == "200":
                return True
        except Exception:  # noqa: BLE001 - a failed control is simply not green
            pass
        if attempt + 1 < max(1, attempts):
            time.sleep(CONTROL_RETRY_SLEEP[min(attempt, len(CONTROL_RETRY_SLEEP) - 1)])
    return False


def control_backoff(streak: int) -> float:
    """Seconds to wait after ``streak`` consecutive failed controls."""
    return CONTROL_BACKOFF[min(max(streak, 1) - 1, len(CONTROL_BACKOFF) - 1)]


def read_frontier(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    keys = [(r["origin"], r["mode"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("frontier contains duplicate origin/mode keys")
    for row in rows:
        if row["mode"] not in {"cc", "direct"}:
            raise ValueError(f"live frontier contains non-live mode: {row['mode']}")
        if row["scheme"] not in {"http", "https"}:
            raise ValueError(f"frontier contains invalid scheme: {row['scheme']!r}")
        if row["origin"] != f"{row['scheme']}://{row['authority']}":
            raise ValueError(
                f"frontier origin does not preserve scheme/authority: {row['origin']}")
    return rows


def evidence_defect(receipt: dict) -> str | None:
    """RETRACTED GATE — frozen, and load-bearing for nothing but the past.

    Tim RETRACTED this gate on 2026-08-28 (task 466, Q3(a)): it was never
    ratified by any ADR or task doc, it overrode receipts the ratified resolver
    had already resolved as permissive, and it misfired on spark.apache.org,
    numpy.org, rubyonrails.org and hudoc.echr.coe.int — 70,314 points held on a
    rule nobody adopted.

    The body is kept BYTE-FAITHFUL so ``classify_pre_retraction`` can still
    re-derive artifacts sealed while it was live; every defect it names is
    fixed in ``evidence_observation``.  Never call it to decide anything.
    """
    if receipt.get("status") != "ok" or receipt.get("search_ok") is not True:
        return None
    robots = (receipt.get("signals") or {}).get("robots") or {}
    if robots.get("http_status") != 200:
        return None
    body = str(receipt.get("_robots_body") or "")
    lower = body.lstrip().lower()
    if lower.startswith("<!doctype html") or lower.startswith("<html"):
        return "robots_http_200_html_body"
    # A non-empty 200 with neither a UA group nor a structured signal is not
    # evidence of permission.  Empty 200 remains the resolver's documented
    # no-rules-published case.
    if (body.strip() and not robots.get("ua_tokens_seen")
            and not robots.get("content_signal")
            and not robots.get("content_usage")):
        return "robots_http_200_no_policy_directives"
    return None


def evidence_observation(receipt: dict) -> str | None:
    """DIAGNOSTIC. Describes the robots evidence; decides nothing.

    Two defects of the retracted gate are fixed here so the observation is at
    least true:

      * a body carrying only non-group records (a bare ``Sitemap:``) is a
        WELL-FORMED robots.txt and an allow-all under RFC 9309 §2.2.1 — the
        misfire on spark.apache.org, numpy.org and rubyonrails.org;
      * a directive group hidden behind a junk prefix is now recovered by the
        resolver's own parser, so ``ua_tokens_seen`` is no longer empty for
        asvhg-foot.com and its kind.

    What remains observable is a body that is not robots syntax at all (an
    HTML page served at ``/robots.txt`` — hudoc.echr.coe.int) or one with no
    parsable record in it.  Worth knowing; never a disposition.
    """
    if receipt.get("status") != "ok" or receipt.get("search_ok") is not True:
        return None
    robots = (receipt.get("signals") or {}).get("robots") or {}
    if robots.get("http_status") != 200:
        return None
    body = str(receipt.get("_robots_body") or "")
    if not body.strip():
        return None                     # empty 200 = no rules published (open)
    shape = cr.robots_body_shape(body)
    if shape["html"]:
        return "robots_http_200_html_body"
    if shape["directive_lines"] == 0:
        return "robots_http_200_no_parsable_record"
    return None


def classify(receipt: dict) -> tuple[str, dict]:
    """The RATIFIED consent rule, and only it (task 466).

    Two corrections carry Tim's 2026-08-28 rulings:
      (a) the ``evidence_defect`` gate is retracted — the resolver's verdict
          stands, and its observation rides along as a diagnostic;
      (b) an explicit refusal recorded on a record whose OTHER leg never
          answered is a decline, not a held_unknown.  Amendment 3's own text
          says explicit refusals are never overridden, and the euipo precedent
          classes a host that refuses our probes as declined residue.
    """
    observation = evidence_observation(receipt)
    diagnostic = {"evidence_observation": observation} if observation else {}
    if receipt.get("status") != "ok":
        if cr.explicit_search_refusal(receipt):
            return "declined", {"resolver_classification_preserved": True,
                                "explicit_refusal_on_unresolved_record": True,
                                **diagnostic}
        return "held_unknown", dict(diagnostic)
    if receipt.get("search_ok") is not True:
        # ADR-096 remains proposed.  Record Content-Signal in the raw receipt,
        # but never reinterpret the resolver's ratified classification here.
        return "declined", {"resolver_classification_preserved": True,
                            **diagnostic}
    train_ok = receipt.get("train_ok") is True
    return ("admitted" if train_ok else "admitted_search_only", {
        "resolver_classification_preserved": True,
        "effective_search_ok": True,
        "effective_train_ok": train_ok,
        **diagnostic,
    })


def classify_pre_retraction(receipt: dict) -> tuple[str, dict]:
    """The classifier as it ran before 2026-08-28, for re-deriving old rows.

    Sealed artifacts must stay verifiable against the code that produced them;
    a validator that cannot re-derive a sealed row cannot tell a superseded
    rule from a corrupted file.  Nothing new is ever classified with this.
    """
    defect = evidence_defect(receipt)
    if defect:
        return "held_invalid_evidence", {"evidence_defect": defect}
    if receipt.get("status") != "ok":
        return "held_unknown", {}
    if receipt.get("search_ok") is not True:
        return "declined", {"resolver_classification_preserved": True}
    train_ok = receipt.get("train_ok") is True
    return ("admitted" if train_ok else "admitted_search_only", {
        "resolver_classification_preserved": True,
        "effective_search_ok": True,
        "effective_train_ok": train_ok,
    })


def validate_results(frontier_rows: list[dict], recorded: list[dict]) -> dict:
    """Bind private results one-to-one to the sealed frontier.

    Missing keys are valid only while a run is in progress.  Duplicates,
    unexpected keys, altered source accounting, and unknown dispositions are
    integrity failures at every stage.
    """
    source_by_key = {(row["origin"], row["mode"]): row for row in frontier_rows}
    seen = set()
    disposition_keys = Counter()
    disposition_documents = Counter()
    source_fields = (
        "origin", "scheme", "authority", "host", "mode", "documents",
        "waves", "sample_urls",
    )
    for ordinal, row in enumerate(recorded, 1):
        key = (row.get("origin"), row.get("mode"))
        if key in seen:
            raise ValueError(f"duplicate result key at row {ordinal}: {key}")
        seen.add(key)
        source = source_by_key.get(key)
        if source is None:
            raise ValueError(f"result key absent from frontier at row {ordinal}: {key}")
        for field in source_fields:
            if row.get(field) != source.get(field):
                raise ValueError(
                    f"result row {ordinal} changed frontier field {field!r}")
        disposition = row.get("disposition")
        if disposition not in DISPOSITIONS:
            raise ValueError(
                f"result row {ordinal} has invalid disposition {disposition!r}")
        disposition_keys[disposition] += 1
        disposition_documents[disposition] += int(row["documents"])
    frontier_keys = set(source_by_key)
    missing = frontier_keys - seen

    def key_set_sha256(keys) -> str:
        encoded = json.dumps(sorted(keys), ensure_ascii=False,
                             separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    return {
        "unique": True,
        "exact_key_set": not missing,
        "frontier_keys": len(frontier_keys),
        "result_keys": len(seen),
        "missing_keys": len(missing),
        "extra_keys": 0,
        "frontier_key_set_sha256": key_set_sha256(frontier_keys),
        "result_key_set_sha256": key_set_sha256(seen),
        "origin_mode_keys_by_disposition": dict(sorted(disposition_keys.items())),
        "documents_by_disposition": dict(sorted(disposition_documents.items())),
    }


def batch_groups(rows: list[dict], batch_size: int):
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["scheme"], row["mode"]), []).append(row)
    for (scheme, mode), values in sorted(groups.items()):
        values.sort(key=lambda r: r["origin"])
        for start in range(0, len(values), batch_size):
            yield scheme, mode, values[start:start + batch_size]


def run(frontier: Path, frontier_run: Path, out_dir: Path,
        batch_size: int, concurrency: int) -> dict:
    assert_private(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "private-cache"
    batch_root = out_dir / "batch-work"
    cache_root.mkdir(exist_ok=True)
    batch_root.mkdir(exist_ok=True)
    results_path = out_dir / "results.jsonl"
    log_path = out_dir / "driver.log"
    rows = read_frontier(frontier)
    frontier_manifest = json.loads(frontier_run.read_text())
    artifact_claim = frontier_manifest["artifacts"]["live_frontier"]
    if sha256_file(frontier) != artifact_claim["compressed_sha256"]:
        raise ValueError("frontier compressed SHA-256 mismatch")

    recorded = []
    if results_path.exists():
        with results_path.open(encoding="utf-8") as fh:
            recorded = [json.loads(line) for line in fh if line.strip()]
    validate_results(rows, recorded)
    done = {(row["origin"], row["mode"]) for row in recorded}
    todo = [row for row in rows if (row["origin"], row["mode"]) not in done]
    with log_path.open("a", buffering=1) as log:
        log.write(f"{time.strftime('%FT%TZ', time.gmtime())} start "
                  f"todo={len(todo)} done={len(done)} frontier={len(rows)}\n")
        batch_number = 0
        control_failures = 0

        def control_failed(kind: str, size: int) -> bool:
            """Record a failed control, pace the next attempt, and say whether
            this pass should end. Returning to the caller's `continue` leaves
            the batch for a later pass exactly as before."""
            nonlocal control_failures
            control_failures += 1
            log.write(f"batch={batch_number} control_{kind}=FAIL discarded={size}"
                      f" streak={control_failures}\n")
            if control_failures >= CONTROL_FAIL_STREAK:
                log.write(f"pass ends: {control_failures} consecutive failed "
                          f"controls against {CONTROL_ORIGIN} — pausing rather "
                          f"than re-probing a control that is not answering\n")
                return True
            time.sleep(control_backoff(control_failures))
            return False

        for scheme, mode, batch in batch_groups(todo, batch_size):
            batch_number += 1
            batch_dir = batch_root / f"batch-{batch_number:05d}"
            if batch_dir.exists():
                shutil.rmtree(batch_dir)
            batch_dir.mkdir()
            if not control_ok():
                # The throw-away dir is removed on every discard path; leaving
                # it behind grew batch-work to 1,505 empty directories on the
                # task 464 sweep.
                shutil.rmtree(batch_dir)
                if control_failed("pre", len(batch)):
                    break
                continue
            hosts = [row["authority"] for row in batch]
            cr.resolve_hosts(
                hosts, cache_dir=str(batch_dir), max_age_days=0,
                concurrency=concurrency, progress=False,
                direct_crawl=mode == "direct", origin_scheme=scheme)
            if not control_ok():
                shutil.rmtree(batch_dir)
                if control_failed("post", len(batch)):
                    break
                continue
            control_failures = 0

            emitted = []
            for source in batch:
                authority = source["authority"]
                receipt_path = cr._cache_path(
                    str(batch_dir), authority,
                    direct_crawl=mode == "direct", origin_scheme=scheme)
                receipt = json.loads(receipt_path.read_text())
                disposition, effective = classify(receipt)
                emitted.append({
                    **source,
                    "disposition": disposition,
                    "effective": effective,
                    "receipt": receipt,
                    "control": {"pre": True, "post": True,
                                "origin": CONTROL_ORIGIN},
                })
                stable_path = cr._cache_path(
                    str(cache_root), authority,
                    direct_crawl=mode == "direct", origin_scheme=scheme)
                stable_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(receipt_path, stable_path)
            with results_path.open("a", encoding="utf-8") as out:
                for record in emitted:
                    out.write(json.dumps(record, ensure_ascii=False,
                                         sort_keys=True) + "\n")
            counts = Counter(r["disposition"] for r in emitted)
            log.write(f"batch={batch_number} scheme={scheme} mode={mode} "
                      f"recorded={len(emitted)} counts={dict(counts)}\n")
            shutil.rmtree(batch_dir)

    # This may be incomplete after a control failure; only a complete frontier
    # is sealed.  The same namespace is safely resumable.
    recorded = []
    if results_path.exists():
        with results_path.open(encoding="utf-8") as fh:
            recorded = [json.loads(line) for line in fh if line.strip()]
    integrity = validate_results(rows, recorded)
    complete = integrity["exact_key_set"]
    run_receipt = {
        "schema": SCHEMA,
        "status": "complete" if complete else "running",
        "producer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "resolver": str(Path(cr.__file__).resolve()),
            "resolver_sha256": sha256_file(Path(cr.__file__).resolve()),
            "revision": os.environ.get("AW_PRODUCER_REVISION", "uncommitted"),
        },
        "frontier": {"path": str(frontier), "sha256": sha256_file(frontier),
                     "keys": len(rows)},
        "results": {"path": str(results_path), "keys": len(recorded),
                    "sha256": sha256_file(results_path) if results_path.exists() else None},
        "counts": {
            "origin_mode_keys": integrity["origin_mode_keys_by_disposition"],
            "documents": integrity["documents_by_disposition"],
        },
        "result_integrity": {
            "unique": integrity["unique"],
            "exact_key_set": integrity["exact_key_set"],
            "missing_keys": integrity["missing_keys"],
            "extra_keys": integrity["extra_keys"],
            "frontier_key_set_sha256": integrity["frontier_key_set_sha256"],
            "result_key_set_sha256": integrity["result_key_set_sha256"],
            "documents_rederived_from_validated_results": True,
        },
        "conservation": {"lhs": integrity["result_keys"],
                         "rhs": integrity["frontier_keys"],
                         "pass": complete},
        "safety": {"private_cache": str(cache_root), "shared_cache_writes": False,
                   "verdict_application": False, "production_writes": False},
    }
    (out_dir / "RUN.json").write_text(
        json.dumps(run_receipt, indent=2, sort_keys=True) + "\n")
    return run_receipt


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("--frontier", required=True, type=Path)
    parser.add_argument("--frontier-run", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.concurrency not in {1, 2}:
        parser.error("batch-size must be positive and concurrency must be 1 or 2")
    receipt = run(args.frontier, args.frontier_run, args.out_dir,
                  args.batch_size, args.concurrency)
    print(json.dumps(receipt["counts"], sort_keys=True))
    return 0 if receipt["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
