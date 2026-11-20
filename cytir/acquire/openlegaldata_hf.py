"""Resumably acquire (and verify) the OpenLegalData HF dump-20260520 parquet shards.

One module, two verbs (task 1235; formerly two pipeline scripts):
``python -m cytir.acquire.openlegaldata_hf acquire ...`` and ``... verify ...``.

Adapts the OpenAlex snapshot-parts acquisition pattern (resumable
range GETs, per-file sha256 receipts, sealed acquisition RUN) to a gated
HuggingFace dataset instead of an anonymous S3 bucket. Task 489 executes the
base-layer acquisition route staged by ``sources/de/openlegaldata.yaml``
(``routes[0].acquire``, ``source_owned_bulk_dump``) once task 473/474's
account/token gate was cleared by a human (plan.md ``## Blocked`` item 1).

WHAT DIFFERS FROM THE OPENALEX PATTERN (and why)
-------------------------------------------------
* Auth: every request needs ``Authorization: Bearer <token>``. The token is
  read once from ``--token-file`` (never logged, never placed in argv) and
  written to a private ``curl -K`` config file (``header = "Authorization: ..."``)
  so it never appears in ``ps`` output either. The config file is 0600 and
  removed at process exit.
* Manifest: OpenAlex had a static, pre-existing S3 file manifest. HuggingFace
  has no equivalent static file; the manifest is *discovered live* from the
  dataset's own ``/api/datasets/...`` metadata (siblings list) at run start,
  and the repo commit SHA it reports is pinned for every subsequent request
  so a mid-run upstream change can't quietly swap file identity underneath a
  resumed download (see ``discover_frontier``).
* Identity headers: HF's resolve endpoint answers with a 302 to a signed,
  short-lived Xet-CDN URL. The *redirect response itself* carries the
  file's true identity (``x-linked-size``, ``x-linked-etag``, matching
  OpenAlex's ``content-length``/``etag``) plus ``x-repo-commit`` (matching
  OpenAlex's ``x-amz-version-id``). ``expected_bytes`` is therefore *learned*
  from this HEAD response, not pre-supplied by the manifest (verified live,
  2026-08-31, from the build box: HEAD on ``resolve/<rev>/<path>`` returns the 302
  with these headers directly, no need to follow it).
* Body GETs use ``--location`` (follow the redirect) since the real bytes
  live behind it; Range/``--continue-at`` resume was verified live to survive
  the redirect correctly (two sequential ranged GETs against the same URL
  returned distinct, correctly-offset byte ranges). curl's ``%{json}``
  ``url`` field (the as-requested URL) is checked instead of ``url_effective``
  (which reports the CDN target, and rotates every 302).
* Sequential only (task instruction: "spindle courtesy irrelevant, writes
  are sequential and modest") — no ThreadPoolExecutor, one shard at a time.
  Per-part flock is still taken so two concurrent invocations can't race.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time

import pyarrow.parquet as pq

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS
import cytir.core.user_agent as UA


VERSION = "openlegaldata-hf-dump-acquisition.v1"
PRODUCER = "aw/acquire/openlegaldata_hf.py"

DATASET = "openlegaldata/court-decisions-germany"
CONFIG = "dump-20260520"
SHARD_COUNT = 54
API_URL = f"https://huggingface.co/api/datasets/{DATASET}"
RESOLVE_TMPL = (
    f"https://huggingface.co/datasets/{DATASET}/resolve/{{revision}}/{{path}}")
SHARD_RE = re.compile(
    rf"^{re.escape(CONFIG)}/train-(\d{{5}})-of-{SHARD_COUNT:05d}\.parquet$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

# The pack's own declared totals (sources/de/openlegaldata.yaml
# admission.evidence; cross-checked live 2026-08-31 against
# cardData.dataset_info[config_name=dump-20260520]: splits[0].num_examples /
# download_size). Reported as a cross-check in the RUN only -- the row-count
# TRUTH comes from reading the downloaded parquet files (a separate
# verification script), never from this declared metadata, per the counting-
# boundary discipline in notes/enumerated-is-not-staged.md.
EXPECTED_ROWS_HINT = 423941
EXPECTED_DOWNLOAD_BYTES_HINT = 10678094516


def _vantage() -> str:
    """The acquisition vantage, from the one acquisition policy (data, not a literal here)."""
    from cytir.core.paths import data_path
    policy = json.loads(data_path("acquisition-policy.v1.json").read_text())
    return policy["direct_fetch"]["vantage_order"][0]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_once(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"immutable evidence collision: {path}")
    else:
        atomic_publish(path, value)


def _last_header(headers: bytes, name: str) -> str | None:
    matches = re.findall(rf"(?im)^{re.escape(name)}:\s*([^\r\n]+)",
                         headers.decode("iso-8859-1"))
    return matches[-1].strip() if matches else None


class TokenConfig:
    """A private curl ``-K`` file carrying the bearer header, never argv."""

    def __init__(self, token_file: Path):
        mode = stat.S_IMODE(token_file.stat().st_mode)
        if mode != 0o600:
            raise ValueError(
                f"HF token file must be 0600, found {oct(mode)}: {token_file}")
        token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError(f"HF token file is empty: {token_file}")
        fd, name = tempfile.mkstemp(prefix=".hf-curl-auth-", suffix=".cfg")
        self.path = Path(name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f'header = "Authorization: Bearer {token}"\n')
        except BaseException:
            self.path.unlink(missing_ok=True)
            raise
        del token  # never retained beyond writing the private config file

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, *exc) -> None:
        self.path.unlink(missing_ok=True)


def _curl_json(url: str, *, token_config: Path, ua: str, extra: list[str] | None = None
              ) -> tuple[dict, dict, bytes]:
    result = subprocess.run(
        ["curl", "-K", str(token_config), "-4", "--fail-with-body", "--silent",
         "--show-error", "--location", "--connect-timeout", "20", "--max-time", "60",
         "--user-agent", ua, "--write-out", "\n%{json}"] + (extra or []) + [url],
        capture_output=True, check=False)
    stdout = result.stdout
    body, _, metrics_line = stdout.rpartition(b"\n")
    metrics = json.loads(metrics_line) if metrics_line else {}
    if result.returncode != 0:
        raise ValueError(
            f"HF API request failed (curl rc={result.returncode}, "
            f"http={metrics.get('http_code')}): {url}\n"
            f"{result.stderr.decode(errors='replace')[:2000]}")
    return json.loads(body), metrics, body


def discover_frontier(*, token_config: Path, ua: str, root: Path
                      ) -> tuple[list[dict], str, dict, list[Path]]:
    """Fetch the dataset's own file listing + pinned commit, live.

    Returns (rows, revision, declared_hint, evidence_artifact_paths). Each
    row carries ``relative_path`` and ``url`` only -- ``expected_bytes`` is
    learned per-part from the HEAD response (see module docstring).

    Resumable like ``acquire_part``: on a re-run, the previously fetched
    ``dataset_info.json`` is reused verbatim rather than re-fetched. This is
    required, not cosmetic -- curl's ``%{json}`` write-out metrics embed live
    timing/connection fields that differ on every request even when the body
    is byte-identical, so treating a fresh fetch's metrics as write-once
    evidence would collide on every resume (hit live 2026-08-31, task 489:
    the first run's crash on shard 3 required a restart, and the naive
    always-refetch version failed immediately on this exact collision).
    """
    manifest_root = root / "original/manifest"
    body_path = manifest_root / "dataset_info.json"
    metrics_path = manifest_root / "dataset_info.curl-metrics.json"
    if body_path.exists():
        raw_body = body_path.read_bytes()
        info = json.loads(raw_body)
    else:
        info, metrics, raw_body = _curl_json(API_URL, token_config=token_config, ua=ua)
        manifest_root.mkdir(parents=True, exist_ok=True)
        _write_once(body_path, raw_body)
        _write_once(metrics_path, (canonical_json(metrics) + "\n").encode())
    if info.get("id") != DATASET:
        raise ValueError(f"unexpected HF dataset identity: {info.get('id')!r}")
    revision = str(info.get("sha") or "")
    if not REVISION_RE.fullmatch(revision):
        raise ValueError(f"HF dataset_info missing a valid pinned revision: {revision!r}")
    siblings = info.get("siblings") or []
    matched: dict[int, str] = {}
    for sibling in siblings:
        name = str((sibling or {}).get("rfilename") or "")
        match = SHARD_RE.match(name)
        if match:
            matched[int(match.group(1))] = name
    if sorted(matched) != list(range(SHARD_COUNT)):
        raise ValueError(
            f"HF siblings do not list exactly shards 0..{SHARD_COUNT - 1} of "
            f"{CONFIG}: found indices {sorted(matched)}")
    rows = [{"relative_path": matched[index],
             "url": RESOLVE_TMPL.format(revision=revision, path=matched[index]),
             "revision": revision}
            for index in range(SHARD_COUNT)]

    declared_hint = {"revision": revision, "gated": info.get("gated")}
    for entry in ((info.get("cardData") or {}).get("dataset_info") or []):
        if entry.get("config_name") == CONFIG:
            splits = entry.get("splits") or [{}]
            declared_hint.update({
                "declared_num_examples": splits[0].get("num_examples"),
                "declared_download_size": entry.get("download_size"),
                "declared_dataset_size": entry.get("dataset_size"),
            })
            break

    return rows, revision, declared_hint, [body_path, metrics_path]


def _validate_head(headers: Path, metrics: Path, row: dict) -> dict:
    metric = json.loads(metrics.read_text(encoding="utf-8"))
    raw = headers.read_bytes()
    status = int(metric.get("http_code") or 0)
    linked_size = _last_header(raw, "x-linked-size")
    linked_etag = _last_header(raw, "x-linked-etag")
    repo_commit = _last_header(raw, "x-repo-commit")
    if (status != 302 or metric.get("url") != row["url"]
            or _last_header(raw, "accept-ranges") != "bytes"
            or not linked_size or not linked_size.isdigit()
            or not linked_etag or not repo_commit):
        raise ValueError(f"OpenLegalData HF HEAD identity mismatch: {row['relative_path']}")
    if repo_commit != row["revision"]:
        raise ValueError(
            f"OpenLegalData HF repo commit changed mid-run for "
            f"{row['relative_path']}: pinned {row['revision']!r}, HEAD saw "
            f"{repo_commit!r} -- upstream mutated the dataset during this "
            f"acquisition; stop and re-discover the frontier.")
    return {"http_status": status, "content_length": int(linked_size),
            "x_linked_etag": linked_etag, "x_repo_commit": repo_commit}


def _artifact(path: Path, root: Path) -> dict:
    return {"uri": f"artifact://{path.relative_to(root).as_posix()}",
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _close_interrupted(attempts: Path, partial: Path) -> None:
    for started in sorted(attempts.glob("attempt-*.started")):
        prefix = started.with_suffix("")
        completed = Path(str(prefix) + ".completed")
        if completed.exists():
            continue
        _write_once(Path(str(prefix) + ".completed"), (_now() + "\n").encode())
        _write_once(Path(str(prefix) + ".rc"), b"130\n")
        _write_once(Path(str(prefix) + ".end_offset"),
                    (str(partial.stat().st_size if partial.exists() else 0) + "\n").encode())
        metrics = Path(str(prefix) + ".metrics")
        if not metrics.exists() or not metrics.read_text(errors="ignore").strip():
            _write_once(Path(str(prefix) + ".interrupted.json"),
                        (json.dumps({"status": "interrupted_no_terminal_curl_metrics"},
                                    sort_keys=True) + "\n").encode())


def _attempt_rows(attempts: Path, partial: Path, expected_bytes: int) -> tuple[list[dict], list[Path]]:
    _close_interrupted(attempts, partial)
    values, evidence = [], []
    previous_end = None
    for started in sorted(attempts.glob("attempt-*.started")):
        prefix = started.with_suffix("")
        required = [Path(str(prefix) + suffix) for suffix in
                    (".started", ".completed", ".rc", ".start_offset", ".end_offset")]
        if not all(path.is_file() for path in required):
            raise ValueError("incomplete OpenLegalData range-attempt evidence")
        start, end = int(required[3].read_text()), int(required[4].read_text())
        if start < 0 or end < start or end > expected_bytes:
            raise ValueError("invalid OpenLegalData attempt offsets")
        if previous_end is not None and start != previous_end:
            raise ValueError("non-conservative OpenLegalData attempt offset chain")
        previous_end = end
        extras = [Path(str(prefix) + suffix) for suffix in
                  (".headers", ".metrics", ".interrupted.json")]
        present = required + [path for path in extras if path.is_file()]
        evidence.extend(present)
        values.append({"attempt": prefix.name, "start_offset": start,
                       "end_offset": end, "return_code": int(required[2].read_text()),
                       "evidence": [{"name": path.name, "bytes": path.stat().st_size,
                                     "sha256": sha256_file(path)} for path in present]})
    return values, evidence


def acquire_part(*, root: Path, row: dict, token_config: Path, ua: str,
                 limit_rate: str) -> dict:
    key = hashlib.sha256(row["relative_path"].encode()).hexdigest()
    receipt_path = root / "receipts" / key[:2] / f"{key}.json"
    body = root / "original/parquet" / row["relative_path"]
    partial = Path(str(body) + ".partial")
    evidence_root = root / "transport" / key[:2] / key
    attempts = evidence_root / "attempts"
    head_headers = evidence_root / "head.headers"
    head_metrics = evidence_root / "head.metrics.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("relative_path") != row["relative_path"]
                or not body.is_file() or sha256_file(body) != receipt["body"]["sha256"]
                or body.stat().st_size != receipt["body"]["bytes"]):
            raise ValueError("invalid immutable OpenLegalData part receipt")
        return receipt
    attempts.mkdir(parents=True, exist_ok=True)
    if not head_headers.exists() or not head_metrics.exists():
        tmp_h, tmp_m = Path(str(head_headers) + ".tmp"), Path(str(head_metrics) + ".tmp")
        with tmp_m.open("wb") as metrics_out:
            subprocess.run(["curl", "-K", str(token_config), "-4", "--fail-with-body",
                            "--silent", "--show-error", "--retry", "5",
                            "--retry-all-errors", "--connect-timeout", "30",
                            "--max-time", "60", "--user-agent", ua, "--head",
                            "--dump-header", str(tmp_h), "--write-out", "%{json}",
                            "--output", "/dev/null", row["url"]],
                           check=True, stdout=metrics_out)
        os.replace(tmp_h, head_headers); os.replace(tmp_m, head_metrics)
    head = _validate_head(head_headers, head_metrics, row)
    expected_bytes = head["content_length"]
    body.parent.mkdir(parents=True, exist_ok=True)
    if body.exists() and body.stat().st_size != expected_bytes:
        raise ValueError("unreceipted OpenLegalData body has wrong byte count")
    _close_interrupted(attempts, partial)
    failures = 0
    while not body.exists() and (partial.stat().st_size if partial.exists() else 0) < expected_bytes:
        sequence = len(list(attempts.glob("attempt-*.started"))) + 1
        prefix = attempts / f"attempt-{sequence:04d}"
        start = partial.stat().st_size if partial.exists() else 0
        _write_once(Path(str(prefix) + ".start_offset"), (str(start) + "\n").encode())
        _write_once(Path(str(prefix) + ".started"), (_now() + "\n").encode())
        headers, metrics = Path(str(prefix) + ".headers"), Path(str(prefix) + ".metrics")
        command = ["curl", "-K", str(token_config), "-4", "--fail-with-body",
                   "--location", "--silent", "--show-error", "--continue-at", "-",
                   "--connect-timeout", "30", "--max-time", "0", "--limit-rate",
                   limit_rate, "--user-agent", ua, "--dump-header", str(headers),
                   "--write-out", "%{json}", row["url"], "--output", str(partial)]
        with metrics.open("wb") as metrics_out:
            completed = subprocess.run(command, stdout=metrics_out, check=False)
        _write_once(Path(str(prefix) + ".completed"), (_now() + "\n").encode())
        _write_once(Path(str(prefix) + ".rc"), (str(completed.returncode) + "\n").encode())
        # A connect-stage failure (e.g. TCP refused before any bytes/headers
        # arrive) can leave curl's --output target never created at all, even
        # on a fresh (start_offset==0) attempt -- guard the same way every
        # other read of this file already does, or a transient network blip
        # crashes the whole run instead of retrying (observed live 2026-08-31:
        # "Connection refused" on shard 3's first attempt with no .partial
        # file yet).
        _write_once(Path(str(prefix) + ".end_offset"),
                    (str(partial.stat().st_size if partial.exists() else 0) + "\n").encode())
        failures = 0 if completed.returncode == 0 else failures + 1
        if failures >= 20:
            raise ValueError("too many consecutive OpenLegalData transfer failures")
        if completed.returncode:
            time.sleep(5)
    source_body = body if body.exists() else partial
    if source_body.stat().st_size != expected_bytes:
        raise ValueError("OpenLegalData part exceeds/undershoots learned byte count")
    attempts_rows, attempt_evidence = _attempt_rows(attempts, partial, expected_bytes)
    if not attempts_rows or attempts_rows[-1]["return_code"] != 0:
        raise ValueError("no terminal successful OpenLegalData transfer attempt")
    terminal = attempts_rows[-1]
    terminal_prefix = attempts / terminal["attempt"]
    terminal_metrics = json.loads(Path(str(terminal_prefix) + ".metrics").read_text())
    terminal_headers = Path(str(terminal_prefix) + ".headers").read_bytes()
    status = int(terminal_metrics.get("http_code") or 0)
    if terminal_metrics.get("url") != row["url"] or status not in (200, 206):
        raise ValueError("invalid terminal OpenLegalData HTTP response")
    if terminal["start_offset"] > 0:
        content_range = _last_header(terminal_headers, "content-range")
        expected_range = f"bytes {terminal['start_offset']}-{expected_bytes - 1}/{expected_bytes}"
        if status != 206 or content_range != expected_range:
            raise ValueError("resumed OpenLegalData response has non-conservative Content-Range")
    if not body.exists():
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, body)
        dir_fd = os.open(body.parent, os.O_RDONLY); os.fsync(dir_fd); os.close(dir_fd)
    receipt = {
        "schema": "aw.openlegaldata-hf-parquet-part-receipt.v1", "version": VERSION,
        "status": "complete", **row, "source_head": head,
        "body": _artifact(body, root),
        "transport_evidence": [_artifact(path, root) for path in
                               [head_headers, head_metrics] + attempt_evidence],
        "attempts": attempts_rows,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_publish(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return receipt


def run(*, root: Path, token_file: Path, limit_rate: str = "8M") -> dict:
    root.mkdir(parents=True, exist_ok=True)
    ua = UA.user_agent("openlegaldata-hf-dump")
    with TokenConfig(token_file) as token_config:
        rows, revision, declared_hint, manifest_artifacts = discover_frontier(
            token_config=token_config, ua=ua, root=root)
        frontier_path = root / "frontier.jsonl"
        body = b"".join((canonical_json(row) + "\n").encode() for row in rows)
        _write_once(frontier_path, body)
        receipts = []
        completed_bytes = 0
        for index, row in enumerate(rows, 1):
            key = hashlib.sha256(row["relative_path"].encode()).hexdigest()
            lock_path = root / "locks" / key[:2] / f"{key}.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            import fcntl
            with lock_path.open("a+b") as handle:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError(
                        f"OpenLegalData part already has an active worker: "
                        f"{row['relative_path']}") from error
                receipt = acquire_part(root=root, row=row, token_config=token_config,
                                       ua=ua, limit_rate=limit_rate)
            receipts.append(receipt)
            completed_bytes += receipt["body"]["bytes"]
            print(f"PROGRESS shard={index}/{len(rows)} bytes={completed_bytes}",
                  flush=True)

    artifact_paths = [root / "frontier.jsonl"] + manifest_artifacts
    for receipt in receipts:
        key = hashlib.sha256(receipt["relative_path"].encode()).hexdigest()
        artifact_paths.append(root / "receipts" / key[:2] / f"{key}.json")
        artifact_paths.append(root / receipt["body"]["uri"].removeprefix("artifact://"))
        artifact_paths.extend(root / item["uri"].removeprefix("artifact://")
                              for item in receipt["transport_evidence"])
    selected_source_bytes = sum(r["body"]["bytes"] for r in receipts)
    summary = {
        "selected_parts": len(rows), "selected_source_bytes": selected_source_bytes,
        "pinned_revision": revision,
        "declared_vs_measured_bytes_drift": selected_source_bytes - EXPECTED_DOWNLOAD_BYTES_HINT,
    }
    stage = SS.seal(
        artifact_root=root, artifact_paths=artifact_paths,
        receipt_path=root / "stages/original/receipt.json", records=len(rows),
        fmt="exact-openlegaldata-hf-parquet+dataset-info+HF-HEAD+range-transport",
        logical_sha256=hashlib.sha256(body).hexdigest(),
        producer_id=PRODUCER, producer_version=VERSION)
    result = {
        "schema": "aw.openlegaldata-hf-dump-acquisition-run.v1", "version": VERSION,
        "status": "complete", "dataset": DATASET, "config": CONFIG,
        "route": "source_owned_bulk_dump", "vantage": _vantage(),
        "license": "public_domain_official_work (UrhG §5)",
        "counts": summary, "declared_hint": declared_hint,
        "stages": {"original": stage},
        "downstream": ("row-count + 100-row sample verification "
                       "(``verify`` in cytir/acquire/openlegaldata_hf.py), then "
                       "the materializer (not yet built) "
                       "for materialization -- out of scope for this run"),
    }
    result_path = root / "RUN.json"
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if result_path.exists():
        if sha256_file(result_path) != hashlib.sha256(data).hexdigest():
            raise ValueError("immutable RUN collision")
    else:
        atomic_publish(result_path, data)
    return result


def _add_acquire_args(parser) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True,
                        help="Path to the HF token, 0600, e.g. "
                             "~/.cache/huggingface/token. Never echoed/logged.")
    parser.add_argument("--limit-rate", default="8M",
                        help="curl --limit-rate ceiling per house bandwidth-"
                             "shaping rule (default 8M = 8 MB/s).")


def _acquire_cli(args) -> int:
    result = run(root=args.root.resolve(), token_file=args.token_file.resolve(),
                 limit_rate=args.limit_rate)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# verify (task 489; formerly a separate script)
# --------------------------------------------------------------------------- #
VERIFY_VERSION = "openlegaldata-hf-dump-verification.v1"
VERIFY_PRODUCER = "aw/acquire/openlegaldata_hf.py#verify"
SCHEMA = "aw.openlegaldata-hf-dump-verification-run.v1"

EXPECTED_COLUMNS = {
    "id", "slug", "court", "file_number", "date", "created_date",
    "updated_date", "type", "ecli", "content", "markdown_content",
    "reference_markers",
}
SAMPLE_PER_SHARD = 2
SAMPLE_TARGET = 100
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _parquet_paths(root: Path, stage_receipt: dict) -> list[Path]:
    paths = sorted(
        root / artifact["uri"].removeprefix("artifact://")
        for artifact in stage_receipt["artifacts"]
        if artifact["uri"].startswith("artifact://original/parquet/")
        and artifact["uri"].endswith(".parquet"))
    if len(paths) != SHARD_COUNT:
        raise ValueError(
            f"expected {SHARD_COUNT} parquet shards in the sealed stage, "
            f"found {len(paths)}")
    return paths


def _sample_rows(path: Path, take: int) -> list[dict]:
    pf = pq.ParquetFile(path)
    columns = ["id", "court", "date", "ecli", "content"]
    batch = next(pf.iter_batches(batch_size=take, columns=columns), None)
    if batch is None:
        return []
    return batch.to_pylist()[:take]


# Minimum fraction of the sample that must carry non-empty decision text.
# NOT 100%: a live check against the real dump (task 489, 2026-08-31) found
# genuine source rows with empty `content` sitting alongside populated ones
# from the SAME court in the SAME shard (e.g. Bundesfinanzhof id 115574:
# content="", ecli="", vs. neighbouring BFH ids 115575-115585 with full text)
# -- a real gap in OpenLegalData's own extraction, not an acquisition defect.
# Per verification-doctrine ("the bar matches the change"): a content/corpus
# property gets a net-non-negative bar with a named tripwire, not a per-row
# zero-loss bar. 90% is a conservative floor -- well below anything observed
# live but tight enough to catch a systemically broken shard.
CONTENT_NONEMPTY_FLOOR = 0.90


def _check_sample_row(row: dict) -> dict:
    content = (row.get("content") or "").strip()
    court = row.get("court") or {}
    date = str(row.get("date") or "")
    ecli = str(row.get("ecli") or "").strip()
    return {
        "id": row.get("id"),
        "content_nonempty": bool(content),
        "content_len": len(content),
        "court_name_nonempty": bool(str(court.get("name") or "").strip()),
        # NOT a sanity requirement: `jurisdiction` names the court BRANCH
        # (Arbeitsgerichtsbarkeit, Sozialgerichtsbarkeit, Ordentliche
        # Gerichtsbarkeit, ...) and is legitimately null for federal courts
        # (Bundesfinanzhof, BGH, ...) -- confirmed live, task 489: every
        # sampled Bundesfinanzhof row carries jurisdiction=None regardless of
        # content. Reported as a rate only, never gated.
        "court_jurisdiction_nonempty": bool(str(court.get("jurisdiction") or "").strip()),
        "date_sane": bool(DATE_RE.match(date)),
        "ecli_nonempty": bool(ecli),
    }


def verify(*, root: Path, out_path: Path) -> dict:
    run_path = root / "RUN.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if (run.get("schema") != "aw.openlegaldata-hf-dump-acquisition-run.v1"
            or run.get("status") != "complete"
            or run.get("config") != CONFIG):
        raise ValueError("unexpected or incomplete OpenLegalData acquisition RUN")

    stage_receipt = _validate_stage(run["stages"]["original"], "openlegaldata.original",
                                    artifact_root=root, deep=True)
    parquet_paths = _parquet_paths(root, stage_receipt)

    shard_reports = []
    total_rows = 0
    schema_mismatches = []
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        columns = set(pf.schema_arrow.names)
        num_rows = pf.metadata.num_rows
        total_rows += num_rows
        if columns != EXPECTED_COLUMNS:
            schema_mismatches.append({
                "shard": path.name, "missing": sorted(EXPECTED_COLUMNS - columns),
                "unexpected": sorted(columns - EXPECTED_COLUMNS)})
        shard_reports.append({"shard": path.name, "num_rows": num_rows,
                              "sha256": sha256_file(path)})

    if schema_mismatches:
        raise ValueError(f"OpenLegalData parquet schema drift: {schema_mismatches}")

    sample_checks = []
    for path in parquet_paths:
        sample_checks.extend(
            _check_sample_row(row) for row in _sample_rows(path, SAMPLE_PER_SHARD))
    sample_checks = sample_checks[:SAMPLE_TARGET]

    sample_size = len(sample_checks)
    content_nonempty = sum(1 for c in sample_checks if c["content_nonempty"])
    content_rate = (content_nonempty / sample_size) if sample_size else 0.0
    court_failures = [c for c in sample_checks if not c["court_name_nonempty"]]
    date_failures = [c for c in sample_checks if not c["date_sane"]]
    jurisdiction_nonempty = sum(1 for c in sample_checks if c["court_jurisdiction_nonempty"])
    ecli_nonempty = sum(1 for c in sample_checks if c["ecli_nonempty"])

    # Hard requirements (structural -- every admitted decision names its
    # court and carries a well-formed date; a failure here means a genuinely
    # broken record, not source sparsity).
    if court_failures:
        raise ValueError(
            f"{len(court_failures)}/{sample_size} sampled rows have no court "
            f"name: {court_failures}")
    if date_failures:
        raise ValueError(
            f"{len(date_failures)}/{sample_size} sampled rows have an "
            f"unparseable date: {date_failures}")
    # Tripwire, not a per-row gate (see CONTENT_NONEMPTY_FLOOR comment above):
    # OpenLegalData's own extraction leaves some decisions text-empty.
    if content_rate < CONTENT_NONEMPTY_FLOOR:
        raise ValueError(
            f"decision-text fill rate {content_rate:.1%} in the {sample_size}-row "
            f"sample is below the {CONTENT_NONEMPTY_FLOOR:.0%} floor -- looks like "
            f"a systemic extraction problem, not source sparsity")

    expected_rows = EXPECTED_ROWS_HINT
    result = {
        # No wall-clock field here on purpose: this result is a pure function
        # of the acquisition RUN + the downloaded parquet bytes, so re-running
        # verification is idempotent and the immutable-write guard below is
        # meaningful (matches the acquisition RUN.json,
        # which carries no timestamp for the same reason). The wall-clock is
        # still visible -- printed to stderr by main(), never persisted.
        "schema": SCHEMA, "version": VERIFY_VERSION, "task": 489,
        "producer": VERIFY_PRODUCER,
        "acquisition_run_sha256": sha256_file(run_path),
        "dataset": DATASET, "config": CONFIG,
        "counts": {
            "shards": len(parquet_paths),
            "measured_total_rows": total_rows,
            "expected_total_rows_hint": expected_rows,
            "row_count_drift": total_rows - expected_rows,
        },
        "schema_check": {"expected_columns": sorted(EXPECTED_COLUMNS), "status": "pass"},
        "sample_verification": {
            "method": (f"systematic: first {SAMPLE_PER_SHARD} rows of every "
                      f"shard (every shard represented), trimmed to "
                      f"{SAMPLE_TARGET}"),
            "sample_size": sample_size,
            "court_name_present": sample_size - len(court_failures),
            "date_sane": sample_size - len(date_failures),
            "content_nonempty_count": content_nonempty,
            "content_nonempty_rate": content_rate,
            "content_nonempty_floor": CONTENT_NONEMPTY_FLOOR,
            "jurisdiction_nonempty_count": jurisdiction_nonempty,
            "jurisdiction_nonempty_rate": (jurisdiction_nonempty / sample_size) if sample_size else None,
            "jurisdiction_note": ("not gated -- null for federal courts "
                                  "(Bundesfinanzhof etc.), populated for "
                                  "Land-level courts; observed live, task 489"),
            "ecli_nonempty_count": ecli_nonempty,
            "ecli_nonempty_rate": (ecli_nonempty / sample_size) if sample_size else None,
            "status": "pass",
        },
        "shards": shard_reports,
        "verdict": ("PASS: schema matches, 100-row sample sane, row-count "
                    "drift named above" if total_rows == expected_rows else
                    "PASS_WITH_DRIFT: schema matches, 100-row sample sane, "
                    "measured row count differs from the pack's declared "
                    "expectation -- see counts.row_count_drift"),
    }
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if out_path.exists():
        if sha256_file(out_path) != hashlib.sha256(data).hexdigest():
            raise ValueError(f"immutable verification-result collision: {out_path}")
    else:
        atomic_publish(out_path, data)
    return result


def _add_verify_args(parser) -> None:
    parser.add_argument("--root", type=Path, required=True,
                        help="Acquisition root containing RUN.json.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Where to write VERIFY.json.")


def _verify_cli(args) -> int:
    result = verify(root=args.root.resolve(), out_path=args.out.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"{_now()} {result['verdict']}", file=sys.stderr)
    # Task 529: the verdict was printed and the process exited 0, so a driver
    # checking $? was checking nothing. A gate that cannot fail is a comment.
    return 0 if str(result.get("verdict", "")).upper().startswith("PASS") else 1


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog or "openlegaldata_hf", description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    _add_acquire_args(sub.add_parser("acquire", help="resumably acquire the 54 shards"))
    _add_verify_args(sub.add_parser("verify", help="verify a sealed acquisition RUN"))
    args = parser.parse_args(argv)
    return _acquire_cli(args) if args.verb == "acquire" else _verify_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
