"""`cyt ops mirror-audit`: twin against prod, by CONTENT, for a declared class of ids (task 837, G3).

WHY. Until task 833 nothing compared a twin row with a prod row (task 830: 6,247 rows rewritten in
place on the twin stayed stale on prod, every count equal). 833 fixed the two places that could
catch it at write time (G1 finalize, G2 the prod-leg readback). G3 is the third net: a scheduled,
READ-ONLY audit that asks the same question later, for the rows that were rewritten in place,
whatever path they took (a flip, a restore, a resume, a writer outside the queue).

WHAT IT DOES. It builds a class of ids, reads each id on the twin and on prod BY ID (never a
scroll, a filter or a search), and compares the two rows with `cytir.index.mirror_compare.Comparator`
(the twin row in prod's stored form, every payload key, the dense vector). Each id lands in one
bucket:

    EQUAL            the rows match
    DIFF-declared    they differ only on keys a DECLARATION allows (see below)
    DIFF-undeclared  they differ on a key nobody declared
    missing-twin     prod holds the id, the twin does not
    missing-prod     the twin holds the id, prod does not
    missing-both     neither holds it (a later tombstone; consistent, but not a comparison)

THE CLASSES (`--class KIND [ARG]`, repeatable; the ids are the union):

    inplace-today        the in-place ids of every pack whose artifact (`<pack>-inplace-ids.json`,
                         task 833) was recorded today (UTC)
    inplace-since DATE   the same since DATE (YYYY-MM-DD or an ISO time)
    ids FILE             an explicit list: one id per line, or a JSON list
    declared             every id of every declared set (the known divergences: they must STAY
                         inside their declared keys; a NEW key on them is a finding)

An artifact recorded less than `--settle-min` minutes ago is skipped and counted: the twin leg of a
pack lands minutes before its prod leg, and a read between the two is not a divergence.

THE DECLARATIONS (`data/mirror-audit-declared.v1.json`, schema `aw.mirror-audit-declared.v1`).
A divergence is DECLARED when a data file says so: an id set (by reference to the file that lists
it), a reason, a ruling, the keys allowed to differ and optionally an expiry. Nothing is declared
in code. `global_allowed_keys` holds keys allowed to differ everywhere (empty by default: every
entry needs a reason). `not_compared` documents what the audit does NOT look at.

NOT COMPARED (documented, review 836 S2): the sparse `bm25` vector. Prod stores it pruned under
the aw_e9 BM25 rule; the twin stores it raw; the two differ by design on every row, so a compare
would flag all of them. The payload carries the rule epoch (`rule_epochs`), which IS compared, and
the sparse text itself is a function of the compared payload.

EXIT CODES. 0 only when no undeclared divergence was found AND at least one id was actually
compared on both sides. 3 = at least one undeclared divergence (DIFF-undeclared, missing-twin or
missing-prod). 4 = REFUSED, nothing can be claimed: a replay (`AW_REPLAY=1`: zero prod calls), the
prod read budget, a read error, an empty class, a declared set that cannot be loaded, nothing
compared. A cap (`--max-ids`, `--deadline-min`) that leaves ids unread is NOT silent: they are
counted as `skipped`, printed, and `complete` is false in the receipt.

PROD IS READ ONLY BY ID, paced and counted. `PD.fetch_rows` goes through `prod_guard.check` (the
budget of 400 POST calls per process; no scroll; no query in a replay) and `PG.pace` follows each
prod batch. At 500 ids per call the default cap (190,000 ids) is 380 calls. No write is issued
to any Qdrant, and the only file written is the receipt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Callable, Iterable

from cytir.index import mirror_compare as MC
from cytir.index import point_delete as PD
from cytir.index import prod_guard as PG
from cytir.index import out_of_queue as OOQ

RECEIPT_SCHEMA = "aw.mirror-audit-receipt.v1"
DECLARED_SCHEMA = "aw.mirror-audit-declared.v1"
INPLACE_SCHEMA = "aw.content-pack-inplace-ids.v1"
EXIT_OK, EXIT_DIVERGED, EXIT_REFUSED = 0, 3, 4
APP_ROOT = pathlib.Path(__file__).resolve().parents[2]
DECLARED_DEFAULT = APP_ROOT / "data" / "mirror-audit-declared.v1.json"
BATCH = 500                       # ids per retrieve, both sides
DEFAULT_MAX_IDS = 190_000         # 380 prod calls of 500: under the 400-call budget
DEFAULT_DEADLINE_MIN = 30
DEFAULT_SETTLE_MIN = 180
DEFAULT_PROD_URL = "http://127.0.0.1:%d" % PG.PROD_PORT
EXAMPLES = MC.EXAMPLES            # 20
SOURCE_BASES = ("news_work", "work_root")     # the settings a ledger-pack source may be rooted at
UNDECLARED_BUCKETS = ("DIFF-undeclared", "missing-twin", "missing-prod")
CLASS_KINDS = ("inplace-today", "inplace-since", "ids", "declared")


class Refused(RuntimeError):
    """The audit cannot claim a result (exit 4)."""


# ------------------------------------------------------------------------- the declarations
class DeclaredSet:
    def __init__(self, entry: dict):
        self.id = entry["id"]
        self.reason = entry["reason"]
        self.ruling = entry["ruling"]
        self.allowed_keys = frozenset(entry["allowed_keys"])
        self.allow_unstamped_rule_epochs = bool(entry.get("allow_unstamped_rule_epochs", False))
        self.allow_missing = bool(entry.get("allow_missing", False))
        self.expires = entry.get("expires")
        self.source = entry["source"]
        self.ids: set[str] = set()
        self.id_keys: dict[str, frozenset] = {}      # id -> EXTRA allowed keys of that id (id-keys-json sources)

    def active(self, today: dt.date) -> bool:
        return not self.expires or today <= dt.date.fromisoformat(self.expires)


class Declared:
    def __init__(self, doc: dict, path: pathlib.Path | None):
        self.path = path
        self.sets = [DeclaredSet(e) for e in doc.get("sets", [])]
        self.global_allowed = {e["key"]: e["reason"] for e in doc.get("global_allowed_keys", [])}
        self.not_compared = doc.get("not_compared", [])

    @classmethod
    def empty(cls) -> "Declared":
        return cls({}, None)


def validate_declared(doc: Any) -> list[str]:
    """The schema problems of a declared-divergence document ([] = valid)."""
    bad: list[str] = []
    if not isinstance(doc, dict) or doc.get("schema") != DECLARED_SCHEMA:
        return [f"schema must be {DECLARED_SCHEMA!r}"]
    for i, e in enumerate(doc.get("global_allowed_keys", [])):
        if not (isinstance(e, dict) and e.get("key") and str(e.get("reason", "")).strip()):
            bad.append(f"global_allowed_keys[{i}] needs a key and a reason")
    for i, e in enumerate(doc.get("not_compared", [])):
        if not (isinstance(e, dict) and e.get("what") and str(e.get("why", "")).strip()):
            bad.append(f"not_compared[{i}] needs what and why")
    seen: set[str] = set()
    for i, e in enumerate(doc.get("sets", [])):
        where = f"sets[{i}]"
        if not isinstance(e, dict):
            bad.append(f"{where} is not an object")
            continue
        for k in ("id", "reason", "ruling"):
            if not str(e.get(k, "")).strip():
                bad.append(f"{where} needs a non-empty {k}")
        if e.get("id") in seen:
            bad.append(f"{where}: duplicate id {e.get('id')!r}")
        seen.add(e.get("id"))
        keys = e.get("allowed_keys")
        if not (isinstance(keys, list) and keys and all(isinstance(k, str) and k for k in keys)):
            bad.append(f"{where}: allowed_keys must be a non-empty list of key names")
        src = e.get("source")
        if not isinstance(src, dict) or src.get("kind") not in _SOURCE_KINDS:
            bad.append(f"{where}: source.kind must be one of {sorted(_SOURCE_KINDS)}")
        else:
            bad += [f"{where}: {p}" for p in _SOURCE_KINDS[src["kind"]]["check"](src)]
        if e.get("expires"):
            try:
                dt.date.fromisoformat(e["expires"])
            except (TypeError, ValueError):
                bad.append(f"{where}: expires must be YYYY-MM-DD")
    return bad


def _check_results_source(src: dict) -> list[str]:
    ok = (isinstance(src.get("path"), str) and src["path"] and isinstance(src.get("classes"), list)
          and src["classes"] and isinstance(src.get("expected_ids"), int))
    return [] if ok else ["results-jsonl-gz needs path, classes[], expected_ids"]


def _check_ledger_source(src: dict) -> list[str]:
    ok = (src.get("base") in SOURCE_BASES and src.get("ledger") and src.get("pack_dir")
          and isinstance(src.get("expected_ids"), int))
    return [] if ok else [f"apply-ledger-pack needs base in {SOURCE_BASES}, ledger, pack_dir, expected_ids"]


def _read_results_ids(src: dict, *, settings) -> list[str]:
    path = _resolve(src["path"], APP_ROOT)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    except (OSError, ValueError) as exc:
        raise Refused(f"declared source {src['path']!r} cannot be read: {exc}") from exc
    want = set(src["classes"])
    return [r["id"] for r in rows if r.get("class") in want]


def _check_idkeys_source(src: dict) -> list[str]:
    ok = (isinstance(src.get("path"), str) and src["path"] and isinstance(src.get("expected_ids"), int)
          and isinstance(src.get("sha256"), str) and len(src["sha256"]) == 64)
    return [] if ok else ["id-keys-json needs path, sha256, expected_ids"]


def _read_idkeys(src: dict, *, settings) -> dict[str, list[str]]:
    """{id: [extra allowed keys]} of a per-id declaration file. The file's sha256 is pinned in the
    declaration: a changed file is a refusal, never a silent re-declaration."""
    path = _resolve(src["path"], APP_ROOT)
    try:
        raw = path.read_bytes()
        doc = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise Refused(f"declared source {src['path']!r} cannot be read: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != src["sha256"]:
        raise Refused(f"declared source {src['path']!r}: sha256 is not the declaration's")
    m = doc.get("ids") if isinstance(doc, dict) else None
    if not (isinstance(m, dict) and all(isinstance(v, list) and v and all(isinstance(k, str) for k in v)
                                        for v in m.values())):
        raise Refused(f"declared source {src['path']!r}: `ids` must map each id to a non-empty key list")
    return m


def _read_ledger_ids(src: dict, *, settings) -> list[str]:
    """The ids of a pack that a writer outside the queue applied. The apply ledger names the pack
    artifact by its sha256 and its row count; the ids are the artifact's own rows."""
    base = pathlib.Path(getattr(settings, src["base"]))
    ledger = base / src["ledger"]
    try:
        entries = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    except (OSError, ValueError) as exc:
        raise Refused(f"apply ledger {ledger} cannot be read: {exc}") from exc
    named = [e for e in entries if e.get("artifact_sha256")]
    if not named:
        raise Refused(f"apply ledger {ledger} names no artifact_sha256")
    sha, n = named[0]["artifact_sha256"], named[0].get("artifact_ids")
    points = base / src["pack_dir"] / f"news-points.{sha}.jsonl.gz"
    try:   # one reader of an out-of-queue pack artifact, shared with the aw_e10 catch-up (task 847)
        return OOQ.pack_point_ids(str(points), sha, n)
    except OOQ.OutOfQueueError as exc:
        raise Refused(str(exc)) from exc


_SOURCE_KINDS = {
    "results-jsonl-gz": {"check": _check_results_source, "read": _read_results_ids},
    "apply-ledger-pack": {"check": _check_ledger_source, "read": _read_ledger_ids},
    "id-keys-json": {"check": _check_idkeys_source, "read": None},     # read by fill_declared_ids
}


def _resolve(rel: str, root: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(rel)
    return p if p.is_absolute() else root / p


def load_declared(path: pathlib.Path | None, *, required: bool) -> Declared:
    """The declarations. A missing DEFAULT file is an empty list (a stranger's checkout does not
    ship it); a missing file that was asked for, or any malformed one, is a refusal."""
    if path is None or not path.exists():
        if required:
            raise Refused(f"declared-divergence file {path} does not exist")
        return Declared.empty()
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise Refused(f"declared-divergence file {path} is not JSON: {exc}") from exc
    problems = validate_declared(doc)
    if problems:
        raise Refused(f"declared-divergence file {path} is invalid: " + "; ".join(problems[:5]))
    return Declared(doc, path)


def fill_declared_ids(declared: Declared, settings) -> None:
    """Read the id list of every declared set from the file it references; the count must be the
    one the declaration states (a source that changed is a refusal, never a silent re-declaration)."""
    for s in declared.sets:
        if s.source["kind"] == "id-keys-json":
            s.id_keys = {i: frozenset(k) for i, k in _read_idkeys(s.source, settings=settings).items()}
            ids = list(s.id_keys)
        else:
            ids = _SOURCE_KINDS[s.source["kind"]]["read"](s.source, settings=settings)
        s.ids = set(ids)
        if len(s.ids) != s.source["expected_ids"]:
            raise Refused(f"declared set {s.id!r}: the source holds {len(s.ids)} distinct ids, "
                          f"the declaration says {s.source['expected_ids']}")


# ------------------------------------------------------------------------- the classes
def _slice_of(pid: str, n: int) -> int:
    """Stable slice of an id (0..n-1): the same id lands in the same slice on every box and day."""
    return int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16) % n


def _parse_when(text: str) -> dt.datetime:
    try:
        d = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Refused(f"not a date or an ISO time: {text!r}") from exc
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def _read_inplace(artifact_dir: pathlib.Path, since: dt.datetime, settled_before: dt.datetime
                  ) -> tuple[list[tuple[str, str]], dict]:
    """(id, pack) newest artifact first, and the counts.

    An artifact is used only when its PROD leg is recorded and settled: the twin leg lands first, and a
    pack whose prod leg is pending (or aborted) is legitimately divergent, which is the drain's own state
    and not a finding (review 838 S1). Those are counted as `prod_leg_pending`. The date and the settle
    time are those of the prod record (the leg that finishes last)."""
    found: list[tuple[str, str, str]] = []
    info = {"artifact_dir": str(artifact_dir), "artifacts": 0, "deferred_unsettled": 0,
            "prod_leg_pending": 0, "unreadable": 0}
    for f in sorted(artifact_dir.glob("*-inplace-ids.json")):
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            info["unreadable"] += 1
            continue
        if rec.get("schema") != INPLACE_SCHEMA:
            info["unreadable"] += 1
            continue
        prod = rec.get("prod")
        if not isinstance(prod, dict) or not prod.get("at"):
            info["prod_leg_pending"] += 1
            continue
        at = _parse_when(prod["at"])
        if at < since:
            continue
        if at > settled_before:
            info["deferred_unsettled"] += 1
            continue
        info["artifacts"] += 1
        ids = set(prod.get("ids", [])) | set((rec.get("twin") or {}).get("ids", []))
        found.extend((prod["at"], i, rec.get("pack_id", f.name)) for i in sorted(ids))
    found.sort(key=lambda t: t[0], reverse=True)
    return [(i, p) for _, i, p in found], info


def _read_ids_file(path: pathlib.Path) -> list[str]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise Refused(f"id file {path} cannot be read: {exc}") from exc
    s = text.lstrip()
    if s.startswith("["):
        try:
            return [str(x) for x in json.loads(s)]
        except ValueError as exc:
            raise Refused(f"id file {path} is not a JSON list: {exc}") from exc
    return [ln.split("#")[0].strip() for ln in text.splitlines() if ln.split("#")[0].strip()]


def build_class(specs: list[list[str]], *, declared: Declared, artifact_dir: pathlib.Path | None,
                now: dt.datetime, settle_min: int, declared_slice: int = 1
                ) -> tuple[list[str], dict[str, str], dict]:
    """(ordered distinct ids, id -> origin label, class info). Order = the order of the specs,
    newest artifact first within a class, so a cap drops the oldest and the declared ids last."""
    if not specs:
        raise Refused("no class given: use --class inplace-today | inplace-since DATE | ids FILE | declared")
    ids: list[str] = []
    origin: dict[str, str] = {}
    info: dict[str, Any] = {"classes": [" ".join(s) for s in specs]}
    settled_before = now - dt.timedelta(minutes=settle_min)

    def add(pairs: Iterable[tuple[str, str]]):
        for i, label in pairs:
            if i not in origin:
                origin[i] = label
                ids.append(i)

    for spec in specs:
        kind, args = spec[0], spec[1:]
        if kind not in CLASS_KINDS:
            raise Refused(f"unknown class {kind!r}: one of {', '.join(CLASS_KINDS)}")
        if kind in ("inplace-today", "inplace-since"):
            if artifact_dir is None or not artifact_dir.is_dir():
                raise Refused(f"in-place artifact directory {artifact_dir} does not exist")
            if kind == "inplace-today":
                since = now.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                if len(args) != 1:
                    raise Refused("inplace-since needs a date")
                since = _parse_when(args[0])
            pairs, i = _read_inplace(artifact_dir, since, settled_before)
            info.setdefault("inplace", []).append(i)
            add((i_, f"inplace:{p}") for i_, p in pairs)
        elif kind == "ids":
            if len(args) != 1:
                raise Refused("ids needs a file")
            add((i, "ids-file") for i in _read_ids_file(pathlib.Path(args[0])))
        else:
            if declared.path is None:
                raise Refused("class 'declared' needs a declared-divergence file")
            index = now.date().toordinal() % declared_slice
            total = kept = 0
            for ds in declared.sets:
                mine = [i for i in sorted(ds.ids) if _slice_of(i, declared_slice) == index]
                total += len(ds.ids)
                kept += len(mine)
                add((i, f"declared:{ds.id}") for i in mine)
            info["declared_slice"] = {"n": declared_slice, "index": index, "ids_in_slice": kept,
                                      "ids_declared": total}
    return ids, origin, info


# ------------------------------------------------------------------------- the compare
class Tally:
    def __init__(self):
        self.counts = {"EQUAL": 0, "DIFF-declared": 0, "DIFF-undeclared": 0,
                       "missing-twin": 0, "missing-prod": 0, "missing-both": 0}
        self.by_key: dict[str, int] = {}
        self.examples: list[dict] = []
        self.by_set: dict[str, dict[str, int]] = {}
        self.checked = 0

    @property
    def undeclared(self) -> int:
        c = self.counts
        return c["DIFF-undeclared"] + c["missing-twin"] + c["missing-prod"]

    @property
    def compared(self) -> int:
        return self.checked - self.counts["missing-both"]

    def add(self, bucket: str, pid: str, origin: str, keys: list[str], dset: DeclaredSet | None,
            outside: list[str]):
        self.checked += 1
        self.counts[bucket] += 1
        if dset is not None:
            row = self.by_set.setdefault(dset.id, {})
            row[bucket] = row.get(bucket, 0) + 1
        if bucket in ("DIFF-undeclared", "missing-twin", "missing-prod"):
            for k in (outside or keys):
                self.by_key[k] = self.by_key.get(k, 0) + 1
            if len(self.examples) < EXAMPLES:
                self.examples.append({"id": pid, "bucket": bucket, "keys": keys,
                                      "undeclared_keys": outside, "origin": origin})


def classify(pid: str, tw: dict | None, pr: dict | None, *, strict: MC.Comparator,
             lenient: MC.Comparator, dset: DeclaredSet | None, global_allowed: set[str],
             user_lenient: bool) -> tuple[str, list[str], list[str]]:
    """(bucket, differing keys, the keys no declaration allows) for one id."""
    if tw is None and pr is None:
        return "missing-both", [], []
    if tw is None or pr is None:
        bucket = "missing-twin" if tw is None else "missing-prod"
        key = "<absent-on-twin>" if tw is None else "<absent-on-prod>"
        if dset is not None and dset.allow_missing:
            return "DIFF-declared", [key], []
        return bucket, [key], [key]
    use_lenient = user_lenient or (dset is not None and dset.allow_unstamped_rule_epochs)
    keys = (lenient if use_lenient else strict).diff(tw, pr)
    if not keys:
        return "EQUAL", [], []
    allowed = set(global_allowed) | (set(dset.allowed_keys) if dset is not None else set())
    if dset is not None:
        allowed |= dset.id_keys.get(pid, frozenset())
    outside = [k for k in keys if k not in allowed]
    return ("DIFF-declared" if not outside else "DIFF-undeclared"), keys, outside


def audit(ids: list[str], origin: dict[str, str], *, twin_url: str, twin_coll: str, prod_url: str,
          prod_coll: str, declared: Declared, today: dt.date, allow_unstamped: bool,
          max_ids: int, deadline_s: float, fetch: Callable = PD.fetch_rows,
          pace: Callable = PG.pace, clock: Callable[[], float] = time.monotonic,
          log: Callable = print, progress: dict | None = None, recheck_s: float = 0,
          sleep: Callable = time.sleep) -> dict:
    """Read the ids in batches of 500 on both sides and classify them. Returns the findings and the
    counts of what was read; raises `Refused` on a read error, a budget refusal or a replay.

    A ROW THAT LOOKS UNDECLARED is a suspect, not yet a finding (review 838 S1): after the pass the
    suspects are read once more `recheck_s` seconds later and only those that still differ count. This
    covers a drain that was mid-write on a row, or a manual drain that started during the run. The
    recheck is skipped (and said so) when it would not fit the prod budget."""
    strict = MC.Comparator(prod_coll)
    lenient = MC.Comparator(prod_coll, allow_unstamped_rule_epochs=True, projector=strict.projector)
    active = [s for s in declared.sets if s.active(today)]
    expired = [s.id for s in declared.sets if not s.active(today)]
    owner: dict[str, DeclaredSet] = {}
    for s in active:
        for i in s.ids:
            owner.setdefault(i, s)             # an id in two sets belongs to the FIRST (tested)
    glob = set(declared.global_allowed)
    todo, skipped_cap = ids[:max_ids], max(0, len(ids) - max_ids)
    tally = Tally()
    prod_calls = twin_calls = 0
    timing = {"sum": 0.0, "max": 0.0}
    progress = progress if progress is not None else {}
    start, skipped_deadline = clock(), 0
    suspects: dict[str, tuple] = {}            # id -> its first reading

    def read(chunk: list[str], where: str) -> tuple[dict, dict]:
        nonlocal prod_calls, twin_calls
        try:
            twin_rows = fetch(twin_url, twin_coll, chunk, batch=BATCH)
            twin_calls += 1
            prod_calls += 1                       # counted before the call: a refused call is still a call
            progress.update(prod_calls=prod_calls, twin_calls=twin_calls)
            t0 = time.perf_counter()
            prod_rows = fetch(prod_url, prod_coll, chunk, batch=BATCH)
            dt_s = time.perf_counter() - t0
            timing["sum"] += dt_s
            timing["max"] = max(timing["max"], dt_s)
            pace(prod_url, prod_coll)
        except PG.ProdReadRefused as exc:
            raise Refused(f"prod read refused ({prod_calls} prod call(s) attempted): {exc}") from exc
        except Exception as exc:                  # a read error is never a pass
            raise Refused(f"read error at ids {where} "
                          f"(prod calls so far {prod_calls}): {type(exc).__name__}: {exc}") from exc
        return twin_rows, prod_rows

    def judge(pid, twin_rows, prod_rows):
        dset = owner.get(pid)
        return dset, classify(pid, twin_rows.get(pid), prod_rows.get(pid), strict=strict,
                              lenient=lenient, dset=dset, global_allowed=glob, user_lenient=allow_unstamped)

    for lo in range(0, len(todo), BATCH):
        if clock() - start > deadline_s:
            skipped_deadline = len(todo) - lo
            log(f"[mirror-audit] DEADLINE: {skipped_deadline} ids not read")
            break
        chunk = todo[lo:lo + BATCH]
        twin_rows, prod_rows = read(chunk, f"{lo}..{lo + len(chunk)}")
        for pid in chunk:
            dset, (bucket, keys, outside) = judge(pid, twin_rows, prod_rows)
            if bucket in UNDECLARED_BUCKETS and recheck_s > 0:
                suspects[pid] = (bucket, keys, outside, dset)
            else:
                tally.add(bucket, pid, origin.get(pid, "?"), keys, dset, outside)
    recheck = {"suspects": len(suspects), "cleared": 0, "after_s": recheck_s if suspects else 0,
               "skipped": False}
    if suspects:
        names = list(suspects)
        need = -(-len(names) // BATCH)
        limit = PG._budget()
        if limit is not None and prod_calls + need > limit:
            recheck["skipped"] = True             # the first reading stands, and the receipt says so
            log(f"[mirror-audit] recheck of {len(names)} suspects skipped: it would exceed the prod budget")
            for pid, (bucket, keys, outside, dset) in suspects.items():
                tally.add(bucket, pid, origin.get(pid, "?"), keys, dset, outside)
        else:
            log(f"[mirror-audit] {len(names)} suspect(s): reading them again in {recheck_s:g} s")
            sleep(recheck_s)
            for lo in range(0, len(names), BATCH):
                chunk = names[lo:lo + BATCH]
                twin_rows, prod_rows = read(chunk, f"recheck {lo}..{lo + len(chunk)}")
                for pid in chunk:
                    dset, (bucket, keys, outside) = judge(pid, twin_rows, prod_rows)
                    if bucket not in UNDECLARED_BUCKETS:
                        recheck["cleared"] += 1
                    tally.add(bucket, pid, origin.get(pid, "?"), keys, dset, outside)
    return {"tally": tally, "prod_calls": prod_calls, "twin_calls": twin_calls,
            "skipped_cap": skipped_cap, "skipped_deadline": skipped_deadline,
            "expired_declarations": expired, "recheck": recheck,
            "prod_seconds_sum": round(timing["sum"], 3), "prod_seconds_max": round(timing["max"], 3)}


# ------------------------------------------------------------------------- the receipt
def _verdict(code: int) -> str:
    return {EXIT_OK: "PASS", EXIT_DIVERGED: "DIVERGED"}.get(code, "REFUSED")


def build_receipt(*, code: int, started: dt.datetime, finished: dt.datetime, args: dict,
                  class_info: dict | None, requested: int, result: dict | None,
                  refusal: str | None, declared: Declared, progress: dict | None = None) -> dict:
    t: Tally | None = result["tally"] if result else None
    skipped = (result["skipped_cap"] + result["skipped_deadline"]) if result else requested
    return {
        "schema": RECEIPT_SCHEMA, "exit_code": code, "verdict": _verdict(code),
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_at": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_epoch": int(finished.timestamp()),
        "refusal": refusal, "args": args, "class": class_info,
        "ids_in_class": requested, "checked": t.checked if t else 0,
        "compared": t.compared if t else 0, "counts": t.counts if t else {},
        "undeclared": t.undeclared if t else None, "by_key": dict(sorted(t.by_key.items())) if t else {},
        "examples": t.examples if t else [], "declared_sets": t.by_set if t else {},
        "skipped": skipped, "complete": bool(result) and skipped == 0,
        "prod_calls": result["prod_calls"] if result else (progress or {}).get("prod_calls", 0),
        "twin_calls": result["twin_calls"] if result else (progress or {}).get("twin_calls", 0),
        "expired_declarations": result["expired_declarations"] if result else [],
        "recheck": result["recheck"] if result else None,
        "prod_seconds_sum": result["prod_seconds_sum"] if result else 0,
        "prod_seconds_max": result["prod_seconds_max"] if result else 0,
        "declared_file": str(declared.path) if declared.path else None,
        "not_compared": declared.not_compared,
    }


def write_receipt(receipt: dict, directory: pathlib.Path, stamp: str) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stamp}.json"
    tmp = directory / f".{stamp}.json.tmp"
    tmp.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def summary_lines(r: dict) -> list[str]:
    c = r["counts"]
    out = [f"[mirror-audit] {r['verdict']} (exit {r['exit_code']})"
           + (f": {r['refusal']}" if r["refusal"] else "")]
    if c:
        out.append("[mirror-audit] " + "  ".join(f"{k}={c[k]}" for k in c)
                   + f"  | compared={r['compared']} of {r['ids_in_class']} in class, "
                     f"prod_calls={r['prod_calls']}")
    if r["skipped"] and r["checked"]:
        out.append(f"[mirror-audit] SKIPPED {r['skipped']} ids (cap or deadline): the result covers "
                   f"only the ids read; complete=false")
    rc = r.get("recheck")
    if rc and rc["suspects"]:
        out.append(f"[mirror-audit] RECHECK after {rc['after_s']:g} s: {rc['suspects']} suspect(s), "
                   f"{rc['cleared']} cleared" + (" (SKIPPED: prod budget)" if rc["skipped"] else ""))
    if r["expired_declarations"]:
        out.append(f"[mirror-audit] EXPIRED declarations (treated as undeclared): "
                   f"{', '.join(r['expired_declarations'])}")
    for ex in r["examples"]:
        out.append(f"[mirror-audit] UNDECLARED {ex['bucket']} {ex['id']} {ex['keys']} ({ex['origin']})")
    return out


# ------------------------------------------------------------------------- the verb
def _parser(prog: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--class", dest="classes", action="append", nargs="+", metavar="KIND [ARG]",
                    help="inplace-today | inplace-since DATE | ids FILE | declared (repeatable)")
    ap.add_argument("--declared", help="declared-divergence file (default data/mirror-audit-declared.v1.json)")
    ap.add_argument("--twin-url", help="default: the twin setting")
    ap.add_argument("--twin-collection", help="default: the twin setting")
    ap.add_argument("--prod-url", help=f"default: $CYTIR_PROD_URL or {DEFAULT_PROD_URL}")
    ap.add_argument("--prod-alias", default="aw_prod")
    ap.add_argument("--prod-collection", help="default: what the alias names on prod")
    ap.add_argument("--artifact-dir", help="default: <queue_dir>/_artifacts")
    ap.add_argument("--receipt-dir", help="default: <work_root>/mirror-audit")
    ap.add_argument("--max-ids", type=int, default=DEFAULT_MAX_IDS)
    ap.add_argument("--deadline-min", type=float, default=DEFAULT_DEADLINE_MIN)
    ap.add_argument("--settle-min", type=int, default=DEFAULT_SETTLE_MIN,
                    help="skip in-place artifacts recorded less than this many minutes ago")
    ap.add_argument("--declared-slice", type=int, default=1, metavar="N",
                    help="read the `declared` class in N rotating slices (stable hash of the id; the slice "
                         "of the day is date.toordinal() %% N): each declared id is read once every N nights. "
                         "Default 1 = all of it")
    ap.add_argument("--recheck-s", type=float, default=60,
                    help="read every id that looks undeclared once more after this many seconds and keep "
                         "only those that still differ (0 = off)")
    ap.add_argument("--allow-unstamped-rule-epochs", action="store_true",
                    help="a prod row carried by a build or a flip predates the BM25 epoch stamp: "
                         "accept the unstamped form (declared sets say so themselves)")
    ap.add_argument("--plan", action="store_true", help="resolve the class and print the plan; read nothing")
    return ap


def _alias_collection(prod_url: str, alias: str) -> str:
    result = PD._api(prod_url, "/aliases")
    for e in (result or {}).get("aliases", []):
        if e.get("alias_name") == alias:
            return e["collection_name"]
    raise Refused(f"alias {alias!r} not found on {prod_url} (GET /aliases)")


def run(argv: list[str], *, prog: str = "cyt ops mirror-audit", now: Callable[[], dt.datetime] | None = None,
        settings=None, fetch: Callable = PD.fetch_rows, pace: Callable = PG.pace,
        clock: Callable[[], float] = time.monotonic, out=print, sleep: Callable = time.sleep) -> int:
    args = _parser(prog).parse_args(argv)
    clock_now = now or (lambda: dt.datetime.now(dt.timezone.utc))
    started = clock_now()
    declared = Declared.empty()
    class_info: dict | None = None
    requested = 0
    result = None
    progress: dict = {}
    code, refusal = EXIT_OK, None
    arg_doc = {k: v for k, v in vars(args).items() if v is not None}
    receipt_dir = None
    try:
        if settings is None:
            from cytir.core import paths
            settings = paths.load_settings()
        receipt_dir = pathlib.Path(args.receipt_dir) if args.receipt_dir else \
            pathlib.Path(settings.work_root) / "mirror-audit"
        if PG.replay_active():
            raise Refused(f"{PG.REPLAY_ENV}=1: a replay sends no query to prod, so there is nothing "
                          f"to compare and no result may be claimed")
        from cytir.core import twin as TW
        twin_url = (args.twin_url or TW.url()).rstrip("/")
        twin_coll = args.twin_collection or TW.collection()
        prod_url = (args.prod_url or os.environ.get("AW_PROD_URL") or DEFAULT_PROD_URL).rstrip("/")
        if PG.is_prod(twin_url, twin_coll):
            raise Refused(f"the twin endpoint {twin_url} ({twin_coll}) is a prod target")
        needs_declared = any(c[0] == "declared" for c in args.classes or [])
        declared = load_declared(pathlib.Path(args.declared) if args.declared else DECLARED_DEFAULT,
                                 required=bool(args.declared) or needs_declared)
        fill_declared_ids(declared, settings)
        art_dir = pathlib.Path(args.artifact_dir) if args.artifact_dir else \
            pathlib.Path(settings.queue_dir) / "_artifacts"
        ids, origin, class_info = build_class(args.classes or [], declared=declared, artifact_dir=art_dir,
                                              now=started, settle_min=args.settle_min,
                                              declared_slice=max(1, args.declared_slice))
        requested = len(ids)
        if not ids:
            raise Refused(f"the class is empty ({class_info}): nothing to compare")
        calls = -(-min(len(ids), args.max_ids) // BATCH)
        budget = PG._budget()
        if budget is not None and calls > budget:
            raise Refused(f"the class needs {calls} prod call(s) of {BATCH} ({min(len(ids), args.max_ids)} ids), "
                          f"over the prod read budget of {budget}: lower --max-ids; no call was made")
        if args.plan:
            out(f"[mirror-audit] PLAN: {len(ids)} ids ({class_info['classes']}), "
                f"{calls} prod call(s) of {BATCH}, skipped by the cap {max(0, len(ids) - args.max_ids)}; "
                f"declared sets: " + ", ".join(f"{s.id}={len(s.ids)}" for s in declared.sets))
            return EXIT_OK
        prod_coll = args.prod_collection or _alias_collection(prod_url, args.prod_alias)
        result = audit(ids, origin, twin_url=twin_url, twin_coll=twin_coll, prod_url=prod_url,
                       prod_coll=prod_coll, declared=declared, today=started.date(),
                       allow_unstamped=args.allow_unstamped_rule_epochs, max_ids=args.max_ids,
                       deadline_s=args.deadline_min * 60, fetch=fetch, pace=pace, clock=clock, log=out, progress=progress,
                       recheck_s=args.recheck_s, sleep=sleep)
        t = result["tally"]
        if t.compared < 1:
            raise Refused(f"nothing was compared on both sides ({t.checked} ids read, "
                          f"{t.counts['missing-both']} absent from both)")
        code = EXIT_DIVERGED if t.undeclared else EXIT_OK
    except Refused as exc:
        code, refusal = EXIT_REFUSED, str(exc)
    except Exception as exc:                      # nothing may end the run without a receipt
        code, refusal = EXIT_REFUSED, f"{type(exc).__name__}: {exc}"
    if args.plan and code == EXIT_REFUSED:
        out(f"[mirror-audit] REFUSED: {refusal}")
        return code
    finished = clock_now()
    receipt = build_receipt(code=code, started=started, finished=finished, args=arg_doc,
                            class_info=class_info, requested=requested, result=result,
                            refusal=refusal, declared=declared, progress=progress)
    for line in summary_lines(receipt):
        out(line)
    try:
        if receipt_dir is None:
            raise OSError("no receipt directory (settings unreadable)")
        path = write_receipt(receipt, receipt_dir, started.strftime("%Y%m%dT%H%M%SZ"))
        out(f"[mirror-audit] receipt {path}")
    except OSError as exc:
        out(f"[mirror-audit] REFUSED: the receipt cannot be written: {exc}")
        code = EXIT_REFUSED
    return code


def main(argv: list[str] | None = None, prog: str = "cyt ops mirror-audit") -> int:
    return run(list(sys.argv[1:] if argv is None else argv), prog=prog)


if __name__ == "__main__":
    raise SystemExit(main())
