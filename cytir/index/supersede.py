#!/usr/bin/env python3
# Moved from `upsert_supersede.py` (task 677, task 643 S5); the old path is an alias shim.
"""upsert_supersede.py — supersede-on-upsert for every delta/upsert path.

THE BUG THIS KILLS (task 348; found in task 342 §4): I1 identity (identity.md
§2) makes a Tier-B point id content-addressed — `uuid5(NS, canon_url + "#" +
digest)` — so a re-crawled page whose content CHANGED mints NEW point ids, and
a plain qdrant upsert inserts them WITHOUT touching the old points of the same
page. Result: score-tied duplicates (both twins servable, retrieve-order decides
which the user sees), and stale chunks of shrunk pages survive forever.
Empirically verified in task 348 repro348.py: a 3-chunk page re-crawled
as {1 changed, 1 unchanged, 1 removed, 1 new} ends at FIVE points, not three.

THE RULE (one home, imported by every applier — engineering doctrine 1):
after a delta's upserts land, every canonical page GROUP the delta touched is
reconciled to exactly the incoming record set: existing points whose group
matches an incoming group and whose id is NOT in the incoming keep-set are
deleted, batched, with wait=True and a retrieve-back verification.

GROUPING:
  group_key = aw_common.canonical_url(stored url)  [+ "|card" for entity_card]
  * fedlex-class law records embed the per-article anchor in the url fragment,
    and canonical_url() KEEPS fragments — each article is its own group, so a
    law delta touching art_336 never disturbs sibling articles.
  * entity cards group separately from content chunks of the same url (a card
    refresh must never delete the page's content records, and vice versa).
  * everything else groups page-level: ALL chunks of a page form one group, so
    a shorter new version deletes ALL old chunks, not just colliding ordinals.

THE COMPLETENESS CONTRACT (what makes group-reconcile sound): an applier input
must carry, for every canonical page it touches, the COMPLETE current record
set of that page. Full staged packs and full collection dumps satisfy this by
construction; a dump_netnew changed-set dump satisfies it only since format 2
(unchanged ids emitted as slim `_delta:"unchanged"` marker lines + manifest
`supersede_complete: true`). Appliers MUST refuse to supersede on inputs that
cannot prove completeness — deleting "absent" siblings that were merely
unchanged-and-omitted would be data loss, not hygiene.

GUARANTEES (also the reasons this is structurally safe):
  * a group only exists because >=1 incoming record was observed for it, so a
    supersede can REPLACE a page's records but never EMPTY a page;
  * ids observed anywhere in the input are globally protected from deletion,
    whatever group they bucket into;
  * an UNCHANGED record (same digest -> same id) is observed as incoming, so
    the freshness re-upsert path stays a clean in-place overwrite (342's
    working case, preserved);
  * deletions are wait=True and verified gone by retrieve before success is
    claimed (assert_durable stance: "the server said ok" is not evidence);
  * gold-AWARE, not gold-blocking: superseding a gold page's old version by
    its new version is correct — but it is checked against the eval gold sets
    (hygiene/gold_guard semantics) and logged loudly. If the gold
    data is not present on this box the check reports itself OFF instead of
    silently finding nothing (an error is not an empty).

LOOKUP: existing group members are found via a keyword payload index on `url`,
matched over the page's url VARIANTS (stored urls keep scheme/www as captured;
the canonical form folds them — so we enumerate {as-given, tracking-stripped,
https/http x bare/www of the canonical}). A stored variant outside this set
(exotic tracking params from another era) is not found — such residuals are
cleanup-sweep material, not supersede material. The `url` index is REQUIRED:
without it this would be a full-scan filter per page (the
prod-serving-box-fragile-scrolls incident class). `ensure_url_index(create=
False)` fails loudly; auto-create is the OFFLINE appliers' choice only.

ROLLBACK-COMPLETENESS GUARD (task 574; found in task 572 §4): a pre-apply
export is not a real rollback just because it exists. Task 572 exported a
copy keyed by a MATERIALIZE-time ledger (the roster's own scan of
`old_point_ids`, `task 565 split_and_export_565.py`) and only
discovered, after the delete had already run, that the live group-scroll
this module's `run()` actually deletes by is a DIFFERENT, wider selector: it
catches every existing point of a touched page's url group, including ones
staged by a lane the roster never scanned. 62,023 exported vs. 62,338
actually deleted — 315 points gone with no copy, unrecoverable. The fix is
not a bigger export; it is refusing to let the two selectors diverge
silently: `run(export_path=..., expect_ids=...)` calls `_scan_stale()`
exactly ONCE — that result IS both "the set about to be exported" and "the
set about to be deleted" — writes the export (dense + sparse vectors + full
payload, `encode_point_vectors`) from that SAME set before touching a
single point, and, when a caller ALSO supplies `expect_ids` (e.g. a
roster-keyed export computed earlier, upstream of staging), refuses BEFORE
any delete if it does not match exactly, naming both counts and the
symmetric difference. `restore_from_export()` is the named inverse: it
verifies `points_count` arithmetic AND a sample readback (payload + vector,
not just presence) — a count match proves arithmetic, never content.

CLI (after-the-fact reconcile of an already-applied dump — e.g. repairing a
wave that was applied before this fix existed):

  python3 upsert_supersede.py --in wave.jsonl.gz \
      --qdrant-url http://localhost:6390 --collection aw_vN \
      [--assume-complete] [--create-url-index] [--dry-run]
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import random
import socket
import subprocess
import time
import urllib.parse
from collections import defaultdict

from cytir.core import common as AC  # noqa: E402  — THE canonicalizer (identity.md §1.2)
from cytir.core.paths import SettingRefused, operator_settings  # noqa: E402
from cytir.core.qdrant import qdrant_connect  # noqa: E402
from cytir.core import twin as TWIN  # noqa: E402
from cytir.index import point_delete as PD  # noqa: E402  — the one delete-and-verify home (task 641)
from cytir.index import prod_guard as PG  # noqa: E402  — no prod scroll without a reason (2026-10-07)

try:
    from qdrant_client import QdrantClient, models as qm
except ImportError:  # pure-helper use (unit tests of key/variant logic)
    QdrantClient = qm = None

CARD_SUFFIX = "|card"
URL_INDEX_FIELD = "url"


# --------------------------------------------------------------------------
# pure helpers (unit-testable without qdrant)
# --------------------------------------------------------------------------
def is_card(source_type: str | None = None, entity_kind=None, qid=None) -> bool:
    """True for an entity card. The TYPE is `source_type == "entity_card"`,
    but a wikidata entity row can be typed `reference` and still carry
    `entity_kind` / `qid` (4 twin keys, task 698 third review R2:
    `srf.ch/sendungen/archiv/`, `srf.ch/radio-srf-virus/`, `rts.ch/archives/`,
    `rsi.ch`). Those are cards too: a page re-capture must never group with
    them, so never tombstone them."""
    if isinstance(source_type, str) and source_type.strip().lower() == "entity_card":
        return True
    return bool(entity_kind) or bool(qid)


def group_key(url: str, source_type: str | None = None, *,
              entity_kind=None, qid=None) -> str:
    """Canonical page group a record belongs to. Cards group apart from the
    content chunks of the same url (a card refresh must not delete them)."""
    key = AC.canonical_url(url or "")
    if is_card(source_type, entity_kind, qid):
        key += CARD_SUFFIX
    return key


def payload_group_key(payload: dict | None) -> str:
    """`group_key` of a payload, entity markers included. THE call-site
    spelling: every composer and applier groups a row through this (or
    `Superseder.observe_payload`) so that composer and drain agree on what a
    card is (`is_card`, one implementation)."""
    pl = payload or {}
    return group_key(pl.get("url", ""), pl.get("source_type"),
                     entity_kind=pl.get("entity_kind"), qid=pl.get("qid"))


# --------------------------------------------------------------------------
# entity-card identity (one definition: composer, superseder and drain)
# --------------------------------------------------------------------------
#: Spellings of the one Swiss business-identifier scheme found in stored
#: `register_scheme` values; all fold to `CHE`. The id itself is folded the
#: same way (`CHE-105.974.401` and `CHE105974401` are one token).
_SWISS_SCHEMES = frozenset({"CHE", "CHEUID", "CHUID"})


#: Identity kinds, strongest first (`card_replaces`, `primary_identity`).
IDENTITY_PRECEDENCE = ("reg:", "lei:", "qid:")


def _fold(value) -> str:
    """Upper-case and strip every non-alphanumeric character; "" for a
    missing or non-string value."""
    if not isinstance(value, str):
        return ""
    return "".join(ch for ch in value.upper() if ch.isalnum())


def card_identity(payload: dict | None) -> frozenset[str]:
    """The identity tokens of a card payload: `reg:<SCHEME>:<ID>` (register id),
    `lei:<LEI>`, `qid:<Q...>`. Reads ONLY `register_id`, `register_scheme`,
    `lei` and `qid`. It never reads the payload `point_id`: on the 337,702
    stored cards that field is the URL-only Tier C id even where the real
    point id differs (151,740 rows), so it names no entity.

    The three spellings of the Swiss scheme (`CHE`, `CHE-UID`, `CH-UID`) fold
    to one, and so do the id spellings (`CHE-105.974.401` == `CHE105974401`);
    a register id that begins `CHE` is Swiss whatever scheme it carries. An
    empty set means an anonymous card."""
    pl = payload or {}
    out: set[str] = set()
    rid = _fold(pl.get("register_id"))
    if rid:
        scheme = _fold(pl.get("register_scheme"))
        if scheme in _SWISS_SCHEMES or rid.startswith("CHE"):
            scheme = "CHE"
        out.add(f"reg:{scheme}:{rid}")
    lei = _fold(pl.get("lei"))
    if lei:
        out.add(f"lei:{lei}")
    qid = _fold(pl.get("qid"))
    if qid:
        out.add(f"qid:{qid}")
    return frozenset(out)


def primary_identity(payload: dict | None) -> str | None:
    """The one identity token that names the card: register id, else LEI, else
    QID (a register id names the legal person; a QID can name a building or an
    archive that shares the website). None for an anonymous card. Several
    tokens of one kind: the smallest, so the answer is stable."""
    ids = card_identity(payload)
    for prefix in IDENTITY_PRECEDENCE:
        hit = sorted(t for t in ids if t.startswith(prefix))
        if hit:
            return hit[0]
    return None


def payload_is_card(payload: dict | None) -> bool:
    """`is_card` on a payload's own fields (type, `entity_kind`, `qid`)."""
    pl = payload or {}
    return is_card(pl.get("source_type"), pl.get("entity_kind"), pl.get("qid"))


def _raw_type(payload: dict) -> str:
    st = payload.get("source_type")
    return st.strip().lower() if isinstance(st, str) else ""


def _card_url(payload: dict) -> str:
    """Canonical url of a payload, "" when it has no url or no host (the
    canonicalizer maps "" to a host-less `https:///`, which equals itself)."""
    raw = payload.get("url")
    if not isinstance(raw, str) or not raw.strip():
        return ""
    canon = AC.canonical_url(raw)
    try:
        return canon if urllib.parse.urlsplit(canon).netloc else ""
    except ValueError:
        return ""


def card_replaces(new_payload: dict | None, stored_payload: dict | None) -> bool:
    """True when the incoming card `new_payload` replaces (so may delete) the
    stored card `stored_payload`: BOTH are cards (`payload_is_card`), their
    canonical URLs are equal (and non-empty), their RAW `source_type` is equal
    (an `entity_card` never replaces an `official` row that carries card
    markers, nor the reverse), and their identities overlap BY PRECEDENCE:
    register id, then LEI, then QID. The first kind that BOTH cards carry
    decides alone. Two different register ids are two entities even when a QID
    or LEI matches (one QID can name a group, a building or an archive that
    shares a website); a lower-precedence overlap counts only when at least one
    side lacks the higher-precedence id.

    A stored card with NO identity keeps today's rule. Today's rule is: the
    supersede group of a card is `canonical_url + "|card"`, so ANY incoming
    card at that canonical URL replaces ANY stored card there, whatever its
    identity (the raw-type filter lived only in compose's
    `filter_cross_family_stale` and in the drain). Here that is: stored
    anonymous -> replaced by any incoming card of the same raw type at the
    URL. An incoming card with no identity never replaces a stored card that
    has one (no overlap)."""
    new = new_payload or {}
    stored = stored_payload or {}
    if not (payload_is_card(new) and payload_is_card(stored)):
        return False
    url = _card_url(new)
    if not url or url != _card_url(stored):
        return False
    if _raw_type(new) != _raw_type(stored):
        return False
    stored_ids = card_identity(stored)
    if not stored_ids:
        return True
    new_ids = card_identity(new)
    for prefix in IDENTITY_PRECEDENCE:
        a = {t for t in stored_ids if t.startswith(prefix)}
        b = {t for t in new_ids if t.startswith(prefix)}
        if a and b:
            return bool(a & b)
    return False


def url_variants(url: str) -> set[str]:
    """Stored-url spellings under which this page's existing points may live.
    Stored urls keep scheme + www as captured (aw_common.strip_tracking_url);
    canonical folds both — enumerate the fold preimages we can reconstruct."""
    out = {u for u in (url or "",) if u}
    stripped = AC.strip_tracking_url(url or "")
    if stripped:
        out.add(stripped)
    canon = AC.canonical_url(url or "")
    try:
        p = urllib.parse.urlsplit(canon)
    except ValueError:
        return out
    host = (p.netloc or "").lower()
    if not host:
        return out
    bare = host[4:] if host.startswith("www.") else host
    for scheme in ("https", "http"):
        for h in (bare, "www." + bare):
            out.add(urllib.parse.urlunsplit(
                (scheme, h, p.path, p.query, p.fragment)))
    return out


def guard_scroll(client, collection: str, *, url: str | None = None,
                 reason: str | None = None) -> None:
    """The prod gate for a `client.scroll(collection, ...)`: refuse it when the target is prod
    (the serving alias or collection, the prod tunnel, a listed URL) unless `reason` is named.
    The endpoint is `url`, or what the client was built with; a client that tells nothing is
    judged by its collection name alone."""
    endpoint = url or getattr(getattr(client, "_client", None), "rest_uri", None) or ""
    PG.check(endpoint, "POST", f"/collections/{collection}/points/scroll",
             allow_prod_scroll=bool((reason or "").strip()), reason=reason)


def served_payloads_by_url(client, collection: str, urls, fields, *,
                           group_batch: int = 64) -> dict[str, list[dict]]:
    """For every url in `urls`, the payloads (only `fields`) of the points
    CURRENTLY stored in `collection` under any spelling of that url: one batched
    url-variant scroll per `group_batch` urls, never a per-url round trip. Keyed
    by the STORED `url` value; a caller that joins a different spelling folds both
    sides with `group_key`. A url with nothing stored is simply absent."""
    if qm is None:
        raise RuntimeError("qdrant_client is required for a served-url lookup")
    variant_groups: dict[str, set[str]] = {}
    for url in urls:
        variant_groups.setdefault(url, set()).update(url_variants(url))
    all_urls = sorted(variant_groups)
    want = sorted({"url", *fields})
    out: dict[str, list[dict]] = {}
    for i in range(0, len(all_urls), group_batch):
        chunk = all_urls[i:i + group_batch]
        vals = sorted(set().union(*(variant_groups[u] for u in chunk)))
        if not vals:
            continue
        flt = qm.Filter(must=[qm.FieldCondition(
            key=URL_INDEX_FIELD, match=qm.MatchAny(any=vals))])
        offset = None
        while True:
            guard_scroll(client, collection)        # twin only: prod is refused
            pts, offset = client.scroll(
                collection, scroll_filter=flt, limit=1000, offset=offset,
                with_payload=want, with_vectors=False)
            for pt in pts:
                pl = pt.payload or {}
                out.setdefault(pl.get("url", ""), []).append(pl)
            if offset is None:
                break
    return out


def read_dump_manifest(dump_path: str) -> dict:
    """The dump's sidecar manifest (`<dump>.count`, written by dump_netnew).
    Missing/unreadable -> {} (the caller decides whether that blocks)."""
    try:
        with open(dump_path + ".count") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def dump_is_changed_set(first_record: dict) -> bool:
    """A record carrying `_delta` marks a dump_netnew CHANGED-SET dump (every
    line carries it), as opposed to a full `cyt dump` dump."""
    return "_delta" in first_record


# --------------------------------------------------------------------------
# the rollback export/restore codec (task 574) — ONE HOME for the vector
# encoding a pre-apply export and its restore must agree on byte-for-byte.
# `task 565 split_and_export_565.py` defined this same pair
# independently (task 565, before this module owned rollback at all); moved
# here rather than left duplicated because a mirror-contract pair (encode at
# export time, decode at restore time) belongs in exactly one place
# (engineering-doctrine 1/6) — 565's script now imports it from here.
# --------------------------------------------------------------------------
def encode_point_vectors(vec):
    """A point's named vectors, JSON-safe and LOSSLESS. Dense vectors are a
    plain list; qdrant_client returns a sparse leg as a `SparseVector`
    object that `json.dumps` refuses, so it is encoded explicitly rather
    than coerced — an export that silently dropped the sparse side would
    restore rows the sparse leg (the whole BM25 regression) cannot see."""
    if vec is None:
        return None
    out = {}
    for name, v in vec.items():
        idx = getattr(v, "indices", None)
        if idx is not None:
            out[name] = {"__sparse__": True, "indices": list(idx),
                         "values": list(getattr(v, "values", []))}
        else:
            out[name] = list(v)
    return out


def decode_point_vectors(vec):
    """The inverse of `encode_point_vectors`, for the restore path."""
    if vec is None:
        return None
    out = {}
    for name, v in vec.items():
        if isinstance(v, dict) and v.get("__sparse__"):
            out[name] = qm.SparseVector(indices=v["indices"],
                                        values=v["values"])
        else:
            out[name] = v
    return out


def read_export_ids(path: str) -> set[str]:
    """The id set an export file (this module's own `export_stale` output,
    or an equivalent externally-produced file — e.g. a roster-keyed export
    from an upstream materialize/split step) carries. One line of JSON per
    point, an `id` key; the rest of the line is ignored, so this reads a
    full export (`export_stale`) exactly as well as an ids-only manifest."""
    opener = gzip.open if path.endswith(".gz") else open
    ids: set[str] = set()
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ids.add(str(json.loads(line)["id"]))
    return ids


# --------------------------------------------------------------------------
# gold awareness (reuses gold_guard/gate_authority semantics — never a copy)
# --------------------------------------------------------------------------
def _gold_audit(urls: list[str], log) -> int:
    """Log (never block) superseded urls that are literal eval golds. Returns
    the number of gold hits; -1 => the check could not run (said loudly)."""
    if not urls:
        return 0
    try:
        from cytir.gates import gold_guard as gg
        from cytir.gates import authority as G
        if not any(os.path.exists(p) for _, p in G.SETS):
            raise FileNotFoundError(f"no eval set files under {G.DATA}")
    except Exception as e:  # noqa: BLE001 — loud OFF, not a silent empty
        log(f"[supersede] gold-awareness OFF ({type(e).__name__}: {e}) — "
            f"superseded urls NOT checked against eval golds on this box")
        return -1
    gold_hits, _, _ = gg.audit(urls)
    for h in gold_hits:
        log(f"[supersede] *** GOLD superseded (old version replaced by new): "
            f"{h['url']}  ({h['set']}/{h['case']}) ***")
    return len(gold_hits)


# --------------------------------------------------------------------------
# twin-mirror (task 359 / ADR-081 amendment — the structural fix for the
# 2026-07-30 divergence: a wave was merged live to aw_prod, then a SEPARATE
# background-agent command was supposed to replay it to the build-box twin —
# that command died silently and the twin sat ~8h and one full pack behind,
# only caught by the next morning's cron. The doctrine lesson (background
# waiters banned for appliers) demands the twin write happen INSIDE the same
# process as the prod write, so there is no gap in which a second process can
# die. Everything below is ONE HOME for that: config (env, never a literal —
# engineering doctrine #2), the connection (direct or via a private SSH
# tunnel — twin qdrant is loopback-only on its host, same as
# deploy/scripts/twin-sync-check.sh), and the delta-verification that decides
# whether a prod apply is allowed to report success (engineering doctrine #3
# fail loud / #6 mirror contracts).
# --------------------------------------------------------------------------
def _twin_ssh_host():
    """The twin ssh host, or None when unconfigured: only the tunnel path needs it, and it
    refuses by name (SettingRefused) there. The env is read fresh on every call, as before."""
    if os.environ.get("AW_TWIN_SSH_HOST"):
        return os.environ["AW_TWIN_SSH_HOST"]
    try:
        return operator_settings().twin_ssh_host
    except SettingRefused:
        return None


def twin_config() -> dict:
    """Live-prod-applier twin config. Read fresh on every call (never frozen
    at import) so tests and re-runs never need a code edit to point it
    elsewhere — config is data, not prose."""
    return {
        # one resolve for "the twin" (cytir.core.twin: env -> overlay ->
        # default); AW_TWIN_QDRANT_URL stays the direct-reach override of a scratch pair.
        "url": os.environ.get("AW_TWIN_QDRANT_URL") or TWIN.url(),
        "collection": TWIN.collection(),
        "ssh_host": _twin_ssh_host(),
        "tunnel_local_port": int(os.environ.get("AW_TWIN_TUNNEL_PORT", "16421")),
        "remote_port": int(os.environ.get("AW_TWIN_REMOTE_PORT") or (urllib.parse.urlparse(TWIN.url()).port or 6333)),
        "connect_timeout": float(os.environ.get("AW_TWIN_CONNECT_TIMEOUT", "10")),
        # the escape hatch: genuine twin-down emergencies only. Every caller
        # that honours this MUST log it loudly and ledger it (never a silent
        # fallback — engineering doctrine #3).
        "no_twin": os.environ.get("AW_APPLY_NO_TWIN", "") == "1",
    }


class SSHTunnel:
    """A single-purpose `ssh -L` tunnel, owned start-to-finish by THIS
    process: Popen'd directly (never `-f` + pkill-by-pattern) so the process
    that opened it is the only thing that can close it, and closing it can
    never race another tunnel matching the same fuzzy pattern."""

    def __init__(self, host: str, local_port: int, remote_port: int, *,
                 log=print, timeout: float = 20.0):
        self.host, self.local_port, self.remote_port = host, local_port, remote_port
        self.log, self.timeout = log, timeout
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             "-L", f"{self.local_port}:localhost:{self.remote_port}", self.host],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            rc = self.proc.poll()
            if rc is not None:
                err = self.proc.stderr.read().decode(errors="replace")
                raise SystemExit(
                    f"FATAL: twin ssh tunnel to {self.host!r} exited "
                    f"immediately (rc={rc}): {err.strip()}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex(("127.0.0.1", self.local_port)) == 0:
                    self.log(f"[twin] tunnel up: 127.0.0.1:{self.local_port} "
                             f"-> {self.host}:{self.remote_port}")
                    return self
            time.sleep(0.5)
        self.close()
        raise SystemExit(f"FATAL: twin ssh tunnel to {self.host!r} did not "
                         f"come up within {self.timeout:.0f}s")

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@contextlib.contextmanager
def connect_twin(*, log=print):
    """Yields (QdrantClient, collection) for the twin — direct if
    AW_TWIN_QDRANT_URL is already reachable (the scratch prod+twin test
    pair, or any future direct-network topology), else through a private
    SSH tunnel to AW_TWIN_SSH_HOST (the real build box/serving box topology).
    Raises (SystemExit / the connection error) if neither path reaches it —
    fails loud, before any write; the caller decides whether
    AW_APPLY_NO_TWIN makes that survivable, this function never decides it."""
    cfg = twin_config()
    direct_cli = None
    try:
        cli = qdrant_connect(cfg["url"], timeout=cfg["connect_timeout"])
        cli.get_collections()
        direct_cli = cli
    except Exception as e:  # noqa: BLE001 — fall through to the tunnel path
        log(f"[twin] direct connect to {cfg['url']} failed "
            f"({type(e).__name__}: {e}) — trying ssh tunnel to "
            f"{cfg['ssh_host']!r}")
    if direct_cli is not None:
        log(f"[twin] direct: {cfg['url']}")
        yield direct_cli, cfg["collection"]
        return
    with SSHTunnel(cfg["ssh_host"] or operator_settings().twin_ssh_host,
                   cfg["tunnel_local_port"], cfg["remote_port"],
                   log=log):
        tunnel_url = f"http://127.0.0.1:{cfg['tunnel_local_port']}"
        cli = qdrant_connect(tunnel_url, timeout=max(cfg["connect_timeout"], 60))
        cli.get_collections()
        log(f"[twin] via tunnel: {tunnel_url} -> {cfg['ssh_host']}:"
            f"{cfg['remote_port']}")
        yield cli, cfg["collection"]


def verify_count_delta(client, collection: str, before: int, after: int,
                       expected_delta: int, *, label: str, log=print) -> None:
    """The mirror-contract check (engineering doctrine #6): a write is not
    'done', it is done-and-verified. Raises SystemExit on mismatch — the
    caller must not print a success line / return 0 past this call."""
    actual = after - before
    if actual != expected_delta:
        raise SystemExit(
            f"FATAL: {label} count delta MISMATCH on {collection!r} — "
            f"{before:,} -> {after:,} (delta {actual:+,}) but expected "
            f"{expected_delta:+,}. Apply is NOT reportable as success; "
            f"collection state untrustworthy — do not proceed to gate/promote.")
    log(f"[{label}] VERIFIED — delta {actual:+,} matches expected on "
        f"{collection!r}")


# --------------------------------------------------------------------------
# the card-overwrite guard for every BY-ID writer outside the drain (task 820)
# --------------------------------------------------------------------------
class CardOverwriteRefused(RuntimeError):
    """A by-id writer's batch would overwrite a stored entity card with a row that is not a card
    that replaces it. The text names the writer, each id and the stored card's url (the drain's
    own reason text, `promote_queue.check_card_overwrites`, prefixed with the writer)."""


def client_url(client) -> str:
    """The REST url a `QdrantClient` talks to (the card check reads over HTTP, like the drain)."""
    inner = getattr(client, "_client", None)
    url = getattr(inner, "rest_uri", None)
    if not isinstance(url, str) or not url:
        raise CardOverwriteRefused(
            f"cannot tell which endpoint {type(client).__name__} talks to -- refusing the write "
            "rather than skipping the card check")
    return url


def load_card_waivers(path: str | None, sha256: str | None) -> dict:
    """The waiver rows a writer was handed (`--card-waivers PATH --card-waivers-sha256 SHA`): the
    same sealed file format the drain reads (`card_waivers.load_waivers`). None -> no waivers."""
    if bool(path) != bool(sha256):
        raise CardOverwriteRefused("--card-waivers and --card-waivers-sha256 go together")
    if not path:
        return {}
    from cytir.index import card_waivers as CW      # lazy: only a caller with waivers needs it
    import pathlib
    try:
        return CW.load_waivers(pathlib.Path(path), expected_sha256=sha256)
    except CW.CardWaiverError as exc:
        raise CardOverwriteRefused(f"card waiver file refused: {exc}") from exc


def guard_card_overwrites(rows, url: str, collection: str, *, writer: str,
                          waivers: dict | None = None, chunk: int = 5000, log=print) -> dict:
    """THE card guard of a by-id writer: refuse the WHOLE batch before the first write when any
    stored card at `collection` (read at `url`) would be overwritten by id with a row that is not
    a card replacing it. It runs `promote_queue.check_card_overwrites` -- the drain's check, no
    second implementation -- with `hold_out=False`, so a refusal is the drain's own reason text.

    `rows`: an iterable of `{"id", "payload"}` dicts or objects with `.id`/`.payload`; pass only
    the rows the writer will WRITE. They are checked `chunk` at a time (one indexed id retrieve per
    500 ids through `PSP.retrieve_payloads`: no scroll, paced and replay-refused by `prod_guard`),
    so memory stays flat. Returns `{"writer", "ids_read", "stored_cards", "waived"}`; each waiver
    used is logged. A read that fails is NOT a pass: the error propagates."""
    from cytir.index import promote_queue as PQ     # lazy: promote_queue imports this module

    report = {"writer": writer, "ids_read": 0, "stored_cards": 0, "waived": []}
    buf: list[dict] = []

    def _flush() -> None:
        if not buf:
            return
        try:
            got = PQ.check_card_overwrites(buf, url, collection, hold_out=False, waivers=waivers)
        except PQ.QueueError as exc:
            raise CardOverwriteRefused(f"{writer}: {exc}") from exc
        report["ids_read"] += got["ids_read"]
        report["stored_cards"] += got["stored_cards"]
        for w in got["waived"]:
            report["waived"].append(w)
            log(f"[card-guard] {writer}: waiver admitted id {w['id']} pr={w['pr']} "
                f"evidence_sha256={w['evidence_sha256']}")
        buf.clear()

    for row in rows:
        if isinstance(row, dict):
            rid, payload = row["id"], row.get("payload")
        else:
            rid, payload = row.id, getattr(row, "payload", None)
        buf.append({"id": rid, "payload": payload or {}})
        if len(buf) >= chunk:
            _flush()
    _flush()
    log(f"[card-guard] {writer}: {report['ids_read']} stored row(s) read on {collection}, "
        f"{report['stored_cards']} stored card(s), no overwrite refused")
    return report


# --------------------------------------------------------------------------
# the named inverse of a supersede (task 574)
# --------------------------------------------------------------------------
def restore_overwritten_from_export(dst_client, collection: str,
                                    export_path: str, *, batch: int = 256,
                                    sample: int = 20, log=print) -> dict:
    """Put back the PRE-IMAGES `cyt index pack-delta --export-overwritten` saved.

    `restore_from_export` above is the inverse of a DELETE: it verifies
    `before - removed + restored == after`. Re-upserting an id that still
    exists changes no count, so that arithmetic is the wrong instrument here
    and using it would either fail honestly or, if softened, hide a restore
    that never landed. This is the inverse of an OVERWRITE, and its checks
    are the ones that can fail for it:

      * `points_count` must be UNCHANGED (an overwrite-restore that changes
        the count wrote an id the target did not already hold);
      * EVERY id in the export must already exist in the target before the
        restore -- an absent one means this file is not the pre-image of the
        current state (e.g. the delta was already reverted another way);
      * a deterministic sample is read back WITH payload and vectors and
        compared byte-for-byte against the export, through the same codec it
        was written with.

    Task 576. Raises SystemExit on any mismatch.
    """
    opener = gzip.open if export_path.endswith(".gz") else open
    rows: list[dict] = []
    with opener(export_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by_id = {str(r["id"]): r for r in rows}
    if not rows:
        log(f"[restore-overwritten] {export_path}: no pre-images; nothing "
            "to put back")
        return {"restored": 0, "count_before": None, "count_after": None}

    before = dst_client.count(collection, exact=True).count
    ids = [r["id"] for r in rows]
    present = set()
    for i in range(0, len(ids), batch):
        present.update(str(p.id) for p in dst_client.retrieve(
            collection, ids=ids[i:i + batch], with_payload=False,
            with_vectors=False))
    absent = sorted(set(by_id) - present)
    if absent:
        raise SystemExit(
            f"FATAL: {len(absent):,} of {len(rows):,} pre-image id(s) are not "
            f"in {collection!r} (first {absent[:3]}). This file is not the "
            "pre-image of the current state; restoring it would ADD rows "
            "rather than put overwritten ones back.")

    log(f"[restore-overwritten] {collection}: {before:,} before, "
        f"{len(rows):,} pre-image(s) to put back")
    restored = 0
    pending: list = []
    for r in rows:
        pending.append(qm.PointStruct(
            id=r["id"], vector=decode_point_vectors(r["vector"]),
            payload=r["payload"]))
        if len(pending) >= batch:
            dst_client.upsert(collection, points=pending, wait=True)
            restored += len(pending)
            pending = []
    if pending:
        dst_client.upsert(collection, points=pending, wait=True)
        restored += len(pending)

    after = dst_client.count(collection, exact=True).count
    if after != before:
        raise SystemExit(
            f"FATAL: overwrite-restore changed {collection!r}'s count "
            f"{before:,} -> {after:,}. Every pre-image id was verified present "
            "beforehand, so a count change means the collection moved under "
            "the restore. Do not report this as done.")

    sample_ids = sorted(by_id)
    if len(sample_ids) > sample:
        sample_ids = random.Random(0).sample(sample_ids, sample)
    back = {str(p.id): p for p in dst_client.retrieve(
        collection, ids=sample_ids, with_payload=True, with_vectors=True)}
    for pid in sample_ids:
        got = back.get(pid)
        if got is None:
            raise SystemExit(f"FATAL: restored id {pid} not readable back")
        want = by_id[pid]
        if got.payload != want["payload"]:
            raise SystemExit(
                f"FATAL: restored id {pid} carries a different payload than "
                "the pre-image held. A count match proves arithmetic, never "
                "content.")
        if encode_point_vectors(got.vector or {}) != want["vector"]:
            raise SystemExit(
                f"FATAL: restored id {pid} carries a different vector than "
                "the pre-image held.")
    log(f"[restore-overwritten] {restored:,} pre-image(s) restored, count "
        f"unchanged at {after:,}, {len(sample_ids)} sampled id(s) read back "
        "byte-identical")
    return {"restored": restored, "count_before": before, "count_after": after,
            "sampled": len(sample_ids)}


def restore_from_export(dst_client, collection: str, export_path: str, *,
                        remove_ids: list | None = None, batch: int = 256,
                        sample: int = 20, log=print) -> dict:
    """Undo one `Superseder.run(export_path=...)` write: delete
    `remove_ids` (the points a delta ADDED, if any — the mirror of
    `task 565 restore_superseded_565.py`'s step 1) and re-insert
    every record `export_path` holds, decoded via `decode_point_vectors`.

    Two verifications, not one (task 574 — a count match proves
    arithmetic, never content):
      * `points_count` before/after reconciles to `before - removed +
        restored` — `restore_superseded_565.py`'s own check, unchanged.
      * a SAMPLE of the restored ids (up to `sample`, deterministic
        selection) is retrieved back WITH payload and vectors and compared
        byte-for-byte (via `encode_point_vectors`, the same codec the
        export was written with) against what the export file held — the
        check a bare count cannot make, and the one that would have caught
        a restore that landed the wrong payload/vector on a right id.

    Raises SystemExit on any mismatch — a restore that cannot prove it
    worked must not be reported as done.
    """
    opener = gzip.open if export_path.endswith(".gz") else open
    rows: list[dict] = []
    with opener(export_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by_id = {str(r["id"]): r for r in rows}

    before = dst_client.count(collection, exact=True).count
    log(f"[restore] {collection}: {before:,} before "
       f"({len(rows):,} preserved point(s) to re-insert)")

    remove_ids = [str(i) for i in (remove_ids or [])]
    for i in range(0, len(remove_ids), batch):
        dst_client.delete(collection, points_selector=qm.PointIdsList(
            points=remove_ids[i:i + batch]), wait=True)

    restored = 0
    pending: list = []
    for r in rows:
        pending.append(qm.PointStruct(
            id=r["id"], vector=decode_point_vectors(r["vector"]),
            payload=r["payload"]))
        if len(pending) >= batch:
            dst_client.upsert(collection, points=pending, wait=True)
            restored += len(pending)
            pending = []
    if pending:
        dst_client.upsert(collection, points=pending, wait=True)
        restored += len(pending)

    after = dst_client.count(collection, exact=True).count
    expected = before - len(remove_ids) + restored
    if after != expected:
        raise SystemExit(
            f"FATAL: restore count MISMATCH on {collection!r} — "
            f"{before:,} -> {after:,} (delta {after - before:+,}) but "
            f"expected {expected - before:+,} (removed {len(remove_ids):,}, "
            f"restored {restored:,}). Collection state untrustworthy — do "
            f"not report this restore as done.")

    # -- sample readback: prove the restored points are what the export
    # held, not merely that the right NUMBER of points exist (task 574).
    sample_ids = sorted(by_id)
    if len(sample_ids) > sample:
        sample_ids = random.Random(0).sample(sample_ids, sample)
    back = {str(p.id): p for p in dst_client.retrieve(
        collection, ids=sample_ids, with_payload=True, with_vectors=True)}
    missing = [i for i in sample_ids if i not in back]
    if missing:
        raise SystemExit(
            f"FATAL: restore sample readback on {collection!r} — "
            f"{len(missing)} of {len(sample_ids)} sampled id(s) not found "
            f"after restore: {missing[:5]}. Collection state untrustworthy.")
    mismatched = [
        i for i in sample_ids
        if (back[i].payload or {}) != by_id[i]["payload"]
        or encode_point_vectors(back[i].vector) != by_id[i]["vector"]]
    if mismatched:
        raise SystemExit(
            f"FATAL: restore sample readback on {collection!r} — "
            f"{len(mismatched)} of {len(sample_ids)} sampled id(s) do not "
            f"match the export (payload or vector differs): "
            f"{mismatched[:5]}. Collection state untrustworthy.")

    log(f"[restore] {collection}: {before:,} -> {after:,} "
       f"(removed {len(remove_ids):,}, restored {restored:,}); sample "
       f"readback OK ({len(sample_ids)}/{len(rows)} checked, byte-exact)")
    return {"before": before, "after": after, "removed": len(remove_ids),
           "restored": restored, "sample_checked": len(sample_ids),
           "sample_ok": True}


# --------------------------------------------------------------------------
# the superseder
# --------------------------------------------------------------------------
class Superseder:
    """Accumulates the incoming record set during an applier's streaming pass
    (observe), then reconciles every touched page group (run)."""

    def __init__(self, client, collection: str, *, log=print, dry_run=False,
                url: str | None = None, prod_scroll_reason: str | None = None):
        if qm is None:
            raise RuntimeError("qdrant_client not installed — Superseder "
                               "requires it (pure helpers do not)")
        self.c = client
        self.coll = collection
        # a scroll of a PROD target is refused unless an operator names a reason (2026-10-07)
        self.prod_scroll_reason = prod_scroll_reason
        self.log = log
        self.dry_run = dry_run
        # task 641: an explicit REST url routes the actual delete through
        # `point_delete.delete_points_verified`, the one home this class and
        # `cytir/index/promote_queue.py`'s tombstone apply now share. Optional,
        # additive: every EXISTING caller constructs without `url` and keeps
        # the byte-identical qdrant_client delete+retrieve below -- this
        # never changes behaviour for a caller that doesn't opt in.
        self.url = url
        self.keep: dict[str, set] = defaultdict(set)      # group -> incoming ids
        self.variants: dict[str, set] = defaultdict(set)  # group -> url spellings
        self.global_keep: set = set()                     # every incoming id
        # group -> the identity-bearing fields of each incoming CARD; a stored
        # card is stale only when `card_replaces` holds against one of them
        self.cards: dict[str, list[dict]] = defaultdict(list)
        # Populated by run() in deterministic order for callers that bind an
        # explicit predecessor-pointer manifest before a canonical write.
        self.last_stale_ids: list[str] = []
        self.stats = {"groups": 0, "existing_seen": 0, "superseded": 0,
                      "gold_superseded": 0, "unmatched_hits": 0}

    # -- accumulation ------------------------------------------------------
    def observe(self, point_id, url: str, source_type: str | None = None, *,
                entity_kind=None, qid=None, register_id=None,
                register_scheme=None, lei=None):
        """Record one incoming record (upserted OR unchanged-marker). An
        incoming card's identity fields (`register_id`, `register_scheme`,
        `lei`, `qid`) decide which stored cards it replaces."""
        if not url:
            return
        g = group_key(url, source_type, entity_kind=entity_kind, qid=qid)
        if is_card(source_type, entity_kind, qid):
            self.cards[g].append({
                "url": url, "source_type": source_type, "entity_kind": entity_kind,
                "qid": qid, "register_id": register_id,
                "register_scheme": register_scheme, "lei": lei})
        self.keep[g].add(point_id)
        self.variants[g].update(url_variants(url))
        self.global_keep.add(point_id)

    def observe_payload(self, point_id, payload: dict | None):
        """`observe` with the row's own entity markers and identity fields."""
        pl = payload or {}
        self.observe(point_id, pl.get("url", ""), pl.get("source_type"),
                     entity_kind=pl.get("entity_kind"), qid=pl.get("qid"),
                     register_id=pl.get("register_id"),
                     register_scheme=pl.get("register_scheme"), lei=pl.get("lei"))

    # -- index precondition ------------------------------------------------
    def ensure_url_index(self, create: bool = False) -> None:
        """The keyword payload index on `url` is the lookup's precondition.
        Missing + create=False -> SystemExit (fail loud, BEFORE any write —
        an unindexed filter is a full scan per page: the fragile-scrolls
        incident class). create=True is for OFFLINE clones/scratch only."""
        info = self.c.get_collection(self.coll)
        schema = info.payload_schema or {}
        ent = schema.get(URL_INDEX_FIELD)
        if ent is not None and str(getattr(ent, "data_type", "")).lower() \
                .endswith("keyword"):
            return
        if not create:
            raise SystemExit(
                f"FATAL: collection {self.coll!r} has no keyword payload index "
                f"on {URL_INDEX_FIELD!r} — supersede-on-upsert needs it. Create "
                f"it once (offline appliers: --create-url-index / this module's "
                f"CLI flag; live prod: an ops change, mirror it on the twin), "
                f"or pass --no-supersede to run the legacy duplicate-leaving "
                f"behaviour on purpose.")
        self.log(f"[supersede] creating keyword payload index on "
                 f"{URL_INDEX_FIELD!r} in {self.coll} ...")
        self.c.create_payload_index(self.coll, field_name=URL_INDEX_FIELD,
                                    field_schema=qm.PayloadSchemaType.KEYWORD,
                                    wait=True)

    # -- selector (task 574) ------------------------------------------------
    def _scan_stale(self, group_batch: int = 64):
        """The live group-scroll that decides what WOULD be deleted — the
        ONE selector. `run()` and `export_stale()` both call this rather
        than re-deriving "what will be deleted" independently: task 572's
        315-point gap was exactly two different answers to that question
        (a materialize-time roster ledger vs. this live scroll), never
        reconciled before the delete ran."""
        groups = sorted(self.keep)
        self.stats["groups"] = len(groups)
        stale_ids: list = []
        stale_urls: list[str] = []
        per_domain = defaultdict(int)

        for i in range(0, len(groups), group_batch):
            chunk = groups[i:i + group_batch]
            vals = sorted(set().union(*(self.variants[g] for g in chunk)))
            flt = qm.Filter(must=[qm.FieldCondition(
                key=URL_INDEX_FIELD, match=qm.MatchAny(any=vals))])
            offset = None
            while True:
                guard_scroll(self.c, self.coll, url=self.url, reason=self.prod_scroll_reason)
                pts, offset = self.c.scroll(
                    self.coll, scroll_filter=flt, limit=1000, offset=offset,
                    with_payload=[URL_INDEX_FIELD, "source_type", "domain",
                                  "last_seen", "entity_kind", "qid",
                                  "register_id", "register_scheme", "lei"],
                    with_vectors=False)
                for p in pts:
                    pl = p.payload or {}
                    g = group_key(pl.get("url", ""), pl.get("source_type"),
                                  entity_kind=pl.get("entity_kind"),
                                  qid=pl.get("qid"))
                    if g not in self.keep:
                        # a card/content sibling outside the delta, or an
                        # unobserved group a variant grazed — never touch it
                        self.stats["unmatched_hits"] += 1
                        continue
                    self.stats["existing_seen"] += 1
                    if p.id in self.keep[g] or p.id in self.global_keep:
                        continue
                    if g.endswith(CARD_SUFFIX) and not any(
                            card_replaces(inc, pl) for inc in self.cards[g]):
                        # a card of another entity at the same url (or of
                        # another raw type): never replaced by this pack
                        self.stats["unmatched_hits"] += 1
                        continue
                    stale_ids.append(p.id)
                    stale_urls.append(pl.get("url", ""))
                    per_domain[pl.get("domain") or "?"] += 1
                if offset is None:
                    break

        self.last_stale_ids = sorted(str(point_id) for point_id in stale_ids)
        return stale_ids, stale_urls, per_domain

    # -- rollback export (task 574) -----------------------------------------
    def export_stale(self, path: str, stale_ids: list | None = None,
                     batch: int = 256) -> dict:
        """Write the FULL rollback copy (dense + sparse vectors + payload,
        `encode_point_vectors`) of `stale_ids` to `path` (gzip iff it ends
        `.gz`) — defaulting to a fresh `_scan_stale()` when `stale_ids` is
        not given. Called with no `stale_ids`, this scans the exact same
        live selector `run()` deletes by, so the export cannot under-cover
        the delete by construction — the replacement for a
        materialize-time/roster-keyed pre-export computed by a different
        route, hours earlier. Returns `{"path", "count", "ids"}`."""
        if stale_ids is None:
            stale_ids, _, _ = self._scan_stale()
        opener = gzip.open if path.endswith(".gz") else open
        exported = 0
        ids_out: set[str] = set()
        with opener(path, "wt", encoding="utf-8") as out:
            for i in range(0, len(stale_ids), batch):
                part = stale_ids[i:i + batch]
                if not part:
                    continue
                got = self.c.retrieve(self.coll, ids=part, with_payload=True,
                                      with_vectors=True)
                for pt in got:
                    out.write(json.dumps({
                        "id": str(pt.id), "payload": pt.payload,
                        "vector": encode_point_vectors(pt.vector)},
                        ensure_ascii=False) + "\n")
                    exported += 1
                    ids_out.add(str(pt.id))
        return {"path": path, "count": exported, "ids": ids_out}

    def _export_and_verify(self, stale_ids: list, export_path, expect_ids):
        """Shared by the dry-run and real-run branches of `run()`: writes
        `export_path` (if given) from `stale_ids` and/or refuses when
        `expect_ids` does not match it exactly — BOTH before any delete.
        Additive to `self.stats`; never removes or renames an existing key,
        so a caller that passes neither argument sees the pre-574 stats
        dict byte-for-byte (task 574)."""
        if export_path is not None:
            result = self.export_stale(export_path, stale_ids=stale_ids)
            self.stats["export_path"] = export_path
            self.stats["export_count"] = result["count"]
        if expect_ids is not None:
            live_set = {str(i) for i in stale_ids}
            expect_set = {str(i) for i in expect_ids}
            self.stats["expect_count"] = len(expect_set)
            self.stats["live_delete_count"] = len(live_set)
            if live_set != expect_set:
                sym_diff = live_set ^ expect_set
                self.stats["rollback_verified"] = False
                self.stats["symmetric_difference"] = len(sym_diff)
                self.log(
                    f"[supersede] REFUSED — the pre-apply export/expected "
                    f"id set does not match the live delete set: expected "
                    f"{len(expect_set):,} id(s), live delete set "
                    f"{len(live_set):,} id(s), symmetric difference "
                    f"{len(sym_diff):,} id(s). An apply whose rollback "
                    f"cannot restore everything it is about to delete is "
                    f"not reportable as safe.")
                raise SystemExit(
                    f"FATAL: rollback export/expected id set "
                    f"({len(expect_set):,}) != live delete set "
                    f"({len(live_set):,}) on {self.coll!r} — symmetric "
                    f"difference {len(sym_diff):,}. Refusing to supersede "
                    f"(task 574).")
            self.stats["rollback_verified"] = True
            self.log(
                f"[supersede] rollback verified — expected and live delete "
                f"id sets match exactly ({len(live_set):,} id(s))")

    # -- reconcile ---------------------------------------------------------
    def run(self, group_batch: int = 64, delete_batch: int = 1000, *,
            export_path: str | None = None,
            expect_ids=None) -> dict:
        """Reconcile all observed groups. Returns the stats dict. Deletions
        are wait=True and verified gone (retrieve must return nothing).

        `export_path` / `expect_ids` (task 574) are additive and OFF by
        default: a caller that passes neither sees byte-for-byte the
        pre-574 stats and log lines. When given, both are checked BEFORE
        any delete (dry-run or real) against `_scan_stale()`'s own result —
        never a second, independently-derived set: see `_export_and_verify`.
        """
        stale_ids, stale_urls, per_domain = self._scan_stale(group_batch)

        if export_path is not None or expect_ids is not None:
            self._export_and_verify(stale_ids, export_path, expect_ids)

        if self.dry_run:
            self.stats["superseded"] = len(stale_ids)
            self.log(f"[supersede] DRY RUN — would delete {len(stale_ids):,} "
                     f"stale points across {self.stats['groups']:,} groups")
            return self.stats

        if self.url:
            # task 641: the shared primitive. Same semantics (wait=true per
            # batch, re-read to verify gone) -- routed through the REST API
            # rather than this instance's qdrant_client, because
            # `cytir/index/promote_queue.py`'s tombstone apply has no open
            # QdrantClient of its own and this is the one home both now use.
            try:
                PD.delete_points_verified(self.url, self.coll, stale_ids,
                                          batch=delete_batch, log=self.log)
            except PD.PointDeleteError as exc:
                raise SystemExit(f"FATAL: {exc} — do not proceed to "
                                 "gate/promote") from exc
        else:
            for i in range(0, len(stale_ids), delete_batch):
                ids = stale_ids[i:i + delete_batch]
                self.c.delete(self.coll,
                              points_selector=qm.PointIdsList(points=ids),
                              wait=True)
                back = self.c.retrieve(self.coll, ids=ids, with_payload=False,
                                       with_vectors=False)
                if back:  # deletion is not done until it is VERIFIED gone
                    raise SystemExit(
                        f"FATAL: {len(back)} superseded ids still present after "
                        f"delete(wait=True) — collection state untrustworthy, "
                        f"do not proceed to gate/promote")
        self.stats["superseded"] = len(stale_ids)

        gold_n = _gold_audit(stale_urls, self.log)
        self.stats["gold_superseded"] = max(gold_n, 0)
        top = sorted(per_domain.items(), key=lambda kv: -kv[1])[:8]
        self.log(f"[supersede] {self.stats['superseded']:,} stale points "
                 f"deleted across {self.stats['groups']:,} touched page groups "
                 f"(existing seen {self.stats['existing_seen']:,}; gold hits "
                 f"{self.stats['gold_superseded']}"
                 f"{' [check OFF]' if gold_n < 0 else ''})"
                 + (f"; top domains: {top}" if top else ""))
        return self.stats


# --------------------------------------------------------------------------
# CLI — after-the-fact reconcile of an already-applied dump
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="reconcile page groups of an already-applied wave dump")
    ap.add_argument("--in", dest="inp", required=True,
                    help="the jsonl(.gz) dump whose records define keep-sets")
    ap.add_argument("--qdrant-url", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--assume-complete", action="store_true",
                    help="assert the dump carries the COMPLETE record set of "
                         "every page it touches (required for changed-set "
                         "dumps without a supersede_complete manifest)")
    ap.add_argument("--create-url-index", action="store_true",
                    help="create the url keyword index if missing (OFFLINE "
                         "clones/scratch only — never casually on live prod)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run:
        # task 864 (review 854 S6): a reconcile deletes rows; on a held twin only the drain writes
        from cytir.index import twin_write_hold as TWH
        try:
            TWH.refuse_if_held(args.collection, tool="cytir.index.supersede")
        except TWH.TwinWriteHeld as exc:
            raise SystemExit(str(exc)) from exc

    c = qdrant_connect(args.qdrant_url, timeout=600)
    sup = Superseder(c, args.collection, dry_run=args.dry_run)
    sup.ensure_url_index(create=args.create_url_index)

    opener = gzip.open if args.inp.endswith(".gz") else open
    first = None
    n = 0
    with opener(args.inp, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if first is None:
                first = rec
                if dump_is_changed_set(rec) and not args.assume_complete and \
                        not read_dump_manifest(args.inp).get("supersede_complete"):
                    raise SystemExit(
                        "FATAL: changed-set dump without supersede_complete "
                        "manifest — its keep-sets are NOT page-complete and "
                        "reconciling would delete unchanged siblings. Re-dump "
                        "with the format-2 dump_netnew, or pass "
                        "--assume-complete if you can prove completeness.")
            pl = rec.get("payload") or {}
            sup.observe_payload(rec["id"], pl)
            n += 1
    print(f"[supersede-cli] observed {n:,} records "
          f"({len(sup.keep):,} page groups) from {args.inp}")
    sup.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
