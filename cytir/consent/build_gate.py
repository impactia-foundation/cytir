"""Builder consent gate: `ConsentGate`, the ADR-076 dump-basis evidence
predicate and the host helpers (moved from the b1 builder script, task 2077).

The code below is the original block, unchanged. It is NOT merged with
`cytir.consent.gate` (a later dedupe)."""
from __future__ import annotations

import json
import os
from collections import defaultdict

from cytir.core.paths import data_path

UNDATED_GOV_HOST = json.loads(
    data_path("consent-undated-gov-hosts.v1.json").read_text("utf-8"))["undated_gov_host"]


# --------------------------------------------------------------------------
# ADR-076 dump-basis admission
# --------------------------------------------------------------------------
# "An explicit open license on an official bulk distribution constitutes the
# admission basis (`search_ok`) for records built from that distribution,
# independent of the canonical host's robots/WAF state… A robots-based decline
# of the host does NOT evict dump-based records; conversely the dump basis
# extends only to what the license actually covers — `train_ok` follows the
# license terms, not the admission ruling."  (ADR-076, ratified Tim 2026-07-25;
# the ConsentGate exemption it names as required next-rebuild work.)
DUMP_CONSENT_STATE = "dump"
# A CLOSED set, not free text. `consent_source` names the distribution and is
# written by each ingester; `consent_basis` is the ruling under which it was
# admitted, and only a basis this build recognizes may override a host decline.
DUMP_CONSENT_BASES = frozenset({"official-dump-license"})


def dump_basis_evidence(payload: dict | None) -> dict | None:
    """Return the ADR-076 dump-basis evidence on `payload`, else None.

    ADR-076 admits on the LICENSE GRANT of an official bulk distribution, so
    the gate requires the evidence of that grant rather than the word "dump":
    a bare `consent_state` stamp would otherwise become a way to launder an
    ordinary robots decline into an admission. All four are required —

      consent_state == "dump"               the route declaration
      consent_basis in DUMP_CONSENT_BASES   a RECOGNIZED basis (closed set)
      consent_source  non-blank             the named distribution granting it
      license         non-blank             the explicit license ADR-076 needs

    — and all four are carried by both live dump packs (read 2026-08-19):
    `aw_dfrlaw_cov` (DILA / `Etalab-2.0`) and `aw_dfedlex2_cov` (Fedlex Akoma
    Ntoso / `SwissGov-Fedlex-Terms`), the fedlex route ADR-076 cites as the
    precedent this generalizes.

    This decides `search_ok` ONLY. `train_ok` follows the licence terms and is
    never written, inferred or upgraded here (ADR-076's explicit bound).
    """
    if not payload:
        return None
    if str(payload.get("consent_state") or "").strip().lower() != DUMP_CONSENT_STATE:
        return None
    basis = str(payload.get("consent_basis") or "").strip().lower()
    if basis not in DUMP_CONSENT_BASES:
        return None
    source = str(payload.get("consent_source") or "").strip()
    licence = str(payload.get("license") or "").strip()
    if not source or not licence:
        return None
    return {"consent_basis": basis, "consent_source": source,
            "license": licence}


# --------------------------------------------------------------------------
# consent gate — 30d grace, fail-closed (architecture.md §5 step 2)
# --------------------------------------------------------------------------
class ConsentGate:
    """Build-stage consent. Verdict sources, in order:
      0. ADR-076 dump basis on the record payload (admit-only; overrides a
         canonical-host decline for records built from an openly-licensed
         official bulk distribution — see dump_basis_evidence)
      1. corpus row consent_search_ok/consent_train_ok/consent_checked_at
      2. persistent host cache (this run + prior runs)
      3. live resolve_host (mode='live' only)

    Exclusion policy (task 319 ruling, after the 318 gate FAIL evicted 1.79M
    points on absence-of-data; cites: architecture.md §5 step 2 "a record whose
    consent endpoint is unreachable keeps its last-known verdict for a bounded
    grace window (default 30 days), then fails closed"; the aw_b1 consent
    census precedent where the 7.7% unreachable-canary-clean class was
    retry-queued while its records kept serving in prod; ADR-073 for what
    counts as a resolved DECLINE):

      * EVICTION happens only on a RESOLVED DECLINE (search_ok=False — robots
        readable and disallowing, or robots-4xx per ADR-073) or on grace
        EXPIRY, never on absence-of-data.
      * UNREACHABLE / resolver-error / census-ABSENT hosts are RETAINED under
        their last-known verdict (they are serving in prod today) with a
        grace stamp (consent_state=grace + consent_grace_until) and pushed to
        consent_retry_queue.txt; the weekly delta job re-resolves and evicts
        on decline or on grace expiry without resolution.
      * verdict() reasons: "grace:…" = admitted-with-grace (caller stamps the
        payload); "defer:…" = not yet resolved (pipelined main pass only —
        queued for the deferred pass). After the census is complete
        (post_census=True, set by the deferred/staged phases and by
        --consent-census-complete in slice mode) a cache-MISS is a
        census-absent host -> grace-retain, NOT exclude: the 318 build's
        deferred pass excluded 100% of 929,826 uncensused points (whole
        domains of a US central bank, a university encyclopedia, an academic wiki, …).

    In mode='trust' (slice default) a missing verdict is ALLOWED-with-flag
    (counted consent_unresolved) so the slice stays tractable without
    thousands of fetches.
    """

    def __init__(self, mode: str, grace_days: int, cache_path: str, log=print):
        self.mode = mode          # 'live' | 'trust'
        self.grace = grace_days
        self.cache_path = cache_path
        self.log = log
        self.host_cache: dict[str, dict] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                self.host_cache = json.load(open(cache_path))
            except Exception:  # noqa: BLE001
                self.host_cache = {}
        self._client = None
        self.stats = defaultdict(int)
        # post_census: the host census is COMPLETE (CONSENT.DONE) — a cache
        # miss now means census-absent, which grace-retains instead of defers.
        self.post_census = False
        import datetime as _dt
        self.grace_until = (_dt.date.today()
                            + _dt.timedelta(days=grace_days)).isoformat()
        # unreachable-but-canary-clean hosts -> recoverable via a weekly delta
        # re-check (Tim's consent refinement); persisted next to the cache.
        self.retry_hosts: set[str] = set()
        self.retry_path = (os.path.join(os.path.dirname(cache_path),
                                        "consent_retry_queue.txt")
                           if cache_path else "")
        if self.retry_path and os.path.exists(self.retry_path):
            self.retry_hosts = {ln.strip() for ln in open(self.retry_path) if ln.strip()}

    def _fresh(self, checked_at) -> bool:
        if not checked_at:
            return False
        try:
            import datetime as dt
            t = str(checked_at).replace("Z", "+00:00")
            d = dt.datetime.fromisoformat(t)
            age = (dt.datetime.now(dt.timezone.utc) - d.astimezone(dt.timezone.utc)).days
            return age <= self.grace
        except Exception:  # noqa: BLE001
            return False

    def _live(self, host: str) -> dict:
        if host in self.host_cache:
            return self.host_cache[host]
        if self.mode != "live":
            return {}
        try:
            import cytir.consent.resolver as CR
            if self._client is None:
                self._client = CR._client()
            rec = CR.resolve_host(self._client, host)
        except Exception as e:  # noqa: BLE001
            rec = {"status": "unknown", "search_ok": None, "train_ok": None,
                   "reasons": [f"resolver error: {e}"]}
        self.host_cache[host] = rec
        return rec

    def reload(self):
        """Re-read the (periodically rsync'd) cache — for pipelined mode where the
        the consent runners fill it live during the base pass."""
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                self.host_cache = json.load(open(self.cache_path))
            except Exception:  # noqa: BLE001
                pass

    def verdict(self, host: str, corpus_row: dict | None,
                payload: dict | None = None) -> tuple[bool, str]:
        """Return (admit, reason). reason starting 'defer:' = not yet resolved
        (pipelined mode) -> the caller queues it for the deferred pass."""
        # 0) ADR-076 DUMP BASIS — before every other branch.
        #
        # The record was built from an official bulk distribution whose open
        # license IS the admission basis, so the canonical host's robots/WAF
        # state does not govern it. This sits at the TOP deliberately: the
        # pipelined branch below returns without ever consulting the corpus
        # row, and a full rebuild runs pipelined, so an exemption placed only
        # ahead of the stale-positive re-resolution would never fire on the
        # path that actually rebuilds (task 322 predicted exactly this: the
        # staged pack re-projected through the robots gate and DECLINED).
        #
        # Narrow by construction — it can only ADMIT, only for a record
        # carrying complete dump evidence (dump_basis_evidence), and it never
        # touches `train_ok`. Absence-of-data handling below is untouched
        # (task 318/319: eviction only on a resolved DECLINE or grace expiry).
        dump = dump_basis_evidence(payload)
        if dump is not None:
            self.stats["dump_basis"] += 1
            # The reason carries the closed-set basis, never the free-text
            # consent_source: exclusion accounting keys on substrings and an
            # admit reason must never read as "declined" (318 counter-lie).
            return True, f"dump: {dump['consent_basis']} (ADR-076)"
        # PIPELINED: consent is being resolved concurrently (on the consent runner box). A cache-hit
        # is applied (ok->admit / declined->exclude / unreachable-clean->retry);
        # a cache-MISS is DEFERRED (queued, re-checked after consent completes).
        if self.mode == "pipelined":
            rec = self.host_cache.get(host)
            if rec is None:
                if self.post_census:
                    # census COMPLETE and this host is absent from it: not an
                    # opt-out, an enumeration gap. RETAIN + grace + retry
                    # (task 319 fix 1; the 318 build excluded 929,826 points
                    # of exactly this class).
                    self.retry_hosts.add(host)
                    self.stats["uncensused_grace"] += 1
                    return True, "grace:uncensused"
                self.stats["deferred"] += 1
                return False, "defer:uncached"
            return self._apply_live(rec, host)
        # 1) corpus verdict
        if corpus_row is not None:
            so = corpus_row.get("consent_search_ok")
            ca = corpus_row.get("consent_checked_at")
            if so is False:
                self.stats["declined"] += 1
                return False, "corpus: search_ok=False (declined)"
            if so is True and self._fresh(ca):
                self.stats["corpus_fresh"] += 1
                return True, "corpus: search_ok fresh"
            if so is True:
                self.stats["corpus_stale"] += 1
                # stale positive verdict: re-resolve (live/cache), else trust
                if self.mode in ("live", "cache"):
                    rec = self._live(host)
                    return self._apply_live(rec, host)
                return True, "corpus: search_ok stale (trust mode)"
        # 2/3) no usable corpus verdict
        if self.mode in ("live", "cache"):
            # 'cache' = the pre-resolved (runner-box) cache is authoritative; a miss
            # is fail-closed (no many-host sweep on the build box — coordinator policy).
            rec = self._live(host)
            return self._apply_live(rec, host)
        self.stats["unresolved_allowed"] += 1
        return True, "no verdict (trust mode: allow-flagged)"

    def _apply_live(self, rec: dict, host: str) -> tuple[bool, str]:
        so = rec.get("search_ok")
        status = rec.get("status")
        reason = rec.get("reason")   # runner reason-tag: declined|unreachable|error
        # DECLINED: robots actually fetched (status ok) but disallows us -> a
        # PERMANENT exclude (respected; naturally re-checked at each future build).
        if so is False:
            self.stats["declined"] += 1
            return False, "declined: search_ok=False"
        # ADMIT only on a REAL robots fetch (status ok) + search_ok True.
        # resolve_host leaves the DEFAULT search_ok=True on an UNREACHABLE host —
        # trusting that would fail-OPEN, so we require status=='ok' (architecture.md
        # §5 "unreachable -> fail closed").
        if status == "ok" and so is True:
            self.stats["live_ok"] += 1
            return True, "ok: search_ok=True (robots fetched)"
        # NON-ok (unreachable / resolver error / held): NO resolved verdict
        # exists. Task 319 fix 2 — RETAIN under the last-known verdict with a
        # grace stamp + retry queue; never evict on absence-of-data (the 318
        # build excluded ~334k points of this class: a Swiss federal statistics host 160k on a
        # tdmrep ConnectError, a German federal law portal 106k, a French statistics office, a Swiss migration office — all
        # serving in prod, all "held, retryable" per the resolver itself).
        # Eviction for this class happens only at grace EXPIRY, applied by the
        # weekly retry job, not by the build (architecture.md §5 grace rule).
        self.retry_hosts.add(host)
        if reason == "unreachable" and rec.get("canary_ok"):
            self.stats["unreachable_grace"] += 1
            return True, "grace:unreachable(canary-clean)"
        self.stats["unresolved_grace"] += 1
        return True, f"grace:unresolved(status={status} reason={reason})"

    def save(self):
        # In pipelined mode the cache file is owned by the consent runners (rsync);
        # never overwrite it from here (would race the rsync). Only persist the
        # retry queue.
        if self.cache_path and self.mode != "pipelined":
            try:
                json.dump(self.host_cache, open(self.cache_path, "w"))
            except Exception:  # noqa: BLE001
                pass
        # persist the unreachable-retry queue (for the weekly delta re-check)
        if self.retry_path and self.retry_hosts:
            try:
                with open(self.retry_path, "w") as fh:
                    for h in sorted(self.retry_hosts):
                        fh.write(h + "\n")
            except Exception:  # noqa: BLE001
                pass




def _host_of(payload: dict) -> str:
    h = payload.get("host") or payload.get("domain") or ""
    if not h:
        u = payload.get("url") or ""
        try:
            from urllib.parse import urlsplit
            h = urlsplit(u).netloc
        except Exception:  # noqa: BLE001
            h = ""
    return (h or "").lower()


def _is_gov_host(host: str) -> bool:
    """The undated government domain or any subdomain (task 321 ch-media undated-gov ruling)."""
    return host == UNDATED_GOV_HOST or host.endswith("." + UNDATED_GOV_HOST)

