#!/usr/bin/env python3
# Moved from pipeline/gate_authority.py (task 677, task 643 S5); the old alias shim was retired (task 1311, D22).
"""AW_AUTHORITY gate (ADR-019 product-1 / prd-link-graph.md ship amendment).

ADR-054 persisted-script discipline. Runs the IN-PROCESS scorer
(cytir/serve/retrieval.search) against the build-box TWIN aw_v4_sat
(READ-ONLY — never prod, never a full-scan) with AW_AUTHORITY ON vs OFF over
every gold class + the Slice-B probe set, and asserts:

  (1) ADR-043 per-class ZERO-REGRESSION on EVERY gold: per class h@1, h@10, MRR
      must NOT drop, and NO per-gold rank regression (a gold whose rank got
      strictly worse / dropped out). The strict bar for a ranking/serving change.
  (2) a measured AUTHORITY-RANKING LIFT: among golds whose authoritative host is a
      high-authority table entry, ON must IMPROVE rank-1 / rank vs OFF on net —
      the point of the feature. (Zero-regression + a strictly-positive lift =>
      pure upside.)
      **SCOPE (task 394 / F5, 2026-08-07): criterion (2) is enforced ONLY when the
      two arms actually differ on some `AW_AUTHORITY*` variable.** It measures a
      lever; when the lever did not move it cannot be satisfied by construction.
      The composite runner is also used for CONTENT/corpus A/Bs (identical flags,
      corpus is the only variable) — task 393 shipped 8,964 cards with zero
      regressions across 1,404 queries and still exited 1 on `authority-lift=FAIL`,
      forcing a founder ruling for what ADR-043's content bar already answered.
      Identical arms now report `authority-lift=N/A` and the criterion is excluded
      from the verdict and the exit code. When the arms DO differ the check is
      unchanged and unwaivable — it is the flood insurance for the authority layer.

COEXISTENCE with the just-shipped AW_HUBFLOOR + trust_tier is the experiment: BOTH
legs run the LIVE prod flag set INCLUDING hubfloor at its deployed tuning
(AW_HUBFLOOR=1 W=-1.5 MINBETA=0.5 DISTINCT_MAX=0.10 DENSE_MAX=0.88); only
AW_AUTHORITY* differs. authority DEFERS to hubfloor/country per candidate
(retrieval.py), so the two never double-count — this gate proves it on the golds.

Query vectors are embedded ONCE into a shared cache (--build-cache, GPU) and both
legs read it, so the two runs use IDENTICAL vectors (the A/B delta is the lever,
not embedding noise) and scoring needs no GPU.

  # 0. build the shared vec cache once (real arctic embedder, GPU):
  python3 -m cytir.gates.authority --build-cache --cache /tmp/auth_vecs.json
  # 1/2. score both legs (same baseline env, only AW_AUTHORITY* differs):
  AW_AUTHORITY=0  <baseline env>  python3 -m cytir.gates.authority --cache ... --out /tmp/auth_off.json
  AW_AUTHORITY=1 AW_AUTHORITY_W=0.5 ... <baseline env> python3 -m cytir.gates.authority --cache ... --out /tmp/auth_on.json
  # 3. compare:
  python3 -m cytir.gates.authority --compare /tmp/auth_off.json /tmp/auth_on.json

Baseline env (set the SAME for both legs — the live prod flags + shipped hubfloor):
  AW_COLLECTION=aw_v4_sat AW_QDRANT_URL=http://localhost:6333 AW_TEI_URL=
  AW_EMBEDDER=arctic AW_RERANK=0 AW_TWO_STAGE=1 AW_BM25_PER_TOKEN=1 AW_AUTHLEG=1
  AW_KSTAB=1 AW_COMMERCIAL=1 AW_DEDUP_V2=1 AW_RANKHEAD=1 AW_QHEAD=1
  AW_QHEAD_KNOBS=country_scope,off_country_mag AW_OFF_COUNTRY_W=-2.0
  AW_OFF_COUNTRY_NONBOUND_W=0.0 AW_OFF_COUNTRY_NONBOUND_CLASSES=ref
  AW_HUBFLOOR=1 AW_HUBFLOOR_W=-1.5 AW_HUBFLOOR_MINBETA=0.5
  AW_HUBFLOOR_DISTINCT_MAX=0.10 AW_HUBFLOOR_DENSE_MAX=0.88
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.parse

import yaml

APP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cytir.core.serve_layout import SERVER_REL  # noqa: E402
DATA = os.path.join(APP, "data")

# ---------------------------------------------------------------------------- #
# GATE-SET VERSION — the identity of the scored yardstick.
#
# verification-doctrine 2 ("the yardstick is immutable"): metrics are comparable
# only because the scored set never moves. It has moved exactly twice, and each
# move is a NEW VERSION scored alongside the old, never an edit of a frozen one:
#
#   v1  (task 318, 2026-07-25) — 663 core golds = authoritative 478 + foundation
#       172 + legal 6 + names 7, plus the report-only classes and slice_b.
#       Every gate from 318 through 393 ran v1. Those results stay comparable to
#       each other and to their v1 baselines; nothing below rewrites them.
#   v2  (task 394, 2026-08-07) — v1 + `entity_card_golds` (16 adjudicated
#       entity-card golds, `data/eval_entity_card_golds.yaml`). Superset-only:
#       no existing set, row or scoring rule changed, so v1 classes remain
#       readable class-by-class inside a v2 run. Motivation (393 F4): the
#       `entity_cards` class carries gold=0, so a card pack could prove no-harm
#       but structurally could not show benefit — the instrument had no way to
#       measure the thing the corpus lane was shipping.
#   v3  (task 531, 2026-09-06) — same rows, same scoring rule: 9
#       `data/eval_authoritative.yaml` rows (sw-spec-001..009, the
#       `authoritative` class) had one of their three `gold_urls` corrected
#       from `rfc-editor.org/info/rfcN/` to `/rfc/rfcN.txt` (task 527 §8.1 —
#       the corpus and the live site use the `/rfc/*.txt` form; `/info/rfcN/`
#       is a dead alias left over from task 181's 2026-07-12 site-revamp
#       note). Each of these 9 rows also carries a `datatracker.ietf.org`
#       fallback gold_url that already resolves, so this fix is not expected
#       to move the class's hit-rate on its own — it removes a permanently-
#       dead alternative, it does not add a first one. No row added/removed,
#       no other field on these 9 rows touched. See
#       `data/CHANGELOG-eval-gold.md`.
#
#   v4  (task 602, 2026-09-17, ADR-107 amendment 3 — Tim's ruling "remove") —
#       the FIRST version that REMOVES rows: 24 `data/eval_authoritative.yaml`
#       freshness rows (the `authoritative` class, 478 -> 454) whose answers had
#       been written into the corpus. Task 106 authored one
#       `data/refresh_manifest.json` passage per failing freshness query, named
#       each by `eval_id`, and recorded at the time that real fetched HTML
#       scored WORSE than the authored text (freshness dom@10 1.00 -> 0.97).
#       That is teaching-to-the-test, so the queries are a mirror and not a
#       benchmark (verification-doctrine 2). The 24 ids live in
#       `data/flip-gate-contract.v1.json` -> instruments.1418.contamination,
#       derived from task 586's `eval-contamination.v1.json`; see
#       task 602.
#       Scored population: 1,420 -> 1,396 rows, 1,089 -> 1,065 gold-bearing.
#       The pinned query-vector cache (sha256 bd5e59e3…, 1,418 vectors) is
#       UNCHANGED and stays pinned: it is a query->vector lookup table, the
#       scorer iterates ROWS and not cache keys, so a removed row's vector is
#       inert and every surviving query keeps the byte-identical vector it was
#       scored with under v3. What moved is the row set, which the cache digest
#       cannot see — that is why the flip-gate contract now pins this version
#       too (`instruments.1418.gate_set_version`), and why no v3 arm may be
#       compared with a v4 one.
#
# The version is STAMPED INTO EVERY LEG JSON and `--compare` REFUSES to compare
# legs from different versions (a cross-version delta is a corrupt measurement,
# not a regression). Legs written before v2 carry no stamp; two such legs still
# compare, with a loud pre-v2 warning — that is what keeps the 393/365/318
# artifacts replayable.
#
# BUMP PROTOCOL: adding/removing a SETS entry, or editing any scored yaml, is a
# version bump here + a row in the table above + the first run under the new
# version captures a FRESH BASELINE (it may not be diffed against a v1 baseline).
GATE_SET_VERSION = "v4"

# ---------------------------------------------------------------------------
# TWIN FIDELITY (task 399, 2026-08-07) — the `before` leg must prove the replica
# BEHAVES like the baseline, not merely that it COUNTS like it.
#
# MOTIVATING INCIDENT. The task-399 urban pack failed this gate, so it was not
# promoted and its 8,416 points were deleted from the twin by exact id. The
# twin's `points_count` returned to 27,959,338 and every count-based check
# passed — but the twin then reproduced the gate's AFTER leg on 16/16
# entity-card golds and BEFORE on only 15/16. Qdrant SOFT-deletes: deleted
# vectors keep contributing to BM25 document frequencies until their segment is
# VACUUMED, and `deleted_threshold: 0.2` only vacuums a segment once 20% of it
# is deleted. That delete was 0.03%, so it would have vacuumed NEVER. The
# contamination was permanent, invisible to a count assertion, and would have
# silently poisoned the baseline of every subsequent gate.
#
# THE CHECK. Capture a digest of a fixed set of sentinel queries' result URLs
# while the twin is known-clean; re-verify it at the start of every later
# `before` leg. A count match plus a digest match is evidence the replica is
# faithful; a count match alone is not.
#
# Sentinels are drawn deterministically from the SCORED sets (not a bespoke
# list) so they move with the gate version rather than rotting separately.
FIDELITY_SENTINEL_N = 24

# SETTLE CONTROL (task 399, 2026-08-07) — both legs must score a CONSOLIDATED
# index, not one mid-reorganisation.
#
# MOTIVATING EVIDENCE. The urban pack's first gate reported 2 per-gold rank
# regressions and FAILED. Re-run on a consolidated replica it reported ZERO and
# PASSED, deterministically. The first run's BEFORE leg scored a long-settled
# index; applying 8,416 points triggered segment optimisation, so its AFTER leg
# scored a REBUILT HNSW graph — and approximate-NN results shift on rebuild
# independently of content. Corroboration: vacuuming the twin moved the
# `authoritative` class from h@1=177 to h@1=175 with ZERO content change, the
# same magnitude as the "regression" it had produced.
#
# So ANN re-organisation jitter (~2 golds at this corpus size) is
# indistinguishable from a real content regression unless the index is stable on
# both sides of the measurement. `--settle-check` samples the sentinel digest
# repeatedly and only returns once consecutive samples agree: the index has
# stopped moving. Green status alone is not sufficient — green means "no
# pending optimizations", which is true again the instant one finishes.
SETTLE_STABLE_SAMPLES = 2
SETTLE_INTERVAL_SEC = 20
SETTLE_MAX_SEC = 3600
# ---------------------------------------------------------------------------

SETS = [
    ("authoritative", f"{DATA}/eval_authoritative.yaml"),
    # 2026-07-25 (task 318, aw_b1 acceptance gate): + foundation so the runner
    # covers the FULL 663-gold composition (auth 478 + foundation 172 + legal 6
    # + names 7 — the set the 289-era gates scored), and + rel_vs_trust as the
    # committed trust-flood tripwire set. Superset-only change: existing sets
    # and scoring untouched, so prior AW_AUTHORITY gate runs stay comparable
    # class-by-class.
    ("foundation", f"{DATA}/eval_foundation.yaml"),
    ("rel_vs_trust", f"{DATA}/eval_relevance_vs_trust.yaml"),
    ("names", f"{DATA}/eval_names.yaml"),
    ("legal", f"{DATA}/eval_legal_authority.yaml"),
    ("jurisdiction", f"{DATA}/eval_jurisdiction.yaml"),
    ("commercial", f"{DATA}/eval_commercial.yaml"),
    ("general_flood", f"{DATA}/eval_general.yaml"),
    ("entity_cards", f"{DATA}/eval_entity_cards.yaml"),
    # 2026-08-07 (task 394, gate-set v2): the entity-card GOLD set. `entity_cards`
    # above is the task-163 RETURN-BAND set — it scores flood/cap behaviour via
    # `official_domains` and carries no `gold_urls`, hence gold=0 and no lift is
    # measurable on it. This set is url-golded, so card packs can finally register
    # benefit. Separate file, superset-only; the 163 band set is NOT touched.
    ("entity_card_golds", f"{DATA}/eval_entity_card_golds.yaml"),
    ("zero_signal", f"{DATA}/eval_zero_signal.yaml"),
    ("freshness", f"{DATA}/eval_freshness_heldout.yaml"),
    ("vocab_robustness", f"{DATA}/eval_vocab_robustness.yaml"),
    ("dom_civic", f"{DATA}/eval/eval-civic.yaml"),
    ("dom_coding", f"{DATA}/eval/eval-coding.yaml"),
    ("dom_entity", f"{DATA}/eval/eval-entity.yaml"),
    ("dom_legal", f"{DATA}/eval/eval-legal.yaml"),
    ("dom_reference", f"{DATA}/eval/eval-reference.yaml"),
    ("dom_research", f"{DATA}/eval/eval-research.yaml"),
]
SLICEB = f"{APP}/probes/runs/2026-07-19/sample_sliceb.jsonl"

# Authority-lift is measured on the golds whose authoritative host is a strongly-
# endorsed table entry (authority >= this) — those are the golds the feature is
# meant to lift. Rank improvements elsewhere still count as long as no class
# regresses (bar 1).
LIFT_MIN_AUTH = 0.5


def norm_url(u):
    u = (u or "").strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = urllib.parse.unquote(u).replace("_", " ")
    return u.rstrip("/")


def rank_regressed(before, after):
    """ADR-043's per-gold rule, in ONE place: a gold REGRESSED when it was found (`before` is a
    rank) and is now strictly worse or gone (`after` is a larger rank or None). A gold that was not
    found before cannot regress, and one that improves or stays put does not. `compare` below and
    `aw gate` (cytir/gates/slice_gate.py) both call this."""
    return before is not None and (after is None or after > before)


def reg_domain(u):
    u = (u or "").strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    return u.split("/")[0].split(":")[0].rstrip(".")


def _walk_rows(obj):
    if isinstance(obj, dict):
        if obj.get("query") or obj.get("q"):
            yield obj
        for v in obj.values():
            yield from _walk_rows(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_rows(v)


def load_rows(path):
    rows = []
    for doc in (d for d in yaml.safe_load_all(open(path)) if d):
        rows.extend(_walk_rows(doc))
    seen, out = set(), []
    for r in rows:
        q = r.get("query") or r.get("q")
        key = (q, r.get("country"))
        if q and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def load_sliceb(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def gold_of(r):
    gold_urls = ([r["target_url"]] if r.get("target_url") else []) \
        + list(r.get("gold_urls") or [])
    gold_domains = [reg_domain(g) for g in (r.get("gold_domains") or [])]
    if gold_urls and r.get("intent") != "freshness":
        gold_domains = []
    return gold_urls, gold_domains


def rank_of(urls, gold_urls, gold_domains):
    gu = {norm_url(u) for u in gold_urls}
    gu_page = {u.split("#")[0] for u in gu}
    for i, u in enumerate(urls, 1):
        n = norm_url(u)
        if gu and (n in gu or n.split("#")[0] in gu_page):
            return i
        if gold_domains:
            host = reg_domain(u)
            if any(host == g or host.endswith("." + g) for g in gold_domains):
                return i
    return None


# ---- authority table (for identifying the authority-golds in the lift metric) --- #

def _load_auth_table():
    p = os.path.join(APP, SERVER_REL, "authority_hosts.json")
    try:
        cfg = json.load(open(p))
        out = {}
        for d, v in (cfg.get("hosts") or {}).items():
            d = d.lower().lstrip(".")
            if d.startswith("www."):
                d = d[4:]
            out[d] = float(v.get("authority", 0.0)) if isinstance(v, dict) else float(v)
        return out
    except Exception:
        return {}


_AUTH = _load_auth_table()


def gold_authority(gold_urls, gold_domains):
    """Max authority of the gold host(s) — for the lift-eligibility filter."""
    hosts = [reg_domain(u) for u in gold_urls] + list(gold_domains or [])
    best = 0.0
    for h in hosts:
        a = _AUTH.get(h)
        if a is None:
            parts = h.split(".")
            for i in range(len(parts) - 1):
                a = _AUTH.get(".".join(parts[i:]))
                if a is not None:
                    break
        if a:
            best = max(best, a)
    return best


# ---- what the two arms actually differ on (task 394 / F5 lift scoping) ------- #

AUTHORITY_PREFIX = "AW_AUTHORITY"

# The env keys legs written BEFORE gate-set v2 recorded under `env`. A pre-v2 leg
# has no exhaustive `AW_AUTHORITY*` capture, so the fallback can only see these
# six. That is sound for every leg that exists (no other AW_AUTHORITY* knob was
# ever set) but it is a whitelist, so `authority_arms_differ` says so out loud
# rather than letting a legacy leg silently look like an identical arm.
_LEGACY_AUTHORITY_KEYS = (
    "AW_AUTHORITY", "AW_AUTHORITY_W", "AW_AUTHORITY_MIN", "AW_AUTHORITY_RELGATE",
    "AW_AUTHORITY_MAXBOOST", "AW_AUTHORITY_MIN_CENTRALITY",
)


def capture_authority_env(environ=None):
    """Every `AW_AUTHORITY*` variable actually set, exhaustively.

    Not a whitelist: a knob added to retrieval.py tomorrow is captured without
    touching this file, so a true A/B can never be misread as an identical pair.
    """
    environ = os.environ if environ is None else environ
    return {k: v for k, v in sorted(environ.items()) if k.startswith(AUTHORITY_PREFIX)}


def _authority_env_of(leg):
    """(authority-env dict, source) for one leg json. Unset == absent, so a
    pre-v2 leg (`env` holds None for unset keys) compares equal to a v2 leg."""
    ae = leg.get("authority_env")
    if ae is not None:
        return {k: v for k, v in ae.items() if v is not None}, "authority_env"
    env = leg.get("env") or {}
    legacy = {k: env[k] for k in _LEGACY_AUTHORITY_KEYS
              if env.get(k) is not None}
    return legacy, "legacy-env-whitelist"


def authority_arms_differ(off_leg, on_leg, say=print):
    """True iff the arms differ on any AW_AUTHORITY* variable.

    This is the F5 scope test: criterion (2) is a claim about a lever, so it is
    only enforceable on runs that moved the lever.
    """
    a, src_a = _authority_env_of(off_leg)
    b, src_b = _authority_env_of(on_leg)
    if "legacy" in src_a or "legacy" in src_b:
        say(f"  [warn] pre-v2 leg(s) — AW_AUTHORITY* read from the legacy "
            f"{len(_LEGACY_AUTHORITY_KEYS)}-key whitelist "
            f"(off={src_a}, on={src_b}); a knob outside that whitelist would be "
            f"invisible on such a leg.")
    say(f"  off AW_AUTHORITY*: {a or '{}'}")
    say(f"  on  AW_AUTHORITY*: {b or '{}'}")
    diff = sorted(set(a) | set(b))
    diff = [k for k in diff if a.get(k) != b.get(k)]
    if diff:
        say(f"  arms DIFFER on: {', '.join(diff)}")
    return bool(diff)


# ------- shared vec cache (build once w/ real embedder; both legs read it) ---- #

def all_queries():
    qs = []
    for _n, p in SETS:
        if os.path.exists(p):
            for r in load_rows(p):
                q = r.get("query") or r.get("q")
                if q:
                    qs.append(q)
    if os.path.exists(SLICEB):
        for r in load_sliceb(SLICEB):
            if r.get("query"):
                qs.append(r["query"])
    return sorted(set(qs))


def build_cache(cache_path):
    sys.path.insert(0, APP)
    from cytir.serve import models  # noqa: E402
    import numpy as np
    qs = all_queries()
    print(f"embedding {len(qs)} unique queries (real arctic embedder)...", flush=True)
    vecs = {}
    for i, q in enumerate(qs):
        models.embed_query(q)
        full = models.last_full_query_vec()
        vecs[q] = np.asarray(full, dtype="float32").tolist()
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(qs)}", flush=True)
    json.dump(vecs, open(cache_path, "w"))
    print(f"wrote {cache_path} ({len(vecs)} vecs)", flush=True)


def install_cached_embedder(cache_path):
    sys.path.insert(0, APP)
    from cytir.serve import models  # noqa: E402
    import numpy as np
    VECS = json.load(open(cache_path))
    MRL = models.MRL_DIM
    miss = [0]

    def _emb(q):
        full = VECS.get(q)
        if full is None:
            miss[0] += 1
            raise KeyError(f"no cached vec: {q!r}")
        full = np.asarray(full, dtype="float32")
        models._tls.full_qvec = full
        v = full[:MRL]
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).tolist(), None

    models.embed_query = _emb
    return miss


# ------------------------------- run a leg ---------------------------------- #

def fusion_weights_provenance() -> dict:
    """The weights file THIS arm serves, resolved and hashed by the arm itself.

    Task 544: task 519's `seal_arm.py` used to re-derive this from its
    OWN process environment, and the 1,418 sealer never exported
    AW_FUSION_WEIGHTS -- so every sealed 1,418 arm from tasks 533/537/538
    claims prod's `fusion_weights.json`, including the dense-only and
    sparse-only arms that demonstrably served something else. An arm that does
    not record its own configuration cannot be compared with another one; the
    leg records it here, at the moment it is used, and the sealer prefers it.
    """
    path = os.environ.get(
        "AW_FUSION_WEIGHTS",
        os.path.join(APP, SERVER_REL, "fusion_weights.json"))
    out = {"path": os.path.abspath(path), "basis": "arm process env"}
    try:
        with open(path, "rb") as fh:
            out["sha256"] = hashlib.sha256(fh.read()).hexdigest()
    except OSError as exc:                     # named, never silently absent
        out["sha256"] = None
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def run(out_path, cache_path):
    miss = install_cached_embedder(cache_path)
    sys.path.insert(0, APP)
    from cytir.serve import retrieval  # noqa: E402
    print(f"AW_AUTHORITY={os.environ.get('AW_AUTHORITY','0')} "
          f"W={os.environ.get('AW_AUTHORITY_W','-')} "
          f"MIN={os.environ.get('AW_AUTHORITY_MIN','-')} "
          f"MAXBOOST={os.environ.get('AW_AUTHORITY_MAXBOOST','-')} "
          f"RELGATE={os.environ.get('AW_AUTHORITY_RELGATE','-')} "
          f"MINCENT={os.environ.get('AW_AUTHORITY_MIN_CENTRALITY','-')} "
          f"| auth_table={len(retrieval._AUTH_HOSTS)} hosts "
          f"hub_table={len(retrieval._HUB_HOSTS)} "
          f"HUBFLOOR={os.environ.get('AW_HUBFLOOR','0')} "
          f"COLLECTION={os.environ.get('AW_COLLECTION','?')}", flush=True)

    result = {"label": os.environ.get("AW_AUTHORITY", "0"),
              "gate_set_version": GATE_SET_VERSION,
              "fusion_weights": fusion_weights_provenance(),
              # task 578 — what the PROCESS resolved, not what the launcher
              # intended and not what an env whitelist remembered to list.
              # The 629 arms get this for free off `/health`; the 1,418 leg
              # scores in-process and had no equivalent, so every 1,418 arm's
              # lever state rested on a whitelist somebody had to extend.
              # `effective_config()` is the server's own report and grows with
              # the server, so a knob added tomorrow is captured without
              # touching this file.
              "served_config": retrieval.effective_config(),
              # exhaustive AW_AUTHORITY* capture — the F5 scope test reads this,
              # `env` below stays as-is so old readers keep working
              "authority_env": capture_authority_env(),
              "env": {k: os.environ.get(k) for k in (
                  "AW_AUTHORITY", "AW_AUTHORITY_W", "AW_AUTHORITY_MIN",
                  "AW_AUTHORITY_RELGATE", "AW_AUTHORITY_MAXBOOST",
                  "AW_AUTHORITY_MIN_CENTRALITY",
                  "AW_HUBFLOOR", "AW_HUBFLOOR_W", "AW_RANKHEAD", "AW_QHEAD",
                  "AW_COMMERCIAL", "AW_COLLECTION",
                  # task 544 -- the fused arm's weights are a lever like any
                  # other, and were the one lever no 1,418 artifact recorded
                  "AW_FUSION_WEIGHTS",
                  # task 404 levers — recorded so an arm's lever state is
                  # readable off the leg artifact, never off memory
                  "AW_NAMEGATE", "AW_NAMEGATE_W", "AW_NAMEGATE_COV",
                  "AW_REFCANON", "AW_REFCANON_W", "AW_REFCANON_BETA",
                  "AW_REFCANON_MAXBETA", "AW_REFCANON_EXTRA_CLASSES",
                  # task 544 -- REF-CANON's blanket `reference` demotion is
                  # what separates prod from the aw_e8 candidate on the bare
                  # name queries, and MINCENT gates its trigger; an arm that
                  # does not echo them cannot be read
                  "AW_REFCANON_MINCENT",
                  # task 406 person-name legs — same rule: an arm's lever
                  # state is read off the leg artifact, never off memory
                  "AW_NAMEDENSE", "AW_NAMEDENSE_SCALE", "AW_NAMECOV_W",
                  "AW_NAMEGATE_TRUSTKEEP", "AW_TRUSTFLOOR_UNDEMOTED",
                  "AW_NAMEGATE_SHAPE_V2", "AW_TRUSTFLOOR",
                  "AW_TRUSTFLOOR_REL_BETA",
                  # task 578 authority-signal levers
                  "AW_REFCANON_TARGETS", "AW_PRIMACY", "AW_PRIMACY_KEEP",
                  "AW_PRIMACY_BETA", "AW_PRIMACY_MINCENT",
                  "AW_PRIMACY_EXTRA_CLASSES",
                  "AW_RANKHEAD_V2", "AW_RANKHEAD_V2_PATH")},
              "sets": {}}

    all_sets = [(n, load_rows(p)) for n, p in SETS if os.path.exists(p)]
    if os.path.exists(SLICEB):
        all_sets.append(("slice_b", load_sliceb(SLICEB)))

    for name, rows in all_sets:
        per_q = {}
        h1 = h10 = ngold = 0
        mrr_sum = 0.0
        for j, r in enumerate(rows):
            q = r.get("query") or r.get("q")
            country = r.get("country")
            lang = r.get("lang")
            try:
                hits = retrieval.search(q, k=10, lang=lang, country=country)
            except Exception as e:
                per_q[str(r.get("id") or r.get("qid") or f"{name}-{j}")] = {
                    "urls": [f"__ERROR__:{type(e).__name__}"], "rank": None}
                continue
            urls = [h.get("url") or "" for h in hits]
            gold_urls, gold_domains = gold_of(r)
            rk = rank_of(urls, gold_urls, gold_domains) if (gold_urls or gold_domains) else None
            qid = str(r.get("id") or r.get("qid") or f"{name}-{j}")
            ga = gold_authority(gold_urls, gold_domains) if (gold_urls or gold_domains) else 0.0
            per_q[qid] = {"urls": urls, "rank": rk, "gold_auth": round(ga, 3)}
            if gold_urls or gold_domains:
                ngold += 1
                if rk == 1:
                    h1 += 1
                if rk and rk <= 10:
                    h10 += 1
                if rk:
                    mrr_sum += 1.0 / rk
        result["sets"][name] = {"n": len(rows), "n_gold": ngold, "h1": h1,
                                "h10": h10, "mrr": round(mrr_sum / ngold, 4) if ngold else None,
                                "per_q": per_q}
        print(f"  {name:18s} n={len(rows):>4} gold={ngold:>4} h@1={h1:>4} h@10={h10:>4} "
              f"mrr={result['sets'][name]['mrr']}", flush=True)
    result["cache_miss"] = miss[0]
    json.dump(result, open(out_path, "w"))
    print(f"wrote {out_path}  cache_miss={miss[0]}", flush=True)


def compare(off_path, on_path):
    A = json.load(open(off_path))   # AW_AUTHORITY=0
    B = json.load(open(on_path))    # AW_AUTHORITY=1
    print("\n=== AW_AUTHORITY GATE  off -> on ===")
    print(f"off env: {A.get('env')}")
    print(f"on  env: {B.get('env')}")

    # --- gate-set identity: never diff two legs scored on different yardsticks -
    va, vb = A.get("gate_set_version"), B.get("gate_set_version")
    if va != vb:
        print(f"\nERROR: gate-set version mismatch — off={va!r} on={vb!r}. The two "
              f"legs did not score the same gold set; their deltas are not a "
              f"regression signal. Re-run both legs on one version.")
        return 3
    if va is None:
        print("\n[warn] legs carry no gate_set_version — pre-v2 artifacts "
              "(<= task 393). Comparing them to each other is fine; do NOT "
              "compare either one against a v2 leg.")
    else:
        print(f"gate-set version: {va}")

    # --- F5 scope test: is criterion (2) even applicable to this run? ----------
    print("\n--- arm scope (AW_AUTHORITY* lever) ---")
    lift_applicable = authority_arms_differ(A, B)
    print("  => authority-lift is "
          + ("ENFORCED (true AW_AUTHORITY A/B)" if lift_applicable else
             "N/A — arms identical on AW_AUTHORITY (corpus/content-only change; "
             "the lift criterion measures a lever that did not move)"))

    print(f"\n{'class':18s} {'n':>4} {'gold':>4} {'h@1 Δ':>8} {'h@10 Δ':>8} "
          f"{'MRR Δ':>9} {'reordered':>10}  verdict")
    all_ok = True
    rank_regressions = []
    lift_improve = []          # authority-golds whose rank improved
    lift_rank1 = []            # authority-golds that gained rank-1
    lift_pool = 0
    total_qs = total_diff = 0
    for name in A["sets"]:
        a, b = A["sets"][name], B["sets"].get(name, {})
        d1 = b.get("h1", 0) - a.get("h1", 0)
        d10 = b.get("h10", 0) - a.get("h10", 0)
        dmrr = round((b.get("mrr") or 0.0) - (a.get("mrr") or 0.0), 4)
        diffs = 0
        for qid, av in a["per_q"].items():
            bv = b["per_q"].get(qid, {})
            if av.get("urls") != bv.get("urls"):
                diffs += 1
            ra, rb = av.get("rank"), bv.get("rank")
            if rank_regressed(ra, rb):
                rank_regressions.append((name, qid, ra, rb))
            # lift bookkeeping — authority-golds only
            if (av.get("gold_auth") or 0.0) >= LIFT_MIN_AUTH and ra is not None:
                lift_pool += 1
                if rb is not None and rb < ra:
                    lift_improve.append((name, qid, ra, rb))
                if ra != 1 and rb == 1:
                    lift_rank1.append((name, qid, ra, rb))
        total_qs += len(a["per_q"])
        total_diff += diffs
        ok = (d1 >= 0 and d10 >= 0 and dmrr >= -0.0005)
        all_ok = all_ok and ok
        print(f"{name:18s} {a['n']:>4} {a.get('n_gold',0):>4} {d1:>+8} {d10:>+8} "
              f"{dmrr:>+9} {diffs:>10}  {'OK' if ok else '*** REGRESSED ***'}")
    print(f"\nTOTAL queries={total_qs} reordered={total_diff} "
          f"per-gold rank-regressions={len(rank_regressions)}")
    for name, qid, ra, rb in rank_regressions[:40]:
        print(f"    RANK-REG {name}/{qid}: {ra} -> {rb}")
    if len(rank_regressions) > 40:
        print(f"    ... +{len(rank_regressions) - 40} more")

    print("\n=== AUTHORITY-RANKING LIFT (golds whose authoritative host has "
          f"authority >= {LIFT_MIN_AUTH})"
          + ("" if lift_applicable else " — REPORT-ONLY, not scored") + " ===")
    print(f"  authority-gold pool: {lift_pool}")
    print(f"  rank improved:  {len(lift_improve)}")
    print(f"  gained rank-1:  {len(lift_rank1)}")
    for name, qid, ra, rb in lift_rank1[:30]:
        print(f"    RANK-1 GAIN {name}/{qid}: {ra} -> 1")
    for name, qid, ra, rb in lift_improve[:30]:
        if (name, qid, ra, rb) not in lift_rank1:
            print(f"    IMPROVED   {name}/{qid}: {ra} -> {rb}")

    zero_reg = (all_ok and not rank_regressions)
    lift_win = (len(lift_improve) > 0 or len(lift_rank1) > 0)
    if lift_applicable:
        lift_verdict = "PASS" if lift_win else "FAIL (no measurable lift)"
        ok = zero_reg and lift_win
    else:
        lift_verdict = "N/A (arms identical on AW_AUTHORITY)"
        ok = zero_reg
    print(f"\nVERDICT: per-class zero-regression={'PASS' if zero_reg else 'FAIL'}; "
          f"authority-lift={lift_verdict}")
    return 0 if ok else 1


def fidelity_sentinels():
    """A deterministic, gate-version-scoped slice of the scored golds.

    Deterministic so the digest is reproducible; drawn from SETS so it tracks
    the gate version instead of becoming a separate list that rots.
    """
    picked = []
    for name, path in SETS:
        if not os.path.exists(path):
            continue
        rows = load_rows(path)
        for j, r in enumerate(rows):
            if j % 37 == 0 and (r.get("query") or r.get("q")):
                picked.append((name, str(r.get("id") or r.get("qid") or f"{name}-{j}"), r))
    picked.sort(key=lambda t: (t[0], t[1]))
    return picked[:FIDELITY_SENTINEL_N]


def fidelity_probe(cache_path):
    """{sentinel_id: [urls]} plus a stable digest, scored exactly like a leg."""
    install_cached_embedder(cache_path)
    sys.path.insert(0, APP)
    from cytir.serve import retrieval  # noqa: E402
    per_q = {}
    for name, qid, r in fidelity_sentinels():
        q = r.get("query") or r.get("q")
        try:
            hits = retrieval.search(q, k=10, lang=r.get("lang"),
                                    country=r.get("country"))
            per_q[qid] = [h.get("url") or "" for h in hits]
        except Exception as e:  # noqa: BLE001
            per_q[qid] = [f"__ERROR__:{type(e).__name__}"]
    # Digest over per-query URL SETS, not ordered lists.
    #
    # WHY (measured 2026-08-07): with the ordered list the digest OSCILLATED
    # between two values on a green, stable-count twin. Diffing the two states
    # showed the cause exactly — ONE sentinel (`sw-api-002`) returning the same
    # ten URLs in a flipped order, i.e. a benign score tie. Zero set
    # differences. An order-sensitive digest therefore produces false
    # contamination aborts and would block every gate.
    #
    # Set comparison keeps the check pointed at what fidelity is actually for —
    # "is the replica retrieving the same CONTENT?" — which is precisely what
    # the soft-delete incident changed (triskel.ch LEFT its result set and five
    # other URLs entered). Ranking is the GATE's instrument, not this one's;
    # per-gold rank regressions are scored there.
    sets = {q: sorted(set(urls)) for q, urls in per_q.items()}
    blob = json.dumps(sets, sort_keys=True, ensure_ascii=False)
    # task 406: COUNT the swallowed exceptions. The `except` above turns every
    # search failure into an `__ERROR__:<Type>` string — and those strings are
    # perfectly STABLE, so a dead or starved channel produces a rock-steady
    # digest and `settle_check` below would report SETTLED on an index that is
    # returning nothing at all. That is doctrine 6 ("an error is not an empty")
    # inside the gate's own pre-flight, and it bit for real on 2026-08-13: with
    # the scratch node's dense leg timing out at 60 s on every query, the
    # settle-check ground on silently and would have green-lit the legs.
    # Reported here so callers can refuse; `settle_check` refuses.
    n_err = sum(1 for urls in per_q.values()
                if urls and str(urls[0]).startswith("__ERROR__"))
    return {"gate_set_version": GATE_SET_VERSION,
            "n_sentinels": len(per_q),
            "n_channel_errors": n_err,
            "collection": os.environ.get("AW_COLLECTION"),
            "digest": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
            "digest_basis": "per-query url SET (order-insensitive)",
            "per_q": per_q,
            "per_q_sets": sets}


def fidelity_verify(cache_path, baseline_path):
    """Compare the live replica against a captured baseline. Exit 3 on drift."""
    base = json.load(open(baseline_path))
    now = fidelity_probe(cache_path)
    if base.get("gate_set_version") != now["gate_set_version"]:
        print(f"FIDELITY ABORT: baseline is {base.get('gate_set_version')}, "
              f"runner is {now['gate_set_version']} — recapture required")
        return 3
    if base["digest"] == now["digest"]:
        print(f"FIDELITY OK: {now['n_sentinels']} sentinels, "
              f"digest {now['digest'][:16]}…")
        return 0
    # Derive the set view on either side when absent, so a baseline captured
    # before the set-digest change still compares.
    def _sets(cap):
        s = cap.get("per_q_sets")
        if s:
            return s
        return {q: sorted(set(u)) for q, u in cap.get("per_q", {}).items()}

    base_sets, now_sets = _sets(base), _sets(now)
    drifted = [q for q, urls in now_sets.items() if base_sets.get(q) != urls]
    print("FIDELITY ABORT: the replica COUNTS like the baseline but does not "
          "BEHAVE like it.")
    print(f"  sentinels drifted: {len(drifted)}/{now['n_sentinels']} -> "
          f"{', '.join(drifted[:8])}")
    print("  Most likely cause (task 399): soft-deleted points still "
          "contributing to BM25 document frequencies because their segment was "
          "never vacuumed (deleted_threshold). Vacuum and re-verify; do NOT "
          "score a leg against this replica.")
    return 3


def settle_check(cache_path, interval=SETTLE_INTERVAL_SEC,
                 stable=SETTLE_STABLE_SAMPLES, max_sec=SETTLE_MAX_SEC):
    """Block until the replica's sentinel digest stops changing.

    Returns 0 when `stable` consecutive samples agree, 4 on timeout. This is
    the settle control: it does not care WHAT the digest is (after an apply it
    legitimately differs from the baseline) — only that the index has stopped
    moving, so the leg about to be scored measures content and not an
    in-progress reorganisation.
    """
    import time as _t
    t0 = _t.time()
    seen = []
    while _t.time() - t0 < max_sec:
        probe = fidelity_probe(cache_path)
        d = probe["digest"]
        # task 406 — an ERROR IS NOT AN EMPTY, and a stable error is the most
        # dangerous stability there is: `fidelity_probe` swallows search
        # exceptions into `__ERROR__` strings, which never change, so a dead or
        # timing-out channel converges instantly and settles. Refuse before
        # comparing digests. (Measured 2026-08-13: with the scratch node's dense
        # leg at a 60 s server timeout, this loop ran silently for 25+ minutes
        # with every sentinel erroring.)
        n_err = probe.get("n_channel_errors", 0)
        if n_err:
            print(f"SETTLE ABORT: {n_err}/{probe['n_sentinels']} sentinels "
                  f"returned __ERROR__ — the search channel is not healthy, so "
                  f"a stable digest here means nothing. Fix the channel before "
                  f"scoring; do NOT treat these as misses.", flush=True)
            for qid, urls in probe["per_q"].items():
                if urls and str(urls[0]).startswith("__ERROR__"):
                    print(f"    {qid}: {urls[0]}", flush=True)
                    break
            return 6
        seen.append(d)
        tail = seen[-stable:]
        if len(tail) == stable and len(set(tail)) == 1:
            print(f"SETTLED: digest {d[:16]}… stable over {stable} samples "
                  f"({int(_t.time() - t0)}s)", flush=True)
            return 0
        # flush: without it a redirected settle-check prints NOTHING for its
        # entire (up to an hour) run, so an operator cannot tell a slow index
        # from a hung one.
        print(f"  not settled yet: digest {d[:16]}… "
              f"(sample {len(seen)}, {int(_t.time() - t0)}s)", flush=True)
        _t.sleep(interval)
    print(f"SETTLE TIMEOUT after {max_sec}s — index still moving; do NOT score "
          f"a leg against it", flush=True)
    return 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/auth_vecs.json")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--capture-fidelity", metavar="PATH",
                    help="write a twin-fidelity baseline (run on a KNOWN-CLEAN "
                         "replica only)")
    ap.add_argument("--verify-fidelity", metavar="PATH",
                    help="verify the replica against a fidelity baseline; "
                         "exit 3 on drift")
    ap.add_argument("--settle-check", action="store_true",
                    help="block until the sentinel digest stops changing "
                         "(settle control); exit 4 on timeout")
    args = ap.parse_args()
    if args.build_cache:
        build_cache(args.cache)
        return 0
    if args.capture_fidelity:
        cap = fidelity_probe(args.cache)
        json.dump(cap, open(args.capture_fidelity, "w"), indent=1)
        print(f"captured {cap['n_sentinels']} sentinels -> "
              f"{args.capture_fidelity}  digest={cap['digest'][:16]}…")
        return 0
    if args.settle_check:
        return settle_check(args.cache)
    if args.verify_fidelity:
        return fidelity_verify(args.cache, args.verify_fidelity)
    if args.compare:
        return compare(*args.compare)
    if not args.out:
        ap.error("--out required (or --build-cache / --compare / "
                 "--capture-fidelity / --verify-fidelity)")
    run(args.out, args.cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
