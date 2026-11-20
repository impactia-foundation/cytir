"""The b1 builder's SLICE mode and its acceptance checks (moved from the builder script, task 2123).

`run_slice` builds the ~150k vertical slice into `aw_b1_slice` and runs the acceptance
checks (a-m); `run_acceptance_checks`, `_dense_contract_sample`, `upsert_points`, `_dfilter`,
`_fold_local` and `SLICE_PLAN` move with it. The code below is the original block, with these
differences: `run_slice` takes the corpus directory as the keyword `corpus_dir` (the builder's
module constant `CORPUS_DIR`; its `main` now resolves it from the `b1_corpus_dir` setting),
`QdrantClient` annotations name the client type through `Any`, and the fold-on environment
default sits before the imports as it did in the builder.

Its write sites (`upsert_points`, the collection create/finalize in `run_slice`) are reached
only through `cytir/index/b1_cli.py`, which calls `refuse_cli_write` before it dispatches."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any

# ---- fold ON before importing the tokenizer path (module-level _FOLD_ACCENTS) --
os.environ.setdefault("AW_FOLD_ACCENTS", "1")

import re as _re_mod            # noqa: E402
import unicodedata as _ud       # noqa: E402
import cytir.core.common as AC          # noqa: E402
from cytir.core import bm25_tokenizer as _tokenize  # noqa: E402
import cytir.core.ids as ID     # noqa: E402
from cytir.consent.build_gate import (  # noqa: E402
    ConsentGate, _is_gov_host)
from cytir.index.b1_projection import (  # noqa: E402
    CHMEDIA_PACK, evaluate_dump_basis_emission,
    evaluate_payload_contract, project_record)
from cytir.index.b1_rebuild import (  # noqa: E402
    BASE_COLL, BUILD_URL, STAGING_URL, TWIN_URL, _reembed_explicit_dense,
    check_deferred_tripwire)
from cytir.index.b1_manifest import (  # noqa: E402
    IGNITION_OVERRIDE_ENV, assert_ignition, create_collection, finalize_collection, ignition_check, qc,
    scroll_collection, validate_manifest, write_manifest)
from cytir.index.b1_corpus_kv import (  # noqa: E402
    _canon_key, build_corpus_text_index, load_published_map,
    load_resolve_conf_map)
from qdrant_client import models as qm   # noqa: E402

# force-enable the fold in-process regardless of import order (mirror contract)
from cytir.core import bm25_tokenizer as _TOK  # noqa: E402
_TOK._FOLD_ACCENTS = True

# self-contained token boundary + fold predicate for the acceptance check — do NOT
# depend on aw_common internals (_fold/_TOKEN_RE), which the build copy on the box
# does not expose as module symbols (its fold is inlined). aw_common.bm25_tokens
# still does the real folded tokenization; this only decides "is this token
# accented" for candidate selection.
_TOKEN_RE_LOCAL = _re_mod.compile(r"\w+", _re_mod.UNICODE)


def _fold_local(s: str) -> str:
    return "".join(c for c in _ud.normalize("NFKD", s) if not _ud.combining(c))

CARD_PACKS = {"aw_dzefix_cov", "aw_dretailcards_cov", "aw_dnewscards_cov"}


def upsert_points(client: Any, name: str, points: list[dict],
                  batch: int, log=print):
    n = 0
    for i in range(0, len(points), batch):
        chunk = points[i:i + batch]
        # Slice assembly is a real production-faithful build leg, not a
        # lightweight projection: explicit dense channels must take the same
        # re-embed path as the streaming full-build flush.
        _reembed_explicit_dense(chunk)
        client.upsert(name, points=[
            qm.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in chunk], wait=True)
        n += len(chunk)
        if n % (batch * 10) == 0 or n == len(points):
            log(f"[upsert] {name}: {n:,}/{len(points):,}")

def _dfilter(domain: str):
    return qm.Filter(must=[qm.FieldCondition(key="domain",
                                             match=qm.MatchValue(value=domain))])


# --------------------------------------------------------------------------
# SLICE mode
# --------------------------------------------------------------------------
SLICE_PLAN = [
    # (label, source, selector, cap, corpus_waves_hint)
    ("fedlex_tierA", "twin", {"domain": "fedlex.admin.ch"}, 30000,
     ["swissauth", "swissauth_health_boost", "phase2_all"]),
    ("wiki_tierB", "twin", {"domain": "de.wikipedia.org"}, 30000,
     ["l1_wiki_de_20260601", "p0-era-wiki_1m", "p0-era-wiki_en"]),
    ("gesetze_genericCC_tierB", "twin", {"domain": "gesetze-im-internet.de"}, 40000,
     ["gesetze_v1", "gesetze_v2", "gesetze_v0", "gesetze_v1_partial_mt"]),
    # anchor-less MULTI-CHUNK pages — the GAP A33 chunk-collapse class (task
    # 319): anthropic.com (414 pts / 13 urls on the twin; /constitution alone
    # is 218 chunks) exercises the snippet-fallback id path; huggingface.co
    # docs/model-cards exercise the ambiguous-KV snippet-match path (corpus
    # waves listed carry their chunk texts, anchor=None throughout).
    ("anthropic_anchorless_multichunk", "twin", {"domain": "anthropic.com"}, 1000,
     ["ai_vendor_cc"]),
    ("hf_anchorless_multichunk", "twin", {"domain": "huggingface.co"}, 3000,
     ["ai_hf_model_cards", "ai_framework_docs"]),
    ("zefix_cards_tierC", "staging", {"coll": "aw_dzefix_cov"}, 40000, None),
    ("swissinfo_news", "staging", {"coll": "aw_dswissinfo_cov"}, 3000, None),
    ("chmedia_news", "staging", {"coll": "aw_dchmedia_cov"}, 2000, None),
    ("retail_cards_tierC", "staging", {"coll": "aw_dretailcards_cov"}, 100, None),
    ("news_cards_tierC", "staging", {"coll": "aw_dnewscards_cov"}, 100, None),
]


def run_slice(args, log=print, *, corpus_dir):
    t0 = time.time()
    twin = qc(TWIN_URL)
    stag = qc(STAGING_URL)
    build = qc(BUILD_URL)

    ID.self_test()

    # 1) select source records (keep in memory for the slice)
    selected: list[tuple] = []  # (label, origin, record)
    corpus_waves: set[str] = set()
    for label, src, sel, cap, waves in SLICE_PLAN:
        if args.max_per_source:
            cap = min(cap, args.max_per_source)
        if src == "twin":
            it = scroll_collection(twin, BASE_COLL, cap, _dfilter(sel["domain"]),
                                   with_vectors=True, log=log)
            origin = BASE_COLL
            if waves:
                corpus_waves.update(waves)
        else:
            it = scroll_collection(stag, sel["coll"], cap, None,
                                   with_vectors=True, log=log)
            origin = sel["coll"]
        cnt = 0
        for rec in it:
            selected.append((label, origin, rec))
            cnt += 1
        log(f"[select] {label}: {cnt:,} records")
    log(f"[select] TOTAL selected {len(selected):,} in {time.time()-t0:.0f}s")

    # 2) build the needed corpus text keys (base records only)
    needed = set()
    for label, origin, rec in selected:
        if origin == BASE_COLL:
            pl = rec.payload or {}
            needed.add(_canon_key(pl.get("url"), pl.get("anchor")))
    corpus_idx = build_corpus_text_index(needed, waves=sorted(corpus_waves),
                                         max_rows=args.corpus_max_rows, log=log,
                                         corpus_dir=corpus_dir)

    # 3) sidecars
    pub_map = load_published_map(args.news_published or [], log=log)
    conf_map = load_resolve_conf_map(args.cards_audit or "", log=log)
    gate = ConsentGate(args.consent_mode, args.consent_grace_days,
                       args.consent_cache, log=log)
    if args.consent_census_complete:
        # slice-mode analogue of the post-CONSENT.DONE phases: cache-miss =
        # census-absent -> grace-retain (exercises task 319 fix 1 end to end)
        gate.post_census = True

    # 4) project + dedup by canon_key (staged/fresher wins by digest)
    stats: defaultdict = defaultdict(int)
    by_key: dict[str, dict] = {}
    src_counts: defaultdict = defaultdict(int)
    lic_exc_by_origin: defaultdict = defaultdict(int)
    tier_counts: defaultdict = defaultdict(int)
    admitted_by_label: defaultdict = defaultdict(int)   # for check (g)
    ids_by_label: dict[str, set] = {}                   # minted ids (pre-dedup)
    excl_reasons: defaultdict = defaultdict(int)
    for label, origin, rec in selected:
        pl = rec.payload or {}
        crow = corpus_idx.get(_canon_key(pl.get("url"), pl.get("anchor"))) if origin == BASE_COLL else None
        point, meta = project_record(rec, origin, crow, pub_map, conf_map, gate, stats)
        if point is None:
            excl_reasons[meta.get("reason", "?")] += 1
            continue
        admitted_by_label[label] += 1
        ids_by_label.setdefault(label, set()).add(point["id"])
        key = point["canon_key"]
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = point
        else:
            # dedup: prefer staged (net-new) over base; else keep existing
            prev_staged = prev["origin"] != BASE_COLL
            cur_staged = origin != BASE_COLL
            if cur_staged and not prev_staged:
                by_key[key] = point
                stats["dedup_staged_win"] += 1
            else:
                stats["dedup_skip"] += 1
                continue
    points = list(by_key.values())
    for p in points:
        src_counts[p["origin"]] += 1
        tier_counts[p["tier"]] += 1
        if p["lic_exception"]:
            lic_exc_by_origin[p["origin"]] += 1
    gate.save()
    log(f"[project] {len(points):,} points after dedup; stats={dict(stats)}")
    log(f"[project] tiers={dict(tier_counts)}")
    log(f"[project] exclusion reasons={dict(excl_reasons)}")
    log(f"[project] consent gate classes={dict(gate.stats)}")

    # 5) tail refetch list (no-text base records)
    tail = [(p["payload"]["url"], p["id"]) for p in points
            if not p["has_text"] and p["tok_epoch"] == 1 and p["origin"] == BASE_COLL]
    tail_path = os.path.join(args.out_dir, "aw_b1_slice.tail_refetch.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(tail_path, "w") as fh:
        for u, pid in tail:
            fh.write(json.dumps({"url": u, "point_id": pid}) + "\n")
    log(f"[tail] {len(tail):,} no-text base points -> {tail_path}")

    # 6) assemble aw_b1_slice
    create_collection(build, args.collection, log=log)
    upsert_points(build, args.collection, points, args.batch, log=log)
    finalize_collection(build, args.collection, log=log)
    final_count = build.count(args.collection, exact=True).count
    log(f"[assemble] {args.collection}: {final_count:,} points")

    # 7) MANIFEST (slice)
    sources = []
    for origin, cnt in sorted(src_counts.items()):
        pack = "ch" if ("zefix" in origin or "chmedia" in origin or "swissinfo" in origin) else "mixed"
        sources.append({
            "pack": pack, "source": origin, "version": "slice-2026-07-23",
            "count": cnt, "policy": "POLICY.md@v1",
            "license_exceptions": lic_exc_by_origin.get(origin, 0),
        })
    counts = {"points": final_count, "adds": final_count, "changes": 0, "deletes": 0}
    man = write_manifest(
        os.path.join(args.out_dir, "aw_b1_slice.MANIFEST.json"),
        "aw_b1", None, sources, counts,
        {"golds": 0, "pass": False, "report": "PENDING (slice — gate not run)"},
        log=log)
    man_errs = validate_manifest(man)

    # 8) acceptance checks
    log("\n================ SLICE ACCEPTANCE CHECKS ================")
    results = run_acceptance_checks(build, twin, args.collection, points,
                                    tier_counts, man, man_errs, stats, log=log,
                                    gate=gate, admitted_by_label=admitted_by_label,
                                    ids_by_label=ids_by_label,
                                    excl_reasons=excl_reasons)

    # 9) throughput -> full-run estimate
    dt = time.time() - t0
    thru = len(points) / max(dt, 1)
    full_est_h = 47_824_966 / max(thru, 1) / 3600
    log(f"\n[throughput] slice {len(points):,} pts in {dt:.0f}s = {thru:.0f} pts/s "
        f"(projection+assembly, single box). Naive full-project ~= {full_est_h:.1f}h "
        f"for 47.8M (excludes HNSW m=16 build + corpus full-scan amortization).")

    log("\n[STOP] Slice complete. Full run ignites only on explicit orchestrator go.")
    return results, man, stats


def _dense_contract_sample(points: list[dict], *, generic_limit: int = 200,
                           explicit_limit: int = 200) -> list[dict]:
    """Bound the dense-contract check without making rare channel rows invisible.

    An evenly-spaced 200-point sample of a ~27M build can contain zero of
    Fedlex's ~22k explicitly re-embedded records. Sample the ordinary cohort
    as before, then add a distinct bounded explicit-channel cohort. This is an
    acceptance *coverage* guard, not a metric sample.
    """
    generic = points[:: max(1, len(points) // generic_limit)][:generic_limit]
    explicit = [p for p in points if p.get("dense_reembed_text")][:explicit_limit]
    seen = set()
    sample = []
    for point in generic + explicit:
        if point["id"] not in seen:
            seen.add(point["id"])
            sample.append(point)
    return sample


def run_acceptance_checks(build, twin, coll, points, tier_counts, man, man_errs,
                          stats, log=print, gate=None, admitted_by_label=None,
                          ids_by_label=None, excl_reasons=None):
    results = {}

    # (a) tokenizer parity — folded query md5 hits stored tokens for accented terms.
    # Candidate terms are extracted with the REAL bm25 token boundary (aw_common's
    # _TOKEN_RE = \w+, unicode) and filtered to genuinely-accented tokens (fold
    # changes them). Using \w+ (not a hand ASCII class) avoids fragmenting
    # multi-accent words like "Pyrénées"/"größe" — those fragments are not tokens
    # the index ever stored and would spuriously "miss".
    parity_hits = parity_total = 0
    samples = []
    seen_terms = set()
    for p in points:
        if p["tok_epoch"] != 2 or "bm25" not in p["vector"]:
            continue
        pl = p["payload"]
        text_fields = " ".join(str(pl.get(k) or "") for k in ("title", "snippet"))
        stored = set(p["vector"]["bm25"].indices)
        for tok in _TOKEN_RE_LOCAL.findall(text_fields.lower()):
            if len(tok) < 3 or _fold_local(tok) == tok:   # not accented -> skip
                continue
            qtf = _tokenize.bm25_tokens(tok, pl.get("lang"))  # query side, fold ON
            if not qtf:
                continue
            h = next(iter(qtf))
            parity_total += 1
            hit = h in stored
            parity_hits += hit
            if hit and tok not in seen_terms and len(samples) < 20:
                samples.append(tok)
                seen_terms.add(tok)
            if parity_total >= 400:
                break
        if parity_total >= 400 and len(samples) >= 20:
            break
    rate = parity_hits / max(parity_total, 1)
    results["a_tokenizer_parity"] = {"tested": parity_total, "hits": parity_hits,
                                     "rate": round(rate, 3), "samples": samples[:20]}
    log(f"(a) tokenizer parity: {parity_hits}/{parity_total} = {rate:.3f} "
        f"(folded accented query terms hit stored tokens) "
        f"{'PASS' if rate >= 0.95 else 'CHECK'}")

    # (b) id spec — worked examples reproduce; and every stored point_id is exactly
    # mint(canonical_url, anchor_component_implied_by_stored_tier): Tier A -> stored
    # anchor; Tier B -> stored content_digest; Tier C -> "". This proves the id is
    # self-consistent with the record's stored tier/anchor/digest.
    ID.self_test()
    idok = idbad = 0
    bad_samples = []
    for p in points[:8000]:
        pl = p["payload"]
        tier = pl.get("tier")
        if tier == "A":
            comp = ID._clean_anchor(pl.get("anchor")) or ""
        elif tier == "B":
            # points are built in-process by project_record, never read back from a collection
            comp = pl.get("content_digest") or ""  # canonical row
        else:  # C
            comp = ""
        # 325: a persisted collision salt is part of the id, not of the digest
        comp = ID.salted_anchor_component(comp, pl.get("id_salt") or 0)
        expect = ID.mint_point_id(AC.canonical_url(pl["url"]), comp)
        if expect == p["id"]:
            idok += 1
        else:
            idbad += 1
            if len(bad_samples) < 5:
                bad_samples.append((pl.get("url"), tier))
    results["b_id_spec"] = {"worked_examples": "PASS (#1,#4 exact)",
                            "recomputed_ok": idok, "recomputed_bad": idbad,
                            "bad": bad_samples}
    log(f"(b) id spec: worked examples #1/#4 exact; stored-id consistency "
        f"{idok} ok / {idbad} bad {'PASS' if idbad == 0 else 'FAIL ' + str(bad_samples)}")

    # (c) stamps present on 100%
    missing = 0
    for p in points:
        pl = p["payload"]
        if "rule_epochs" not in pl or "content_digest" not in pl or "build_id" not in pl:
            missing += 1
        else:
            # points are built in-process by project_record, never read back from a collection
            if set(pl["rule_epochs"].keys()) != set(ID.EPOCHS.keys()):  # canonical row
                missing += 1
    results["c_stamps"] = {"points": len(points), "missing_stamps": missing}
    log(f"(c) stamps present: {len(points)-missing}/{len(points)} complete "
        f"{'PASS' if missing == 0 else 'FAIL'}")

    # (d) counts invariant: final == projected unique; manifest points==adds
    final = build.count(coll, exact=True).count
    inv_ok = (final == len(points) == man["counts"]["adds"] == man["counts"]["points"])
    results["d_counts"] = {"final": final, "projected": len(points),
                           "manifest_points": man["counts"]["points"], "ok": inv_ok}
    log(f"(d) counts invariant: final={final:,} projected={len(points):,} "
        f"manifest.points={man['counts']['points']:,} "
        f"{'PASS' if inv_ok else 'FAIL'}")

    # (e) dense byte-identical to source unless its explicit corpus channel
    # demanded re-embedding (Fedlex R2). The latter compares the stored vector
    # to the projected target, not to the knowingly superseded source vector.
    sample = _dense_contract_sample(points)
    ids = [p["id"] for p in sample]
    got = {r.id: (r.vector.get("dense") if isinstance(r.vector, dict) else r.vector)
           for r in build.retrieve(coll, ids=ids, with_vectors=True)}
    de_ok = de_bad = reembed_ok = reembed_bad = 0
    for p in sample:
        stored = got.get(p["id"])
        expected = list(p["vector"]["dense"])
        ok = stored is not None and len(stored) == len(expected) and \
            all(abs(a - b) <= 1e-6 for a, b in zip(stored, expected))
        if p.get("dense_reembedded"):
            reembed_ok += int(ok)
            reembed_bad += int(not ok)
        else:
            de_ok += int(ok)
            de_bad += int(not ok)
    results["e_dense_contract"] = {
        "sampled": len(sample), "reused_ok": de_ok, "reused_bad": de_bad,
        "reembedded_ok": reembed_ok, "reembedded_bad": reembed_bad,
    }
    log(f"(e) dense contract: reused {de_ok}/{de_ok + de_bad} identical; "
        f"reembedded {reembed_ok}/{reembed_ok + reembed_bad} persisted "
        f"{'PASS' if de_bad == 0 and reembed_bad == 0 else 'FAIL'} "
        f"(tol 1e-6; qdrant returns f32 originals)")

    # (f) manifest validates
    results["f_manifest"] = {"errors": man_errs}
    log(f"(f) manifest validates: {'PASS' if not man_errs else 'FAIL: ' + '; '.join(man_errs)}")

    # (g) id-uniqueness vs chunk-count (task 319 / GAP A33 catch): every
    # admitted source record mints a distinct id, per slice label. Card packs
    # are exempt (Tier C is one-id-per-url BY DESIGN). The 318 build scored
    # anthropic.com at 15 ids / 414 chunks = 0.036 — this check fails <0.85.
    g_ok = True
    g_detail = {}
    for label, adm in sorted((admitted_by_label or {}).items()):
        if "tierC" in label or "cards" in label:
            continue
        minted = len((ids_by_label or {}).get(label, set()))
        ratio = minted / max(adm, 1)
        g_detail[label] = {"admitted": adm, "distinct_ids": minted,
                           "ratio": round(ratio, 3)}
        if adm >= 20 and ratio < 0.85:
            g_ok = False
    results["g_id_uniqueness"] = {"ok": g_ok, "per_label": g_detail}
    log(f"(g) id-uniqueness vs chunk-count: "
        f"{'PASS' if g_ok else 'FAIL (chunk collapse!)'} {g_detail}")

    # (h) per-class consent accounting: every exclusion/retention class is
    # named and counted (the 318 STATE reported consent_excluded: 0 while
    # 1.79M were excluded). In pipelined mode the grace classes must be
    # exercised; declines must equal the declined exclusion reasons.
    gs = dict(gate.stats) if gate is not None else {}
    er = dict(excl_reasons or {})
    h_info = {"gate_classes": gs, "exclusion_reasons": er,
              "grace_retained": stats.get("consent_grace_retained", 0)}
    if gate is not None and gate.mode == "pipelined":
        h_ok = ((gs.get("uncensused_grace", 0) > 0 if gate.post_census else True)
                and stats.get("consent_grace_retained", 0) > 0
                and sum(v for k, v in er.items() if "declined" in k)
                == gs.get("declined", 0))
        h_info["ok"] = h_ok
        log(f"(h) consent accounting (pipelined): {'PASS' if h_ok else 'FAIL'} {h_info}")
    else:
        h_info["ok"] = None
        log(f"(h) consent accounting (report-only, mode="
            f"{getattr(gate, 'mode', '?')}): {h_info}")
    results["h_consent_accounting"] = h_info

    # (i) deferred-exclusion tripwire fires on a synthetic 100%-exclusion and
    # stays quiet on a sane rate
    fired = quiet = False
    try:
        check_deferred_tripwire(100_000, 100_000)
    except SystemExit:
        fired = True
    try:
        check_deferred_tripwire(100_000, 2_000)
        quiet = True
    except SystemExit:
        pass
    results["i_tripwire"] = {"fires_on_100pct": fired, "quiet_on_2pct": quiet}
    log(f"(i) deferred tripwire: fires on synthetic 100% exclusion={fired}, "
        f"quiet at 2%={quiet} {'PASS' if fired and quiet else 'FAIL'}")

    # (j) ignition assertion: blocks when provenance is bad (dirty / off-main /
    # non-repo tree), honors the override env. Environment-dependent: on a
    # clean merged checkout would_block=False and that is also a PASS.
    problems = ignition_check()
    would_block = bool(problems)
    raised = False
    saved = os.environ.pop(IGNITION_OVERRIDE_ENV, None)
    try:
        try:
            assert_ignition(log=lambda *a: None)
        except SystemExit:
            raised = True
        os.environ[IGNITION_OVERRIDE_ENV] = "selftest"
        try:
            assert_ignition(log=lambda *a: None)
            override_ok = True
        except SystemExit:
            override_ok = False
    finally:
        if saved is None:
            os.environ.pop(IGNITION_OVERRIDE_ENV, None)
        else:
            os.environ[IGNITION_OVERRIDE_ENV] = saved
    j_ok = (raised == would_block) and override_ok
    results["j_ignition"] = {"problems": problems, "blocks": raised,
                             "override_honored": override_ok, "ok": j_ok}
    log(f"(j) ignition assertion: problems={len(problems)} blocks={raised} "
        f"override_honored={override_ok} {'PASS' if j_ok else 'FAIL'}"
        + (f" | first problem: {problems[0]}" if problems else ""))

    # (k) ch-media undated-gov exclusion (task 321; preflight T8/M21): the
    # counter FIRES on exactly the ruled class and nothing else. Synthetic
    # records make the check deterministic (independent of the live pack's
    # date mix; trust-mode gate keeps it offline); the live slice accounting
    # is reconciled on top (counter == reasons; no undated admin.ch ch-media
    # row admitted).
    from types import SimpleNamespace as _NS
    kst: defaultdict = defaultdict(int)
    kgate = ConsentGate("trust", 30, "", log=lambda *a: None)

    def _krec(url, dom, pub=None):
        pl = {"url": url, "domain": dom, "source_type": "news"}
        if pub:
            pl["published"] = pub
        return _NS(id="k", payload=pl, vector=None)

    u_gov = "https://www.admin.ch/gov/de/start/dokumentation/m321.html"
    # 1) undated admin.ch row in the ch-media pack -> excluded BY THIS RULE
    p1, m1 = project_record(_krec(u_gov, "www.admin.ch"), CHMEDIA_PACK,
                            None, {}, {}, kgate, kst)
    fired = (p1 is None and str(m1.get("reason", "")).startswith("undated_gov")
             and kst["staged_excluded_undated_gov"] == 1)
    # 2) DATED admin.ch row -> passes the rule (payload date)
    _, m2 = project_record(_krec(u_gov, "www.admin.ch", pub="2026-07-01"),
                           CHMEDIA_PACK, None, {}, {}, kgate, kst)
    # 3) undated admin.ch row whose date the SIDECAR carries -> passes
    _, m3 = project_record(_krec(u_gov, "www.admin.ch"), CHMEDIA_PACK, None,
                           {AC.canonical_url(u_gov): "2026-07-01"}, {},
                           kgate, kst)
    # 4) undated NON-admin.ch ch-media row -> passes the rule
    _, m4 = project_record(_krec("https://www.watson.ch/a/m321", "watson.ch"),
                           CHMEDIA_PACK, None, {}, {}, kgate, kst)
    passes_ok = all(not str(m.get("reason", "")).startswith("undated_gov")
                    for m in (m2, m3, m4))
    counter_exact = kst["staged_excluded_undated_gov"] == 1  # only case 1
    # live-slice reconciliation: counter == excluded reasons; zero admitted
    # undated admin.ch rows from the pack
    live_cnt = stats.get("staged_excluded_undated_gov", 0)
    er_cnt = sum(v for r, v in (excl_reasons or {}).items()
                 if str(r).startswith("undated_gov"))
    leaks = sum(1 for p in points
                if p["origin"] == CHMEDIA_PACK
                and _is_gov_host(p["payload"].get("domain") or "")
                and not p["payload"].get("published"))
    k_ok = fired and passes_ok and counter_exact and live_cnt == er_cnt \
        and leaks == 0
    results["k_chmedia_undated_gov"] = {
        "ok": k_ok, "synthetic_fired": fired, "synthetic_passes_ok": passes_ok,
        "slice_excluded": live_cnt, "slice_reasons": er_cnt,
        "admitted_undated_gov_leaks": leaks}
    log(f"(k) ch-media undated-gov exclusion: synthetic fired={fired} "
        f"dated/sidecar/non-gov pass={passes_ok} | slice excluded={live_cnt:,} "
        f"(reasons={er_cnt:,}) leaks={leaks} {'PASS' if k_ok else 'FAIL'}")

    # (l) payload contract: every PAYLOAD_CONTRACT population carries its
    # required field at >= min_rate (task 358 R1 — the canton-stamp-strip
    # class of bug).
    l_ok, l_detail = evaluate_payload_contract(points)  # noqa: E741 (matches a..k)
    results["l_payload_contract"] = {"ok": l_ok, "rules": l_detail}
    log(f"(l) payload contract: {'PASS' if l_ok else 'FAIL'} {l_detail}")

    # (m) dump-basis emission (task 468 / 433 WI-2): every point the consent
    # gate admitted on ADR-076 dump evidence carries that evidence out again,
    # so the exemption survives the next rebuild instead of re-arming one
    # generation later. Exact — no rate, no min_sample.
    m_ok, m_detail = evaluate_dump_basis_emission(points)
    results["m_dump_basis_emission"] = m_detail
    log(f"(m) dump-basis emission: {'PASS' if m_ok else 'FAIL'} {m_detail}")

    log(f"\ntier distribution: {dict(tier_counts)}")
    return results
