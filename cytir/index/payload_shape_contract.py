#!/usr/bin/env python3
"""Does the build's payload LOOK like something the ranker can read?

Conservation counts rows. It cannot see a row that loaded with no
`trust_tier`, no `source_type` and no BM25 posting -- task 514 shipped
11,273,954 and 19,619,656 of those respectively and every check passed -- and it
cannot see a title that is present, non-empty, and missing the handle the query
uses, which is task 516's fedlex defect.

This is the check that can. It reads `data/payload-shape-contract.v1.json` (the
declared thresholds, per set and per family, each with its `why`) and evaluates
it against the binder's own stage receipts, which already carry the per-field
hydration counters, the no-posting count and -- new in task 516 -- the
per-partition title-shape count.

Two entry points, one contract:

  * `evaluate(receipts, contract)` -- from the receipts, at STAGE time, before
    a single row is upserted;
  * the same function again in `assembly_binder verify`, so a tree that was
    staged before the contract existed cannot be loaded and shipped unchecked.

`title_shape` is deliberately OPT-IN per family. Inventing a pattern for a
family whose production titles never carried one would fail a build over a rule
nobody measured a need for -- task 516 section 3 measured exactly that for
jorf, kali, cantonal law, gesetze and legifrance.
"""
from __future__ import annotations

import json
import os
import pathlib
import re

from cytir.core.paths import REPO_ROOT, data_path

SCHEMA = "aw.payload-shape-contract.v1"
DEFAULT_PATH = str(data_path("payload-shape-contract.v1.json"))


def load(path: str | pathlib.Path | None = None) -> dict:
    p = pathlib.Path(path) if path else (
        pathlib.Path(DEFAULT_PATH))
    doc = json.loads(p.read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{p}: not {SCHEMA}")
    doc["_path"] = str(p)
    return doc


def provenance_head_pattern(contract: dict):
    """One compiled alternation over every declared provenance-head shape.

    The class this catches: a materializer writes its own bookkeeping into the
    head of the document, the chunker's section-adjacent policy makes it the
    head of chunk 0, and the row ships with provenance as its snippet AND as
    the head of its dense embedding. Task 516 found it in fedlex (22,913 of
    22,928 rows), task 517 found it one family over in context7 (72,330 of
    126,427) -- both after the collection was built, because nothing measured
    the SHAPE of a chunk's head. This does, per partition, at stage.

    Returns None when the contract declares no pattern, so a contract file
    without the section is a no-op rather than a crash.
    """
    pats = contract.get("provenance_head_patterns") or {}
    parts = [str(v["pattern"]) for v in pats.values()
             if isinstance(v, dict) and v.get("pattern")]
    if not parts:
        return None
    return re.compile("|".join(f"(?:{p})" for p in parts))


def title_pattern(contract: dict, source: str | None):
    """The compiled title contract for a source, or None when it has none."""
    if not source:
        return None
    row = (contract.get("title_shape") or {}).get(source)
    if not isinstance(row, dict) or not row.get("pattern"):
        return None
    return re.compile(row["pattern"]), float(row.get("min_share") or 0.0)


def published_presence_min_share(contract: dict, source: str | None) -> float | None:
    """The declared `published`-presence floor for a source, or None.

    Task 536, same OPT-IN shape as `title_pattern`: a family that never
    stamped `published` in production (almost all of them -- ADR-039's shared
    minimum has no `published` in it at all, task 535 sec.3.5) is not checked
    just because the field now exists. A family becomes checkable by
    DECLARING what the collection this build replaces actually carried --
    the twin's own rate, not an invented one.
    """
    if not source:
        return None
    row = (contract.get("published_presence") or {}).get(source)
    if not isinstance(row, dict) or row.get("min_share") is None:
        return None
    return float(row["min_share"])


def unit_of(receipt: dict) -> tuple[str, str | None]:
    """(unit key, source) for one staged partition.

    NOT the set id alone. `task417_families` pools 123 unrelated sources into
    one set -- arxiv, pubmed, fedlex, cantonal law, 26 context7 tranches -- and
    a per-SET fraction over 10.4M rows hides a source that is 100% defective:
    measured on this build, all 126,427 context7 rows carry no `title` at all
    and the set-level number is 1.2%. The unit a threshold means is the unit a
    producer owns.
    """
    set_id = receipt.get("set_id") or "?"
    shard = receipt.get("shard") or ""
    # Only the PACK route names a source in its shard key
    # ("source=<name>.shard-0000"); a plain "shard-0000" is the set itself.
    # Reading the prefix unconditionally split `task477_direct_capture_...`
    # into nine one-shard "sources" -- units nobody owns and no threshold can
    # be declared about.
    if not shard.startswith("source="):
        return set_id, None
    source = shard.split(".shard-")[0].removeprefix("source=")
    return (f"{set_id}/source={source}", source) if source else (set_id, None)


def _limits(contract: dict, unit: str, source: str | None) -> dict:
    """Defaults, then the set-level override, then the source-level one."""
    per = contract.get("per_set") or {}
    out = dict(contract.get("defaults") or {})
    out.update(per.get(unit.split("/source=")[0]) or {})
    if source:
        out.update(per.get(source) or {})
    return out


def evaluate(receipts: list[dict], contract: dict) -> dict:
    """Per set: missing-serving-signal and no-posting fractions vs the contract;
    per family: the title-shape share. Returns a report; never raises."""
    by_set: dict[str, dict] = {}
    by_source: dict[str, dict] = {}
    by_pub_source: dict[str, dict] = {}
    for r in receipts:
        unit, source = unit_of(r)
        staged = int(r.get("staged") or 0)
        meta = r.get("serving_metadata") or {}
        s = by_set.setdefault(unit, {"set_id": unit, "source": source,
                                     "staged": 0,
                                     "missing_serving_signal": 0,
                                     "no_bm25_posting": 0,
                                     "provenance_head": 0,
                                     "provenance_head_measured": 0,
                                     "partitions": 0})
        s["staged"] += staged
        s["partitions"] += 1
        s["missing_serving_signal"] += int(meta.get("meta_row_missing_serving_signal") or 0)
        s["no_bm25_posting"] += int(r.get("staged_without_sparse") or 0)
        ph = r.get("provenance_head")
        if isinstance(ph, dict):
            s["provenance_head"] += int(ph.get("rows_with_provenance_head") or 0)
            s["provenance_head_measured"] += int(ph.get("rows") or 0)
        collapse = r.get("extraction_collapse")
        if isinstance(collapse, dict) and "extraction_collapse" not in s:
            # Set-level and identical on every receipt of the set: read once,
            # never summed, or a set with N partitions would report N times its
            # own rows.
            s["extraction_collapse"] = collapse
        width = r.get("chunk_width")
        if isinstance(width, dict) and "chunk_width" not in s:
            # Set-level and identical on every receipt of the set, exactly like
            # `extraction_collapse` above: read once, never summed.
            s["chunk_width"] = width
        shape = r.get("title_shape") or {}
        source = shape.get("source")
        if source:
            t = by_source.setdefault(source, {"source": source, "rows": 0,
                                              "matched": 0})
            t["rows"] += int(shape.get("rows") or 0)
            t["matched"] += int(shape.get("matched") or 0)

        pub = r.get("published_presence") or {}
        psource = pub.get("source")
        if psource:
            p = by_pub_source.setdefault(psource, {"source": psource,
                                                    "rows": 0, "present": 0})
            p["rows"] += int(pub.get("rows") or 0)
            p["present"] += int(pub.get("present") or 0)

    findings: list[str] = []
    sets = []
    for sid, s in sorted(by_set.items()):
        lim = _limits(contract, sid, s.get("source"))
        staged = max(1, s["staged"])
        miss_frac = s["missing_serving_signal"] / staged
        sparse_frac = s["no_bm25_posting"] / staged
        max_miss = float(lim.get("max_missing_serving_signal_frac", 1.0))
        max_sparse = float(lim.get("max_no_bm25_posting_frac", 1.0))
        row = {**s, "missing_serving_signal_frac": round(miss_frac, 8),
               "no_bm25_posting_frac": round(sparse_frac, 8),
               "max_missing_serving_signal_frac": max_miss,
               "max_no_bm25_posting_frac": max_sparse,
               "verdict": "PASS"}
        if miss_frac > max_miss:
            row["verdict"] = "FAIL"
            findings.append(
                f"{sid}: {s['missing_serving_signal']:,} of {s['staged']:,} "
                f"staged rows ({miss_frac:.4%}) carry no "
                f"title/trust_tier/source_type/lang, above the declared "
                f"{max_miss:.4%}")
        if sparse_frac > max_sparse:
            row["verdict"] = "FAIL"
            findings.append(
                f"{sid}: {s['no_bm25_posting']:,} of {s['staged']:,} staged "
                f"rows ({sparse_frac:.4%}) carry no BM25 posting, above the "
                f"declared {max_sparse:.4%} -- they are invisible to the "
                f"lexical leg")
        max_prov = float(lim.get("max_provenance_head_frac", 1.0))
        measured = s.get("provenance_head_measured") or 0
        row["max_provenance_head_frac"] = max_prov
        if not measured:
            # An unmeasured unit is NOT a passing one. Task 514 and 515 both
            # shipped this defect class inside a build-wide average nobody
            # decomposed; a silent "no data" here would be the same mistake.
            row["provenance_head_frac"] = None
            if max_prov < 1.0:
                findings.append(
                    f"{sid}: no partition measured the provenance-head shape, "
                    f"so the declared {max_prov:.4%} ceiling was never "
                    f"evaluated. Re-stage, or measure it at verify.")
                row["verdict"] = "FAIL"
        else:
            prov_frac = s["provenance_head"] / measured
            row["provenance_head_frac"] = round(prov_frac, 8)
            if prov_frac > max_prov:
                row["verdict"] = "FAIL"
                findings.append(
                    f"{sid}: {s['provenance_head']:,} of {measured:,} staged "
                    f"rows ({prov_frac:.4%}) open with a provenance block, "
                    f"above the declared {max_prov:.4%} -- the snippet AND the "
                    f"head of the dense embedding are the materializer's own "
                    f"bookkeeping, not the document")
        spec = None
        try:
            spec = extraction_collapse_limits(contract, sid)
        except ValueError as exc:
            row["verdict"] = "FAIL"
            findings.append(str(exc))
        if spec is not None:
            collapse = s.get("extraction_collapse")
            row["extraction_collapse"] = collapse
            row["max_collapsed_row_frac"] = float(spec["max_collapsed_row_frac"])
            if not collapse or not collapse.get("rows"):
                # Declared but unmeasured is a FAIL, for the same reason the
                # provenance-head dimension treats it as one: a contract
                # nothing evaluates is not a contract.
                row["collapsed_row_frac"] = None
                row["verdict"] = "FAIL"
                findings.append(
                    f"{sid}: an extraction-collapse contract is declared "
                    f"(>= {spec['min_cohort_urls']} URLs on one content_digest "
                    f"at <= {spec['sparse_posting_ceiling']} BM25 postings) and "
                    f"no partition measured it. Re-stage, or measure it at "
                    f"verify.")
            else:
                frac = collapse["collapsed_rows"] / max(1, collapse["rows"])
                row["collapsed_row_frac"] = round(frac, 8)
                if frac > float(spec["max_collapsed_row_frac"]):
                    row["verdict"] = "FAIL"
                    top = (collapse.get("top_cohorts") or [{}])[0]
                    findings.append(
                        f"{sid}: {collapse['collapsed_rows']:,} of "
                        f"{collapse['rows']:,} staged rows "
                        f"({frac:.4%}) are {collapse['cohorts']:,} digest "
                        f"cohort(s) of >= {spec['min_cohort_urls']} distinct "
                        f"URLs sharing one body of <= "
                        f"{spec['sparse_posting_ceiling']} BM25 postings, above "
                        f"the declared "
                        f"{float(spec['max_collapsed_row_frac']):.4%}. Largest: "
                        f"{top.get('distinct_urls', 0):,} URLs, "
                        f"{top.get('sparse_postings', 0)} postings, "
                        f"{str(top.get('snippet'))[:70]!r} -- the extractor "
                        f"returned the same fragment for every page")

        # -- chunk width (task 535/541) ---------------------------------
        wspec = None
        try:
            wspec = chunk_width_limits(contract, sid)
        except ValueError as exc:
            row["verdict"] = "FAIL"
            findings.append(str(exc))
        if wspec is not None:
            width = s.get("chunk_width")
            row["chunk_width"] = width
            if not width or not width.get("rows"):
                # Declared but unmeasured is a FAIL, same as every other
                # dimension here: a contract nothing evaluates is not a
                # contract (verification-doctrine 12).
                row["verdict"] = "FAIL"
                findings.append(
                    f"{sid}: a chunk-width contract is declared (policy "
                    f"{str(wspec['policy_sha256'])[:12]}...) and no partition "
                    f"measured it. Re-stage, or measure it at verify.")
            else:
                if width.get("policy_sha256") != wspec["policy_sha256"]:
                    row["verdict"] = "FAIL"
                    findings.append(
                        f"{sid}: staged under chunk policy "
                        f"{str(width.get('policy_sha256'))[:12]}... but the "
                        f"contract declares "
                        f"{str(wspec['policy_sha256'])[:12]}.... A policy is "
                        f"identified by its bytes; these are different rules.")
                present = width["rows_measured"] / max(1, width["rows"])
                row["chunk_n_words_presence"] = round(present, 8)
                minimum = float(wspec["min_chunk_n_words_presence"])
                if present < minimum:
                    row["verdict"] = "FAIL"
                    findings.append(
                        f"{sid}: {width['rows_measured']:,} of "
                        f"{width['rows']:,} staged rows ({present:.4%}) carry "
                        f"`chunk_n_words`, below the declared {minimum:.4%}. "
                        f"A row that cannot state its own width cannot be "
                        f"checked against the policy it claims (task 535 "
                        f"section 2.3: the instrument did not exist).")
                over_frac = width["rows_over_hard_max"] / max(1, width["rows_measured"])
                row["rows_over_hard_max_frac"] = round(over_frac, 8)
                if over_frac > float(wspec["max_rows_over_hard_max_frac"]):
                    row["verdict"] = "FAIL"
                    findings.append(
                        f"{sid}: {width['rows_over_hard_max']:,} of "
                        f"{width['rows_measured']:,} measured rows "
                        f"({over_frac:.4%}, widest {width['max_chunk_n_words']} "
                        f"words) exceed the declared policy's own "
                        f"word_guard.hard_max of {width['hard_max']}. The rows "
                        f"were not cut by the policy this set declares.")
                max_mis = wspec.get("max_mis_selected_frac")
                if max_mis is not None:
                    sel_rows = width.get("rows_selector_measured") or 0
                    if not sel_rows:
                        row["verdict"] = "FAIL"
                        findings.append(
                            f"{sid}: a selector-conformance ceiling "
                            f"({float(max_mis):.4%}) is declared and NO staged "
                            f"row carries a `chunk_selector` stamp. The "
                            f"materializer holds the markdown; it is the only "
                            f"stage that can record which family the page "
                            f"belongs to.")
                    else:
                        mis_frac = width["rows_mis_selected"] / sel_rows
                        row["mis_selected_frac"] = round(mis_frac, 8)
                        if mis_frac > float(max_mis):
                            row["verdict"] = "FAIL"
                            findings.append(
                                f"{sid}: {width['rows_mis_selected']:,} of "
                                f"{sel_rows:,} stamped rows ({mis_frac:.4%}) "
                                f"sit on pages the ratified selector assigns to "
                                f"a different chunk policy family than "
                                f"{width.get('policy_family')!r}, above the "
                                f"declared {float(max_mis):.4%}. This is task "
                                f"535's defect stated as a refusal: the rows "
                                f"are correctly declared and the PAGES are in "
                                f"the wrong family.")
        sets.append(row)

    families = []
    for source, spec in sorted((contract.get("title_shape") or {}).items()):
        if not isinstance(spec, dict) or not spec.get("pattern"):
            continue
        t = by_source.get(source)
        if t is None or not t["rows"]:
            findings.append(
                f"{source}: a title-shape contract is declared "
                f"(min_share {spec.get('min_share')}) but the stage measured no "
                f"rows for it. A contract nothing evaluates is not a contract; "
                f"either the family is absent from this build or the source "
                f"name in the contract does not match the staged partition.")
            families.append({"source": source, "rows": 0, "matched": 0,
                             "share": None, "min_share": spec.get("min_share"),
                             "verdict": "NOT EVALUATED"})
            continue
        share = t["matched"] / max(1, t["rows"])
        minimum = float(spec.get("min_share") or 0.0)
        verdict = "PASS" if share >= minimum else "FAIL"
        if verdict == "FAIL":
            findings.append(
                f"{source}: {t['matched']:,} of {t['rows']:,} titles "
                f"({share:.2%}) match /{spec['pattern']}/, below the declared "
                f"{minimum:.0%}. Example of the shape this family's titles must "
                f"carry: {spec.get('sample')!r}")
        families.append({"source": source, **t, "share": round(share, 6),
                         "min_share": minimum, "pattern": spec["pattern"],
                         "verdict": verdict})

    published_families = []
    for source, spec in sorted((contract.get("published_presence") or {}).items()):
        if not isinstance(spec, dict) or spec.get("min_share") is None:
            continue
        p = by_pub_source.get(source)
        minimum = float(spec["min_share"])
        if p is None or not p["rows"]:
            findings.append(
                f"{source}: a published-presence contract is declared "
                f"(min_share {minimum}) but the stage measured no rows for "
                f"it. A contract nothing evaluates is not a contract; either "
                f"the family is absent from this build or the source name in "
                f"the contract does not match the staged partition.")
            published_families.append({
                "source": source, "rows": 0, "present": 0, "share": None,
                "min_share": minimum, "verdict": "NOT EVALUATED"})
            continue
        share = p["present"] / max(1, p["rows"])
        verdict = "PASS" if share >= minimum else "FAIL"
        if verdict == "FAIL":
            findings.append(
                f"{source}: {p['present']:,} of {p['rows']:,} staged rows "
                f"({share:.2%}) carry `published`, below the declared "
                f"{minimum:.0%} -- the collection this build replaces carried "
                f"it at that rate (task 535 sec.3.1); a drop here is the same "
                f"projection defect task 536 fixed, recurring.")
        published_families.append({"source": source, **p,
                                   "share": round(share, 6),
                                   "min_share": minimum, "verdict": verdict})

    return {"schema": "aw.payload-shape-report.v1",
            "contract": contract.get("_path"),
            "sets": sets, "families": families,
            "published_families": published_families, "findings": findings,
            "verdict": "FAIL" if findings else "PASS"}


def format_report(report: dict, log=print) -> None:
    log(f"payload-shape contract: {report['verdict']} "
        f"({len(report['sets'])} set(s), {len(report['families'])} "
        f"title-shape family/families)")
    for f in report["families"]:
        log(f"  title shape {f['source']:<36} "
            f"{f.get('matched', 0):>9,}/{f.get('rows', 0):<9,} "
            f"= {('%.2f%%' % (100 * f['share'])) if f.get('share') is not None else 'n/a':>8} "
            f"(min {f['min_share']:.0%})  {f['verdict']}")
    for pf in report.get("published_families") or []:
        log(f"  published  {pf['source']:<36} "
            f"{pf.get('present', 0):>9,}/{pf.get('rows', 0):<9,} "
            f"= {('%.2f%%' % (100 * pf['share'])) if pf.get('share') is not None else 'n/a':>8} "
            f"(min {pf['min_share']:.0%})  {pf['verdict']}")
    for row in report["sets"]:
        if row["verdict"] != "PASS":
            pf = row.get("provenance_head_frac")
            log(f"  [!] {row['set_id']}: missing-signal "
                f"{row['missing_serving_signal']:,} ({row['missing_serving_signal_frac']:.4%}) "
                f"no-posting {row['no_bm25_posting']:,} "
                f"({row['no_bm25_posting_frac']:.4%}) "
                f"provenance-head {row.get('provenance_head', 0):,} "
                f"({('%.4f%%' % (100 * pf)) if pf is not None else 'not measured'})")
    for finding in report["findings"]:
        log(f"  FAIL {finding}")


def measure_partition_titles(payload_path, pattern) -> tuple[int, int]:
    """(rows, matched) over a staged partition's payload file.

    The staged payload is the ground truth for what the load will upsert, so a
    tree staged before the title contract existed can still be measured against
    it -- at verify time, loudly, rather than being reported as unchecked. Reads
    the file once, streams it, and holds nothing.
    """
    import gzip
    rows = matched = 0
    with gzip.open(payload_path, "rt") as fh:
        for line in fh:
            title = (json.loads(line).get("payload") or {}).get("title") or ""
            rows += 1
            if pattern.search(title):
                matched += 1
    return rows, matched


def measure_partition_provenance_heads(payload_path, pattern) -> tuple[int, int]:
    """(rows, rows whose snippet opens with a declared provenance block).

    The snippet is the first thing the chunk's text became, so a match here is
    also a statement about the head of that row's dense embedding.
    """
    import gzip
    rows = hits = 0
    with gzip.open(payload_path, "rt") as fh:
        for line in fh:
            rows += 1
            if pattern.match((json.loads(line).get("payload") or {}).get("snippet") or ""):
                hits += 1
    return rows, hits


# --- extraction collapse (task 523) -----------------------------------------
# The class: ONE stored body serving MANY distinct URLs. It is invisible to
# every count check (the rows are all present, conservation balances) and to
# the provenance-head rule (the text is a real fragment of a real page, just
# the same wrong fragment for every page -- the task-517 contract flagged 5 of
# 127,074 F2-HTML rows while 17,850 of them were collapsed).
#
# MEASURED, task 521/523, on `task472_f2_html`: 846 digest cohorts spanning
# 10,143 distinct URLs and 17,850 rows = 14.05% of the lane. The head of that
# distribution, by cohort size and dense tokens:
#     eeoc.gov 3,010 urls @ 72 tokens (the ".gov websites use HTTPS" banner)
#     has-sante.fr 1,970 @ 693 | edudoc.ch 1,082 @ 36 | edpb.europa.eu 529 @ 117
#     citiesalliance.org 260 @ 148 | ies.ed.gov 129 @ 59 | cedefop 101 @ 155
#     rki.de 46 @ 282
# The token ceiling separates the two mechanisms that produce the signature: a
# SHORT body repeated across many pages is boilerplate the extractor mistook
# for the page (eeoc, edudoc, edpb), while a LONG one is the origin genuinely
# serving the same document at many URLs (has-sante.fr at 693 tokens: four
# nested-path URLs, four identical <title>s, extraction correct). Only the
# first is an extraction defect, so only the first is refused here.

# The length unit is DISTINCT BM25 POSTINGS, not snippet words: the stored
# snippet is truncated to 25 words by `aw_b1_projection.extractive_snippet`, so
# a snippet-length ceiling could not tell a 72-token banner from a 700-token
# document. `bm25.indices` is the sparse encoding of the chunk's whole sparse
# surface and is not truncated, so it is the payload's only honest measure of
# how much text a row actually carries.
def _sparse_postings(row: dict) -> int:
    bm25 = row.get("bm25") or {}
    indices = bm25.get("indices")
    return len(indices) if isinstance(indices, list) else 0


def measure_extraction_collapse(payload_paths, min_cohort_urls: int,
                                sparse_posting_ceiling: int) -> dict:
    """Digest cohorts spanning many distinct URLs, over a whole set's payloads.

    Two passes so memory is bounded by the COHORTS, not by the set: pass one
    counts rows per `content_digest` (a cohort needs at least `min_cohort_urls`
    rows before it can have that many URLs), pass two revisits only the digests
    that cleared it and collects their distinct URLs and body length. A cohort
    is a collapse when it spans >= `min_cohort_urls` distinct URLs and the
    shared body carries at most `sparse_posting_ceiling` distinct BM25
    postings.
    """
    import gzip
    paths = [str(x) for x in payload_paths]
    rows = 0
    row_count: dict[str, int] = {}
    for path in paths:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                digest = (json.loads(line).get("payload") or {}).get("content_digest")
                if not digest:
                    continue
                rows += 1
                row_count[digest] = row_count.get(digest, 0) + 1
    candidates = {d for d, n in row_count.items() if n >= min_cohort_urls}
    del row_count
    urls: dict[str, set] = {d: set() for d in candidates}
    hits: dict[str, int] = {d: 0 for d in candidates}
    tokens: dict[str, int] = {}
    sample: dict[str, str] = {}
    for path in paths:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                row = json.loads(line)
                payload = row.get("payload") or {}
                digest = payload.get("content_digest")
                if digest not in candidates:
                    continue
                urls[digest].add(payload.get("url"))
                hits[digest] += 1
                if digest not in tokens:
                    tokens[digest] = _sparse_postings(row)
                    sample[digest] = (payload.get("snippet") or "")[:120]
    cohorts = []
    for digest in candidates:
        distinct = len(urls[digest])
        if distinct < min_cohort_urls or tokens[digest] > sparse_posting_ceiling:
            continue
        cohorts.append({"content_digest": digest, "distinct_urls": distinct,
                        "rows": hits[digest], "sparse_postings": tokens[digest],
                        "snippet": sample[digest]})
    cohorts.sort(key=lambda c: -c["distinct_urls"])
    return {"rows": rows, "min_cohort_urls": min_cohort_urls,
            "sparse_posting_ceiling": sparse_posting_ceiling,
            "cohorts": len(cohorts),
            "collapsed_rows": sum(c["rows"] for c in cohorts),
            "collapsed_urls": sum(c["distinct_urls"] for c in cohorts),
            "top_cohorts": cohorts[:20],
            "partitions": len(paths)}


def chunk_width_limits(contract: dict, unit: str) -> dict | None:
    """The declared chunk-width rule for a set, or None when it declares none.

    Task 535/541. The epoch-4 generic packer cut long, densely-sectioned
    reference documents at 3.73x the width the incumbent collection serves --
    82 hosts, ~1.01M rows down, 150 of 262 lost golds -- and NOTHING in the
    repo could see it, because 84 of 86 sealed families declare
    `chunk_policy.policy_id: "unknown"` and no receipt ever carried a width.
    This dimension is where a lane states what it cut with and the stage
    refuses it when the rows disagree.

    OPT-IN per set, like `title_shape` and `extraction_collapse`: a family
    whose route has no markdown page (source-native records, cards, preserved
    passages) has no width to declare, and inventing one for it would fail a
    build over a rule nobody measured a need for.
    """
    spec = (contract.get("chunk_width") or {}).get(unit.split("/source=")[0])
    if not isinstance(spec, dict):
        return None
    for field in ("policy_sha256", "max_rows_over_hard_max_frac",
                  "min_chunk_n_words_presence"):
        if field not in spec:
            raise ValueError(
                f"chunk_width[{unit}] declares no {field}: the contract must "
                f"state it, never inherit a default")
    return spec


def measure_partition_chunk_width(payload_paths, policy_sha256: str,
                                  hard_max: int,
                                  expect_policy_family: str | None = None) -> dict:
    """Three counts over a set's staged payloads, all from the row itself.

      rows_measured       carry `chunk_n_words` at all. Before task 536 the
                          chunker computed it and `project_record` discarded
                          it, so a row could be any width and the payload could
                          not say -- which is why 535 had to argue the density
                          question from a chunk-count RATIO and could not
                          measure it.
      rows_over_hard_max  wider than the declared policy's own
                          `word_guard.hard_max`. This is a structural invariant
                          of the chunker, not a tuned threshold: a row above it
                          was not cut by the policy the lane declares.
      rows_mis_selected   the page's own selector verdict (`chunk_selector`,
                          stamped by the materializer that had the markdown)
                          names a DIFFERENT policy family than the one the row
                          was cut under. This is the arm that would have caught
                          the original defect, in which every row was correctly
                          declared epoch-4 and the pages should not have been.
    """
    import gzip
    paths = [str(x) for x in payload_paths]
    rows = measured = over = mis = sel_measured = 0
    max_words = 0
    declared_rules: dict[str, int] = {}
    for path in paths:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                payload = (json.loads(line).get("payload") or {})
                rows += 1
                rule = payload.get("chunk_rule")
                if rule:
                    declared_rules[rule] = declared_rules.get(rule, 0) + 1
                n = payload.get("chunk_n_words")
                if isinstance(n, int):
                    measured += 1
                    max_words = max(max_words, n)
                    if n > hard_max:
                        over += 1
                sel = payload.get("chunk_selector") or {}
                fam = sel.get("policy_family")
                if fam:
                    sel_measured += 1
                    if expect_policy_family and fam != expect_policy_family:
                        mis += 1
    return {"rows": rows, "rows_measured": measured,
            "rows_over_hard_max": over, "max_chunk_n_words": max_words,
            "hard_max": hard_max, "policy_sha256": policy_sha256,
            "policy_family": expect_policy_family,
            "rows_selector_measured": sel_measured, "rows_mis_selected": mis,
            "declared_chunk_rules": declared_rules,
            "partitions": len(paths)}


def extraction_collapse_limits(contract: dict, unit: str) -> dict | None:
    """The declared collapse rule for a set, or None when it declares none.

    OPT-IN, like `title_shape` and for the same reason: N and the token ceiling
    are judgements about a lane's own content shape, and a number nobody
    measured for a lane must not decide whether that lane ships.
    """
    spec = (contract.get("extraction_collapse") or {}).get(
        unit.split("/source=")[0])
    if not isinstance(spec, dict):
        return None
    for field in ("min_cohort_urls", "sparse_posting_ceiling",
                  "max_collapsed_row_frac"):
        if field not in spec:
            raise ValueError(
                f"extraction_collapse[{unit}] declares no {field}: the contract "
                f"must state it, never inherit a default")
    return spec


def backfill_extraction_collapse(receipts, staged_dir, contract, log=print) -> int:
    """Measure the collapse dimension for every SET that declares it.

    Set-level, not partition-level: a digest cohort spans partitions, so the
    question cannot be answered one shard at a time. The measurement is written
    onto every receipt of the set (identical dict, read once by `evaluate`), so
    it survives the stage->verify handoff the same way the other dimensions do.
    """
    by_set: dict[str, list] = {}
    for r in receipts:
        by_set.setdefault(r["set_id"], []).append(r)
    measured = 0
    for set_id, group in sorted(by_set.items()):
        spec = extraction_collapse_limits(contract, set_id)
        if spec is None or any(r.get("extraction_collapse") for r in group):
            continue
        paths = []
        for r in group:
            path = pathlib.Path(staged_dir) / f"{r['set_id']}.{r['shard']}.payload.jsonl.gz"
            if path.is_file():
                paths.append(path)
        if not paths:
            log(f"  [!] {set_id}: an extraction-collapse contract is declared "
                f"and no staged payload file can answer it")
            continue
        result = measure_extraction_collapse(
            paths, int(spec["min_cohort_urls"]),
            int(spec["sparse_posting_ceiling"]))
        for r in group:
            r["extraction_collapse"] = result
        measured += 1
        log(f"  extraction collapse measured for {set_id}: "
            f"{result['collapsed_rows']:,}/{result['rows']:,} rows in "
            f"{result['cohorts']:,} cohort(s)")
    return measured


def backfill_chunk_width(receipts, staged_dir, contract, log=print) -> int:
    """Measure the chunk-width dimension for every SET that declares it.

    Set-level and written onto every receipt of the set, the same shape as
    `backfill_extraction_collapse`, so it survives the stage->verify handoff
    identically. The declared policy's `word_guard.hard_max` and the registry's
    policy family are read from the pinned artifacts, never restated here
    (engineering-doctrine 1).
    """
    from cytir.chunk import policy as CPOL
    import cytir.chunk.registry as CREG

    repo = REPO_ROOT
    registry = None
    by_set: dict[str, list] = {}
    for r in receipts:
        by_set.setdefault(r["set_id"], []).append(r)
    measured = 0
    for set_id, group in sorted(by_set.items()):
        spec = chunk_width_limits(contract, set_id)
        if spec is None or any(r.get("chunk_width") for r in group):
            continue
        sha = spec["policy_sha256"]
        if registry is None:
            registry = CREG.load()
        ent = CREG.entry(sha, registry)
        policy = CPOL.load_chunk_policy(repo / ent["file"])
        if policy.sha256 != sha:
            raise ValueError(
                f"chunk_width[{set_id}]: {ent['file']} hashes to "
                f"{policy.sha256}, not the declared {sha}")
        paths = []
        for r in group:
            path = pathlib.Path(staged_dir) / f"{r['set_id']}.{r['shard']}.payload.jsonl.gz"
            if path.is_file():
                paths.append(path)
        if not paths:
            log(f"  [!] {set_id}: a chunk-width contract is declared and no "
                f"staged payload file can answer it")
            continue
        result = measure_partition_chunk_width(
            paths, sha, int(policy.word_hard_max),
            expect_policy_family=ent["policy_family"])
        for r in group:
            r["chunk_width"] = result
        measured += 1
        log(f"  chunk width measured for {set_id}: "
            f"{result['rows_measured']:,}/{result['rows']:,} rows carry "
            f"chunk_n_words, {result['rows_over_hard_max']:,} over "
            f"{policy.word_hard_max}w, widest {result['max_chunk_n_words']}w")
    return measured


def _measure_head_job(args):
    path, pattern_src = args
    return measure_partition_provenance_heads(path, re.compile(pattern_src))


def backfill_provenance_head(receipts, staged_dir, contract, log=print,
                             workers: int | None = None) -> int:
    """Measure the provenance-head shape for any receipt whose stage did not.

    Same posture as `backfill_title_shape`: a tree staged before a contract
    dimension existed is measured loudly rather than shipped unchecked.

    PARALLEL, unlike the title backfill, because this one is not scoped to a
    declared family: it reads EVERY partition. Measured on this build's tree,
    594 partitions and 40.9 GB of gzipped payloads, that is ~3.4 CPU-hours
    single-threaded and about 9 minutes across 24 workers -- and it runs twice,
    at stage and at verify. A check nobody can afford to run is a check that
    gets turned off.
    """
    pattern = provenance_head_pattern(contract)
    if pattern is None:
        return 0
    todo = []
    for r in receipts:
        if r.get("provenance_head"):
            continue
        path = pathlib.Path(staged_dir) / f"{r['set_id']}.{r['shard']}.payload.jsonl.gz"
        if not path.is_file():
            log(f"  [!] {r['set_id']}.{r['shard']}: the provenance-head contract "
                f"cannot be answered -- neither the stage receipt nor "
                f"{path.name} has it")
            continue
        todo.append((r, path))
    if not todo:
        return 0
    n = workers or min(24, os.cpu_count() or 8)
    log(f"  measuring the provenance-head shape on {len(todo):,} partition(s) "
        f"with {n} worker(s)")
    if n <= 1:
        results = [measure_partition_provenance_heads(p, pattern) for _, p in todo]
    else:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(_measure_head_job,
                                  [(str(p), pattern.pattern) for _, p in todo],
                                  chunksize=1))
    for (r, _), (rows, hits) in zip(todo, results):
        r["provenance_head"] = {
            "pattern": pattern.pattern, "rows": rows,
            "rows_with_provenance_head": hits,
            "measured_at": "backfilled (this partition's stage predates the "
                           "contract dimension)"}
    log(f"  provenance-head shape measured for {len(todo):,} partition(s)")
    return len(todo)


def backfill_title_shape(receipts, staged_dir, contract, log=print) -> int:
    """Fill `title_shape` on any receipt whose source HAS a contract and whose
    stage did not measure one. Returns the number of receipts backfilled."""
    filled = 0
    for r in receipts:
        if r.get("title_shape"):
            continue
        source = r["shard"].split(".shard-")[0].removeprefix("source=")
        rule = title_pattern(contract, source)
        if rule is None:
            continue
        path = pathlib.Path(staged_dir) / f"{r['set_id']}.{r['shard']}.payload.jsonl.gz"
        if not path.is_file():
            log(f"  [!] {source}: a title contract is declared and neither the "
                f"stage receipt nor {path.name} can answer it")
            continue
        rows, matched = measure_partition_titles(path, rule[0])
        r["title_shape"] = {"source": source, "pattern": rule[0].pattern,
                            "rows": rows, "matched": matched,
                            "measured_at": "verify (the stage predates the contract)"}
        filled += 1
        log(f"  title shape measured at VERIFY for {source}: "
            f"{matched:,}/{rows:,} = {matched / max(1, rows):.2%}")
    return filled
