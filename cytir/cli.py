"""cyt — the CLI entry point (task 646 S1; `dump`/`apply` added task 650 S8).

Target shape (task 643 §6.1): `cyt source {plan,run} · cyt embed · cyt index
{build,apply,promote} · cyt gate · cyt dump · cyt apply · cyt serve`. Verbs are
added one at a time as their module moves into the package, alongside the
shim at its old `pipeline/`/`tasks/` path (task 643 §6.2's "no behaviour
change" contract). Each verb's own module owns its argument parsing (it is
also runnable standalone, e.g. `python -m cytir.distribute.dump`) — this file's
subparsers are thin dispatch, not a second copy of any verb's flags.
"""
from __future__ import annotations

import argparse
import sys

from . import __version__


def _version_text() -> str:
    """`cyt X.Y` plus where its `data/` files come from (task 733: a stranger's install reads
    package data, a checkout reads `data/`; the line says which)."""
    from .core.paths import data_dir_line
    return f"cyt {__version__}\n{data_dir_line()}"


def _prog_name() -> str:
    """The command name shown in usage lines: always `cyt` (the only console script; task 2982)."""
    return "cyt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_prog_name(),
        formatter_class=argparse.RawDescriptionHelpFormatter,  # --version keeps its 2 lines
        description="Cytir -- an open web search engine and open web search index "
                     "whose primary user is an AI agent.",
    )
    parser.add_argument("--version", action="version", version=_version_text())
    subparsers = parser.add_subparsers(dest="verb")

    dump_parser = subparsers.add_parser(
        "dump", help="export a Qdrant collection (twin only) into the "
                     "public dump format -- specs/dump-format.md",
        add_help=False)
    dump_parser.set_defaults(_verb_main=_dump_main)

    apply_parser = subparsers.add_parser(
        "apply", help="apply a public dump or delta pack to your own "
                      "Qdrant -- specs/dump-format.md §7",
        add_help=False)
    apply_parser.set_defaults(_verb_main=_apply_main)

    rehearse_parser = subparsers.add_parser(
        "rehearse", help="release rehearsal on the sealed fixture: `cyt rehearse "
                         "run` (dump, sign, verify, tamper, apply, query; the CI "
                         "step runs this same verb), `cyt rehearse build-fixture` "
                         "(twin only, read-only)",
        add_help=False)
    rehearse_parser.set_defaults(_verb_main=_rehearse_main)

    torrent_parser = subparsers.add_parser(
        "torrent", help="deterministic .torrent of a dump/pack directory from its "
                        "manifest's file list: `cyt torrent make DIR`, `cyt torrent "
                        "verify FILE DIR`",
        add_help=False)
    torrent_parser.set_defaults(_verb_main=_torrent_main)

    source_parser = subparsers.add_parser(
        "source", help="run or plan a source pack -- `cyt source run PACK "
                       "[--plan|--dry-run|...]`, `cyt source plan PACK` "
                       "(ADDING-A-SOURCE.md); `cyt source hn-freshness` ingests the HN freshness window; "
                       "`cyt source tech-product {frontier,consent,pages}` "
                       "builds the technical-manufacturer product frontier",
        add_help=False)
    source_parser.set_defaults(_verb_main=_source_main)

    embed_parser = subparsers.add_parser(
        "embed", help="embed text: `cyt embed shards` / `cyt embed stage` write the "
                      "sealed full-width vector files the promotion path reads; "
                      "`cyt embed models [--check]` fetches and verifies the pinned model "
                      "files (`cyt embed index` is retired: it names `cyt index build`)",
        add_help=False)
    embed_parser.set_defaults(_verb_main=_embed_main)

    serve_parser = subparsers.add_parser(
        "serve", help="start the API/MCP server on a named collection by configuration "
                      "only; refuses a collection that is not its declared serving profile "
                      "and writes a start receipt: `cyt serve --collection NAME --qdrant URL`",
        add_help=False)
    serve_parser.set_defaults(_verb_main=_serve_main)

    read_parser = subparsers.add_parser(
        "read", help="read a source URL as text on this machine (JavaScript pages via an installed browser): "
                     "`cyt read URL [--read-url U] [--max-chars N] [--renderer R]` prints JSON",
        add_help=False)
    read_parser.set_defaults(_verb_main=_read_main)

    ops_parser = subparsers.add_parser(
        "ops", help="maintainer operations: `cyt ops cut --out DIR` exports the public cut "
                    "(data/public-cut.v1.json) and refuses a new leak",
        add_help=False)
    ops_parser.set_defaults(_verb_main=_ops_main)

    consent_parser = subparsers.add_parser(
        "consent", help="`cyt consent read-page URL... --out DIR` records a licence/terms page read (--quote REGEX prints the first matching sentence); "
                        "`cyt consent seal` writes the sealed per-host probe file the contracts "
                        "pin (run on the corpus box); consent sweeps: "
                        "`cyt consent sweep {run,verify,drive,watchdog}` "
                        "(re-resolution runner, its verifier, the resume-and-verify driver, "
                        "the relaunch watchdog); the runner probes live origins",
        add_help=False)
    consent_parser.set_defaults(_verb_main=_consent_main)

    news_parser = subparsers.add_parser(
        "news", help="news tools: `cyt news enrich-tier1 IN.jsonl HOST_META.json OUT.jsonl` stamps country and "
                     "cited licence on projection rows; `cyt news seal-source-evidence` turns one sealed "
                     "news arm into a "
                     "tier-2 content-pack `sources[]` entry (writes receipts under --out); "
                     "`cyt news stage` (fetch -> consent gate -> "
                     "staging collections) and `cyt news delta` (staged arms vs the canonical "
                     "twin -> promote delta; inputs come from the CYTIR_* environment); `cyt news "
                     "truncated-cc-recovery` recovers legacy-truncated news HTML from pinned "
                     "Common Crawl WARC records into a preserved stage (fetches CC, rate <= 4/s); "
                     "`cyt news routine-apply [--dry-run]` drives one sealed pack through the armed "
                     "ADR-098 transaction under the ADR-043 am.4 daily gate (CYTIR_* environment)",
        add_help=False)
    news_parser.set_defaults(_verb_main=_news_main)

    chunk_parser = subparsers.add_parser(
        "chunk", help="chunkers: `cyt chunk hn-discussion` splits HN discussion records into chunk.generic chunks; "
                      "`cyt chunk reference-select` classifies one markdown page as chunk.reference or "
                      "chunk.generic under a selector artifact (read-only); `cyt chunk law-sub --self-check` "
                      "cuts the bundled law-article fixtures with the chunk.law.sub policy and checks the "
                      "conservation contract",
        add_help=False)
    chunk_parser.set_defaults(_verb_main=_chunk_main)

    index_parser = subparsers.add_parser(
        "index", help="`cyt index build` creates a collection from the served profile and applies "
                      "sealed packs to it; `cyt index apply` adds further packs to an existing "
                      "collection; `cyt index b1-rebuild --mode {slice,full}` runs the b1 corpus-replay builder; `cyt index assemble` runs the ADR-101 binder (five steps); `cyt index delta-journal {init,record,verify,replay}` keeps the chunk@2 post-cutoff delta journal and replays it into a candidate; `cyt index rebuild {status,plan,run}` reconstructs a collection from one "
                      "sealed production manifest; `cyt index surface-delta` applies a re-surfaced pack as an ADR-083 delta; `cyt index promote` runs the hourly drain's FUNCTION "
                      "(promote_queue.main), NOT its unit: it does not take the unit's flock, "
                      "reset the checkout to origin/main or set the budget and throttle flags, and "
                      "on the configured LIVE queue it REFUSES (start the unit: "
                      "`systemctl start aw-promotion-nightly.service`) and runs on any other queue",
        add_help=False)
    index_parser.set_defaults(_verb_main=_index_main)

    gate_parser = subparsers.add_parser(
        "gate", help="score a pinned gold slice against a running `cyt serve`, over HTTP, with "
                     "the ADR-043 per-gold scorer; pass = no gold ranks worse than in the "
                     "baseline verdict: `cyt gate --api URL --slice FILE [--baseline FILE]`. "
                     "`cyt gate gold-presence --qdrant URL --collection C --control-url U` is the coverage "
                     "gate (a gold the incumbent holds and the candidate lacks; exit 0/2/3); "
                     "`cyt gate serve-parity {prepare,run}` compares two source trees' rankings on "
                     "the twin (a serving-code move must change nothing); `cyt gate serve-parity "
                     "check` exits 0 only for a PASS report of exactly (prod sha, new sha) "
                     "(`release-serving.sh` calls it); `cyt gate news-daily-slice [--check]` freezes/verifies the ADR-043 daily-gate slice; `cyt gate p0-eval` scores hit-rate@k of an eval "
                     "yaml against a running API (search queries only); `cyt gate content-guard --baseline F "
                     "--candidate F` is the catch-up waiver's dense-leg content guard (exit 0 pass, 1 stop, "
                     "2 unknown); `cyt gate flip-config --url U [--rollback]` reads the running api's /health back "
                     "against the ratified flip config (flip-day readback, task 2152); "
                     "`cyt gate qhead {--out F | --compare OFF ON}` is the ADR-062 AW_QHEAD byte-identical gate; `cyt gate promote-verify --qdrant-url U --collection C [--api URL]` is the "
                     "ADR-066 promote gate (tier 1 structural, tier 2 ranking; exit 0/1)",
        add_help=False)
    gate_parser.set_defaults(_verb_main=_gate_main)

    corpus_parser = subparsers.add_parser(
        "corpus", help="corpus facet tools: `cyt corpus facet-compact` compacts canonical append-only "
                       "facet event waves into the current-state parquet; `cyt corpus "
                       "card-content-dry-run` prepares entity-card content enrichments and their "
                       "event wave (prepares only: it never fetches, embeds or applies); `cyt corpus "
                       "card-website` captures Common-Crawl-first entity-card website snapshots",
        add_help=False)
    corpus_parser.set_defaults(_verb_main=_corpus_main)

    extract_parser = subparsers.add_parser(
        "extract", help="source-record projections: `cyt extract bfs-dam-current` projects the "
                        "exact current BFS DAM records from the sealed official catalog; "
                        "`cyt extract loda-current` projects exact current LODA article IDs from "
                        "sealed DILA LEGI XML (reads sealed artifacts only; never fetches)",
        add_help=False)
    extract_parser.set_defaults(_verb_main=_extract_main)

    retune_parser = subparsers.add_parser(
        "retune", help="the retune family: `cyt retune emit` (fusion weights candidate + report, never deploys), "
                       "`cyt retune policy-train` (typed query-policy head -> candidate-only qhead; ADR-036/084), "
                       "`cyt retune rankhead-train` (ADR-036 trust head -> candidate-only head file), "
                       "`cyt retune qhead-scope-train` (country_scope head -> candidate-only qhead)",
        add_help=False)
    retune_parser.set_defaults(_verb_main=_retune_main)

    bench_parser = subparsers.add_parser(
        "bench", help="benchmark tools: `cyt bench scope {probe,report}` scores an external benchmark run in "
                      "two scope-separated halves (in-scope hit rate and out-of-scope share; task 1026)",
        add_help=False)
    bench_parser.set_defaults(_verb_main=_bench_main)

    return parser


def _bench_main(argv: list[str]) -> int:
    """`cyt bench scope ...` == `python -m cytir.bench.scope ...` (task 2546; cytir.corpus.l0_census is reached through it)."""
    if argv[:1] == ["scope"]:
        from cytir.bench.scope import main as scope_main
        return int(scope_main(argv[1:]) or 0)
    print("usage: cyt bench scope {probe,report} [options]", file=sys.stderr)
    return 2 if argv[:1] not in (["-h"], ["--help"]) else 0


def _retune_main(argv: list[str]) -> int:
    """`cyt retune emit ...` == `python -m cytir.retune.emit ...`; `cyt retune policy-train ...` trains the
    candidate-only query-policy head (cytir.retune.policy_train; refuses to overwrite a base or live qhead)."""
    if argv[:1] == ["emit"]:
        from cytir.retune.emit import main as emit_main
        return int(emit_main(argv[1:]) or 0)
    if argv[:1] == ["policy-train"]:
        from cytir.retune.policy_train import run as train_run
        return int(train_run(argv[1:], prog="cyt retune policy-train") or 0)
    if argv[:1] == ["rankhead-train"]:
        from cytir.retune.rankhead_train import run as rankhead_run
        return int(rankhead_run(argv[1:], prog="cyt retune rankhead-train") or 0)
    if argv[:1] == ["qhead-scope-train"]:
        from cytir.retune.qhead_scope_train import run as scope_run
        return int(scope_run(argv[1:], prog="cyt retune qhead-scope-train") or 0)
    print("usage: cyt retune {emit,policy-train,rankhead-train,qhead-scope-train} [options]", file=sys.stderr)
    return 2 if argv[:1] not in (["-h"], ["--help"]) else 0


def _extract_main(argv: list[str]) -> int:
    """`cyt extract bfs-dam-current ...` == `python -m cytir.extract.bfs_dam_current ...` (task 1884);
    `cyt extract loda-current ...` == `python -m cytir.extract.loda_current ...` (task 1883)."""
    if argv[:1] == ["bfs-dam-current"]:
        from cytir.extract.bfs_dam_current import main as bfs_main
        return int(bfs_main(argv[1:], prog="cyt extract bfs-dam-current") or 0)
    if argv[:1] == ["loda-current"]:
        from cytir.extract.loda_current import main as loda_main
        return int(loda_main(argv[1:], prog="cyt extract loda-current") or 0)
    if argv[:1] == ["wikipedia-dump"]:   # task 2360
        from cytir.extract.wikipedia_dump import main as wiki_main
        return int(wiki_main(argv[1:], prog="cyt extract wikipedia-dump") or 0)
    print("usage: cyt extract {bfs-dam-current,loda-current,wikipedia-dump} [options]", file=sys.stderr)
    return 2 if argv[:1] not in (["-h"], ["--help"]) else 0


def _corpus_main(argv: list[str]) -> int:
    """`cyt corpus facet-compact ...` == `python -m cytir.corpus.facet_cli ...`;
    `cyt corpus card-content-dry-run ...` == `python -m cytir.corpus.card_content_dry_run ...` (task 1836)."""
    if argv[:1] == ["facet-compact"]:
        from cytir.corpus.facet_cli import main as compact_main
        return int(compact_main(argv[1:], prog="cyt corpus facet-compact") or 0)
    if argv[:1] == ["card-content-dry-run"]:
        from cytir.corpus.card_content_dry_run import main as dry_run_main
        return int(dry_run_main(argv[1:], prog="cyt corpus card-content-dry-run") or 0)
    if argv[:1] == ["card-website"]:
        from cytir.corpus.card_website_capture import main as card_website_main
        return int(card_website_main(argv[1:], prog="cyt corpus card-website") or 0)
    print("usage: cyt corpus {facet-compact,card-content-dry-run,card-website} [options]", file=sys.stderr)
    return 2 if argv[:1] not in (["-h"], ["--help"]) else 0


def _news_main(argv: list[str]) -> int:
    """`cyt news enrich-tier1 ...` == `python -m cytir.news.enrich_tier1 ...` (task 1948);
    `cyt news truncated-cc-recovery ...` == `python -m cytir.news.truncated_cc_recovery ...` (task 1950);
    `cyt news seal-source-evidence ...` == `python -m cytir.news.seal_source_evidence ...` (task 1949);
    `cyt news routine-apply [--dry-run]` == `python -m cytir.news.routine_apply` (task 2379);
    `cyt news stage|delta` == the live refresh chain (task 1989)."""
    if argv[:1] == ["enrich-tier1"]:
        from cytir.news.enrich_tier1 import main as enrich_main
        return int(enrich_main(argv[1:], prog="cyt news enrich-tier1") or 0)
    if argv[:1] == ["seal-source-evidence"]:
        from cytir.news.seal_source_evidence import main as seal_main
        return int(seal_main(argv[1:], prog="cyt news seal-source-evidence") or 0)
    if argv[:1] == ["truncated-cc-recovery"]:
        from cytir.news.truncated_cc_recovery import main as recovery_main
        return int(recovery_main(argv[1:], prog="cyt news truncated-cc-recovery") or 0)
    if argv[:1] == ["page-date-refetch"]:   # task 2360
        from cytir.news.page_date_refetch import main as refetch_main
        return int(refetch_main(argv[1:], prog="cyt news page-date-refetch") or 0)
    if argv[:1] == ["routine-apply"]:
        from cytir.news.routine_apply import main as routine_main
        return int(routine_main(argv[1:], prog="cyt news routine-apply") or 0)
    from cytir.news.refresh_chain import main as chain_main
    return chain_main(argv, prog="cyt news")


def _gate_main(argv: list[str]) -> int:
    """`cyt gate ...` == `python -m cytir.gates.slice_gate ...`;
    `cyt gate gold-presence ...` == `python -m cytir.gates.gold_presence ...`;
    `cyt gate content-guard ...` == `python -m cytir.gates.content_guard ...` (the catch-up waiver content guard, task 1888);
    `cyt gate serve-parity {prepare,run,check} ...` == `python -m cytir.gates.serve_parity ...` (S10's code-move
    gate on the twin, task 752)."""
    if argv[:1] == ["serve-parity"]:
        from cytir.gates import serve_parity
        return int(serve_parity.main(argv[1:]) or 0)
    if argv[:1] == ["gold-presence"]:
        from cytir.gates import gold_presence
        return int(gold_presence.main(argv[1:]) or 0)
    if argv[:1] == ["flip-config"]:
        from cytir.gates import flip_config_verify
        return int(flip_config_verify.main(argv[1:], prog="cyt gate flip-config") or 0)
    if argv[:1] == ["p0-eval"]:
        from cytir.gates import p0_eval
        return int(p0_eval.main(argv[1:], prog="cyt gate p0-eval") or 0)
    if argv[:1] == ["content-guard"]:
        from cytir.gates import content_guard
        return int(content_guard.main(argv[1:], prog="cyt gate content-guard") or 0)
    if argv[:1] == ["news-daily-slice"]:
        from cytir.gates import news_daily_slice
        return int(news_daily_slice.main(argv[1:], prog="cyt gate news-daily-slice") or 0)
    if argv[:1] == ["stamp-harness-weights"]:   # task 2383 (S5 W8)
        from cytir.gates import stamp_harness_weights
        return int(stamp_harness_weights.main(argv[1:], prog="cyt gate stamp-harness-weights") or 0)
    if argv[:1] == ["legs"]:   # task 2383 (S5 W8): the flip gate's leg reads, `cyt gate legs {prod,scratch}`
        import importlib
        sub = {"prod": "cytir.gates.prod_legs", "scratch": "cytir.gates.scratch_legs"}
        if argv[1:2] and argv[1] in sub:
            return int(importlib.import_module(sub[argv[1]]).main(argv[2:], prog=f"cyt gate legs {argv[1]}") or 0)
        print("usage: cyt gate legs {prod,scratch} [options]", file=sys.stderr)
        return 2 if argv[1:2] not in (["-h"], ["--help"]) else 0
    if argv[:1] == ["qhead"]:   # task 2489: the ADR-062 byte-identical AW_QHEAD gate (ADR-054 persisted-script discipline)
        from cytir.gates import gate_qhead
        return int(gate_qhead.main(argv[1:], prog="cyt gate qhead") or 0)
    if argv[:1] == ["promote-verify"]:
        from cytir.gates import promote_verify
        return int(promote_verify.main(argv[1:], prog="cyt gate promote-verify") or 0)
    from cytir.gates.slice_gate import main as gate_main
    return int(gate_main(argv, prog="cyt gate") or 0)


def _read_main(argv: list[str]) -> int:
    """`cyt read ...` == `python -m cytir.reader.cli ...` (task 2710)."""
    from cytir.reader.cli import main as read_main
    return int(read_main(argv) or 0)


def _serve_main(argv: list[str]) -> int:
    """`cyt serve ...` == `python -m cytir.ops.serve ...`."""
    from cytir.ops.serve import run as serve_run
    return int(serve_run(argv) or 0)


# task 2361 (S5 W5): operator tools behind doors. apply-delta WRITES to the --qdrant-url/--collection you name (an
# offline aw_vN clone); restore-pack only READS collections and composes a pack dir (`compose` writes
# under --queue-dir); landing-check, verify-supersede, qdrant-hop-verify read.
_INDEX_TOOL_VERBS = {
    "apply-delta": "cytir.index.apply_delta",
    "verify-supersede": "cytir.index.verify_supersede_by_id",
    "layout-check": "cytir.index.layout_check",   # task 2405 (S6): read-only, one GET
}
_OPS_TOOL_VERBS = {
    "landing-check": "cytir.ops.landing_check",
    "restore-pack": "cytir.ops.restore_pack",
    "qdrant-hop-verify": "cytir.ops.qdrant_hop_verify",
    "http-drift": "cytir.ops.http_drift",             # task 2383 (S5 W8)
    "prod-env-run": "cytir.ops.prod_env_run",
    "guards": "cytir.ops.guards",   # task 2390: every public-cut guard in one pre-push run
    "registry-census": "cytir.ops.registry_census",   # task 2494: run / merge / stars over the package-registry repo census
    "experiments-ledger": "cytir.ops.experiments_ledger",   # task 2516: validate data/experiments-ledger.v1.jsonl
    "github-docs-gold": "cytir.ops.github_docs_gold",       # task 2516: validate data/eval_github_docs_gold.v1.jsonl
    "agent-probe": "cytir.ops.agent_probe",                 # task 2684: agent-side probe through opencode (RULING 71)
}


def _index_main(argv: list[str]) -> int:
    """`cyt index build ...` / `cyt index apply ...` == `cytir.index.build` (the Single transaction of
    the drain); `cyt index promote ...` == `cytir.index.promote_queue.main` (twin-first by default,
    `--target-url/--target-collection` for the Single transaction). `promote` is the drain's
    function, not the hourly UNIT (its wrapper `promote_nightly.sh`): the unit's own
    flock, checkout reset, budget and throttle flags are not applied here (the gap is the unit's
    wrapper, named in the verb's help); two rules ARE enforced here: `PQ.refuse_live_queue_run`
    (the live queue, whatever the environment holds) and `PQ.hold_unit_lock` (a hand run takes the
    unit's lock when the box has one)."""
    if argv[:1] in (["apply-delta"], ["verify-supersede"], ["layout-check"]):   # task 2361 (S5 W5), 2405
        import importlib
        mod = importlib.import_module(_INDEX_TOOL_VERBS[argv[0]])
        return int(mod.main(argv[1:], prog=f"cyt index {argv[0]}") or 0)
    doors = ("build", "apply", "promote", "loda-resurface", "rebuild", "surface-delta", "pack-delta", "delta-journal", "b1-rebuild", "assemble")
    if not argv or argv[0] not in doors:
        print("usage: cyt index {build,apply,promote,apply-delta (writes),verify-supersede,layout-check,loda-resurface,rebuild,surface-delta,pack-delta,delta-journal,b1-rebuild,assemble} [options]  "
              "(options: see `cyt index <verb> --help`)", file=sys.stderr)
        return 2 if argv[:1] not in (["-h"], ["--help"]) else 0
    sub, rest = argv[0], argv[1:]
    if sub == "rebuild":             # task 1234: reconstruct a collection from one sealed manifest
        from cytir.index.rebuild_from_manifest import main as rebuild_main
        return int(rebuild_main(rest, prog="cyt index rebuild") or 0)
    if sub == "surface-delta":       # task 1822: the ADR-083 vector delta (held writer; no default target)
        from cytir.index.surface_delta import main as surface_delta_main
        return int(surface_delta_main(rest, prog="cyt index surface-delta") or 0)
    if sub == "pack-delta":          # task 2062: merge a staged native pack collection into a collection (held writer)
        from cytir.index.pack_delta import main as pack_delta_main
        return int(pack_delta_main(rest, prog="cyt index pack-delta") or 0)
    if sub == "loda-resurface":      # task 886: the E6 loda dense store for `aw_e10` (twin read only)
        from cytir.index.loda_resurface import main as resurface_main
        return int(resurface_main(rest, prog="cyt index loda-resurface") or 0)
    if sub == "b1-rebuild":          # task 2123: the b1 corpus-replay builder {--mode slice,--mode full}
        from cytir.index.b1_cli import main as b1_main
        return int(b1_main(rest, prog="cyt index b1-rebuild") or 0)
    if sub == "assemble":            # task 2118: the ADR-101 assembly binder {guards,plan,stage,load,verify}
        from cytir.index.assemble_cli import main as assemble_main
        return int(assemble_main(rest, prog="cyt index assemble") or 0)
    if sub == "delta-journal":       # task 1898: the chunk@2 post-cutoff delta journal + candidate replay
        from cytir.index.delta_journal import main as journal_main
        return int(journal_main(rest, prog="cyt index delta-journal") or 0)
    if sub == "promote":
        from cytir.index import promote_queue as PQ
        refusal = PQ.refuse_live_queue_run(rest)
        if refusal:
            print(f"[REFUSED] {refusal}", file=sys.stderr)
            return 2
        if any(a in ("-h", "--help") for a in rest):
            return int(PQ.main(rest) or 0)
        try:
            with PQ.hold_unit_lock():
                return int(PQ.main(rest) or 0)
        except PQ.UnitLockRefused as exc:
            print(f"[REFUSED] {exc}", file=sys.stderr)
            return 2
    from cytir.index.build import main as build_main
    return int(build_main(rest, mode=sub, prog=f"cyt index {sub}") or 0)


# Each verb regenerates a committed file (task 1074): authority_hosts.json, geo_gazetteer.json, labels_search_arena.jsonl,
# and the data/assembly manifests/claims.
_PRODUCER_VERBS = {
    "build-authority-table": "cytir.ops.build_authority_table",
    "build-gazetteer": "cytir.ops.build_gazetteer",
    "extract-arena-labels": "cytir.ops.extract_arena_labels",
    "bind-assembly-vector-rollups": "cytir.ops.bind_assembly_vector_rollups",
    "produce-assembly-artifacts": "cytir.ops.produce_assembly_artifacts",
    "scan-membership-assembly-facts": "cytir.ops.scan_membership_assembly_facts",
    "graduate-qhead-country": "cytir.ops.graduate_qhead_country",
    "eval-general-run": "cytir.ops.eval_general_run",              # task 1088; needs a served tree + judge key
    "general-pool-build": "cytir.ops.general_pool_build",          # task 1088; the candidate pool upstream of eval_general.yaml
    "scan-mechanism-callers": "cytir.ops.scan_mechanism_callers",  # task 1088; data/mechanism-callers.v1.json
    "assembly-guards": "cytir.ops.assembly_guards",
    "build-assembly-plan": "cytir.ops.build_assembly_plan",
    "fold-assembly-claim-fragments": "cytir.ops.fold_assembly_claim_fragments",
    "rii-dump-basis-backfill": "cytir.ops.rii_dump_basis_backfill",
    "extract-agentic-eval": "cytir.ops.extract_agentic_eval",
    "source-cards": "cytir.ops.source_cards",                       # task 1235 (port S2): build / embed / search the source-authority cards
    "lessons": "cytir.ops.lessons_ledger",                         # task 1873; validate data/lessons-ledger.v2.jsonl
    "pipeline-reach-census": "cytir.ops.pipeline_reach_census",     # task 1736; data/pipeline-reach-census.v1.json (--write / --check)
    # task 2357 (S5 W6): operator tools, each its own verb; no committed artifact
    "entry-doors": "cytir.ops.entry_doors",   # task 2894: every non-verb `main` is declared in data/entry-doors.v1.json (--check/--list)
    "writer-locks": "cytir.ops.writer_locks",   # task 2913: read-only stale-writer-lock check (--check)
    "docs-roster": "cytir.ops.docs_roster",
    "extras-smoke": "cytir.ops.extras_smoke",
    "family-census": "cytir.ops.family_census",
    "health-host-vet": "cytir.ops.health_host_vet",
    "tombstones-index-text": "cytir.ops.index_text_tombstones",   # ADR-123 pack
    "tombstones-spam": "cytir.ops.spam_tombstones",               # ADR-123 pack
}


# `cyt ops cc-tranche` (task 2355, S5 W3): subverb -> (module, argv prefix the module's main expects, one-line purpose)
CC_TRANCHE_VERBS: dict[str, tuple[str, list[str], str]] = {
    "outlinks": ("cytir.acquire.extract_outlinks", [], "endorsement graph: outlinks of curated seeds"),
    "roster": ("cytir.ops.cc_tranche_roster", [], "make a tranche roster from outlink candidates"),
    "roster-fix": ("cytir.ops.cc_tranche_roster_fix", [], "fix a never-queued roster in place"),
    "queue": ("cytir.ops.cc_tranche_queue", [], "split the never-queued roster into tranche queue files"),
    "delta": ("cytir.ops.cc_tranche_delta", [], "seal the delta of a tranche that kept growing"),
    "retry-domains": ("cytir.ops.cc_tranche_retry", [], "retry named domains of a sealed tranche"),
    "depth-roster": ("cytir.ops.cc_tranche_depth", ["roster"], "depth continuation: roster of capped hosts"),
    "depth-fetch": ("cytir.ops.cc_tranche_depth", ["fetch"], "depth continuation: fetch the new band"),
    "depth-gate": ("cytir.ops.cc_tranche_depth", ["gate"], "depth continuation: gate"),
}




def _ops_main(argv: list[str]) -> int:
    """`cyt ops cut ...` == `python -m cytir.ops.public_cut ...`; `cyt ops cc-tranche retry-domains ...` ==
    `python -m cytir.ops.cc_tranche_retry ...` (task 760)."""
    if argv[:1] == ["cc-tranche"]:         # tasks 760, 2194, 2355 (S5 W3): one door
        from cytir.ops.cc_tranche_door import main as cc_tranche_main
        return cc_tranche_main(CC_TRANCHE_VERBS, argv[1:])
    if argv[:2] == ["receipt", "backfill-distinct-urls"]:
        from cytir.ops.receipt_backfill import main as backfill_main
        return int(backfill_main(argv[2:], prog="cyt ops receipt backfill-distinct-urls") or 0)
    if argv[:1] == ["mirror-audit"]:       # task 837 (G3): read-only twin vs prod content audit
        from cytir.ops.mirror_audit import main as audit_main
        return int(audit_main(argv[1:], prog="cyt ops mirror-audit") or 0)
    if argv[:1] == ["payload-profile"]:    # task 1241 (S5a): payload-profile roundtrip / reconcile / published backfill
        from cytir.index.payload_profile_reconcile import main as profile_main
        return int(profile_main(argv[1:], prog="cyt ops payload-profile") or 0)
    if argv[:1] == ["snippet-repatch"]:    # task 2734: re-derive live snippets from text into a patch file
        from cytir.ops.snippet_repatch import main as repatch_main
        return int(repatch_main(argv[1:], prog="cyt ops snippet-repatch") or 0)
    if argv[:1] == ["twin-patch-fields"]:  # task 845 (T17): closed-key payload patch on TWIN rows by id
        from cytir.ops.twin_patch_fields import main as patch_main
        return int(patch_main(argv[1:], prog="cyt ops twin-patch-fields") or 0)
    if argv[:2] in (["github-admission", "roster"], ["github-admission", "licence-read"]):   # task 986 / 995
        from cytir.ops.github_admission import main as gh_main
        return int(gh_main(argv[1:], prog="cyt ops github-admission") or 0)
    if argv[:1] == ["build-persisted-index"]:    # task 1234: the one-time sqlite persisted-input index
        from cytir.corpus.persisted_build import main as persisted_build_main
        return int(persisted_build_main(argv[1:], prog="cyt ops build-persisted-index") or 0)
    if argv[:1] == ["check-qdrant-temp-path"]:   # task 1234: P18 monitor (snapshot staging volume)
        from cytir.ops.check_qdrant_temp_path import main as p18_main
        return int(p18_main(argv[1:], prog="cyt ops check-qdrant-temp-path") or 0)
    if argv[:1] == ["serve-split-map"]:    # task 1061 (S10): the serve split's def map + knob census
        from cytir.ops.serve_split_map import main as split_main
        return int(split_main(argv[1:]) or 0)
    if argv[:1] == ["serve-settings-table"]:   # task 1071 (S10 step 2): every env read -> its setting
        from cytir.ops.serve_settings_table import main as table_main
        return int(table_main(argv[1:]) or 0)
    if argv[:1] and argv[0] in _OPS_TOOL_VERBS:   # task 2361 (S5 W5)
        import importlib
        mod = importlib.import_module(_OPS_TOOL_VERBS[argv[0]])
        return int(mod.main(argv[1:], prog=f"cyt ops {argv[0]}") or 0)
    if argv[:1] and argv[0] in _PRODUCER_VERBS:   # task 1074: producers of committed artifacts
        import importlib
        mod = importlib.import_module(_PRODUCER_VERBS[argv[0]])
        return int(mod.main(argv[1:], prog=f"cyt ops {argv[0]}") or 0)
    if not argv or argv[0] != "cut":
        print("usage: cyt ops {" + ",".join([*_PRODUCER_VERBS, *_OPS_TOOL_VERBS]) + "} [options]   (producers of committed artifacts; see --help)\n"
              "       cyt ops cut {--out DIR | --check | --verify TREE | --tighten | --record-baseline | --admit-moved FROM TO}\n"
              "       cyt ops cc-tranche {outlinks,roster,roster-fix,queue,delta,retry-domains,depth-roster,depth-fetch,depth-gate} ...\n"
              "       cyt ops receipt backfill-distinct-urls PACK_DIR\n"
              "       cyt ops serve-split-map [--json | --write SPEC | --check SPEC]\n"
              "       cyt ops check-qdrant-temp-path [--container NAME] [--min-free-gb N] [--json]\n"
              "       cyt ops build-persisted-index --index-dir DIR [--registry-path F] [--root DIR]\n"
              "       cyt ops serve-settings-table --base REV [--out MD] [--json FILE] [--check]\n"
              "       cyt ops mirror-audit --class {inplace-today|inplace-since DATE|ids FILE|declared} [--plan]\n"
              "       cyt ops github-admission roster --out-dir DIR --index-repos FILE [--max-repos N]\n"
              "       cyt ops github-admission licence-read --roster FILE --out FILE [--summary FILE]\n"
              "       cyt ops payload-profile {roundtrip,reconcile,published,published-from-pages} [options]\n"
              "       cyt ops snippet-repatch --domain D --text-source pack=DIR|parquet=DIR --out DIR",
              "       cyt ops twin-patch-fields --from FILE --fields license[,rule_epochs,stamps,snippet] [--apply --export PATH]",
              file=sys.stderr)
        return 2
    from cytir.ops.public_cut import main as cut_main
    return int(cut_main(argv[1:], prog="cyt ops cut") or 0)


def _chunk_main(argv: list[str]) -> int:
    """`cyt chunk hn-discussion ...` (task 1873): the HN discussion chunker;
    `cyt chunk reference-select ...` == `python -m cytir.chunk.reference_selector ...` (task 1887)."""
    if argv[:2] == ["registry", "repoint-policy"]:   # task 2535: re-seal of a registered chunk policy
        from cytir.chunk.registry_repoint import repoint_main
        return int(repoint_main(argv[2:], prog="cyt chunk registry repoint-policy") or 0)
    if argv[:1] == ["hn-discussion"]:
        from cytir.chunk.hn_discussion import main as hn_main
        return int(hn_main(argv[1:], prog="cyt chunk hn-discussion") or 0)
    if argv[:1] == ["reference-select"]:
        from cytir.chunk.reference_selector import main as select_main
        return int(select_main(argv[1:], prog="cyt chunk reference-select") or 0)
    if argv[:1] == ["law-sub"]:
        from cytir.chunk.law_subchunk import main as law_sub_main
        return int(law_sub_main(argv[1:], prog="cyt chunk law-sub") or 0)
    print("usage: cyt chunk registry repoint-policy --old-sha OLD --new-sha NEW --reason R [--dry-run]\n"
          "       cyt chunk hn-discussion --in FILE --out FILE --chunk-policy FILE "
          "[--seal-root DIR [--seal-artifact-root DIR]]\n"
          "       cyt chunk reference-select [options]\n"
          "       cyt chunk law-sub [options]", file=sys.stderr)
    return 2 if argv[:1] not in (["-h"], ["--help"]) else 0


def _consent_main(argv: list[str]) -> int:
    """`cyt consent sweep run|verify|drive|watchdog`: the 427/479 consent scripts, ported (task 721)."""
    if argv and argv[0] == "seal":
        from cytir.consent.seal import SealRefused, main as seal_main
        try:
            return int(seal_main(argv[1:], prog="cyt consent seal") or 0)
        except SealRefused as exc:
            print(f"[REFUSED] {exc}", file=sys.stderr)
            return 2
    if argv and argv[0] == "read-page":    # task 2804: sealed licence/terms-page reads
        from cytir.consent.read_page import main as read_page_main
        return int(read_page_main(argv[1:], prog="cyt consent read-page") or 0)
    if argv and argv[0] == "backfill":     # task 1241 (S5a): the ADR-115 consent-flag backfill driver
        from cytir.consent.apply import main as backfill_main
        return int(backfill_main(argv[1:], prog="cyt consent backfill") or 0)
    doors = ("run", "verify", "drive", "watchdog")
    if len(argv) < 2 or argv[0] != "sweep" or argv[1] not in doors:
        print("usage: cyt consent seal --probe-box NAME ... | "
              "cyt consent read-page URL... --out DIR [--quote REGEX] | "
              "cyt consent backfill {twin,prod-sample,prod-plan} ... | "
              "cyt consent sweep {run,verify,drive,watchdog} [options]  "
              "(options: see `cyt consent sweep <verb> --help`)", file=sys.stderr)
        return 2 if argv[:1] not in (["-h"], ["--help"]) else 0
    verb, rest = argv[1], argv[2:]
    prog = f"cyt consent sweep {verb}"
    if verb == "run":
        from cytir.consent.sweep import main as run_main
        return int(run_main(rest, prog=prog) or 0)
    if verb == "verify":
        from cytir.consent.sweep_verify import main as verify_main
        return int(verify_main(rest, prog=prog) or 0)
    from cytir.consent.sweep_driver import main as driver_main
    return int(driver_main([verb, *rest], prog=prog) or 0)


def _embed_main(argv: list[str]) -> int:
    """`cyt embed index ...` == `python -m cytir.embed.collection_stage ...` (RETIRED: it
    refuses and names `cyt index build`); `cyt embed shards ...` == `python -m
    cytir.embed.shards ...` (sealed 1024-d fp16 files); `cyt embed stage ...` == the
    GPU-lock wrapper around shards (`python -m cytir.embed.stage ...`); `cyt embed models ...` ==
    `python -m cytir.embed.fetch_models ...` (the pinned model files)."""
    doors = ("index", "shards", "stage", "models")
    if not argv or argv[0] not in doors:
        print("usage: cyt embed {index,shards,stage,models} [options]  "
              "(options: see `cyt embed <verb> --help`)", file=sys.stderr)
        return 2 if argv[:1] not in (["-h"], ["--help"]) else 0
    sub, rest = argv[0], argv[1:]
    if sub == "index":
        from cytir.embed.collection_stage import main as index_main
        return int(index_main(rest, prog="cyt embed index") or 0)
    if sub == "models":
        from cytir.embed.fetch_models import main as models_main
        return int(models_main(rest, prog="cyt embed models") or 0)
    if sub == "shards":
        from cytir.embed.shards import main as shards_main
        return int(shards_main(rest) or 0)
    from cytir.embed.stage import main as stage_main
    return int(stage_main(rest) or 0)


# task 2360 (S5 W7): operator source tools, `cyt source <tool> [<step>] ...`; each module's
# `main(argv, prog)` builds its parser under the verb's own prog (door convention).
_SOURCE_TOOL_VERBS = {
    "eric-pdf": {"frontier": "cytir.sources.eric.frontier", "fetch": "cytir.sources.eric.pdf_fetch"},
    "stackoverflow": {"frontier": "cytir.sources.stackoverflow.frontier"},
    "openlegaldata-hf": {None: "cytir.acquire.openlegaldata_hf"},
}


def _source_tool_main(argv: list[str]) -> int:
    import importlib
    tool = _SOURCE_TOOL_VERBS[argv[0]]
    if None in tool:
        return int(importlib.import_module(tool[None]).main(argv[1:], prog=f"cyt source {argv[0]}") or 0)
    if argv[1:2] and argv[1] in tool:
        return int(importlib.import_module(tool[argv[1]]).main(
            argv[2:], prog=f"cyt source {argv[0]} {argv[1]}") or 0)
    print(f"usage: cyt source {argv[0]} {{{','.join(tool)}}} [options]", file=sys.stderr)
    return 2 if argv[1:2] not in (["-h"], ["--help"]) else 0


def _source_main(argv: list[str]) -> int:
    """`cyt source run ...` == `run_source.py ...`; `cyt source plan PACK` ==
    `run_source.py PACK --plan` (no network, no crawler identity); `cyt source plan FAMILY` ==
    `cyt source run FAMILY --dry-run` (task 1379: one verb, one meaning for a pack and a family)."""
    if argv[:1] == ["hn-freshness"]:
        from cytir.sources.hn.freshness_ingest import main as hn_freshness_main
        return int(hn_freshness_main(argv[1:], prog="cyt source hn-freshness") or 0)
    if argv[:1] == ["tech-product"]:
        return _tech_product_main(argv[1:])
    if argv[:1] == ["fedlex-law-meta"]:   # task 2535: re-seal of data/fedlex-law-names.v1.json
        from cytir.sources.fedlex_law_meta import main as fedlex_meta_main
        return int(fedlex_meta_main(argv[1:], prog="cyt source fedlex-law-meta") or 0)
    if argv[:1] == ["fedlex-versions"]:   # task 2709: the side table behind `read_url`
        from cytir.sources.fedlex_versions import main as fedlex_versions_main
        return int(fedlex_versions_main(argv[1:], prog="cyt source fedlex-versions") or 0)
    if argv[:1] == ["audit"]:   # task 2352 (S5 W4): one door to the per-family pack audits
        from cytir.sources.audit_door import main as audit_door_main
        return audit_door_main(argv[1:])
    if argv[:1] and argv[0] in _SOURCE_TOOL_VERBS:
        return _source_tool_main(argv)
    if argv[:1] == ["hn-discussion"]:   # task 1899: HN Firebase discussion ingest (live daily stage)
        from cytir.sources.hn.discussion_ingest import main as hn_discussion_main
        return int(hn_discussion_main(argv[1:], prog="cyt source hn-discussion") or 0)
    if not argv or argv[0] not in ("run", "plan", "families", "http-evidence", "bulk-finalize",
                                   "refresh-cadence", "refresh-new-refs", "refresh-periodic",
                                   "refresh-direct-compose", "audit", "fedlex-versions"):
        print("usage: cyt source {run,plan} PACK|FAMILY [options]  |  cyt source families\n"
              "       cyt source http-evidence --body F --headers F ...  (canonical curl evidence)\n"
              "       cyt source bulk-finalize {sec,companies-house,gleif} ...  (fail-closed ZIP finaliser)\n"
              "       cyt source tech-product {frontier,consent,pages} [options]\n"
              "       cyt source fedlex-law-meta reseal [--dry-run] [--write-artifact]  (Tim's re-seal)\n"
              "       cyt source fedlex-versions [--out F] [--elis-from F]  (read_url side table)\n"
              "       cyt source refresh-cadence [--check]\n"
              "       cyt source hn-freshness [options]\n"
              "       cyt source refresh-new-refs|refresh-periodic [--armed] [--now ISO] ...  (refresh lane)\n"
              "       cyt source refresh-direct-compose [--pack-id P] [--since-days N] ...  (refresh lane feed composer)\n"
              "       cyt source audit {hn,rfc-editor,service-public,routec,stackoverflow,context7,finma-partitions,tech-product} [options]\n"
              "       cyt source hn-discussion [options]  (HN Firebase discussion ingest)\n"
              "       cyt source eric-pdf {frontier,fetch} | stackoverflow frontier | openlegaldata-hf {acquire,verify}\n"
              "  plan FAMILY == run FAMILY --dry-run (discover + consent, nothing read or written)\n"
              "  PACK   a sources/**/*.yaml path or pack id (`cyt source run --help`)\n"
              "  FAMILY a registered source-family id (`cyt source families`), "
              "`cyt source run FAMILY --help`", file=sys.stderr)
        return 2 if argv[:1] not in (["-h"], ["--help"]) else 0
    if argv[0] == "refresh-cadence":     # task 1821: data/refresh-cadence.v1.json generator
        from cytir.sources.refresh_cadence import main as cadence_main
        return cadence_main(argv[1:])
    if argv[0] == "refresh-new-refs":    # task 1918: daily new-refs discovery lane
        from cytir.sources.refresh_direct.new_refs import main as new_refs_main
        return new_refs_main(argv[1:], "cyt source refresh-new-refs")
    if argv[0] == "refresh-periodic":    # task 1918: periodic re-fetch lane
        from cytir.sources.refresh_direct.periodic import main as periodic_main
        return periodic_main(argv[1:], "cyt source refresh-periodic")
    if argv[0] == "refresh-direct-compose":   # task 2116: feed/docs composer of the refresh lane
        from cytir.sources.refresh_direct.direct_compose import main as direct_compose_main
        return int(direct_compose_main(argv[1:], "cyt source refresh-direct-compose") or 0)
    if argv[0] == "refresh-retire-old-commit":   # task 2634: gh-repo-docs old-commit retire pack (prepare only)
        from cytir.sources.refresh_direct.retire_old_commit import main as retire_main
        return int(retire_main(argv[1:], "cyt source refresh-retire-old-commit") or 0)
    if argv[0] == "families":
        from cytir.sources.run_family import list_main
        return list_main(argv[1:])
    if argv[0] == "http-evidence":
        from cytir.sources.official_bulk.http_evidence import main as evidence_main
        return int(evidence_main(argv[1:]) or 0)
    if argv[0] == "bulk-finalize":
        return _bulk_finalize_main(argv[1:])
    if argv[0] in ("run", "plan") and len(argv) > 1 and not argv[1].startswith("-"):
        # a registered family id drives the source-family framework (task 683, S7); a pack
        # path or pack id keeps the pack runner. A path that exists is always a pack.
        # `plan FAMILY` is `run FAMILY --dry-run`: the dry run already is the plan.
        import os
        if not os.path.exists(argv[1]):
            from cytir.sources import registry
            if argv[1] in registry.registered():
                from cytir.sources.run_family import main as family_main
                return family_main(argv[1:] + (["--dry-run"] if argv[0] == "plan" else []))
    from cytir.sources.runner import main as runner_main
    sub, rest = argv[0], argv[1:]
    return int(runner_main(rest + (["--plan"] if sub == "plan" else [])) or 0)


def _bulk_finalize_main(argv: list[str]) -> int:
    """`cyt source bulk-finalize {sec,companies-house,gleif} ...` -- the fail-closed finaliser of
    one official bulk ZIP fetch (task 1870; the acquisition drivers call it)."""
    names = ("sec", "companies-house", "gleif")
    if argv and argv[0] in ("-h", "--help"):
        print("usage: cyt source bulk-finalize {sec,companies-house,gleif} [options]\n"
              "  `cyt source bulk-finalize NAME --help` lists that finaliser's flags")
        return 0
    if not argv or argv[0] not in names:
        print("usage: cyt source bulk-finalize {sec,companies-house,gleif} [options]",
              file=sys.stderr)
        return 2
    if argv[0] == "sec":
        from cytir.sources.official_bulk.finalize_sec import main as fin_main
    elif argv[0] == "companies-house":
        from cytir.sources.official_bulk.finalize_companies_house import main as fin_main
    else:
        from cytir.sources.official_bulk.finalize_gleif import main as fin_main
    return int(fin_main(argv[1:]) or 0)


def _tech_product_main(argv: list[str]) -> int:
    """`cyt source tech-product {frontier,pages,consent} ...` (task 1871): the technical-manufacturer
    product-frontier steps, formerly three `pipeline/` scripts and a chain."""
    verbs = {"frontier": ("cytir.sources.technical_manufacturer.sitemap_frontier", "main"),
             "pages": ("cytir.sources.technical_manufacturer.page_frontier", "main"),
             "consent": ("cytir.sources.technical_manufacturer.host_consent", "main")}
    if argv[:1] and argv[0] in verbs:
        import importlib
        mod = importlib.import_module(verbs[argv[0]][0])
        mod.main(argv[1:], prog=f"cyt source tech-product {argv[0]}")
        return 0
    print("usage: cyt source tech-product {frontier,consent,pages} [options]", file=sys.stderr)
    return 2 if argv[:1] not in (["-h"], ["--help"]) else 0


def _dump_main(argv: list[str]) -> int:
    if argv[:1] == ["shadow-compare"]:   # task 2603: ruling 68 (2) shadow period
        from cytir.distribute.shadow_compare import main as shadow_main
        return shadow_main(argv[1:])
    from cytir.distribute.dump import main as dump_main
    return dump_main(argv)


def _apply_main(argv: list[str]) -> int:
    from cytir.distribute.apply import main as apply_main
    return apply_main(argv)


def _torrent_main(argv: list[str]) -> int:
    from cytir.distribute.torrent import TorrentRefusal, main as torrent_main
    try:
        return torrent_main(argv, prog="cyt torrent")
    except TorrentRefusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1


def _rehearse_main(argv: list[str]) -> int:
    from cytir.distribute.rehearsal import main as rehearse_main
    return rehearse_main(argv, prog="cyt rehearse")


def main(argv: list[str] | None = None) -> int:
    import warnings
    # F10 (task 3008): two third-party notices that print on a stranger's first command and mean nothing to them.
    warnings.filterwarnings("ignore", message=r".*doesn't match a supported version.*")   # urllib3/requests PySocks etc.
    warnings.filterwarnings("ignore", message=r".*Payload indexes have no effect in the local Qdrant.*")
    parser = build_parser()
    # A verb owns everything after its own name -- parse only far enough to
    # find it, then hand the REST of argv to that verb's own parser (its
    # flags are not re-declared here; see this module's docstring).
    args, rest = parser.parse_known_args(argv)
    verb_main = getattr(args, "_verb_main", None)
    if verb_main is None:
        if getattr(args, "verb", None):
            parser.error(f"unknown verb: {args.verb!r}")
        # No verb given -- print help so a bare `aw` invocation is
        # informative rather than silently doing nothing.
        parser.print_help()
        return 0
    try:
        return verb_main(rest)
    except ModuleNotFoundError as e:
        hint = missing_extra_message(args.verb, e)
        if hint is None:
            raise
        print(f"REFUSED: {hint}", file=sys.stderr)
        return 1   # the refusal code of the other precondition refusals here (`cyt torrent`); 2 is argparse usage


#: Top-level module -> the distribution (as named in pyproject.toml) that ships it. Only a module
#: in this table gets the "install the extra" line; any other missing module re-raises. A test checks
#: every distribution here against the extras of pyproject.toml.
EXTRA_MODULES = {
    "qdrant_client": "qdrant-client", "httpx": "httpx", "yaml": "PyYAML", "fastapi": "fastapi",
    "uvicorn": "uvicorn", "mcp": "mcp", "pyarrow": "pyarrow", "duckdb": "duckdb",
    "sklearn": "scikit-learn", "trafilatura": "trafilatura", "bs4": "beautifulsoup4", "lxml": "lxml",
    "jsonschema": "jsonschema", "htmldate": "htmldate", "zstandard": "zstandard", "boto3": "boto3",
}
#: Which extras to try, first match wins, per verb (a verb's own extra first, then the rest).
_VERB_EXTRAS = {"read": ("reader", "dev"), "embed": ("pipeline", "serve", "dev"),
                "chunk": ("pipeline", "dev"), "extract": ("pipeline", "dev"),
                "news": ("pipeline", "dev"), "corpus": ("pipeline", "dev")}
_DEFAULT_EXTRAS = ("dev", "serve", "reader", "pipeline")


def _norm(dist_name: str) -> str:
    return dist_name.lower().replace("_", "-")


def _extra_dists(extra: str) -> set[str]:
    """Normalised distribution names an extra declares, from the installed package metadata."""
    import re
    from importlib import metadata
    out = set()
    for req in metadata.requires("cytir") or []:
        if f"extra == '{extra}'" in req.replace('"', "'"):
            out.add(_norm(re.split(r"[\s\[<>=!~;(]", req, maxsplit=1)[0]))
    return out


def missing_extra_message(verb: str, exc: ModuleNotFoundError) -> str | None:
    """One line for a verb whose DECLARED third-party module is not installed, else None (re-raise:
    a submodule of an installed package, a module no extra declares, or Cytir's own, is a bug)."""
    dist = EXTRA_MODULES.get(exc.name or "")   # a dotted name (`yaml.x`) never matches
    if dist is None:
        return None
    extra = None
    try:
        for cand in _VERB_EXTRAS.get(verb, _DEFAULT_EXTRAS):
            if _norm(dist) in _extra_dists(cand):
                extra = cand
                break
    except Exception:  # noqa: BLE001  (no package metadata, e.g. a bare checkout on PYTHONPATH)
        return None
    if extra is None:
        return None
    return (f'cyt {verb} needs the "{extra}" extra (missing module: {exc.name}): '
            f'uv tool install "cytir[{extra}]"  (or pip install "cytir[{extra}]" in a virtualenv)')


if __name__ == "__main__":
    raise SystemExit(main())
