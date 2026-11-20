"""Tests for cytir/ops/docs_roster.py (task 694): the roster is derived from registry rows, every row
reaches exactly one terminal, our own consent failures are never recorded as declines, and the
queue entry is what discovery_ingest reads."""
from __future__ import annotations

import json

from cytir.ops import docs_roster as D


def _pkg(registry, rank, name, doc=None, home=None, lic="MIT", repo_home=None):
    return {"registry": registry, "rank": rank, "name": name, "downloads": 1000 - rank,
            "documentation_url": doc, "homepage": home, "repo_homepage": repo_home,
            "licenses": lic}


def test_docs_target_folds_www_and_keeps_prefix_only_on_shared_hosts():
    assert D.docs_target("https://www.Example.org/docs/") == ("example.org", "/")
    assert D.docs_target("docs.python.org/3/library") == ("docs.python.org", "/")
    assert D.docs_target("https://learn.microsoft.com/en-us/dotnet/api/x") == \
        ("learn.microsoft.com", "/en-us/dotnet")
    assert D.docs_target("mailto:a@b.c") is None
    assert D.docs_target("") is None and D.docs_target(None) is None
    assert D.docs_target("https://localhost/x") is None


def test_licence_class_reads_dual_licence_and_never_guesses():
    assert D.licence_class("MIT OR Apache-2.0") == "permissive"
    assert D.licence_class("GPL-3.0-only") == "copyleft"
    assert D.licence_class("MIT OR GPL-2.0") == "permissive"       # a permissive arm is offered
    assert D.licence_class(None) == "unstated"
    assert D.licence_class("SomethingBespoke-1.0") == "other"


def test_homepage_is_docs_evidence_only_when_it_looks_like_docs():
    rows = [_pkg("pypi.org", 1, "marketingco", home="https://marketingco.com/"),
            _pkg("pypi.org", 2, "realdocs", home="https://realdocs.readthedocs.io/en/latest/"),
            _pkg("pypi.org", 3, "gh", home="https://github.com/o/gh"),
            _pkg("npmjs.org", 1, "viaexplicit", doc="https://viaexplicit.dev/docs")]
    hosts = {r["host"] for r in D.build_roster(rows)}
    assert "marketingco.com" not in hosts                          # vendor-content boundary
    assert "realdocs.readthedocs.io" in hosts and "viaexplicit.dev" in hosts
    gh = [r for r in D.build_roster(rows) if r["host"] == "github.com"]
    assert gh == []                                                # not a docs host at all


def test_build_groups_by_host_ranks_by_rank_decay_and_keeps_platforms_visible():
    rows = [_pkg("pypi.org", 1, "a", doc="https://docs.a.org/"),
            _pkg("npmjs.org", 1, "a-js", doc="https://docs.a.org/js"),
            _pkg("pypi.org", 2, "b", doc="https://b.dev/"),
            _pkg("pypi.org", 3, "c", doc="https://github.com/c/c/wiki"),
            _pkg("crates.io", 9, "d", doc="https://docs.rs/d")]
    roster = D.build_roster(rows, top_n=2)
    by = {r["host"]: r for r in roster}
    assert by["docs.a.org"]["n_packages"] == 2 and by["docs.a.org"]["usage_rank"] == 1
    assert by["docs.a.org"]["registries"] == ["npmjs.org", "pypi.org"]
    assert by["github.com"]["platform"] and by["docs.rs"]["platform"]
    assert [r["host"] for r in roster if not r["platform"]] == ["docs.a.org", "b.dev"]


def test_vendor_proprietary_docs_are_not_given_the_package_licence():
    rows = [_pkg("pypi.org", 1, "boto3", doc="https://docs.aws.amazon.com/boto3/latest/",
                 lic="Apache-2.0")]
    (r,) = D.build_roster(rows)
    assert r["licence_class"] == "proprietary_vendor" and r["path_prefix"] == "/boto3"
    assert "CODE licence" not in r["licence_basis"]


def test_consent_class_never_turns_our_403_into_a_decline():
    assert D.consent_class(None) == "unresolved"
    ok = {"status": "ok", "search_ok": True, "train_ok": True, "signals": {}}
    assert D.consent_class(ok) == "ok"
    ccbot = dict(ok, train_ok=False)
    assert D.consent_class(ccbot) == "search_only"                # train_ok only (ADR-024 am.5)
    forbidden = {"status": "ok", "search_ok": False, "train_ok": False,
                 "signals": {"robots": {"fetched": True, "http_status": 403}}}
    assert D.consent_class(forbidden) == "held_unreadable"
    unreachable = {"status": "ok", "search_ok": False, "train_ok": False,
                   "signals": {"robots": {"fetched": False}}}
    assert D.consent_class(unreachable) == "held_unreadable"
    held = {"status": "unknown", "search_ok": True, "train_ok": True, "signals": {}}
    assert D.consent_class(held) == "held_unresolved"
    disallow = {"status": "ok", "search_ok": False, "train_ok": False,
                "signals": {"robots": {"fetched": True, "http_status": 200}}}
    assert D.consent_class(disallow) == "declined_search"


def _row(host, **kw):
    base = {"host": host, "project": host, "path_prefix": "/", "scope": "host", "platform": False,
            "licence_class": "permissive", "licence_spdx": "MIT", "usage_score": 0.1,
            "usage_rank": 1, "consent_state": "ok", "cc_urls": 400, "served_points": 0,
            "served_docs_points": 0}
    base.update(kw)
    return base


def test_scope_separates_docs_hosts_from_mixed_and_root_sites():
    assert D.scope_of("docs.a.org", "/", "https://docs.a.org/", False) == ("host", "/")
    assert D.scope_of("x.readthedocs.io", "/", "https://x.readthedocs.io/en/latest/", False)[0] == "host"
    assert D.scope_of("lang.org", "/", "https://lang.org/", True) == ("host", "/")        # language site
    assert D.scope_of("mongodb.com", "/", "https://www.mongodb.com/docs/atlas/", False) == \
        ("path_prefix", "/docs")
    assert D.scope_of("plotly.com", "/", "https://plotly.com/python/", False) == \
        ("path_prefix", "/python")
    assert D.scope_of("networkx.org", "/", "https://networkx.org/", False) == ("site_root", "/")


def test_every_row_reaches_exactly_one_terminal_and_estimates_use_the_measured_ratio():
    cal = {"pooled": 5.0, "host_median": 2.0}
    roster = [_row("good.dev"),
              _row("agg.io", platform=True),
              _row("gpl.org", licence_class="copyleft"),
              _row("held.org", consent_state="held_unreadable"),
              _row("nope.org", consent_state="declined_search"),
              _row("tiny.org", cc_urls=3),
              _row("nocc.org", cc_urls=None),
              _row("prefixed.org", scope="path_prefix", path_prefix="/x"),
              _row("done.org", served_docs_points=5000),
              _row("fresh.org", consent_state=None),
              _row("root.org", scope="site_root"),
              _row("launched.dev", tranche="tranche6_docs_roster"),
              _row("huge.org", cc_urls=50_000)]
    D.rank_rows(roster, cal, cap=100)
    why = {r["host"]: r["ineligible_reason"] for r in roster}
    assert why == {
        "good.dev": None, "agg.io": "platform_aggregator", "gpl.org": "licence_class_copyleft",
        "held.org": "consent_held_unreadable", "nope.org": "consent_declined_search",
        "tiny.org": "cc_urls_below_floor", "nocc.org": "cc_not_counted",
        "prefixed.org": "path_prefix_scope_needs_prefix_fetch",
        "done.org": "already_served_to_estimate", "fresh.org": "consent_not_resolved",
        "root.org": "site_root_scope_unclassified", "launched.dev": "in_tranche6_docs_roster", "huge.org": "host_too_large_to_assume_docs_only"}
    good = next(r for r in roster if r["host"] == "good.dev")
    assert good["est_points"] == 500 and good["est_net_new"] == 500      # min(400,100) * 5
    assert good["eligible"] and sum(r["eligible"] for r in roster) == 1
    assert "conservation: in=13 eligible=1 ineligible=12 -> OK" in D.summarise(roster)


def test_queue_entry_has_what_discovery_ingest_reads():
    q = D.queue_entry(_row("good.dev", project="good"), cap=800)
    assert q["domain"] == "good.dev" and q["class"] == "tech"
    assert q["class_basis"] == "self-signal" and q["cap"] == 800 and q["n_endorser_regs"] == 0


def test_chunks_per_page_is_measured_from_rows(tmp_path):
    p = tmp_path / "passages.jsonl"
    rows = ([{"url": f"https://a.dev/p{i}"} for i in range(4) for _ in range(6)]
            + [{"url": "https://b.dev/p0"} for _ in range(2)])
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cal = D.chunks_per_page(p)
    assert cal["rows"] == 26 and cal["urls"] == 5 and cal["hosts"] == 2
    assert cal["pooled"] == 26 / 5 and cal["host_median"] in (2.0, 6.0)


def test_wikidata_licence_labels_map_to_spdx_and_unknown_labels_are_kept_raw():
    assert D.label_to_spdx("MIT License") == "MIT"
    assert D.label_to_spdx("Apache License 2.0 | MIT License") == "Apache-2.0 OR MIT"
    assert D.label_to_spdx("GNU General Public License") == "GPL-3.0"
    assert D.label_to_spdx("Some Bespoke Terms") == "Some Bespoke Terms"
    assert D.label_to_spdx(None) is None
    assert D.licence_class(D.label_to_spdx("Some Bespoke Terms")) == "other"
    rows = [{"registry": "wikidata", "rank": 1, "name": "Lua", "wd_class": "programming language",
             "homepage": "https://lua.example.org/", "licenses": "MIT License", "sitelinks": 90},
            {"registry": "wikidata", "rank": 2, "name": "Std", "wd_class": "technical standard",
             "homepage": "https://std.example.org/", "licenses": None, "sitelinks": 40}]
    by = {r["host"]: r for r in D.build_roster(rows)}
    assert by["lua.example.org"]["licence_class"] == "permissive"
    assert "P275" in by["lua.example.org"]["licence_basis"]
    assert by["std.example.org"]["licence_class"] == "unstated"      # never guessed


def test_rank_with_a_launched_queue_removes_those_hosts_from_the_eligible_set(tmp_path):
    roster = tmp_path / "roster.jsonl"
    queue = tmp_path / "tranche9_docs_roster.json"
    cal = tmp_path / "passages.jsonl"
    roster.write_text("".join(json.dumps(_row(h)) + "\n" for h in ("a.dev", "b.dev")))
    queue.write_text(json.dumps([D.queue_entry(_row("a.dev"), 10)]))
    cal.write_text("".join(json.dumps({"url": f"https://c.dev/p{i}"}) + "\n" for i in range(4)))
    assert D.main(["rank", "--roster", str(roster), "--calibration", str(cal),
                   "--queue", str(queue)]) == 0
    rows = {r["host"]: r for r in map(json.loads, roster.read_text().splitlines())}
    assert rows["a.dev"]["ineligible_reason"] == "in_tranche9_docs_roster"
    assert rows["b.dev"]["eligible"] and rows["b.dev"]["est_points"] == 400
