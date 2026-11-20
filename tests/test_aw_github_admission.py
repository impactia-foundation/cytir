"""Tests for cytir.ops.github_admission (task 986): band splitting, the decision table, resumable roster. No network."""
import datetime as dt
import json

import pytest

from cytir.ops import github_admission as G

NOW = dt.date(2026, 10, 19)


def _item(name, stars, lic="MIT", pushed="2026-09-01T00:00:00Z", archived=False, fork=False):
    return {"full_name": name, "stargazers_count": stars, "pushed_at": pushed, "archived": archived,
            "fork": fork, "license": {"spdx_id": lic} if lic else None, "default_branch": "main"}


class FakeSearch:
    """Search universe: repos with stars; q parsed for the star range and created window (ignored: one date)."""

    def __init__(self, repos):
        self.repos, self.queries = repos, []

    def __call__(self, args):
        a = " ".join(args)
        if "search/repositories" in a:
            q = next(x for x in args if x.startswith("q="))[2:]
            page = 1
            for i, x in enumerate(args):
                if x.startswith("page="):
                    page = int(x[5:])
            rng = q.split()[0][len("stars:"):]
            lo, hi = (rng[2:], None) if rng.startswith(">=") else rng.split("..")
            lo, hi = int(lo), (int(hi) if hi else None)
            hit = [r for r in self.repos if r["stargazers_count"] >= lo and (hi is None or r["stargazers_count"] <= hi)]
            self.queries.append(q)
            body = {"total_count": len(hit), "incomplete_results": False, "items": hit[(page - 1) * 100:page * 100]}
            return 200, {}, json.dumps(body)
        return 404, {}, "{}"


def _gh(repos):
    fs = FakeSearch(repos)
    return G.Gh(run=fs, sleep=lambda s: None), fs


def test_bands_split_until_every_band_is_under_cap_and_nothing_is_lost():
    repos = [_item(f"o/r{i}", 5000 + i) for i in range(2500)]      # 2,500 repos in 5000..7499
    gh, fs = _gh(repos)
    items, log = G.enumerate_candidates(gh, now=NOW)
    assert len(items) == 2500
    assert all(b.get("fetched", 0) <= G.RESULT_CAP for b in log)
    assert all("warning" not in b for b in log)


def test_open_top_band_is_split():
    repos = [_item(f"o/t{i}", 20000 + i) for i in range(1500)]
    gh, _ = _gh(repos)
    items, _ = G.enumerate_candidates(gh, now=NOW)
    assert len(items) == 1500


def test_secondary_rate_limit_stops():
    gh = G.Gh(run=lambda a: (403, {}, '{"message":"You have exceeded a secondary rate limit"}'), sleep=lambda s: None)
    with pytest.raises(G.RateLimitStop):
        gh.call(["x"], "core")


def test_calls_are_spaced():
    sleeps, t = [], [100.0]
    gh = G.Gh(run=lambda a: (200, {}, "{}"), sleep=lambda s: (sleeps.append(s), t.__setitem__(0, t[0] + s)),
              clock=lambda: t[0])
    gh.call(["a"], "search")
    gh.call(["b"], "search")
    gh.call(["c"], "core")
    gh.call(["d"], "core")
    assert sleeps[0] == pytest.approx(G.SEARCH_DELAY) and sleeps[-1] == pytest.approx(G.CORE_DELAY)


@pytest.mark.parametrize("over,want", [
    ({}, ("admit", "permissive+docs-tree")),
    ({"archived": True}, ("refuse", "archived")),
    ({"pushed_at": "2024-01-01T00:00:00Z"}, ("refuse", "stale")),
    ({"fork": True}, ("refuse", "fork")),
    ({"license": None}, ("link-only", "licence-unknown")),
    ({"license": "NOASSERTION"}, ("link-only", "licence-unknown")),
    ({"license": "GPL-3.0"}, ("link-only", "copyleft-tagged")),
    ({"docs_files": 0}, ("refuse", "no-docs-tree")),
    ({"docs_files": 19}, ("refuse", "docs-tree-too-small")),
    ({"tree_truncated": True}, ("refuse", "tree-truncated")),
    ({"tree_status": 404}, ("refuse", "tree-unavailable")),
])
def test_decision_table(over, want):
    row = {"repo": "o/r", "archived": False, "fork": False, "pushed_at": "2026-09-01T00:00:00Z", "license": "MIT",
           "tree_status": 200, "tree_truncated": False, "docs_files": 20}
    row.update(over)
    assert G.decide(row, now=NOW) == want


def test_docs_tree_stats_counts_docs_dirs_not_root_files():
    nodes = [{"type": "blob", "path": "README.md", "sha": "a" * 40, "size": 5}] + [
        {"type": "blob", "path": f"docs/p{i}.md", "sha": "a" * 40, "size": 10} for i in range(3)]
    st = G.docs_tree_stats(nodes)
    assert st["docs_files"] == 3 and st["tree_files"] == 4 and st["tree_bytes"] == 35


def test_summary_invariant_and_estimate():
    rows = [{"repo": "a/a", "decision": "admit", "reason": "x", "license": "MIT", "docs_files": 30, "tree_files": 31,
             "in_index": True},
            {"repo": "b/b", "decision": "link-only", "reason": "licence-unknown", "license": None},
            {"repo": "c/c", "decision": "refuse", "reason": "no-docs-tree", "license": "MIT", "docs_files": 0}]
    s = G.summarize(rows)
    assert s["invariant_in_equals_sum"] and s["decisions"] == {"admit": 1, "link-only": 1, "refuse": 1}
    assert s["estimate"]["rows_median_1600"] == 1600 and s["in_index"] == 1


def test_roster_writes_files_and_resumes_after_stop(tmp_path):
    repos = [_item("o/a", 6000), _item("o/b", 7000, lic=None), _item("o/c", 8000)]
    tree = {"truncated": False, "tree": [{"type": "blob", "path": f"docs/p{i}.md", "sha": "a" * 40, "size": 9}
                                         for i in range(25)]}
    fs = FakeSearch(repos)
    state = {"trees": 0, "stop_at": 1}

    def run(args):
        a = " ".join(args)
        if "git/trees" in a:
            state["trees"] += 1
            if state["trees"] == state["stop_at"] + 1:
                return 403, {}, '{"message":"secondary rate limit"}'
            return 200, {}, json.dumps(tree)
        if args[0] == "graphql":
            return 200, {}, json.dumps({"data": {f"r{j}": {"defaultBranchRef": {"target": {"oid": "f" * 40}}}
                                                 for j in range(3)}})
        return fs(args)

    idx = tmp_path / "idx.txt"
    idx.write_text("o/a\n")

    class A:
        out_dir, index_repos, min_stars, max_repos, refresh = str(tmp_path / "o"), str(idx), 5000, 0, False

    gh = G.Gh(run=run, sleep=lambda s: None)
    assert G.roster(A, gh=gh, now=NOW) == 3                      # stopped mid-way, partial file kept
    state["stop_at"] = 999
    assert G.roster(A, gh=gh, now=NOW) == 0                      # resumes: no repeated search, no re-done repos
    rows = [json.loads(x) for x in (tmp_path / "o" / "phase1-roster.v1.jsonl").read_text().splitlines()]
    got = {r["repo"]: (r["decision"], r["in_index"], r["default_branch_sha"]) for r in rows}
    assert got["o/a"] == ("admit", True, "f" * 40) and got["o/b"][0] == "link-only" and got["o/c"][0] == "admit"
    assert json.loads((tmp_path / "o" / "phase1-summary.json").read_text())["decisions"] == {"admit": 2, "link-only": 1}


# ---- licence-file read (task 995): text matcher on sealed headers, decision rule on four rows. No network. ----

BSD3_FULL = ("Redistribution and use in source and binary forms, with or without modification, are permitted "
             "provided that the following conditions are met:\n1. Redistributions of source code must retain.\n"
             "2. Redistributions in binary form must reproduce.\n3. Neither the name of the copyright holder nor "
             "the names of its contributors may be used to endorse.\n")


@pytest.mark.parametrize("text, want", [
    ("MIT License\n\nCopyright (c) 2020 Example\n\nPermission is hereby granted, free of charge", "MIT"),
    ("                                 Apache License\n                           Version 2.0, January 2004",
     "Apache-2.0"),
    (BSD3_FULL, "BSD-3-Clause"),
    ("ISC License\n\nCopyright (c) 2020 Example\n\nPermission to use, copy, modify, and/or distribute", "ISC"),
    ("                    GNU GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007", "GPL"),
    ("Custom terms: you may look but not touch.", "unrecognised"),
])
def test_licence_text_matcher(text, want):
    assert G.match_licence_text(text) == want


@pytest.mark.parametrize("after, docs, read, want", [
    ("MIT", 25, "ok", ("admit", "permissive+docs-tree")),
    ("MIT", 19, "ok", ("link-only", "docs_tree_small")),
    ("GPL", None, "ok", ("link-only", "licence_not_permissive")),
    ("none", None, "none", ("link-only", "no_licence_file")),
])
def test_licence_decision_rule(after, docs, read, want):
    assert G.licence_decision(after, docs, read) == want
