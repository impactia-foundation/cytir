"""The serve-parity gate's operator slice (task 2652): one request per operator path of /search,
the whole body (the `filters` echo included) compared, a 400 accepted where a 400 is the answer."""
from __future__ import annotations

import argparse
import re

from cytir.gates import serve_parity as sp
from cytir.serve.query import parse_operators

REQS = {r["id"].split(":", 1)[1]: r for r in sp.operator_requests()}
STABLE_HOSTS = {"fedlex.admin.ch", "admin.ch", "github.com", "stackoverflow.com", "wikipedia.org"}


def _q(rid):
    return REQS[rid]["params"]["q"]


def test_one_request_per_operator_path():
    assert "site:fedlex.admin.ch" in _q("site-exact") and "site:admin.ch" in _q("site-parent")
    assert _q("site-multi").count("site:") == 2
    assert "-site:" in _q("minus-site")
    assert "source:official" in _q("source") and _q("source-multi").count("source:") == 2
    assert "lang:de" in _q("lang") and "country:CH" in _q("country")
    c = _q("combo")
    assert all(t in c for t in ("site:admin.ch", "-site:wikipedia.org", "lang:fr", "country:CH"))
    assert REQS["param-site"]["params"]["site"] == ["fedlex.admin.ch"]
    assert REQS["param-exclude-site"]["params"]["exclude_site"] == ["wikipedia.org"]
    assert REQS["param-source"]["params"]["source"] == ["official"]
    assert "site:" in _q("param-and-operator") and REQS["param-and-operator"]["params"]["site"]
    assert "lang:fr" in _q("override-note") and REQS["override-note"]["params"]["lang"] == "de"
    assert REQS["authorities"]["params"]["authorities"] == 1
    assert "site:no-such-host.invalid" in _q("unknown-host")
    assert "source:zzznotasource" in _q("unknown-source")


def test_operator_only_queries_expect_400():
    for rid in ("only-operator", "only-operators"):
        assert REQS[rid]["expect_status"] == 400
        assert parse_operators(_q(rid))[0].strip() == ""
    assert all("expect_status" not in r for rid, r in REQS.items() if not rid.startswith("only-"))


def test_hosts_are_the_stable_ones():
    named = set()
    for r in REQS.values():
        named |= set(re.findall(r"site:(\S+)", r["params"]["q"]))
        named |= set(r["params"].get("site", ())) | set(r["params"].get("exclude_site", ()))
    assert named - {"no-such-host.invalid"} <= STABLE_HOSTS


def test_embed_text_is_what_the_server_embeds():
    """The vector cache is keyed on the cleaned text: a drift here is a cache miss, an ERROR."""
    for rid, r in REQS.items():
        if r.get("expect_status") == 400:
            continue
        clean, ops = parse_operators(r["params"]["q"])
        if rid == "unknown-source":   # the app puts an unknown source: value back (cytir/serve/app.py)
            clean = f"{clean} source:zzznotasource".strip()
        assert (r.get("embed_q") or r["params"]["q"]) == clean, rid


def test_slice_is_a_default_set_and_plain_sets_are_untouched():
    ns = argparse.Namespace(sets="operators", chooser="", fixture="", release="", probe_runs=None,
                            probe_days=7, limit=0)
    got = sp.collect(ns)
    assert [r["id"] for r in got] == [r["id"] for r in sp.operator_requests()]
    p = argparse.ArgumentParser()
    sp._add_sets(p)
    assert "operators" in p.parse_args([]).sets.split(",")
    # a plain request is byte-identical to before: q, k and the three old params only
    assert sp._req("x", "1", "abc", lang="de") == {"id": "x:1", "set": "x",
                                                   "params": {"q": "abc", "k": 10, "lang": "de"}}


def _ans(body, status=200):
    return {"status": status, "body": body}


def test_filters_echo_is_compared():
    base = {"query": "q", "results": [{"url": "https://a/"}],
            "filters": {"hosts": ["admin.ch"], "note": None}}
    dropped = {"query": "q", "results": [{"url": "https://a/"}]}        # filter gone, results kept
    other = {**base, "filters": {"hosts": ["github.com"], "note": None}}

    def seq(head):
        return {"base": _ans(base), "null": _ans(base), "head": _ans(head)}

    assert sp.classify(seq(base), 0.0)["state"] == "EQUAL"
    assert sp.classify(seq(dropped), 0.0)["state"] == "DIFF"
    d = sp.classify(seq(other), 0.0)
    assert d["state"] == "DIFF" and "filters" in d["diffs"][0]


def test_a_400_is_equal_when_expected_and_an_error_otherwise():
    body = {"error": "query has no terms besides the operators", "status": "bad_request"}
    seq = {k: _ans(body, 400) for k in ("base", "null", "head")}
    assert sp.classify(seq, 0.0, 400)["state"] == "EQUAL"
    assert sp.classify(seq, 0.0)["state"] == "ERROR"
    ok = {k: _ans({"results": []}) for k in ("base", "null", "head")}
    assert sp.classify(ok, 0.0, 400)["state"] == "ERROR"      # the 400 went missing
    assert sp.classify({**seq, "head": _ans(body, 200)}, 0.0, 400)["state"] == "ERROR"
