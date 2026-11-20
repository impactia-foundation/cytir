"""cytir.core.surtkey: the rule table, the sealed parity fixture, the run-start guard.

The key is a LOOKUP KEY on the Common Crawl index. A key that differs makes a
lookup miss, and a miss reads as "no capture", so this file is the proof that
`surt_key` gives Common Crawl's own `url_surtkey`:

* `test_rule_table` -- one row per documented rule (see the module docstring).
* `test_sealed_parity_fixture` -- every row of `fixtures/surtkey_parity.v1.jsonl`.
  Rows with `src == "cc"` are (url, url_surtkey) pairs read from Common Crawl's
  own columnar index: the ground truth. Rows with `src == "oracle"` are pairs
  the retired `surt` package produced for inputs the CC sample does not hold
  (rare shapes); they pin behaviour the package had.
* `test_parity_guard_*` -- the run-start guard that refuses a wrong key.
Running the differential against the package itself: `test_surtkey_oracle.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cytir.core import surtkey as subject

FIXTURE = Path(__file__).parent / "fixtures" / "surtkey_parity.v1.jsonl"

# (name, input, key). Inputs that are not URLs stay in: the key of a bad input
# must not change silently either.
RULES = [
    ("https_same_as_http", "https://example.com/a", "com,example)/a"),
    ("scheme_case", "HTTP://EXAMPLE.COM/A", "com,example)/a"),
    ("doubled_scheme", "https://https://example.com/a", "com,example)/a"),
    ("doubled_scheme_mixed", "https://http://www.example.com:8080/a", "com,example:8080)/a"),
    ("no_scheme_host_path", "example.com/x", "com,example)/x"),
    ("protocol_relative", "//example.com/x", "com,example)/x"),
    ("ftp_same", "ftp://example.com/x", "com,example)/x"),
    ("www_dropped", "http://www.example.com/x", "com,example)/x"),
    ("www_digits_dropped", "http://www12345.example.com/x", "com,example)/x"),
    ("www_dropped_once", "http://www.www.example.com/x", "com,example,www)/x"),
    ("www_without_dot_kept", "http://www/x", "www)/x"),
    ("www_before_tld", "http://www.com/x", "com)/x"),
    ("wwww_kept", "http://wwww.example.com/x", "com,example,wwww)/x"),
    ("ww2_kept", "http://ww2.example.com/x", "com,example,ww2)/x"),
    ("www_dash_kept", "http://www-1.example.com/x", "com,example,www-1)/x"),
    ("www_not_leading_kept", "http://sub.www.example.com/x", "com,example,www,sub)/x"),
    ("www_after_subdomain_dropped_leading", "http://www.sub.example.com/x", "com,example,sub)/x"),
    ("host_lowercased", "http://WWW.Example.COM/X", "com,example)/x"),
    ("deep_host_reversed", "http://a.b.c.d.example.co.uk/x", "uk,co,example,d,c,b,a)/x"),
    ("trailing_dot", "http://example.com./x", "com,example)/x"),
    ("leading_dot", "http://.example.com/x", "com,example)/x"),
    ("double_dot", "http://example..com/x", "com,example)/x"),
    ("underscore_kept", "http://exa_mple.com/x", "com,exa_mple)/x"),
    ("host_percent_decoded", "http://exa%6Dple.com/x", "com,example)/x"),
    ("bare_host", "example.com", "com,example)/"),
    ("bare_host_with_www", "www.example.com", "com,example)/"),
    ("http_80_dropped", "http://example.com:80/x", "com,example)/x"),
    ("https_443_dropped", "https://example.com:443/x", "com,example)/x"),
    ("other_port_kept", "http://example.com:8080/x", "com,example:8080)/x"),
    ("https_80_kept", "https://example.com:80/x", "com,example:80)/x"),
    ("http_443_kept", "http://example.com:443/x", "com,example:443)/x"),
    ("empty_port", "http://example.com:/x", "com,example)/x"),
    ("port_zero_dropped", "http://example.com:0/x", "com,example)/x"),
    ("port_leading_zeros", "http://example.com:00080/x", "com,example)/x"),
    ("port_with_www_and_dot", "http://www.example.com.:8080/x", "com,example:8080)/x"),
    ("userinfo_dropped", "http://user:pw@example.com/x", "com,example)/x"),
    ("userinfo_with_at", "http://user:p@ss@example.com/x", "com,example)/x"),
    ("userinfo_and_port", "http://user:pw@www.example.com:8080/x", "com,example:8080)/x"),
    ("at_in_path_kept", "http://example.com/x@y", "com,example)/x@y"),
    ("idn_host_and_path", "http://m\u00fcnchen.de/stra\u00dfe", "de,xn--mnchen-3ya)/stra%c3%9fe"),
    ("idn_sharp_s_folds", "http://fa\u00df.de/x", "de,fass)/x"),
    ("idn_uppercase", "http://EX\u00c4MPLE.com/x", "com,xn--exmple-cua)/x"),
    ("punycode_kept", "http://xn--mnchen-3ya.de/x", "de,xn--mnchen-3ya)/x"),
    ("idn_cjk_tld", "http://\u4e2d\u6587.\u4e2d\u56fd/x", "xn--fiqs8s,xn--fiq228c)/x"),
    ("idn_rejected_label_escaped", "http://\u00e9..com/x", "com,%c3%a9)/x"),
    ("ipv4", "http://192.168.1.1/x", "1,1,168,192)/x"),
    ("ipv4_port", "http://192.168.1.1:81/x", "1,1,168,192:81)/x"),
    ("ipv4_single_number", "http://2130706433/x", "1,0,0,127)/x"),
    ("ipv4_short_form", "http://127.1/x", "1,0,0,127)/x"),
    ("ipv4_octal_parts", "http://010.010.010.010/x", "8,8,8,8)/x"),
    ("ipv4_hex_untouched", "http://0x7f.0.0.1/x", "1,0,0,0x7f)/x"),
    ("ipv4_out_of_range_untouched", "http://256.1.1.1/x", "1,1,1,256)/x"),
    ("ipv6", "http://[::1]/x", "::1)/x"),
    ("ipv6_port", "http://[2001:db8::1]:8080/x", "2001:db8::1:8080)/x"),
    ("ipv6_default_port_dropped", "http://[::1]:80/x", "::1)/x"),
    ("empty_path_is_slash", "https://example.com", "com,example)/"),
    ("root_slash", "https://example.com/", "com,example)/"),
    ("trailing_slash_dropped", "http://example.com/a/", "com,example)/a"),
    ("trailing_slash_dropped_with_query", "http://example.com/a/?x=1", "com,example)/a?x=1"),
    ("double_slash_collapsed", "http://example.com//a//b", "com,example)/a/b"),
    ("dot_segments", "http://example.com/a/./b/../c", "com,example)/a/c"),
    ("unresolvable_dotdot_stays", "http://example.com/../b", "com,example)/../b"),
    ("dotdot_pairs_cancel", "http://a.com/../../b", "com,a)/b"),
    ("path_lowercased", "http://example.com/Path/A", "com,example)/path/a"),
    ("index_html_kept", "http://example.com/index.html", "com,example)/index.html"),
    ("backslash_kept", "http://example.com/a\\b", "com,example)/a\\b"),
    ("semicolon_params_kept", "http://example.com/a;b=c/d", "com,example)/a;b=c/d"),
    ("space_escaped", "http://example.com/a b", "com,example)/a%20b"),
    ("percent_20_kept", "http://example.com/a%20b", "com,example)/a%20b"),
    ("unreserved_unescaped", "http://example.com/%7Euser", "com,example)/~user"),
    ("encoded_slash_decoded", "http://example.com/a%2Fb", "com,example)/a/b"),
    ("double_encoded_decoded", "http://example.com/%2541", "com,example)/a"),
    ("lone_percent_escaped", "http://example.com/%", "com,example)/%25"),
    ("bad_hex_percent_escaped", "http://example.com/%zz", "com,example)/%25zz"),
    ("hash_escaped", "http://example.com/%23x", "com,example)/%23x"),
    ("utf8_literal", "http://example.com/\u4f60\u597d?q=\u4f60", "com,example)/%e4%bd%a0%e5%a5%bd?q=%e4%bd%a0"),
    ("utf8_percent_lowercased", "http://example.com/%E4%BD%A0", "com,example)/%e4%bd%a0"),
    ("non_ascii_upper_not_folded", "http://example.com/\u00c4", "com,example)/%c3%84"),
    ("invalid_utf8_kept", "http://example.com/%ff", "com,example)/%ff"),
    ("control_char_escaped", "http://example.com/a%00b", "com,example)/a%00b"),
    ("tab_deleted", "http://example.com/a\tb", "com,example)/ab"),
    ("fragment_dropped", "http://example.com/a#frag", "com,example)/a"),
    ("fragment_with_query", "http://example.com/a?x=1#y?z=2", "com,example)/a?x=1"),
    ("hashbang_dropped", "http://example.com/#!/x", "com,example)/"),
    ("query_sorted_and_lowercased", "http://www.Example.COM/Path/A?b=2&a=1&C=3#frag", "com,example)/path/a?a=1&b=2&c=3"),
    ("duplicate_keys_sorted_by_value", "http://example.com/a?b=2&a=1&a=0", "com,example)/a?a=0&a=1&b=2"),
    ("name_sorts_before_name_dash", "http://example.com/a?a=2&a-b=1&a=1", "com,example)/a?a=1&a=2&a-b=1"),
    ("bare_name_before_name_eq", "http://example.com/a?a=&a&a=1", "com,example)/a?a&a=&a=1"),
    ("empty_query_dropped", "http://example.com/a?", "com,example)/a"),
    ("empty_pieces_kept", "http://example.com/a?b=c&&d=e", "com,example)/a?&b=c&d=e"),
    ("trailing_amp_kept", "http://example.com/a?x=1&", "com,example)/a?&x=1"),
    ("query_on_empty_path", "http://example.com?x=1", "com,example)/?x=1"),
    ("plus_kept", "http://example.com/a?q=a+b", "com,example)/a?q=a+b"),
    ("query_percent_decoded", "http://example.com/a?q=a%2Bb&r=%C3%A4", "com,example)/a?q=a+b&r=%c3%a4"),
    ("encoded_amp_splits", "http://example.com/a?x=%26y", "com,example)/a?x=&y"),
    ("second_question_mark_kept", "http://example.com/a?b=c?d", "com,example)/a?b=c?d"),
    ("utm_params_kept", "http://www.example.com/a?q=1&utm_source=x", "com,example)/a?q=1&utm_source=x"),
    ("sid_param_removed", "http://example.com/a?sid=0123456789abcdef0123456789abcdef&x=1", "com,example)/a?x=1"),
    ("sid_param_last_leaves_amp", "http://example.com/a?x=1&sid=0123456789abcdef0123456789abcdef", "com,example)/a?&x=1"),
    ("sid_wrong_length_kept", "http://example.com/a?sid=0123456789abcdef", "com,example)/a?sid=0123456789abcdef"),
    ("jsessionid_param_removed", "http://example.com/a?jsessionid=0123456789abcdef0123456789abcdef", "com,example)/a"),
    ("phpsessid_param_removed", "http://example.com/a?PHPSESSID=0123456789abcdef0123456789abcdef&x=1", "com,example)/a?x=1"),
    ("aspsessionid_param_removed", "http://example.com/a?aspsessionidABCDEFGH=abcdefghijklmnopqrstuvwx", "com,example)/a"),
    ("cfid_cftoken_removed", "http://example.com/a?cfid=1&cftoken=0", "com,example)/a"),
    ("cfid_alone_kept", "http://example.com/a?cfid=1", "com,example)/a?cfid=1"),
    ("two_sids_remove_last", "http://example.com/a?sid=0123456789abcdef0123456789abcdef&sid=0123456789abcdef0123456789abcdef&a=1", "com,example)/a?a=1&sid=0123456789abcdef0123456789abcdef"),
    ("path_jsessionid_kept", "http://example.com/a;jsessionid=0123456789abcdef0123456789abcdef/b", "com,example)/a;jsessionid=0123456789abcdef0123456789abcdef/b"),
    ("aspnet_segment_removed", "http://example.com/(S(abcdefghij0123456789klmn))/page.aspx", "com,example)/page.aspx"),
    ("aspnet_segment_kept_without_aspx", "http://example.com/(S(abcdefghij0123456789klmn))/page.html", "com,example)/(s(abcdefghij0123456789klmn))/page.html"),
    ("aspnet_segment_in_middle", "http://example.com/notaries/(S(abcdefghij0123456789klmn))/Menu.aspx?x=1", "com,example)/notaries/menu.aspx?x=1"),
    ("aspnet_segment_wrong_length_kept", "http://example.com/(S(abcdefghij0123456789klm))/page.aspx", "com,example)/(s(abcdefghij0123456789klm))/page.aspx"),
    ("aspnet_two_groups", "http://example.com/(A(abcdefghij0123456789klmn)F(abcdefghij0123456789klmn))/p.aspx", "com,example)/p.aspx"),
    ("empty_input", "", "-"),
    ("whitespace_trimmed", "  http://example.com/x \t", "com,example)/x"),
    ("no_host", "http://", "http:"),
    ("no_host_with_path", "http:///", "http:/"),
    ("mailto_not_a_url", "mailto:A@B.com", "mailto:a@b.com"),
    ("empty_host_promotes_segment", "http:///example.com/x", "com,example)/x"),
]


@pytest.mark.parametrize("name,url,key", RULES, ids=[r[0] for r in RULES])
def test_rule_table(name, url, key):
    assert subject.surt_key(url) == key


def test_rule_names_are_unique():
    names = [r[0] for r in RULES]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("url", [
    "http://example.com:99999/x",     # above 65535
    "http://example.com:abc/x",       # not a number
    "http://example.com:+80/x",       # a sign is not a digit
    "http://example.com:80:80/x",     # two ports
    "http://::1/x",                   # an unbracketed IPv6 host
])
def test_bad_port_raises_and_none_form_returns_none(url):
    with pytest.raises(ValueError):
        subject.surt_key(url)
    assert subject.surt_key_or_none(url) is None


def test_lone_surrogate_cannot_be_keyed():
    assert subject.surt_key_or_none("http://example.com/a\ud800b") is None


def test_pure_and_idempotent_on_its_own_output_shape():
    # the same input, the same key, every time (no state, no clock, no network)
    u = "https://www.Example.com/A/b/?z=1&a=2#f"
    assert {subject.surt_key(u) for _ in range(5)} == {"com,example)/a/b?a=2&z=1"}


def _fixture_rows():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_sealed_parity_fixture():
    rows = _fixture_rows()
    assert len(rows) >= 2000, "the sealed sample shrank"
    assert {r["src"] for r in rows} == {"cc", "oracle"}
    cc = [r for r in rows if r["src"] == "cc"]
    assert len(cc) >= 1500, "the Common Crawl ground-truth part shrank"
    wrong = [(r["u"], r["k"], subject.surt_key_or_none(r["u"]))
             for r in rows if subject.surt_key_or_none(r["u"]) != r["k"]]
    assert not wrong, f"{len(wrong)} keys differ, first: {wrong[:3]}"


def test_fixture_covers_the_tricky_classes():
    rows = _fixture_rows()
    classes = {r.get("c") for r in rows}
    for needed in ("uniform", "idn", "port", "pct", "query", "longquery", "dblslash", "aspnet_session"):
        assert needed in classes, f"fixture lost the {needed!r} class"


# ---------------------------------------------------------------- the run-start guard


def test_parity_probes_hold_for_the_shipped_function():
    subject.assert_key_parity(force=True)
    assert len(subject.PARITY_PROBES) >= 12
    assert all(k.count(")") >= 1 for _, k in subject.PARITY_PROBES)


def test_parity_guard_refuses_a_wrong_key(monkeypatch):
    real = subject.surt_key
    monkeypatch.setattr(subject, "surt_key", lambda u: real(u).replace(")", ",)"))
    with pytest.raises(RuntimeError, match="surt key parity"):
        subject.assert_key_parity(force=True)


def test_cc_lookup_path_runs_the_guard_and_does_not_swallow_it(monkeypatch):
    pytest.importorskip("httpx")
    import cytir.acquire.cc_columnar as ccx

    monkeypatch.setattr(subject, "_PARITY_OK", False)
    real = subject.surt_key
    monkeypatch.setattr(subject, "surt_key", lambda u: real(u) + "x")
    with pytest.raises(RuntimeError, match="surt key parity"):
        ccx.url_surtkey("https://example.com/")
    monkeypatch.setattr(subject, "surt_key", real)
    monkeypatch.setattr(subject, "_PARITY_OK", False)
    assert ccx.url_surtkey("https://www.example.com/a/") == "com,example)/a"
    assert ccx.url_surtkey("https://example.com:99999/") is None   # a bad input is still None
    assert ccx.domain_surtkey_prefix("estv.admin.ch") == "ch,admin,estv)"
    assert ccx.domain_surtkey_prefix("www.lib4ri.ch") == "ch,lib4ri)"
