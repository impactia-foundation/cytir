"""cytir.core.surtkey against the retired `surt` package, used ONLY as an oracle.

Skipped unless `surt` is importable. It is not declared anywhere (ADR-121: the
package is AGPL-3.0): to run this file, `pip install surt==0.3.1` in a scratch
venv. CI proves parity from the sealed fixture in `test_surtkey.py` instead.

The generators are seeded, so a failure repeats. They draw from the shapes the
tests/fixtures hold plus the odd ones the rule table names. A difference on a
generated input is a bug in `surt_key` (or a new rule to measure), never a
reason to loosen the assertion.
"""
from __future__ import annotations

import random

import pytest

surt_pkg = pytest.importorskip("surt")

from cytir.core.surtkey import surt_key  # noqa: E402
from tests.test_surtkey import RULES  # noqa: E402


def _oracle(url: str):
    try:
        return surt_pkg.surt(url)
    except Exception:  # noqa: BLE001 -- the package raised: so must we
        return "RAISES"


def _mine(url: str):
    try:
        return surt_key(url)
    except Exception:  # noqa: BLE001
        return "RAISES"


def test_rule_table_equals_the_package():
    for name, url, _key in RULES:
        assert _mine(url) == _oracle(url), name


def test_package_and_function_both_match_the_sealed_fixture():
    """The fixture's Common Crawl keys are the truth; the package matched all of
    them when the fixture was cut, and so must the function."""
    import json
    from tests.test_surtkey import FIXTURE
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["u"] for r in rows if _oracle(r["u"]) != r["k"]] == []
    assert [r["u"] for r in rows if _mine(r["u"]) != r["k"]] == []


S32 = "0" * 32
T24 = "a1" * 12
PATH_TOKENS = ["a", "b", "..", ".", "", "%2e", "%2e%2e", "%2f", "%252f", "%252e%252e", "%25", "%2525",
               "A", "%41", "%2541", "%20", "%2520", "...", "%5c", "%3f", "%253f", "%23", "x.aspx", ".aspx",
               "a.ASPX", "(S(" + T24 + "))", "(s(" + T24.upper() + "))", "(A(" + T24 + ")F(" + T24 + "))",
               "%28S%28" + T24 + "%29%29", "ä", "Ä", "€", "x.html"]
QUERY_TOKENS = ["a", "b", "A", "a=1", "a=2", "a=", "=1", "%26", "%2526", "%3d", "%253d", "%25", "%2541",
                "sid=" + S32, "%73id=" + S32, "%2573id=" + S32, "jsessionid=" + S32, "phpsessid=" + S32,
                "aspsessionidabcdefgh=" + "a" * 24, "cfid=1", "cftoken=2", "x=1%26y=2", "", "&", "%2e",
                "%2f", "%23", "%3f", "?", "é=1", "k=ä", "a-b=1", "a_b=1"]
HOST_TOKENS = ["a", "B", "www", "www1", "WWW", "", "%2e", "%41", "%2541", "é", "É", "ß",
               "%c3%a9", "1", "010", "0", "99999", "4294967296", "x_y", "a b", "%20", "。", "xn--9ca",
               "​", "-", "08", "255", "256", "0x7f", "012", "00", "1a", "com", "ch", "co", "uk"]


def _cases(n: int, seed: int, make):
    rnd = random.Random(seed)
    return [make(rnd) for _ in range(n)]


def _path_case(rnd):
    return "http://x.com" + "/".join([""] + [rnd.choice(PATH_TOKENS) for _ in range(rnd.randint(0, 6))]) + rnd.choice(["", "", "/"])


def _query_case(rnd):
    sep = rnd.choice(["&", "&", "&", "%26"])
    return "http://x.com/p?" + sep.join(rnd.choice(QUERY_TOKENS) for _ in range(rnd.randint(1, 5)))


def _host_case(rnd):
    host = ".".join(rnd.choice(HOST_TOKENS) for _ in range(rnd.randint(1, 5)))
    return "http://" + host + rnd.choice(["", ":8080", ":80", "."]) + "/x"


@pytest.mark.parametrize("name,make,seed", [("path", _path_case, 1), ("query", _query_case, 2), ("host", _host_case, 3)])
def test_generated_inputs_equal_the_package(name, make, seed):
    wrong = []
    for url in _cases(6000, seed, make):
        if _mine(url) != _oracle(url):
            wrong.append((url, _oracle(url), _mine(url)))
    assert not wrong, f"{len(wrong)} of 6000 {name} cases differ, first: {wrong[:3]}"


def test_mixed_url_shapes_equal_the_package():
    rnd = random.Random(4)
    schemes = ["http://", "https://", "HTTP://", "https://https://", "http://http://"]
    ports = ["", ":80", ":443", ":8080", ":0", ":00080", ":65535"]
    users = ["", "u@", "u:p@"]
    wrong = []
    for _ in range(6000):
        # a host that cannot decode to nothing: "no host" inputs are in the rule table
        url = (rnd.choice(schemes) + rnd.choice(users) + "h." + _host_case(rnd)[7:-2] + rnd.choice(ports)
               + _path_case(rnd)[12:])
        if rnd.random() < 0.6:
            url += "?" + _query_case(rnd).split("?", 1)[1]
        if rnd.random() < 0.15:
            url += "#" + rnd.choice(["", "f", "!/x", "?a=1"])
        if _mine(url) != _oracle(url):
            wrong.append((url, _oracle(url), _mine(url)))
    assert not wrong, f"{len(wrong)} of 6000 differ, first: {wrong[:3]}"
