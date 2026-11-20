#!/usr/bin/env python3
"""Tests for `gate_payload_contract.py` -- the guard task 580 did not have.

The two fixtures are the two real shapes, reduced: task 579's tranche (every
scored payload field present -- the tranche that read +13 h@1) and task 580's
(`title`/`trust_tier`/`lang`/`domain` absent on every row -- the tranches that
read -10, -17 and -9 and were reverted as a text-restore failure).

What is asserted, and why each case exists:

  * the 580 shape FAILS with a non-zero exit code. This is the whole point:
    `main()` returning a verdict nobody reads is decoration.
  * the 579 shape PASSES, so the gate is not a blanket refusal.
  * an unreadable input, a missing file and an empty input all FAIL. A gate
    that skips on absent input reports green over unchecked ground.
  * the failure NAMES the fields, because "FAIL" without the field is a
    verdict a builder cannot act on.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
from cytir.gates import gate_payload_contract as G

GOOD = {"url": "https://www.w3.org/TR/css-flexbox-1/",
        "source_type": "spec", "title": "CSS Flexible Box Layout Module Level 1",
        "trust_tier": 1.0, "lang": "en", "domain": "www.w3.org",
        "snippet": "Flex layout is...", "country": "INTL"}
# the same row with no `country` -- honest in the corpus (94.2% present on a
# uniform sample of aw_e8), so the fixture carries a realistic share of them
# and the country floor is exercised rather than assumed away.
GOOD_NO_COUNTRY = dict(GOOD, country=None)
BAD = {"url": "https://www.service-public.gouv.fr/particuliers/vosdroits/F1420",
       "source_type": "official", "snippet": "Vous pouvez...",
       "country": "FR"}


def _pack(rows, name):
    p = pathlib.Path(tempfile.mkdtemp()) / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def test_the_580_shape_fails_with_a_nonzero_exit():
    path = _pack([BAD] * 50, "task580ish.jsonl")
    rc = G.main(["--pack", path])
    assert rc == 1, f"the 580 shape must FAIL; got exit {rc}"


def test_the_579_shape_passes():
    path = _pack([GOOD] * 45 + [GOOD_NO_COUNTRY] * 5, "task579ish.jsonl")
    rc = G.main(["--pack", path])
    assert rc == 0, f"the 579 shape must PASS; got exit {rc}"


def test_the_verdict_names_the_missing_fields():
    path = _pack([BAD] * 50, "named.jsonl")
    out = str(pathlib.Path(path).with_suffix(".verdict.json"))
    G.main(["--pack", path, "--out", out])
    v = json.load(open(out))
    assert v["verdict"] == "FAIL"
    named = " ".join(v["failures"])
    for f in ("title", "trust_tier", "lang", "domain"):
        assert f in named, f"the verdict must name {f}: {v['failures']}"
    assert v["rows_checked"] == 50


def test_a_missing_file_fails_closed():
    rc = G.main(["--pack", "/nonexistent/path/nope.jsonl"])
    assert rc == 2, f"a missing input must FAIL, not skip; got exit {rc}"


def test_an_unreachable_collection_fails_closed():
    # port 1 is reserved and refuses instantly: a transport error must be a
    # failure, never an empty census reported as a pass.
    rc = G.main(["--qdrant", "http://127.0.0.1:1", "--collection", "nope"])
    assert rc == 2, f"an unreachable qdrant must FAIL; got exit {rc}"


def test_an_empty_input_fails_closed():
    p = pathlib.Path(tempfile.mkdtemp()) / "empty.jsonl"
    p.write_text("")
    rc = G.main(["--pack", str(p)])
    assert rc == 2, f"an empty input must FAIL; got exit {rc}"


def test_no_force_flag_exists():
    # A waiver belongs to whoever owns the standard, not to the run that needs
    # excusing. If someone adds --force, this test is the conversation.
    src = pathlib.Path(G.__file__).read_text()
    for flag in ("--force", "--allow-fail", "--no-fail", "--warn-only"):
        assert f'add_argument("{flag}"' not in src, \
            f"the payload contract has no in-band waiver, and {flag} is one"


def test_a_partial_shape_still_fails():
    # half the rows carry a title: 50% is far under the 99% floor, and a
    # majority-good pack must not pass on its majority.
    path = _pack([GOOD] * 25 + [BAD] * 25, "half.jsonl")
    assert G.main(["--pack", path]) == 1


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"  OK   {name}")
        except Exception as e:                              # noqa: BLE001
            fails += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print("PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


def test_qdrant_prod_scroll_refused_before_any_http(monkeypatch, capsys):
    """Task 1536: a prod-looking --qdrant target without a named reason makes no HTTP call."""
    def boom(*a, **k):
        raise AssertionError("HTTP called against a prod target")
    monkeypatch.setattr(G, "_api", boom)
    monkeypatch.setattr(G.PG, "is_prod", lambda url, coll=None: True)
    rc = G.main(["--qdrant", "http://prod.example:6333", "--collection", "aw_prod"])
    assert rc == 2
    assert "REFUSED scroll on prod" in capsys.readouterr().err


def test_qdrant_prod_scroll_allowed_with_named_reason(monkeypatch):
    calls = []
    monkeypatch.setattr(G.PG, "is_prod", lambda url, coll=None: True)
    monkeypatch.setattr(G, "_api", lambda b, p, body: calls.append(p) or
                        {"result": {"points": [], "next_page_offset": None}})
    n, _ = G.from_collection("http://prod.example:6333", "aw_prod", prod_scroll_reason="test reason")
    assert n == 0 and len(calls) == 1
