"""`cyt news stage` / `cyt news delta` (task 1989): the Python port of the two live shell steps keeps
every command line, the consent literals and every receipt byte.

Byte-for-byte equality with the recorded old-shell run lives in the unshipped
characterization test; this module asserts the Python side only.
"""
from __future__ import annotations

import hashlib
import json

import pytest

import news_refresh_chain_harness as H
from cytir.news import refresh_chain as RC

@pytest.fixture()
def twin():
    t = H.Twin()
    yield t
    t.close()


def test_stage_runs_the_manifest_by_module_with_one_inline_call(tmp_path):
    sc = H.Scenario("stage", tmp_path)
    rc, text = sc.run_python()
    assert rc == 0, text
    # the in-process edits: the concat and bucket heredocs are gone; the manifest start is by module
    assert [c["argv"][:3] for c in sc.inline_calls() if c["tool"] == "qpy"] == \
        [["-m", "cytir.news.refresh_chain", "manifest"]]
    assert len(sc.inline_calls()) == 1


def test_delta_starts_no_heredoc_or_one_liner(tmp_path, twin):
    sc = H.Scenario("delta", tmp_path, twin)
    rc, text = sc.run_python()
    assert rc == 0, text
    assert sc.inline_calls() == []          # no heredoc, no `-c` one-liner is started any more


def test_consent_literals_reach_stage_pack_b1_unchanged(tmp_path):
    sc = H.Scenario("stage", tmp_path)
    assert sc.run_python()[0] == 0
    packs = [c["argv"] for c in sc.calls() if c["argv"][:2] == ["-m", "cytir.index.stage_pack_b1"]]
    assert packs, "no arm was staged"
    for argv in packs:
        assert argv[argv.index("--consent-basis") + 1] == "robots-live-resolve"
        assert argv[argv.index("--consent-source") + 1] == "consent_gate.py direct-crawl receipt"
        assert argv[argv.index("--consent-state") + 1] == "resolved"
    assert (RC.CONSENT_BASIS, RC.CONSENT_SOURCE) == ("robots-live-resolve",
                                                      "consent_gate.py direct-crawl receipt")


def test_one_hash_recipe_writes_and_verifies(tmp_path):
    sc = H.Scenario("stage", tmp_path)
    assert sc.run_python()[0] == 0
    manifest = json.loads((sc.spool / "stage_manifest.json").read_text())
    body = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
    want = hashlib.sha256((json.dumps(body, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
    assert manifest["manifest_payload_sha256"] == want == RC.manifest_payload_sha256(manifest)
    assert RC.verify_stage_manifest(sc.spool / "stage_manifest.json") == \
        [a["collection"] for a in manifest["arms"].values()]
    manifest["arms"]["hn"]["points_count"] += 1
    (sc.spool / "stage_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="hash mismatch"):
        RC.verify_stage_manifest(sc.spool / "stage_manifest.json")


def test_delta_refuses_a_red_manifest_and_an_unhealthy_twin(tmp_path, twin):
    sc = H.Scenario("delta", tmp_path, twin)
    H.hand_manifest(sc.spool, {"hn": {"collection": "c", "points_count": 0, "status": "green"}})
    with pytest.raises(SystemExit, match="green/nonempty"):
        RC.run_delta(dict(sc.env))
    sc.env["AW_NEWS_TWIN_COLLECTION"] = "absent"
    with pytest.raises(SystemExit, match="canonical twin unavailable"):   # real Qdrant 404s
        RC.run_delta(dict(sc.env))


def test_stage_refuses_the_optional_langgate_and_missing_pins(tmp_path):
    sc = H.Scenario("stage", tmp_path)
    assert RC.run_stage({**sc.env, "AW_LANGGATE_OPTIONAL": "1"}) == 2
    no_pin = {k: v for k, v in sc.env.items() if k != "AW_NEWS_RELEASE_SHA"}
    assert RC.run_stage(no_pin) == 1
    assert not sc.log.exists()              # nothing started


def test_a_failing_child_ends_the_run_with_its_status(tmp_path):
    sc = H.Scenario("stage", tmp_path)
    (sc.bin / "lg").write_text("#!/bin/sh\nexit 7\n")
    assert RC.run_stage(dict(sc.env)) == 7


def test_cli_registers_stage_and_delta(capsys):
    from cytir import cli
    with pytest.raises(SystemExit) as exc:
        cli.main(["news", "stage", "--help"])
    assert exc.value.code == 0
    assert "fetch -> consent gate" in capsys.readouterr().out
    with pytest.raises(SystemExit) as bad:
        cli.main(["news", "bogus"])
    assert bad.value.code == 2
