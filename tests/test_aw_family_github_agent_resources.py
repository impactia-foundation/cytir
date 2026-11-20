"""`github_agent_resources` on the Protocol: receipts and documents identical to the legacy acquirer's (task 1115).

`data/samples/github-agent-resources/root/` is a preserved root the UNMODIFIED legacy acquirer wrote
for four sealed repositories (docs tree with an RST file, a heading-less file, a 0-byte file and
unparsable link; a Git-blob-API fallback; a repo with no docs; a copyleft repo that takes only the
root document). `expected_candidates.jsonl` is what the UNMODIFIED refresh-lane reader returned for
it. `rows-recording.json` is what the UNMODIFIED lane path (build + stamp) returned for those documents on the
build host with the real models (task 1169); the gate and tokenizer are STUBBED FROM IT, as in
`test_aw_chunk_markdown_rows.py`. No network.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from cytir.acquire import fetch_process as fp
from cytir.chunk import chunker as CH
from cytir.chunk import markdown_rows as MR
from cytir.extract import github_agent_resources as G
from cytir.sources import run_family
from cytir.sources.family import RunContext
from cytir.sources.families.github_agent_resources import GithubAgentResourcesFamily

REPO = pathlib.Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "samples" / "github-agent-resources"
ROOT = SAMPLE / "root"
REC = json.loads((SAMPLE / "rows-recording.json").read_text(encoding="utf-8"))
_h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()   # noqa: E731
OK = {"bucket": "ok", "search_ok": True, "train_ok": True, "status": "ok"}


def _ctx(tmp_path):
    return RunContext(out_dir=tmp_path, today="2026-10-23", log=lambda m: None,
                      options={"root": str(ROOT), "trust_tier": "A"})


def _drive(tmp_path):
    fam, ctx = GithubAgentResourcesFamily(), _ctx(tmp_path)
    out = {}
    for u in fam.discover(ctx):
        try:
            caps = fam.acquire(u, ctx)
            out[u.key] = [d for c in caps for d in fam.extract(c, ctx)], caps
        except RuntimeError as exc:
            out[u.key] = exc, None
    return out


def test_documents_identical_to_the_legacy_reader(tmp_path):
    want = [json.dumps(json.loads(x), sort_keys=True, ensure_ascii=False)
            for x in (SAMPLE / "expected_candidates.jsonl").read_text().splitlines()]
    got = [json.dumps(d.extra["candidate"], sort_keys=True, ensure_ascii=False)
           for k, (docs, _) in _drive(tmp_path).items() if not isinstance(docs, Exception) for d in docs]
    assert got == want and len(want) == 6


def test_receipt_is_sealed_byte_for_byte_and_quarantine_is_named(tmp_path):
    out = _drive(tmp_path)
    for repo, (docs, caps) in out.items():
        if caps:
            assert caps[0].body == (ROOT / "receipts" / G.receipt_name(repo)).read_bytes()
    assert str(out["acme/nodocs"][0]) == "receipt_quarantined:no_eligible_documentation"


@pytest.fixture()
def recorded_models(monkeypatch):
    """Gate and tokenizer answer from the recording (see the module docstring); fail loudly on a new question."""
    def detect_lang(text, url, default):
        key = json.dumps([_h(text), url, default])
        if key not in REC["gate"]:
            raise AssertionError(f"language gate asked a question the recording never saw: {url}")
        return tuple(REC["gate"][key])

    def token_len(text):
        if _h(text) not in REC["tokens"]:
            raise AssertionError("tokenizer asked to count a text the recording never saw")
        return REC["tokens"][_h(text)]

    monkeypatch.setattr(fp, "detect_lang", detect_lang)
    monkeypatch.setattr(CH, "_token_len", token_len)


_needs_selector = pytest.mark.skipif(
    not MR._SELECTOR_ARTIFACT.is_file(),
    reason="the sha-pinned chunk selector is not in this tree (public cut; task 1157 decision D5)")


def test_recording_declares_real_models():
    prov = REC["provenance"]
    assert prov["tokenizer_status"] == "ok" and prov["tokenizer_is_fast"] is True and prov["lid_sha256"]
    assert prov["today_frozen"] == "2026-10-23" and len(prov["git_sha"]) == 40


@_needs_selector
def test_chunk_rows_equal_the_lane_recording(tmp_path, recorded_models, word_only_chunking_allowed):
    ctx = _ctx(tmp_path)
    fam, got = GithubAgentResourcesFamily(), []
    for repo, (docs, _) in _drive(tmp_path).items():
        if not isinstance(docs, Exception):
            for d in docs:
                got += fam.chunk(d, ctx)
    want = [r for c in REC["cases"] for r in c["rows"]]
    key = lambda r: r["point_id"]   # noqa: E731
    assert len(want) == 5
    assert json.dumps(sorted(got, key=key), sort_keys=True) == json.dumps(sorted(want, key=key), sort_keys=True)
    assert {r["last_seen"] for r in got} == {"2026-10-23"}      # the clock is ctx.today, not the box's


@_needs_selector
def test_run_family_writes_a_pack_with_the_recorded_rows(tmp_path, recorded_models, word_only_chunking_allowed):
    res = run_family.run_family(
        "github_agent_resources", out_dir=tmp_path / "o", today="2026-10-23", log=lambda m: None,
        consent=lambda h, d: {**OK, "host": h}, options={"root": str(ROOT), "trust_tier": "A"})
    assert res
    rows = [json.loads(x) for p in (tmp_path / "o").rglob("*.jsonl") for x in p.read_text().splitlines()
            if '"point_id"' in x]
    assert {r["point_id"] for r in rows} >= {r["point_id"] for c in REC["cases"] for r in c["rows"]}


def test_the_adapter_is_not_marked_partial():
    assert not hasattr(GithubAgentResourcesFamily, "PARTIAL")


@_needs_selector
def test_the_pinned_policy_resolves_through_data_path():
    assert MR._pinned_policy_path(json.loads(MR._SELECTOR_ARTIFACT.read_text())["generic_policy"]["file"]).is_file()
