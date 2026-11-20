"""The Markdown row builder (`cytir.chunk.markdown_rows`) cuts the rows the refresh lane always cut (task 1157, S7).

`data/samples/refresh-lane-markdown-rows/recording.json` is what the UNMODIFIED lane builder returned for the 11
sealed Markdown inputs beside it (a long doc, headings only, a code-fence-heavy doc, Japanese, Russian and German
docs, an empty and a one-line doc, an explicit title, a missing title, and a news page on the legacy policy). It was
recorded on the build host with the real models: the lid.176 language gate and the pinned embedder tokenizer.

THE MODELS ARE STUBBED FROM THE RECORDING. This test runs on CI without them. It replaces exactly two functions:
`cytir.acquire.fetch_process.detect_lang` (the language gate) and `cytir.chunk.chunker._token_len` (the tokenizer's
count), each answering from the recording and failing loudly for a question the recording never saw. Everything
else (the pinned policy resolution against the registry, the structured chunker, quality and junk gates, the
`Passage` stamping, provenance ids) runs for real, so the rows must equal the recorded dicts byte for byte.
A real-model run is the recorder's job, not this test's.
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib

import pytest

from cytir.acquire import fetch_process as fp
from cytir.chunk import chunker as CH
from cytir.chunk import markdown_rows as MR

REPO = pathlib.Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "samples" / "refresh-lane-markdown-rows"
REC = json.loads((SAMPLE / "recording.json").read_text(encoding="utf-8"))
CASES = json.loads((SAMPLE / "cases.json").read_text(encoding="utf-8"))
_h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()   # noqa: E731


@pytest.fixture()
def recorded_models(monkeypatch):
    def detect_lang(text, url, default):
        key = json.dumps([_h(text), url, default])
        if key not in REC["gate"]:
            raise AssertionError(f"language gate asked a question the recording never saw: {url}")
        lang, confident = REC["gate"][key]
        return lang, confident

    def token_len(text):
        if _h(text) not in REC["tokens"]:
            raise AssertionError("tokenizer asked to count a text the recording never saw")
        return REC["tokens"][_h(text)]

    monkeypatch.setattr(fp, "detect_lang", detect_lang)
    monkeypatch.setattr(CH, "_token_len", token_len)


def test_recording_declares_real_models():
    prov = REC["provenance"]
    assert prov["tokenizer_status"] == "ok" and prov["lid_sha256"]
    assert len(CASES) >= 6 and {c["id"] for c in CASES} == {c["id"] for c in REC["cases"]}


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_rows_equal_the_recording(case, recorded_models, word_only_chunking_allowed):
    md = (SAMPLE / case["input"]).read_text(encoding="utf-8")
    rows = MR.build_rows_for_markdown(case["url"], md, title=case["title"], meta=case["meta"])
    want = next(c["rows"] for c in REC["cases"] if c["id"] == case["id"])
    assert json.dumps(rows, sort_keys=True) == json.dumps(want, sort_keys=True)


def test_the_recording_exercises_every_branch(recorded_models):
    by_id = {c["id"]: c["rows"] for c in REC["cases"]}
    assert len(by_id["long_doc"]) > 1 and by_id["code_fence_heavy"] and by_id["german"]
    assert by_id["japanese"] == by_id["russian"] == by_id["empty"] == []      # gate / nothing to cut
    assert {r["stamps"]["chunk"] for r in by_id["long_doc"]} == {4}           # docs: pinned generic epoch 4
    assert {r["stamps"]["chunk"] for r in by_id["news_legacy_policy"]} == {1}  # news: legacy, untouched
    assert by_id["title_override"][0]["title"] == "An Explicit Title"
    assert by_id["title_none_falls_back_to_url"][0]["title"] == "https://example.org/docs/untitled"


def test_docs_selection_is_pinned_and_only_for_docs():
    assert MR.docs_chunk_selection({"source_type": "news"}) is None
    sel = MR.docs_chunk_selection({"source_type": "docs"})
    assert sel.epoch == 4 and MR.docs_chunk_selection({"source_type": "docs"}) is sel   # resolved once


def test_the_module_loads_no_model_of_its_own():
    """One way to reach the models: through the chunker and the language gate, never a second loader here."""
    tree = ast.parse((REPO / "cytir" / "chunk" / "markdown_rows.py").read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
           {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in mods if m.split(".")[0] in {"fasttext", "transformers", "torch", "tokenizers"}}
