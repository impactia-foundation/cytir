#!/usr/bin/env python3
"""Real-record guards for `fetch_process.detect_lang_scored` (task 652).

`detect_lang` is the audit-pinned interface (task 166/236/238) every other
caller in the pipeline uses; these tests prove `detect_lang_scored` holds
the one real implementation and `detect_lang` is a thin, behaviour-
preserving delegator -- never a second copy of the threshold logic. The
real fasttext model is not available in this environment (no numpy<2 /
fasttext wheel here -- that only runs in the build box's `.venv-langgate`), so
the gate is stubbed via `_GATE`/`assert_langgate_active`, exactly the way
a real probe would leave it: `_GATE["model"]` is a fake with a `.predict()`
of the same shape `fasttext` returns.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cytir.acquire import fetch_process as FP  # noqa: E402


class _FakeModel:
    """`.predict(text, k=1) -> ([label], [prob])`, fasttext's own shape."""

    def __init__(self, label: str, prob: float):
        self.label = label
        self.prob = prob

    def predict(self, _text, k=1):
        return ([f"__label__{self.label}"], [self.prob])


def _activate(monkeypatch, model):
    monkeypatch.setattr(FP, "assert_langgate_active",
                        lambda context="": {"active": True, "reason": None,
                                            "model_path": "stub"})
    monkeypatch.setitem(FP._GATE, "model", model)
    monkeypatch.setitem(FP._GATE, "active", True)


def test_detect_lang_scored_confident_pass(monkeypatch):
    _activate(monkeypatch, _FakeModel("de", 0.97))
    text = "echter deutscher text hier, lang genug fuer den lid-schwellenwert von funfzig zeichen"
    result = FP.detect_lang_scored(text, "https://x.ch", "en")
    assert result == {"lang": "de", "confident": True, "raw_label": "de",
                      "raw_prob": 0.97, "text_chars": len(text)}


def test_detect_lang_scored_low_confidence_still_reports_the_score(monkeypatch):
    """The whole point of this function over `detect_lang`: a row that
    FAILS the confidence check still carries its raw label/prob, not just
    a bare `confident: False`."""
    _activate(monkeypatch, _FakeModel("rm", 0.42))
    text = "x" * 60  # >= _LID_MIN_LEN so predict() actually runs
    result = FP.detect_lang_scored(text, "https://e-lir.ch/x", "en")
    assert result["confident"] is False
    assert result["raw_label"] == "rm"
    assert result["raw_prob"] == 0.42
    assert result["text_chars"] == 60
    # falls back to the URL heuristic for `lang` itself, same as detect_lang
    assert result["lang"] == "en"  # no /de//fr//it//en/ marker in this url


def test_detect_lang_scored_text_under_min_len_never_calls_predict(monkeypatch):
    calls = []
    model = _FakeModel("en", 0.99)
    model.predict = lambda *a, **k: calls.append(1) or (["__label__en"], [0.99])
    _activate(monkeypatch, model)
    result = FP.detect_lang_scored("too short", "https://example.com", "en")
    assert calls == []
    assert result["raw_label"] is None
    assert result["raw_prob"] is None
    assert result["confident"] is False


def test_detect_lang_delegates_to_scored_and_unpacks_the_pair(monkeypatch):
    """`detect_lang`'s own pinned (lang, confident) contract, now served by
    `detect_lang_scored` underneath -- unchanged from every existing
    caller's point of view."""
    _activate(monkeypatch, _FakeModel("fr", 0.9))
    text = "un vrai texte francais ici, assez long pour depasser le seuil"
    lang, confident = FP.detect_lang(text, "https://x.fr", "en")
    assert (lang, confident) == ("fr", True)


def test_detect_lang_gate_inactive_raises(monkeypatch):
    def _raise(context=""):
        raise FP.LangGateInactive("simulated: no model")
    monkeypatch.setattr(FP, "assert_langgate_active", _raise)
    try:
        FP.detect_lang_scored("some text", "https://x", "en")
        raise AssertionError("expected LangGateInactive")
    except FP.LangGateInactive:
        pass
