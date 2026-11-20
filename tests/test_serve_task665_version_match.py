#!/usr/bin/env python3
"""AW_VERSION_MATCH (task 665) -- dotted version tokens in the stage-2 lexical
features.

Pins the pure decisions the lever rests on, on the REAL titles that motivated
it (task 665 §1); the ranking claims live in the slice and
battery arms, not here (verification-doctrine 1):

  1. the tokenizer: "GLM-5.3" yields "5.3"; percentages, money and a digit
     glued to a longer number do not; a letter prefix is allowed.
  2. the matcher: equal or a dot-boundary refinement, never a plain prefix
     ("10.15" must not name "10.1", "5.30" must not name "5.3").
  3. OFF / NO-VERSION IS BYTE-IDENTICAL: with no query version the
     version-aware title/exact functions return exactly the shipped values,
     and the flag defaults off.
  4. the motivating case, as arithmetic: for "GLM 5.3" the multi-version HF
     title stops outscoring the HN title that names 5.3.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cytir.serve import retrieval as R  # noqa: E402
from cytir.serve import schema as S  # noqa: E402


def test_version_tokens_on_real_titles():
    assert S.version_tokens("GLM-5.3: Frontier coding") == {"5.3"}
    assert S.version_tokens("GLM-5, GLM-5.1, GLM-5.2") == {"5.1", "5.2"}
    assert S.version_tokens("glm5.3") == {"5.3"}
    assert S.version_tokens("Qwen3.8-27B") == {"3.8"}
    assert S.version_tokens("vLLM v0.28.0") == {"0.28.0"}
    assert S.version_tokens("Racket v9.3.") == {"9.3"}
    assert S.version_tokens("France reaches 94.9% fiber coverage") == frozenset()
    assert S.version_tokens("Private prisons announce $1.4B") == frozenset()
    assert S.version_tokens("Art. 336 OR") == frozenset()
    assert S.version_tokens(None) == frozenset()


def test_version_match_is_dot_boundary_not_prefix():
    assert S.version_match("5.3", "5.3")
    assert S.version_match("3.12", "3.12.2")
    assert not S.version_match("10.1", "10.15")
    assert not S.version_match("5.3", "5.30")
    assert not S.version_match("3.12.2", "3.12")


def test_flag_defaults_off():
    if os.environ.get("AW_VERSION_MATCH", "0") != "1":
        assert R._VERSION_MATCH is False
    assert R.effective_config()["version_match"] == {
        "on": R._VERSION_MATCH, "w": R._VERSION_MATCH_W}


def test_no_query_version_is_byte_identical():
    q_content = R._content_terms("GLM 5.3", None)
    for title in ("GLM-5, GLM-5.1, GLM-5.2", "GLM-5.3 is now open-weight", "GLM"):
        assert R._title_centrality_v(q_content, frozenset(), title) == \
            R._title_centrality(q_content, title)
        pl = {"title": title, "snippet": "x"}
        q_terms = R._query_terms("GLM 5.3")
        assert R._exact_v(q_terms, frozenset(), pl) == R._exact(q_terms, pl)


def test_glm53_title_arithmetic():
    q = "GLM 5.3"
    qc, qt, qv = R._content_terms(q, None), R._query_terms(q), S.version_tokens(q)
    hf, hn = "GLM-5, GLM-5.1, GLM-5.2", "GLM-5.3 is now open-weight"
    # shipped: the version is invisible, the multi-version page wins the title term
    assert R._title_centrality(qc, hf) == 1.0
    assert R._title_centrality(qc, hn) < 0.3
    # version-aware: the page naming 5.3 wins it
    assert R._title_centrality_v(qc, qv, hn) > R._title_centrality_v(qc, qv, hf)
    assert R._exact_v(qt, qv, {"title": hn}) == 1.0
    assert R._exact_v(qt, qv, {"title": hf}) == 0.5
    assert R._title_versions_match(qv, hn) and not R._title_versions_match(qv, hf)
