"""`aw_common.require_index_fold` -- the guard for the measurement that lied.

Task 564: task 562's bisect script imports the QUERY-side tokenizer
(`aw_server.schema.bm25_tokens`, whose `_FOLD_ACCENTS` defaults OFF) with no
runner setting `AW_FOLD_ACCENTS`, while its sibling `run_probe_562.sh` runs
under `prod_env_run.py`, which does. An unfolded hash for `chômage` matches a
tok@1 row and misses the tok@2 row that holds the same word, so the bisect
reported four golds losing words their own row carries
(task 564's refold-words artifact).

The guard fails CLOSED and names both sides' actual setting, because the
failure mode is silence: nothing crashes, the counts all balance, and only the
accented terms are wrong.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cytir.core.bm25_tokenizer as AC  # noqa: E402  (task 2333: the tokenizer moved)


def _with_fold(value, fn, query_side=None):
    """Hold BOTH tokenizers at a known setting for the duration of `fn`.

    `query_side=None` means "leave the query side out of it" -- the module is
    dropped from `sys.modules` so the guard sees only the write side. Whether
    `aw_server.schema` happens to be imported depends on which other test ran
    first in the same process, and a test whose verdict depends on that is not
    a test (it failed exactly that way on the first full-suite run).
    """
    old = AC._FOLD_ACCENTS
    saved = {m: sys.modules[m] for m in list(sys.modules)
             if m == "cytir.serve.schema"}
    for m in saved:
        del sys.modules[m]
    if query_side is not None:
        import importlib
        old_env = os.environ.get("AW_FOLD_ACCENTS")
        os.environ["AW_FOLD_ACCENTS"] = "1" if query_side else "0"
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))
        importlib.import_module("cytir.serve.schema")
    AC._FOLD_ACCENTS = value
    try:
        return fn()
    finally:
        AC._FOLD_ACCENTS = old
        for m in [x for x in sys.modules if x == "cytir.serve.schema"]:
            del sys.modules[m]
        sys.modules.update(saved)
        if query_side is not None:
            if old_env is None:
                os.environ.pop("AW_FOLD_ACCENTS", None)
            else:
                os.environ["AW_FOLD_ACCENTS"] = old_env


def test_passes_when_the_write_side_is_folded():
    assert _with_fold(True, lambda: AC.require_index_fold(2)) is True


def test_refuses_an_unfolded_process_against_a_tok2_index():
    def go():
        try:
            AC.require_index_fold(2, who="a probe")
        except RuntimeError as exc:
            assert "tok@2" in str(exc) and "aw_common" in str(exc)
            return "refused"
        return "passed"
    assert _with_fold(False, go) == "refused"


def test_a_deliberate_pre_fold_run_states_its_epoch():
    assert _with_fold(False, lambda: AC.require_index_fold(1)) is False


def test_env_supplies_the_default_epoch():
    old = os.environ.get("AW_TOK_EPOCH")
    os.environ["AW_TOK_EPOCH"] = "1"
    try:
        assert _with_fold(False, lambda: AC.require_index_fold()) is False
    finally:
        if old is None:
            os.environ.pop("AW_TOK_EPOCH", None)
        else:
            os.environ["AW_TOK_EPOCH"] = old


def test_an_unknown_epoch_is_a_refusal_not_a_default():
    try:
        AC.require_index_fold(7)
    except RuntimeError as exc:
        assert "7" in str(exc)
        return
    raise AssertionError("an unknown tokenizer epoch must refuse")


def test_the_query_side_is_checked_too_when_it_is_loaded():
    """The 562 case exactly: the write side is pinned (an `aw_b1_projection`
    import would do it), the QUERY side was imported bare, and only the query
    side is wrong."""
    def go():
        try:
            AC.require_index_fold(2, who="a probe")
        except RuntimeError as exc:
            assert "cytir.serve.schema" in str(exc)
            return "refused"
        return "passed"
    assert _with_fold(True, go, query_side=False) == "refused"


def test_both_sides_folded_passes():
    assert _with_fold(True, lambda: AC.require_index_fold(2),
                      query_side=True) is True
