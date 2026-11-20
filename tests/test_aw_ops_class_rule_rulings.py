"""Direct contract tests for cytir.ops.assembly_guards.class_rule_rulings (review 1331)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cytir.ops import assembly_guards as M

SCHEMA = "aw.adr101-class-precedence.v1"
DOC = {"schema": SCHEMA, "ranks": [
    {"rank": 1, "classes": ["strong", "strong2"], "basis": "B-strong"},
    {"rank": 2, "classes": ["weak"], "basis": "B-weak"}]}
SETS = [{"set_id": "s_strong", "class": "strong"},
        {"set_id": "s_strong2", "class": "strong2"},
        {"set_id": "s_weak", "class": "weak"},
        {"set_id": "s_unknown", "class": "no_such_class"}]


def _ctx(tmp_path: Path, doc=DOC):
    if doc is not None:
        target = tmp_path / M.CLASS_PRECEDENCE
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(doc))
    return SimpleNamespace(root=tmp_path, manifests={"sets": copy.deepcopy(SETS)})


def test_ruled_pair_names_the_lower_rank_winner_and_its_basis(tmp_path):
    out = M.class_rule_rulings(_ctx(tmp_path), {"s_weak x s_strong"})
    line = out[frozenset({"s_weak", "s_strong"})]
    assert (line["winner"], line["loser"]) == ("s_strong", "s_weak")
    assert line["ruling"] == "class" and line["basis"] == "B-strong"
    assert line["pair"] == "s_weak x s_strong"
    assert line["source"] == M.CLASS_PRECEDENCE and line["id"] == f"class-rule:{SCHEMA}"


def test_pair_order_does_not_change_key_or_winner(tmp_path):
    ctx = _ctx(tmp_path)
    a = M.class_rule_rulings(ctx, {"s_strong x s_weak"})
    b = M.class_rule_rulings(ctx, {"s_weak x s_strong"})
    assert a.keys() == b.keys() == {frozenset({"s_strong", "s_weak"})}
    key = next(iter(a))
    assert a[key]["winner"] == b[key]["winner"] == "s_strong"


@pytest.mark.parametrize("pair", [
    "s_strong x s_strong2",   # equal rank decides nothing
    "s_strong x s_unknown",   # class absent from the table
    "s_strong x s_missing",   # set absent from the manifests
    "no separator here"])     # not a 'LEFT x RIGHT' pair
def test_unruled_pairs_are_absent(tmp_path, pair):
    assert M.class_rule_rulings(_ctx(tmp_path), {pair}) == {}


def test_missing_table_rules_nothing(tmp_path):
    assert M.class_rule_rulings(_ctx(tmp_path, doc=None), {"s_weak x s_strong"}) == {}


def test_wrong_schema_is_refused(tmp_path):
    ctx = _ctx(tmp_path, doc={**DOC, "schema": "other"})
    with pytest.raises(M.GuardInputError):
        M.class_rule_rulings(ctx, {"s_weak x s_strong"})


def test_inputs_are_not_mutated(tmp_path):
    ctx = _ctx(tmp_path)
    pairs = {"s_weak x s_strong", "s_strong x s_strong2"}
    before = (copy.deepcopy(ctx.manifests), set(pairs))
    M.class_rule_rulings(ctx, pairs)
    assert (ctx.manifests, pairs) == before
