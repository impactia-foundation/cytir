"""Task 758: `search()` ranks in a TOTAL order (finding D6, plan T11).

`search()` ranks a set of string point ids. Python salts string hashes per process, so a sort by
score alone left exact ties in set-iteration order and every restart re-drew them (213 of 2,272
gate requests changed with PYTHONHASHSEED alone; tasks 752, 757, 758). The fix is a total order:
stage 1 (score, point id), stage 2 (fused, shorter URL without scheme, point id).

What this file pins (the ranking effect itself is measured on the twin, task 758):
  1. the two order helpers on tied fixtures;
  2. the helpers give the same answer under three hash seeds, and the old sort does NOT (so the
     seed test can see the defect it guards);
  3. `search()` sorts its candidates only through the helpers (a new score-only sort of `ids` or
     `scored` fails here before it ships);
  4. review 779 M1: the authority-leg ids join the pool in the leg's order (not a Python set's), and
     the entity-card boost picks among tied cards by a total key, under any hash seed; and no set
     inside `search()` is turned into an ordered sequence.
"""
from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cytir.serve import fusion, levers, retrieval  # noqa: E402

IDS = {f"{c}{i:02d}-uuid" for c in "abcdefgh" for i in range(8)}      # 64 string ids, a set


def test_stage1_orders_by_score_then_point_id():
    score = {"p3": 2.0, "p1": 1.0, "p2": 1.0, "p0": 1.0, "p9": 0.5}
    assert fusion._order_ids(set(score), score.get) == ["p3", "p0", "p1", "p2", "p9"]


def test_stage1_without_ties_is_the_score_order():
    score = {p: float(i) for i, p in enumerate(sorted(IDS))}
    assert fusion._order_ids(IDS, score.get) == sorted(IDS, key=score.get, reverse=True)


def test_stage2_breaks_a_tie_by_the_shorter_url_then_point_id():
    pl = lambda u: {"url": u}                                      # noqa: E731
    rows = [(1.0, "id-z", pl("https://x.test/a?utm_source=feed")),
            (1.0, "id-y", pl("https://x.test/a")),
            (1.0, "id-b", pl("http://x.test/b")),              # same length as /a once the
            (1.0, "id-a", pl("https://x.test/c")),             # scheme is not counted
            (2.0, "id-q", pl("https://x.test/a-much-longer-page")),
            (1.0, "id-n", None)]                                # no payload: length 0, first tie
    fusion._order_scored(rows)
    assert [r[1] for r in rows] == ["id-q", "id-n", "id-a", "id-b", "id-y", "id-z"]


def test_url_tie_length_ignores_the_scheme():
    assert fusion._url_tie_len("http://x.test/a") == fusion._url_tie_len("https://x.test/a")
    assert fusion._url_tie_len(None) == 0


_PROBE = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from cytir.serve import fusion, levers
ids = {f"{c}{i:02d}-uuid" for c in "abcdefgh" for i in range(8)}
rows = [(1.0, p, {"url": "https://x.test/" + p[:1]}) for p in ids]
fusion._order_scored(rows)
class P:
    def __init__(self, i, s): self.id, self.score = i, s
ann = [P(p, 1.0) for p in sorted(ids)]                    # the leg's (fixed) answer order
pool = fusion._extend_pool(["zz-pre"], fusion._auth_admit(ann, 0.5))
cards = [(1.0, 0, 0.5, [p, {"url": "https://" + p[:3] + ".test/"}]) for p in ids]
print(json.dumps({"new1": fusion._order_ids(ids, lambda p: 1.0),
                  "new2": [r[1] for r in rows],
                  "pool": pool,
                  "cards": [e[3][0] for e in levers._card_winners(cards)],
                  "old": sorted(ids, key=lambda p: 1.0, reverse=True),
                  "old_pool": list(dict.fromkeys(["zz-pre"] + list({p.id for p in ann})))}))
"""


def _probe(seed: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env["PYTHONHASHSEED"] = seed
    out = subprocess.run([sys.executable, "-c", _PROBE, str(HERE), str(HERE.parent)], env=env, check=True,
                         capture_output=True, text=True, cwd=str(HERE)).stdout
    import json
    return json.loads(out.strip().splitlines()[-1])


def test_the_order_does_not_depend_on_the_hash_seed():
    runs = {s: _probe(s) for s in ("0", "1", "7")}
    assert runs["0"]["new1"] == runs["1"]["new1"] == runs["7"]["new1"] == sorted(IDS)
    assert runs["0"]["new2"] == runs["1"]["new2"] == runs["7"]["new2"]
    assert runs["0"]["pool"] == runs["1"]["pool"] == runs["7"]["pool"] == ["zz-pre"] + sorted(IDS)
    assert runs["0"]["cards"] == runs["1"]["cards"] == runs["7"]["cards"]
    # the control: the old score-only sort of a set DOES move with the seed, so this test can
    # see the defect (64 ids in one tie: the chance that three seeds agree by luck is nil)
    assert len({tuple(r["old"]) for r in runs.values()}) > 1
    assert len({tuple(r["old_pool"]) for r in runs.values()}) > 1


def _search_sorts() -> list[str]:
    src = pathlib.Path(retrieval.__file__).read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "search")
    bad = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        target = None
        if isinstance(f, ast.Name) and f.id == "sorted" and node.args:
            target = node.args[0]
        elif isinstance(f, ast.Attribute) and f.attr == "sort":
            target = f.value
        names = ({n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
                 if target is not None else set())
        if names & {"ids", "scored", "_ordered", "pre", "elig", "best_by_dom", "winners"}:
            bad.append(f"line {node.lineno}: {ast.unparse(node)[:90]}")
    return bad


def test_search_sorts_its_candidates_only_through_the_total_order():
    assert _search_sorts() == [], ("search() sorts candidates outside _order_ids/_order_scored; "
                                   "a score-only sort of a set re-draws exact ties per process (task 758)")


def test_entity_card_winner_ties_break_by_shorter_url_then_point_id():
    row = lambda pid, url: [pid, {"url": url}]                 # noqa: E731
    elig = [(1.0, 3, 0.5, row("p2", "https://devis-debarras.ch/")),
            (1.0, 3, 0.5, row("p1", "https://devisdebarras.ch/")),
            (1.0, 3, 0.5, row("p0", "https://www.xyz.ch/")),     # shortest; same length as p3, lower id
            (1.0, 3, 0.5, row("p3", "https://www.xyw.ch/")),
            (0.9, 9, 0.9, row("p4", "https://a.ch/")),           # a worse name match loses
            (1.0, 3, 0.4, row("p5", "https://devisdebarras.ch/fr"))]  # same domain as p1, lower rel
    assert [e[3][0] for e in levers._card_winners(elig)] == ["p0", "p3", "p1", "p2", "p4"]


def test_authority_admit_keeps_the_leg_order_and_its_beta():
    class P:
        def __init__(self, i, s):
            self.id, self.score = i, s
    pts = [P("c", 1.0), P("a", 0.9), P("b", 0.4)]
    assert list(fusion._auth_admit(pts, 0.5)) == ["c", "a"]
    assert fusion._auth_admit([], 0.5) == {}
    assert fusion._extend_pool(["a", "x"], {"c": None, "a": None}) == ["a", "x", "c"]


def _search_sets_made_ordered() -> list[str]:
    """Names `search()` binds to a set, then turns into an ORDERED sequence: `list(s)`,
    `tuple(s)`, `dict.fromkeys(s)`, `enumerate(s)`, or a `for`/comprehension over `s` that is not
    an order-free reduction (max/min/sum/any/all/len/sorted). Each one would carry hash-salt
    order into the ranking (review 779 M1: `auth_admit`)."""
    src = pathlib.Path(retrieval.__file__).read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "search")

    def is_set(v):
        if isinstance(v, (ast.Set, ast.SetComp)):
            return True
        if isinstance(v, ast.Call):
            f = v.func
            return ((isinstance(f, ast.Name) and f.id in ("set", "frozenset"))
                    or (isinstance(f, ast.Attribute) and f.attr in ("union", "intersection", "difference")))
        return False
    sets = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and is_set(n.value):
            sets |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and n.value is not None and is_set(n.value):
            if isinstance(n.target, ast.Name):
                sets.add(n.target.id)
    parent = {c: p for p in ast.walk(fn) for c in ast.iter_child_nodes(p)}
    order_free = {"max", "min", "sum", "any", "all", "len", "sorted", "_order_ids"}
    bad = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id in sets:
            f = n.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            if name in ("list", "tuple", "enumerate", "fromkeys"):
                bad.append(f"line {n.lineno}: {ast.unparse(n)[:90]}")
        it = n.iter if isinstance(n, (ast.For, ast.comprehension)) else None
        if isinstance(it, ast.Name) and it.id in sets:
            gen = parent.get(n)
            call = parent.get(gen) if isinstance(gen, ast.GeneratorExp) else None
            fname = getattr(getattr(call, "func", None), "id", None)
            if fname not in order_free:
                bad.append(f"line {getattr(n, 'lineno', getattr(it, 'lineno', '?'))}: iterates set {it.id}")
    return bad


def test_search_never_turns_a_set_into_an_order():
    assert _search_sets_made_ordered() == [], (
        "search() turns a Python set into an ordered sequence; string sets iterate in hash-salt "
        "order, so the ranking would depend on PYTHONHASHSEED (task 758, review 779 M1)")
