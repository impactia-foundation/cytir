"""cytir.gates.serve_parity -- the S10 parity gate, on fake server trees (task 752).

The fixture is two (or three) throwaway source trees, each with a tiny `cytir/serve` whose
`search()` is a deterministic function of the cached query vector. The real harness runs them:
real worker processes, the real boot (the tree first on sys.path), the real FastAPI TestClient,
the real comparison and verdict. Only the server is fake, so these tests prove the instrument,
not a ranking: a PASS on equal trees, FAIL on a ranking or a deep-component change, INCONCLUSIVE
on a base that disagrees with itself, ERROR on a cache miss, FAIL on a configuration change.
"""
from __future__ import annotations

import json
import pathlib
import socket

import pytest

from cytir.gates import serve_parity as sp

APP_PY = '''
from fastapi import FastAPI, Query
from . import retrieval
api = FastAPI()

@api.get("/search")
def search(q: str = Query(..., min_length=1), k: int = 10, lang: str | None = None,
           country: str | None = None, authorities: int = 0):
    res = retrieval.search(q, k=k, lang=lang, country=country)
    return {"query": q, "count": len(res), "results": res}
'''

MODELS_PY = '''
import threading
MRL_DIM = 4
TEI_URL = ""
_tls = threading.local()
def last_full_query_vec():
    return getattr(_tls, "full_qvec", None)
def embed_query(q):
    raise RuntimeError("the harness must replace the embedder")
'''

RETRIEVAL_PY = '''
import os, pathlib
from . import models
WEIGHTS = {"dense": 1.0}
_RANKHEAD_MODEL = {"loaded": True}
_LAST_SCORE_DEBUG = []
_LAST_QUERY_PLAN_TRACE = {}
TWEAK = @TWEAK@
COMPONENT = @COMPONENT@
CONFIG_EXTRA = @CONFIG_EXTRA@
COUNTER = pathlib.Path(__file__).with_name("calls.txt")

def effective_config():
    return {"collection": os.environ.get("AW_COLLECTION"), "extra": CONFIG_EXTRA,
            "armed": os.environ.get("AW_DARK_LEVER"), "stray": os.environ.get("AW_STRAY"),
            "fusion_weights": {"path": os.path.join(os.path.dirname(__file__), "w.json")}}

def search(q, k=10, lang=None, country=None):
    dense, _ = models.embed_query(q)
    full = models.last_full_query_vec()
    base = sum(float(x) for x in full)
    n = 0
    if TWEAK in ("drift", "deepdrift"):   # a base that disagrees with itself
        n = int(COUNTER.read_text()) if COUNTER.exists() else 0
        COUNTER.write_text(str(n + 1))
    if TWEAK == "drift":
        base += n
    urls = [f"https://example.org/{q.replace(' ', '-')}/{i}" for i in range(k)]
    if TWEAK == "setorder":        # equal scores, order = iteration order of a set of strings
        urls = list({f"https://example.org/{q.replace(' ', '-')}/tie-{i}" for i in range(12)})[:k]
    if TWEAK == "swap" and "swap" in q:
        urls[0], urls[1] = urls[1], urls[0]
    _LAST_SCORE_DEBUG.clear()
    _LAST_QUERY_PLAN_TRACE.clear()
    if os.environ.get("AW_DEBUG_SCORES") == "1":
        for i, u in enumerate(urls):
            _LAST_SCORE_DEBUG.append({"pid": f"p{i}", "url": u, "fused": base - i,
                                      "components": {"rel.dense": COMPONENT * dense[0],
                                                     "rel.bm25": 1e-6 * n}})
    if os.environ.get("AW_QHEAD_COUNTRY_TRACE") == "1":
        _LAST_QUERY_PLAN_TRACE.update({"final_ranked": [{"id": f"p{i}", "url": u}
                                                        for i, u in enumerate(urls)],
                                       "stage1_request_ms": 1.0})
    if TWEAK == "setorder":
        return [{"url": u, "score": round(base, 4)} for u in urls]
    return [{"url": u, "score": round(base - i, 4)} for i, u in enumerate(urls)]
'''


def make_tree(root: pathlib.Path, name: str, *, tweak: str = "", component: float = 1.0,
              config_extra: str = "") -> pathlib.Path:
    tree = root / name
    srv = tree / "cytir" / "serve"
    srv.mkdir(parents=True)
    (tree / "cytir").mkdir(exist_ok=True)
    (tree / "cytir" / "__init__.py").write_text("")
    (tree / "webapp").mkdir()
    (tree / "webapp" / "index.html").write_text("<html></html>")
    (srv / "__init__.py").write_text("")
    (srv / "app.py").write_text(APP_PY)
    (srv / "models.py").write_text(MODELS_PY)
    (srv / "retrieval.py").write_text(RETRIEVAL_PY.replace("@TWEAK@", repr(tweak))
                                      .replace("@COMPONENT@", repr(component))
                                      .replace("@CONFIG_EXTRA@", repr(config_extra)))
    return tree


def free_url() -> str:
    """A loopback URL on a port nothing listens on (the harness only reads its point count, and
    records an unreadable count as unknown)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


QUERIES = ["alpha", "beta gamma", "swap me", "delta"]


def write_cache(path: pathlib.Path, queries=QUERIES) -> pathlib.Path:
    vecs = {q: [float(i + 1), 0.5, 0.25, 0.125] for i, q in enumerate(queries)}
    path.write_text(json.dumps({"schema": sp.CACHE_SCHEMA, "vectors": vecs}))
    return path


def requests(queries=QUERIES):
    return [sp._req("t", str(i), q, k=3) for i, q in enumerate(queries)]


def run(tmp_path, base, head, *, reqs=None, deep=True, seed_probe=False, arm_extra=None):
    guarded = []
    rep = sp.run_parity(reqs or requests(), base_tree=base, head_tree=head, collection="twin_x",
                        qdrant_url=free_url(), vec_cache=write_cache(tmp_path / "v.json"),
                        out_dir=tmp_path / "out", shards=2, deep=deep, seed_probe=seed_probe,
                        guard=lambda url, coll: guarded.append(coll),
                        declared_env={"AW_FAKE_KNOB": "1"}, arm_extra=arm_extra,
                        say=lambda m: None)
    assert guarded == ["twin_x"], "the twin guard must run before any arm starts"
    return rep


def test_identical_code_in_two_trees_passes_with_a_clean_null_pair(tmp_path):
    base = make_tree(tmp_path, "base")
    head = make_tree(tmp_path, "head")
    rep = run(tmp_path, base, head)
    assert rep["verdict"] == "PASS", sp.summary(rep)
    assert rep["counts"]["EQUAL"] == len(QUERIES)
    assert rep["null_pair"] == {"first_attempt_unstable": 0, "unstable_after_retries": 0}
    assert rep["trees"]["same_server_code"] is True
    assert rep["config_change"] is None          # one declared env: nothing to record
    assert rep["fingerprint_diffs"] == []        # tree paths are masked, not compared
    assert (tmp_path / "out" / "parity.json").is_file()


def test_a_ranking_change_fails_and_names_the_request(tmp_path):
    base = make_tree(tmp_path, "base")
    head = make_tree(tmp_path, "head", tweak="swap")
    rep = run(tmp_path, base, head, deep=False)
    assert rep["verdict"] == "FAIL"
    assert [r["id"] for r in rep["not_equal"]] == ["t:2"]
    assert rep["not_equal"][0]["final"]["tier"] == "served"
    assert rep["trees"]["same_server_code"] is False


def test_a_component_change_with_the_same_answer_fails_on_the_deep_tier(tmp_path):
    base = make_tree(tmp_path, "base")
    head = make_tree(tmp_path, "head", component=1.5)
    rep = run(tmp_path, base, head)
    assert rep["verdict"] == "FAIL", "an aggregate cannot acquit a component"
    assert {r["final"]["tier"] for r in rep["not_equal"]} == {"deep"}


def test_a_base_that_disagrees_with_itself_is_inconclusive_never_pass(tmp_path):
    base = make_tree(tmp_path, "base", tweak="drift")
    head = make_tree(tmp_path, "head", tweak="drift")
    rep = run(tmp_path, base, head, reqs=requests(["alpha"]), deep=False)
    assert rep["verdict"] == "INCONCLUSIVE"
    assert rep["not_equal"][0]["attempts"] == 1 + sp.RETRIES


def test_a_deep_record_that_moves_under_the_same_code_is_unstable_not_a_diff(tmp_path):
    """The task-752 first run: a twin write between base/deep and head/deep shifted BM25 scores
    in the 5th digit while the rounded served answer stayed equal. The deep null must catch it."""
    base = make_tree(tmp_path, "base", tweak="deepdrift")
    head = make_tree(tmp_path, "head", tweak="deepdrift")
    rep = run(tmp_path, base, head, reqs=requests(["alpha"]))
    assert rep["verdict"] == "INCONCLUSIVE", sp.summary(rep)
    assert "deep" in rep["not_equal"][0]["final"]["why"]


def test_a_missing_vector_is_an_error_not_a_diff(tmp_path):
    base = make_tree(tmp_path, "base")
    head = make_tree(tmp_path, "head")
    rep = run(tmp_path, base, head, reqs=requests(["not in the cache"]), deep=False)
    assert rep["verdict"] == "ERROR"


def test_a_configuration_change_fails_even_when_every_answer_is_equal(tmp_path):
    base = make_tree(tmp_path, "base")
    head = make_tree(tmp_path, "head", config_extra="moved")
    rep = run(tmp_path, base, head, deep=False)
    assert rep["verdict"] == "FAIL"
    assert rep["counts"]["EQUAL"] == len(QUERIES)
    assert any("effective_config.extra" in d for d in rep["fingerprint_diffs"])


def test_the_twin_guard_refusal_stops_the_run_before_any_arm(tmp_path):
    base = make_tree(tmp_path, "base")

    def refuse(url, coll):
        raise SystemExit("REFUSED: not the twin")

    with pytest.raises(SystemExit, match="not the twin"):
        sp.run_parity(requests(), base_tree=base, head_tree=base, collection="prod_x",
                      qdrant_url=free_url(), vec_cache=write_cache(tmp_path / "v.json"),
                      out_dir=tmp_path / "out", guard=refuse, declared_env={})
    assert not (tmp_path / "out").exists()


def test_numbers_compare_within_tolerance_and_structure_exactly():
    out: list = []
    sp._num_diff({"a": [1.0, 2.0]}, {"a": [1.0, 2.0 + 1e-12]}, 1e-9, "x", out)
    assert out == []
    sp._num_diff({"a": [1.0, 2.0]}, {"a": [2.0, 1.0]}, 1e-9, "x", out)
    assert len(out) == 2
    out = []
    sp._num_diff({"a": 1}, {"b": 1}, 1e-9, "x", out)
    assert out and "keys" in out[0]
    assert sp.max_abs_delta({"s": [1.0, 3.0]}, {"s": [1.0, 3.5]}) == 0.5


def test_probe_requests_keep_the_parameters_the_probe_sent(tmp_path):
    day = tmp_path / "runs" / "2026-10-08"
    day.mkdir(parents=True)
    (day / "sample.jsonl").write_text(json.dumps(
        {"query": "q1", "probe_id": "p1", "params": {"k": 5, "authorities": 1, "lang": "en"}}) + "\n")
    (day / "sample_sliceb.jsonl").write_text(json.dumps(
        {"qid": "b1", "query": "q2", "lang": "de", "country": "CH"}) + "\n")
    (tmp_path / "runs" / "README").write_text("not a day")
    reqs = sp.probe_requests(tmp_path / "runs", days=7)
    assert [r["params"] for r in reqs] == [
        {"q": "q1", "k": 5, "lang": "en", "authorities": 1},
        {"q": "q2", "k": 10, "lang": "de", "country": "CH"}]
    dup = sp.dedupe(reqs + [sp._req("other", "x", "q1", k=5, lang="en", authorities=1)])
    assert len(dup) == 2 and dup[0]["also"] == ["other:x"]


def test_a_dark_lever_is_armed_in_every_arm_and_a_stray_knob_in_none(tmp_path, monkeypatch):
    monkeypatch.setenv("AW_STRAY", "from the caller's shell")
    base = make_tree(tmp_path, "base")
    head = make_tree(tmp_path, "head")
    rep = run(tmp_path, base, head, deep=False, arm_extra={"AW_DARK_LEVER": "1"})
    assert rep["verdict"] == "PASS"
    assert rep["arm_extra_env"] == {"AW_DARK_LEVER": "1"}
    for arm in ("base", "head"):
        cfg = rep["fingerprints"][arm]["effective_config"]
        assert cfg["armed"] == "1" and cfg["stray"] is None


def test_the_hash_seed_pin_is_load_bearing(tmp_path, monkeypatch):
    """Finding D6: a server whose tie order is a set's iteration order answers differently in two
    processes unless PYTHONHASHSEED is pinned. With the pin: PASS, and the seed probe sees it.
    Without the pin (simulated: every arm salted at random): never PASS."""
    base = make_tree(tmp_path, "base", tweak="setorder")
    head = make_tree(tmp_path, "head", tweak="setorder")
    rep = run(tmp_path, base, head, deep=False, seed_probe=True)
    assert rep["verdict"] == "PASS", sp.summary(rep)
    assert rep["seed_probe"]["sensitive"] > 0
    env, _ = sp.arm_env({}, base, collection="c", qdrant_url=free_url(), deep=False)
    assert env["PYTHONHASHSEED"] == sp.SEED
    monkeypatch.setattr(sp, "SEED", "random")
    (tmp_path / "unpinned").mkdir()
    unpinned = run(tmp_path / "unpinned", base, head, deep=False)
    assert unpinned["verdict"] != "PASS", "the gate passed without a pinned hash seed"


def test_aw_gate_serve_parity_is_the_door(monkeypatch):
    from cytir import cli
    seen = []
    monkeypatch.setattr(sp, "main", lambda argv=None: seen.append(argv) or 0)
    assert cli.main(["gate", "serve-parity", "run", "--x"]) == 0
    assert seen == [["run", "--x"]]


# ---- task 841: `_comment` keys are documentation, not configuration ------------------------- #

def _write(p, doc):
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return p


def test_a_comment_only_difference_is_no_config_difference(tmp_path):
    a = _write(tmp_path / "a.json", {"_comment": "built by the old builder script", "hosts": {"x.ch": 1},
                                     "n": [{"_comment": "inner", "v": 2}]})
    b = _write(tmp_path / "b.json", {"_comment": "built by `cyt index`", "hosts": {"x.ch": 1},
                                     "n": [{"v": 2}]})
    assert sp._config_sha(a) == sp._config_sha(b)

    # ...and the loaded-table fingerprint (resolved path) sees no difference either
    class R:
        _GAZ_PATH = str(a)

    class S:
        _GAZ_PATH = str(b)
    assert sp._loaded_state(R)["_GAZ_PATH"]["sha256"] == sp._loaded_state(S)["_GAZ_PATH"]["sha256"]


def test_a_one_value_data_difference_in_the_same_file_is_a_config_difference(tmp_path):
    a = _write(tmp_path / "a.json", {"_comment": "same", "hosts": {"x.ch": 1}})
    b = _write(tmp_path / "b.json", {"_comment": "same", "hosts": {"x.ch": 2}})
    c = _write(tmp_path / "c.json", {"_comment": "same", "hosts": {"x.ch": 1}, "extra": 0})
    d = _write(tmp_path / "d.json", {"_comments": "x", "hosts": {"x.ch": 1}})  # only the exact key
    assert len({sp._config_sha(p) for p in (a, b, c, d)}) == 4


def test_non_json_and_unparseable_files_stay_byte_compared(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("x")
    b = tmp_path / "b.txt"
    b.write_text("y")
    assert sp._config_sha(a) != sp._config_sha(b)
    c = tmp_path / "c.json"
    c.write_text("{not json _comment")
    assert sp._config_sha(c) == sp._sha256(c)


def test_the_tree_digest_still_binds_the_raw_bytes(tmp_path):
    """Tree binding = bytes; config equality = semantics. A comment edit changes the tree digest
    (the base and head trees stay distinguishable) while it leaves the config hash alone."""
    d1, d2 = tmp_path / "t1" / "serve", tmp_path / "t2" / "serve"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)
    _write(d1 / "t.json", {"_comment": "old", "v": 1})
    _write(d2 / "t.json", {"_comment": "new", "v": 1})
    assert sp._source_digest(d1) != sp._source_digest(d2)
    assert sp._config_sha(d1 / "t.json") == sp._config_sha(d2 / "t.json")


def test_the_layout_constants_agree():
    from cytir.core import serve_layout
    assert sp.SERVER_REL == serve_layout.SERVER_REL        # the harness cannot import it (see serve_parity.BOOT)


def test_fixture_requests_are_the_ci_gold_slice_as_the_gate_sends_them(tmp_path):
    slice_ = tmp_path / "s.json"
    slice_.write_text(json.dumps({"golds": [
        {"id": "g1", "query": "alpha beta", "lang": "de"},
        {"id": "g2", "query": "gamma", "country": "CH"},
        {"id": "g3", "query": "alpha beta", "lang": "de"}]}))
    reqs = sp.dedupe(sp.fixture_requests(slice_))
    assert [r["id"] for r in reqs] == ["fixture:g1", "fixture:g2"]
    assert reqs[0]["params"] == {"q": "alpha beta", "k": 10, "lang": "de"}
    assert reqs[1]["params"] == {"q": "gamma", "k": 10, "country": "CH"}
    assert reqs[0]["also"] == ["fixture:g3"]
    with pytest.raises(SystemExit):
        sp.fixture_requests(tmp_path / "missing.json")


def test_the_shipped_fixture_slice_is_a_set_of_the_collector():
    import argparse
    ap = argparse.ArgumentParser()
    sp._add_sets(ap)
    reqs = sp.collect(ap.parse_args(["--sets", "fixture"]))
    assert len(reqs) >= 20 and all(r["set"] == "fixture" for r in reqs)


# --- task 2816: a config-only PR (the trees' declared envs differ, their serving inputs do not)

def with_compose(tree: pathlib.Path, env: dict) -> pathlib.Path:
    """Give a fake tree the deploy files `serving_env` reads: a compose aw-api block, no overlays."""
    deploy = tree / "deploy"
    deploy.mkdir(exist_ok=True)
    lines = ["services:", "  aw-api:", "    environment:"]
    lines += [f'      {k}: "{v}"' for k, v in env.items()] or ["      {}"]
    (deploy / "docker-compose.yaml").write_text("\n".join(lines) + "\n")
    (deploy / "compose-overlays.manifest").write_text("# none\n")
    return tree


def run_declared(tmp_path, base, head):
    """`run` with each tree's OWN declared env (no `declared_env` override), as CI runs it."""
    return sp.run_parity(requests(), base_tree=base, head_tree=head, collection="twin_x",
                         qdrant_url=free_url(), vec_cache=write_cache(tmp_path / "v.json"),
                         out_dir=tmp_path / "out", shards=1, deep=True,
                         guard=lambda url, coll: None, say=lambda m: None)


def test_a_config_only_change_runs_every_arm_under_the_head_env_and_passes(tmp_path):
    base = with_compose(make_tree(tmp_path, "base"), {"AW_FAKE_KNOB": "1", "AW_QDRANT_TIMEOUT": "5"})
    head = with_compose(make_tree(tmp_path, "head"), {"AW_FAKE_KNOB": "1", "AW_DARK_LEVER": "1",
                                                      "AW_QDRANT_TIMEOUT": "9"})
    rep = run_declared(tmp_path, base, head)
    assert rep["verdict"] == "PASS", sp.summary(rep)
    cc = rep["config_change"]
    assert cc["env_source"] == "head"
    # the timing knob differs too, but it is forced per arm: not a configuration change
    assert cc["changed"] == {"AW_DARK_LEVER": {"base": None, "head": "1"}}
    assert cc["serving_inputs_sha256"] == sp.serving_inputs_digest(base)
    for arm in ("base", "head"):     # the base tree ran the NEW env too
        assert rep["fingerprints"][arm]["effective_config"]["armed"] == "1"
    assert "CONFIG-ONLY" in sp.summary(rep)
    # never a release report: the trees hold the same server code
    bad = sp.check_report(rep, base_sha="b", head_sha="h", head_server_dir=head / "cytir" / "serve",
                          base_server_dir=base / "cytir" / "serve")
    assert any("compares a tree with itself" in x for x in bad)
    # nor through the two declared modes: the same-code refusal precedes the mode branch
    for mode in ({"ranking": {"ref": "", "max_diff": 99, "null_report": None, "repo_dir": tmp_path}},
                 {"envelope": {"ref": "", "fields": ["count"], "repo_dir": tmp_path}}):
        bad = sp.check_report(rep, base_sha="b", head_sha="h", head_server_dir=head / "cytir" / "serve",
                              base_server_dir=base / "cytir" / "serve", **mode)
        assert any("compares a tree with itself" in x for x in bad), mode


def with_profile_value(tree: pathlib.Path, value: str) -> pathlib.Path:
    """The declared serving-profile value as the package holds it (cytir/core/serving_config.py)."""
    core = tree / "cytir" / "core"
    core.mkdir(exist_ok=True)
    (core / "serving_config.py").write_text(f'OVERSAMPLING = "{value}"\n')
    return tree


def pr2197_shape(tmp_path, *, head_tweak=""):
    """The shape of the 256-pq8 oversampling PR: the compose env AND the value in cytir/core change,
    the serving code in cytir/serve is the same (or, with head_tweak, also moves)."""
    base = with_profile_value(with_compose(make_tree(tmp_path, "base"),
                                           {"AW_FAKE_KNOB": "1", "AW_DENSE_OVERSAMPLING": "2.0"}), "2.0")
    head = with_profile_value(with_compose(make_tree(tmp_path, "head", tweak=head_tweak),
                                           {"AW_FAKE_KNOB": "1", "AW_DENSE_OVERSAMPLING": "3.0"}), "3.0")
    return base, head


def test_a_pure_value_change_passes_with_the_head_code_under_the_base_env(tmp_path):
    base, head = pr2197_shape(tmp_path)
    assert sp.env_changes(sp.serving_env(base), sp.serving_env(head))
    assert sp.serving_inputs_digest(base) != sp.serving_inputs_digest(head)
    rep = run_declared(tmp_path, base, head)
    assert rep["verdict"] == "PASS", sp.summary(rep)
    cc = rep["config_change"]
    assert cc["env_source"] == "base" and cc["head_code_under_base_env"] is True
    assert cc["changed"] == {"AW_DENSE_OVERSAMPLING": {"base": "2.0", "head": "3.0"}}
    assert cc["serving_inputs_sha256"] == {"base": sp.serving_inputs_digest(base),
                                           "head": sp.serving_inputs_digest(head)}
    assert f"head code under base env = EQUAL on {len(QUERIES)} queries" in cc["receipt"]
    assert "'2.0' -> '3.0'" in cc["receipt"] and "serving inputs differ" in cc["receipt"]
    assert cc["moved_inputs"] == ["cytir/core/serving_config.py"]
    assert "cytir/core/serving_config.py" in cc["receipt"]
    assert cc["receipt"] in sp.summary(rep)
    # the head tree really ran the BASE env, not its own declared one
    for arm in ("base", "head"):
        assert rep["fingerprints"][arm]["effective_config"]["armed"] is None
    # never a release report, even though the server code (here) is the same
    bad = sp.check_report(rep, base_sha="b", head_sha="h", head_server_dir=head / "cytir" / "serve",
                          base_server_dir=base / "cytir" / "serve")
    assert any("under the BASE env" in x for x in bad)


def test_a_value_change_plus_a_serve_change_is_refused_before_any_run(tmp_path):
    base, head = pr2197_shape(tmp_path, head_tweak="swap")
    with pytest.raises(SystemExit, match=r"AW_DENSE_OVERSAMPLING.*cytir/serve/retrieval.py.*Split the PR"):
        run_declared(tmp_path, base, head)
    assert not (tmp_path / "out" / "parity.json").exists()


def test_a_serve_change_that_only_acts_under_the_new_value_is_refused(tmp_path):
    base, head = pr2197_shape(tmp_path, head_tweak="swap")
    p = head / "cytir" / "serve" / "retrieval.py"
    p.write_text(p.read_text().replace(
        'TWEAK == "swap" and',
        'TWEAK == "swap" and __import__("os").environ.get("AW_DENSE_OVERSAMPLING") == "3.0" and'))
    with pytest.raises(SystemExit, match=r"cytir/serve/retrieval.py.*Split the PR"):
        run_declared(tmp_path, base, head)
    assert not (tmp_path / "out" / "parity.json").exists()


def test_a_core_change_that_moves_an_answer_under_the_base_env_is_refused_with_the_diff_count(tmp_path):
    base, head = pr2197_shape(tmp_path, head_tweak="")
    for tree, v in ((base, "off"), (head, "on")):
        tweak = tree / "cytir" / "serve" / "retrieval.py"
        tweak.write_text(tweak.read_text().replace("TWEAK = ''", 'TWEAK = "swap"').replace(
            'TWEAK == "swap" and',
            'TWEAK == "swap" and (pathlib.Path(__file__).parents[1] / "core" / "flag.txt")'
            '.read_text() == "on" and'))
        (tree / "cytir" / "core" / "flag.txt").write_text(v)
    assert sp.moved_inputs(base, head) == ["cytir/core/flag.txt", "cytir/core/serving_config.py"]
    with pytest.raises(SystemExit, match=r"AW_DENSE_OVERSAMPLING.*[1-9]\d* DIFF.*Split the PR"):
        run_declared(tmp_path, base, head)
    rep = json.loads((tmp_path / "out" / "parity.json").read_text())
    assert rep["verdict"] == "FAIL" and rep["counts"]["DIFF"] >= 1


def test_a_config_change_with_an_unrelated_dark_lever_env_and_a_code_change_is_refused(tmp_path):
    base = with_compose(make_tree(tmp_path, "base"), {"AW_FAKE_KNOB": "1"})
    head = with_compose(make_tree(tmp_path, "head", tweak="swap"),
                        {"AW_FAKE_KNOB": "1", "AW_DARK_LEVER": "1"})
    with pytest.raises(SystemExit, match="AW_DARK_LEVER.*Split the PR"):
        run_declared(tmp_path, base, head)
    assert not (tmp_path / "out" / "parity.json").exists()


def test_a_config_change_with_a_data_change_is_refused_too(tmp_path):
    base = with_compose(make_tree(tmp_path, "base"), {"AW_FAKE_KNOB": "1"})
    head = with_compose(make_tree(tmp_path, "head"), {"AW_FAKE_KNOB": "2"})
    for tree, v in ((base, "1"), (head, "2")):
        (tree / "data").mkdir()
        (tree / "data" / "table.json").write_text(v)
    with pytest.raises(SystemExit, match=r"data/table.json.*Split the PR"):
        run_declared(tmp_path, base, head)
    assert not (tmp_path / "out" / "parity.json").exists()


def test_the_same_declared_env_in_both_trees_is_the_old_path(tmp_path):
    base = with_compose(make_tree(tmp_path, "base"), {"AW_DARK_LEVER": "1"})
    head = with_compose(make_tree(tmp_path, "head", tweak="swap"), {"AW_DARK_LEVER": "1"})
    rep = run_declared(tmp_path, base, head)
    assert rep["config_change"] is None
    assert rep["verdict"] == "FAIL"              # a code change under one env, as before


def test_the_serving_inputs_digest_covers_code_and_data_not_the_derived_env(tmp_path):
    tree = make_tree(tmp_path, "t")
    (tree / "data").mkdir()
    (tree / "data" / "a.json").write_text("1")
    (tree / "cytir" / "ops").mkdir()
    default = tree / "cytir" / "ops" / "serving_env.default.json"
    default.write_text('{"env": {"AW_X": "1"}}')
    d0 = sp.serving_inputs_digest(tree)
    default.write_text('{"env": {"AW_X": "2"}}')       # regenerated with the compose: the env itself
    assert sp.serving_inputs_digest(tree) == d0
    assert sp.serving_env(tree) == {"AW_X": "2"}      # no deploy/: the TREE's shipped default
    (tree / "cytir" / "serve" / "__pycache__").mkdir()
    (tree / "cytir" / "serve" / "__pycache__" / "x.pyc").write_bytes(b"\0")
    assert sp.serving_inputs_digest(tree) == d0
    (tree / "data" / "a.json").write_text("2")
    d1 = sp.serving_inputs_digest(tree)
    assert d1 != d0
    (tree / "cytir" / "core").mkdir()
    (tree / "cytir" / "core" / "m.py").write_text("X = 1\n")
    assert sp.serving_inputs_digest(tree) != d1


def test_the_digest_in_a_git_checkout_skips_ignored_files_not_untracked_ones(tmp_path):
    import subprocess
    tree = make_tree(tmp_path, "g")
    (tree / "data").mkdir()
    (tree / "data" / "a.json").write_text("1")
    (tree / ".gitignore").write_text("data/state.json\n")
    git = ["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "t"], check=True)
    d0 = sp.serving_inputs_digest(tree)
    (tree / "data" / "state.json").write_text("left by a job")      # ignored: not an input
    assert sp.serving_inputs_digest(tree) == d0
    (tree / "data" / "new.json").write_text("{}")                   # untracked, not ignored
    assert sp.serving_inputs_digest(tree) != d0


def test_an_empty_serving_code_listing_is_refused_never_an_empty_digest(tmp_path, monkeypatch):
    tree = tmp_path / "empty"
    (tree / sp.PACKAGE_NAMES[0]).mkdir(parents=True)
    (tree / sp.PACKAGE_NAMES[0] / "__init__.py").write_text("")
    monkeypatch.setattr(sp, "_input_files", lambda t, r: [])        # a listing that finds nothing
    with pytest.raises(SystemExit, match="no files"):
        sp.serving_inputs_digest(tree)
    monkeypatch.undo()
    with pytest.raises(SystemExit, match="no serving package"):
        sp.serving_inputs_digest(tmp_path / "nothing")
