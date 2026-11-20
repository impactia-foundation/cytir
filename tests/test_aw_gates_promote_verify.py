#!/usr/bin/env python3
"""The promote gate judges the RATIFIED shape, not a literal it kept a copy of.

Task 529, from the task-529 gate consolidation. `verify_promote.py` is the tool
the task 508 flip-day runbook names for the promote verification, and until
this change its Tier-1 shape check read:

    shape_ok = (dense["size"] == 128 and dense["on_disk"] is True
                and "bm25" in sparse and hnsw_m == 16 and quant == "int8")

Four literals describing the RETIRED 128-int8 "config B". Against the ratified
256-pq8 flip candidate both `dense["size"] == 128` and `quant == "int8"` are
false, so a perfectly correct candidate scored **FAIL** — and `quant` was
parsed only out of `quantization_config.scalar`, so a product-quantized
collection read back as `None` whatever it actually held.

The fix is a deletion, not a parameterisation: the rules live in
`cytir.core.serving_config`, which already had the same check for the
client-side path, and both sides now share `collection_shape_problems`.
Engineering doctrine 1 — a copy is a forked contract, not parity.

The collection bodies below are the shapes qdrant actually returns, taken from
the sealed arm seals of tasks 519 and 524 (`aw_e6`/`aw_e8`: dense size 256,
`float16`, `quantization_config.product`, hnsw m=16, bm25 sparse).

Run: python3 -m pytest tests/test_aw_gates_promote_verify.py
"""
from __future__ import annotations

import copy
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parents[1] / "cytir" / "gates"
from cytir.core import serving_config as sc  # noqa: E402
from cytir.gates import promote_verify as vp  # noqa: E402


def _collection(size, datatype, quant_key, *, points=30_897_110,
                indexed=30_897_110, m=16, status="green", sparse=("bm25",)):
    quant = {quant_key: {"compression": "x8", "always_ram": False}} \
        if quant_key else None
    return {
        "status": status,
        "points_count": points,
        "indexed_vectors_count": indexed,
        "config": {
            "params": {
                "vectors": {"dense": {"size": size, "distance": "Cosine",
                                      "on_disk": True, "datatype": datatype}},
                "sparse_vectors": {k: {} for k in sparse},
            },
            "hnsw_config": {"m": m, "ef_construct": 100, "on_disk": True},
            "quantization_config": quant,
        },
    }


#: The ratified flip candidate, as `aw_e8` reports itself.
E8 = _collection(256, "float16", "product")
#: The retired config B, as the old literal described it.
B128 = _collection(128, "float32", "scalar")


def _flip_cfg():
    return sc.get(sc.FLIP_SERVING_CONFIG)


# ------------------------------------------------- the defect, stated as a test
def test_the_ratified_flip_candidate_passes():
    """256-pq8 is the shape the flip ships. The old check FAILED it."""
    problems = sc.collection_shape_problems(sc.shape_from_json(E8),
                                            _flip_cfg())
    assert problems == [], problems


def test_the_retired_config_b_fails_against_the_flip_config():
    problems = sc.collection_shape_problems(sc.shape_from_json(B128),
                                            _flip_cfg())
    assert any("dense size 128" in p for p in problems), problems
    assert any("quantizer read back as 'scalar'" in p for p in problems), \
        problems


def test_a_product_quantizer_is_not_read_back_as_none():
    """The old parse looked only in `quantization_config.scalar`."""
    assert sc.shape_from_json(E8)["quantizer_kind"] == "product"


def test_verify_promote_keeps_no_copy_of_the_shape_literals():
    src = (HERE / "promote_verify.py").read_text()
    assert 'dense["size"] == 128' not in src
    assert '"int8"' not in src, \
        "a quantizer literal reappeared in promote_verify.py"
    assert "collection_shape_problems" in src
    assert "serving_config" in src


def test_the_gate_reads_its_target_shape_from_serving_config():
    """`--serving-config` defaults to the ratified flip config, not a literal."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--serving-config", default=sc.FLIP_SERVING_CONFIG)
    assert ap.parse_args([]).serving_config == sc.FLIP_SERVING_CONFIG
    assert sc.FLIP_SERVING_CONFIG == "256-pq8"


# ------------------------------------------- the rules that are this gate's own
def test_a_missing_bm25_leg_is_a_failure():
    """The lexical leg is not optional: ADR-004's dense+BM25 fusion."""
    no_sparse = _collection(256, "float16", "product", sparse=())
    shape_problems = sc.collection_shape_problems(
        sc.shape_from_json(no_sparse), _flip_cfg())
    assert shape_problems == []          # the SERVING config says nothing here
    sparse = list((no_sparse["config"]["params"].get("sparse_vectors") or {}))
    assert "bm25" not in sparse          # ... so promote_verify must check it


def test_a_bulk_build_left_at_m_zero_is_a_failure():
    """A collection created at m=0 and never finalised answers every query by
    full scan (memory: qdrant-bulk-build-dim-change)."""
    assert vp.EXPECTED_HNSW_M == 16
    unfinalised = _collection(256, "float16", "product", m=0)
    assert unfinalised["config"]["hnsw_config"]["m"] != vp.EXPECTED_HNSW_M


# --------------------------------------------- the appendable-segment tripwire
def test_an_unoptimised_candidate_is_refused():
    """Product quantization is SKIPPED in appendable segments, so an
    unoptimised collection answers from an UNQUANTIZED scan — measuring it is
    measuring a different index (task 412)."""
    lagging = _collection(256, "float16", "product",
                          points=30_897_110, indexed=1_000_000)
    problems = sc.collection_shape_problems(sc.shape_from_json(lagging),
                                            _flip_cfg())
    assert any("indexed_vectors_count" in p for p in problems), problems


def test_a_non_green_collection_is_refused():
    yellow = _collection(256, "float16", "product", status="yellow")
    problems = sc.collection_shape_problems(sc.shape_from_json(yellow),
                                            _flip_cfg())
    assert any("not green" in p for p in problems), problems


def test_check_indexed_false_skips_only_the_residency_rules():
    lagging = _collection(256, "float16", "product", indexed=1)
    assert sc.collection_shape_problems(sc.shape_from_json(lagging),
                                        _flip_cfg(),
                                        check_indexed=False) == []


# ----------------------------------------------------- the two readers agree
class _FakeDense:
    def __init__(self, size, datatype):
        self.size, self.datatype = size, datatype


class _FakeQC:
    def __init__(self, kind):
        for k in ("scalar", "product", "binary", "turbo"):
            setattr(self, k, {} if k == kind else None)


class _FakeInfo:
    def __init__(self, body):
        params = body["config"]["params"]
        dense = params["vectors"]["dense"]
        kind = next((k for k in ("scalar", "product", "binary", "turbo")
                     if (body["config"]["quantization_config"] or {}).get(k)),
                    None)
        self.config = type("C", (), {
            "params": type("P", (), {
                "vectors": {"dense": _FakeDense(dense["size"],
                                                dense["datatype"])}})(),
            "quantization_config": _FakeQC(kind) if kind else None})()
        self.points_count = body["points_count"]
        self.indexed_vectors_count = body["indexed_vectors_count"]
        self.status = body["status"]


@pytest.mark.parametrize("body", [E8, B128])
def test_the_client_reader_and_the_json_reader_agree(body):
    """A mirror contract: two ways to read one collection must not disagree.

    The whole reason this defect existed is that the promote gate read the
    collection over raw HTTP and the builder read it through the client, so
    the two grew apart.
    """
    from_json = sc.shape_from_json(body)
    from_info = sc.shape_from_info(_FakeInfo(copy.deepcopy(body)))
    assert from_json == from_info


# ---------------------------------------------- the CLI, against local stubs
# Task 2154. The CLI is the contract the promote runbook runs on the boxes, so
# both tiers are driven end to end against a stdlib HTTP stub on 127.0.0.1:
# never a real qdrant, never a real API.
import http.server  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


class _Stub:
    """Qdrant REST + search API in one stdlib server."""

    def __init__(self, collections, domain_counts=None, search_top="docs.python.org"):
        stub = self
        self.collections = collections          # name -> collection body
        self.domain_counts = domain_counts or {}  # (name, domain) -> n
        self.search_top = search_top

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.startswith("/search?"):
                    return self._send({"results": [
                        {"url": f"https://{stub.search_top}/x"}]})
                name = self.path.split("/")[2]
                self._send({"result": stub.collections[name]})

            def do_POST(self):
                name = self.path.split("/")[2]
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                must = (body.get("filter") or {}).get("must")
                if must:
                    n = stub.domain_counts.get((name, must[0]["match"]["value"]), 1)
                else:
                    n = stub.collections[name]["points_count"]
                self._send({"result": {"count": n}})

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def stub():
    made = []

    def make(*a, **kw):
        s = _Stub(*a, **kw)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


def _run(args, capsys):
    rc = vp.main(args)
    return rc, capsys.readouterr().out


def test_tier1_passes_the_ratified_candidate(stub, capsys):
    s = stub({"cand": E8, "base": E8})
    rc, out = _run(["--qdrant-url", s.url, "--collection", "cand",
                    "--base-collection", "base", "--expect", "30897110"], capsys)
    assert rc == 0
    assert "== TIER 1 STRUCTURAL: cand @ " + s.url + " ==" in out
    assert "  OK count\n" in out
    assert "  OK 256-pq8 shape\n" in out
    assert "  OK no gold-domain regression vs base\n" in out
    assert "TIER1: PASS\n" in out and out.endswith("VERIFY: PASS\n")


def test_tier1_refuses_a_wrong_count_and_a_gold_domain_regression(stub, capsys):
    s = stub({"cand": E8, "base": E8},
             domain_counts={("base", "arxiv.org"): 5, ("cand", "arxiv.org"): 2})
    rc, out = _run(["--qdrant-url", s.url, "--collection", "cand",
                    "--base-collection", "base", "--expect", "1"], capsys)
    assert rc == 1
    assert "  FAIL count 30,897,110 != expect 1\n" in out
    assert "gold domains REGRESSED (base>cand): [('arxiv.org', 5, 2)]" in out
    assert "TIER1: FAIL (2 checks)\n" in out and out.endswith("VERIFY: FAIL (2 total)\n")


def test_tier1_refuses_the_retired_config_b_shape(stub, capsys):
    s = stub({"cand": B128})
    rc, out = _run(["--qdrant-url", s.url, "--collection", "cand"], capsys)
    assert rc == 1
    assert "  FAIL 256-pq8 shape: " in out
    assert "WARN 0 gold domains absent" in out


def test_tier2_passes_and_refuses_a_flooded_rank_one(stub, capsys):
    ok = stub({}, search_top="docs.python.org")
    rc, out = _run(["--api", ok.url], capsys)
    assert rc == 0
    assert "  no eval queries loaded (need --eval-dir with yaml) — flood-only\n" in out
    assert "  flood probe rank-1: docs.python.org -> ok\n" in out
    assert "TIER2: PASS\n" in out
    bad = stub({}, search_top="canada.ca")
    rc, out = _run(["--api", bad.url], capsys)
    assert rc == 1
    assert "  FAIL flood query rank-1 is a gov/reference flood host\n" in out
    assert out.endswith("VERIFY: FAIL (1 total)\n")


def test_nothing_to_do_exits_with_the_old_message():
    with pytest.raises(SystemExit) as e:
        vp.main([])
    assert "nothing to do: pass --qdrant-url/--collection (tier1)" in str(e.value)


@pytest.mark.parametrize("door", [["-m", "cytir.gates.promote_verify"],
                                  ["-m", "cytir.cli", "gate", "promote-verify"]])
def test_both_doors_run_the_same_gate(stub, door):
    s = stub({"cand": E8})
    r = subprocess.run([sys.executable, *door, "--qdrant-url", s.url,
                        "--collection", "cand", "--expect", "30897110"],
                       cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "TIER1: PASS" in r.stdout


def test_api_search_is_not_routed_through_the_internal_host_guard(monkeypatch):
    """`--api` may be the operator's public API on any host (task 2160/2162): an https name that is
    neither loopback nor a configured service must still be reached, by the public-self helper."""
    import io
    from cytir.core import internal_http as ih
    from cytir.sources import http as sources_http
    api = "https://api.selfhost.example"
    with pytest.raises(sources_http.InternalHostRefused):
        sources_http.make_internal_client(api).close()      # the internal guard WOULD refuse this host
    seen = []

    def fake(url, *, timeout, **kw):
        seen.append((url, timeout))
        return io.BytesIO(b'{"results": [{"url": "https://x.example/x"}]}')
    monkeypatch.setattr(ih, "public_self_urlopen", fake)
    res = vp.api_search("python", 3, api, "")
    assert res and res[0]["url"].endswith("/x")
    assert seen == [("https://api.selfhost.example/search?q=python&k=3", 30)]
