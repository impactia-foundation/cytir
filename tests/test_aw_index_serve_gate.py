"""`cyt index build` -> `cyt serve` -> `cyt gate`, end to end, on the fixture packs (slice 8 step 5;
specs/aw-index-serve-gate.md section 4). This is the body of the CI job `index-serve-gate`, and it is
what a stranger runs: seven packs (`data/index-fixture/`, 96 rows of seven families) become a
collection, a server answers on it, and the gold slice is scored against the committed baseline.

WHAT A GREEN RUN PROVES, plainly: the PLUMBING. The vectors are STAND-INS (`tests/standin_embedder.py`:
a hashed bag of words for documents and queries, served to `cyt serve` by a TEI-shaped stub); the real
embedder needs torch and 2.3 GB of weights this job does not install. The BM25 leg, its IDF, the
projection, the fusion and the ranking code are the real ones; the dense leg's geometry is not the
model's. The verdict says so itself (`proves`, set from the embedder identity the stub announces to
`cyt serve`), not only by a caller's label. 96 points sit below Qdrant's indexing
threshold, so no PQ and no HNSW is exercised either.

Every verb runs as a subprocess of the real CLI (`python -m cytir.cli ...`), with only `AW_WORK_ROOT` set
(a stranger's environment). The Qdrant is the session's throwaway one (`tests/throwaway_qdrant.py`);
without `AW_TEST_QDRANT_URL` the module skips, and under `AW_REQUIRE_QDRANT=1` (the CI job) it fails.
"""
from __future__ import annotations

import gzip
import hashlib
import http.server
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

pytest.importorskip("yaml")
pytest.importorskip("qdrant_client")

import standin_embedder as SE  # noqa: E402  (tests/)
import throwaway_qdrant  # noqa: E402  (tests/: the shared gate of every Qdrant-backed test)
from cytir.core import serving_profile as SP  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = REPO / "data" / "index-fixture"
SLICE = REPO / "data" / "gate-slices" / "ci-fixture.v1.json"
BASELINE = REPO / "data" / "gate-slices" / "ci-fixture.baseline.v1.json"
FAMILIES = ("loda_exact", "fedlex_exact", "dila_kali", "service_public_fiche",
            "stackexchange_nonso_qa_p0", "openalex_academic_dump", "stackoverflow_qa")
#: the family whose stage the test MAKES with `cyt source run` (the others are sealed hand stages)
SOURCE_RUN = {"dila_kali": (REPO / "data" / "samples" / "dila-kali", "2026-10-03"),
              "service_public_fiche": (REPO / "data" / "samples" / "service-public-fiche",
                                       "2026-10-03"),
              "stackexchange_nonso_qa_p0": (REPO / "data" / "samples" / "stackexchange-nonso",
                                            "2026-10-03"),
              "openalex_academic_dump": (REPO / "data" / "samples" / "openalex-academic",
                                         "2026-10-03"),
              "stackoverflow_qa": (REPO / "data" / "samples" / "stackoverflow-qa", "2026-10-03")}
COLLECTION = "aw_gate_fixture"
AS_OF = "2026-10-03T00:00:00Z"                 # pins consent_checked_at and the IDF table's stamp
SHIPPED_TABLE = REPO / "cytir" / "serve" / "bm25_idf.json"
LABELS = ["--label", f"vectors=stand-in:{SE.KIND}", "--label", "packs=data/index-fixture"]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _env(work: pathlib.Path) -> dict:
    """A stranger's environment: the system's own variables and ONE AW setting."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AW_", "CYTIR_"))}
    work.mkdir(parents=True, exist_ok=True)         # the lock directory lives under it and must exist
    env["AW_WORK_ROOT"] = str(work)
    return env


def _aw(args, env, **kw):
    return subprocess.run([sys.executable, "-m", "cytir.cli", *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=600, **kw)


def _sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _source_run_stage(family: str, dst: pathlib.Path, work: pathlib.Path) -> None:
    """The family door, as a stranger runs it: `cyt source run FAMILY --from DIR --out STAGE`."""
    src, today = SOURCE_RUN[family]
    res = _aw(["source", "run", family, "--from", str(src), "--out", str(dst), "--today", today],
              _env(work))
    assert res.returncode == 0, res.stdout[-1500:] + res.stderr[-1500:]


def _stages(root: pathlib.Path, *, drop_url: str | None = None,
            rewrite: dict[str, str] | None = None) -> list[pathlib.Path]:
    """Copy the fixture stages and give them stand-in vectors. `drop_url` removes that row (and
    its vector) from its stage: the negative control. `rewrite={url: text}` replaces that row's text
    (title, snippet and both surfaces too) with `text`: the row stays PRESENT in the index, but says
    something else."""
    out = []
    for fam in FAMILIES:
        dst = root / fam
        if fam in SOURCE_RUN:
            _source_run_stage(fam, dst, root.parent / "stage-work")
        else:
            shutil.copytree(FIXTURES / fam, dst)
        shard = dst / "chunks" / "shard-0000.texts.jsonl.gz"
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            rows = [json.loads(x) for x in fh]
        kept = [r for r in rows if r["url"] != drop_url]
        for r in kept:
            if rewrite and r["url"] in rewrite:
                r["title"] = ""
                for field in ("text", "dense_text", "sparse_text", "snippet"):
                    if field in r:
                        r[field] = rewrite[r["url"]]
                if not r.get("anchor"):
                    # a digest-addressed row (I1 tier B, e.g. fedlex): its id follows its text, so the
                    # edited row gets the id its new text mints (an anchored row keeps its id)
                    from cytir.core import ids as aw_ids
                    from cytir.index import id_rule
                    r["point_id"], _why = id_rule.mint_i1(
                        {**r, "tier": "B", "content_digest": aw_ids.content_digest(r["text"])})
        if drop_url or rewrite:
            with open(shard, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                                         mtime=0) as gz:
                gz.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept).encode())
            meta = json.loads((dst / "PACK.json").read_text())
            meta["rows"] = len(kept)
            (dst / "PACK.json").write_text(json.dumps(meta))
        SE.write_vectors(dst)
        out.append(dst)
    return out


class _Stack:
    """A built collection, the stub query embedder and one `cyt serve` process on it."""

    def __init__(self, tmp: pathlib.Path, url: str, collection: str, stages, env):
        self.tmp, self.url, self.collection, self.env = tmp, url, collection, env
        self.run_dir = tmp / "run"
        self.cmd_build = ["index", "build", "--qdrant", url, "--collection", collection,
                          "--out", str(self.run_dir), "--as-of", AS_OF,
                          "--consent-cache-dir", str(tmp / "consent-cache"), "--recreate"]
        (tmp / "consent-cache").mkdir(exist_ok=True)
        self.stages = stages
        self.proc = self.tei = self.log = None
        self.port = _free_port()
        self.receipt = tmp / "serve-receipt.json"

    def build(self, run_dir=None):
        cmd = list(self.cmd_build)
        if run_dir:
            cmd[cmd.index("--out") + 1] = str(run_dir)
        for s in self.stages:
            cmd += ["--pack", str(s)]
        res = _aw(cmd, self.env)
        assert res.returncode == 0, res.stdout[-1500:] + res.stderr[-1500:]
        return json.loads((pathlib.Path(run_dir or self.run_dir) / "build-manifest.json").read_text())

    def _spawn(self):
        if self.log is not None:
            self.log.close()
        self.log = (self.tmp / "server.log").open("w")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "cytir.cli", "serve", "--collection", self.collection,
             "--qdrant", self.url, "--tei-url", f"http://127.0.0.1:{self.tei.server_address[1]}",
             "--idf-table", str(self.run_dir / "bm25_idf.json"), "--port", str(self.port),
             "--receipt", str(self.receipt)],
            cwd=REPO, env=self.env, stdout=self.log, stderr=subprocess.STDOUT)

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.api}/health", timeout=5) as r:
                return json.load(r).get("status") == "ok"
        except Exception:  # noqa: BLE001
            return False

    def serve(self):
        self.tei = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SE.tei_handler())
        threading.Thread(target=self.tei.serve_forever, daemon=True).start()
        for attempt in range(1, 4):
            self._spawn()
            deadline = time.time() + 180             # imports + uvicorn; a loaded runner is slow
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    log = (self.tmp / "server.log").read_text()
                    if "address already in use" in log and attempt < 3:
                        # task 2480: 24 runners share one box; the port picked by _free_port() was
                        # taken between the pick and the bind. Pick another and start again.
                        self.port = _free_port()
                        break
                    raise AssertionError(log[-3000:])
                if self._healthy():
                    return
                time.sleep(0.5)
            else:
                raise AssertionError("cyt serve did not become healthy:\n"
                                     + (self.tmp / "server.log").read_text()[-3000:])

    @property
    def api(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def gate(self, *extra):
        return _aw(["gate", "--api", self.api, "--slice", str(SLICE),
                    "--serve-receipt", str(self.receipt), *LABELS, *extra], self.env)

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()                  # the exact pid: `cyt serve` exec'd uvicorn
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.tei is not None:
            self.tei.shutdown()
        if self.log is not None:
            self.log.close()


def _delete(url: str, collection: str) -> None:
    from cytir.core.qdrant import qdrant_client
    throwaway_qdrant.assert_proven(url)
    client = qdrant_client(url, timeout=60)
    if client.collection_exists(collection):
        client.delete_collection(collection)



def test_serve_starts_again_on_a_new_port_when_the_port_was_taken_before_bind(tmp_path, monkeypatch):
    """task 2480: a port taken between _free_port() and the bind retries, on another port."""
    stack = _Stack(tmp_path, "http://127.0.0.1:1", COLLECTION, [], {})
    seen = []

    class Exited:
        def poll(self):
            return 3

    class Running:
        def poll(self):
            return None

    def fake_spawn(self):
        seen.append(self.port)
        if len(seen) == 1:
            self.proc = Exited()
            (self.tmp / "server.log").write_text(
                "[Errno 98] error while attempting to bind on address ('127.0.0.1', "
                f"{self.port}): address already in use\n")
        else:
            self.proc = Running()

    monkeypatch.setattr(_Stack, "_spawn", fake_spawn)
    monkeypatch.setattr(_Stack, "_healthy", lambda self: True)
    try:
        stack.serve()
    finally:
        stack.tei.shutdown()
        stack.tei.server_close()
    assert len(seen) == 2 and seen[0] != seen[1]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    url = throwaway_qdrant.require_throwaway()
    shipped_before = _sha(SHIPPED_TABLE)
    root = tmp_path_factory.mktemp("gate-world")
    env = _env(root / "work")
    stack = _Stack(root, url, COLLECTION, _stages(root / "stages"), env)
    t0 = time.time()
    first = stack.build(run_dir=root / "run-first")        # the build twice: same sidecars (below)
    second = stack.build()
    built = time.time() - t0
    stack.serve()
    try:
        yield {"stack": stack, "first": first, "manifest": second, "built_twice_s": built,
               "shipped_before": shipped_before, "url": url, "root": root}
    finally:
        stack.stop()
        # CI sets this so the NEXT step (`cyt index layout-check`, task 2405) reads the very collection
        # this world built; the throwaway Qdrant dies with the job, so nothing outlives it.
        if os.environ.get("AW_GATE_KEEP_COLLECTION") != "1":
            _delete(url, COLLECTION)


# ------------------------------------------------------------------ the build
def test_the_build_counts_balance_per_pack_and_in_total(world):
    m = world["manifest"]
    got = {p["pack_id"]: (p["records"], p["admitted"], p["written"], p["status"]) for p in m["packs"]}
    assert got == {"loda-exact-v1-local-fb45826c249c": (3, 3, 3, "LANDED"), "fedlex-exact-v1-local-779c076b42c1": (63, 63, 63, "LANDED"),
                   "dila-kali-v1-local-7105286712e7": (1, 1, 1, "LANDED"),
                   "service-public-fiche-v1-local-4a3a2e491a54": (6, 6, 6, "LANDED"),
                   "stackexchange-nonso-qa-p0-v1-local-ea6e89b9db45": (7, 7, 7, "LANDED"),
                   "openalex-academic-dump-v1-local-d75bceb26216": (6, 6, 6, "LANDED"),
                   "stackoverflow-qa-v1-local-27ffc4e09546": (10, 10, 10, "LANDED")}
    assert m["layout"]["points_count"] == 96 and m["fingerprints"]["count"] == 96
    # task 1271 (ruling R1): the unqualified `cyt index build --pack` is PROD_E9, no longer `served`
    assert m["profile"]["name"] == "prod-e9" and m["profile"]["digest"] == SP.PROD_E9.digest()
    # 96 points sit below the indexing threshold: `settled` says so, it is reported and not hidden
    assert any("below the indexing threshold" in x for x in m["layout_settled"])


def test_the_idf_is_this_builds_own_the_shipped_table_is_untouched(world):
    m = world["manifest"]
    assert m["idf_snapshot"]["n"] == m["layout"]["points_count"] == 96
    table = json.loads((world["stack"].run_dir / "bm25_idf.json").read_text())
    assert table["collection"] == COLLECTION and table["points_count"] == 96
    assert table["measured_at"] == AS_OF and table["fold_accents"] is True
    assert table["n_measured"] == m["idf_snapshot"]["vocabulary"] > 1000
    assert m["idf_table"]["sha256"] == _sha(world["stack"].run_dir / "bm25_idf.json")
    # measured through Qdrant == the integer-count formula over the snapshot (a fresh build has no
    # stale postings), to the table's 4-decimal rounding
    from cytir.distribute import idf as SNAP
    df = dict(SNAP.iter_snapshot(world["stack"].run_dir / "bm25-idf.tsv.gz"))
    worst = max(abs(SNAP.idf(96, df[int(h)]) - v) for h, v in table["idf"].items())
    assert worst < 1e-3, worst
    assert _sha(SHIPPED_TABLE) == world["shipped_before"]            # prod's table: never touched
    assert table["collection"] != json.loads(SHIPPED_TABLE.read_text())["collection"]


def test_a_second_build_gives_the_same_sidecars_and_the_same_table(world):
    a, b = world["first"], world["manifest"]
    assert a["fingerprints"]["sha256"] == b["fingerprints"]["sha256"]
    assert a["idf_snapshot"]["sha256"] == b["idf_snapshot"]["sha256"]
    assert a["idf_table"]["sha256"] == b["idf_table"]["sha256"]


# ------------------------------------------------------------------ the server
def test_the_server_receipt_names_the_profile_the_env_files_and_this_builds_table(world):
    stack = world["stack"]
    r = json.loads(stack.receipt.read_text())
    assert r["collection"] == COLLECTION and r["points_count"] == 96
    assert r["profile"]["digest"] == SP.PROD_E9.digest()
    assert r["idf_table"]["explicit"] is True and r["idf_table"]["matches_target"] is True
    assert r["idf_table"]["sha256"] == world["manifest"]["idf_table"]["sha256"]
    assert r["query_embedding"]["mode"] == "tei"             # the stand-in stub, not the model
    assert r["env_sha256"] and r["env"]["AW_MRL_DIM"]["source"] == "profile prod-e9 dense_dim"


# ------------------------------------------------------------------ the gate
def test_the_gate_passes_against_the_committed_baseline(world, tmp_path):
    stack = world["stack"]
    out = tmp_path / "verdict.json"
    res = stack.gate("--baseline", str(BASELINE), "--verdict-out", str(out))
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    v = json.loads(out.read_text())
    assert v["pass"] is True and v["baseline"]["regressions"] == []
    assert v["slice"]["sha256"] == _sha(SLICE) and v["slice"]["n_golds"] == 24
    assert v["serve_receipt"]["sha256"] == _sha(stack.receipt)
    assert v["serve_receipt"]["profile"]["digest"] == SP.PROD_E9.digest()
    assert v["server"]["points"] == 96 and v["server"]["collection"] == COLLECTION
    assert all(g["hit_at_k"] for g in v["golds"])
    # the table of THIS build is what the baseline measured on: same slice, same packs, same IDF
    assert v["baseline"]["identity_equal"] == {"serve_receipt.profile.digest": True,
                                               "serve_receipt.idf_table.sha256": True}


def test_the_baseline_is_a_verdict_of_this_slice_and_says_it_proves_plumbing_only():
    b = json.loads(BASELINE.read_text())
    assert b["schema"] == "aw.gate-verdict.v1" and b["slice"]["sha256"] == _sha(SLICE)
    assert b["labels"]["vectors"] == f"stand-in:{SE.KIND}"
    # a first-class field, set by the gate from the embedder identity the stub announced
    assert b["proves"] == "plumbing-only (stand-in vectors; not a ranking claim)"
    assert b["serve_receipt"]["query_embedding"]["info"]["stand_in"] == SE.KIND
    assert len(b["golds"]) == 24 and b["pass"] is None


def test_every_verdict_of_a_stand_in_server_says_plumbing_only(world, tmp_path):
    out = tmp_path / "v.json"
    res = world["stack"].gate("--baseline", str(BASELINE), "--verdict-out", str(out))
    assert res.returncode == 0
    assert json.loads(out.read_text())["proves"] == "plumbing-only (stand-in vectors; not a ranking claim)"
    assert "proves: plumbing-only" in res.stdout


def _row_text(url: str) -> str:
    for fam in FAMILIES:
        with gzip.open(FIXTURES / fam / "chunks" / "shard-0000.texts.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["url"] == url:
                    return row["text"]
    raise KeyError(url)


def _control(world, tmp_path_factory, suffix, **stage_kw):
    """The same build and server over a doctored copy of the fixture stages. The doctored index
    has another IDF table than the baseline's, so the gate REFUSES the comparison until it is told
    (`--allow-identity-mismatch`); that is checked here, then the verdict is read with the flag."""
    root = tmp_path_factory.mktemp(f"gate-{suffix}")
    stack = _Stack(root, world["url"], f"{COLLECTION}_{suffix}", _stages(root / "stages", **stage_kw),
                   _env(root / "work"))
    stack.build()
    stack.serve()
    try:
        refused = stack.gate("--baseline", str(BASELINE))
        assert refused.returncode == 3, refused.stdout[-1500:] + refused.stderr[-1500:]
        assert "serve_receipt.idf_table.sha256" in refused.stderr and "baseline" in refused.stderr
        out = root / "verdict.json"
        res = stack.gate("--baseline", str(BASELINE), "--allow-identity-mismatch",
                         "--verdict-out", str(out))
        return res, (json.loads(out.read_text()) if out.exists() else None)
    finally:
        stack.stop()
        _delete(world["url"], f"{COLLECTION}_{suffix}")


def test_the_gate_is_red_when_one_gold_document_is_missing_from_the_index(world, tmp_path_factory):
    """The negative control: the same build and server with ONE gold's row left out of its pack.
    The gate must fail on that gold, by name."""
    drop = "https://www.fedlex.admin.ch/eli/cc/1948/195_183_155/de#art_14"
    res, v = _control(world, tmp_path_factory, "minus", drop_url=drop)
    assert res.returncode == 1, res.stdout[-2000:] + res.stderr[-2000:]
    assert v["pass"] is False and v["baseline"]["identity_mismatch_allowed"] is True
    assert {r["id"] for r in v["baseline"]["regressions"]} == {"ci-fedlex-de-1"}
    gold = next(g for g in v["golds"] if g["id"] == "ci-fedlex-de-1")
    assert gold["rank_exact"] is None and drop not in gold["top"]


def test_the_gate_is_red_when_a_gold_is_still_present_but_ranks_worse(world, tmp_path_factory):
    """The second negative control, "worse but present": the gold `ci-fedlex-fr-1` keeps its row in
    the index, but the row now holds only the first 150 characters of its text, so the sibling
    articles of its page answer its query better. The gold is still in the top-k at exact rank >= 2;
    the gate must fail on it, by name, on `rank_exact`, and on no other gold."""
    gold_id = "ci-fedlex-fr-1"
    gold = next(g for g in json.loads(SLICE.read_text())["golds"] if g["id"] == gold_id)
    res, v = _control(world, tmp_path_factory, "worse",
                      rewrite={gold["target_url"]: _row_text(gold["target_url"])[:150]})
    assert res.returncode == 1, res.stdout[-2000:] + res.stderr[-2000:]
    row = next(g for g in v["golds"] if g["id"] == gold_id)
    # still present (in the top-k) and worse by the exact URL: rank 1 -> >= 2. The page-level `rank`
    # may stay 1, because a sibling article of the same page answers: `rank_exact` is why the gate
    # scores both.
    assert row["hit_at_k"] is True and row["rank_exact"] is not None and row["rank_exact"] >= 2
    assert [(r["id"], r["measure"]) for r in v["baseline"]["regressions"]] == [(gold_id, "rank_exact")]


@pytest.mark.skipif(os.environ.get("AW_GATE_RECORD_BASELINE") != "1",
                    reason="set AW_GATE_RECORD_BASELINE=1 to (re)record the committed baseline")
def test_record_the_baseline(world):
    """How `data/gate-slices/ci-fixture.baseline.v1.json` is made: this same world, no baseline
    given. Run it on purpose, review the diff, commit it. It writes the baseline and its
    `golds_outcome_digest`, never the sealed contract: a moved gold outcome fails the seal test
    until the contract file changes in a POLICY PR."""
    res = world["stack"].gate("--verdict-out", str(BASELINE))
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
