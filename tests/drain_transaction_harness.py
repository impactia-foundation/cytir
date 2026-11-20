"""Shared harness of the drain-transaction tests (slice 8 step 2, specs/aw-index-serve-gate.md §2/§5).

Drives ONE sealed real pack through the live drain against throwaway Qdrant collections and returns
everything the drain leaves behind, normalised, so two runs (before/after a change, or twin-first vs
single) can be compared byte for byte.

The pack is real: the four CASS/JADE decisions of `tests/fixtures/dila_juri/`, run through the
`dila_juri` family door, staged, and composed by the live `compose_pack`. Only the vector shard is
a stand-in (deterministic pseudo-random 1024-d fp16 rows in the sealed shard format): the drain's
write path never looks at the vector VALUES, and the embedder has its own tests.

NEVER point this at a live Qdrant. `qdrant_url()` asks the endpoint what it HOLDS before the first
write (`tests/throwaway_qdrant.py`: no alias, no declared live collection, and a marker the session
put on an EMPTY instance), and every write or delete in this module is checked against that proof.
The port list is a third belt only.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import urllib.parse
import urllib.request

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent          # the app root: where `data/promotion-gate-contract.v1.json` lives
for _p in (str(HERE.parent), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import throwaway_qdrant as TQ  # noqa: E402  (this directory, put on sys.path above)

FAMILY = "dila_juri"
MRL = 128
TWIN = "aw_e8"        # a collection NAME the payload-profile declaration knows (profile `prod`)
PROD = "aw_e9"        # ... and the one prod serves through the alias (profile `normalized`)
ALIAS = "aw_prod"
FIXTURE_DIR = HERE / "fixtures" / "dila_juri"


def qdrant_url() -> str:
    """The throwaway Qdrant's URL, PROVEN by what it holds (see `tests/throwaway_qdrant.py`)."""
    return TQ.require_throwaway()


def api(url: str, path: str, body=None, method: str | None = None):
    if body is not None or (method or "GET") != "GET":
        TQ.assert_proven(url)                 # before ANY write or delete: the proof of the session
    req = urllib.request.Request(
        url.rstrip("/") + path, method=method or ("PUT" if body is not None else "GET"),
        data=None if body is None else json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r).get("result")


def reset_collection(url: str, name: str, *, dim: int = MRL) -> None:
    """An EMPTY collection of the served shape's names: dense (cosine) + bm25 (idf sparse)."""
    TQ.assert_proven(url)       # before even the listing: nothing is sent to an unproven endpoint
    existing = {c["name"] for c in api(url, "/collections")["collections"]}
    if name in existing:
        api(url, f"/collections/{name}", method="DELETE")
    api(url, f"/collections/{name}", {
        "vectors": {"dense": {"size": dim, "distance": "Cosine", "on_disk": True}},
        "sparse_vectors": {"bm25": {"modifier": "idf"}},
        "on_disk_payload": True})


def set_alias(url: str, alias: str, collection: str) -> None:
    TQ.assert_proven(url)
    api(url, "/collections/aliases", {"actions": [
        {"delete_alias": {"alias_name": alias}},
        {"create_alias": {"alias_name": alias, "collection_name": collection}}]}, method="POST")


def drop_alias(url: str, alias: str) -> None:
    TQ.assert_proven(url)       # outside the try below: a refusal is never swallowed
    try:
        api(url, "/collections/aliases", {"actions": [{"delete_alias": {"alias_name": alias}}]},
            method="POST")
    except Exception:    # noqa: BLE001 -- an alias that does not exist is fine
        pass


LIVE_NAMES = (TWIN, PROD, "source_cards")     # names a build's target endpoint must not hold


def clear_live(url: str) -> None:
    """Make the throwaway endpoint look like a STRANGER's: no prod alias, none of the collections a
    live box declares. A build transaction refuses an endpoint that holds them (by what it HOLDS,
    not how its URL is spelled), so a test that runs the twin-first drain and then a build on the
    one throwaway Qdrant clears the twin-first world between the two."""
    TQ.assert_proven(url)
    drop_alias(url, ALIAS)
    held = {c["name"] for c in api(url, "/collections")["collections"]}
    for name in LIVE_NAMES:
        if name in held:
            api(url, f"/collections/{name}", method="DELETE")


def write_vectors(vectors_dir: pathlib.Path, ids: list[str], dim: int = 1024) -> None:
    """The sealed shard format `promote_sealed_pack.read_vectors` reads, with deterministic rows
    (a function of the id only), so every run of every test sees the same vectors."""
    vectors_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in ids:
        seed = int.from_bytes(hashlib.sha256(pid.encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(dim).astype("float32")
        rows.append((v / np.linalg.norm(v)).astype("<f2"))
    np.stack(rows).tofile(vectors_dir / "shard-0000.vectors.f16")
    with open(vectors_dir / "shard-0000.row_ids.jsonl", "w") as fh:
        for pid in ids:
            fh.write(json.dumps({"point_id": pid}) + "\n")
    from cytir.core.model_pin import EMBED_MODEL_ID, embed_model_revision
    (vectors_dir / "shard-0000.vectors.manifest.json").write_text(json.dumps(
        {"schema": "aw.413a.vectors.v1", "shard": "shard-0000", "n": len(ids), "dim": dim,
         "dtype": "float16",
         # the record `cyt embed shards` writes, so `cyt index build`'s pin check has something to
         # read; the rows are NOT real embeddings, and the manifest says so
         "embedder": {"model": EMBED_MODEL_ID, "revision_pinned": embed_model_revision()},
         "stand_in_for_tests": True}))


AS_OF = "2026-10-03T00:00:00Z"


def _empty_dir(p: pathlib.Path) -> pathlib.Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sealed_pack(tmp: pathlib.Path, *, collection: str = TWIN, queue_name: str = "queue") -> pathlib.Path:
    """Run the dila_juri family over the four real decisions, stage it, give it vectors, and compose
    it into `tmp/<queue_name>/<pack_id>/` with the live `compose_pack`. Returns the queue dir."""
    from cytir.consent import stamp as CONSENT_STAMP
    from cytir.index import compose
    from cytir.sources import family_stage as FS, run_family

    root = tmp / "dump"
    d = root / "cass"
    d.mkdir(parents=True, exist_ok=True)
    for f in sorted(FIXTURE_DIR.glob("*.xml")):
        shutil.copy(f, d / f.name)
    out = tmp / "run"
    run_family.run_family(FAMILY, out_dir=out, today="2026-10-03",
                          consent=lambda h, dd: pytest.fail("dump route: no host consent"),
                          options={"root": str(root)}, log=lambda m: None)
    FS.stage_family_pack(out / "pack", corpus_root=tmp / "corpus", family_id=FAMILY,
                         log=lambda m: None)
    (_, stage), = FS.list_stages(tmp / "corpus", FAMILY)
    with gzip.open(stage / "chunks" / "shard-0000.texts.jsonl.gz", "rt", encoding="utf-8") as fh:
        ids = [json.loads(x)["point_id"] for x in fh]
    vectors = tmp / "vectors"
    write_vectors(vectors, ids)
    queue = tmp / queue_name
    res = compose.compose_pack(
        pack_id="dila-juri-v1-shard0", family=FAMILY, chunks_dir=str(stage / "chunks"),
        vectors_dir=str(vectors), collection=collection, mrl=MRL, queue_dir=str(queue),
        # an EMPTY consent cache dir: a dump-route family derives consent from its declared basis
        # and reads no host verdict, and a CI runner has no ~/aw-data/consent to read
        consent_ctx=CONSENT_STAMP.load_context(as_of=AS_OF, cache_dir=str(_empty_dir(tmp / "consent-cache"))))
    # `detect_language: False` is the documented per-pack override every drain-level test uses: it
    # keeps the fasttext detector (a model file this test must not need) out of the content check.
    mp = pathlib.Path(res["pack_dir"]) / "manifest.json"
    manifest = json.loads(mp.read_text())
    manifest["detect_language"] = False
    body = json.dumps(manifest, indent=1, sort_keys=True).encode()
    mp.write_bytes(body)
    (mp.parent / "READY").write_text(hashlib.sha256(body).hexdigest() + "\n")
    return queue


# ---------------------------------------------------------------- the fakes of the host's locks
class FakeLock:
    """`twin_first_apply.ProductionWriterLock` without the ssh to the prod host."""
    log: list = []

    def __init__(self, *_a, **_k):
        pass

    def acquire(self) -> None:
        FakeLock.log.append("prod-lock-acquire")

    def release(self) -> None:
        FakeLock.log.append("prod-lock-release")


def install_fakes(monkeypatch, Q, PSP) -> list:
    """Stub the two things a drain needs a HOST for: the writer locks, and the persisted-input index
    (ADR-105: a corpus-wide sqlite tree the test has no use for). Everything else is real: Qdrant,
    the projection, the plan, the content check, the transaction protocol."""
    calls: list = []
    FakeLock.log = calls

    class FakeRB:
        from cytir.ops.write_lock import LOCKS

        @staticmethod
        def assert_locks_free(task):
            calls.append(("assert_locks_free", task))
            return {}

        @staticmethod
        def acquire(name, task, purpose, *, pid=None):
            calls.append(("acquire", name, task, purpose))

        @staticmethod
        def release(name, task):
            calls.append(("release", name, task))

    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", FakeLock)
    # the drain asks the settings for the production writer-lock HOST before it takes the lock; the lock
    # is faked, so the host is a constant: the tests need no operator overlay (a stranger's box has none)
    monkeypatch.setattr(Q, "prod_lock_host", lambda: "lockhost")
    monkeypatch.setattr(Q, "_load_run_batch", lambda: FakeRB)
    monkeypatch.setattr(PSP.PI, "assert_pages_persisted",
                        lambda pages, **k: {"verdict": "PASS", "pages": len(list(pages)),
                                            "stubbed_by": "drain_transaction_harness"})
    return calls


# ---------------------------------------------------------------- what a run leaves behind
_TS = re.compile(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|\+00:00)?")
_SECONDS = re.compile(r'("(?:write_seconds|plan_seconds|seconds|elapsed_s|duration_s)": )[0-9.eE+-]+')


#: The manifest's bytes embed this run's temp directory, so ITS sha differs run to run by
#: construction; every other digest in a receipt is data and is compared as it is.
_MANIFEST_SHA = re.compile(r'("manifest_sha": ")[0-9a-f]{64}')


#: `consent_stamp.contracts_sha256` in a COMPOSED manifest is the sha of `data/source-family-contracts.v1.json`
#: at compose time: an input of the pack, not something the drain writes, and a policy file that
#: changes with every family ruling. Masked ONLY when it equals the current file's sha (a manifest
#: that carries any other sha still differs from the golden). The golden was recorded with an older
#: contracts file; the one-value difference is the mask, nothing else in it changed.
CONTRACTS_PATH = REPO / "data" / "source-family-contracts.v1.json"


def _contracts_sha() -> str:
    return hashlib.sha256(CONTRACTS_PATH.read_bytes()).hexdigest()


#: What a layout fingerprint holds that is a TIMING of the optimizer, not a fact of what was written
#: (segment split, indexed counts, stale ratios): dropped, so the comparison cannot flake on a race.
_FP_STABLE = ("schema", "collection", "physical", "points_count", "named_vectors", "status")


def _stabilise(obj):
    if isinstance(obj, dict):
        if obj.get("schema") == "aw.collection-layout-fingerprint.v1":
            return {k: obj[k] for k in _FP_STABLE if k in obj}
        return {k: _stabilise(v) for k, v in obj.items()}      # key ORDER is kept
    if isinstance(obj, list):
        return [_stabilise(v) for v in obj]
    return obj


#: The gate contract and the payload declaration quote the private build tree's paths in their prose;
#: the receipt copies that prose. The golden does not carry those paths (the public cut ships this
#: directory and no path into the private tree), so both sides are normalised the same way.
_PRIVATE = re.compile(r"(?<![\w.-])(?:pipe" + "line|ta" + "sks)/")


def normalise(text: str, tmp: pathlib.Path) -> str:
    text = text.replace(str(tmp), "<TMP>")
    text = text.replace(str(REPO), "<REPO>")      # the checkout the drain's gate contract is read from
    text = text.replace(os.environ.get("AW_TEST_QDRANT_URL", "<no-url>"), "<QDRANT>")
    text = _PRIVATE.sub("<private-tree>/", text)
    text = text.replace(_contracts_sha(), "<CONTRACTS_SHA256>")
    text = _TS.sub("<TS>", text)
    text = _MANIFEST_SHA.sub(r"\1<MANIFEST_SHA>", text)
    text = _SECONDS.sub(r'\1"<N>"', text)
    try:                                   # one JSON document (receipt, manifest, id list)
        return json.dumps(_stabilise(json.loads(text)), indent=1, ensure_ascii=False) + "\n"
    except ValueError:
        pass
    lines = []                             # or JSON lines (the ledger)
    for line in text.splitlines():
        try:
            lines.append(json.dumps(_stabilise(json.loads(line)), ensure_ascii=False))
        except ValueError:
            return text                    # plain text (markers): as it is
    return "\n".join(lines) + "\n"


def snapshot_tree(root: pathlib.Path, tmp: pathlib.Path) -> dict[str, str]:
    """Every file under `root`, relpath -> normalised text (the drain writes only small text files
    here: receipts, ledgers, id lists, markers)."""
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            text = normalise(p.read_text(encoding="utf-8"), tmp)
            if p.name == "READY":                    # the sha of the manifest bytes (see above)
                text = "<MANIFEST_SHA>\n"
            out[str(p.relative_to(root))] = text
    return out


def dump_points(url: str, collection: str) -> list[dict]:
    """Every point of a (tiny) collection, id-sorted, vectors and payload included."""
    pts, offset = [], None
    while True:
        body = {"limit": 100, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        res = api(url, f"/collections/{collection}/points/scroll", body, method="POST")
        pts += res["points"]
        offset = res.get("next_page_offset")
        if offset is None:
            break
    return sorted(pts, key=lambda p: str(p["id"]))
