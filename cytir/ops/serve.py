"""`cyt serve`: start the API/MCP server on a NAMED collection, by configuration only
(refactor slice 8, step 4; specs/aw-index-serve-gate.md section 2).

    cyt serve --collection NAME --qdrant URL [--profile served] [--tei-url URL]
             [--host 127.0.0.1] [--port 8080] [--idf-table PATH] [--set K=V ...]
             [--receipt PATH] [--print-env]

WHAT THE ENV IS. `cytir.ops.prod_env.resolve_env`: the compose `aw-api` block PLUS every overlay
in `deploy/compose-overlays.manifest` (container paths mapped to this checkout); in a tree without
`deploy/` (the public cut) the shipped default `cytir/ops/serving_env.default.json`, generated from those
same files. Either way the receipt names the file each value came from. Then the
target: the collection, its Qdrant, the query embedder, the width the profile declares, and
the IDF table. A flag a caller wants changed goes through `--set K=V`, which the receipt
records. An `AW_*` variable that is only in the caller's shell is NOT passed to the server
(it would be invisible state); the receipt names every one that was dropped.

WHAT IT REFUSES (exit 3, nothing started):
  * the collection is missing or unreadable;
  * its shape differs from the declared `ServingProfile`: vector width/dtype, quantization,
    sparse config, payload indexes, HNSW graph, on-disk flags. Every differing field is
    named. Optimizer settings and the HNSW indexing-thread cap are runtime-tunable, not what
    a query reads, and are not compared;
  * `--idf-table` names another collection, or another point count, than the target;
  * there is no way to embed a query (no `--tei-url` and no in-process model installed).

WHAT IT WRITES. `serve-receipt.json`: collection, profile name + digest, the resolved env
with the file each value came from, the IDF table's identity, how queries are embedded. It
is printed too. Then `os.execvpe` uvicorn on `cytir.serve.app:api`: one process.

The unit files that run `cytir/ops/prod_env_run.py` are NOT repointed here (an operator change).
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import hashlib
import importlib.util
import json
import os
from cytir.core import envnames as _envnames
import pathlib
import re
import sys

from cytir.core import serving_profile as SP
from cytir.ops import prod_env
from cytir.core.serve_layout import SERVER_REL

REPO = prod_env.REPO
RECEIPT_SCHEMA = "aw.serve-receipt.v1"
DEFAULT_PROFILE = SP.PROD_E9.name     # task 1271 (R1): `served` is the pre-e9 shape, still selectable by name
EXIT_REFUSED = 3
#: shape fields a query does not read and an operator tunes at runtime
NOT_COMPARED = ("optimizer.", "hnsw.max_indexing_threads")
# a credential-shaped NAME (not `AW_BM25_PER_TOKEN`, a ranking flag): the receipt and
# `--print-env` show a digest instead of the value
_SECRET = re.compile(r"(SECRET|PASSW|CREDENTIAL|API_?KEY|ACCESS_?KEY|AUTH_?TOKEN|BEARER|_KEY$)", re.I)


class ServeRefused(RuntimeError):
    """The server must not start on this collection / configuration."""


# --------------------------------------------------------------------------- the env
def build_env(profile, *, collection: str, qdrant_url: str | None = None,
              tei_url: str | None = None, idf_table: str | None = None,
              sets=(), resolved: prod_env.ResolvedEnv | None = None) -> tuple[dict, dict]:
    """`(env, sources)` for a server on `collection`: compose base + overlays, then the
    target, then `--set`. `sources[key]` names where each value came from."""
    r = resolved or prod_env.resolve_env(overlays=True)
    env, src = dict(r.env), dict(r.sources)

    def put(k, v, why):
        env[k], src[k] = str(v), why

    put("AW_COLLECTION", collection, "aw serve --collection")
    if qdrant_url:
        put("AW_QDRANT_URL", qdrant_url, "aw serve --qdrant")
    # the compose value is a container host name; outside the container it resolves to
    # nothing, so a run without --tei-url embeds queries in-process, said in the receipt
    if tei_url:
        put("AW_TEI_URL", tei_url, "aw serve --tei-url")
    else:
        _envnames.pop_knob(env, "AW_TEI_URL")
        src["AW_TEI_URL"] = "aw serve: no --tei-url, queries embed in-process"  # rename-codemod: keep (persisted: source label in serve-receipt.json)
    put("AW_MRL_DIM", profile.dense_dim, f"profile {profile.name} dense_dim")
    webapp = REPO / "webapp" / "index.html"
    if webapp.is_file():
        put("AW_WEBAPP", webapp, "this checkout (the compose value is a container path)")
    else:
        _envnames.pop_knob(env, "AW_WEBAPP")
        src["AW_WEBAPP"] = "absent: no webapp/index.html in this checkout"
    if idf_table:
        put("AW_BM25_IDF_TABLE", idf_table, "aw serve --idf-table")
    for item in sets:
        k, eq, v = str(item).partition("=")
        if not eq or not k:
            raise ServeRefused(f"--set {item!r}: expected K=V")
        put(k, v, "aw serve --set")  # rename-codemod: keep (persisted: source label in serve-receipt.json)
    return env, src


def process_env(env: dict, ambient=None) -> tuple[dict, list]:
    """The environment the server process gets: the ambient non-AW variables (PATH, HOME,
    HF_HOME ...) and exactly the resolved `AW_*` ones. Returns it and the ambient `AW_*`
    names that were NOT passed on."""
    ambient = dict(os.environ if ambient is None else ambient)
    dropped = sorted(k for k in ambient if _envnames.is_knob(k) and k != _envnames.MARKER and k not in env and _envnames.twin(k) not in env)
    out = {k: v for k, v in ambient.items() if not _envnames.is_knob(k)}
    out.update(env)
    out["PYTHONPATH"] = os.pathsep.join(
        [str(REPO)] + ([out["PYTHONPATH"]] if out.get("PYTHONPATH") else []))
    return out, dropped


def _shown(k: str, v: str) -> str:
    if _SECRET.search(k):
        return "<redacted sha256:" + hashlib.sha256(v.encode()).hexdigest()[:12] + ">"
    return v


def env_digest(env: dict) -> str:
    blob = json.dumps({k: env[k] for k in sorted(env)}, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


# ------------------------------------------------------------------------ the checks
def shape_problems(client, collection: str, profile) -> list[str]:
    """Every field where the collection read back differs from `profile`, one line each
    (`vectors.dense.size: collection 128, profile 256`). Raises `ServeRefused` when the
    collection is missing or cannot be read: an error is never an empty list."""
    try:
        read = SP.read_shape(client, collection)
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if "NotFound" in name or "404" in str(exc) or "doesn't exist" in str(exc):
            raise ServeRefused(f"collection {collection!r} does not exist") from exc
        raise ServeRefused(f"collection {collection!r} could not be read: {name}: {exc}") from exc
    diffs = SP.shape_diff(read, profile.expected_shape())
    return [f"{k}: collection {a!r}, profile {profile.name!r} declares {b!r}"
            for k, (a, b) in sorted(diffs.items())
            if not any(k == n or k.startswith(n) for n in NOT_COMPARED)]


def idf_identity(path: pathlib.Path | str, *, explicit: bool, collection: str,
                 points_count: int | None) -> dict:
    """The IDF table's identity for the receipt. `matches_target` is None when the table is
    not a measured table, else whether its collection and point count are the target's."""
    p = pathlib.Path(path)
    raw = p.read_bytes()
    out = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "explicit": explicit}
    try:
        doc = json.loads(raw)
    except ValueError:
        out["matches_target"] = None
        return out
    out.update({k: doc.get(k) for k in ("collection", "points_count", "measured_at", "fold_accents")})
    out["matches_target"] = (doc.get("collection") == collection
                             and doc.get("points_count") == points_count)
    return out


def default_idf_table() -> pathlib.Path:
    return REPO / SERVER_REL / "bm25_idf.json"


def _tei_info(url: str) -> dict | None:
    """The embedder's own identity (`GET /info` of a text-embeddings-inference server: model id and
    revision; a stand-in announces itself with `stand_in`), recorded in the receipt so a gate can
    say what its verdict proves. `None` when the server does not answer: the receipt then says
    the identity is unknown, never a guess."""
    try:
        with IH.urlopen(url.rstrip("/") + "/info", timeout=5) as r:
            doc = json.load(r)
    except Exception:  # noqa: BLE001  the identity is optional evidence, never a reason to refuse
        return None
    if not isinstance(doc, dict):
        return None
    return {k: doc[k] for k in ("model_id", "model_sha", "stand_in") if k in doc}


def query_embedding(env: dict) -> dict:
    """How this server embeds queries, and a refusal when it cannot."""
    if env.get("AW_TEI_URL"):
        return {"mode": "tei", "tei_url": env["AW_TEI_URL"], "info": _tei_info(env["AW_TEI_URL"])}
    if importlib.util.find_spec("sentence_transformers") is None:
        raise ServeRefused(
            "no way to embed a query: pass --tei-url (a text-embeddings-inference server "
            "of the pinned model) or install the in-process model "
            "(`pip install sentence-transformers`)")
    return {"mode": "in-process", "tei_url": None, "info": _in_process_info()}


def _in_process_info() -> dict:
    """The identity of the model the server loads to embed queries in-process, read with no network
    call: the pinned id and revision (`cytir/serve/models.py` loads exactly these), the device
    and dtype sentence-transformers picks here, and the library versions. Same keys as TEI `/info`
    (`model_id`, `model_sha`) so `cyt gate` judges both the same way."""
    from importlib import metadata
    from cytir.core import model_pin

    def ver(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except ImportError:
        cuda = False
    return {"model_id": model_pin.EMBED_MODEL_ID, "model_sha": model_pin.embed_model_revision(),
            "device": "cuda" if cuda else "cpu", "dtype": "float32",
            "sentence_transformers": ver("sentence-transformers"), "torch": ver("torch"),
            "source": "pin"}


# -------------------------------------------------------------------------- the verb
def build_parser(prog: str = "cyt serve") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--collection", required=True)
    ap.add_argument("--profile", default=DEFAULT_PROFILE,
                    help=f"a named serving profile (default {DEFAULT_PROFILE!r}, what prod runs; `served` is the pre-e9 shape); the "
                         "collection must match it")
    ap.add_argument("--qdrant", default=None, help="Qdrant URL (required unless --print-env)")
    ap.add_argument("--tei-url", default=None,
                    help="embed queries through a TEI server of the pinned model; without it "
                         "the server embeds in-process")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--idf-table", default=None,
                    help="a measured BM25 IDF table for THIS collection (refused when its "
                         "collection or point count differ); default: the shipped table")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="K=V",
                    help="override one env value; recorded in the receipt, never silent")
    ap.add_argument("--receipt", default="serve-receipt.json")
    ap.add_argument("--print-env", action="store_true",
                    help="print the resolved env and where each value came from, then exit "
                         "(no Qdrant needed, nothing started)")
    return ap


def run(argv=None, *, exec_fn=os.execvpe, client=None, ambient=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = SP.get(args.profile)
    except KeyError as exc:
        print(f"REFUSED: {exc.args[0]}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        env, src = build_env(profile, collection=args.collection, qdrant_url=args.qdrant,
                             tei_url=args.tei_url, idf_table=args.idf_table, sets=args.sets)
        if args.print_env:
            print(json.dumps({"profile": profile.name, "env": {k: _shown(k, env[k]) for k in sorted(env)},
                              "sources": {k: src[k] for k in sorted(src)}}, indent=1))
            return 0
        if not args.qdrant:
            raise ServeRefused("--qdrant URL is required (the compose value is a container host)")
        embed = query_embedding(env)
        if client is None:
            from cytir.core.qdrant import qdrant_client
            client = qdrant_client(args.qdrant, timeout=30)
        problems = shape_problems(client, args.collection, profile)
        if problems:
            raise ServeRefused(
                f"collection {args.collection!r} is not the declared profile {profile.name!r} "
                f"(digest {profile.digest()[:12]}); {len(problems)} field(s) differ:\n  - "
                + "\n  - ".join(problems))
        points = client.get_collection(args.collection).points_count
        table = pathlib.Path(args.idf_table) if args.idf_table else default_idf_table()
        idf = idf_identity(table, explicit=bool(args.idf_table), collection=args.collection,
                           points_count=points)
        if args.idf_table and not idf["matches_target"]:
            raise ServeRefused(
                f"--idf-table {args.idf_table} was measured on collection "
                f"{idf.get('collection')!r} with {idf.get('points_count')!r} points; the target "
                f"is {args.collection!r} with {points!r}")
        penv, dropped = process_env(env, ambient)
    except ServeRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    argv_exec = [sys.executable, "-m", "uvicorn", "cytir.serve.app:api",
                 "--host", args.host, "--port", str(args.port)]
    receipt = {
        "schema": RECEIPT_SCHEMA, "collection": args.collection, "qdrant_url": args.qdrant,
        "points_count": points,
        "profile": {"name": profile.name, "digest": profile.digest(),
                    "profile_schema_version": SP.PROFILE_SCHEMA_VERSION},
        "shape_check": {"result": "equal", "not_compared": list(NOT_COMPARED)},
        "env": {k: {"value": _shown(k, env[k]), "source": src[k]} for k in sorted(env)},
        "env_not_set": {k: src[k] for k in sorted(src) if k not in env},
        "env_sha256": env_digest(env),
        "set": sorted(k for k in (s.partition("=")[0] for s in args.sets)),
        "ambient_aw_variables_not_passed": dropped,
        "query_embedding": embed, "idf_table": idf,
        "listen": {"host": args.host, "port": args.port}, "argv": argv_exec,
    }
    text = json.dumps(receipt, indent=1, sort_keys=True)
    pathlib.Path(args.receipt).write_text(text + "\n")
    print(text, flush=True)
    exec_fn(argv_exec[0], argv_exec, penv)
    return 0     # only reached by a test's exec_fn


def main(argv=None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
