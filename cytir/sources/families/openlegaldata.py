"""`de_openlegaldata_v1` -- German court decisions from the OpenLegalData dump (task 1373).

The declared family of task 633: dump route (the HuggingFace dataset `openlegaldata/court-decisions-germany`,
`dump-20260520`, 54 parquet shards), UrhG par. 5 (a court decision is an official work), chunked under the
declared `chunk.generic` policy. The extractor is `cytir.extract.openlegaldata` (identity line, title, URL and
document id: the same functions the shard chain has always called, not rewritten). The adapter
adds only the door: one unit per parquet shard, one document per decision, the declared chunker.

Options (`--opt KEY=VALUE`, or `--from DIR`): `root` (required; a directory that holds the shards as
`original/parquet/dump-20260520/train-NNNNN-of-00054.parquet`, or any `.parquet` files below it).
A shard ends `admitted`, or `excluded(no_decision_with_text)`. A decision with an empty
`markdown_content` (about 0.4 % of shard 0, a gap in the source's own extraction) is not dropped
silently: the old chain writes it to an inventory, and this door counts it in the pack manifest
(`notes.per_shard[<shard>].empty_body`).

L0: the shard is a public, content-addressed file the operator already holds; the framework seals a
POINTER to it (relative path, size, sha256), and every row's `source_part_sha256` carries that hash.

A dump-route family contacts no host: no per-host consent is resolved (ADR-076). Nothing is fetched
from de.openlegaldata.io or from huggingface.co.

L1 (`--opt shards=2-11,14`, optional `--opt corpus_root=DIR`, read only): `l1(ctx)` materializes the Markdown
of each named shard through the extractor functions and writes the per-shard L1 release under
`<out>/l1/release=openlegaldata-v1-shard<N>/` (the release name the earlier chain used, so a shard already
filed is recognised). A shard that `is_l1_filed` finds under `corpus_root` is skipped. Filing into the corpus
is the operator step, run per shard against that out dir:

    cyt source run de_openlegaldata_v1 --from DUMP_DIR --opt shards=N --opt corpus_root=CORPUS --out OUT
    python -m cytir.corpus.register_l1_release --source OUT/l1/release=openlegaldata-v1-shardN \
        --release openlegaldata-v1-shardN --corpus-root CORPUS --recursive
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import re
import shutil

from cytir.acquire.gzip_pinned import deterministic_gzip
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

NO_DECISION_WITH_TEXT = "no_decision_with_text"
POINTER_SCHEMA = "aw.openlegaldata-shard-pointer.v1"
DOMAIN = "de.openlegaldata.io"
BLOCK = 1 << 20
STAGES = "stages/de-openlegaldata-v1"
SHARDS_TOTAL = 54                    # task 489: dump-20260520
#: the chunker writes a heading as its breadcrumb, not as `##` marks: the marks are the declared
#: transformation, so the staged-text guard compares the text without them
#: The strip is the declared transformation also used by `cytir/sources/families/pdf_replay.py` (`guard_text`); the structured
#: chunker in `cytir/chunk/chunker.py` carries headings as a breadcrumb prefix, so the words still appear in a row.
_HEADING_MARK = re.compile(r"^#{1,6} ", re.M)


def parse_shards(spec: str) -> list[int]:
    """'2-11,14' -> [2,...,11,14]. Refuses anything outside 0..53."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    bad = [n for n in out if not 0 <= n < SHARDS_TOTAL]
    if bad:
        raise ValueError(f"shards {bad} are outside 0..{SHARDS_TOTAL - 1}")
    return sorted(set(out))


def shard_dirs(artifact_root, shard: int) -> tuple[pathlib.Path, pathlib.Path]:
    """(materialize dir, chunk-stage dir) of one acquisition shard under the corpus root."""
    base = pathlib.Path(artifact_root) / STAGES
    return base / f"materialize-shard{shard}", base / f"chunks-shard{shard}"


def is_chunked(chunks_dir) -> bool:
    """A shard is chunked when its stage published `PACK.json` (the chunker publishes by rename)."""
    return (pathlib.Path(chunks_dir) / "PACK.json").is_file()


def l1_release_name(shard: int) -> str:
    return f"openlegaldata-v1-shard{shard}"


def is_l1_filed(artifact_root, shard: int) -> bool:
    """True when the shard's Markdown is filed in the registered L1 tree (ADR-105)."""
    return (pathlib.Path(artifact_root) / "l1" / f"release={l1_release_name(shard)}"
            / "_register_manifest.json").is_file()


def _policy_selection(ctx: RunContext):
    """The family's declared chunk policy as an explicit structured selection (the file the
    registry names for the contract's sha256, and the contract's tokenizer revision): the run
    needs no `AW_CHUNK*` variable, and a policy that is not the declared one never runs."""
    from cytir.chunk import registry as chunk_registry
    from cytir.core import common as C
    from cytir.chunk import dispatch as _dispatch

    cp = ctx.contract["chunk_policy"]
    entry = chunk_registry.entry(cp["sha256"])
    repo = pathlib.Path(__file__).resolve().parents[3]
    return _dispatch.explicit_structured_chunker_selection(
        str(repo / entry["file"]), cp["sha256"], cp["tokenizer_revision"])


def _sha256_file(path) -> tuple[int, str]:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(BLOCK), b""):
            h.update(block)
            n += len(block)
    return n, h.hexdigest()


@register_family
class OpenLegalDataFamily:
    family_id = "de_openlegaldata_v1"
    route = "source_owned_bulk_dump"
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED
    require_records = True     # a run that admits 0 passages refuses (smoke-test-yield)

    def preflight(self, ctx: RunContext) -> None:
        self._selection = _policy_selection(ctx)

    def l1(self, ctx: RunContext) -> dict:
        """Persist the per-shard L1 release (ADR-105 am.1) under `ctx.out_dir/l1/`, one per shard in
        `--opt shards=`. `--opt corpus_root=` is read only: a shard it already files is skipped."""

        from cytir.acquire.artifact_io import canonical_json
        from cytir.corpus.register_l1_release import register_l1_release
        from cytir.extract import openlegaldata as E

        spec = ctx.options.get("shards")
        if not spec:
            return {"shards": [], "note": "no shards option; nothing persisted"}
        root = ctx.options.get("root")
        if not root or not pathlib.Path(root).is_dir():
            raise ValueError(f"{self.family_id}: l1 needs --opt root=<directory holding the shards>")
        corpus = ctx.options.get("corpus_root")
        done = []
        for n in parse_shards(spec):
            name = l1_release_name(n)
            if corpus and is_l1_filed(corpus, n):
                done.append({"shard": n, "release": name, "status": "already_filed"})
                continue
            if (ctx.out_dir / "l1" / f"release={name}" / "_register_manifest.json").is_file():
                done.append({"shard": n, "release": name, "status": "persisted"})
                continue
            found = sorted(pathlib.Path(root).rglob(f"train-{n:05d}-of-{SHARDS_TOTAL:05d}.parquet"))
            if len(found) != 1:
                raise RuntimeError(f"{self.family_id}: shard {n}: {len(found)} parquet files under root")
            _, digest = _sha256_file(found[0])
            parent = {"relative_path": found[0].relative_to(root).as_posix(), "sha256": digest}
            raws = E.read_shard_rows(found[0])
            rows = [E.build_markdown_row(raw, parent_archive=parent)
                    for raw in raws if E.build_markdown(raw) is not None]
            src = ctx.out_dir / "_l1_src" / f"shard{n}"
            (src / "data").mkdir(parents=True, exist_ok=True)
            body = b"".join((canonical_json(r) + "\n").encode() for r in rows)
            (src / "data" / "shard-000000.jsonl.gz").write_bytes(deterministic_gzip(body))
            register_l1_release(str(src), name, corpus_root=str(ctx.out_dir), recursive=True)
            shutil.rmtree(src)
            done.append({"shard": n, "release": name, "status": "persisted", "documents": len(rows),
                         "skipped_empty": len(raws) - len(rows), "source_rows": len(raws)})
        shutil.rmtree(ctx.out_dir / "_l1_src", ignore_errors=True)
        return {"shards": done}

    def discover(self, ctx: RunContext):
        root = ctx.options.get("root")
        if not root:
            raise ValueError(f"{self.family_id} needs --opt root=<directory holding the dump's "
                             "parquet shards> (or --from DIR)")
        if not pathlib.Path(root).is_dir():
            raise ValueError(f"{self.family_id}: root {root!r} is not a directory")
        found = sorted(p.relative_to(root).as_posix() for p in pathlib.Path(root).rglob("*.parquet"))
        if not found:
            raise RuntimeError(f"{self.family_id}: no .parquet shard under {root!r}; "
                               "an empty walk is a wrong root, not an empty source")
        ctx.notes.update({"shards": len(found)})
        return [Unit(key=rel, url=(pathlib.Path(root) / rel).resolve().as_uri(),
                     meta={"path": str(pathlib.Path(root) / rel), "relative_path": rel})
                for rel in found]

    def acquire(self, unit: Unit, ctx: RunContext):
        size, digest = _sha256_file(unit.meta["path"])
        pointer = {"schema": POINTER_SCHEMA, "relative_path": unit.meta["relative_path"],
                   "bytes": size, "sha256": digest,
                   "dataset": "https://huggingface.co/datasets/openlegaldata/court-decisions-germany"}
        body = (json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n").encode()
        return [Captured(unit, body, {"member": unit.key, "sha256": digest, "bytes": size})]

    def extract(self, cap: Captured, ctx: RunContext):
        from cytir.extract import openlegaldata as E

        rel, shard_sha = cap.unit.meta["relative_path"], cap.meta["sha256"]
        parent = {"relative_path": rel, "sha256": shard_sha}
        docs, empty = [], 0
        for raw in E.read_shard_rows(pathlib.Path(cap.unit.meta["path"])):
            if E.build_markdown(raw) is None:
                empty += 1
                continue
            row = E.build_markdown_row(raw, parent_archive=parent)
            docs.append(Doc(url=row["url"], text=row["markdown"], title=row["title"],
                            lang=row["lang"],
                            source_text=_HEADING_MARK.sub("", row["markdown"]),
                            extra={"document_id": row["document_id"], "shard_sha256": shard_sha,
                                   "court": (raw.get("court") or {}).get("name") or "",
                                   "metadata": row["source_metadata"], "license": row["license"],
                                   "trust_tier": row["trust_tier"]}))
        ctx.notes.setdefault("per_shard", {})[rel] = {"rendered": len(docs), "empty_body": empty}
        if not docs:
            raise UnitExcluded(NO_DECISION_WITH_TEXT)
        return docs

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        x = doc.extra
        out = []
        stamps = self._selection.stamps()
        digest = C.sha1(doc.text)
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(doc.text,
                                                                 selection=self._selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(unit.dense_text or unit.text, 25),
                lang=doc.lang, domain=DOMAIN, tier="A", source_type="official",
                trust_tier=x["trust_tier"], last_seen=ctx.today, content_digest=digest,
                license=x["license"], attribution=x["court"],
                anchor=f"document:{x['document_id']}:chunk-{ordinal}",
                master={"document_id": x["document_id"], "ecli": x["metadata"]["ecli"],
                        "file_number": x["metadata"]["file_number"],
                        "decision_date": x["metadata"]["date"],
                        "decision_type": x["metadata"]["type"],
                        "source_part_sha256": x["shard_sha256"]})
            rec = C.passage_source_record(p)
            rec["document_id"], rec["chunk_ordinal"] = x["document_id"], ordinal
            out.append(rec)
        return out
