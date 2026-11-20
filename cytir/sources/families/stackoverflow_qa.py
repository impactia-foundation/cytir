"""`stackoverflow_qa` -- Stack Overflow questions with their accepted answer (task 696).

The declared family of task 629: bulk-dump route, content under CC BY-SA (2.5, 3.0 or 4.0, read from
each post's own `ContentLicense`), chunked under the declared `chunk.generic` policy. The extractor is
`cytir.extract.stackoverflow_qa` (the selection of the production frontier) with the renderer the live
tranches used (`cytir.extract.stackoverflow_pilot.materialize`): one question and its accepted answer
become one Markdown document under the question's address, and the declared chunker cuts it into one
or more passages (the live tranches measured 1.28 per document).

Options (`--opt KEY=VALUE`): `root` (required; a directory holding the official dump's `Posts.xml`
and, when an owner has no display name on the post, `Users.xml`; the files of the dump, extracted).
Optional: `ids` (a text file of question ids, one per line, `.gz` allowed) to take only those
questions, as a tranche of the live path does; without it every question the frontier rule admits is
taken. The unit is the `Posts.xml`; it ends `admitted` (one or more passages) or
`excluded(no_accepted_question_scores_10_or_more)`.

L0: the dump is a public, content-addressed file that the operator already holds (the posts archive is
23 GB; extracted, about 90 GB), so the framework seals a POINTER to it: the files' relative paths,
sizes and sha256 (read in 1 MiB blocks, never whole). The extractor streams the rows out of the file.

Persons (ADR-039; founder ruling 7): the owners of the posts are natural persons or deleted accounts.
Their display names ARE in the embedded text (the dense and sparse surfaces), exactly as the served
rows of this family have them: the surface is a contract. They are NOT in the snippet's first chunk and
not in any compact field: the names also go in the master-only `_master` (`attribution` is the answer's
author), and the licence name stays compact. Neither ships in the compact dump.

A dump-route family contacts no host: no per-host consent is resolved (ADR-076; the licence is the
consent basis, ADR-115 r2). Nothing is fetched from stackoverflow.com.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import os
import pathlib

from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

POSTS = "Posts.xml"
USERS = "Users.xml"
NO_QUESTION_ELIGIBLE = "no_accepted_question_scores_10_or_more"
POINTER_SCHEMA = "aw.stackoverflow-dump-pointer.v1"
DOMAIN = "stackoverflow.com"
BLOCK = 1 << 20


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


def _open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(
        path, encoding="utf-8")


def _sha256_file(path) -> tuple[int, str]:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(BLOCK), b""):
            h.update(block)
            n += len(block)
    return n, h.hexdigest()


def _licence_name(dump_value: str) -> tuple[str, str]:
    """`CC BY-SA 2.5` (the dump's words) -> (`CC-BY-SA-2.5`, `2.5`): the SPDX-style name of
    ADR-039 and the master-only version."""
    version = dump_value.rsplit(" ", 1)[-1]
    return dump_value.replace(" ", "-"), version


def _name(attribution: dict) -> str:
    return str(attribution.get("owner_display_name") or attribution.get("owner_user_id") or "")


@register_family
class StackOverflowQaFamily:
    family_id = "stackoverflow_qa"
    route = "source_owned_bulk_dump"
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED
    require_records = True     # a run that admits 0 passages refuses (smoke-test-yield)

    def preflight(self, ctx: RunContext) -> None:
        sel = _policy_selection(ctx)
        ctx.notes["chunker"] = {"mode": sel.mode, "epoch": sel.epoch, "legend": sel.legend,
                                "policy_sha256": sel.policy_sha256,
                                "tokenizer_revision": sel.tokenizer_revision}
        self._selection = sel

    def discover(self, ctx: RunContext):
        root = ctx.options.get("root")
        if not root:
            raise ValueError(f"{self.family_id} needs --opt root=<directory holding the dump's "
                             f"{POSTS}> (or --from DIR)")
        if not os.path.isdir(root):
            raise ValueError(f"{self.family_id}: root {root!r} is not a directory")
        posts = pathlib.Path(root) / POSTS
        if not posts.is_file():
            raise RuntimeError(f"{self.family_id}: no {POSTS} under {root!r}; "
                               "an empty walk is a wrong root, not an empty source")
        ids = ctx.options.get("ids")
        if ids:
            from cytir.extract import stackoverflow_qa as E

            with _open_text(ids) as fh:
                self._ids = E.read_ids(fh)
            ctx.notes["ids"] = len(self._ids)
        else:
            self._ids = None
        ctx.notes.update({"root": str(root)})
        return [Unit(key=POSTS, url=posts.resolve().as_uri(),
                     meta={"path": str(posts), "users": str(posts.with_name(USERS))})]

    def acquire(self, unit: Unit, ctx: RunContext):
        size, digest = _sha256_file(unit.meta["path"])
        pointer = {"schema": POINTER_SCHEMA, "relative_path": unit.key, "bytes": size,
                   "sha256": digest, "licence": "CC BY-SA 2.5/3.0/4.0, per post",
                   "dump": "https://archive.org/details/stackexchange"}
        users = pathlib.Path(unit.meta["users"])
        if users.is_file():
            usize, udigest = _sha256_file(users)
            pointer["users"] = {"relative_path": USERS, "bytes": usize, "sha256": udigest}
        body = (json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n").encode()
        return [Captured(unit, body, {"member": unit.key, "sha256": digest, "bytes": size})]

    def extract(self, cap: Captured, ctx: RunContext):
        from cytir.extract import stackoverflow_qa as E

        pairs, counts = E.select_pairs(cap.unit.meta["path"], ids=self._ids)
        users, users_path = {}, pathlib.Path(cap.unit.meta["users"])
        wanted = E.owner_ids(pairs)
        if wanted:
            if not users_path.is_file():
                raise ValueError(
                    f"{self.family_id}: {len(wanted)} selected post(s) name an owner by id only "
                    f"and {users_path} is not there; put the dump's {USERS} beside {POSTS}")
            with open(users_path, "rb") as fh:
                users = E.read_users(fh, wanted)
        admitted, why = E.admit(pairs, users)
        docs, refused = [], {}
        for pair in admitted:
            row, quarantine = E.render(pair, users)
            if quarantine is not None:
                refused[quarantine["reason"]] = refused.get(quarantine["reason"], 0) + 1
                continue
            docs.append(Doc(
                url=row["url"], text=row["markdown"], title=row["title"].strip(), lang=row["lang"],
                extra={"row": row, "source_sha256": cap.meta["sha256"]}))
        ctx.notes.setdefault("per_unit", {})[cap.unit.key] = {
            **counts, **why, "rendered": len(docs), "refused_by_the_renderer": refused}
        if not docs:
            raise UnitExcluded(NO_QUESTION_ELIGIBLE)
        return docs

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        row = doc.extra["row"]
        q, a = row["question"], row["accepted_answer"]
        licence, version = _licence_name(a["license"])
        out = []
        stamps = self._selection.stamps()
        digest = C.sha1(doc.text)
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(doc.text,
                                                                 selection=self._selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                # the live rule: the first 25 words of the embedded text (`projection.project_record`)
                snippet=C.extractive_snippet(unit.dense_text or unit.text, 25),
                lang=doc.lang, domain=DOMAIN, tier="B", source_type="community",
                trust_tier=0.55, last_seen=ctx.today, content_digest=digest,
                license=licence, license_version=version,
                attribution=_name(a["attribution"]),
                # the live anchor: the point id is minted over it (identity_check `pack-declared`)
                anchor=f"document:{row['document_id']}:chunk-{ordinal}",
                master={"question_id": q["post_id"], "answer_id": a["post_id"],
                        "question_score": q["score"], "answer_score": a["score"],
                        "question_license": q["license"], "answer_license": a["license"],
                        "question_author": _name(q["attribution"]),
                        "author_url": a["attribution"]["owner_profile_url"],
                        "document_id": row["document_id"],
                        "question_raw_xml_sha256": q["raw_xml_sha256"],
                        "answer_raw_xml_sha256": a["raw_xml_sha256"],
                        "source_sha256": doc.extra["source_sha256"]})
            rec = C.passage_source_record(p)
            # the two fields the live stage rows carry; `cytir.index.compose` mints the identity check
            # from them (its default anchor scheme), so no per-family entry is needed there
            rec["document_id"], rec["chunk_ordinal"] = row["document_id"], ordinal
            out.append(rec)
        return out
