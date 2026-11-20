"""`stackexchange_nonso_qa_p0` -- Stack Exchange sites other than Stack Overflow (task 696).

The declared family of task 631: bulk-dump route, content under CC BY-SA (2.5, 3.0 or 4.0 by the
answer's date, ADR-039), chunked under the declared `chunk.generic` policy. The extractor is
`cytir.extract.stackexchange_qa` (the task-226 logic, unchanged): a question and its accepted or top
answers become one passage per answer, under the answer's own permalink (`https://<site>/a/<id>`).

Options (`--opt KEY=VALUE`): `root` (required; a directory with one sub-directory per site, named
by the site's host, each holding that site's `Posts.xml` and, optionally, its `Users.xml` -- the
files of the official data dump, extracted). A unit is one site's `Posts.xml`; it ends `admitted`
(one or more passages) or `excluded(no_question_clears_the_score_bar)`.

Attribution (ADR-039): the answer author's display name goes in the master-only `attribution`, the
profile URL, the post ids, the score and the licence version in the master-only part of the row.
Neither ships in the compact dump; the licence name does.

A dump-route family contacts no host: no per-host consent is resolved (ADR-076; the licence is the
consent basis). No Stack Exchange site is ever fetched.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
import re

from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

POSTS = "Posts.xml"
USERS = "Users.xml"
NO_QUESTION_CLEARS_THE_BAR = "no_question_clears_the_score_bar"
_HOST = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")


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


def _after_title(text: str, title: str) -> str:
    rest = text[len(title):].strip() if title and text.startswith(title) else text
    return rest or text


@register_family
class StackExchangeNonSoQaFamily:
    family_id = "stackexchange_nonso_qa_p0"
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
            raise ValueError(f"{self.family_id} needs --opt root=<directory with one "
                             f"sub-directory per site, each holding {POSTS}> (or --from DIR)")
        if not os.path.isdir(root):
            raise ValueError(f"{self.family_id}: root {root!r} is not a directory")
        units = []
        for dirpath, _dirs, files in sorted(os.walk(root)):
            if POSTS not in files:
                continue
            host = os.path.basename(dirpath)
            if os.path.abspath(dirpath) == os.path.abspath(root) or not _HOST.match(host):
                raise ValueError(f"{self.family_id}: {os.path.join(dirpath, POSTS)} is not under a "
                                 "directory named by a site host (e.g. <root>/security."
                                 f"stackexchange.com/{POSTS})")
            path = os.path.join(dirpath, POSTS)
            units.append(Unit(key=os.path.relpath(path, root),
                              url=pathlib.Path(path).resolve().as_uri(),
                              meta={"path": path, "host": host}))
        if not units:
            raise RuntimeError(f"{self.family_id}: no {POSTS} under {root!r}; "
                               "an empty walk is a wrong root, not an empty source")
        ctx.notes.update({"root": str(root), "sites": len(units)})
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        body = pathlib.Path(unit.meta["path"]).read_bytes()
        return [Captured(unit, body, {"member": unit.key, "host": unit.meta["host"],
                                      "sha256": hashlib.sha256(body).hexdigest()})]

    def extract(self, cap: Captured, ctx: RunContext):
        import io

        from cytir.extract import stackexchange_qa as E

        host = cap.unit.meta["host"]
        users_path = pathlib.Path(cap.unit.meta["path"]).with_name(USERS)
        users = {}
        if users_path.is_file():
            with users_path.open("rb") as fh:
                users = E.users_map(E.iter_rows(fh))
        recs = list(E.passages(E.iter_rows(io.BytesIO(cap.body)), host=host, users=users,
                               last_seen=ctx.today))
        if not recs:
            raise UnitExcluded(NO_QUESTION_CLEARS_THE_BAR)
        return [Doc(url=r["url"], text=r["text"], title=r["title"], lang=r["lang"],
                    extra={"rec": r, "source_sha256": cap.meta["sha256"]}) for r in recs]

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        r = doc.extra["rec"]
        out = []
        stamps = self._selection.stamps()
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(doc.text,
                                                                 selection=self._selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=r["snippet"] if ordinal == 0 else
                C.extractive_snippet(_after_title(unit.text, doc.title), 25),
                lang=doc.lang, domain=r["domain"], tier=r["tier"], source_type=r["source_type"],
                trust_tier=r["trust_tier"], last_seen=ctx.today, content_digest=r["content_digest"],
                license=r["license"], license_version=r["master_license_version"],
                attribution=r["master_author"], anchor=f"chunk-{ordinal}",
                master={"author_url": r["master_author_url"], "post_id": r["master_post_id"],
                        "question_id": r["master_question_id"],
                        "answer_score": r["master_answer_score"],
                        "is_accepted": r["master_is_accepted"],
                        "source_sha256": doc.extra["source_sha256"]})
            out.append(C.passage_source_record(p))
        return out
