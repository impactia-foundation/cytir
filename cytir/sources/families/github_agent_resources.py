"""`github_agent_resources` -- repository documentation from GitHub (task 1115, S7).

THE FAMILY IS FULL ON THE PROTOCOL (task 1169). `discover`, `acquire` and `extract` are proven identical to the
legacy acquirer; `chunk` is the refresh lane's `docs_markdown` row path, proven by recording (below). Not on a
live path: the refresh lane and the legacy acquirer driver still make the served rows.

What the legacy driver is. `acquire_github_repo_docs.py` produces NO rows. Per repository it pins a
branch to a commit, preserves the exact API responses and blobs, writes Markdown and an outlink
inventory, and seals ONE terminal receipt (`complete` or `quarantined`, with its reason). The rows
come later, from the refresh lane (`build_rows_for_markdown` over the receipt's Markdown).

Split. Enumerate/fetch (`GitHubClient`, `acquire_repository`, the run loop) stays in the CLI driver:
its output is a preserved ROOT. The family reads that root (option `root`, as in tasks 1101/1108).
Everything pure moved to `cytir.extract.github_agent_resources` and the driver imports it back.

Options (`--opt KEY=VALUE`): `root` (required): a preserved root with `receipts/`; `queue`
(default `<root>/queue.jsonl`): the driver's queue (`owner_repo`, `branch`, `license`); `trust_tier`
(required): the pack's tier, copied onto each document as the refresh lane does.

Dispositions, said. A unit is one repository. Its receipt is the capture: the framework seals the
receipt file's own bytes (identical to the driver's). `complete` -> its documents are the Docs,
equal to `candidates_from_receipt` (the refresh lane's reader). `quarantined` -> the unit is
QUARANTINED with `receipt_quarantined:<reason>`, 1:1 with the driver's terminal (it is NOT
`excluded`: the driver's reasons mix origin facts and our own failures). A missing or invalid receipt
raises: the unit is quarantined, never an empty. Order is queue order. Consent: the GitHub API and
raw hosts are resolved live per unit; no consent field is added or defaulted, no `unknown`.

Chunk (task 1169). `chunk` = `build_rows_for_markdown` + `stamp_docs_row` (both in `cytir.chunk.markdown_rows`) with
the `docs_markdown` pack meta; rows equal the lane's, replayed against a recording, made on the build host, of the unmodified
lane path over the four sample repos (`data/samples/github-agent-resources/rows-recording.json`). The lane's
twin-digest skip is a Qdrant read and stays in the lane. `Doc.source_text` is left unset: the staged-text guard
counts these docs `unchecked`, because gate drops and policy cuts are not a transformation it can price.
Content kinds: the driver handles Markdown/MDX/RST documents only (no
code files; release notes are the separate `github_releases` adapter, not this family).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from cytir.chunk.markdown_rows import build_rows_for_markdown, stamp_docs_row
from cytir.acquire.preserved_stage_io import PreservedArtifactError
from cytir.extract import github_agent_resources as G
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit
from cytir.sources.registry import register_family

HOSTS = ("api.github.com", "raw.githubusercontent.com")


#: The lane's `PACK_META["us-github-repo-docs-tracked"]` minus its labels and gate (a test in the lane's own suite
#: keeps the two equal): what `build_rows_for_markdown` and `stamp_docs_row` read.
DOCS_MARKDOWN_PACK_META = {"tier": "A", "source_type": "docs", "trust_tier": 0.75, "route": "docs_markdown"}


@register_family
class GithubAgentResourcesFamily:
    family_id = "github_agent_resources"
    route = "documented_source_api"
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED

    def __init__(self) -> None:
        self._root = ""
        self._records: dict[str, dict] = {}

    def discover(self, ctx: RunContext):
        root = ctx.options.get("root")
        if not root:
            raise ValueError(f"{self.family_id} needs --opt root=<preserved root with receipts/>")
        if not ctx.options.get("trust_tier"):
            raise ValueError(f"{self.family_id} needs --opt trust_tier=<the pack's tier>")
        self._root = Path(root)
        queue = ctx.options.get("queue") or os.path.join(root, "queue.jsonl")
        rows = []
        with open(queue, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    row["owner_repo"] = G.normalize_owner_repo(str(row["owner_repo"]))
                    rows.append(row)
        if not rows:
            raise RuntimeError(f"{self.family_id}: {queue!r} lists no repository")
        self._records = {r["owner_repo"]: r for r in rows}
        ctx.notes.update({"root": root, "queue": queue, "repositories": len(rows)})
        return [Unit(key=r["owner_repo"], url=f"https://api.github.com/repos/{r['owner_repo']}",
                     meta={"license": str(r.get("license") or "")}, consent_hosts=HOSTS)
                for r in rows]

    def acquire(self, unit: Unit, ctx: RunContext):
        path = os.path.join(str(self._root), "receipts", G.receipt_name(unit.key))
        if G.load_terminal(self._root, unit.key) is None:
            raise RuntimeError(f"{unit.key}: no valid terminal receipt at {path!r}")
        body = open(path, "rb").read()
        return [Captured(unit, body, {"receipt": os.path.basename(path)})]

    def extract(self, cap: Captured, ctx: RunContext):
        receipt = json.loads(cap.body.decode("utf-8"))
        if receipt["status"] != "complete":
            raise RuntimeError("receipt_quarantined:" + receipt["quarantine"]["reason"])
        repo = {"owner_repo": cap.unit.key, "license": cap.unit.meta["license"],
                "trust_tier": ctx.options["trust_tier"]}
        try:
            cands = G.candidates_from_receipt(self._root, receipt, repo)
        except PreservedArtifactError as exc:
            raise RuntimeError(f"{cap.unit.key}: {exc}") from exc
        return [Doc(url=c["url"], text=c["markdown"], title=c.get("title") or "", lang="en",
                    extra={"candidate": c}) for c in cands]

    def chunk(self, doc: Doc, ctx: RunContext):
        """The refresh lane's `docs_markdown` branch, line for line (task 1169): the pack's meta with the
        candidate's own `trust_tier` over it, `build_rows_for_markdown`, then `stamp_docs_row` with the
        file's commit date (none today), the candidate's licence (else the pack's: none) and `ctx.today`.
        The lane's twin-digest skip (`unchanged_against_twin`) reads Qdrant and is a write-path decision,
        not a chunk: it is not here."""
        cand = doc.extra["candidate"]
        meta = {**DOCS_MARKDOWN_PACK_META, "trust_tier": cand.get("trust_tier") or DOCS_MARKDOWN_PACK_META["trust_tier"]}
        rows = build_rows_for_markdown(doc.url, doc.text, title=cand.get("title"), meta=meta)
        stamp = (cand.get("committed_at") or "")[:10] or None
        return [stamp_docs_row(r, meta=DOCS_MARKDOWN_PACK_META, published=stamp,
                               license_=cand.get("license") or DOCS_MARKDOWN_PACK_META.get("license"),
                               today=ctx.today)
                for r in rows]
