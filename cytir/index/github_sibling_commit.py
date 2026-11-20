"""Sibling-commit urls of a GitHub blob (spec `gh-repo-docs-retire-old-commit` section 4).

A refresh of a repo's docs stores every file under a NEW commit url
(`https://github.com/<owner>/<repo>/blob/<40-hex sha>/<path>`), so the old commit's rows are
never the same url group as the new rows. For a url of exactly this shape, "the same page" is
"same owner/repo (case-folded), same path (exact), other sha". A url of any other shape keeps
the drain's canonical-url-group rule.

Pure, no I/O and no import from the drain, so `cytir/index/promote_queue.py` can import it later
without a cycle. The retire-pack builder (`cytir.sources.refresh_direct.retire_old_commit`) and the
drain's survivor check must share this one definition (engineering-doctrine #1).
"""
from __future__ import annotations

import re

_BLOB = re.compile(
    r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<sha>[0-9a-f]{40})/(?P<path>.+)$")


def parse_blob_url(url: "str | None") -> "tuple[str, str, str, str] | None":
    """`(owner, repo, sha, path)` of a github blob url at a 40-hex commit, else None.
    Owner and repo keep their spelling; compare them case-folded."""
    if not isinstance(url, str):
        return None
    m = _BLOB.match(url)
    return (m["owner"], m["repo"], m["sha"], m["path"]) if m else None


def is_sibling_commit_url(old_url: str, candidate_url: str) -> bool:
    """True iff both urls are github blob urls at a commit sha, owner and repo are equal
    (case-folded), the path is equal (exact) and the sha differs."""
    a, b = parse_blob_url(old_url), parse_blob_url(candidate_url)
    if a is None or b is None:
        return False
    return (a[0].lower(), a[1].lower(), a[3]) == (b[0].lower(), b[1].lower(), b[3]) and a[2] != b[2]
