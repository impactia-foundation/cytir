"""Hit-rank rule of the v2 chooser, and the aggregate of one arm. Pure."""
from __future__ import annotations

import urllib.parse


def norm(u: str) -> str:
    if not u:
        return ""
    p = urllib.parse.urlsplit(u.strip())
    return f"{p.netloc.lower()}{p.path.rstrip('/')}"


def hit_rank(target: str, urls: list[str]) -> int | None:
    """1-based rank of the first URL that names `target` (the rule task 533's chooser arms used;
    `tests/test_aw_retune.py` pins it)."""
    t = norm(target)
    for i, u in enumerate(urls):
        n = norm(u)
        if n == t or n.startswith(t) or t.startswith(n):
            return i + 1
    return None


def aggregate(per_q: dict) -> dict:
    """h@1, h@3, h@10, MRR over the golds of one set (`per_q[qid]["rank"]` is None for a miss)."""
    n = len(per_q)
    ranks = [v.get("rank") for v in per_q.values()]
    return {"n": n, "n_gold": n,
            "h1": sum(1 for r in ranks if r == 1),
            "h3": sum(1 for r in ranks if r and r <= 3),
            "h10": sum(1 for r in ranks if r and r <= 10),
            "mrr": round(sum(1.0 / r for r in ranks if r) / n, 4) if n else None}
