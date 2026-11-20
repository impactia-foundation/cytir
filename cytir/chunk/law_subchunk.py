#!/usr/bin/env python3
"""`chunk.law.sub` -- sub-chunk the BODY of a long law article, keep the article.

Task 573, experiment E573-1 (`data/experiments-ledger.v1.jsonl`). PROPOSED, not
ratified: ratification is ADR-090 amendment 2 and is Tim's merge. Until then the
policy artifact's registry status is `proposed`, which
`chunk_policy_registry.assert_selectable` refuses for any lane that does not
declare itself an experiment.

WHAT THIS IS, AND WHAT IT REFUSES TO BE
---------------------------------------
ADR-090 3 split the chunker family in two: `chunk.generic` (the structured
packer) and `chunk.law` (per-article splitters over source-native anchors).
ADR-090 4 then bound "standardize on 1152" to `chunk.generic` alone and said
it may NEVER be read as authority to sweep `chunk.law` into the generic policy.
ADR-090 amendment 1 restated the same bound in the other direction when it
returned the density class to production granularity: "`chunk.law` is
untouched -- ADR-090 4's bound binds this policy in both directions."

This module keeps that bound. The ARTICLE stays the outer boundary. Nothing
here merges two articles, moves a boundary between articles, or lets the
generic packer decide where an article begins. The only thing it changes is
what happens INSIDE one article that is wider than the tail: instead of one
288+ word chunk, several, each still the same article.

WHY THAT MIGHT MATTER
---------------------
Task 569 measured the dilution mechanism on the density class, one variable, on
one fixed page set: dense-only h@1 28 -> 21 -> 15 -> 6 as the width goes
192 -> 384 -> 768 -> 1024, and over 79 gold-channel pairs the dense leg prefers
narrow 42:1. Task 567 9 leaves a 14 h@1 residual whose largest named factor is
DL-NORM -- the candidate's rows wider than production's. ADR-090 amendment 1
exempted legifrance and `gesetze_*` from the return to production granularity
because they are "1.000-1.012 chunks/doc -- nothing to merge". That argument is
about merging and says nothing about splitting: task 573's census found 21.73%
of legifrance articles and 17.12% of gesetze-im-internet articles longer than
288 words (183,394 law article rows corpus-wide).

THE PAYLOAD CONTRACT (corpus-doctrine 1: the representation IS the contract)
---------------------------------------------------------------------------
Every sub-chunk of an article carries:

  * `anchor`      -- the article's source-native anchor, VERBATIM. A sub-chunk
                     that lost its anchor would be a passage the URL fragment
                     no longer reaches, which is the whole value of the law
                     class. `assert_subchunk_contract` refuses it.
  * `article_id`  -- the article's own identifier within its law.
  * `law_id`      -- the instrument the article belongs to.
  * the article's IDENTITY PREFIX inside the embedded text of every sub-chunk
    OF A SPLIT ARTICLE. This clause exists because it has already been violated
    once: the epoch-4 re-materialization dropped fedlex's law name /
    abbreviation / SR number from the dense surface, and no count, hash or width
    check could see it (tasks 524-527; memory `embedding-surface-is-a-contract`).
    Slice 3 of an article whose first slice carried "SR 220 -- Art. 336 OR" and
    whose third carried nothing would be exactly that defect, one generation
    later.

    It binds only where a split happened. An article emitted whole keeps the
    surface it has today, byte for byte -- the guard may not demand of an
    unchanged row a property the row never had, or "short articles unchanged"
    stops being true. That asymmetry means a SPLIT row differs from its
    whole-article baseline in two ways, boundary and prefix, so E573-1 carries a
    third arm (`compose_prefix=False`) that changes only the boundary and lets
    the two be told apart.

IDENTITY AND PRECEDENCE
-----------------------
  * An article that fits in ONE chunk is emitted byte-identically and keeps the
    point_id `chunk.law` mints today (`law_sub_anchor_component` returns the
    component unchanged at `n_sub == 1`). Short articles are unchanged in text
    AND in id.
  * An article that is SPLIT mints a NEW id for every sub-chunk, first included
    (`<component>~s<k>`, 1-based). The old single-chunk id is superseded, and
    superseding happens at the PAGE GROUP -- the unit task 565 used -- so a
    page is never half old-generation and half new.
  * Tier is resolved by `aw_b1_identity.classify_tier`, never re-implemented
    here: a positional or title-equal anchor still demotes to Tier B, where the
    per-chunk digest already separates the slices and no sub-ordinal is needed.

    cyt chunk law-sub --self-check      # cuts the bundled fixtures
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from cytir.core.paths import data_path
import cytir.core.ids as ident  # the only legal id/anchor home
from cytir.chunk import chunker as CH  # the one chunker; never re-implemented here
import cytir.chunk.registry as REG
from cytir.chunk.policy import ChunkPolicy, load_chunk_policy

REPO = pathlib.Path(__file__).resolve().parents[2]
POLICY_PATH = ("data/chunk-policy-law-article-sub-192w-o0-tail-288-tok2560"
               ".v1.json")
POLICY_FAMILY = "chunk.law.sub"
SCHEMA = "aw.law-subchunk.v1"


class LawSubchunkError(Exception):
    """An article this module cannot honestly cut, or a contract it breaks."""


@dataclass(frozen=True)
class SubChunk:
    """One passage of one law article."""
    ordinal: int                 # 1-based, within the article
    n_sub: int                   # how many the article produced
    text: str                    # what gets embedded
    n_words: int
    anchor: Optional[str]        # the ARTICLE's native anchor, verbatim
    article_id: str
    law_id: Optional[str]
    identity_prefix: str
    tier: str
    digest: str
    anchor_component: str
    point_id: str
    chunk_epoch: int
    chunk_rule: str
    split: bool                  # n_sub > 1
    #: Whether `identity_prefix` was composed INTO `text`. False is the
    #: `compose_prefix=False` control arm: the prefix is still recorded, because
    #: it is a property of the article and the guard needs it to tell a peeled
    #: header from a deleted sentence.
    prefix_composed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def policy(path: str | pathlib.Path | None = None) -> ChunkPolicy:
    """The pinned `chunk.law.sub` artifact, checked against the registry.

    `experiment=True` is passed because the policy is `proposed`: an
    unratified policy is selectable by a measurement arm and by nothing else
    (`chunk_policy_registry.EXPERIMENT_SELECTABLE`). When ADR-090 amendment 2
    is merged the registry entry becomes `current` and this call keeps working
    unchanged -- the flag stops being load-bearing rather than being removed.
    """
    p = load_chunk_policy(path or data_path(pathlib.Path(POLICY_PATH).name))
    fam = REG.family_of(p.sha256)
    if fam != POLICY_FAMILY:
        raise LawSubchunkError(
            f"{p.policy_id}: registry family {fam!r} is not {POLICY_FAMILY!r}")
    REG.assert_selectable(p, experiment=True)
    return p


@contextlib.contextmanager
def _active(p: ChunkPolicy):
    """Install `p` in `chunk`'s module globals and PUT THE PREVIOUS ONE BACK.

    Copied in shape, deliberately, from `chunk_reference_selector._active`,
    whose docstring records why: the chunker's active policy is process-wide
    state, and a caller that leaves the last one installed silently
    re-configures every later caller in the same process.
    """
    before = CH.active_policy()
    CH._apply_policy(p, allow_env_overrides=False)
    try:
        yield
    finally:
        CH._apply_policy(before, allow_env_overrides=True)


def default_identity_prefix(text: str, title: str | None) -> str:
    """The article's identity line: its own first line, else its title.

    Both shapes exist in the store and neither is guessed at cut time:
    `gesetze-im-internet`, `laws-lois.justice.gc.ca` and the Swiss cantonal
    collections put "<law> -- <article>" on the article's first line, while
    `fedlex` rows start straight into "Art. 62 ..." and carry "SR 101 -- Art.
    62 ..." only in `title`. A caller that knows its family may pass its own
    prefix; this is the default, not a rule.
    """
    head = (text or "").split("\n", 1)[0].strip()
    body_follows = "\n" in (text or "")
    if head and body_follows and len(head.split()) <= 60:
        return head
    return (title or "").strip()


def subchunk_article(canonical_url: str, text: str, *,
                     anchor: str | None,
                     article_id: str,
                     law_id: str | None = None,
                     title: str | None = None,
                     source_type: str | None = None,
                     identity_prefix: str | None = None,
                     compose_prefix: bool = True,
                     chunk_policy: ChunkPolicy | None = None,
                     stats: dict | None = None) -> list[SubChunk]:
    """Cut ONE law article into one or more passages of that same article.

    Returns a single `SubChunk` holding `text` verbatim when the article fits
    the policy's tail; several when it does not. Never merges across articles,
    and never returns an empty list for a non-empty article -- every article
    reaches exactly one terminal (corpus-doctrine 2).
    """
    if not isinstance(canonical_url, str) or not canonical_url.strip():
        raise LawSubchunkError("canonical_url must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        raise LawSubchunkError(f"{canonical_url}: article text is empty")
    if not isinstance(article_id, str) or not article_id.strip():
        raise LawSubchunkError(
            f"{canonical_url}: article_id is required -- a sub-chunk that "
            "cannot name its article is not a law row")

    p = chunk_policy or policy()
    prefix = (identity_prefix if identity_prefix is not None
              else default_identity_prefix(text, title))
    prefix = prefix.strip()
    if not prefix:
        raise LawSubchunkError(
            f"{canonical_url}#{article_id}: no identity prefix. Splitting an "
            "article whose slices cannot each name their law is the epoch-4 "
            "fedlex defect (tasks 524-527); refused rather than shipped.")

    st = dict(collisions=0, articles=0, split=0, whole=0) if stats is None else stats
    for k in ("collisions", "articles", "split", "whole"):
        st.setdefault(k, 0)
    st["articles"] += 1

    # Does it need splitting at all? The threshold is the policy's, read from
    # the artifact -- not a literal here (engineering-doctrine 2).
    threshold = int(p.law_article["split_body_above_words"]) if p.law_article \
        else p.word_hard_max
    if CH.word_count(text) <= threshold:
        st["whole"] += 1
        return [_mint(canonical_url, text, ordinal=1, n_sub=1, anchor=anchor,
                      article_id=article_id, law_id=law_id, title=title,
                      source_type=source_type, identity_prefix=prefix,
                      policy=p, stats=st, used_ids=set())]

    # Split the BODY, not the article-plus-header. The prefix is peeled off
    # first when the article's own text carries it, because leaving it in makes
    # the packer emit it as its own 8-word section -- a slice that is the title
    # and nothing else, measured on the legifrance fixture before this line
    # existed. Peeled off and re-composed, every slice carries the identity AND
    # a body.
    #
    # The peel happens in BOTH arms, `compose_prefix` or not. It decides where
    # the cuts fall, and the control arm exists to vary the composition alone:
    # peeling in one arm and not the other would move the boundaries too and
    # the third arm would stop isolating anything.
    body = text
    if body.startswith(prefix):
        body = body[len(prefix):].lstrip("\n")
        if not body.strip():
            body = text          # a header-only article: nothing to peel
    with _active(p):
        slices = [c.text for c in CH.chunk_markdown(body)]
    if not slices:
        raise LawSubchunkError(
            f"{canonical_url}#{article_id}: the packer returned no chunks for "
            f"a {CH.word_count(text)}-word article -- an error is not an empty")
    if len(slices) == 1:
        # The token/word packer declined to split (one indivisible unit). That
        # is a legal outcome, not a failure: emit the article whole, with the
        # id it has today.
        st["whole"] += 1
        return [_mint(canonical_url, text, ordinal=1, n_sub=1, anchor=anchor,
                      article_id=article_id, law_id=law_id, title=title,
                      source_type=source_type, identity_prefix=prefix,
                      policy=p, stats=st, used_ids=set())]

    st["split"] += 1
    # `compose_prefix=False` is the CONTROL arm of E573-1: the same boundaries
    # with no identity re-stated on the slices. It exists so the experiment can
    # say which of the two changes -- narrower passages, or each passage naming
    # its law -- moved the number, instead of reporting one arm that changed
    # both (verification-doctrine 3).
    composed = ([s if s.startswith(prefix) else f"{prefix}\n{s}" for s in slices]
                if compose_prefix else list(slices))
    n = len(composed)
    used: set[str] = set()
    out = [_mint(canonical_url, t, ordinal=i, n_sub=n, anchor=anchor,
                 article_id=article_id, law_id=law_id, title=title,
                 source_type=source_type, identity_prefix=prefix,
                 prefix_composed=compose_prefix,
                 policy=p, stats=st, used_ids=used)
           for i, t in enumerate(composed, 1)]
    return out


def _mint(canonical_url: str, text: str, *, ordinal: int, n_sub: int,
          anchor, article_id: str, law_id, title, source_type,
          identity_prefix: str, policy: ChunkPolicy, stats: dict,
          used_ids: set[str], prefix_composed: bool = True) -> SubChunk:
    digest = ident.content_digest(text)
    tier, comp = ident.classify_tier(source_type, anchor, True, digest,
                                     title=title)
    # Tier A carries a real per-article anchor, so the sub-ordinal is what
    # separates the slices. Tier B/C already have a per-chunk digest as the
    # component and need no ordinal -- adding one there would invent identity
    # where content already provides it.
    if tier == "A":
        comp = ident.law_sub_anchor_component(comp, ordinal, n_sub)
    salt = 0
    pid = ident.mint_point_id(canonical_url, comp)
    while pid in used_ids:
        salt += 1
        stats["collisions"] += 1
        pid = ident.mint_point_id(
            canonical_url, ident.salted_anchor_component(comp, salt))
    used_ids.add(pid)
    return SubChunk(
        ordinal=ordinal, n_sub=n_sub, text=text, n_words=CH.word_count(text),
        anchor=(None if anchor in (None, "") else str(anchor)),
        article_id=article_id, law_id=law_id, identity_prefix=identity_prefix,
        tier=tier, digest=digest,
        anchor_component=ident.salted_anchor_component(comp, salt),
        point_id=pid, chunk_epoch=policy.epoch, chunk_rule=policy.rule_legend,
        split=n_sub > 1, prefix_composed=prefix_composed)


# ---------------------------------------------------------------------------
# The guard. Wired into every producer, fails closed, carries an exit code.
# ---------------------------------------------------------------------------

def assert_subchunk_contract(subs: Iterable[SubChunk], *,
                             anchor, article_id: str, article_text: str,
                             identity_prefix: str | None = None) -> None:
    """Refuse a cut that lost the article.

    This is the check that would have caught the defect this policy is most
    likely to introduce, and the one the ledger says this project re-learns at
    full price when it lives only in prose (engineering-doctrine 10). It runs
    on every article the materializer cuts, not on request.

    Raises `LawSubchunkError` -- callers propagate; `__main__` exits non-zero.
    """
    subs = list(subs)
    if not subs:
        raise LawSubchunkError(
            f"{article_id}: zero sub-chunks for a non-empty article")
    n = subs[0].n_sub
    want_anchor = None if anchor in (None, "") else str(anchor)
    prefix = (identity_prefix or subs[0].identity_prefix).strip()

    for s in subs:
        if s.n_sub != n:
            raise LawSubchunkError(
                f"{article_id}: sub-chunk {s.ordinal} claims n_sub={s.n_sub}, "
                f"sibling claims {n}")
        if s.anchor != want_anchor:
            raise LawSubchunkError(
                f"{article_id}: sub-chunk {s.ordinal} carries anchor "
                f"{s.anchor!r}, the article's is {want_anchor!r} -- a "
                "sub-chunk that lost its anchor is a passage the URL fragment "
                "no longer reaches")
        if s.article_id != article_id:
            raise LawSubchunkError(
                f"{article_id}: sub-chunk {s.ordinal} carries article_id "
                f"{s.article_id!r}")
        if n > 1 and prefix and s.prefix_composed and prefix not in s.text:
            raise LawSubchunkError(
                f"{article_id}: sub-chunk {s.ordinal}/{n} does not carry the "
                f"article's identity prefix {prefix[:60]!r} -- this is the "
                "epoch-4 fedlex surface defect (tasks 524-527)")
        if not s.text.strip():
            raise LawSubchunkError(f"{article_id}: empty sub-chunk text")

    if len({s.ordinal for s in subs}) != len(subs):
        raise LawSubchunkError(f"{article_id}: duplicate sub-chunk ordinals")
    if sorted(s.ordinal for s in subs) != list(range(1, n + 1)):
        raise LawSubchunkError(
            f"{article_id}: ordinals {sorted(s.ordinal for s in subs)} are not "
            f"1..{n}")
    if len({s.point_id for s in subs}) != len(subs):
        raise LawSubchunkError(
            f"{article_id}: two sub-chunks share a point_id -- the A33 collapse")

    if n == 1:
        if subs[0].text != article_text:
            raise LawSubchunkError(
                f"{article_id}: a whole article must be emitted byte-identically; "
                "a short article that changed text changed its point_id too")
        return

    # Word conservation, as a MULTISET and not as a sum.
    #
    # The sum was the first version of this check and it was wrong in both
    # directions, which the cantonal collection proved within minutes of the
    # first real run. Too STRICT for the no-prefix arm: the article's own
    # header line is peeled off before the split, so 297 words in came back as
    # 292 and 390 perfectly good articles were refused. Too LOOSE for the
    # prefix arm: every slice repeats the header, so an article could lose real
    # sentences and still sum higher than it started -- padding hiding a
    # deletion, which is the exact shape of failure this guard exists to catch.
    #
    # A multiset is immune to both. Every word of the article must appear in
    # the slices, except words the peeled identity prefix accounts for.
    have = collections.Counter(" ".join(s.text for s in subs).split())
    want = collections.Counter(article_text.split())
    missing = want - have
    if prefix:
        missing = missing - collections.Counter(prefix.split())
    if missing:
        sample = ", ".join(sorted(missing)[:8])
        raise LawSubchunkError(
            f"{article_id}: {sum(missing.values())} word(s) of the article are "
            f"in no sub-chunk ({sample}) -- text was lost in the split")


# ---------------------------------------------------------------------------

FIXTURES = REPO / "tests" / "fixtures" / "law_subchunk"


def _self_check() -> int:
    p = policy()
    print(f"policy {p.policy_id}@{p.epoch} sha256={p.sha256[:12]}... "
          f"family={REG.family_of(p.sha256)} status={REG.status_of(p.sha256)}")
    if not FIXTURES.exists():
        raise SystemExit(f"FATAL: fixtures missing at {FIXTURES}")
    rc = 0
    for path in sorted(FIXTURES.glob("*.json")):
        fx = json.loads(path.read_text())
        subs = subchunk_article(
            fx["canonical_url"], fx["text"], anchor=fx.get("anchor"),
            article_id=fx["article_id"], law_id=fx.get("law_id"),
            title=fx.get("title"), source_type=fx.get("source_type"),
            chunk_policy=p)
        try:
            assert_subchunk_contract(subs, anchor=fx.get("anchor"),
                                     article_id=fx["article_id"],
                                     article_text=fx["text"])
        except LawSubchunkError as exc:
            print(f"  FAIL {path.name}: {exc}")
            rc = 1
            continue
        print(f"  ok {path.name:<34} words={CH.word_count(fx['text']):>5} -> "
              f"{len(subs)} sub-chunk(s) "
              f"{[s.n_words for s in subs]} tier={subs[0].tier}")
    return rc


def main(argv: list[str] | None = None, prog: str = "cyt chunk law-sub") -> int:
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args(argv)
    if a.self_check:
        return _self_check()
    ap.error("nothing to do: pass --self-check")
    return 2


if __name__ == "__main__":
    sys.exit(main())
