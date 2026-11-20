"""cytir.index.id_rule -- the ADR-067 I1 id of a row (am.1: cards per entity), ONE definition.

Three callers, one rule:

* the `aw_e10` rebuild (deleted builder `aw.index.rebuild_e10`, task 1312; spec `specs/archive/aw-e10-rebuild.md` E3): `rekey` PROVES a
  stored row's `point_id` with the formula of its tier (and the canon epoch that minted it) and returns
  the row's new id;
* a content-pack composer (`key_points_for`, task 864): the points of a pack for a collection that
  declares I1 are keyed by `mint_i1` before the pack is sealed (the drain mints the WRITTEN id of a
  sealed pack itself: `promote_sealed_pack.project_rows(id_rule=...)`);
* the drain's writer contract after the flip (task 847, spec section 6 step 2): a collection whose
  declaration names `"id_rule": "I1"` (`data/collection-payload-profiles.v1.json`) takes a point only
  when its id IS the I1 id of its own row, minted with today's canon (`mint_i1`). The check runs on the
  pack row BEFORE the payload projection (the projection of profile `rebuild_e10` drops `tier` and
  `point_id`, which the formula reads), and refuses the WHOLE batch before the first PUT.

The I1 formula (ADR-067, `cytir.core.ids.mint_point_id`): `uuid5(AW_NAMESPACE, canonical_url + "#" +
component)`, the component by `tier`: A the native `anchor`, B the 64-hex `content_digest`, C "".
A card (`supersede.is_card`) is `canonical_url + "#card:" + primary_identity`, or the url-only id when
it has no identity (am.1).

Imports `cytir.core` and `cytir.index.supersede` (the card test) only; no import from the operator tool tree.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from cytir.core.canon import CANON_EPOCH, CANON_EPOCHS, canonical_url, canonical_url_at, strip_tracking_url
from cytir.core.ids import mint_point_id
from cytir.index.supersede import is_card, primary_identity

#: The value of a declaration's `id_rule` key that switches the guard on.
ID_RULE_I1 = "I1"
ID_RULES = (ID_RULE_I1,)


class IdRuleRefused(RuntimeError):
    """A batch holds a point whose id is not the I1 id of its own row."""

    def __init__(self, collection: str, bad: list[dict]):
        self.collection, self.bad = collection, bad
        super().__init__(
            "REFUSED: %d point(s) for %r are not keyed by the I1 id of their own row (ADR-067 I1 + "
            "am.1; the collection declares \"id_rule\": \"I1\"), e.g. %s. A pack composed for the "
            "old id space must be re-composed. A tombstone with an old id must be "
            "re-composed with the I1 id of its row (the old-to-new id map of the deleted aw_e10 builder is in git history). "
            "Nothing was written."
            % (len(bad), collection, json.dumps(bad[:3])))


def rekey(pl: Mapping[str, Any]) -> tuple[str | None, str, str]:
    """(new id, kind, why). E3, ADR-067 I1 + am.1, spec #611 section 4.3.

    The GENERATION-1 id of a row is `mint_point_id(canonical_url(url), component)` with the
    component of its `tier`: A the native `anchor`, B the 64-hex `content_digest`, C "" (url only).
    It must equal the stored `point_id`; a row where it does not is returned with id None (the
    caller refuses: an id the formula cannot reproduce is not the ratified id).
    A card (`supersede.is_card`) then moves to `canon + "#card:" + primary_identity`, or keeps the
    url-only id when it has no identity. Every other row's new id IS its generation-1 id."""
    url = pl.get("url") or ""
    if not url:
        return None, "P", "no-url"
    cu = canonical_url(url)
    tier = pl.get("tier")
    if tier == "A":
        comp = pl.get("anchor")
        if not comp:
            return None, "P", "tier-A-without-anchor"
    elif tier == "B":
        comp = pl.get("content_digest") or ""
        if len(comp) != 64:
            return None, "P", "tier-B-without-64-hex-digest"
    elif tier == "C":
        comp = ""
    else:
        return None, "P", f"tier-{tier!r}"
    gen1 = mint_point_id(cu, comp)
    proved = ""
    if gen1 != pl.get("point_id"):
        # canon epochs (ADR-067; review 773 M1): a stored id is PROVED with the epoch that minted it
        # (`canon@1` = before #704 added `authuser`); the new id is MINTED with today's canon, so an
        # identical-text `?authuser=` variant collides with its canonical page (an E3' group whose
        # survivor is the canonical spelling) and the url is rewritten at write (`rewrite_url`).
        for ep in sorted(CANON_EPOCHS, reverse=True):
            if ep == CANON_EPOCH:
                continue
            cu_ep = canonical_url_at(url, ep)
            if cu_ep != cu and mint_point_id(cu_ep, comp) == pl.get("point_id"):
                proved = f"+canon@{ep}"
                break
        else:
            return None, "P", f"tier-{tier}-point_id-not-reproduced"
    if is_card(pl.get("source_type"), pl.get("entity_kind"), pl.get("qid")):
        tok = primary_identity(dict(pl))
        if tok:
            return mint_point_id(cu, f"card:{tok}"), "C", "card:" + tok.split(":", 1)[0] + proved
        return mint_point_id(cu, ""), "C", "card:none" + proved
    return gen1, "P", f"tier-{tier}" + proved


def rewrite_url(url: str) -> str | None:
    """The stored url a row gets at the rebuild, or None to keep it: `strip_tracking_url(url)` when
    today's canon strips a key that the canon of an earlier epoch kept (the row was captured before
    that key joined the tracking set, review 773 section 5.3). Every other url stays as captured."""
    if not url:
        return None
    new = strip_tracking_url(url)
    for ep in CANON_EPOCHS:
        if ep != CANON_EPOCH and strip_tracking_url(url, epoch=ep) != new:
            return new if new != url else None
    return None


def mint_i1(pl: Mapping[str, Any]) -> tuple[str | None, str]:
    """(the I1 id of a row minted with TODAY's canon, why). None when the row cannot have one (no
    url, an unknown tier, tier A without an anchor, tier B without a 64-hex digest). A new row is
    minted with today's canon, so unlike `rekey` no earlier canon epoch is consulted."""
    url = pl.get("url") or ""
    if not url:
        return None, "no-url"
    cu = canonical_url(url)
    if is_card(pl.get("source_type"), pl.get("entity_kind"), pl.get("qid")):
        tok = primary_identity(dict(pl))
        return (mint_point_id(cu, f"card:{tok}"), "card") if tok else (mint_point_id(cu, ""), "card:none")
    tier = pl.get("tier")
    if tier == "A":
        comp = pl.get("anchor")
        if not comp:
            return None, "tier-A-without-anchor"
    elif tier == "B":
        comp = pl.get("content_digest") or ""
        if len(comp) != 64:
            return None, "tier-B-without-64-hex-digest"
    elif tier == "C":
        comp = ""
    else:
        return None, f"tier-{tier!r}"
    return mint_point_id(cu, comp), f"tier-{tier}"


def check_points(points: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Every point whose id is not `mint_i1` of its own payload: `[{id, expected, why}]`."""
    bad = []
    for p in points:
        want, why = mint_i1(p.get("payload") or {})
        if want is None or str(p.get("id")) != want:
            bad.append({"id": str(p.get("id")), "expected": want, "why": why})
    return bad


def declared_id_rule(collection: str, *, profiles_path=None) -> str | None:
    """The `id_rule` the PHYSICAL collection's declaration names, or None (undeclared, or no key:
    the guard is off -- every collection before `aw_e10`). An unknown value is refused."""
    from cytir.index import payload_profile as PP
    declared = PP.load_profiles(profiles_path)
    rule = (declared.get(collection) or {}).get("id_rule")
    if rule is not None and rule not in ID_RULES:
        raise PP.PayloadProfileError("%s: collection %s declares unknown id_rule %r (known: %s)"
                                     % (PP.PROFILES_PATH.name, collection, rule, list(ID_RULES)))
    return rule


def key_points_for(collection: str, points: list[dict], *,
                   profiles_path=None) -> tuple[list[dict], dict[str, str]]:
    """A CONTENT-PACK composer's door to the target's id rule (task 864). A content pack carries
    built points, so its composer, not the drain, names the ids. `collection` is the pack's target
    (the manifest's `collection`). It declares no `id_rule` (every collection before the aw_e10
    flip): the points come back unchanged with an empty map. It declares I1: each point's id
    becomes `mint_i1` of its own payload (ADR-067 I1 + am.1: a card with an identity is keyed
    `canon#card:<identity>`), and `{old id: new id}` names every id that moved.

    Refused (`IdRuleRefused`, nothing returned): a point with no I1 id, and two points that mint
    ONE I1 id (a composer never merges two rows by accident; the drain would write one over the
    other). The drain MERGES such rows of a sealed pack instead (`choose_survivor`): a sealed pack
    is staged rows, a content pack is points its composer already chose (review 869 nit b)."""
    if declared_id_rule(collection, profiles_path=profiles_path) is None:
        return points, {}
    bad, out, moved, seen = [], [], {}, {}
    for p in points:
        old = str(p.get("id"))
        new, why = mint_i1(p.get("payload") or {})
        if new is None:
            bad.append({"id": old, "expected": None, "why": why})
            continue
        if new in seen:
            bad.append({"id": old, "expected": new, "why": f"same I1 id as {seen[new]}"})
            continue
        seen[new] = old
        if new != old:
            moved[old] = new
        out.append(dict(p, id=new))
    if bad:
        raise IdRuleRefused(collection, bad)
    return out, moved


def choose_survivor(members: list[tuple[str, Mapping]], kind: str) -> str:
    """Spec section 2 step 4 (E3'), ONE rule for the aw_e10 rebuild AND the drain (review 869 S2:
    two rows of one pack that mint one I1 id keep the same row the rebuild would). Pages: the row whose URL is the canonical spelling, else the newest
    `last_seen`, else the smallest old id. Cards: the register-sourced card (officers come only
    from the register), else the newest, else the smallest old id."""
    def key(m):
        old, pl = m
        url = pl.get("url") or ""
        first = (bool(pl.get("register_id")) if kind == "C" else url == canonical_url(url))
        return (not first, _neg_date(pl.get("last_seen")), old)
    return sorted(members, key=key)[0][0]


def _neg_date(v) -> str:
    """Sort key so that a NEWER `last_seen` sorts first and a missing one last."""
    s = str(v or "")
    return "".join(chr(0x10FFFF - ord(c)) for c in s) if s else "\U0010FFFF" * 11


def refuse_unless_i1(collection: str, points: list[Mapping[str, Any]], *, profiles_path=None) -> int:
    """The drain guard: when `collection` declares `id_rule`, raise `IdRuleRefused` if one point of
    the batch is not keyed by its I1 id. Returns the number of points checked (0: guard off)."""
    if declared_id_rule(collection, profiles_path=profiles_path) is None:
        return 0
    bad = check_points(points)
    if bad:
        raise IdRuleRefused(collection, bad)
    return len(points)
