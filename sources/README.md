# sources/ — the declarative source layer

Packs (`<pack>/<source>.yaml`, schema in `_schema/`) + the legislative files:
`POLICY.md` (admission law) and `PORTFOLIO.md` (allocation).

## Change process (Tim, 2026-07-25; narrowed by ADR-110, Tim 2026-09-23)

> Superseded in part by ADR-110: the bullet below over-read "pack
> `admission:` blocks" as always legislative. A TRANCHE of a family/host set
> `PORTFOLIO.md` already admits is executive (index-governance.md §2) — the
> orchestrator merges it on green CI, like any receipt PR. What stays
> legislative, unchanged: `POLICY.md`, `PORTFOLIO.md`'s matrix, a NEW
> family/host not yet in PORTFOLIO, a consent class, the gold set, and a
> gate contract.

- **`POLICY.md`, `PORTFOLIO.md`, a NEW family/host's `admission:` block, a
  consent class, the gold set, and gate contracts change ONLY via GitHub
  pull request, approved/merged by the board (Tim).** Merging the PR IS the
  ratification act (POLICY §9); the PR history is the public legislative
  record. The curator (Bender) authors these PRs, never self-merges them.
- A TRANCHE `admission:` change — more rows/hosts for a family/host set
  PORTFOLIO already admits — is executive (ADR-110): the orchestrator opens
  and merges it on green CI, same as any receipt PR.
- Everything else (specs, tasks, code, pack non-admission fields) lands
  direct-to-main per normal flow.
- Conversational ratifications recorded in `decisions.md` before this date
  remain valid (POLICY v1, ADR-067…075 era).

## Acquisition and preservation

Every pack pins `acquisition_policy: aw-acquisition-v1` and
`outlink_policy: capture-and-receipt`. The policy in
`data/acquisition-policy.v1.json` requires feed/dump/API/Common-Crawl-first
acquisition, preserves original → Markdown → chunks → outlinks → vectors, and
uses direct origin crawling only as a bounded fallback. An empty external-link
result is still a required receipted stage.
