"""`fr_insee_sirene_cards` -- INSEE SIRENE legal-unit register cards, on the Protocol (task 2728), PARTIAL.

The family was the last S7 row with neither an adapter nor a declaration. What exists in the package is
the ACQUIRE half: `cytir.sources.official_bulk.sirene_legal_units` fetches the StockUniteLegale parquet once
with `curl`, records its HTTP evidence (body, headers, transport metadata) and hands that set to
`cytir.sources.official_bulk.bulk`. This adapter is the Protocol door onto exactly that: the shared
official-bulk stages (`discover`, `acquire`, `seal_original`; see `official_bulk.py`), with the SIRENE
source identity pinned to the extractor's own constants (URL, content scope, next projection), never
repeated here.

Consent and licence are read from the contract row (`consent.search_ok` true, `train_ok` true, Etalab-2.0,
resolution `explicit-grant`); this module asserts none of it.

WHAT STAYS OUT (`PARTIAL = extract, chunk`), and why it is not invented here:
  * `extract`: no code turns a SIRENE row into a card. The stock file holds NO address: the ADR-060 recipe
    wants `{commune} ({departement})`, which lives in StockEtablissement (a second file, not fetched), and
    the NAF label expansion (activitePrincipaleUniteLegale -> label) has no table in the package.
    `compose_register_card` also needs a website `domain` the register does not carry. Rows to keep
    (active public legal persons, minus SCI 6540 and loi-1901 associations 9220) are the card_recipe's
    words, not code.
  * `chunk`: follows extract. The contract declares `chunk-policy-atomic-record.v1` (one card, one row).
  * The contract's `identity` unknown: the per-entity `#card:<SIREN>` point id of ADR-067 amendment 1 is
    ratified but not in code; the card step must adopt it.

Options (`--opt KEY=VALUE`): as the official-bulk families: `body`, `headers`, `metadata`, `requested_url`,
`root`, `source_id`, `source_name`, `content_scope`, `next_projection`, `fetch_vantage`, `country`.
"""
from __future__ import annotations

from cytir.sources.families.official_bulk import _OfficialBulkFamily
from cytir.sources.registry import register_family


@register_family
class FrInseeSireneCardsFamily(_OfficialBulkFamily):
    family_id = "fr_insee_sirene_cards"
    PARTIAL = ("extract", "chunk")
