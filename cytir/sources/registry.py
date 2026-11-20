"""The source-family registry (task 683, S7): which families exist, and whether they may run.

A family is registered by decorating its class with `@register_family`. It is ADMITTED to run
only when `data/source-family-contracts.v1.json` declares it: an undeclared family is a
refusal, not a default (corpus-doctrine #7). The registry READS the contracts file through
the one validator (`cytir.sources.contract`, moved here from the retired script directory in this task so the
package can use it); it keeps no copy of any declared value.

Adding a family:
  1. declare it in data (a POLICY PR: `data/source-family-contracts.v1.json`);
  2. write one class implementing `cytir.sources.family.SourceFamily`;
  3. name its module in `FAMILY_MODULES` below (or, out of tree, in `$AW_FAMILY_MODULES`).
`cyt source families` lists what is registered and what each one's declaration says.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys
from typing import Callable

# Started by path (`python3 aw/...py`), `sys.path[0]` is this file's own directory, not the repo
# root: put this checkout's root first so `import cytir` resolves HERE and never to a foreign editable
# install (the shim defensive-import and script-start-guard tests).
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cytir.chunk import registry as chunk_registry
from cytir.sources import contract as contract_mod
from cytir.sources.family import (CHUNK_DECLARED, CHUNK_DISPATCHER, NATIVE_POLICIES, ChunkSpec,
                               SourceFamily)

#: Modules that register the in-tree families. One line per adapter module.
FAMILY_MODULES = (
    "cytir.sources.families.fedlex",
    "cytir.sources.families.sitemap",
    "cytir.sources.families.cc",
    "cytir.sources.families.loda",
    "cytir.sources.families.dila_juri",
    "cytir.sources.families.dila_kali",
    "cytir.sources.families.dila_jorf",
    "cytir.sources.families.openlegaldata",
    "cytir.sources.families.eric_official_pdf",
    "cytir.sources.families.finma_circulars",
    "cytir.sources.families.news_daily",
    "cytir.sources.families.technical_manufacturer_catalog",
    "cytir.sources.families.rfc_editor",
    "cytir.sources.families.de_laender_juris",
    "cytir.sources.families.us_federal_register_ecfr",
    "cytir.sources.families.govuk_content",
    "cytir.sources.families.service_public_fiche",
    "cytir.sources.families.service_public_dila_exact",
    "cytir.sources.families.stackexchange_nonso_qa_p0",
    "cytir.sources.families.openalex_academic_dump",
    "cytir.sources.families.stackoverflow_qa",
    "cytir.sources.families.bfs_dam",
    "cytir.sources.families.statcan_wds",
    "cytir.sources.families.statcan_wds_preserved",
    "cytir.sources.families.doab_monographs",
    "cytir.sources.families.entscheidsuche",
    "cytir.sources.families.hf_model_cards",
    "cytir.sources.families.ch_cantonal_law",
    "cytir.sources.families.github_agent_resources",
    "cytir.sources.families.gesetze",
    "cytir.sources.families.official_bulk",
    "cytir.sources.families.authority_packs",
    "cytir.sources.families.ctgov_psy",
    "cytir.sources.families.justice_canada",
    "cytir.sources.families.context7_exact",
    "cytir.sources.families.bfs_pxweb",
    "cytir.sources.families.ch_zefix_cards",
    "cytir.sources.families.fr_insee_sirene_cards",
    "cytir.sources.families.pubmed",
    "cytir.sources.families.routec_generic",
    "cytir.sources.families.rights_canon",
    "cytir.sources.families.exact_url_preserved",
    "cytir.sources.families.ietf_drafts",
    "cytir.sources.families.w3c",
    "cytir.sources.families.swisstopo",
    "cytir.sources.families.exact_url_preserved",
)
#: Out-of-tree adapters: comma-separated importable module names.
EXTRA_MODULES_ENV = "AW_FAMILY_MODULES"

_FACTORIES: dict[str, Callable[[], SourceFamily]] = {}


class FamilyRefused(Exception):
    """The registry will not run this family (undeclared, retired, or at odds with its
    declaration). The message names the rule."""


def register_family(factory):
    """Class decorator. The factory (usually the class) builds one adapter and must carry a
    `family_id` class attribute: the key of its row in the contracts file."""
    fid = getattr(factory, "family_id", None)
    if not fid or not isinstance(fid, str):
        raise TypeError(f"{factory!r} has no `family_id`: a family names its contract row")
    prior = _FACTORIES.get(fid)
    if prior is not None and prior is not factory and (
            getattr(prior, "__module__", None) != getattr(factory, "__module__", None)
            or getattr(prior, "__qualname__", None) != getattr(factory, "__qualname__", None)):
        raise ValueError(f"family {fid!r} is already registered by {prior!r}: "
                         "a family has one adapter")
    _FACTORIES[fid] = factory
    return factory


def load_modules(extra: str | None = None) -> None:
    """Import every module that registers a family (idempotent)."""
    names = list(FAMILY_MODULES)
    extra = extra if extra is not None else os.environ.get(EXTRA_MODULES_ENV, "")
    names += [n.strip() for n in extra.split(",") if n.strip()]
    for name in names:
        importlib.import_module(name)


def registered() -> list[str]:
    load_modules()
    return sorted(_FACTORIES)


def resolve_chunk_spec(declared: dict, policy_id: str) -> ChunkSpec:
    """The family's chunk policy, from the contract's declaration, checked against the
    registry (`data/chunk-policy-registry.v1.json`). `policy_id` is what the ADAPTER says it
    chunks under; it must agree with the declaration."""
    cp = declared.get("chunk_policy") or {}
    want = cp.get("policy_id")
    if want in (None, contract_mod.UNKNOWN):
        if policy_id not in (*NATIVE_POLICIES, CHUNK_DISPATCHER):   # CHUNK_DECLARED needs a policy
            raise FamilyRefused(
                f"{declared.get('family_id')}: the adapter chunks under {policy_id!r} but the "
                "contract declares no chunk policy (`declared` needs one). Declare it (POLICY PR), or have the "
                "adapter say it chunks natively (`CHUNK_NATIVE`) or by the dispatcher's "
                "pinned selection (`CHUNK_DISPATCHER`).")
        return ChunkSpec(declared=policy_id)
    if policy_id not in (want, CHUNK_DECLARED):
        raise FamilyRefused(
            f"{declared.get('family_id')}: the contract declares chunk policy {want!r} but "
            f"the adapter chunks under {policy_id!r}. A declaration and its plugin must agree.")
    if want in NATIVE_POLICIES:
        # `native` / `native-chunked` are explicit declared values (ruling 26 item 20c, ruling 27
        # T6 a): the chunk is the source's own article (the second may split an over-window one).
        # No registry entry, no sha256, no epoch (the contract validator checks that).
        if policy_id != want:
            raise FamilyRefused(
                f"{declared.get('family_id')}: the contract declares chunk policy {want!r} but "
                f"the adapter chunks under {policy_id!r}. A declaration and its plugin must agree.")
        return ChunkSpec(declared=want)
    sha = cp.get("sha256")
    try:
        reg = chunk_registry.entry(sha)
    except chunk_registry.ChunkPolicyRegistryError as exc:
        raise FamilyRefused(f"{declared.get('family_id')}: {exc}") from exc
    # identity is the sha256 (the contract's `policy_id` spells the file name, the registry's
    # the bare policy name; comparing the two spellings would refuse a correct declaration)
    return ChunkSpec(declared=want, sha256=sha, epoch=cp.get("epoch"),
                     policy_family=reg.get("policy_family"),
                     registry_status=reg.get("status"))


def admit(family_id: str, *, strict: bool = False,
          contracts_path: str | pathlib.Path | None = None) -> tuple[SourceFamily, dict, ChunkSpec]:
    """Return `(adapter, its contract row, its chunk spec)` or raise `FamilyRefused`.

    Refuses: an unregistered id; a family with no contract row; a retired family; a row that
    fails the one validator (`--strict` also refuses every `unknown` value); an adapter whose
    chunk policy disagrees with its declaration."""
    load_modules()
    factory = _FACTORIES.get(family_id)
    if factory is None:
        raise FamilyRefused(f"no family {family_id!r} is registered "
                            f"(registered: {sorted(_FACTORIES)})")
    doc = contract_mod.load(contracts_path)
    row = contract_mod.by_id(doc).get(family_id)
    if row is None:
        raise FamilyRefused(
            f"family {family_id!r} has an adapter but no declaration in "
            f"{doc.get('_path')}. An undeclared family cannot run: declare it "
            "(POLICY PR, ratified by Tim's merge) first (corpus-doctrine #7).")
    if row.get("status") == "retired":
        raise FamilyRefused(f"family {family_id!r} is retired in its declaration; "
                            "a retired family does not run")
    try:
        contract_mod.assert_declared([family_id], doc, strict=strict, stage="cyt source run")
    except contract_mod.FamilyContractError as exc:
        raise FamilyRefused(str(exc)) from exc
    adapter = factory()
    # an adapter whose policy depends on its row (`SitemapFamily`: declared once the row names one)
    # answers per row; every other adapter has the one class constant
    policy_for = getattr(adapter, "chunk_policy_for", None)
    spec = resolve_chunk_spec(row, policy_for(row) if policy_for else adapter.chunk_policy_id)
    return adapter, row, spec
