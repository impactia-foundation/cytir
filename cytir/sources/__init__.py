"""cytir.sources -- the source-family interface and its adapters (task 643 S7, task 683).

    family.py      the `SourceFamily` Protocol: discover -> acquire -> extract -> chunk, with
                   consent, the L0 seal, the terminals, embed and pack composition owned by
                   the framework (read its docstring first)
    registry.py    `@register_family`, and `admit()`: a family runs only if
                   `data/source-family-contracts.v1.json` declares it
    run_family.py  `cyt source run <family_id>`: the framework; `cyt source families` lists them
    contract.py    the one validator of the contracts file (was cytir/sources/contract.py)
    pack_loader.py, runner.py   the pack YAML door (`cyt source run <pack.yaml>`), unchanged
    families/      one module per family: fedlex (dump), sitemap (a pack's sitemap route),
                   cc (Common Crawl), news, openlegaldata; `web.py` is the shared HTML tail

Adding a source is a declaration (a POLICY PR on the contracts file) plus one class in
`families/`. See ADDING-A-SOURCE.md, "Writing the adapter".
"""
from __future__ import annotations
