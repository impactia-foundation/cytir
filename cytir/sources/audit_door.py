"""`cyt source audit <family>`: the one door to the per-family pack audits (task 2352, S5 W4).

Eight audit modules each own a family's checks and their own required flags; this
dispatcher only names them. Each module exposes `main(argv, prog)` (task 2380 convention):
the door passes the family's flags as `argv` and `prog="cyt source audit <family>"`, so
no `sys.argv` is rebound. The modules stay where they are: moving them would break
their imports and tests for no gain in reach.
"""
from __future__ import annotations

import importlib
import sys

FAMILIES = {
    "hn": "cytir.sources.hn.audit_profile",
    "rfc-editor": "cytir.sources.rfc_editor.audit_profile",
    "service-public": "cytir.sources.service_public.audit_profile",
    "routec": "cytir.sources.routec.audit_profile",
    "stackoverflow": "cytir.sources.stackoverflow.audit_profile",
    "context7": "cytir.ops.audits.context7_software_docs_pack",
    "finma-partitions": "cytir.ops.audits.finma_web_pdf_partitions",
    "tech-product": "cytir.ops.audits.technical_product_profile",
}


def usage() -> str:
    return ("usage: cyt source audit {" + ",".join(FAMILIES) + "} [family options]\n"
            "  one door to the per-family pack audits; `cyt source audit FAMILY --help` "
            "lists that family's flags")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(usage(), file=sys.stderr if not argv else sys.stdout)
        return 2 if not argv else 0
    if argv[0] not in FAMILIES:
        print(f"unknown audit family {argv[0]!r}\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(FAMILIES[argv[0]])
    return int(module.main(argv[1:], prog=f"cyt source audit {argv[0]}") or 0)
