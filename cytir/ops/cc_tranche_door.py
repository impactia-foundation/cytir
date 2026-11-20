"""`cyt ops cc-tranche <subverb>`: the one door to the CC tranche tooling (task 2355, S5 W3).

Each step of the CC growth pipeline lives in its own module and reads its own flags. This
dispatcher only names them; modules are imported lazily, so `--help` and a refused subverb
cost no heavy import. Every module behind the door has `main(argv, prog)` (DOOR CONVENTION).
The verb table itself lives in `cytir/cli.py` (`CC_TRANCHE_VERBS`), where the reach scoreboard reads it.
"""
from __future__ import annotations

import importlib
import sys

def usage(table: dict) -> str:
    lines = ["usage: cyt ops cc-tranche {" + ",".join(table) + "} [options]",
             "  one door to the CC tranche tooling; `cyt ops cc-tranche SUBVERB --help` lists its flags"]
    lines += [f"  {name:<14} {why}" for name, (_, _, why) in table.items()]
    return "\n".join(lines)


def main(table: dict, argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(usage(table), file=sys.stderr if not argv else sys.stdout)
        return 2 if not argv else 0
    if argv[0] not in table:
        print(f"unknown cc-tranche subverb {argv[0]!r}\n{usage(table)}", file=sys.stderr)
        return 2
    name = argv[0]
    module_name, prefix, _ = table[name]
    module = importlib.import_module(module_name)
    prog = f"cyt ops cc-tranche {name}"
    args = [*prefix, *argv[1:]]
    return int(module.main(args, prog=prog) or 0)
