"""`cyt ops source-cards {build,embed,search}`: the door for the source-authority card commands.

The logic lives in `cytir.corpus.source_cards` (task 1235, ported from the former pipeline script);
this module is the one `cyt ops` entry for it, so the verb table stays "one verb, one door".
"""
from __future__ import annotations

from cytir.corpus.source_cards import main as _main


def main(argv: list[str] | None = None, prog: str = "cyt ops source-cards") -> int:
    return int(_main(argv, prog=prog) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
