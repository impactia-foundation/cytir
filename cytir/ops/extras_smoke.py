"""Prove one pyproject extra installs AND imports (task 2153, scoreboard S1).

    python -m cytir.ops.extras_smoke <extra>

Imports every module of ``EXTRA_MODULES[<extra>]`` and prints one line each
(``ok <module>`` or ``FAIL <module>: <exception>``). Exit 1 on any failure.
charsplit is in no extra (PyPI refuses direct-URL requirements; task 2989), so it is not smoked here.
Stdlib only, so it runs in a venv that holds just that extra.

The map is the ONE place that says what each extra must make importable:
the third-party top-level modules its pins provide, plus the ``aw`` modules
that import them at module top level. ``tests/test_aw_extras_smoke.py`` holds
it equal to ``pyproject.toml``. ``dev`` is the CI test install and is proven
by the whole suite, so it has no entry here.
"""

from __future__ import annotations

import importlib
import sys

EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    # build pipeline (embedding, extraction, fetch lanes)
    "pipeline": (
        "FlagEmbedding",
        "sentence_transformers",
        "transformers",
        "qdrant_client",
        "trafilatura",
        "httpx",
        "bs4",
        "lxml",
        "mwparserfromhell",
        "mwxml",
        "duckdb",
        "yaml",
        "orjson",
        "tqdm",
        "zstandard",
        "pyarrow",
        "jsonschema",
        "htmldate",
        "boto3",
        "lmdb",
        "py7zr",
        "pypdf",
        "playwright",
        "aiohttp",
        "sklearn",
        "cytir.embed.model",
        "cytir.extract.wikipedia_dump",
        "cytir.index.b1_corpus_kv",
        "cytir.retune.policy_train",
    ),
    # language scope gate: a separate venv (numpy<2)
    "langgate": (
        "numpy",
        "fasttext",
        "trafilatura",
        "bs4",
        "lxml",
        "httpx",
        "yaml",
        "orjson",
        "duckdb",
        "tqdm",
    ),
    # PDF and table extraction
    "pdf": (
        "docling",
        "liteparse",
        "cytir.extract.pdf",
    ),
    # agent-probe .reader variant and `cyt read --mcp` (task 2779)
    "reader": (
        "httpx",
        "lxml",
        "mcp",
        "trafilatura",
        "cytir.reader.mcp",
    ),
    # MCP / HTTP serving stack
    "serve": (
        "fastapi",
        "uvicorn",
        "httpx",
        "qdrant_client",
        "mcp",
        "cytir.serve.app",
        "cytir.serve.mcp_server",
        "cytir.serve.models",
    ),
    # the torch carriers of the serve path: in-process query embedding, the opt-in reranker (task 2281)
    "embed-local": (
        "FlagEmbedding",
        "sentence_transformers",
        "transformers",
    ),
}


def smoke(extra: str) -> list[tuple[str, str | None]]:
    """Return ``(module, error or None)`` per module of ``extra``."""
    out: list[tuple[str, str | None]] = []
    for name in EXTRA_MODULES[extra]:
        try:
            importlib.import_module(name)
            out.append((name, None))
        except BaseException as exc:  # a broken wheel may raise anything
            out.append((name, f"{type(exc).__name__}: {exc}"))
    return out


def main(argv: list[str] | None = None, prog: str | None = None) -> int:   # prog accepted, unused (hand-parsed)
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in EXTRA_MODULES:
        print(f"usage: python -m cytir.ops.extras_smoke {{{'|'.join(EXTRA_MODULES)}}}", file=sys.stderr)
        return 2
    failed = 0
    for name, err in smoke(args[0]):
        if err is None:
            print(f"ok {name}")
        else:
            failed += 1
            print(f"FAIL {name}: {err}")
    print(f"extra {args[0]}: {failed} failing of {len(EXTRA_MODULES[args[0]])}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
