"""The news source queue: the one reader of the checked-in `data/news-queue.example.json`.

The file lists the publisher hosts the news refresh fetches (one row per host: domain, sitemap
sources, section routes). It ships beside the news scripts of a checkout or a release, and four
places used to open it by hand: the roster census, the sealed source-evidence builder, the release
file list, and the stage script's default queue. They all go through this module now, so a move of
the file is a one-line change here.

    python -m cytir.news.source_queue path [ROOT]     print the queue file's path under ROOT (default: this checkout)
"""
from __future__ import annotations

import json
import pathlib
import sys

from cytir.core.paths import data_path

#: The queue file's name and its directory under a checkout or release root.
QUEUE_DIR = "data"
QUEUE_NAME = "news-queue.example.json"
#: The host table beside it (task 2600: moved here from the old withheld directory).
HOST_META_NAME = "news-host-meta.v1.json"

# Task 2107: the file moved from the old withheld shell-script directory (as news_queue.example.json) to
# `data/`. No fallback to the old place: every reader of a release root (cron preflight, stage
# script, chain, seal) runs the code OF that release (PYTHONPATH=<release>), so code and file move
# together; a release cut before this change carries old code that reads its own old path.


def host_meta_relpath() -> str:
    """The host table's path relative to a checkout or release root, with `/` separators."""
    return f"{QUEUE_DIR}/{HOST_META_NAME}"


def queue_relpath() -> str:
    """The queue file's path relative to a checkout or release root, with `/` separators."""
    return f"{QUEUE_DIR}/{QUEUE_NAME}"


def queue_path(root: pathlib.Path | str | None = None) -> pathlib.Path:
    """The queue file of the checkout or release at `root` (default: the checkout this module is in)."""
    if root is None:
        return data_path(QUEUE_NAME)
    return pathlib.Path(root) / QUEUE_DIR / QUEUE_NAME


def load_queue(root: pathlib.Path | str | None = None) -> list[dict]:
    """The queue rows, parsed."""
    return json.loads(queue_path(root).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ["path"] or len(argv) > 2:
        print("usage: python -m cytir.news.source_queue path [ROOT]", file=sys.stderr)
        return 64
    print(queue_path(argv[1] if len(argv) == 2 else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
