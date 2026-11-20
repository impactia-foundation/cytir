"""The sole chunker entry point for recurring generic prose/doc sources.

Native-anchor law/article splitters deliberately do not import this module:
ADR-079 preserves their source-specific boundaries.  Every generic source that
constructs ``Passage`` records directly must use this wrapper so the tracked
build profile (or an explicitly pinned candidate) is never bypassed.
"""
from __future__ import annotations

from typing import Optional

from cytir.core.common import ChunkTextChannels
from cytir.chunk.dispatch import ChunkerSelection, chunk_markdown_channels, chunk_markdown_units


def chunk_generic_markdown(markdown: str) -> tuple[list[str], Optional[dict]]:
    """Backward-compatible text-only view of the verified central profile."""
    return chunk_markdown_units(markdown)


def chunk_generic_records(
        markdown: str, selection: Optional[ChunkerSelection] = None
        ) -> list[ChunkTextChannels]:
    """Principal generic interface with canonical, dense and sparse surfaces."""
    if selection is None:
        # Preserve the historical one-argument call contract for recurring
        # producer adapters and their byte-identity mocks.
        return chunk_markdown_channels(markdown)
    return chunk_markdown_channels(markdown, selection=selection)
