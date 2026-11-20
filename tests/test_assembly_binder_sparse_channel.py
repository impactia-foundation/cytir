"""The binder's two-channel read, and the scope that keeps it from spreading.

`assembly_binder.py:1389` fed ONE surface to both legs; every ADR-060 card and
every task-417 law row therefore had its BM25 posting built from the dense
prose (task 527 §5: jaccard 0.168 against production's posting, 0 of 40
matching).  The fix reads the record's own `sparse_text` -- but only for a
family whose representation contract DECLARES one, because eight epoch-4 chunk
populations also seal a `sparse_text` and reading theirs would be a second
unruled representation change (the 528a stage-channel probe).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cytir.index import assemble_stage as AB  # noqa: E402


def test_the_card_family_declares_its_own_sparse_surface():
    fam = AB._representation_family("task467_entity_card_1024_reembed", None)
    assert fam.get("sparse_surface_field") == "sparse_text"


def test_the_law_pack_family_resolves_by_its_source_dir():
    fam = AB._representation_family("task417_families", "gesetze-official-current-v1")
    assert fam.get("sparse_surface_field") == "sparse_text"


def test_the_epoch4_chunk_lanes_do_not_declare_one():
    """These seal a `sparse_text` too. They must keep the single-surface path
    until someone measures what changing them does."""
    for sid in ("task472_f2_html_1024_embed",
                "task472_cc_materializer_1024_embed",
                "task492_crawl30_cc_chunks",
                "task482_f2_consent_supplement_chunks"):
        fam = AB._representation_family(sid, None)
        assert fam, f"{sid} has no representation family at all"
        assert not fam.get("sparse_surface_field"), sid


def test_an_unlisted_set_resolves_to_the_safe_default():
    assert AB._representation_family("no_such_set_id_at_all", None) == {}


def test_channel_surfaces_consumes_the_pair_atomically():
    """The consumer the binder hands the pair to: both keys or neither, and a
    half-set pair fails loud rather than becoming an unmeasured hybrid."""
    import pytest
    pytest.importorskip("qdrant_client",
                        reason="build_aw_b1 imports the qdrant client at module "
                               "level; present on the build host, not on every "
                               "test runner")
    from cytir.index import b1_projection as B

    assert B._channel_surfaces({"text": "body"}, "body") == ("body", "body", False)
    assert B._channel_surfaces(
        {"text": "prose", "dense_text": "prose", "sparse_text": "hooks"},
        "prose") == ("prose", "hooks", True)
    try:
        B._channel_surfaces({"text": "prose", "sparse_text": "hooks"}, "prose")
    except ValueError:
        pass
    else:
        raise AssertionError("an incomplete channel pair must fail loud")
