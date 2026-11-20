"""cytir.gates.scratch_legs -- the gate's leg arms read on a candidate or scratch collection (task 863):
the leg comes from `cytir.gates.leg_definition` (one definition for both sides), prod and live names are
refused, the query sets are built from the committed instruments."""
from __future__ import annotations

import pathlib

import pytest

from cytir.gates import leg_definition as LD
from cytir.gates import scratch_legs as SL


def test_the_query_sets_are_the_committed_instruments():
    from cytir.core.paths import data_path
    if not pathlib.Path(data_path("eval_tasks.v2.yaml")).exists():
        pytest.skip("the gate's query sets are not in the public cut")
    q629 = SL.queries_of("629")
    assert len(q629) == 629 and len({q["id"] for q in q629}) == 629
    assert all(q["gold_urls"] and q["query"] for q in q629)
    ho = SL.queries_of("heldout")
    assert len(ho) > 1000 and len({q["id"] for q in ho}) == len(ho)
    assert all(q["gold_urls"] or q["gold_domains"] for q in ho)
    with pytest.raises(SL.ScratchLegRefused):
        SL.queries_of("nope")


def test_url_variants_cover_scheme_www_and_the_trailing_slash_without_the_fragment():
    v = SL.url_variants("https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_336")
    assert "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de" in v
    assert "http://fedlex.admin.ch/eli/cc/27/317_321_377/de/" in v
    assert not any("#" in x for x in v) and len(v) == 8
    assert SL.url_variants("") == []


def test_prod_and_live_names_are_refused_before_any_read(tmp_path):
    env = tmp_path / "e.env"
    env.write_text("AW_RANKHEAD=1\n")
    with pytest.raises(SL.ScratchLegRefused, match="prod"):
        SL.read("http://prod.invalid:1", "aw_prod", leg="full-stack", weights=None, env_file=str(env),
                tei="http://prod.invalid:1", instrument="629", out=str(tmp_path / "o.json"))


def test_the_leg_is_the_one_definition():
    with pytest.raises(LD.LegUndefined):
        LD.leg_env("dense-only", "/w.json", "0" * 64)
    assert LD.leg_env("full-stack", None, None) == {}
    assert SL.parse_env_text("AW_A=1\nPATH=/x\n# c\nAW_B=x=y\n") == {"AW_A": "1", "AW_B": "x=y"}
