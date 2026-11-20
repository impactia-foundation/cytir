"""Task 2254: the `twin-fixture` role of `cytir.retune.guard.refuse_non_twin`.

The CI job's throwaway collection `aw_gate_fixture` is declared `twin-fixture`. The guard accepts it
only under an explicit operator overlay that names the endpoint, and never for prod, the prod alias
or a serving collection. The twin's own tests (tests/test_aw_retune.py) stay as they are.
"""
from __future__ import annotations

import json

import pytest

from cytir.retune import guard

FIXTURE = "aw_gate_fixture"
SS_NONE = "LISTEN 0 128 127.0.0.1:6999 0.0.0.0:* users:((\"python3\",pid=1,fd=3))\n"


def _overlay(tmp_path, monkeypatch, port=6701):
    f = tmp_path / "operator.json"
    f.write_text(json.dumps({"schema": "aw.operator-overlay.v1",
                         "settings": {"twin_url": f"http://127.0.0.1:{port}"}}))
    monkeypatch.setenv("AW_OPERATOR_OVERLAY", str(f))
    return f"http://127.0.0.1:{port}"


def test_the_fixture_is_declared_and_is_not_a_twin_collection():
    assert FIXTURE in guard.fixture_collections()
    assert FIXTURE not in guard.twin_collections()


def test_fixture_accepted_under_overlay_endpoint_and_no_alias(tmp_path, monkeypatch):
    url = _overlay(tmp_path, monkeypatch)
    guard.refuse_non_twin(url, FIXTURE, aliases={}, ss_text=SS_NONE)


def test_fixture_refused_without_the_overlay(tmp_path, monkeypatch):
    monkeypatch.delenv("AW_OPERATOR_OVERLAY", raising=False)
    from cytir.core import twin
    with pytest.raises(SystemExit) as e:
        guard.refuse_non_twin(twin.configured_url(), FIXTURE, aliases={}, ss_text=SS_NONE)
    assert e.value.code == guard.EXIT_REFUSED and "overlay" in str(e.value)


def test_fixture_refused_on_an_endpoint_the_overlay_did_not_name(tmp_path, monkeypatch):
    url = _overlay(tmp_path, monkeypatch, port=6701)
    with pytest.raises(SystemExit):
        guard.refuse_non_twin(url.replace("6701", "6702"), FIXTURE, aliases={}, ss_text=SS_NONE)


def test_fixture_refused_when_an_alias_of_prod_resolves_to_it(tmp_path, monkeypatch):
    url = _overlay(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        guard.refuse_non_twin(url, FIXTURE, aliases={"aw_prod": FIXTURE}, ss_text=SS_NONE)
    with pytest.raises(SystemExit):
        guard.refuse_non_twin(url, FIXTURE, aliases={"aw_prod": "x", FIXTURE: "x"}, ss_text=SS_NONE)


def test_fixture_refused_when_the_alias_list_is_unreadable(tmp_path, monkeypatch):
    url = _overlay(tmp_path, monkeypatch)
    monkeypatch.setattr(guard, "fetch_aliases", lambda u: None)
    with pytest.raises(SystemExit):
        guard.refuse_non_twin(url, FIXTURE, ss_text=SS_NONE)


@pytest.mark.parametrize("coll", ["aw_prod", "aw_e10", "aw_e9"])
def test_prod_alias_and_serving_collections_stay_refused_under_the_overlay(tmp_path, monkeypatch, coll):
    url = _overlay(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        guard.refuse_non_twin(url, coll, aliases={"aw_prod": "aw_e10"}, ss_text=SS_NONE)


def test_a_fixture_marked_serving_is_not_a_fixture_collection(monkeypatch):
    from cytir.index import payload_profile as PP
    real = PP.load_profiles()
    bad = {n: dict(e) for n, e in real.items()}
    bad[FIXTURE]["serving"] = True
    monkeypatch.setattr(PP, "load_profiles", lambda path=None: bad)
    assert FIXTURE not in guard.fixture_collections()
    assert FIXTURE not in guard.twin_collections()


def test_a_fixture_is_not_a_live_name_but_a_serving_one_still_is(monkeypatch):
    from cytir.index import live_target as LT
    from cytir.index import payload_profile as PP
    assert FIXTURE not in LT.live_collection_names()
    assert "aw_e10" in LT.live_collection_names()
    bad = {n: dict(e) for n, e in PP.load_profiles().items()}
    bad[FIXTURE]["serving"] = True
    monkeypatch.setattr(PP, "load_profiles", lambda path=None: bad)
    assert FIXTURE in LT.live_collection_names()


def test_fixture_refused_when_the_overlay_does_not_name_the_endpoint(tmp_path, monkeypatch):
    f = tmp_path / "operator.json"
    f.write_text(json.dumps({"schema": "aw.operator-overlay.v1", "settings": {"twin_collection": "aw_e8"}}))
    monkeypatch.setenv("AW_OPERATOR_OVERLAY", str(f))
    from cytir.core import twin
    with pytest.raises(SystemExit) as e:
        guard.refuse_non_twin(twin.configured_url(), FIXTURE, aliases={}, ss_text=SS_NONE)
    assert e.value.code == guard.EXIT_REFUSED and "twin_url" in str(e.value)


def test_fixture_refused_on_an_endpoint_that_holds_aw_prod_even_if_prod_guard_cannot_read(tmp_path, monkeypatch):
    from cytir.index import prod_guard
    url = _overlay(tmp_path, monkeypatch)
    monkeypatch.setattr(prod_guard, "_fetch_aliases", lambda u: None)
    prod_guard._ALIASES.clear()
    with pytest.raises(SystemExit) as e:
        guard.refuse_non_twin(url, FIXTURE, aliases={"aw_prod": "aw_e10"}, ss_text=SS_NONE)
    assert e.value.code == guard.EXIT_REFUSED and "aw_prod" in str(e.value)
