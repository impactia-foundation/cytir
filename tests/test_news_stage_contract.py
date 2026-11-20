from __future__ import annotations

import json
from pathlib import Path

import pytest

import cytir.news.stage_contract as subject
from tests.helpers.repo_files import repo_file_or_skip


def _state(**statuses: str) -> dict:
    return {"hosts": {host: {"status": statuses.get(host, "ok")}
                      for host in subject.PUBLISHERS}}


def test_source_contract_accepts_complete_or_explicitly_empty_publishers():
    state = _state(**{"www.swissinfo.ch": "empty"})
    result = subject.validate_sources(state, {"completeness": "time-window-complete"})
    assert result["publishers"]["www.swissinfo.ch"] == "empty"


def test_source_contract_rejects_transport_error_instead_of_staging_partial_run():
    state = _state(**{"www.swissinfo.ch": "error"})
    with pytest.raises(subject.NewsStageContractError, match="swissinfo.ch=error"):
        subject.validate_sources(state, {"completeness": "time-window-complete"})


def test_source_contract_rejects_best_effort_hn_fallback():
    with pytest.raises(subject.NewsStageContractError, match="HN source boundary incomplete"):
        subject.validate_sources(_state(), {"completeness": "best-effort-current-lists"})


def test_source_contract_cli_has_no_implicit_partial_success(tmp_path: Path, monkeypatch):
    state = tmp_path / "state.json"
    state.write_text(json.dumps(_state(**{"www.rts.ch": "error"})))
    receipt = tmp_path / "hn.json"
    receipt.write_text(json.dumps({"completeness": "time-window-complete"}))
    monkeypatch.setattr("sys.argv", ["contract", "source-boundary", "--state", str(state),
                                      "--hn-receipt", str(receipt)])
    with pytest.raises(SystemExit, match="rts.ch=error"):
        subject.main()


# task 508 / task 500 §7.1 — the one home for the news lane's per-day,
# per-arm staging collection name, shared by news_refresh_stage.sh (which
# stages) and the task-452 tier-2 apply script (which reads the same
# collection back). Two independent shell re-derivations of this string is
# exactly the mirror-contract fork this project's doctrine forbids.

def test_collection_name_at_the_current_r7_width_is_unchanged():
    """Every collection this lane has EVER staged at width 128 was named
    without a width suffix. That name must never change under this
    refactor -- news_tier2_apply.sh's `collection_of()` reads it back."""
    assert subject.news_stage_collection_name("2026-08-28", "chmedia", 128) \
        == "aw_dnews_20260828_chmedia"
    assert subject.news_stage_collection_name("2026-09-02", "swissinfo", 128) \
        == "aw_dnews_20260902_swissinfo"


def test_collection_name_at_a_non_default_width_is_a_different_named_collection():
    """The flip-day repoint to 1024 (task 500 §7.1) must target a collection
    a 128-dim upsert could never collide with by name."""
    name = subject.news_stage_collection_name("2026-09-08", "chmedia", 1024)
    assert name == "aw_dnews1024_20260908_chmedia"
    assert name != subject.news_stage_collection_name("2026-09-08", "chmedia", 128)


def test_collection_name_rejects_a_non_positive_width():
    with pytest.raises(ValueError):
        subject.news_stage_collection_name("2026-09-08", "chmedia", 0)


def test_collection_name_cli_matches_the_python_function(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["contract", "collection-name", "--day", "2026-09-08",
                                      "--label", "chmedia", "--mrl-dim", "1024"])
    subject.main()
    printed = capsys.readouterr().out.strip()
    assert printed == subject.news_stage_collection_name("2026-09-08", "chmedia", 1024)


# task 677 (follow-up of 676): a run that deliberately covers one publisher names it.

def test_expected_hosts_default_is_all_four_and_stays_strict():
    assert subject.parse_expected_hosts(None) == subject.PUBLISHERS
    assert subject.parse_expected_hosts("") == subject.PUBLISHERS
    state = _state(**{"www.srf.ch": "error"})
    with pytest.raises(subject.NewsStageContractError, match="srf.ch=error"):
        subject.validate_sources(state, {"completeness": "time-window-complete"})


def test_expected_hosts_checks_only_the_named_publisher():
    # the other three hosts were never fetched: absent from state, as in a real
    # one-publisher backfill spool
    state = {"hosts": {"www.srf.ch": {"status": "ok"}}}
    hn = {"completeness": "time-window-complete"}
    result = subject.validate_sources(state, hn, subject.parse_expected_hosts("www.srf.ch"))
    assert result["publishers"] == {"www.srf.ch": "ok"}
    with pytest.raises(subject.NewsStageContractError, match="www.srf.ch=missing"):
        subject.validate_sources({"hosts": {}}, hn, ("www.srf.ch",))
    # and the default still refuses the very same state
    with pytest.raises(subject.NewsStageContractError, match="swissinfo.ch=missing"):
        subject.validate_sources(state, hn)


@pytest.mark.parametrize("bad", ["www.nzz.ch", "www.srf.ch,srf.ch", " , "])
def test_expected_hosts_refuses_unknown_names_and_an_empty_list(bad):
    with pytest.raises(subject.NewsStageContractError):
        subject.parse_expected_hosts(bad)


def test_expected_host_in_error_still_refuses():
    state = {"hosts": {"www.rsi.ch": {"status": "error"}}}
    with pytest.raises(subject.NewsStageContractError, match="rsi.ch=error"):
        subject.validate_sources(state, {"completeness": "time-window-complete"},
                                 ("www.rsi.ch",))


def test_arms_for_hosts_follow_the_stage_routes():
    assert subject.arms_for_hosts(subject.PUBLISHERS) == ("chmedia", "swissinfo")
    assert subject.arms_for_hosts(("www.swissinfo.ch",)) == ("swissinfo",)
    assert subject.arms_for_hosts(("www.rts.ch", "www.rsi.ch")) == ("chmedia",)


def test_stage_script_routes_match_the_contract_host_arm_map():
    """The stage script no longer keeps a literal routes dict: it asks this module (`routes`), which reads the
    publisher registry (tests/test_news_publisher_registry.py pins the daily four). HOST_ARM holds every
    registered host; PUBLISHERS only the daily run's required ones."""
    text = repo_file_or_skip(Path("deploy") / "scripts" / "news_refresh_stage.sh").read_text()
    assert '"swissinfo.ch": "swissinfo"' not in text and "cytir.news.stage_contract routes" in text
    routes = {"swissinfo.ch": "swissinfo", "srf.ch": "chmedia",
              "rts.ch": "chmedia", "rsi.ch": "chmedia"}
    for host, arm in routes.items():
        assert subject.HOST_ARM["www." + host] == arm
    assert set(subject.PUBLISHERS) <= set(subject.HOST_ARM)
    assert set(subject.HOST_ARM) == set(subject.ALL_PUBLISHERS)


def test_cli_expected_hosts_and_arms(tmp_path: Path, monkeypatch, capsys):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"hosts": {"www.swissinfo.ch": {"status": "empty"}}}))
    receipt = tmp_path / "hn.json"
    receipt.write_text(json.dumps({"completeness": "time-window-complete"}))
    monkeypatch.setattr("sys.argv", ["contract", "source-boundary", "--state", str(state),
                                      "--hn-receipt", str(receipt),
                                      "--expected-hosts", "www.swissinfo.ch"])
    subject.main()
    assert json.loads(capsys.readouterr().out)["publishers"] == {"www.swissinfo.ch": "empty"}
    monkeypatch.setattr("sys.argv", ["contract", "source-boundary", "--state", str(state),
                                      "--hn-receipt", str(receipt)])
    with pytest.raises(SystemExit, match="srf.ch=missing"):
        subject.main()
    monkeypatch.setattr("sys.argv", ["contract", "arms", "--expected-hosts", "www.srf.ch"])
    subject.main()
    assert capsys.readouterr().out.strip() == "chmedia"
