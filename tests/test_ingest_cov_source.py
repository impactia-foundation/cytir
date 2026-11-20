#!/usr/bin/env python3
"""Equivalence tests for the Task-351 source-runner adapter."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.helpers.repo_files import repo_file_or_skip

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from cytir.sources.cov import adapter as ADAPTER  # noqa: E402
import cytir.sources.pack_loader as PL  # noqa: E402


def _regdocs_plans():
    return PL.plans(PL.load_pack(repo_file_or_skip("sources/ch/regulator-docs.yaml")))


def test_regulator_pack_compiles_all_runnable_routes():
    plans = _regdocs_plans()
    assert len(plans) == 13
    assert sum(p.route == "source_owned_sitemap_fetch" for p in plans) == 12
    assert sum(p.route == "documented_source_api" for p in plans) == 1
    assert all(p.engine == "extractor" for p in plans)
    assert all(p.extractor == "cytir/sources/cov/adapter.py" for p in plans)


def test_every_regulator_plan_binds_one_exact_legacy_source():
    """The migration points at Task-351 config; it does not fork it."""
    seen = set()
    for plan in _regdocs_plans():
        binding = ADAPTER.resolve_binding(
            pack_id=plan.pack_id, route_id=plan.route_id,
            config_ref=plan.config_ref)
        legacy = binding["source"]
        seen.add((binding["config_pack"], binding["source_key"]))
        assert legacy["host"] == plan.route_id
        assert float(legacy["delay"]) == plan.delay_s
        assert int(legacy["cap"]) == int(plan.cap)
        assert legacy["license"] == plan.effective_license()
        assert float(legacy["trust"]) == plan.effective_trust()
        if legacy["mode"] == "sitemap":
            assert plan.route == "source_owned_sitemap_fetch"
            assert legacy["sitemaps"] == plan.params["urls"]
        else:
            assert legacy["mode"] == "sitecore_widget"
            assert plan.route == "documented_source_api"
            assert plan.params["endpoint"] == legacy["base"] + legacy["api_path"]
    assert len(seen) == 13


def test_suva_declares_both_origins_in_plan_and_legacy_engine():
    suva = next(p for p in _regdocs_plans() if p.route_id == "www.suva.ch")
    assert suva.effective_hosts() == [
        "www.suva.ch", "suva-p-001.sitecorecontenthub.cloud"]
    binding = ADAPTER.resolve_binding(
        pack_id=suva.pack_id, route_id=suva.route_id,
        config_ref=suva.config_ref)
    assert binding["source"]["ok_hosts"] == [
        "www.suva.ch", "suva-p-001.sitecorecontenthub.cloud"]


@pytest.mark.parametrize("host,kept,dropped,lang", [
    ("www.swissmedic.ch",
     "https://www.swissmedic.ch/swissmedic/fr/home/medicaments/a.html",
     "https://www.swissmedic.ch/swissmedic/fr/home/ueber-uns/karriere/a.html", "fr"),
    ("www.oak-bv.admin.ch",
     "https://www.oak-bv.admin.ch/fr/weisungen/w-01",
     "https://www.oak-bv.admin.ch/fr/organisation/equipe", "fr"),
    ("www.suva.ch",
     "https://www.suva.ch/it-ch/download/schede-tematiche/a",
     "https://www.suva.ch/it-ch/download/videos/a", "it"),
    ("www.enisa.europa.eu",
     "https://www.enisa.europa.eu/publications/report-a",
     "https://www.enisa.europa.eu/topics/report-a", "en"),
])
def test_compiled_selection_and_language_examples(host, kept, dropped, lang):
    plan = next(p for p in _regdocs_plans() if p.route_id == host)
    assert plan.predicate()(kept)
    assert not plan.predicate()(dropped)
    assert plan.lang_fn()(kept) == lang


def test_adapter_plan_is_read_only_and_reports_config_digest(capsys):
    rc = ADAPTER.main([
        "--pack-id", "ch-regulator-docs", "--route-id", "www.finma.ch",
        "--config-path", "data/cov351_routes.yaml",
        "--config-pack", "regdocs1", "--config-source", "finma_docs2",
        "--stage", "plan"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "sitecore_widget"
    assert len(report["config_sha256"]) == 64


def test_binding_refuses_cross_route_config():
    with pytest.raises(ADAPTER.AdapterError, match="cross-route"):
        ADAPTER.resolve_binding(
            pack_id="ch-regulator-docs", route_id="www.weko.admin.ch",
            config_ref={"path": "data/cov351_routes.yaml",
                        "pack": "regdocs1", "source": "finma_docs2"})


def test_routes_file_ships_under_data_and_its_urls_files_resolve_under_the_work_dir(monkeypatch):
    """Task 825: the routes live in data/ (shipped); a relative `urls_file` names no box path."""
    import yaml
    from cytir.sources.cov import engine as COV
    routes = REPO / "data" / "cov351_routes.yaml"
    assert routes.is_file() and not (HERE / "cov351_routes.yaml").exists()
    files = [s["urls_file"] for p in yaml.safe_load(routes.read_text())["packs"].values()
             for s in p["sources"].values() if s.get("urls_file")]
    assert files and not any(Path(f).is_absolute() for f in files)
    monkeypatch.setenv("COV351_WORK", "/w")
    assert COV.resolve_urls_file(files[0]) == "/w/" + files[0]
    assert COV.resolve_urls_file("/abs/x.jsonl") == "/abs/x.jsonl"
