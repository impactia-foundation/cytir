"""Offline orchestration tests for the entity-card website snapshot stage."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

import cytir.corpus.card_website_capture as driver
import cytir.corpus.entity_card_website as ECW


def association(entity="CHE-341.859.062", domain="example.ch"):
    return {
        "schema": ECW.ASSOCIATION_RECEIPT_SCHEMA,
        "entity": entity, "domain": domain, "status": "admitted",
        "evidence_class": "uid_on_page", "source": "fixture",
        "observed_at": "2026-07-14", "evidence_sha256": "e" * 64,
        "source_artifact_sha256": "d" * 64,
        "artifact_pointer": "master://task275/cards.jsonl#line=1",
        "resolve_confidence": 1.0, "resolve_method": "fixture_uid_on_page",
    }


def seed(entity="CHE-341.859.062", domain="example.ch"):
    return {
        "register_id": entity, "domain": domain,
        "url": f"https://www.{domain}/", "name": "Example Consulting AG",
        "title": "Example Consulting AG",
        "site_association_receipt": association(entity, domain),
    }


def capture(url="https://www.example.ch/"):
    return {
        "url": url, "warc_filename": "crawl-data/x.warc.gz",
        "offset": 10, "length": 500, "digest": "sha1:ABC",
        "fetch_time": "2026-07-22T08:00:00Z",
        "content_truncated": False, "truncation_reason": None,
    }


def policy(mode="cc", robots=""):
    return {
        "status": "ok", "search_ok": True, "train_ok": False,
        "checked_at": "2026-08-04T10:00:00Z", "mode": mode,
        "_robots_body": robots,
    }


class FakeServices:
    def __init__(self, tmp_path, *, candidate=(None, "miss"), fail_fetch=False):
        self.candidate_cache = {"example.ch": candidate}
        self.candidate_errors = {}
        self.cc_route = "common-crawl-s3-warc-range"
        self.tmp_path = pathlib.Path(tmp_path)
        self.fail_fetch = fail_fetch
        self.fetches = []
        self.direct_calls = []
        self.exact_calls = []

    def consent(self, url, *, direct):
        return policy("direct" if direct else "cc")

    def fetch_cc(self, record):
        self.fetches.append(record["url"])
        if self.fail_fetch:
            return None
        if record["url"].endswith("/about"):
            return b"<html><title>About Example Consulting AG</title>about body</html>"
        return (b'<html><title>Example Consulting AG</title>'
                b'<a href="/about">About</a>homepage body</html>')

    def decode_cc(self, body):
        return body.decode()

    def store_l0(self, url, fetched_at, body, extra):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        path = self.tmp_path / (hashlib.sha256(url.encode()).hexdigest() + ".zst")
        path.write_bytes(body)
        return {"path": path.name, "sha256": hashlib.sha256(body).hexdigest()}

    def resolve_exact(self, crawl, urls, *, complete):
        self.exact_calls.append((crawl, tuple(urls), complete))
        if complete and any(url.endswith("/about") for url in urls):
            return {url: capture(url) for url in urls if url.endswith("/about")}
        return {}

    def resolve_domains(self, crawl, domains, *, complete):
        return {domain: [] for domain in domains}

    def direct_get(self, url, domain):
        self.direct_calls.append(url)
        return {
            "url": url, "raw": b"<html>direct body</html>",
            "html": "<html>direct body</html>",
            "fetched_at": "2026-08-04T11:00:00Z",
            "retrieved_at": "2026-08-04T11:00:01Z",
            "policy": policy("direct"),
        }


def test_cc_hit_archives_once_per_page_and_keeps_l1_in_pinned_crawl(
        tmp_path, monkeypatch):
    services = FakeServices(tmp_path, candidate=(capture(), "exact"))
    monkeypatch.setattr(
        driver, "normalized_surface",
        lambda html, url, depth: (("Example Consulting AG homepage content"
                                   if depth == 0 else
                                   "Example Consulting AG services and products"),
                                  "fixture"))
    snapshot = driver.process_site(
        services, domain="example.ch", seed=seed(), crawl="CC-MAIN-2026-30",
        manifest_sha="a" * 64, max_l1=3, direct_fallback=True)
    assert snapshot["status"] == "ADMITTED"
    assert snapshot["route"] == "common-crawl-s3-warc-range"
    assert services.fetches == ["https://www.example.ch/",
                                "https://www.example.ch/about"]
    assert services.direct_calls == []
    assert snapshot["l1_refresh_complete"] is True
    assert snapshot["l1_pages"][0]["acquisition"]["crawl"] == "CC-MAIN-2026-30"
    assert snapshot["homepage"]["raw_sha256"] == hashlib.sha256(
        b'<html><title>Example Consulting AG</title>'
        b'<a href="/about">About</a>homepage body</html>').hexdigest()


def test_genuine_cc_miss_uses_direct_but_unusable_and_error_do_not(
        tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "normalized_surface",
                        lambda html, url, depth: ("Example Consulting content", "fixture"))
    missed = FakeServices(tmp_path, candidate=(None, "miss"))
    result = driver.process_site(
        missed, domain="example.ch", seed=seed(), crawl="crawl",
        manifest_sha="a" * 64, max_l1=0, direct_fallback=True)
    assert result["route"] == "direct-crawl-fallback"
    assert missed.direct_calls == ["https://www.example.ch/"]

    unusable = FakeServices(tmp_path, candidate=(None, "unusable"))
    result = driver.process_site(
        unusable, domain="example.ch", seed=seed(), crawl="crawl",
        manifest_sha="a" * 64, max_l1=0, direct_fallback=True)
    assert result["status"] == "CC_UNUSABLE"
    assert unusable.direct_calls == []

    broken = FakeServices(tmp_path, candidate=(capture(), "exact"), fail_fetch=True)
    with pytest.raises(driver.RetryableSiteError, match="WARC"):
        driver.process_site(
            broken, domain="example.ch", seed=seed(), crawl="crawl",
            manifest_sha="a" * 64, max_l1=0, direct_fallback=True)
    assert broken.direct_calls == []


def test_ccbot_path_rule_is_checked_for_the_selected_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "normalized_surface",
                        lambda html, url, depth: ("Example Consulting content", "fixture"))
    services = FakeServices(
        tmp_path, candidate=(capture("https://www.example.ch/private"), "exact"))
    services.consent = lambda url, direct=False: policy(
        "direct" if direct else "cc",
        "User-agent: CCBot\nDisallow: /private\n")
    result = driver.process_site(
        services, domain="example.ch", seed=seed(), crawl="crawl",
        manifest_sha="a" * 64, max_l1=0, direct_fallback=False)
    assert result["status"] == "CC_CONSENT_DECLINED_DIRECT_DISABLED"
    assert services.fetches == []

    fallback = FakeServices(
        tmp_path, candidate=(capture("https://www.example.ch/private"), "exact"))
    fallback.consent = lambda url, direct=False: policy(
        "direct" if direct else "cc",
        "User-agent: CCBot\nDisallow: /private\n")
    result = driver.process_site(
        fallback, domain="example.ch", seed=seed(), crawl="crawl",
        manifest_sha="a" * 64, max_l1=0, direct_fallback=True)
    assert result["route"] == "direct-crawl-fallback"
    assert fallback.direct_calls == ["https://www.example.ch/"]


def test_batch_candidate_planning_uses_four_calls_not_four_per_site():
    services = driver.Services.__new__(driver.Services)
    services.candidate_cache, services.candidate_errors = {}, {}
    calls = []

    def exact(crawl, urls, *, complete):
        calls.append(("exact", complete, len(urls)))
        return {}

    def domains(crawl, domains, *, complete):
        calls.append(("domain", complete, len(domains)))
        return {domain: [] for domain in domains}

    services.resolve_exact = exact
    services.resolve_domains = domains
    seeds = {f"example{i}.ch": seed(f"CHE-{i}", f"example{i}.ch")
             for i in range(10)}
    services.prepare_candidates("crawl", seeds, batch_sites=10)
    assert len(calls) == 4
    assert set(services.candidate_cache) == set(seeds)
    assert all(value == (None, "miss") for value in services.candidate_cache.values())


def test_two_cards_on_one_domain_form_one_site_group(tmp_path):
    path = tmp_path / "cards.jsonl"
    rows = [seed("CHE-1"), seed("CHE-2")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    groups = driver.load_inventory(path, 1, 0, 0)
    assert list(groups) == ["example.ch"]
    assert len(groups["example.ch"]) == 2


def test_retryable_interruption_resumes_and_manifest_is_published_last(
        tmp_path, monkeypatch):
    source = tmp_path / "cards.jsonl"
    source.write_text(json.dumps(seed()) + "\n")
    out, audit, manifest = (tmp_path / "out.jsonl", tmp_path / "audit.jsonl",
                            tmp_path / "manifest.json")
    work, l0 = tmp_path / "work", tmp_path / "l0"
    state = {"fail": True}

    class MainServices(FakeServices):
        def __init__(self, **kwargs):
            super().__init__(l0, candidate=(capture(), "exact"),
                             fail_fetch=state["fail"])

        def prepare_candidates(self, crawl, seeds, batch_sites):
            self.candidate_cache = {domain: (capture(), "exact") for domain in seeds}

        def close(self):
            pass

    monkeypatch.setattr(driver, "Services", MainServices)
    monkeypatch.setattr(driver, "normalized_surface",
                        lambda html, url, depth: ("Example Consulting AG content", "fixture"))
    argv = ["snapshot", "--in", str(source), "--out", str(out),
            "--audit-out", str(audit), "--manifest", str(manifest),
            "--work-root", str(work), "--l0-root", str(l0), "--wave", "test",
            "--crawl", "CC-MAIN-2026-30", "--cc-transport", "signed-s3",
            "--cc-index-manifest-sha256", "a" * 64,
            "--source-git-sha", "c" * 40,
            "--runtime-lock-sha256", "b" * 64,
            "--max-l1", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="INCOMPLETE"):
        driver.main(sys.argv[1:], "snapshot")
    assert not manifest.exists() and not out.exists() and not audit.exists()
    identities = list(work.rglob("_run.identity.json"))
    assert identities
    identity = json.loads(identities[0].read_text())
    assert identity["limit"] == 0
    assert identity["l0_root"] == str(l0.resolve())
    assert identity["runtime_lock_sha256"] == "b" * 64
    assert identity["git_sha"] == "c" * 40
    assert identity["cc_transport"] == "signed-s3"
    assert list(work.rglob("retry/*.json"))

    state["fail"] = False
    driver.main(sys.argv[1:], "snapshot")
    assert manifest.exists() and out.exists() and audit.exists()
    payload = json.loads(manifest.read_text())
    assert payload["site_statuses"] == {"ADMITTED": 1}
    emitted = json.loads(out.read_text())
    assert emitted["website_receipt"]["train_ok"] is False
    assert emitted["website_receipt"]["association_receipt"]["status"] == "admitted"


def task275_row(entity, domain, tier, uid=False, reachable="ok"):
    return {
        "point_id": entity, "url": f"https://{domain}/", "title": "Example AG",
        "register_id": entity, "domain": domain, "source_type": "entity_card",
        "provenance": "zefix_ccjoin", "reachable": reachable,
        "last_seen": "2026-07-14",
        "resolve_confidence": 0.9, "resolve_tier": tier,
        "resolve_uid_on_page": uid,
    }


def test_snapshot_cli_help_smoke(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["snapshot", "--help"])
    with pytest.raises(SystemExit) as exc:
        driver.main(sys.argv[1:], "snapshot")
    assert exc.value.code == 0
    assert "Common-Crawl-first" in capsys.readouterr().out


