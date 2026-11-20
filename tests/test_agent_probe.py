"""Task 2684: the agent probe's trace extractor, day selection, config derivation and findings page. No network.

The fixture is a real opencode 2.2.7 `run --format json` stream (Art. 336c CO; recorded with webfetch off, so it holds no fetch), with each
search result list trimmed to three rows. `tasks.json` / `opencode.json` are copies of the operator task set and
probe config (those live outside the shipped tree).
"""
import datetime as dt
import json
from pathlib import Path

import pytest

from cytir.ops import agent_probe as ap

FIXTURE = Path(__file__).parent / "fixtures" / "agent_probe" / "opencode-run-a1.jsonl"
TASKS = Path(__file__).parent / "fixtures" / "agent_probe" / "tasks.json"
CONFIG = Path(__file__).parent / "fixtures" / "agent_probe" / "opencode.json"


def _events():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line]


def _task(tid="a1-co-336c"):
    return next(t for t in ap.load_tasks(TASKS) if t["id"] == tid)


def test_extract_real_trace():
    r = ap.extract(_events(), _task())
    assert r["n_cytir_calls"] == 2 and r["searched"]
    first, second = r["cytir_calls"]
    assert first["tool"] == "search"
    assert first["query"].endswith("site:fedlex.admin.ch")
    assert first["operators"] == {"site": ["fedlex.admin.ch"]}
    assert first["named"] == {"k": 5, "country": "CH"} and first["k"] == 5
    assert first["result_count"] == 3 and len(first["urls"]) == 3
    # host-level gold is rank 1 (Art. 984 on fedlex), the article itself only rank 2: URL-level gold sees it
    assert first["gold_rank"] == 1 and first["gold_url_rank"] == 2
    assert second["gold_url_rank"] == 1
    assert r["answer_source"] == "snippets_only" and r["webfetch_urls"] == []
    assert r["caption_misuse"] == bool(r["cited_urls"]) and r["search_engine_fetches"] == []
    assert r["grounded"] and r["gold_cited"] and r["gold_url_cited"]
    assert r["cited_from_cytir"] == ["https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en#art_336_c"]
    assert r["first_gold_call"] == 1 and r["requery_after_gold"] == 1
    assert r["steps"] == 3 and r["finish_reason"] == "stop"
    assert r["tokens"]["output"] == 593 and r["cost_usd"] == pytest.approx(0.03405, abs=1e-4)
    assert r["answer_text"].startswith("Based on the search results")


def test_extract_no_search_and_webfetch():
    task = _task("c1-http-429")
    ev = [{"type": "text", "timestamp": 1, "part": {"type": "text", "text": "429, defined in RFC 6585."}},
          {"type": "step_finish", "timestamp": 2, "part": {"reason": "stop", "cost": 0.01, "tokens": {"input": 1}}}]
    r = ap.extract(ev, task)
    assert r["answer_source"] == "no_search" and not r["caption_misuse"] and not r["grounded"] and r["cited_urls"] == []
    u = "https://www.rfc-editor.org/rfc/rfc6585.html"
    out = json.dumps({"count": 1, "results": [{"url": u + "#section-4"}]})
    ev = [{"type": "tool_use", "part": {"tool": "cytir-websearch_search",
                                        "state": {"status": "completed", "input": {"query": "429 RFC"}, "output": out}}},
          {"type": "tool_use", "part": {"tool": "webfetch", "state": {"status": "completed", "input": {"url": u}}}},
          {"type": "text", "part": {"text": f"client = Client('http://localhost:6333')\nSee {u}."}}]
    r = ap.extract(ev, task)
    assert r["answer_source"] == "read_returned_url" and r["read_returned_url"] and not r["caption_misuse"]
    assert r["cited_urls"] == [u] and r["grounded"] and r["gold_url_cited"]


def test_parse_query_operators():
    text, ops = ap.parse_query("minimum wage -site:wikipedia.org source:official lang:de country:CH")
    assert text == "minimum wage"
    assert ops == {"-site": ["wikipedia.org"], "source": ["official"], "lang": ["de"], "country": ["CH"]}


def test_select_runs_default_day_is_eight_runs_one_config():
    tasks = ap.load_tasks(TASKS)
    runs = ap.select_runs(tasks, dt.date(2026, 11, 11))
    assert len(runs) == 8 and {c for _, c in runs} == {"webfetch"}
    assert sum(t["set"] == "core" for t, _ in runs) == 6 and sum(t["set"] == "rotating" for t, _ in runs) == 2
    nxt = ap.select_runs(tasks, dt.date(2026, 11, 12))
    assert [t["id"] for t, _ in nxt if t["set"] == "core"] == [t["id"] for t, _ in runs if t["set"] == "core"]
    assert {t["id"] for t, _ in nxt} != {t["id"] for t, _ in runs}
    assert [t["id"] for t, _ in ap.select_runs(tasks, dt.date(2026, 11, 11), ids=["a1"])] == ["a1-co-336c"]


def _check_config(base):
    cfg = ap.derive_config(base)
    assert cfg["websearch"] is False and "webfetch" not in cfg["tools"]
    assert not any(r["resource"] == "webfetch" for r in cfg["permissions"])  # tool.use deny is gone
    assert ap.mcp_server_name(base) == "cytir-websearch"
    bad = dict(base, agents={"build": {"steps": 40}})
    with pytest.raises(ValueError):
        ap.derive_config(bad)
    off = json.loads(json.dumps(base))
    off["tools"]["webfetch"] = False
    with pytest.raises(ValueError):
        ap.derive_config(off)
    lacking = json.loads(json.dumps(base))
    lacking["permissions"] = [r for r in lacking["permissions"] if "bing.com" not in r["resource"]]
    with pytest.raises(ValueError):
        ap.derive_config(lacking)


def test_fixture_config_webfetch_on_search_engines_denied():
    _check_config(json.loads(CONFIG.read_text(encoding="utf-8")))


@pytest.mark.parametrize("url,blocked", [
    ("https://www.bing.com/search?q=x", True), ("https://duckduckgo.com/?q=x", True),
    ("https://html.duckduckgo.com/html/?q=x", True), ("https://www.google.com/search?q=x", True),
    ("https://search.yahoo.com/search?p=x", True), ("https://yahoo.com/search?p=x", True),
    ("https://yandex.ru/search/?text=x", True), ("https://www.yandex.com/search", True),
    ("https://search.brave.com/search?q=x", True), ("https://brave.com/search/?q=x", True),
    ("https://www.baidu.com/s?wd=x", True), ("https://startpage.com/do/search", True),
    ("https://www.ecosia.org/search", True), ("https://kagi.com/search", True),
    ("https://www.perplexity.ai/search", True), ("https://you.com/search", True),
    ("https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de", False), ("https://notbing.com/", False),
    ("https://brave.com/about", False), ("https://finance.yahoo.com/", False), ("", False), (None, False)])
def test_is_search_engine(url, blocked):
    assert ap.is_search_engine(url) is blocked


def _fetch(u):
    return {"type": "tool_use", "part": {"tool": "webfetch", "state": {"status": "completed", "input": {"url": u}}}}


def _search(*urls):
    out = json.dumps({"count": len(urls), "results": [{"url": u} for u in urls]})
    return {"type": "tool_use", "part": {"tool": "cytir-websearch_search",
                                         "state": {"status": "completed", "input": {"query": "q"}, "output": out}}}


def _search_read(url, read_url):
    out = json.dumps({"count": 1, "results": [{"url": url, "read_url": read_url}]})
    return {"type": "tool_use", "part": {"tool": "cytir-websearch_search",
                                         "state": {"status": "completed", "input": {"query": "q"}, "output": out}}}


def test_fetch_of_result_read_url_counts_as_returned():
    task = _task("c1-http-429")
    u = "https://www.rfc-editor.org/rfc/rfc6585.html"
    ru = "https://fedlex.data.admin.ch/filestore/x/rfc6585.html"
    ans = {"type": "text", "part": {"text": f"See {u}."}}
    r = ap.extract([_search_read(u, ru), _fetch(ru), ans], task)
    assert r["answer_source"] == "read_returned_url" and r["read_via_read_url"] is True
    r = ap.extract([_search_read(u, ru), _fetch("https://example.org/other"), ans], task)
    assert r["answer_source"] == "read_other_url" and r["read_via_read_url"] is False
    page = ap.render_findings("2026-11-12", {}, [dict(r, config="webfetch", model="openrouter/x/m1", prompt="p")])
    assert "read via read_url: 0 run(s)" in page


def test_answer_source_values_and_search_engine_fetch():
    task = _task("c1-http-429")
    u = "https://www.rfc-editor.org/rfc/rfc6585.html"
    ans = {"type": "text", "part": {"text": f"See {u}."}}
    r = ap.extract([_search(u), _fetch("https://example.org/other"), ans], task)
    assert r["answer_source"] == "read_other_url" and r["grounded"] and not r["caption_misuse"]
    r = ap.extract([_search(u), ans], task)
    assert r["answer_source"] == "snippets_only" and r["grounded"] and r["caption_misuse"]
    r = ap.extract([_search(u), {"type": "text", "part": {"text": "no url cited"}}], task)
    assert r["answer_source"] == "snippets_only" and not r["caption_misuse"]
    r = ap.extract([_search(u), _fetch("https://www.bing.com/search?q=rfc"), _fetch(u), ans], task)
    assert r["answer_source"] == "read_returned_url" and r["search_engine_fetches"] == ["https://www.bing.com/search?q=rfc"]
    page = ap.render_findings("2026-11-11", {}, [dict(r, config="webfetch", model="openrouter/x/m1", prompt="p")])
    assert "read_returned_url" in page and "search_engine_fetch" in page and "1 run(s)" in page
    assert "answered from captions (misuse)" in page


def test_patterns_report_runs_stopped_by_the_runner():
    base = dict(ap.extract(_events(), _task()), config="webfetch", model="openrouter/x/m1", prompt="p")
    capped = dict(base, killed="Cytir call past the cap 8", answer_text="")
    assert "stopped by the runner (cap/timeout): 0 run(s)" in "\n".join(ap._patterns([base]))
    out = "\n".join(ap._patterns([base, capped]))
    assert "1 run(s): m1:a1-co-336c.webfetch (Cytir call past the cap 8; no final answer)" in out


def test_findings_page_section_order_and_evidence_rule():
    row = dict(ap.extract(_events(), _task()), config="webfetch", model="openrouter/x/m1", prompt=_task()["prompt"])
    notes = {"proposals": [{"text": "return the article passage", "evidence": ["m1:a1-co-336c.webfetch"], "n": 1},
                           {"text": "an uncited wish", "evidence": ["zz.webfetch"]}]}
    page = ap.render_findings("2026-11-11", {"model": "openrouter/x", "opencode_version": "2.2.7"}, [row], notes)
    heads = [line for line in page.splitlines() if line.startswith("## ")]
    assert heads == ["## 1. Per-task table", "## 2. Patterns", "## 3. Feature proposals", "## 4. Index-shape evidence"]
    assert "| m1:a1-co-336c.webfetch | a | n | 2 |" in page
    assert "[n=1] return the article passage — evidence: m1:a1-co-336c.webfetch" in page
    assert "refused: no trace cited (1)" in page and "~~an uncited wish~~" in page
    assert "`search` `Art. 336c CO Swiss Code of Obligations notice period illness site:fedlex.admin.ch`" in page


def test_models_and_reasoning_effort_are_declared():
    models = ap.load_models(TASKS)
    assert [ap.model_slug(m) for m in models] == ["claude-haiku-5.5", "gpt-6.1-sol", "deepseek-v4.1-flash-1017"]
    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    for m in models:
        ap.check_reasoning(base, m)
    bad = json.loads(json.dumps(base))
    bad["provider"]["openrouter"]["models"]["openai/gpt-6.1-sol"] = {"settings": {"reasoningEffort": "medium"}}
    with pytest.raises(ValueError):
        ap.check_reasoning(bad, models[1])


def test_run_loops_models_and_caps_cytir_calls_per_model(tmp_path, monkeypatch):
    calls = []

    def fake_run_one(binary, cfg_path, model, prompt, title, server, max_cytir, timeout_s=0):
        calls.append((model, max_cytir))
        return _events(), {"killed": None, "rc": 0, "wall_s": 1.0}

    monkeypatch.setattr(ap, "run_one", fake_run_one)
    monkeypatch.setattr(ap, "opencode_version", lambda b: "2.2.7")
    out = tmp_path / "run"
    argv = ["run", "--date", "2026-11-11", "--tasks", str(TASKS), "--config", str(CONFIG), "--out", str(out),
            "--ids", "a1,a2", "--max-cytir-calls-per-model", "2"]
    assert ap.main(argv) == 0
    rows = [json.loads(x) for x in (out / "summary.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["model"] for r in rows} == set(ap.load_models(TASKS))
    assert len(rows) == 6
    n = rows[0]["n_cytir_calls"]
    assert n == 2 and all(c[1] <= 2 for c in calls)
    assert all(r.get("skipped") for r in rows if r["task_id"] == "a2-bfs-population")  # cap of 2 spent on a1
    assert (out / "claude-haiku-5.5.a1-co-336c.webfetch.jsonl").exists()
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["reasoning_effort"] == "medium" and meta["max_cytir_calls_per_model"] == 2
    assert ap.CAP_PER_MODEL == 60


def test_requery_after_gold_waits_for_every_declared_gold():
    # Task 2730: a two-gold task (b2) is re-queried only once BOTH golds have been returned.
    task = dict(_task("c1-http-429"), gold_hosts=["fedlex.admin.ch", "gesetze-im-internet.de"], gold_cover="all")
    g1, g2 = "https://www.fedlex.admin.ch/eli/cc/1/", "https://www.gesetze-im-internet.de/burlg/"
    ans = {"type": "text", "part": {"text": "done"}}
    assert ap.extract([_search(g1), ans], task)["requery_after_gold"] == 0
    r = ap.extract([_search(g1), _search(g2), ans], task)
    assert r["gold_in_results"] and r["requery_after_gold"] == 0
    assert ap.extract([_search(g1), _search(g2), _search(g1), ans], task)["requery_after_gold"] == 1
    assert ap.extract([_search(g1), _search(g1), _search(g2), ans], task)["requery_after_gold"] == 0
    assert ap.extract([_search(g1), _search(g2), _search(g1), _search(g2), ans], task)["requery_after_gold"] == 2
    # default (any-of: golds are copies of one source): the first gold call already counts
    any_task = dict(task, gold_cover="any")
    assert ap.extract([_search(g1), _search(g1), ans], any_task)["requery_after_gold"] == 1


def test_capped_run_without_final_answer_is_capped_no_answer():
    task = _task("c1-http-429")
    u = "https://www.rfc-editor.org/rfc/rfc6585.html"
    r = ap.extract([_search(u), _fetch(u)], task, killed="Cytir call past the cap 8")
    assert r["answer_source"] == "capped_no_answer"
    ans = {"type": "text", "part": {"text": f"See {u}."}}
    r = ap.extract([_search(u), _fetch(u), ans], task, killed="timeout 600s")
    assert r["answer_source"] == "read_returned_url"
    r = ap.extract([_search(u), _fetch(u)], task)
    assert r["answer_source"] == "read_returned_url"


def test_capped_no_answer_cell_shows_the_stored_cap_reason():
    base = dict(ap.extract(_events(), _task()), config="webfetch", model="openrouter/x/m1", prompt="p")
    capped = dict(base, answer_source="capped_no_answer", killed="Cytir call past the cap 8")
    timed = dict(base, answer_source="capped_no_answer", killed="timeout 600s")
    done = dict(base, answer_source="read_returned_url", killed=None)
    assert ap._answer_cell(capped) == "capped_no_answer (Cytir call past the cap 8)"
    assert ap._answer_cell(timed) == "capped_no_answer (timeout 600s)"
    assert ap._answer_cell(done) == "read_returned_url"

def test_tool_error_records_http_status_and_detail():
    # task 2759: mcp_server api_error_message text carries the upstream status; legacy bare text does not.
    msg = "search API HTTP 503: Qdrant search over its time budget. Transient: retry in a few seconds."
    ev = [{"type": "tool_use", "part": {"tool": "cytir-websearch_search", "state": {
        "status": "error", "error": msg, "input": {"query": "art. 336 CO"}}}}]
    ev_legacy = [{"type": "tool_use", "part": {"tool": "cytir-websearch_search", "state": {
        "status": "error", "error": "Error executing tool search", "input": {"query": "art. 336 CO"}}}}]
    r = ap.extract(ev, _task())
    c = r["cytir_calls"][0]
    assert c["error_status"] == 503 and c["error_detail"] == msg[:160]
    assert r["errors_by_status"] == {"503": 1}
    legacy = ap.extract(ev_legacy, _task())["cytir_calls"][0]
    assert legacy["error_status"] is None and legacy["error_detail"] is None
    assert legacy["error"] == "Error executing tool search"
    assert ap.extract(ev_legacy, _task())["errors_by_status"] == {"unparsed": 1}


def test_reader_variant_config_and_reader_tool_counts_as_a_read():
    """task 2772: `.reader` swaps webfetch for the stdio reader MCP; the daily `webfetch` config is untouched."""
    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    before = json.dumps(base, sort_keys=True)
    daily = ap.derive_config(base)
    assert "cytir-reader" not in daily["mcp"] and daily["tools"].get("webfetch") is not False
    cfg = ap.derive_config(base, "reader", "/venv/bin/python", "/repo")
    assert cfg["tools"]["webfetch"] is False
    srv = cfg["mcp"]["cytir-reader"]
    assert srv["type"] == "local" and srv["command"] == ["/venv/bin/python", "-m", "cytir.reader.mcp"]
    assert srv["environment"] == {"PYTHONPATH": "/repo"}
    assert ap.mcp_server_name(base) == "cytir-websearch" and json.dumps(base, sort_keys=True) == before
    assert [c for _, c in ap.select_runs(ap.load_tasks(TASKS), dt.date(2026, 11, 12), 0, ["a1"], "reader")] == ["reader"]
    task = _task("c1-http-429")
    u = "https://www.rfc-editor.org/rfc/rfc6585.html"
    rd = lambda url: {"type": "tool_use", "part": {"tool": "cytir-reader_read", "state": {
        "status": "completed", "input": {"url": url}}}}  # noqa: E731
    ans = {"type": "text", "part": {"text": f"See {u}."}}
    r = ap.extract([_search(u), rd(u), ans], task)
    assert r["answer_source"] == "read_returned_url" and r["fetch_tools"] == {"cytir-reader_read": 1}
    assert r["n_cytir_calls"] == 1  # the reader is not a search call
    r = ap.extract([_search(u), rd("https://example.org/x"), ans], task)
    assert r["answer_source"] == "read_other_url"


def test_permission_denied_fetch_is_not_a_read():
    task = _task("c1-http-429")
    u = "https://www.rfc-editor.org/rfc/rfc6585.html"
    denied = {"type": "tool_use", "part": {"tool": "webfetch", "state": {
        "status": "error", "input": {"url": u}, "error": "Permission denied: webfetch"}}}
    r = ap.extract([_search(u), denied, {"type": "text", "part": {"text": f"See {u}."}}], task)
    assert r["answer_source"] == "snippets_only" and r["webfetch_urls"] == []
