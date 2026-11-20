"""Agent-side probe (task 2684, RULING 71): drive Cytir through the opencode agent harness and record how it is used.

The content probe asks "does Cytir return the right URL for this query?". This probe asks "what does an agent
harness DO with Cytir?": does it search at all, which tool, with which query, operators and k, does it read the
URLs it gets, does it answer from memory, is the final answer grounded in a returned URL. Spec:
`specs/agent-probe-opencode.md`.

    python -m cytir.ops.agent_probe run --date 2026-11-11 --tasks data/agent-probe-tasks.v1.json \\
        --config <probe opencode.json> --out probes/agent/runs/2026-11-11/ \\
        [--findings probes/agent/findings/2026-11-11.md] [--ids a1,b1] [--notes N.json]
    python -m cytir.ops.agent_probe report --run-dir probes/agent/runs/2026-11-11/ --findings F.md [--notes N.json]

`run` reads the repo's opencode config (never the user's live one; it is passed as OPENCODE_CONFIG), derives the
webfetch-enabled config (webfetch ON for every run, Tim 2026-11-11; search-engine hosts denied by the config's
`webfetch` rules and re-checked post hoc from the trace), runs each task once headless (`opencode run --format json`, private server, empty temp
workspace, 5-minute timeout, step limit <= 15 from the config), keeps the raw event stream per task, extracts one
summary row per task and writes the findings page. It only queries the public MCP: a task is stopped after
the first search past `--max-cytir-calls-per-task` and the day stops launching tasks at `--max-cytir-calls-total`.
`report` re-extracts from the stored traces (no network) and rewrites the page, e.g. after adding notes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from cytir.core.search_engines import SEARCH_ENGINE_HOSTS, host_matches, host_of, is_search_engine

SCHEMA_TASKS = "aw.agent-probe-tasks.v1"
SCHEMA_SUMMARY = "aw.agent-probe-summary.v1"
CONFIG = "webfetch"  # the one config: webfetch ON for every run (Tim 2026-11-11)
READER = "reader"  # task 2772: the same tasks, the cytir-reader MCP instead of the built-in webfetch (opt-in, not in the cron)
CONFIGS = (CONFIG, READER)
READER_SERVER = "cytir-reader"
READER_TOOL = READER_SERVER + "_read"  # opencode names an MCP tool <server>_<tool>
# The search-engine list lives in cytir.core.search_engines (one list; the probe's deny rules under deploy/ mirror it).

MAX_STEPS = 15
REASONING_EFFORT = "medium"  # Tim 2026-11-11
CAP_PER_MODEL = 60
TIMEOUT_S = 300
DEFAULT_BIN = "~/.opencode/bin/opencode"
_OPERATOR = re.compile(r"(?<!\S)(-site|site|source|lang|country):(\S+)")
_URL = re.compile(r"https?://[^\s<>\"'`)\]]+")
_WORD = re.compile(r"[^\W_]{3,}", re.U)


# ---------------------------------------------------------------- task set + config

def load_models(path: str | Path) -> list[str]:
    """The probe's model list: the tasks file's `models` (one place; the cron script names none)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    models = doc.get("models")
    if not isinstance(models, list) or not models or not all(isinstance(m, str) and "/" in m for m in models):
        raise ValueError(f"{path}: `models` must be a non-empty list of provider/model ids")
    return models


def model_slug(model: str) -> str:
    """`openrouter/anthropic/claude-haiku-5.5` -> `claude-haiku-5.5` (file names and run keys)."""
    return model.rsplit("/", 1)[-1]


def check_reasoning(base: dict, model: str, effort: str = REASONING_EFFORT) -> None:
    """Fail if the config does not pass reasoning.effort=<effort> for this model. opencode 2.2.7 forwards
    provider.<p>.models.<m>.options to the AI SDK request body (proved against a capture server); `settings`/`body` are NOT sent."""
    prov, _, mid = model.partition("/")
    got = (base.get("provider", {}).get(prov, {}).get("models", {}).get(mid, {}).get("options", {})
           .get("reasoning", {}).get("effort"))
    if got != effort:
        raise ValueError(f"config lacks provider.{prov}.models.{mid}.options.reasoning.effort={effort!r} (has {got!r})")


def load_tasks(path: str | Path) -> list[dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA_TASKS:
        raise ValueError(f"{path}: schema is {doc.get('schema')!r}, want {SCHEMA_TASKS}")
    tasks, seen = doc["tasks"], set()
    for t in tasks:
        for key in ("id", "family", "set", "prompt", "gold_hosts", "knows_already"):
            if key not in t:
                raise ValueError(f"{path}: task {t.get('id')!r} lacks {key!r}")
        if t["family"] not in ("a", "b", "c") or t["set"] not in ("core", "rotating") or t["id"] in seen:
            raise ValueError(f"{path}: task {t['id']!r}: bad family/set or duplicate id")
        seen.add(t["id"])
    return tasks


def select_runs(tasks: list[dict], day: dt.date, n_rotating: int = 2,
                ids: list[str] | None = None, config: str = CONFIG) -> list[tuple[dict, str]]:
    """The day's (task, config) runs: every core task plus `n_rotating` rotating tasks chosen by day ordinal, all
    with webfetch on (one config). `ids` replaces the selection (prefix match on the id, e.g. `a1`)."""
    ordinal = day.toordinal()
    if ids is not None:
        chosen = []
        for p in ids:
            hit = [t for t in tasks if t["id"] == p or t["id"].startswith(p + "-")]
            if len(hit) != 1:
                raise ValueError(f"task prefix {p!r} matches {len(hit)} tasks")
            chosen.append(hit[0])
    else:
        core = [t for t in tasks if t["set"] == "core"]
        rot = [t for t in tasks if t["set"] == "rotating"]
        start = (ordinal * n_rotating) % len(rot) if rot else 0
        chosen = core + [rot[(start + i) % len(rot)] for i in range(min(n_rotating, len(rot)))]
    return [(t, config) for t in chosen]


def derive_config(base: dict, config: str = CONFIG, reader_python: str | None = None,
                  reader_cwd: str | None = None) -> dict:
    """Validate the repo config (steps, no built-in websearch, webfetch not switched off, search engines denied)."""
    if config not in CONFIGS:
        raise ValueError(config)
    cfg = json.loads(json.dumps(base))
    steps = cfg.get("agents", {}).get("build", {}).get("steps")
    if not isinstance(steps, int) or not 0 < steps <= MAX_STEPS:
        raise ValueError(f"agents.build.steps must be 1..{MAX_STEPS}, is {steps!r}")
    if cfg.get("websearch") is not False:
        raise ValueError("the built-in websearch must be disabled (websearch: false)")
    if cfg.get("tools", {}).get("webfetch") is False or any(
            r.get("resource") == "webfetch" and r.get("effect") == "deny" for r in cfg.get("permissions", [])):
        raise ValueError("webfetch must be ON in the probe config (Tim 2026-11-11)")
    denied = {r["resource"] for r in cfg.get("permissions", [])
              if r.get("action") == "webfetch" and r.get("effect") == "deny"}
    for h in SEARCH_ENGINE_HOSTS:
        if f"http*://{h}/*" not in denied:
            raise ValueError(f"config lacks a webfetch deny rule for search engine {h}")
    if config == READER:
        # Built-in webfetch OFF; the stdio reader MCP is the only page reader. It refuses search-engine URLs itself.
        cfg["tools"]["webfetch"] = False
        cfg["mcp"][READER_SERVER] = {
            "type": "local", "enabled": True,
            "command": [reader_python or sys.executable, "-m", "cytir.reader.mcp"],
            "environment": {"PYTHONPATH": reader_cwd or str(Path.cwd())}}
    return cfg


def mcp_server_name(cfg: dict) -> str:
    servers = cfg.get("mcp", {})
    servers = servers.get("servers", servers)
    names = [k for k in servers if k != "timeout"]
    if len(names) != 1:
        raise ValueError(f"want exactly one MCP server in the probe config, have {names}")
    return names[0]


# ---------------------------------------------------------------- extraction (pure)

def url_matches(url: str, task: dict) -> bool | None:
    """URL-level gold: one of the task's `gold_url_contains` substrings (case-insensitive) on a gold host.
    None when the task declares none (host-level gold only)."""
    subs = task.get("gold_url_contains")
    if not subs:
        return None
    return host_matches(url, task["gold_hosts"]) and any(x.lower() in url.lower() for x in subs)


def _defrag(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/")


def parse_query(query: str) -> tuple[str, dict[str, list[str]]]:
    ops: dict[str, list[str]] = {}
    for name, val in _OPERATOR.findall(query or ""):
        ops.setdefault(name, []).append(val)
    return " ".join(_OPERATOR.sub(" ", query or "").split()), ops


def _words(s: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(s or "")}


_HTTP_STATUS = re.compile(r"HTTP (\d{3})")


def _error_fields(err) -> dict:
    """task 2759: a tool error that carries an upstream HTTP status (mcp_server api_error_message:
    `search API HTTP 503: ...`) keeps the status and the first 160 chars. Bare legacy text: no status."""
    m = _HTTP_STATUS.search(str(err or ""))
    if not m:
        return {"error_status": None, "error_detail": None}
    return {"error_status": int(m.group(1)), "error_detail": str(err)[:160]}


def _errors_by_status(calls) -> dict:
    """Error count per upstream HTTP status; errors with no status parsed are counted as `unparsed`."""
    out: dict[str, int] = {}
    for c in calls:
        if c.get("error"):
            k = str(c.get("error_status") or "unparsed")
            out[k] = out.get(k, 0) + 1
    return out


def _result_urls(output) -> tuple[int | None, list[str]]:
    try:
        doc = json.loads(output) if isinstance(output, str) else output
        res = doc.get("results", [])
        return doc.get("count", len(res)), [r["url"] for r in res if r.get("url")]
    except (ValueError, AttributeError, TypeError, KeyError):
        urls = _URL.findall(output or "") if isinstance(output, str) else []
        return None, urls


def _read_urls(output) -> list[str]:
    """The no-JavaScript `read_url` copies a Cytir search result may carry beside `url`."""
    try:
        doc = json.loads(output) if isinstance(output, str) else output
        return [r["read_url"] for r in doc.get("results", []) if isinstance(r, dict) and r.get("read_url")]
    except (ValueError, AttributeError, TypeError):
        return []


_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "example.com", "example.org")


def _is_local(url: str) -> bool:
    """A URL inside a code sample (a client endpoint), not a cited source."""
    return host_of(url) in _LOCAL_HOSTS


def _clean_url(u: str) -> str:
    return u.rstrip(".,;:!?*")


def _gold_cover_index(cytir: list[dict], task: dict) -> int | None:
    """The call index after which a further query is a re-query. Default (`gold_cover` absent / "any"): the first call
    returning ANY gold host (golds are alternative copies of one source). `"gold_cover": "all"` (b2: two jurisdictions):
    the call at which EVERY declared gold host has appeared (None if one never does)."""
    hosts = task["gold_hosts"]
    if task.get("gold_cover", "any") != "all":
        return next((n for n, c in enumerate(cytir) if c["gold_rank"]), None)
    first: dict[str, int] = {}
    for n, c in enumerate(cytir):
        for h in hosts:
            if h not in first and any(host_matches(u, [h]) for u in c["urls"]):
                first[h] = n
    return max(first.values()) if hosts and len(first) == len(set(hosts)) else None


def extract(events: list[dict], task: dict, server: str = "cytir-websearch", killed: str | None = None) -> dict:
    """One summary row from one opencode `--format json` event stream."""
    prefix = server + "_"
    tool_calls, cytir, webfetch_urls, read_urls, tokens = [], [], [], [], dict(input=0, output=0, reasoning=0, cache_read=0,
                                                                   cache_write=0)
    cost, steps, finish, last_tool_idx, texts, session = 0.0, 0, None, -1, [], None
    stamps = [e["timestamp"] for e in events if isinstance(e.get("timestamp"), (int, float))]
    for i, e in enumerate(events):
        part = e.get("part", {}) or {}
        session = session or e.get("sessionID")
        if e.get("type") == "tool_use":
            name, st = part.get("tool", ""), part.get("state", {}) or {}
            tool_calls.append({"tool": name, "status": st.get("status")})
            last_tool_idx = i
            inp = st.get("input", {}) or {}
            if name.startswith(prefix):
                q = inp.get("query", "")
                text, ops = parse_query(q)
                count, urls = _result_urls(st.get("output"))
                read_urls.extend(_read_urls(st.get("output")))
                t0, t1 = (st.get("time") or {}).get("start"), (st.get("time") or {}).get("end")
                gold_rank = next((r + 1 for r, u in enumerate(urls) if host_matches(u, task["gold_hosts"])), None)
                url_rank = next((r + 1 for r, u in enumerate(urls) if url_matches(u, task)), None)
                err = st.get("error") if st.get("status") == "error" else None
                cytir.append({
                    "tool": name[len(prefix):], "query": q, "query_text": text, "operators": ops,
                    "named": {k: v for k, v in inp.items() if k != "query" and v not in (None, [], "")},
                    "k": inp.get("k"), "status": st.get("status"), "result_count": count, "urls": urls,
                    "gold_rank": gold_rank, "gold_url_rank": url_rank, "duration_ms": (t1 - t0) if t0 and t1 else None,
                    "error": err, **_error_fields(err)})
            elif name == "webfetch" or name == READER_TOOL:
                if st.get("status") == "error" and "Permission denied" in str(st.get("error")):
                    continue  # the config refused the call (reader variant: webfetch is off): no page was read
                webfetch_urls.append(inp.get("url"))  # a read by either tool (the config enables exactly one)
        elif e.get("type") == "text":
            texts.append((i, part.get("text", "")))
        elif e.get("type") == "step_finish":
            steps += 1
            finish = part.get("reason")
            cost += float(part.get("cost") or 0)
            tk = part.get("tokens", {}) or {}
            for k in ("input", "output", "reasoning"):
                tokens[k] += int(tk.get(k) or 0)
            tokens["cache_read"] += int((tk.get("cache") or {}).get("read") or 0)
            tokens["cache_write"] += int((tk.get("cache") or {}).get("write") or 0)
        elif e.get("type") == "error":
            tool_calls.append({"tool": "<error>", "status": str(e.get("error"))[:200]})
    answer = "\n".join(t for i, t in texts if i > last_tool_idx).strip()
    cited = list(dict.fromkeys(u for u in (_clean_url(x) for x in _URL.findall(answer)) if not _is_local(u)))
    returned = [u for c in cytir for u in c["urls"]]
    read_keys = {_defrag(u) for u in read_urls}
    returned_keys = {_defrag(u) for u in returned} | read_keys
    first_gold = next((n for n, c in enumerate(cytir) if c["gold_rank"]), None)
    cover = _gold_cover_index(cytir, task)
    counts: dict[str, int] = {}
    for c in tool_calls:
        counts[c["tool"]] = counts.get(c["tool"], 0) + 1
    fetch_tools = {t: counts[t] for t in ("webfetch", READER_TOOL) if t in counts}
    read_returned = any(u and _defrag(u) in returned_keys for u in webfetch_urls)
    read_via_read_url = any(u and _defrag(u) in read_keys for u in webfetch_urls)
    se_fetches = [u for u in webfetch_urls if is_search_engine(u)]
    if killed and not answer:
        answer_source = "capped_no_answer"  # runner killed the run at its cap/timeout before any final answer
    elif read_returned:
        answer_source = "read_returned_url"
    elif webfetch_urls:
        answer_source = "read_other_url"
    elif cytir:
        answer_source = "snippets_only"
    else:
        answer_source = "no_search"
    overlap = None
    if cytir:
        a, b = _words(cytir[0]["query_text"]), _words(task["prompt"])
        overlap = round(len(a & b) / len(a | b), 3) if a | b else None
    return {
        "schema": SCHEMA_SUMMARY, "task_id": task["id"], "family": task["family"], "set": task["set"],
        "knows_already": task["knows_already"], "gold_hosts": task["gold_hosts"], "session_id": session,
        "steps": steps, "finish_reason": finish,
        "duration_s": round((max(stamps) - min(stamps)) / 1000, 1) if stamps else None,
        "tokens": tokens, "cost_usd": round(cost, 5), "tool_counts": counts,
        "cytir_calls": cytir, "n_cytir_calls": len(cytir), "searched": bool(cytir),
        "errors_by_status": _errors_by_status(cytir),
        "webfetch_urls": webfetch_urls, "fetch_tools": fetch_tools, "read_returned_url": read_returned,
        "read_via_read_url": read_via_read_url, "answer_source": answer_source,
        "search_engine_fetches": se_fetches,
        "answer_text": answer[:6000], "cited_urls": cited,
        "cited_from_cytir": [u for u in cited if _defrag(u) in returned_keys],
        "grounded": any(_defrag(u) in returned_keys for u in cited),
        "caption_misuse": answer_source == "snippets_only" and bool(cited),
        "gold_in_results": first_gold is not None,
        "gold_cited": any(host_matches(u, task["gold_hosts"]) for u in cited),
        "gold_url_in_results": (None if not task.get("gold_url_contains")
                                else any(c["gold_url_rank"] for c in cytir)),
        "gold_url_cited": None if not task.get("gold_url_contains") else any(url_matches(u, task) for u in cited),
        "first_gold_call": None if first_gold is None else first_gold + 1,
        "requery_after_gold": 0 if cover is None else len(cytir) - cover - 1,
        "first_query_prompt_overlap": overlap,
    }


# ---------------------------------------------------------------- findings page (pure)

def _run_key(row: dict) -> str:
    slug = model_slug(row["model"]) if row.get("model") else ""
    return f"{slug}:{row['task_id']}.{row['config']}" if slug else f"{row['task_id']}.{row['config']}"


def _ops_str(c: dict) -> str:
    bits = [f"{k}:{','.join(v)}" for k, v in sorted(c["operators"].items())]
    bits += [f"{k}={v}" for k, v in sorted(c["named"].items()) if k != "k"]
    return " ".join(bits) or "-"


def _md(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def _answer_cell(row: dict) -> str:
    """Answer-source cell; a capped run shows the cap reason the runner stored in `killed`, verbatim."""
    if row["answer_source"] == "capped_no_answer" and row.get("killed"):
        return f"capped_no_answer ({row['killed']})"
    return row["answer_source"]


def render_findings(date: str, meta: dict, rows: list[dict], notes: dict | None = None) -> str:
    """The daily page. Section order is FIXED (spec): 1 per-task table, 2 patterns, 3 feature proposals,
    4 index-shape evidence. A proposal or index note is listed only when it cites >= 1 run key of this page
    (`<model>:<task_id>.<config>`); one that does not is shown under 'refused: no trace cited'."""
    notes = notes or {}
    keys = {_run_key(r) for r in rows}
    ran = [r for r in rows if not r.get("skipped")]
    out = [f"# Agent probe — {date}", "",
           f"Harness: opencode {meta.get('opencode_version', '?')} · models {', '.join('`' + m + '`' for m in meta.get('models') or [meta.get('model', '?')])} · "
           f"reasoning effort {meta.get('reasoning_effort', '?')} · "
           f"MCP `{meta.get('mcp_server', '?')}` · config sha256 {meta.get('config_sha256', {})} · "
           f"runs {len(ran)}/{len(rows)} · Cytir calls {sum(r.get('n_cytir_calls', 0) for r in ran)} · "
           f"cost ${sum(r.get('cost_usd', 0) for r in ran):.3f}", ""]
    if meta.get("sample_note"):
        out += [f"> {meta['sample_note']}", ""]
    out += ["## 1. Per-task table", "",
            "| run (model:task.config) | fam | knows | Cytir calls | other tools | k | operators (1st call) | answer source | gold in results "
            "(call/rank) | gold URL rank per call | cited Cytir URL | gold cited host/URL | steps | s | tokens in/out "
            "| $ |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("skipped"):
            out.append(f"| {_run_key(r)} | {r['family']} | | skipped: {r['skipped']} |" + " |" * 12)
            continue
        cc = r["cytir_calls"]
        other = ", ".join(f"{t}×{n}" for t, n in sorted(r["tool_counts"].items())
                          if not t.startswith(r.get("mcp_server", "cytir-websearch"))) or "-"
        gold = (f"{r['first_gold_call']}/{cc[r['first_gold_call'] - 1]['gold_rank']}"
                if r["first_gold_call"] else "no")
        tk = r["tokens"]
        url_ranks = "-" if r.get("gold_url_in_results") is None or not cc else ",".join(
            str(c.get("gold_url_rank") or "x") for c in cc)
        ucited = {None: "", True: "/yes", False: "/no"}[r.get("gold_url_cited")]
        out.append(
            f"| {_run_key(r)} | {r['family']} | {'y' if r['knows_already'] else 'n'} | {r['n_cytir_calls']} | "
            f"{other} | {','.join(str(c['k']) for c in cc) or '-'} | {_md(_ops_str(cc[0])) if cc else '-'} | "
            f"{_answer_cell(r)}{' ⚠SE:' + str(len(r['search_engine_fetches'])) if r['search_engine_fetches'] else ''} | {gold} | {url_ranks} | {'yes' if r['grounded'] else 'no'} | "
            f"{'yes' if r['gold_cited'] else 'no'}{ucited} "
            f"| {r['steps']} | {r['duration_s']} | {tk['input'] + tk['cache_read'] + tk['cache_write']}/"
            f"{tk['output']} | {r['cost_usd']:.3f} |")
    out += ["", "### Queries, exact, in call order", ""]
    for r in ran:
        out.append(f"- **{_run_key(r)}** — prompt: “{_md(r.get('prompt', ''))}”")
        for n, c in enumerate(r["cytir_calls"], 1):
            out.append(f"  {n}. `{c['tool']}` `{_md(c['query'])}` named={json.dumps(c['named'], ensure_ascii=False)}"
                       f" → {c['result_count']} results, gold host rank {c['gold_rank'] or '-'}, gold URL rank "
                       f"{c.get('gold_url_rank') or '-'}"
                       f"{', ERROR ' + _md(str(c['error'])) if c['error'] else ''}")
        if not r["cytir_calls"]:
            out.append("  (no Cytir call)")
        if r.get("killed"):
            out.append(f"  stopped by the runner: {r['killed']}")
    out += ["", "## 2. Patterns", ""]
    out += _patterns(ran)
    out += ["", "## 3. Feature proposals", ""]
    out += _notes_block(notes.get("proposals", []), keys, "proposal")
    out += ["", "## 4. Index-shape evidence", ""]
    out += _index_evidence(ran)
    out += _notes_block(notes.get("index_notes", []), keys, "index note")
    if notes.get("observations"):
        out += ["", "## Observations (free text, cite run keys)", ""]
        out += _notes_block(notes["observations"], keys, "observation")
    return "\n".join(out).rstrip() + "\n"


def _pct(n: int, d: int) -> str:
    return f"{n}/{d}" + (f" ({100 * n / d:.0f} %)" if d else "")


def _patterns(ran: list[dict]) -> list[str]:
    if not ran:
        return ["(no run)"]
    calls = [c for r in ran for c in r["cytir_calls"]]
    lines = [f"- searched at all: {_pct(sum(r['searched'] for r in ran), len(ran))}; Cytir calls per run: "
             f"{', '.join(str(r['n_cytir_calls']) for r in ran)}",
             f"- tools used: {sorted({c['tool'] for c in calls}) or '-'}",
             f"- calls with an operator in the query: {_pct(sum(bool(c['operators']) for c in calls), len(calls))};"
             f" operators seen: {sorted({k for c in calls for k in c['operators']}) or '-'}; named params seen: "
             f"{sorted({k for c in calls for k in c['named']}) or '-'}",
             f"- k requested: {sorted({str(c['k']) for c in calls}) or '-'}",
             f"- first-query word overlap with the prompt (Jaccard): "
             f"{[r['first_query_prompt_overlap'] for r in ran if r['searched']]}",
             f"- gold host in some result list: {_pct(sum(r['gold_in_results'] for r in ran), len(ran))}; answer "
             f"cites a Cytir-returned URL: {_pct(sum(r['grounded'] for r in ran), len(ran))}; answer cites a gold "
             f"host: {_pct(sum(r['gold_cited'] for r in ran), len(ran))}",
             f"- re-queried after the gold host was already returned (search used as a reader): "
             f"{_pct(sum(r['requery_after_gold'] > 0 for r in ran), sum(r['gold_in_results'] for r in ran))}"]
    fam_c = [r for r in ran if r["knows_already"]]
    if fam_c:
        lines.append(f"- control (knows_already) runs that searched anyway: {_pct(sum(r['searched'] for r in fam_c), len(fam_c))}")
    src = {k: sum(r["answer_source"] == k for r in ran)
           for k in ("read_returned_url", "read_other_url", "snippets_only", "no_search", "capped_no_answer")}
    lines.append("- answer source (snippets are for choosing the URL; reading = fetching it): "
                 + ", ".join(f"{k} {_pct(n, len(ran))}" for k, n in src.items()))
    lines.append(f"- answered from captions (misuse), i.e. snippets_only with a cited URL: "
                 f"{_pct(sum(r['caption_misuse'] for r in ran), len(ran))}")
    stopped = [r for r in ran if r.get("killed")]
    lines.append(f"- stopped by the runner (cap/timeout): {len(stopped)} run(s)"
                 + (": " + ", ".join(f"{_run_key(r)} ({r['killed']}; "
                                     f"{'answered' if r.get('answer_text') else 'no final answer'})" for r in stopped)
                    if stopped else ""))
    rr = [_run_key(r) for r in ran if r.get("read_via_read_url")]
    lines.append(f"- read via read_url: {len(rr)} run(s)" + (": " + ", ".join(rr) if rr else ""))
    se = [(_run_key(r), len(r["search_engine_fetches"])) for r in ran if r["search_engine_fetches"]]
    lines.append(f"- search_engine_fetch (webfetch of a blocked search-engine host): {len(se)} run(s)"
                 + (": " + ", ".join(f"{k} ×{n}" for k, n in se) if se else ""))
    return lines


def _index_evidence(ran: list[dict]) -> list[str]:
    grounding_hosts: dict[str, list[str]] = {}
    for r in ran:
        for u in r["cited_from_cytir"]:
            grounding_hosts.setdefault(host_of(u), []).append(_run_key(r))
    lines = ["- hosts whose Cytir-returned URL the final answer cited: "
             + (", ".join(f"{h} ({', '.join(sorted(set(v)))})" for h, v in sorted(grounding_hosts.items())) or "none")]
    miss = [_run_key(r) for r in ran if r["searched"] and not r["gold_in_results"] and r["gold_hosts"]]
    lines.append(f"- searched but no gold host in any result list (coverage or ranking gap): {miss or 'none'}")
    nos = [_run_key(r) for r in ran if not r["searched"]]
    lines.append(f"- answered with no Cytir call (memory sufficed or harness skipped the tool): {nos or 'none'}")
    ka = [_run_key(r) for r in ran if r["knows_already"] and r["searched"]]
    lines.append(f"- knows_already tasks that still searched (index must cover these too, or the harness wastes a "
                 f"call): {ka or 'none'}")
    return lines


def _notes_block(items: list[dict], keys: set[str], what: str) -> list[str]:
    if not items:
        return [f"(no {what} this run)"]
    ok, bad = [], []
    for it in items:
        ev = [e for e in it.get("evidence", []) if e in keys]
        (ok if ev else bad).append((it, ev))
    lines = [f"- {'[n=' + str(it['n']) + '] ' if 'n' in it else ''}{it['text']} — evidence: "
             f"{', '.join(ev)}" for it, ev in ok]
    if bad:
        lines += ["", f"refused: no trace cited ({len(bad)}):"] + [f"- ~~{it['text']}~~" for it, _ in bad]
    return lines


# ---------------------------------------------------------------- running (network: opencode + public MCP)

def opencode_version(binary: str) -> str:
    out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    return out.split()[-1] if out else "?"


def run_one(binary: str, cfg_path: Path, model: str, prompt: str, title: str, server: str,
            max_cytir: int, timeout_s: int = TIMEOUT_S) -> tuple[list[dict], dict]:
    """One headless opencode run in an empty temp workspace. Returns (events, run info)."""
    env = dict(os.environ, OPENCODE_CONFIG=str(cfg_path.resolve()), OPENCODE_DISABLE_AUTOUPDATE="1",
               OPENCODE_DISABLE_CLAUDE_CODE="1", OPENCODE_DISABLE_PROJECT_CONFIG="1")
    ws = tempfile.mkdtemp(prefix="aw-agent-probe-")
    cmd = [binary, "run", "--standalone", "--format", "json", "--archived", "--model", model, "--title", title, prompt]
    events, info, n_cytir = [], {"killed": None, "rc": None}, 0
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ws, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def _timeout():
        info["killed"] = f"timeout {timeout_s}s"
        proc.kill()
    timer = threading.Timer(timeout_s, _timeout)
    timer.start()
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                ev = {"type": "unparsed", "raw": line[:2000]}
            events.append(ev)
            if ev.get("type") == "tool_use" and str(ev.get("part", {}).get("tool", "")).startswith(server + "_"):
                n_cytir += 1
                if n_cytir > max_cytir and not info["killed"]:
                    info["killed"] = f"Cytir call past the cap {max_cytir}"
                    proc.kill()
        info["rc"] = proc.wait()
        info["stderr_tail"] = (proc.stderr.read() or "")[-1500:]
    finally:
        timer.cancel()
        shutil.rmtree(ws, ignore_errors=True)
    info["wall_s"] = round(time.time() - t0, 1)
    return events, info


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def cmd_run(a: argparse.Namespace) -> int:
    tasks = load_tasks(a.tasks)
    base = json.loads(Path(a.config).read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfgs = {a.config_variant: derive_config(base, a.config_variant, a.reader_python, str(Path.cwd()))}
    server = mcp_server_name(base)
    for c, cfg in cfgs.items():
        (out / f"config.{c}.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    split = lambda s: None if s is None else [x for x in s.split(",") if x]  # noqa: E731
    runs = select_runs(tasks, dt.date.fromisoformat(a.date), a.rotating, split(a.ids), a.config_variant)
    models = [x for x in a.models.split(",") if x] if a.models else load_models(a.tasks)
    for m in models:
        check_reasoning(base, m)
    if len(runs) > a.max_runs:
        print(f"refused: {len(runs)} runs > --max-runs {a.max_runs} (per model)", file=sys.stderr)
        return 2
    binary = os.path.expanduser(a.opencode_bin)
    meta = {"date": a.date, "opencode_version": opencode_version(binary), "models": models,
            "reasoning_effort": REASONING_EFFORT, "mcp_server": server,
            "config_sha256": {c: _sha(v) for c, v in cfgs.items()}, "tasks_file": str(a.tasks),
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "max_cytir_calls_per_task": a.max_cytir_calls_per_task, "max_cytir_calls_per_model": a.max_cytir_calls_per_model,
            "sample_note": a.sample_note}
    rows, totals = [], {}
    with (out / "summary.jsonl").open("w", encoding="utf-8") as fh:
        for model in models:
            slug = model_slug(model)
            totals[model] = 0
            for task, config in runs:
                base_row = {"task_id": task["id"], "family": task["family"], "config": config, "model": model}
                if totals[model] >= a.max_cytir_calls_per_model:
                    row = dict(base_row, skipped=f"{slug} Cytir cap {a.max_cytir_calls_per_model} reached")
                else:
                    left = min(a.max_cytir_calls_per_task, a.max_cytir_calls_per_model - totals[model])
                    events, info = run_one(binary, out / f"config.{config}.json", model, task["prompt"],
                                           f"aw-agent-probe {a.date} {slug} {task['id']} {config}", server, left)
                    with (out / f"{slug}.{task['id']}.{config}.jsonl").open("w", encoding="utf-8") as tf:
                        for ev in events:
                            tf.write(json.dumps(ev, ensure_ascii=False) + "\n")
                    row = dict(extract(events, task, server, killed=info["killed"]), config=config, model=model, prompt=task["prompt"],
                               mcp_server=server, rc=info["rc"], killed=info["killed"], wall_s=info["wall_s"],
                               opencode_version=meta["opencode_version"])
                    if info["rc"] not in (0, None) or info["killed"]:
                        row["stderr_tail"] = info.get("stderr_tail", "")
                    totals[model] += row["n_cytir_calls"]
                    print(f"{slug}:{task['id']}.{config}: cytir={row['n_cytir_calls']} source={row['answer_source']} se_fetch={len(row['search_engine_fetches'])} "
                          f"grounded={row['grounded']} ${row['cost_usd']:.3f} {row['duration_s']}s rc={info['rc']}",
                          flush=True)
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
    meta["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    meta["cytir_calls_per_model"] = totals
    meta["cytir_calls_total"] = sum(totals.values())
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if a.findings:
        _write_findings(out, Path(a.findings), a.notes)
    return 0


def _write_findings(run_dir: Path, findings: Path, notes_path: str | None) -> None:
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_dir / "summary.jsonl").read_text(encoding="utf-8").splitlines() if line]
    notes = json.loads(Path(notes_path).read_text(encoding="utf-8")) if notes_path else None
    findings.parent.mkdir(parents=True, exist_ok=True)
    findings.write_text(render_findings(meta["date"], meta, rows, notes), encoding="utf-8")
    print(f"wrote {findings}")


def cmd_report(a: argparse.Namespace) -> int:
    """Re-extract every stored trace (no network) and rewrite summary.jsonl + the findings page."""
    run_dir = Path(a.run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    tasks = {t["id"]: t for t in load_tasks(a.tasks or meta["tasks_file"])}
    old = [json.loads(line) for line in (run_dir / "summary.jsonl").read_text(encoding="utf-8").splitlines() if line]
    rows = []
    for o in old:
        trace = run_dir / f"{model_slug(o['model'])}.{o['task_id']}.{o['config']}.jsonl"
        if not trace.exists():  # day-1 layout: no model prefix
            trace = run_dir / f"{o['task_id']}.{o['config']}.jsonl"
        if o.get("skipped") or not trace.exists():
            rows.append(o)
            continue
        events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
        keep = {k: o[k] for k in ("config", "model", "prompt", "mcp_server", "rc", "killed", "wall_s",
                                  "opencode_version", "stderr_tail") if k in o}
        rows.append(dict(extract(events, tasks[o["task_id"]], meta["mcp_server"], killed=o.get("killed")), **keep))
    (run_dir / "summary.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                           encoding="utf-8")
    _write_findings(run_dir, Path(a.findings), a.notes)
    return 0


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the day's tasks through opencode (queries the public MCP)")
    r.add_argument("--date", default=dt.datetime.now(dt.timezone.utc).date().isoformat())
    r.add_argument("--tasks", required=True)
    r.add_argument("--config", required=True, help="the repo's opencode config (webfetch-on base)")
    r.add_argument("--out", required=True)
    r.add_argument("--findings", default=None)
    r.add_argument("--notes", default=None, help="JSON {proposals, index_notes, observations}: [{text, evidence}]")
    r.add_argument("--ids", default=None, help="comma list of task ids/prefixes (replaces the day set)")
    r.add_argument("--rotating", type=int, default=2)
    r.add_argument("--max-runs", type=int, default=8)
    r.add_argument("--max-cytir-calls-per-task", type=int, default=8)
    r.add_argument("--max-cytir-calls-per-model", type=int, default=CAP_PER_MODEL)
    r.add_argument("--models", default=None, help="comma list of provider/model ids; default: `models` in --tasks")
    r.add_argument("--config-variant", choices=CONFIGS, default=CONFIG,
                   help="`reader` (task 2772): cytir-reader MCP instead of webfetch; default `webfetch` = the daily config")
    r.add_argument("--reader-python", default=None, help="python with the `mcp` SDK for the reader (default: this one)")
    r.add_argument("--opencode-bin", default=DEFAULT_BIN)
    r.add_argument("--sample-note", default=None)
    p = sub.add_parser("report", help="re-extract stored traces and rewrite the findings page (no network)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--findings", required=True)
    p.add_argument("--tasks", default=None)
    p.add_argument("--notes", default=None)
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    raise SystemExit(main())
