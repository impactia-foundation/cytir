#!/usr/bin/env python3
"""Task 120 — Hugging Face model-card ingest (AI-depth #2, route R2).

Fills the 0-coverage "X model card" grounding hole: prod has HF /docs+/learn
pages but ZERO model cards, so every "license of X / params of X / what is X"
query returns a docs page or Wikipedia. Route = the official HF Hub API
(policy-(a), robots Allow: /), NOT scraping.

The PubMed lesson is BINDING (long-tail flooding once ruined 12% of queries), so
scope is CAPPED to the models agents actually ask about:
  * top ~3,000 by downloads
  * top ~500 by likes
  * every model from the major orgs (meta-llama, mistralai, google, openai,
    deepseek-ai, Qwen, microsoft, anthropic, stabilityai, black-forest-labs,
    sentence-transformers)
dedupe -> ~4-6k models -> ~15-40k passages.

Per model we emit:
  * ONE structured summary passage (params / license / task / languages / dates /
    downloads) — the record that answers "what is X / license of X".
  * the card body (README, YAML-frontmatter stripped) chunked — only if the card
    is non-empty (>=50 words), NOT gated, and passes the junk screen.

Junk screen: skip auto-generated empty cards (<50 words of body -> body dropped,
structured kept); skip quantized-duplicate re-quant spam ENTIRELY (TheBloke-style
GGUF/AWQ/GPTQ re-quants) unless the model is in the top-500-by-downloads or from a
major org. Gated models: metadata-only (structured passage), no body.

Canonical URL = https://huggingface.co/<org>/<model>. Payload matches the live
huggingface.co points (tier C / source_type reference / trust 0.75) + provenance
ai_hf_model_cards, so it's directly mergeable into aw_v4_sat / aw_prod (config-B).

Subcommands (all resumable, HF-polite <=4 req/s + backoff):
  enumerate --out model_list.json
  fetch     --list model_list.json --cache-dir cards/ [--limit N]
  parse     --list model_list.json --cache-dir cards/ --out hfcards.jsonl \
            --stats stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse

from cytir.acquire import fetch_process as fp  # noqa: E402
from cytir.consent.block import ConsentBlocked  # noqa: E402
from cytir.sources.family import HttpSpec  # noqa: E402
from cytir.sources.http import make_client  # noqa: E402
from cytir.extract.hf_model_cards import (  # noqa: E402,F401  # re-exported: read as HFC.card_records/derivative_base by direct_compose.py (live refresh path)
    DOMAIN, MAJOR_ORGS, MAX_BODY_CHUNKS, MIN_BODY_WORDS, PROVENANCE, SOURCE_TYPE, TIER,
    TOP_DOWNLOADS, TOP_LIKES, TRUST, _LANGS_OK, _is_quant_spam, _safe, _structured_text,
    base_model_repos, card_records, derivative_base, new_parse_stats, records_for_model)

API = "https://huggingface.co/api/models"
RAW = "https://huggingface.co/{id}/resolve/main/README.md"
MIN_INTERVAL = 0.28            # <=4 req/s politeness (per-process)


_last_req = [0.0]


def _throttle():
    dt = time.time() - _last_req[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last_req[0] = time.time()


def _client(timeout=60):
    return make_client(HttpSpec(timeout=timeout, user_agent_component="hf-model-cards",
                                follow_redirects=True))


def _get(url, accept_json=True, tries=6):
    """Polite GET with exponential backoff on 429/5xx. Returns (status, body|None)."""
    for attempt in range(tries):
        _throttle()
        try:
            with _client() as client:
                r = client.get(url)
            if r.status_code >= 400:   # urllib raised HTTPError for these
                if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                    wait = min(60, 2 ** attempt) + (attempt * 0.5)
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        wait = max(wait, int(ra))
                    print(f"  backoff {r.status_code} {wait:.0f}s {url[:80]}", flush=True)
                    time.sleep(wait)
                    continue
                return r.status_code, None, {}
            body = r.content.decode("utf-8", "replace")
            return r.status_code, body, r.headers   # httpx.Headers: case-insensitive like urllib
        except ConsentBlocked:
            raise    # a recorded block is final: never retried, never parked
        except Exception as e:  # noqa: BLE001
            # build-box net-blip: don't burn the retry budget per-URL while the
            # box is off-network — park until connectivity returns, then retry
            # the same attempt.
            if "Network is unreachable" in str(e) or "Temporary failure" in str(e):
                print(f"  net down; parking ({url[:60]})", flush=True)
                while True:
                    time.sleep(30)
                    try:
                        with _client(timeout=8) as probe:
                            probe.get(API + "?limit=1").raise_for_status()
                        print("  net back; resuming", flush=True)
                        break
                    except ConsentBlocked:
                        raise
                    except Exception:  # noqa: BLE001
                        continue
                continue
            if attempt < tries - 1:
                time.sleep(min(30, 2 ** attempt))
                continue
            print(f"  ERR {e} {url[:80]}", flush=True)
            return 0, None, {}
    return 0, None, {}


# --------------------------------------------------------------------------- #
# enumerate
# --------------------------------------------------------------------------- #

def _paginate(params, cap):
    """Yield model rows from the list endpoint, following cursor pages up to cap."""
    url = API + "?" + urllib.parse.urlencode(params)
    seen = 0
    while url and seen < cap:
        status, body, headers = _get(url)
        if status != 200 or not body:
            print(f"  list stop status={status}", flush=True)
            break
        rows = json.loads(body)
        if not rows:
            break
        for r in rows:
            yield r
            seen += 1
            if seen >= cap:
                return
        nxt = None
        for part in headers.get("Link", "").split(","):
            if 'rel="next"' in part:
                m = re.search(r"<([^>]+)>", part)
                if m:
                    nxt = m.group(1)
        url = nxt


def cmd_enumerate(args):
    models: dict[str, dict] = {}

    def add(row, reason, rank=None):
        mid = row.get("id") or row.get("modelId")
        if not mid or row.get("private"):
            return
        m = models.setdefault(mid, {"id": mid, "reasons": [], "row": row})
        if reason not in m["reasons"]:
            m["reasons"].append(reason)
        # keep the freshest list row (they carry gated/downloads/likes/tags)
        m["row"] = row
        if reason == "top_downloads" and rank is not None:
            m["dl_rank"] = rank

    print(f"enumerate: top {TOP_DOWNLOADS} by downloads…", flush=True)
    for i, row in enumerate(_paginate(
            {"sort": "downloads", "direction": "-1", "limit": "100",
             "full": "true"}, TOP_DOWNLOADS)):
        add(row, "top_downloads", rank=i + 1)
    print(f"  -> {len(models)} unique so far", flush=True)

    print(f"enumerate: top {TOP_LIKES} by likes…", flush=True)
    for row in _paginate({"sort": "likes", "direction": "-1", "limit": "100",
                          "full": "true"}, TOP_LIKES):
        add(row, "top_likes")
    print(f"  -> {len(models)} unique so far", flush=True)

    for org in MAJOR_ORGS:
        n0 = len(models)
        for row in _paginate({"author": org, "limit": "100", "full": "true"},
                             100000):
            add(row, f"org:{org}")
        print(f"  org {org}: +{len(models) - n0} (total {len(models)})",
              flush=True)

    out = os.path.expanduser(args.out)
    with open(out, "w") as f:
        json.dump({"models": list(models.values()),
                   "counts": {"total": len(models),
                              "orgs": MAJOR_ORGS,
                              "top_downloads": TOP_DOWNLOADS,
                              "top_likes": TOP_LIKES}}, f)
    print(f"WROTE {out}: {len(models)} unique models", flush=True)


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #


def cmd_fetch(args):
    lst = json.load(open(os.path.expanduser(args.list)))["models"]
    cache = os.path.expanduser(args.cache_dir)
    os.makedirs(cache, exist_ok=True)
    if args.limit:
        lst = lst[:args.limit]
    done = skip = 0
    t0 = time.time()
    for i, m in enumerate(lst):
        mid = m["id"]
        base = os.path.join(cache, _safe(mid))
        detail_p, readme_p, meta_p = base + ".detail.json", base + ".README.md", base + ".meta.json"
        if os.path.exists(meta_p):
            skip += 1
            continue
        # detail (cardData license + safetensors params + config)
        status, body, _ = _get(f"{API}/{mid}")
        detail = None
        if status == 200 and body:
            detail = json.loads(body)
            with open(detail_p, "w") as f:
                f.write(body)
        elif status not in (401, 403, 404):
            # transient (network blip / 5xx after retries): do NOT mark done —
            # a later resume pass retries this model (build-box outage lesson)
            continue
        gated = bool((detail or m.get("row") or {}).get("gated"))
        # README body — skip for gated (task rule: metadata-only for gated)
        rstatus = None
        if not gated:
            rstatus, rbody, _ = _get(RAW.format(id=mid), accept_json=False)
            if rstatus == 200 and rbody:
                with open(readme_p, "w") as f:
                    f.write(rbody)
            elif rstatus not in (401, 403, 404):
                continue                     # transient: retry on next pass
        with open(meta_p, "w") as f:
            json.dump({"id": mid, "detail_status": status,
                       "readme_status": rstatus, "gated": gated,
                       "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime())}, f)
        done += 1
        if done % 100 == 0:
            rate = (done) / max(time.time() - t0, 1)
            print(f"  fetched {done} (skip {skip}) {rate:.1f}/s  [{i+1}/{len(lst)}]",
                  flush=True)
    print(f"FETCH done: {done} fetched, {skip} already cached", flush=True)


# --------------------------------------------------------------------------- #
# parse
# --------------------------------------------------------------------------- #

def cmd_parse(args):
    lst = json.load(open(os.path.expanduser(args.list)))["models"]
    cache = os.path.expanduser(args.cache_dir)
    top500 = {m["id"] for m in lst if m.get("dl_rank") and m["dl_rank"] <= 500}
    stats = new_parse_stats(len(lst))
    seen_pid = set()
    out = open(os.path.expanduser(args.out), "w")
    for m in lst:
        for rec in records_for_model(m, cache, top500, seen_pid, stats):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()
    with open(os.path.expanduser(args.stats), "w") as f:
        json.dump(stats, f, indent=2)
    total = stats["structured_passages"] + stats["body_passages"]
    print(f"PARSE done: {stats['kept_models']} models kept -> {total} passages "
          f"({stats['structured_passages']} structured + {stats['body_passages']} body)",
          flush=True)
    print(json.dumps(stats, indent=2), flush=True)



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enumerate")
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_enumerate)
    f = sub.add_parser("fetch")
    f.add_argument("--list", required=True)
    f.add_argument("--cache-dir", required=True)
    f.add_argument("--limit", type=int, default=0)
    f.set_defaults(func=cmd_fetch)
    p = sub.add_parser("parse")
    p.add_argument("--list", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stats", required=True)
    p.set_defaults(func=cmd_parse)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
