#!/usr/bin/env python3
"""Task 169 — WEEKLY-DELTA v1: refresh the curated freshness manifest on a
weekly cadence by re-embedding ONLY the entries whose NATIVE change signal
advanced, and last_seen-touching the rest WITHOUT re-embedding.

This generalises the daily `freshness_ingest.py` loop (task 106/148) with
native-signal change detection, per `specs/weekly-delta-update.md` v1 scope:
the signalled subset of the 199-source manifest. It reuses the freshness
loop's proven rails end-to-end: the same TEI arctic embedder + MRL-128 slice,
the same `aw_common.bm25_tokens`, the same `uuid5(url)` url-addressed point ids
(upsert-in-place), the same `/tmp/aw_prod_writer.lock` + per-batch `/health`
gate. NO new prod-write path.

WHAT IT DOES, per manifest entry:
  * change_signal.type == http_etag           -> conditional GET (If-None-Match)
  * change_signal.type == http_last_modified  -> conditional GET (If-Modified-Since)
  * change_signal.type == sitemap_lastmod     -> fetch sitemap, diff <lastmod>
  * change_signal absent / type == none       -> no signal (WAF-blocked etc.):
        ALWAYS re-embed (current freshness behaviour; 25-word snippets, cheap)

  Classification -> {changed, unchanged, unknown, none}:
    embed set = changed | unknown | none   (re-embed via build_points -> upsert)
    touch set = unchanged                   (set_payload last_seen ONLY — no GPU)

  A tiny mutable sidecar `data/refresh_signal_state.json` (regenerable runtime
  state, NOT committed) stores per-url {etag, last_modified, lastmod,
  last_checked, last_embedded} — the signal cursor. First run of a signalled
  entry has no cursor -> classified `unknown` -> embedded once to seed it.

CONFIG-CHANGE GUARD (mandatory, spec §6): before ANY write, compare the LIVE
collection's dim/distance/quant/sparse config + the runtime embed-model/MRL-dim
against data/build_config.json. Any drift -> REFUSE to run (content-delta and
config-change must never mix; a config change mid-write deadlocked prod once).

  # dry-run (no writes): classify + report only
  python -m cytir.news.weekly_delta --manifest ../data/refresh_manifest.json --dry-run

  # real run (writer-lock + health-gated upserts):
  python -m cytir.news.weekly_delta --manifest ../data/refresh_manifest.json \
      --collection aw_prod --health-url http://localhost:8090/health

Writes a machine + human summary to data/weekly_delta_last_summary.{json,txt}
(a cron on the orchestrator box mails it to Tim — the serving box has no mailer; see
deploy/scripts/weekly-delta-mail.sh).
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import datetime as dt
import email.utils
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter, namedtuple

from cytir.news import freshness_ingest as fi  # noqa: E402  (shared embed/build/lock/consent)
from cytir.core.paths import data_path  # noqa: E402
from cytir.core.qdrant import qdrant_connect  # noqa: E402
from cytir.core.common import mint_card_point_id  # noqa: E402

MRL_DIM = fi.MRL_DIM

# ONE identity, shared with the module that evaluates consent (task 432).
# This used to be a LOCAL UA of its own:
#     "Mozilla/5.0 (compatible; AuthoritativeWebBot/1.0; +https://websear.ch)"
# while the consent gate at the bottom of this file is `fi.consent_ok(url)`,
# which evaluates robots.txt with `fi._ua()`. Python's robotparser resolves an
# agent to the token BEFORE the first slash, so that local string resolved to
# the agent "mozilla" — a site operator's `User-agent: AuthoritativeWebBot`
# rule was honored in our bookkeeping and invisible on the wire. We asked
# permission as one crawler and fetched as another.
#
# Taking fi._ua() makes the identity we ask with the identity we fetch with, and
# leaves exactly one place to change it.

# classification -> action
EMBED_STATUSES = {"changed", "unknown", "none"}
TOUCH_STATUSES = {"unchanged"}

HttpResp = namedtuple("HttpResp", "status headers body")


# --------------------------- HTTP (injectable) ----------------------------- #
def http_get(url: str, headers: dict | None = None, timeout: int = 20) -> HttpResp:
    """Plain/conditional GET. Returns HttpResp(status, headers, body). A 304 is
    returned as status 304 (NOT raised). Network error -> status 0."""
    try:
        r = fi.guarded_get(url, timeout, headers)     # consent-guarded (task 1621)
        if r.status_code >= 300:
            # 304 Not Modified is the happy path for a conditional GET; an error status has no body.
            return HttpResp(r.status_code, dict(r.headers), b"")
        return HttpResp(r.status_code, dict(r.headers), r.content)
    except Exception:  # noqa: BLE001  (DNS/timeout/TLS/ConsentBlocked -> treat as unknown)
        return HttpResp(0, {}, b"")


def _h(headers: dict, name: str):
    """Case-insensitive header lookup."""
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v
    return None


def _parse_http_date(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return email.utils.mktime_tz(email.utils.parsedate_tz(s))
    except Exception:  # noqa: BLE001
        return None


# --------------------------- signal classification ------------------------- #
def check_signal(entry: dict, prev: dict, get=http_get) -> tuple[str, dict]:
    """Classify ONE manifest entry against its stored cursor `prev`.

    Returns (status, new_cursor) where status in
    {changed, unchanged, unknown, none}. `new_cursor` is the cursor fields to
    persist for this url (merged by the caller). Pure w.r.t. `get` (injectable
    for tests) — does no embedding, upserting, or state I/O itself."""
    sig = entry.get("change_signal") or {}
    stype = sig.get("type", "none")
    url = entry["url"]

    if stype in (None, "none"):
        return "none", {}

    if stype == "http_etag":
        stored = prev.get("etag")
        hdrs = {"If-None-Match": stored} if stored else {}
        r = get(url, hdrs)
        if r.status == 304:
            return "unchanged", {}
        if r.status == 200:
            new = _h(r.headers, "ETag")
            if not new:
                return "unknown", {}
            if stored is None:
                return "unknown", {"etag": new}          # first sight -> seed
            return ("unchanged" if new == stored else "changed"), {"etag": new}
        return "unknown", {}

    if stype == "http_last_modified":
        stored = prev.get("last_modified")
        hdrs = {"If-Modified-Since": stored} if stored else {}
        r = get(url, hdrs)
        if r.status == 304:
            return "unchanged", {}
        if r.status == 200:
            new = _h(r.headers, "Last-Modified")
            if not new:
                return "unknown", {}
            if stored is None:
                return "unknown", {"last_modified": new}  # first sight -> seed
            tn, ts = _parse_http_date(new), _parse_http_date(stored)
            if tn is not None and ts is not None:
                return ("changed" if tn > ts else "unchanged"), {"last_modified": new}
            return ("unchanged" if new == stored else "changed"), {"last_modified": new}
        return "unknown", {}

    if stype == "sitemap_lastmod":
        lastmod = _sitemap_lastmod(sig.get("url"), url, get)
        if lastmod is None:
            return "unknown", {}
        stored = prev.get("lastmod")
        if stored is None:
            return "unknown", {"lastmod": lastmod}        # first sight -> seed
        return ("changed" if lastmod > stored else "unchanged"), {"lastmod": lastmod}

    # unrecognised signal type -> be safe, re-embed
    return "unknown", {}


def _sitemap_lastmod(sitemap_url: str | None, page_url: str, get=http_get) -> str | None:
    """Return the <lastmod> for page_url from sitemap_url, or None (not found /
    fetch error -> caller treats as unknown, never as unchanged)."""
    if not sitemap_url:
        return None
    r = get(sitemap_url, {})
    if r.status != 200 or not r.body:
        return None
    try:
        root = ET.fromstring(r.body)
    except Exception:  # noqa: BLE001
        return None
    # strip namespaces
    for el in root.iter():
        el.tag = el.tag.rsplit("}", 1)[-1]
    for url_el in root.iter("url"):
        loc = url_el.findtext("loc")
        if loc and loc.strip() == page_url:
            lm = url_el.findtext("lastmod")
            return lm.strip() if lm else None
    return None


# ------------------------------- config guard ------------------------------ #
def normalize_live_config(coll_info: dict, embed_model: str, mrl_dim: int) -> dict:
    """Flatten a qdrant get_collection() result (+ runtime model/dim) into the
    same shape as data/build_config.json for comparison."""
    cfg = coll_info["config"]["params"]
    dense = cfg["vectors"]["dense"]
    sparse = cfg.get("sparse_vectors", {}).get("bm25", {})
    quant = coll_info["config"].get("quantization_config") or {}
    scalar = quant.get("scalar", {})
    return {
        "embed_model": embed_model,
        "mrl_dim": mrl_dim,
        "dense_size": dense.get("size"),
        "dense_distance": dense.get("distance"),
        "dense_on_disk": dense.get("on_disk"),
        "sparse_bm25_modifier": sparse.get("modifier"),
        "sparse_bm25_on_disk": (sparse.get("index") or {}).get("on_disk"),
        "quant_type": scalar.get("type"),
    }


def config_guard(live: dict, pinned: dict) -> tuple[bool, list[str]]:
    """Compare live-collection config against the pinned build config. Returns
    (ok, diffs). ok is True only when every checked key matches. Pure."""
    keys = ["embed_model", "mrl_dim", "dense_size", "dense_distance",
            "dense_on_disk", "sparse_bm25_modifier", "sparse_bm25_on_disk",
            "quant_type"]
    diffs = []
    for k in keys:
        want, got = pinned.get(k), live.get(k)
        if str(want) != str(got):
            diffs.append(f"{k}: live={got!r} != pinned={want!r}")
    return (not diffs), diffs


def require_supported_chunk_epoch(pinned: dict) -> None:
    """Refuse the URL-addressed v1 writer after a chunk@2 promotion.

    ``freshness_ingest.build_points`` emits one uuid5(url) record and does not
    run the generic chunker or supersede a source group.  Continuing to run it
    after the build profile flips to epoch 2 would mix legacy records into the
    new collection.  The successor must use source-identity replay plus the
    chunk@2 journal.
    """
    if pinned.get("chunk_epoch") != 1 or pinned.get("chunker_mode") != "legacy":
        raise SystemExit(
            "CONFIG-CHANGE GUARD TRIPPED: weekly_delta v1 is url-addressed "
            "chunk@1 only; use the chunk@2 source replay/journal path")


# ------------------------------- state sidecar ----------------------------- #
def load_state(path: str) -> dict:
    try:
        return json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        return {}


def save_state(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)


# --------------------------- classify all entries -------------------------- #
def classify(sources: list[dict], state: dict, get=http_get, now_iso: str = "") -> list[dict]:
    """Classify every source; MUTATES `state` cursors (etag/lastmod/last_checked).
    Returns a list of {entry, status} in source order."""
    now_iso = now_iso or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = []
    for s in sources:
        prev = state.get(s["url"], {})
        status, cursor = check_signal(s, prev, get)
        row = dict(prev)
        row.update(cursor)
        row["last_checked"] = now_iso
        state[s["url"]] = row
        out.append({"entry": s, "status": status})
    return out


# ----------------------------------- main ---------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--build-config",
                    default=str(data_path("build_config.json")))
    ap.add_argument("--state",
                    default=str(data_path("refresh_signal_state.json")))
    ap.add_argument("--summary",
                    default=str(data_path("weekly_delta_last_summary")))
    ap.add_argument("--collection", default="aw_prod")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--tei-url", default="http://localhost:8088")
    ap.add_argument("--health-url", default="http://localhost:8090/health")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true",
                    help="classify + report only; no signal-state write, no upsert")
    args = ap.parse_args()
    # task 864 (review 869 S1): from an aw_e10 T0 to the flip only the drain writes the twin
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(args, tool='cytir/news/weekly_delta.py')

    # Shared with the daily job.  This v1 URL-addressed path cannot form the
    # complete source keep-set required by aw_b1, nor can it mirror a safe
    # supersede transaction.  Abort before config/network/Qdrant work; dry-run
    # remains useful for native-signal diagnosis.
    fi.require_principal_source_replay(dry_run=args.dry_run)

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    sources = manifest["sources"]
    today = dt.date.today().isoformat()
    now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"weekly-delta: {len(sources)} sources; collection={args.collection}; "
          f"dry_run={args.dry_run}", flush=True)

    from qdrant_client import models
    client = qdrant_connect(args.qdrant_url, timeout=120)

    # --- CONFIG-CHANGE GUARD (before anything else) --------------------------
    pinned = json.load(open(args.build_config, encoding="utf-8"))
    require_supported_chunk_epoch(pinned)
    try:
        tei_model = IH.fetch_json(f"{args.tei_url}/info", timeout=15).get("model_id")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"ABORT: cannot read TEI /info for config guard: {e!r}")
    coll = client.get_collection(args.collection).model_dump(mode="json")
    live_cfg = normalize_live_config(coll, tei_model, MRL_DIM)
    ok, diffs = config_guard(live_cfg, pinned)
    if not ok:
        print("CONFIG-CHANGE GUARD TRIPPED — refusing to run:", flush=True)
        for d in diffs:
            print(f"  - {d}", flush=True)
        raise SystemExit("config drift: content-delta and config-change must not mix "
                         "(see specs/weekly-delta-update.md §6). Rebuild+alias-swap "
                         "instead, and update data/build_config.json.")
    print(f"config-guard OK (dim={live_cfg['dense_size']} quant={live_cfg['quant_type']} "
          f"model={tei_model})", flush=True)

    # --- classify via native signals ----------------------------------------
    state = load_state(args.state)
    classified = classify(sources, state, now_iso=now_iso)

    embed_list = [c["entry"] for c in classified if c["status"] in EMBED_STATUSES]
    touch_list = [c["entry"] for c in classified if c["status"] in TOUCH_STATUSES]
    by_status = Counter(c["status"] for c in classified)
    # per-source-type embed/unchanged breakdown
    per_type: dict[str, Counter] = {}
    for c in classified:
        st = c["entry"].get("source_type") or "?"
        per_type.setdefault(st, Counter())[c["status"]] += 1

    print(f"classified: {dict(by_status)} -> embed={len(embed_list)} "
          f"touch={len(touch_list)}", flush=True)

    # consent-gate the embed set (same gate as the daily loop)
    embed_gated = [s for s in embed_list if fi.consent_ok(s["url"])]
    if len(embed_gated) != len(embed_list):
        print(f"consent: {len(embed_gated)}/{len(embed_list)} embed-set passed", flush=True)

    # ADR-105 (task 581): persisted inputs are a precondition to load. Same
    # reasoning as the daily lane -- this path embeds the manifest's own text
    # and seals nothing, so every row it writes is un-rebuildable unless the
    # page is already persisted by some other lane. Checked over the EMBED set
    # only: the touch set writes `last_seen` and no content.
    # An empty embed set is a real weekly outcome (nothing changed), so it is
    # skipped explicitly and SAID so -- not folded into a silent pass.
    import cytir.corpus.persisted as PI
    from cytir.corpus import persist_l1_records as PL1
    from cytir.corpus import authored_text_guard as ATG
    # ADR-106 / task 586. The embed set below is re-embedded from the
    # manifest's OWN `text` -- this lane never fetches the page. That is why
    # the refresh timer could bump `last_seen` for ten weeks without a single
    # fact changing: `classify()` detects that the page moved, and then
    # re-embeds the same authored body with today's date on it. Refuse before
    # the L1 filing, because filing it is what made the ADR-105 guard pass.
    if embed_gated:
        ATG.assert_chunk_embeddable(
            [dict(s, text_origin=s.get("text_origin")
                  or "freshness-manifest:curated text "
                     "(data/refresh_manifest.json)")
             for s in embed_gated],
            label="weekly_delta embed set")
    if embed_gated:
        try:
            written = PL1.write_l1_records(
                ({"url": src["url"], "text": src.get("text"),
                  "title": src.get("title"), "lang": src.get("lang"),
                  "domain": src.get("domain")} for src in embed_gated),
                release="freshness", source=f"weekly-{today}",
                origin="freshness-manifest:curated text (data/refresh_manifest.json)",
                note=f"weekly_delta {args.manifest} -> {args.collection}")
            pv = PI.assert_pages_persisted(
                [{"url": src["url"]} for src in embed_gated], require="l1",
                label=f"weekly-delta embed set -> {args.collection}",
                overlay=PI.overlay_from_paths(written))
        except (PI.PersistedInputsError, PL1.L1PersistError) as exc:
            raise SystemExit(f"REFUSED: {exc}") from None
        print(f"persisted inputs: {pv['pages']:,} embed-set page(s) resolved",
              flush=True)
    else:
        pv = {"verdict": "SKIPPED", "reason": "embed set is empty", "pages": 0}
        print("persisted inputs: embed set is empty, nothing to check",
              flush=True)

    summary = {
        "run_at": now_iso, "collection": args.collection, "dry_run": args.dry_run,
        "persisted_inputs": pv,
        "sources": len(sources), "status_counts": dict(by_status),
        "embed_planned": len(embed_gated), "touch_planned": len(touch_list),
        "per_source_type": {k: dict(v) for k, v in sorted(per_type.items())},
        "embedded": 0, "touched": 0, "points_before": None, "points_after": None,
    }

    if args.dry_run:
        summary["note"] = "dry-run: no signal-state write, no upsert"
        _write_summary(args.summary, summary)
        print("DRY-RUN — no writes", flush=True)
        return

    # --- write path (guarded, health-gated) ---------------------------------
    try:
        summary["points_before"] = client.get_collection(args.collection).points_count
    except Exception:  # noqa: BLE001
        pass

    fi.take_lock(f"task169 weekly-delta (embed {len(embed_gated)}, touch {len(touch_list)})")
    embedded = touched = 0
    try:
        # 1) EMBED changed/unknown/none -> upsert in place (uuid5(url))
        for i in range(0, len(embed_gated), args.batch):
            chunk = embed_gated[i:i + args.batch]
            pts = fi.build_points(models, chunk, args.tei_url, today)
            client.upsert(args.collection, points=pts, wait=True)
            embedded += len(pts)
            try:
                assert IH.send(args.health_url, timeout=15)[0] == 200
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"ABORT: health check failed after embed {embedded}: {e!r}")
            print(f"  embedded {embedded}/{len(embed_gated)} (health OK)", flush=True)
            for s in chunk:
                state.setdefault(s["url"], {})["last_embedded"] = today

        # 2) TOUCH unchanged -> last_seen payload set ONLY (no GPU, cheap)
        for i in range(0, len(touch_list), 200):
            chunk = touch_list[i:i + 200]
            ids = [mint_card_point_id(s["url"]) for s in chunk]
            client.set_payload(args.collection, payload={"last_seen": today},
                               points=ids, wait=True)
            touched += len(ids)
            try:
                assert IH.send(args.health_url, timeout=15)[0] == 200
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"ABORT: health check failed after touch {touched}: {e!r}")
        if touch_list:
            print(f"  touched last_seen on {touched} unchanged (no re-embed)", flush=True)
    finally:
        fi.release_lock()

    save_state(args.state, state)
    try:
        summary["points_after"] = client.get_collection(args.collection).points_count
    except Exception:  # noqa: BLE001
        pass
    summary["embedded"], summary["touched"] = embedded, touched
    _write_summary(args.summary, summary)
    print(f"DONE weekly-delta: embedded={embedded} touched={touched} "
          f"points_after≈{summary['points_after']}", flush=True)


def _write_summary(base: str, summary: dict) -> None:
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    lines = [
        "AW weekly-delta run summary",
        f"  run_at        : {summary['run_at']}",
        f"  collection    : {summary['collection']}  (dry_run={summary['dry_run']})",
        f"  sources       : {summary['sources']}",
        f"  status counts : {summary['status_counts']}",
        f"  embedded      : {summary['embedded']}  (planned {summary['embed_planned']})",
        f"  touched       : {summary['touched']}  (planned {summary['touch_planned']})",
        f"  points        : {summary['points_before']} -> {summary['points_after']}",
        "  per source_type:",
    ]
    for st, counts in summary["per_source_type"].items():
        lines.append(f"    {st:<20} {counts}")
    if summary.get("note"):
        lines.append(f"  note          : {summary['note']}")
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
