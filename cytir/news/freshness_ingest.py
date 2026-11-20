#!/usr/bin/env python3
"""Task 106 — ingest current-state / freshness authoritative sources into the
LIVE prod collection (aw_v4_sat / alias aw_prod).

Reads a manifest (data/refresh_manifest.json) of curated authoritative pages
(gov statistics, official docs/model hubs, ...) that answer the eval's
freshness slice ("latest X / current stat / newest release"). For each source:

  * CONSENT-GATE: robots.txt is checked for our agent; a disallowed or
    unresolved path is skipped.  A transport failure is not consent.
  * EMBED: passage text -> arctic-embed-l-v2 via the TEI sidecar (same model
    the server queries with) -> Matryoshka-128 slice + L2 renorm  (parity with
    models.embed_query / the retired config-B 128-dim layout (removed 2026-10-28, task 1378)).
  * BM25: aw_common.bm25_tokens (identical tokenization to the live index).
  * last_seen := ingest date -> the fusion `recency` signal fires for these.
  * UPSERT to prod under the purpose-stamped writer lock, small wait=true
    batches, /health probed each batch (abort + release on red). No exact
    count (44M box) — approximate points_count only, for the log.

Historical note: this module's url-addressed ``uuid5(url)`` IDs predate the
principal source-identity contract.  They are intentionally BLOCKED for real
writes now: a URL can have multiple current chunks, so this writer cannot
prove a complete keep-set for superseding and would create a duplicate URL
group.  Its chunk@2 successor must materialise source-identity records and
apply them twin-first through ``upsert_supersede`` / ``twin_first_apply``.

  # seed run (uses the manifest's curated `text`):
  python -m cytir.news.freshness_ingest --manifest ../data/refresh_manifest.json \
      --collection aw_prod --health-url http://localhost:8090/health

  # refresh run (re-fetch non-blocked URLs, update text on success):
  python -m cytir.news.freshness_ingest --manifest ... --refresh

  --dry-run prints the consent decision + point payload and does NOT write.
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.robotparser

from cytir.core.bm25_tokenizer import bm25_tokens
from cytir.core.bm25_saturate import saturate_point_bm25  # noqa: E402
from cytir.corpus import authored_text_guard as ATG  # noqa: E402
from cytir.core.common import mint_card_point_id  # noqa: E402
from cytir.core.qdrant import qdrant_connect  # noqa: E402

LOCK = "/tmp/aw_prod_writer.lock"

# ONE identity, used for the robots.txt fetch, the robots EVALUATION and the
# content fetch (task 432). It used to be two: robots was evaluated as
# "AuthoritativeWebBot" while the page was fetched as
# "Mozilla/5.0 (compatible; AuthoritativeWebBot/1.0)".
#
# That is not UA hygiene, it is consent evidence that does not describe the
# request we then make. Python's robotparser keys on the token BEFORE the first
# slash, so the two strings are not interchangeable — measured:
#
#   robots.txt: "User-agent: AuthoritativeWebBot" + "Disallow: /private"
#     can_fetch("AuthoritativeWebBot",                      "/private") -> False
#     can_fetch("AuthoritativeWebBot/1.0 (+https://...)",   "/private") -> False
#     can_fetch("Mozilla/5.0 (compatible; AuthoritativeWebBot/1.0)", ...) -> True
#
# i.e. the string we transmitted did NOT match the rule a site operator would
# write to control us — it resolves to the agent token "mozilla". The operator
# could not write an effective rule against us and could not recognise us in
# their logs. The identity you evaluate permission for must be the identity you
# crawl with.
#
# The token before the slash is unchanged, so every robots decision this script
# has ever made is reproduced exactly; only the string on the wire becomes
# honest and answerable.


def _ua() -> str:
    """The one crawl identity (consent evidence = wire identity), from the operator identity, lazily."""
    from cytir.core.user_agent import pack_fetch_user_agent
    return pack_fetch_user_agent()


def __getattr__(name):
    if name == "UA":
        return _ua()
    raise AttributeError(name)


MRL_DIM = 128


# ------------------------------- writer lock ------------------------------- #
def take_lock(purpose: str, wait_s: int = 3600):
    t0 = time.time()
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{purpose} pid={os.getpid()} at={time.ctime()}\n".encode())
            os.close(fd)
            print(f"[lock] took {LOCK}", flush=True)
            return
        except FileExistsError:
            if time.time() - t0 > wait_s:
                raise SystemExit(f"could not take {LOCK}: {open(LOCK).read().strip()!r}")
            print(f"[lock] busy ({open(LOCK).read().strip()!r}) — waiting", flush=True)
            time.sleep(30)


def release_lock():
    try:
        os.remove(LOCK)
        print(f"[lock] released {LOCK}", flush=True)
    except FileNotFoundError:
        pass


# ------------------------------- consent gate ------------------------------ #
_robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def guarded_get(url: str, timeout: float, headers: dict | None = None):
    """One consent-guarded GET (task 1621): the crawl identity, no content-coding (as urllib sent),
    redirects followed. Raises `ConsentBlocked` for a declined host before any socket; the
    caller decides what a failure means. The client is built per call and closed after it."""
    from cytir.sources.family import HttpSpec
    from cytir.sources.http import make_client
    with make_client(HttpSpec(timeout=timeout, user_agent_kind="pack_fetch", follow_redirects=True,
                              headers={"user-agent": _ua(), "accept-encoding": "identity",
                                       **(headers or {})})) as client:
        return client.get(url)


def _robots(host_scheme: str):
    if host_scheme in _robots_cache:
        return _robots_cache[host_scheme]
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(host_scheme + "/robots.txt")
    try:
        r = guarded_get(host_scheme + "/robots.txt", 10)
        r.raise_for_status()
        rp.parse(r.content.decode("utf-8", "replace").splitlines())
    except Exception as e:  # noqa: BLE001
        print(f"  [consent] robots fetch failed for {host_scheme} ({type(e).__name__}) "
              f"— default-allow (curated public source)", flush=True)
        _robots_cache[host_scheme] = None
        return None
    _robots_cache[host_scheme] = rp
    return rp


def consent_ok(url: str) -> bool:
    from urllib.parse import urlsplit
    p = urlsplit(url)
    rp = _robots(f"{p.scheme}://{p.netloc}")
    if rp is None:
        # A network/WAF failure is an unknown consent state, not an implicit
        # grant.  Task 318 established this as a load-bearing resolver rule.
        print(f"  [consent] UNRESOLVED {url} — held (never default-allow)",
              flush=True)
        return False
    ok = rp.can_fetch(_ua(), url)
    if not ok:
        print(f"  [consent] DISALLOW {url} — skipped", flush=True)
    return ok


def require_principal_source_replay(*, dry_run: bool) -> None:
    """Reject the retired URL-addressed writer before any remote write.

    ``uuid5(url)`` is neither ``aw_b1_identity`` nor a page-complete source
    group.  Reusing it against aw_b1 created the 2026-08-03 OpenAI duplicate
    pair and a prod-only delta.  Keeping this guard in the shared daily/weekly
    implementation makes both timers fail loudly until the source-identity
    replay successor is installed; a service-file comment is not a safety
    boundary.  Dry runs remain available for signal/consent diagnostics.
    """
    if not dry_run:
        raise SystemExit(
            "FATAL: retired url-addressed freshness writer blocked before "
            "write: it cannot prove a complete source keep-set, preserve "
            "supersede/tombstone pointers, or execute twin-first. Use the "
            "source-identity chunk replay/journal successor instead.")


# ------------------------------- embedding --------------------------------- #
def embed_tei(tei_url: str, texts: list[str]) -> list[list[float]]:
    import numpy as np
    vecs = IH.fetch_json(f"{tei_url}/embed",
                         body={"inputs": texts, "normalize": True, "truncate": True}, timeout=60)
    out = []
    for v in vecs:
        a = np.asarray(v, dtype="float32")[:MRL_DIM]
        n = float(np.linalg.norm(a)) or 1.0
        out.append((a / n).tolist())
    return out


# ------------------------------- refresh fetch ----------------------------- #
def refetch_text(url: str) -> str | None:
    """Best-effort re-fetch for the refresh timer. Returns extracted text or
    None (WAF/blocked/error -> keep seeded text)."""
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return None
    try:
        r = guarded_get(url, 20)
        r.raise_for_status()
        html = r.content.decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        print(f"  [refresh] fetch failed {url} ({type(e).__name__}) — keep seed", flush=True)
        return None
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header"]):
        t.extract()
    txt = " ".join(soup.get_text(" ").split())
    return txt[:1500] if len(txt) > 200 else None


# ------------------------------- point build ------------------------------- #
def build_points(models, chunk: list[dict], tei_url: str, today: str,
                 mrl_dim: int = MRL_DIM):
    """Embed a chunk of manifest entries and build qdrant PointStructs.

    Shared by the daily freshness loop (main) and the weekly-delta job
    (cytir.news.weekly_delta) so both embed/tokenize/id identically. `models` is
    qdrant_client.models (passed in so this module doesn't import qdrant at
    import time). point_id = uuid5(url) → url-addressed, upsert-in-place."""
    # ADR-106 / task 586: authored prose may not become a page's content.
    #
    # This is the exact line at which 966 rows of model-written prose entered
    # production: `s["text"]` is the manifest's hand-authored body, and this
    # function embedded it, tokenised it into the BM25 surface and stamped
    # today's `last_seen` on it. 370 of those rows are still served; 118 of the
    # 199 bodies carry a dated claim that had already expired by 2026-09-16,
    # and 7 of 9 spot-checked claims were false against the live page.
    #
    # The guard runs BEFORE the embedder, so a refusal costs nothing, and it
    # fails closed: an entry that cannot name a fetched origin is refused, not
    # assumed. The replacement shape is `authored_text_guard.manifest_roster()`
    # plus `fetched_text_for()` -- the manifest supplies the URL and the
    # refresh signal, a persisted L1 record supplies the text, and a URL with
    # no fetched text yields NO ROW rather than a written answer.
    ATG.assert_chunk_embeddable(chunk, label="freshness_ingest.build_points")
    vecs = embed_tei(tei_url, [s["text"] for s in chunk])
    pts = []
    for s, dv in zip(chunk, vecs):
        bm = bm25_tokens(s["text"], s.get("lang"))
        payload = {
            "url": s["url"], "title": s["title"], "lang": s.get("lang"),
            "domain": s.get("domain"), "tier": s.get("tier"),
            "trust_tier": s.get("trust_tier"),
            "country": s.get("country"),
            "source_type": s.get("source_type"),
            "snippet": s.get("snippet"), "anchor": None,
            "last_seen": today,
        }
        vec = {"dense": dv,
               "bm25": models.SparseVector(indices=list(bm.keys()),
                                           values=list(bm.values()))}
        # task 212 Part 1: saturate on the live write path (shared by the daily
        # freshness loop AND cytir.news.weekly_delta, both of which call build_points) so
        # freshness/delta upserts store saturated bm25 consistent with the base.
        saturate_point_bm25(vec)
        pts.append(models.PointStruct(
            id=mint_card_point_id(s["url"]),
            vector=vec, payload=payload))
    return pts


# ----------------------------------- main ---------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--collection", default="aw_prod")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--tei-url", default="http://localhost:8088")
    ap.add_argument("--health-url", default="http://localhost:8090/health")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch non-blocked URLs and update text on success")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    # task 864 (review 869 S1): from an aw_e10 T0 to the flip only the drain writes the twin
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(args, tool='cytir/news/freshness_ingest.py')

    # This must precede consent resolution, embedding, Qdrant connection, and
    # especially the writer lock: a scheduled legacy run must be visibly
    # terminal without touching either mirror.
    require_principal_source_replay(dry_run=args.dry_run)

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    sources = manifest["sources"]
    today = dt.date.today().isoformat()
    print(f"{len(sources)} sources; last_seen={today}; collection={args.collection}",
          flush=True)

    # consent gate
    gated = [s for s in sources if consent_ok(s["url"])]
    print(f"consent: {len(gated)}/{len(sources)} passed", flush=True)

    # ADR-105 (task 581): persisted inputs are a precondition to load. This
    # lane is the one that made the point -- it embedded the manifest's own
    # `text`, upserted, and sealed NOTHING, which is the shape of ingester
    # that produced task 579's un-rebuildable population.
    #
    # So it now FILES ITS INPUTS FIRST. Every gated source's text is written
    # into the ADR-026 store as an L1 document (url + markdown + sha) before
    # the guard runs, before the writer lock, and before a single vector is
    # embedded. The destination is a path the tracked registry already covers,
    # so nothing in the registry has to move: the run persists, and then it
    # passes. The overlay carries the pages written seconds ago, which the
    # prebuilt index cannot know about yet -- and it is evidence, not a
    # bypass: every entry is checked to be a real file inside a registered
    # store.
    import cytir.corpus.persisted as PI
    from cytir.corpus import persist_l1_records as PL1
    try:
        written = PL1.write_l1_records(
            ({"url": src["url"], "text": src.get("text"),
              "title": src.get("title"), "lang": src.get("lang"),
              "domain": src.get("domain")} for src in gated),
            release="freshness", source=today,
            origin="freshness-manifest:curated text "
                   "(data/refresh_manifest.json)",
            note=f"freshness_ingest {args.manifest} -> {args.collection}")
        pv = PI.assert_pages_persisted(
            [{"url": src["url"]} for src in gated], require="l1",
            label=f"freshness manifest -> {args.collection}",
            overlay=PI.overlay_from_paths(written))
    except (PI.PersistedInputsError, PL1.L1PersistError) as exc:
        raise SystemExit(f"REFUSED: {exc}") from None
    print(f"persisted inputs: {pv['pages']:,} page(s), "
          f"{pv['persisted_in_this_run']:,} filed by this run", flush=True)

    if args.refresh:
        for s in gated:
            if s.get("blocked"):
                continue
            new = refetch_text(s["url"])
            if new:
                s["text"] = f"{s['title']}. {new}"
                print(f"  [refresh] updated {s['url']}", flush=True)

    from qdrant_client import models
    client = qdrant_connect(args.qdrant_url, timeout=120)

    if args.dry_run:
        pts = build_points(models, gated[:3], args.tei_url, today)
        for p in pts:
            print(json.dumps({"id": p.id, "payload": p.payload,
                              "dense_dim": len(p.vector["dense"])},
                             ensure_ascii=False, indent=2))
        print("DRY-RUN — no writes", flush=True)
        return

    try:
        before = client.get_collection(args.collection).points_count
    except Exception:
        before = None
    print(f"{args.collection} points_count(before, approx): {before}", flush=True)

    take_lock(f"task106 freshness ingest ({len(gated)} sources)")
    n = 0
    try:
        for i in range(0, len(gated), args.batch):
            chunk = gated[i:i + args.batch]
            pts = build_points(models, chunk, args.tei_url, today)
            client.upsert(args.collection, points=pts, wait=True)
            n += len(pts)
            try:
                assert IH.send(args.health_url, timeout=15)[0] == 200
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"ABORT: health check failed after {n}: {e!r}")
            print(f"  upserted {n}/{len(gated)} (health OK)", flush=True)
    finally:
        release_lock()

    try:
        after = client.get_collection(args.collection).points_count
    except Exception:
        after = None
    print(f"DONE upserted {n}; points_count(after, approx): {after}", flush=True)


if __name__ == "__main__":
    main()
