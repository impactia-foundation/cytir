#!/usr/bin/env python3
"""Ingest German federal law from gesetze-im-internet.de — the BMJ's OFFICIAL
bulk XML (gii-toc.xml lists every law; each law = /<abbr>/xml.zip). Dumps-first,
no scraping. -> Passage JSONL (one per norm/paragraph).

  cyt source run gesetze   (`python -m` no longer starts this module, task 2943)
  (the old flat script name is a forwarding shim; the family adapter is
  `cytir.sources.families.gesetze`, task 1400.)

# TODO(linkgraph): German norms reference other Gesetze/§§ (e.g. "§ 123 BGB",
# "i.V.m. … Abs."). Parse these into the shared link-graph store (ADR-020) as a
# legal citation graph, same {src, links} shape as the web endorsement
# extractor. Deferred.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import httpx
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client
from lxml import etree

from cytir.core.common import Passage, low_quality
from cytir.core.user_agent import user_agent  # task 432: one home for the crawler identity

TOC = "https://www.gesetze-im-internet.de/gii-toc.xml"
BASE = "https://www.gesetze-im-internet.de"
def _headers() -> dict:
    """Built at request time, not import time: importing the module must not require a crawler identity (cut test, task 1400)."""
    return {"User-Agent": user_agent()}
_TAG = re.compile(r"<[^>]+>"); _WS = re.compile(r"\s+")

# Repealed/placeholder norm bodies — carry no statutory content and must stay
# dropped even under the loosened short-norm rule below.
_PLACEHOLDER = re.compile(
    r"^[\s\-–—.]*$"
    r"|^\(?\s*(weggefallen|aufgehoben|gegenstandslos|entfallen|entfällt|unbesetzt"
    r"|zukünftig in kraft[^)]*)\s*\)?\s*\.?$",
    re.IGNORECASE)

# Politeness: fixed pause after every law fetch (the host throttle-blocked an
# IP once already; with 2 workers this caps us well under ~4 req/s).
POLITE_DELAY = 0.5


def clean(node) -> str:
    if node is None:
        return ""
    raw = etree.tostring(node, encoding="unicode", method="xml")
    return _WS.sub(" ", _TAG.sub(" ", raw)).strip()


def norm_text(norm) -> str:
    """Body text of a norm. Primary path is textdaten/text/Content, but some
    norms put the Content elsewhere (e.g. directly under textdaten/text, or a
    fussnoten/Content). Fall back to any <Content> under the norm so norms with
    an alternate layout are not silently dropped."""
    node = norm.find(".//textdaten/text/Content")
    text = clean(node)
    if len(text) >= 1:
        return text
    # alternate layouts: first <Content> anywhere with real body text
    for c in norm.iter("Content"):
        t = clean(c)
        if t:
            return t
    return ""


def norm_label(norm, enbez: str) -> str:
    """A human label for the norm. Prefer the Einzelnorm-Bezeichnung (e.g.
    '§ 433'); otherwise fall back to the norm's own title or its
    structural heading (Gliederung), so norms without an <enbez> still get a
    meaningful, distinct title instead of being discarded."""
    if enbez:
        return enbez
    titel = (norm.findtext(".//titel") or "").strip()
    if titel:
        return titel
    gbez = (norm.findtext(".//gliederungseinheit/gliederungsbez") or "").strip()
    gtit = (norm.findtext(".//gliederungseinheit/gliederungstitel") or "").strip()
    label = " ".join(p for p in (gbez, gtit) if p).strip()
    return label


def toc_links(content: bytes) -> list[str]:
    """The law ZIP URLs listed by one `gii-toc.xml` body."""
    root = etree.fromstring(content)
    # each <item><link>https://www.gesetze-im-internet.de/<abbr>/xml.zip</link>
    return [el.text for el in root.iter("link") if el.text and el.text.endswith(".zip")]


def law_links(client) -> list[str]:
    return toc_links(client.get(TOC).content)


# Explicit per-phase timeouts. A single shared httpx.Client used from many
# threads can wedge its connection pool when the server slow-throttles under
# concurrency, so each worker thread gets its OWN client (thread-local) and a
# hard read timeout — a stalled socket fails fast and is retried instead of
# hanging the whole run forever.
_TIMEOUT = httpx.Timeout(connect=15.0, read=30.0, write=30.0, pool=30.0)
_TLOCAL = None  # set in main() to a threading.local()


def _client():
    import threading
    global _TLOCAL
    if _TLOCAL is None:
        _TLOCAL = threading.local()
    c = getattr(_TLOCAL, "client", None)
    if c is None:
        c = make_client(HttpSpec(timeout=_TIMEOUT, headers={"user-agent": _headers()["User-Agent"]},
                             follow_redirects=True))
        _TLOCAL.client = c
    return c


def fetch_law_zip(client, zip_url):
    """GET one law ZIP with the driver's retry/back-off and politeness pause.

    Returns the final response (any status) or None when every attempt raised. The one
    place the fetch policy lives: the driver and the family adapter both call it."""
    import time as _t
    r = None
    for attempt in range(4):
        try:
            r = client.get(zip_url)
            if r.status_code == 200:
                break
        except Exception:  # noqa: BLE001
            r = None
        _t.sleep(1.5 * (attempt + 1))   # back off on throttle/error
    _t.sleep(POLITE_DELAY)
    return r


def ingest_law(zip_url, out, lock):
    client = _client()
    r = fetch_law_zip(client, zip_url)
    if r is None or r.status_code != 200:
        return -1   # fetch failure (throttle/network) — distinct from "0 norms"
    try:
        rows = parse_law_zip(zip_url, r.content)
    except Exception:  # noqa: BLE001
        return -1
    if rows:
        with lock:
            for p in rows:
                out.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
            out.flush()
    return len(rows)


def parse_law_zip(zip_url: str, body: bytes) -> list[Passage]:
    """Parse one preserved official XML ZIP into source-native norm rows.

    Both the live downloader and the L0 replay adapter call this function, so
    statutory boundaries, URLs, filtering, text composition, and point IDs are
    one mirror contract rather than parallel implementations.
    """
    abbr = zip_url.rsplit("/", 2)[-2]
    zf = zipfile.ZipFile(io.BytesIO(body))
    xmlnames = [name for name in zf.namelist() if name.endswith(".xml")]
    if not xmlnames:
        return []
    if len(xmlnames) != 1:
        raise ValueError(f"official law ZIP has {len(xmlnames)} XML members: {zip_url}")
    root = etree.fromstring(zf.read(xmlnames[0]))
    law_title = (root.findtext(".//norm[1]//langue") or root.findtext(".//norm[1]//titel")
                 or abbr)
    rows: list[Passage] = []
    seen_urls: set[str] = set()
    for idx, norm in enumerate(root.iter("norm")):
        enbez = (norm.findtext(".//enbez") or "").strip()      # e.g. "§ 433"
        text = norm_text(norm)
        # Repealed placeholders ("-", "(weggefallen)") carry no content — drop.
        if not text or _PLACEHOLDER.match(text):
            continue
        # Loosened filter: emit any norm that has real body text, even when it
        # has no <enbez>. Real statutory ONE-LINERS exist (GG Art 31
        # "Bundesrecht bricht Landesrecht." = 31 chars; GG Art 102, BGB § 857,
        # ZPO § 50 …), so norms WITH an <enbez> only need a tiny floor;
        # enbez-less norms keep the 40-char floor to suppress structural junk.
        # low_quality() still strips boilerplate.
        if len(text) < (15 if enbez else 40) or low_quality(text):
            continue
        label = norm_label(norm, enbez)
        # Drop pure table-of-contents norms: they carry no statutory content,
        # just a list of section headings, and are huge low-signal noise.
        if label in ("Inhaltsübersicht", "Inhaltsverzeichnis"):
            continue
        # Drop gii "(XXXX)" placeholder norms (renumbered/reserved markers):
        # they have no resolvable page and only ever 404 (task 143).
        if label.startswith("(XXXX)"):
            continue
        title = f"{law_title} — {label}" if label else law_title
        # Canonical gesetze-im-internet.de page filename: lowercase the norm
        # LABEL (the enbez — "§ 433", "Art 5", "Anlage 1" — or the structural
        # heading / "Eingangsformel" / "Schlußformel") and map EACH
        # non-alphanumeric char to "_". Verified live (task 143): "§ 433" ->
        # "__433" (§+space => two underscores), "Art 5" -> "art_5",
        # "Eingangsformel" -> "eingangsformel", "Anlage 1" -> "anlage_1".
        # NOTE: do NOT `.replace("§","__")` — that double-counts the space and
        # yields the broken "/___N.html" (three underscores) 404 artifact that
        # task 141 found across the whole German corpus. Label-less norms have
        # no per-norm page -> point at the whole-act index page.
        label_for_url = enbez or label
        if label_for_url:
            frag = re.sub(r"[^a-z0-9]", "_", label_for_url.lower())
            url = f"{BASE}/{abbr}/{frag}.html"
            if url in seen_urls:
                url = f"{BASE}/{abbr}/{frag}_{idx}.html"
        else:
            url = f"{BASE}/{abbr}/index.html"
            if url in seen_urls:
                url = f"{BASE}/{abbr}/index_{idx}.html"
        seen_urls.add(url)
        rows.append(Passage(url=url,
                            title=title[:300], snippet=" ".join(text.split()[:30]),
                            text=f"{title}\n{text}", lang="de",
                            domain="gesetze-im-internet.de", tier="A",
                            source_type="official", trust_tier=1.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    import threading
    global _TLOCAL
    _TLOCAL = threading.local()
    links = law_links(_client())
    print(f"{len(links)} German laws to fetch", flush=True)
    lock = threading.Lock()
    total = [0]
    done = [0]
    laws_ok = [0]
    empty = [0]
    failed: list[str] = []
    with open(args.out, "w", encoding="utf-8") as f:
        def work(u):
            k = ingest_law(u, f, lock)
            if k < 0:
                with lock:
                    failed.append(u)
            elif k == 0:
                empty[0] += 1
            else:
                total[0] += k
                laws_ok[0] += 1
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"  {done[0]}/{len(links)} laws, {total[0]} norms, "
                      f"{len(failed)} fetch-fail, {empty[0]} empty", flush=True)
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(work, links))
        # Second chance for throttle victims: the 2026-06-28 run lost 122
        # entire laws to transient fetch failures (alphabetical bursts).
        # Retry them sequentially and gently at the end, once.
        if failed:
            import time as _t
            print(f"retrying {len(failed)} fetch-failed laws sequentially...",
                  flush=True)
            still_failed = []
            for u in failed:
                _t.sleep(2.0)
                k = ingest_law(u, f, lock)
                if k < 0:
                    still_failed.append(u)
                elif k == 0:
                    empty[0] += 1
                else:
                    total[0] += k
                    laws_ok[0] += 1
            failed = still_failed
    # Machine-greppable summary so undercounts are visible in future runs.
    # Sanity floor: a healthy full run yields ~6k laws_parsed / ~100k+ passages.
    print(f"SUMMARY laws_found={len(links)} laws_parsed={laws_ok[0]} "
          f"passages_out={total[0]} fetch_errors={len(failed)} "
          f"empty_laws={empty[0]} out={args.out}", flush=True)
    if failed:
        print("  unrecovered fetch failures: "
              + " ".join(u.rsplit("/", 2)[-2] for u in failed[:50]))
