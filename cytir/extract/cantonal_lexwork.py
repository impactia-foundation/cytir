"""LexWork (sitrox) cantonal law -> article-level passages (`scratch_ch_cantonal_law_exact`, portal `lexwork`).

Moved out of `ingest_lexwork.py` (task 1108, S7) so the source family
(`cytir.sources.families.ch_cantonal_law`) and the legacy CLI share one builder. `_txt` and
`parse_xhtml_tol` are the driver's own, UNCHANGED. `law_rows` is the pure tail of the driver's
`process_law` (everything after the HTTP GET), cut out so it can run on a captured response.

Task 1488: the LIVE half moved here too (one home per family): the `CANTONS` host table, the global
and per-host throttle, `_get` with retries, `lightweight_index`, `process_law`, `canton_worker`, the
done-file resume and the CLI `main` (bodies as the driver had them). The family adapter
(`cytir.sources.families.ch_cantonal_law`) uses `CANTONS`, `lightweight_index` and `_get` for its live
path; the old `ingest_lexwork.py` driver is a forwarding shim to this module.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client
from cytir.consent.block import ConsentBlocked
from bs4 import BeautifulSoup

from cytir.core.common import Passage, extractive_snippet


def _txt(node):
    if node is None:
        return ""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def parse_xhtml_tol(html, law_title):
    """Split LexWork xhtml_tol into (anchor, article_label, body) passages.

    Structure: a flat sequence under div.document — div.article (number+title)
    followed by sibling div.paragraph / table.enumeration_item blocks until the
    next div.article. Paragraphs before the first article -> a 'preamble' chunk.
    Long articles are windowed (~240 words) so tails aren't lost.
    """
    soup = BeautifulSoup(html, "lxml")
    root = soup.find("div", class_="document") or soup
    arts, cur = [], {"num": None, "title": "", "parts": []}

    def classes(el):
        return el.get("class") or []

    for el in root.find_all(recursive=False):
        cls = classes(el)
        if "article" in cls:
            if cur["parts"]:
                arts.append(cur)
            num = _txt(el.select_one(".article_number .number")) or \
                _txt(el.select_one(".article_number"))
            sym = _txt(el.select_one(".article_symbol"))
            if sym and sym in num:
                num = num.replace(sym, "").strip()
            title = _txt(el.select_one(".article_title .title_text")) or \
                _txt(el.select_one(".article_title"))
            cur = {"num": num, "title": title, "parts": []}
        elif ("paragraph" in cls and "paragraph_post" not in cls) or \
                "enumeration_item" in cls:
            t = _txt(el)
            if t:
                cur["parts"].append(t)
    if cur["parts"]:
        arts.append(cur)

    out, seen = [], {}
    for a in arts:
        num = (a["num"] or "").strip()
        label = (f"Art. {num}" if num else "Präambel")
        if a["title"]:
            label += f" — {a['title']}"
        body = " ".join(a["parts"]).strip()
        if not body:
            continue
        base = "art-" + re.sub(r"[^0-9a-zA-Z]+", "-", num).strip("-").lower() \
            if num else "preamble"
        words = body.split()
        if len(words) <= 300:
            chunks = [body]
        else:
            chunks = [" ".join(words[i:i + 240]) for i in range(0, len(words), 240)]
        for ci, chunk in enumerate(chunks):
            anc = base if len(chunks) == 1 else f"{base}-c{ci+1}"
            n = seen.get(anc, 0); seen[anc] = n + 1
            if n:
                anc = f"{anc}-{n}"
            out.append((anc, label, chunk))
        if len(out) >= 800:            # over-chunk safety cap per law
            break
    return out


def select_html(tol: dict) -> str:
    """The xhtml of the version the driver reads: selected, else current ('' when neither has one)."""
    sv = tol.get("selected_version") or tol.get("current_version") or {}
    return sv.get("xhtml_tol") or ""


def law_rows(host, lang, canton, sysno, idx_title, tol) -> list[dict]:
    """The rows `process_law` appended for one `text_of_law` object (the driver's own lines)."""
    html = select_html(tol)
    law_title = tol.get("title") or idx_title or sysno
    last_seen = (tol.get("text_of_law_dates_str") or tol.get("enactment")
                 or tol.get("date_of_decision") or None)
    url = f"https://{host}/app/{lang}/texts_of_law/{sysno}"
    out_rows: list[dict] = []
    if not html:
        return out_rows
    passages = parse_xhtml_tol(html, law_title)
    if not passages:
        # short/degenerate doc — one whole-doc passage from any text
        whole = _txt(BeautifulSoup(html, "lxml"))
        if len(whole.split()) >= 8:
            passages = [("full", law_title, whole)]
    for anc, label, body in passages:
        title = f"{law_title} — {label}"[:300]
        text = f"{law_title} · {sysno}\n{label}\n{body}"
        p = Passage(
            url=url, title=title, snippet=extractive_snippet(body, 25),
            text=text, lang=lang, domain=host, tier="A",
            source_type="official", trust_tier=1.0, anchor=anc,
            last_seen=last_seen,
        )
        row = {**p.payload(), "text": p.text,
               "country": "CH", "provenance": "ch_cantonal_law", "canton": canton}
        out_rows.append(row)
    return out_rows


# --- live acquisition (moved from the old driver, task 1488) ---
UA = "AuthoritativeWeb/1.0 (+https://websear.ch; grounding index for AI agents)"
HDR = {"User-Agent": UA, "Accept": "application/json"}

# (code, host, [langs]) — Romansh (rm) intentionally omitted (project scope, 2026-06-21).
# Canton-branded official host chosen where it exists, else the sitrox clex.ch host.
CANTONS = [
    ("BS", "www.gesetzessammlung.bs.ch", ["de"]),
    ("AG", "gesetzessammlungen.ag.ch",   ["de"]),
    ("LU", "srl.lu.ch",                  ["de"]),
    ("SG", "www.gesetzessammlung.sg.ch", ["de"]),
    ("ZG", "bgs.zg.ch",                  ["de"]),
    ("SO", "bgs.so.ch",                  ["de"]),
    ("GL", "gesetze.gl.ch",              ["de"]),
    ("BL", "bl.clex.ch",                 ["de"]),
    ("AR", "ar.clex.ch",                 ["de"]),
    ("AI", "ai.clex.ch",                 ["de"]),
    ("SH", "rechtsbuch.sh.ch",           ["de"]),
    ("NW", "nw.clex.ch",                 ["de"]),
    ("OW", "ow.clex.ch",                 ["de"]),
    ("UR", "rechtsbuch.ur.ch",                 ["de"]),
    ("TG", "tg.clex.ch",                 ["de"]),
    ("GR", "gr.clex.ch",                 ["de", "it"]),
    ("BE", "be.clex.ch",                 ["de", "fr"]),
    ("FR", "bdlf.fr.ch",                 ["de", "fr"]),
    ("VS", "lex.vs.ch",                  ["de", "fr"]),
]

_rate_lock = {}          # host -> (threading.Lock, [last_ts])
_state_lock = threading.Lock()
# All sitrox LexWork hosts (clex.ch + canton-branded) share back-end infra, so a
# GLOBAL aggregate limiter is the binding politeness constraint (8-way burst got
# connection-refused). Per-host floor still applies on top.
_global = {"lock": threading.Lock(), "last": 0.0, "interval": 0.25}


def _throttle(host, min_interval=0.9):
    g = _global
    with g["lock"]:
        dt = time.time() - g["last"]
        if dt < g["interval"]:
            time.sleep(g["interval"] - dt)
        g["last"] = time.time()
    lk, last = _rate_lock.setdefault(host, (threading.Lock(), [0.0]))
    with lk:
        dt = time.time() - last[0]
        if dt < min_interval:
            time.sleep(min_interval - dt)
        last[0] = time.time()


_CLIENT = None


def _http():
    """The module's lazy, consent-guarded client (same headers/timeout/redirects as before)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = make_client(HttpSpec(
            timeout=45, follow_redirects=True,
            headers={"user-agent": UA, "accept": "application/json"}))
    return _CLIENT


def _get(host, path, tries=5):
    url = f"https://{host}{path}"
    for i in range(tries):
        _throttle(host)
        try:
            r = _http().get(url)
        except ConsentBlocked:
            raise                        # a refusal is final: never retried
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1)); continue
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(int(r.headers.get("Retry-After", 3 * (i + 1))))
            continue
        r.raise_for_status()
    raise RuntimeError(f"{url} failed after {tries} tries (last {r.status_code})")


def lightweight_index(host, lang):
    """Return list of in-force laws: [{systematic_number, title, id, ...}]."""
    d = _get(host, f"/api/{lang}/texts_of_law/lightweight_index").json()
    laws = []
    for cat_id, items in d.items():
        for it in items:
            if not it.get("abrogated"):
                laws.append(it)
    # dedup by sysno (keep first)
    seen, out = set(), []
    for it in laws:
        s = it.get("systematic_number")
        if s and s not in seen:
            seen.add(s); out.append(it)
    return out


def process_law(host, lang, canton, sysno, idx_title, out_rows):
    r = _get(host, f"/api/{lang}/texts_of_law/{sysno}")
    tol = r.json().get("text_of_law", {})
    rows = law_rows(host, lang, canton, sysno, idx_title, tol)
    out_rows.extend(rows)
    return len(rows)


def canton_worker(canton, host, langs, done, out_path, cap):
    added_rows, added = [], 0
    for lang in langs:
        try:
            laws = lightweight_index(host, lang)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {canton}/{lang} index failed: {e}", flush=True)
            continue
        if cap:
            laws = laws[:cap]
        for i, it in enumerate(laws):
            sysno = it.get("systematic_number")
            key = f"{canton}|{lang}|{sysno}"
            if key in done:
                continue
            try:
                n = process_law(host, lang, canton, sysno,
                                 it.get("title", ""), added_rows)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {key} failed: {e}", flush=True)
                continue
            added += n
            with _state_lock:
                with open(out_path, "a") as f:
                    for r in added_rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                with open(out_path + ".done", "a") as f:
                    f.write(key + "\n")
                added_rows.clear()
            if (i + 1) % 50 == 0:
                print(f"  {canton}/{lang} {i+1}/{len(laws)} laws, "
                      f"{added} passages", flush=True)
    print(f"[{canton}] DONE +{added} passages", flush=True)
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=0, help="max laws per canton/lang (test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-sep canton codes")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out + ".done"):
        with open(args.out + ".done") as f:
            done = {ln.strip() for ln in f if ln.strip()}
        print(f"[resume] {len(done)} laws already done", flush=True)

    cantons = CANTONS
    if args.only:
        keep = set(args.only.split(","))
        cantons = [c for c in CANTONS if c[0] in keep]

    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(canton_worker, c, h, L, done, args.out, args.cap): c
                for (c, h, L) in cantons}
        for fu in as_completed(futs):
            total += fu.result()
    print(f"\nTOTAL passages appended: {total}", flush=True)


if __name__ == "__main__":
    main()
