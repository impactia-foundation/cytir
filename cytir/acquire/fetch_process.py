"""Fetch + extract + chunk for the AW P0 build.

P0 fetch methods (spec §8):
  - crawl     : polite httpx GET of seed URLs (+ shallow same-domain expansion)
  - wikipedia : per-article via the Wikipedia action API (clean extract +
                canonical URL). A curated slice (e.g. the 26 cantons +
                institutions in de/fr/it/en). Full Wikipedia via dumps = P1.
  - cc        : Common Crawl columnar-index range-GET (implemented in cc.py;
                optional at P0 — direct crawl covers the tiny domain set).
"""
from __future__ import annotations

import os
import re
import sys
import time
from typing import Iterator
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

from cytir.embed.model_assets import LID_176 as _LID_ASSET
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

UA = "authoritative-web-bot/0.1 (+https://websear.ch; grounding index for AI agents)"
_SPEC = HttpSpec(timeout=30.0, follow_redirects=True, headers={"user-agent": UA})
_CLIENT: httpx.Client | None = None


def _http() -> httpx.Client:
    """The consent-guarded fetch client, built on first use (spec step 4)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = make_client(_SPEC)
    return _CLIENT


def guess_lang(url: str, default: str) -> str:
    """Cheap language hint from URL path (fedlex/admin use /de/ /fr/ /it/ /en/).

    WARNING (task 166): this is a URL-path-only heuristic that DEFAULTS to
    `default` (usually 'en') whenever the path carries no /de//fr//it//en/
    marker. It never looks at content, so a CC host-capture of a locale mirror
    (e.g. blogs.nvidia.co.jp, whose path has no marker) is silently stamped
    'en' and slips past the EN/DE/FR/IT scope filter. Use `detect_lang` below
    for the authoritative label + scope gate; keep this only as the fallback."""
    u = url.rstrip("/")
    for code in ("de", "fr", "it", "en"):
        if f"/{code}/" in url or u.endswith(f"/{code}") or f"/{code}-" in url or f".{code}." in url:
            return code
    return default


# --- content-based language scope gate (task 166) --------------------------
# AW scope is strictly EN/DE/FR/IT (Romansh dropped 2026-06-21). The URL
# heuristic above cannot enforce it (it defaults to en). detect_lang() reads
# the extracted text via fasttext lid.176 so out-of-scope pages are labeled
# truthfully and can be dropped at the ingest gate (_content_passages).
#
# task 236 — lingua evaluated as a fasttext replacement and REJECTED. On a
# 1120-snippet Wikipedia set (EN/DE/FR/IT + OOS es/pt/zh/nl, half short-text),
# fasttext@prob>=0.85 beat lingua on every accuracy metric that matters to
# this gate: label acc 98.0% vs 89.5% (SHORT 96.1% vs 82.9% — opposite of the
# "lingua wins on short text" hypothesis), and critically 0.0% vs 9.8%
# in-scope FALSE-DROP (16.4% short) — lingua would silently discard ~1-in-6
# short EN/DE/FR/IT pages, violating this gate's precision-favoring design.
# lingua only won on OOS leak (1.2% vs 13%), the cheap error the gate tolerates.
# fasttext also ~41x faster (9023 vs 218 snip/s). See task 236 (lingua vs fasttext language-detection bake-off).
# NOTE: fasttext needs numpy<2 to predict (else predict() raises -> gate OFF,
# the task-233 failure). Run the HTML-extraction pipeline in the DEDICATED
# lang-gate venv (the build box: <langgate-venv>, numpy<2 + fasttext-wheel;
# see deploy/langgate/requirements.txt). The main venv is on numpy>=2
# (docling/opencv require it) and CANNOT run the gate — task 238.
LANG_SCOPE = ("en", "de", "fr", "it")
# Precision-favoring thresholds — MUST match the task-166 audit so the gate
# drops exactly the class the audit measured (no borderline target-lang loss).
_LID_MIN_PROB = 0.85
_LID_MIN_LEN = 50
# Model path: `model_assets.LID_176` is the ONE home for this file's location,
# canonical URL and sha256 (task 432 — before that the path below was the only
# mention of the model anywhere in the repo: no URL, no digest, no fetch step,
# so a cold builder simply could not obtain it). AW_LID_MODEL still wins
# absolutely; the implicit search order keeps <models-dir> FIRST so every
# existing build-box invocation resolves to the byte it always did.
_LID_MODEL_PATH, _LID_MODEL_PRESENT = _LID_ASSET.resolve()

# --- gate health / fail-loud (task 238) ------------------------------------
# The task-233/238 bug is a SILENT inactive gate: fasttext.load_model() succeeds
# on numpy>=2 but predict() then raises, so the old code caught the raise per
# call and silently fell back to a URL heuristic that stamps everything 'en' —
# multilingual pages LEAK into the index mislabeled, and nobody notices.
# Fix: probe the gate ONCE (load + a self-test predict, which is what actually
# breaks under numpy>=2), and refuse to run degraded in SILENCE. Default is
# HARD-FAIL (raise LangGateInactive) so a production ingest aborts before it can
# contaminate the corpus; set AW_LANGGATE_OPTIONAL=1 to downgrade to a screaming
# once-per-process banner + a recorded `gate_inactive` flag (for English-dominant
# sources / offline tooling / tests that accept the URL-heuristic fallback).


class LangGateInactive(RuntimeError):
    """The fastText EN/DE/FR/IT scope gate cannot run (model missing, or
    predict() raises under numpy>=2). Raised instead of silently mislabeling
    out-of-scope pages as the default language (the task-233 corpus leak)."""


# _GATE["active"]: None=unprobed, True/False after probe.
# _GATE["reason"]: human-readable cause when inactive.
# GATE_INACTIVE_REASON (module attr): None when active/unprobed, else the reason
# string — ingest/wave code stamps langgate_manifest_field() onto the wave
# manifest so a degraded run is auditable and re-derivable.
_GATE = {"active": None, "reason": None, "model": None, "screamed": False}
GATE_INACTIVE_REASON: str | None = None
_SELFTEST = "Die Schweiz ist ein Land in Europa mit vier Landessprachen."


def _probe_langgate() -> None:
    """Load the model AND run one self-test predict, then latch _GATE state.
    Load succeeding is NOT enough — predict() is what raises under numpy>=2, so
    we must actually predict to know the gate works."""
    if _GATE["active"] is not None:
        return
    try:
        import numpy
    except ImportError as e:
        # A bare ModuleNotFoundError traceback out of a probe is loud but not
        # actionable. Latch it as an ordinary inactive-gate reason so the caller
        # gets the same LangGateInactive with the same fix instructions.
        _GATE.update(active=False, model=None,
                     reason=f"numpy is not installed ({e}); the lang-gate venv "
                            f"needs numpy<2 + fasttext-wheel "
                            f"(deploy/langgate/requirements.txt)")
        return
    if not os.path.exists(_LID_MODEL_PATH):
        _GATE.update(active=False, model=None,
                     reason=f"lid.176 model not found at {_LID_MODEL_PATH}; "
                            f"fetch it with `python3 -m cytir.embed.fetch_models`")
        return
    try:
        # A present-but-wrong file is worse than a missing one: it loads, it
        # predicts, and it produces language labels nobody can reproduce. Verify
        # the pinned digest before trusting the gate.
        _LID_ASSET.verify(_LID_MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        _GATE.update(active=False, model=None,
                     reason=f"lid.176 model at {_LID_MODEL_PATH} is not the pinned "
                            f"artifact ({e}); re-fetch with "
                            f"`python3 -m cytir.embed.fetch_models --force`")
        return
    try:
        import fasttext
        model = fasttext.load_model(_LID_MODEL_PATH)
        lab, prob = model.predict(_SELFTEST, k=1)  # the numpy>=2 tripwire
        assert lab and float(prob[0]) > 0.0
        _GATE.update(active=True, model=model, reason=None)
    except Exception as e:  # noqa: BLE001
        _GATE.update(active=False, model=None,
                     reason=f"fasttext predict failed on numpy {numpy.__version__} "
                            f"({type(e).__name__}: {e}); need numpy<2 + fasttext-wheel")


def langgate_status() -> tuple[bool, str | None]:
    """(active, reason). Probes on first call. reason is None when active."""
    _probe_langgate()
    return bool(_GATE["active"]), _GATE["reason"]


def _langgate_banner(reason: str) -> str:
    bar = "!" * 72
    return (f"\n{bar}\n"
            f"!! AW LANGUAGE SCOPE-GATE INACTIVE — fastText EN/DE/FR/IT gate is OFF\n"
            f"!! reason : {reason}\n"
            f"!! model  : {_LID_MODEL_PATH}\n"
            f"!! effect : content-based language detection is DISABLED; pages fall\n"
            f"!!          back to a URL heuristic that DEFAULTS to the source lang\n"
            f"!!          (usually 'en') — out-of-scope pages LEAK mislabeled.\n"
            f"!! fix    : run the extraction pipeline in the lang-gate venv\n"
            f"!!          (the operator's langgate interpreter) — numpy<2 + fasttext-wheel\n"
            f"!!          (deploy/langgate/requirements.txt), and fetch the model:\n"
            f"{_LID_ASSET.howto()}\n"
            f"!! NOTE   : running degraded because AW_LANGGATE_OPTIONAL=1. Unset it\n"
            f"!!          to HARD-FAIL instead of leaking. This wave is flagged\n"
            f"!!          gate_inactive.\n"
            f"{bar}\n")


def assert_langgate_active(context: str = "") -> dict:
    """Fail-loud gate check for ingest startup / callers.

    active  -> returns {"active": True, ...}.
    inactive:
      - default              -> raises LangGateInactive (loud, unmissable abort).
      - AW_LANGGATE_OPTIONAL=1 -> prints the screaming banner ONCE, records
        GATE_INACTIVE_REASON, returns {"active": False, "reason": ...} so the
        caller keeps running (URL fallback) but the degraded state is recorded.
    """
    global GATE_INACTIVE_REASON
    active, reason = langgate_status()
    if active:
        return {"active": True, "reason": None, "model_path": _LID_MODEL_PATH}
    ctx = f" [{context}]" if context else ""
    if os.environ.get("AW_LANGGATE_OPTIONAL") == "1":
        GATE_INACTIVE_REASON = reason
        if not _GATE["screamed"]:
            _GATE["screamed"] = True
            print(_langgate_banner(f"{reason}{ctx}"), file=sys.stderr, flush=True)
        return {"active": False, "reason": reason, "model_path": _LID_MODEL_PATH}
    raise LangGateInactive(
        f"Language scope-gate INACTIVE{ctx}: {reason}.\n"
        f"Fix, in order:\n"
        f"  1. get the model (126 MB, checksum-verified):\n{_LID_ASSET.howto()}\n"
        f"  2. run in the lang-gate venv (numpy<2 + fasttext-wheel — fasttext's\n"
        f"     predict() raises on numpy>=2; see deploy/langgate/requirements.txt).\n"
        f"To run degraded ON PURPOSE (English-dominant source / offline tooling), "
        f"set AW_LANGGATE_OPTIONAL=1 — the run is then stamped gate_inactive.")


def langgate_manifest_field() -> dict:
    """Field to merge into a wave manifest so a degraded run is auditable.
    {'lang_gate': 'active'} normally; {'lang_gate': 'inactive', 'lang_gate_reason': ...}
    when the process ran with AW_LANGGATE_OPTIONAL=1 and the gate was off."""
    active, reason = langgate_status()
    if active:
        return {"lang_gate": "active"}
    return {"lang_gate": "inactive", "lang_gate_reason": reason}


def _lid_model():
    """Back-compat accessor: the probed model when the gate is active, else None."""
    _probe_langgate()
    return _GATE["model"]


def detect_lang(text: str, url: str, default: str) -> tuple[str, bool]:
    """Content-based language label + confidence flag.

    Returns (lang, confident). When the detector is confident (top-1 prob
    >= _LID_MIN_PROB on >= _LID_MIN_LEN chars) `lang` is the detected language,
    which MAY be outside LANG_SCOPE — the caller gates on that. Otherwise falls
    back to the URL-path heuristic with confident=False (never drops on a
    low-confidence guess — favors precision / keeps borderline target text).

    If the gate is INACTIVE, this raises LangGateInactive (default) or, under
    AW_LANGGATE_OPTIONAL=1, screams once + records the flag and falls back —
    NEVER silent (task 238).

    Thin wrapper over `detect_lang_scored` -- this function's signature is
    the audit-pinned interface (task 166/236/238) every other caller in the
    pipeline already uses; `detect_lang_scored` holds the one real
    implementation so a caller that also needs the raw score (task 652's
    lang-presence waiver receipt) is not a second detector."""
    result = detect_lang_scored(text, url, default)
    return result["lang"], result["confident"]


def detect_lang_scored(text: str, url: str, default: str) -> dict:
    """Same gate, same thresholds, same fallback as `detect_lang` -- but
    also returns the raw top-1 label and probability the threshold check
    was applied to, even when that fell under `_LID_MIN_PROB` or the text
    was under `_LID_MIN_LEN`. Diagnostic use only (task 652: which score a
    row that failed the confidence check actually got, for a waiver
    receipt and the investigation into why); every other caller keeps
    using `detect_lang`, which delegates here and discards the score.

    Returns {"lang": str, "confident": bool, "raw_label": str|None,
    "raw_prob": float|None, "text_chars": int} -- `raw_label`/`raw_prob`
    are None when `predict()` was never attempted (gate inactive, no
    text, or text under `_LID_MIN_LEN`)."""
    status = assert_langgate_active(context="detect_lang_scored")  # loud if inactive
    m = _GATE["model"] if status["active"] else None
    t = " ".join((text or "").split())[:1000]
    raw_label = None
    raw_prob = None
    if m is not None and t and len(t) >= _LID_MIN_LEN:
        try:
            lab, prob = m.predict(t, k=1)
            raw_label = lab[0].replace("__label__", "")
            raw_prob = float(prob[0])
            if raw_prob >= _LID_MIN_PROB:
                return {"lang": raw_label, "confident": True,
                       "raw_label": raw_label, "raw_prob": raw_prob,
                       "text_chars": len(t)}
        except Exception as e:  # noqa: BLE001
            # The gate self-tested clean at probe time, so a raise here is a
            # NEW runtime failure — do not swallow it silently. Re-latch the
            # gate inactive and go through the loud path.
            _GATE.update(active=False, model=None,
                         reason=f"runtime predict failure ({type(e).__name__}: {e})")
            assert_langgate_active(context="detect_lang_scored/runtime")
    return {"lang": guess_lang(url, default), "confident": False,
           "raw_label": raw_label, "raw_prob": raw_prob, "text_chars": len(t)}


_PW = {"ctx": None}


def fetch_render(url: str, wait_ms: int = 3500) -> str | None:
    """Render a JS-gated page with Playwright (Python) and return its HTML.
    Used for fedlex / ch.ch (SPAs). Returns None if Playwright isn't available."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("  ! playwright not installed; cannot render", url)
        return None
    try:
        if _PW["ctx"] is None:
            pw = sync_playwright().start()
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            _PW["ctx"] = browser.new_context(user_agent=UA)
        page = _PW["ctx"].new_page()
        page.goto(url, wait_until="networkidle", timeout=45000)
        page.wait_for_timeout(wait_ms)
        html = page.content()
        page.close()
        return html
    except Exception as e:  # noqa: BLE001
        print(f"  ! render failed {url}: {e}")
        return None


def close_render():
    """Close the persistent Playwright browser so the process can exit cleanly."""
    ctx = _PW.get("ctx")
    if ctx is not None:
        try:
            ctx.browser.close()
        except Exception:  # noqa: BLE001
            pass
        _PW["ctx"] = None


def fetch_url(url: str) -> str | None:
    try:
        r = _http().get(url)
        if r.status_code == 200 and "text/html" in r.headers.get("content-type", "text/html"):
            return r.text
        if r.status_code == 200:
            return r.text
    except Exception as e:  # noqa: BLE001
        print(f"  ! fetch failed {url}: {e}")
    return None


def extract(html: str, url: str) -> tuple[str, str] | None:
    """Return (title, main_text) or None. Preserves prose; trafilatura keeps
    tables/lists reasonably (spec wants exactness preserved)."""
    text = trafilatura.extract(
        html, url=url, include_tables=True, include_links=False,
        favor_precision=True, no_fallback=False,
    )
    if not text or len(text) < 120:
        return None
    title = ""
    try:
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)
    except Exception:  # noqa: BLE001
        pass
    return (title or url, text)


def chunk(text: str, target_words: int = 180, overlap_words: int = 30) -> list[str]:
    """Paragraph-aware windows (~target_words) for passage-level grounding.
    Article-level anchoring for fedlex is a refinement; P0 uses windows."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur, cur_n = [], [], 0
    for p in paras:
        n = len(p.split())
        if cur_n + n > target_words and cur:
            chunks.append("\n".join(cur))
            # carry overlap
            tail, tn = [], 0
            for q in reversed(cur):
                tail.insert(0, q); tn += len(q.split())
                if tn >= overlap_words:
                    break
            cur, cur_n = tail[:], tn
        cur.append(p); cur_n += n
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if len(c.split()) >= 15] or ([text] if len(text.split()) >= 15 else [])


def crawl(seeds: list[str], domain: str, max_pages: int = 40, delay: float = 0.5) -> Iterator[tuple[str, str]]:
    """BFS within `domain`, starting from seeds, yielding (url, html)."""
    seen: set[str] = set()
    queue = list(dict.fromkeys(seeds))
    while queue and len(seen) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        html = fetch_url(url)
        if not html:
            continue
        yield url, html
        time.sleep(delay)
        # shallow expansion: same-registered-domain links
        try:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                nxt = urljoin(url, a["href"]).split("#")[0]
                if urlparse(nxt).netloc.endswith(domain) and nxt not in seen and len(queue) < max_pages * 3:
                    queue.append(nxt)
        except Exception:  # noqa: BLE001
            pass


def fedlex_articles(html: str) -> tuple[list[tuple[str, str, str]], str]:
    """Parse a rendered fedlex law page into per-article units.

    fedlex renders each article as `<article id="art_N">`. Returns
    ([(anchor, label, text), ...], page_title) so each statute article becomes
    one passage with a `#art_N` anchor — article-level grounding (spec §5.1).
    """
    soup = BeautifulSoup(html, "lxml")
    page_title = ""
    if soup.title and soup.title.string:
        page_title = soup.title.string.strip()
    out: list[tuple[str, str, str]] = []
    for art in soup.find_all("article", id=re.compile(r"^art_")):
        aid = art.get("id")
        text = art.get_text(" ", strip=True)
        if len(text) < 25:
            continue
        h = art.find(re.compile(r"^h[1-6]$"))
        label = h.get_text(" ", strip=True) if h else aid.replace("_", " ").title()
        out.append(("#" + aid, label, text))
    return out, page_title


def wiki_titles() -> dict:
    """Curated Wikipedia titles per language: the large enumerated set
    (data/wiki_titles_large.json, from enumerate_wiki.py) if present, else the
    small built-in Swiss slice."""
    import json
    import os
    here = os.path.dirname(__file__)
    for p in (os.path.join(here, "../../data/wiki_titles_large.json"),
              "../data/wiki_titles_large.json", "data/wiki_titles_large.json"):
        if os.path.exists(p):
            try:
                d = json.load(open(p, encoding="utf-8"))
                if d:
                    return d
            except Exception:  # noqa: BLE001
                pass
    return SWISS_WIKI_TITLES


def fetch_wikipedia(lang: str, title: str) -> tuple[str, str, str] | None:
    """Return (canonical_url, title, plain_text) via the Wikipedia action API."""
    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "format": "json", "prop": "extracts|info",
        "inprop": "url", "explaintext": 1, "redirects": 1, "titles": title,
    }
    try:
        r = _http().get(api, params=params)
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            if "missing" in page:
                return None
            text = page.get("extract", "")
            url = page.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
            if text and len(text) > 200:
                return (url, page.get("title", title), text)
    except Exception as e:  # noqa: BLE001
        print(f"  ! wiki fetch failed {lang}:{title}: {e}")
    return None


# Curated Wikipedia slice for P0: the 26 cantons + key federal institutions,
# in the four target languages. Concrete + verifiable; full Wikipedia = P1.
SWISS_WIKI_TITLES = {
    "en": ["Cantons of Switzerland", "Canton of Vaud", "Canton of Geneva", "Canton of Zurich",
           "Canton of Bern", "Canton of Ticino", "Federal Council (Switzerland)",
           "Federal Assembly (Switzerland)", "Federal Supreme Court of Switzerland",
           "Swiss Federal Constitution", "Swiss franc", "Switzerland"],
    "de": ["Kantone der Schweiz", "Kanton Waadt", "Kanton Genf", "Kanton Zürich",
           "Kanton Bern", "Kanton Tessin", "Schweizerischer Bundesrat",
           "Bundesversammlung (Schweiz)", "Bundesgericht (Schweiz)",
           "Bundesverfassung der Schweizerischen Eidgenossenschaft", "Schweizer Franken", "Schweiz"],
    "fr": ["Cantons suisses", "Canton de Vaud", "Canton de Genève", "Canton de Zurich",
           "Canton de Berne", "Canton du Tessin", "Conseil fédéral (Suisse)",
           "Assemblée fédérale (Suisse)", "Tribunal fédéral (Suisse)",
           "Constitution fédérale de la Confédération suisse", "Franc suisse", "Suisse"],
    "it": ["Cantoni della Svizzera", "Canton Vaud", "Canton Ginevra", "Canton Zurigo",
           "Canton Berna", "Canton Ticino", "Consiglio federale (Svizzera)",
           "Assemblea federale (Svizzera)", "Tribunale federale (Svizzera)",
           "Costituzione federale della Confederazione Svizzera", "Franco svizzero", "Svizzera"],
}
