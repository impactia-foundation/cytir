"""entscheidsuche.ch decision JSON -> ONE locator passage (`scratch_entscheidsuche_exact`).

Moved out of `ingest_entscheidsuche.py` (task 1101, S7) so the source family
(`cytir.sources.families.entscheidsuche`) and the legacy CLI share one builder. The functions are the
driver's own, UNCHANGED. `build_passage(data, qrec)` returns `(Passage, extra)` or `None` when the
decision has no real Regeste/Abstract (the headnote gate) or fails the text-quality gate.
"""
from __future__ import annotations

import html as _html
import re
import urllib.parse

from cytir.core.common import (Passage, extraction_junk, extractive_snippet, low_quality,
                            strip_tracking_url)

# original-court hosts we trust as stable canonical permalinks
_BGER_HOST = re.compile(r"(^|\.)bger\.ch$", re.I)
# markers that make an original URL unstable (session tokens / download / query app)
_UNSTABLE = re.compile(r"jsessionid|nph-omniscgi|OmnisServer|decisionId|jurispub|"
                       r"download|/publiws/", re.I)
_DATE_TAIL = re.compile(r"_(\d{4})(?:-\d{2}-\d{2})?$")
_DATE_ANY = re.compile(r"_(\d{4})-\d{2}-\d{2}$|_(\d{4})$|_nodate$")
_CANTON_FR = {"GE", "VD", "JU", "NE"}
_CANTON_IT = {"TI"}


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")


def clean_html(raw: str) -> str:
    """Strip HTML tags (<br> -> space), decode entities (&eacute; &amp; …),
    collapse whitespace. Regeste/Kopfzeile text carries light HTML markup."""
    if not raw:
        return ""
    raw = re.sub(r"(?i)<br\s*/?>", " ", raw)
    txt = _html.unescape(_TAG.sub(" ", raw))
    return _WS.sub(" ", txt.replace("\xa0", " ")).strip()


def bger_permalink(url: str) -> str:
    """search.bger.ch result URLs carry volatile search-context params
    (from_date/page/rank/number_of_ranks). Keep only the stable doc selector
    (highlight_docid + lang) so the stored URL is a clean, deterministic
    permalink to the decision."""
    try:
        p = urllib.parse.urlsplit(url)
    except ValueError:
        return url
    if "bger.ch" not in (p.netloc or "").lower() or "highlight_docid" not in (p.query or ""):
        return url
    keep = [(k, v) for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=True)
            if k in ("highlight_docid", "lang", "type")]
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path,
                                    urllib.parse.urlencode(keep), ""))


def valid_date(d: str) -> str | None:
    """Real ISO date or None (guards 0000-00-00 / pre-1900 placeholders)."""
    if len(d) == 10 and d[4] == "-" and d[:4].isdigit() and int(d[:4]) >= 1900 \
            and d[5:7].isdigit() and d[8:10].isdigit() and d[5:7] != "00" and d[8:10] != "00":
        return d
    return None


def spider_of_sitemap(loc: str) -> str | None:
    m = re.search(r"/Sitemaps/(.+?)_\d+\.xml$", loc)
    return m.group(1) if m else None


def year_of(viewid: str) -> int:
    m = _DATE_TAIL.search(viewid)
    return int(m.group(1)) if m else 0


def dedup_base(viewid: str) -> str:
    """BGE (and any) date-variant base: strip a trailing _YYYY-MM-DD / _YYYY /
    _nodate so `_1954-01-27`, `_1954`, `_nodate` collapse to one decision."""
    return _DATE_ANY.sub("", viewid)


# ---------------------------------------------------------------------------


def _first_text(entries, lang: str | None):
    """From a [{Sprachen:[...], Text:...}] list pick the lang match, else first."""
    if not entries or not isinstance(entries, list):
        return "", None
    pick = None
    if lang:
        for e in entries:
            if isinstance(e, dict) and lang in (e.get("Sprachen") or []):
                pick = e
                break
    if pick is None:
        pick = entries[0] if isinstance(entries[0], dict) else None
    if not pick:
        return "", None
    langs = pick.get("Sprachen") or []
    return (pick.get("Text") or "").strip(), (langs[0] if len(langs) == 1 else None)


def _dedup_segments(text: str) -> str:
    """Regeste JSON sometimes triples the text ('X;;X;;X' or 'X | X'). Collapse
    consecutive identical segments split on ';;' or ' | '."""
    if not text:
        return text
    for sep in (";;", " | "):
        if sep in text:
            segs = [s.strip() for s in text.split(sep)]
            out = []
            for s in segs:
                if not out or out[-1] != s:
                    out.append(s)
            text = sep.join(out)
    return text.strip()


def guess_lang(spider: str, sprache, abstract_lang, kopf_lang) -> str:
    if isinstance(sprache, str) and sprache.lower() in ("de", "fr", "it"):
        return sprache.lower()
    for cand in (abstract_lang, kopf_lang):
        if cand in ("de", "fr", "it"):
            return cand
    canton = spider.split("_")[0]
    if canton in _CANTON_FR:
        return "fr"
    if canton in _CANTON_IT:
        return "it"
    return "de"


def court_code(spider: str) -> str:
    return {
        "CH_BGE": "BGE", "CH_BGer": "BGer", "CH_BSTG": "BStGer",
        "CH_BPatG": "BPatGer", "CH_BVGer": "BVGer",
    }.get(spider, spider)


def choose_url(data: dict, view_url: str) -> str:
    """Original-court permalink where stable, else the entscheidsuche /view/ page."""
    for key in ("HTML", "PDF"):
        blk = data.get(key)
        if isinstance(blk, dict):
            u = (blk.get("URL") or "").strip()
            if not u.startswith(("http://", "https://")):
                continue
            u = strip_tracking_url(u)  # drops ;jsessionid in the path
            try:
                host = re.sub(r"^https?://", "", u).split("/")[0].lower()
            except Exception:  # noqa: BLE001
                host = ""
            if _BGER_HOST.search(host):
                return bger_permalink(u)          # stable BGer permalink (params trimmed)
            if not _UNSTABLE.search(u):
                return u                      # clean cantonal landing page
    return view_url


def build_passage(data: dict, qrec: dict):
    spider = qrec["spider"]
    sprache = data.get("Sprache")
    kopf_text, kopf_lang = _first_text(data.get("Kopfzeile"), None)
    ab_text, ab_lang = _first_text(data.get("Abstract"), sprache if isinstance(sprache, str) else None)
    kopf_text = clean_html(kopf_text)
    ab_text = clean_html(_dedup_segments(ab_text))
    lang = guess_lang(spider, sprache, ab_lang, kopf_lang)

    num = data.get("Num")
    num_s = (num[0] if isinstance(num, list) and num else (num if isinstance(num, str) else "")) or ""
    datum = (data.get("Datum") or qrec.get("date") or "").strip()

    # header line — court / chamber / date / Aktenzeichen
    header = kopf_text or f"{court_code(spider)} {datum} {num_s}".strip()

    # --- headnote/text-quality gate: keep only decisions with a real Regeste/
    # Abstract (the agent-answerable headnote). Mirrors ingest_rii.py's Leitsatz
    # gate. A bare header with no headnote is not a useful grounding passage.
    if not ab_text or len(re.findall(r"[^\W\d_]{2,}", ab_text, re.UNICODE)) < 3:
        return None

    parts = [header, f"Regeste: {ab_text}"]
    if num_s and num_s not in header:
        parts.append(f"Aktenzeichen: {num_s}")
    text = "\n".join(parts)[:4000]
    if low_quality(text) or extraction_junk(text):
        return None

    url = choose_url(data, qrec["view_url"])
    host = re.sub(r"^https?://", "", url).split("/")[0].lower()
    if host.startswith("www."):
        host = host[4:]
    title = header[:300]
    snippet = extractive_snippet(ab_text or header, 30)
    canton = spider.split("_")[0]
    p = Passage(url=url, title=title, snippet=snippet, text=text, lang=lang,
                domain=host, tier="A", source_type="official", trust_tier=1.0,
                last_seen=valid_date(datum))
    extra = {"country": "CH", "provenance": "entscheidsuche",
             "court": court_code(spider), "canton": canton, "spider": spider,
             "viewid": qrec["viewid"]}
    return p, extra
