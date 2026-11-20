"""Find casino / pharma spam chunks in a staged pack and write an exclusion file (task 699).

WHY. The CC tranche deltas (task 698) hold rows from allowlisted hosts whose pages were
hacked, parked or injected with gambling text (measured 2026-10-03 on t3: whole-page casino
doorways on `marketfox.io`, `thefancy.com`, `certsimple.com`, `shiftpayments.com`; injected
chunks on `georgekaramolegos.com`, `tree.it`, `controlchaos.com`; comment spam on
`techblog.cloudperf.net`). An index of authoritative sources must not serve them, and the
drain's 1 % fail share does not stop them.

RULE (declared, row level, on the chunk the pack embeds: title + text):

    spam_lexicon   a small multilingual list of gambling / pharma-spam phrases. Single words
                   that also have an ordinary meaning (`slot`, `bet`, `bonus`, `poker`, `porn`)
                   are NOT in it: measured, they flag Postgres replication slots, "a bet on
                   LEO", planning poker and essays about porn.
    R1 dense       >= 3 DISTINCT lexicon terms and >= 4 hits in one chunk.
    R2 title       the title itself holds a spam-title phrase (`casino`, `togel`, `judi online`,
                   `slot gacor`, ...) and the chunk holds >= 2 lexicon hits.
    R3 page        a URL with >= 1 R1/R2 chunk and >= 50 % of its chunks holding >= 1 lexicon
                   hit: every chunk of that URL is excluded (a doorway page is spam whole).
    R4 attachment  a WordPress attachment URL (`.../name-65x65`) with one chunk of comment spam (a
                   lexicon or pharma hit): every chunk of it is excluded; an attachment page has
                   no content of its own, only its comments.
    R6 bbcode      a chunk with a rendered BBCode link `[url=http...` (comment spam).
    R7 injected    a chunk holding the injected anchor `Beste Online Casino Deutschland` (task 704):
       anchor      one lexicon hit only, so R1 never fires. Measured on deutsch-plus.de (a genuine
                   NGO site, 2026-10-03): 6 of its 148 rows carry it in the snippet (5 tag pages that
                   hold only the menu with the anchor first, 1 spam sentence); the other 142 are
                   genuine and stay. A row rule, never a host hold-out (4 % share, 6 rows).

    HOST hold-out (task 701, 2026-10-03; the row rule's recall is weak on a fully hijacked host: it
    names casino text, not the pharma / diploma-mill / job-tag-cloud / generated filler that fills
    the rest of the page). A HOST is held out when its SPAM SIGNAL share is >= `HOST_SHARE_MIN`
    (0.50) over >= `HOST_MIN_SIGNAL_ROWS` (20) signal rows on >= `HOST_MIN_SIGNAL_URLS` (3) URLs, or
    when it is on the sealed hold-out list (`data/spam-host-lists.v1.json`, every entry hand-read with
    its evidence). A row has the spam signal when the row rule flags it, or it holds a phrase of the
    narrow host-spam list (`_HOST_SPAM_RE`: pharma, steroid, escort-service, diploma-mill), or its URL
    is a `/job-tag/` tag cloud. (`_HOST_SPAM_RE` needs two DISTINCT phrases in the chunk.) ALL rows of a held-out host go to the exclusion file with reason
    `host_hijacked`. Legitimate hosts with a few hacked pages never reach the threshold and keep the
    row rule only. The same file lists `exceptions`: gambling-INDUSTRY and anti-gambling sites
    (betable, flowplay, ...) that the lexicon flags by topic, not by spam; their rows are exempt from
    the row rule and from the hold-out.

The output is `{"point_id", "reason", "url", "domain"}` rows, the shape
`cytir.index.exclude_file` loads; the pack stays sealed and whole, the rule is data. Apply with

    python -m cytir.index.reseal --queue-dir Q --pack-id P --reason R \\
        --exclude-ids-file F --exclude-ids-reason "casino/pharma spam (cytir.ops.spam_rows R1-R3)"

`scan` writes the exclusion file, the host table and the hand-check samples (hits, and
non-hits from the same hosts); a scan that reads 0 rows refuses. `apply` is the reseal call above.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import re
import sys
import time
from pathlib import Path

_LEX = [
    r"casinos?", r"casino non aams", r"casin[oò]s? en ligne", r"online gambling", r"pokies",
    r"togel", r"judi (?:online|bola|slot)", r"slot gacor", r"situs slot", r"slot (?:online|demo)",
    r"online slots?", r"slot machines?", r"gacor", r"maxwin", r"sbobet", r"bandar (?:judi|bola|togel)",
    r"free spins?", r"welcome bonus", r"no[- ]deposit", r"deposit bonus", r"bonus codes?",
    r"sportsbooks?", r"jackpots?", r"roulette", r"blackjack", r"baccarat", r"wagering",
    r"payid", r"live dealer", r"real money", r"high rollers?", r"free bets?", r"betting sites?",
    r"bookmakers?", r"paris sportifs", r"spielautomaten", r"kasyno", r"bahis", r"rtp slot",
    r"pragmatic play", r"buy (?:viagra|cialis|levitra)", r"(?:cheap|generic) (?:viagra|cialis)",
    r"viagra online", r"online pharmacy", r"payday loans?",
    r"(?:canadian|india|indian|mexican) pharmac(?:y|ies)", r"pharmacie en ligne", r"online apotheke",
    r"sildenafil", r"tadalafil", r"erectile", r"call girls?", r"escort services?", r"pay4d",
    r"nolvadex", r"cytotec", r"misoprostol", r"prescription drugs? online", r"ed medication",
]
_LEX_RE = re.compile(r"(?<![\w])(" + "|".join(_LEX) + r")(?![\w])", re.IGNORECASE)
_TITLE_RE = re.compile(
    r"casino|togel|judi (?:online|bola|slot)|slot gacor|situs slot|pokies|online gambling|"
    r"betting sites?|sbobet|sportsbook|casin[oò]s? en ligne|bahis", re.IGNORECASE)

_INJECTED_ANCHOR_RE = re.compile(r"beste[rns]? online[- ]casinos? deutschland", re.IGNORECASE)
_BBCODE_RE = re.compile(r"\[url=https?://", re.IGNORECASE)
_ATTACH_RE = re.compile(r"/[^/?#]*-\d{2,4}[x\u00d7]\d{2,4}/?(?:[?#].*)?$", re.IGNORECASE)
_PHARMA_RE = re.compile(r"kamagra|valtrex|viagra|cialis|levitra|tramadol|live sex", re.IGNORECASE)

_HOST_SPAM_RE = re.compile(
    r"anavar|clomid|clomiphene|kamagra|nolvadex|dianabol|xanax|tramadol|viagra|cialis|levitra|"
    r"buy steroids|call girls?|escorts? (?:in|service|agency|girls)|aviator oyun|1xbet|mostbet|"
    r"заказать диплом|купить диплом|my mexican drugstore",
    re.IGNORECASE)
_JOBTAG_RE = re.compile(r"/job-tag/", re.IGNORECASE)

HOST_SHARE_MIN = 0.50
HOST_MIN_SIGNAL_ROWS = 20
HOST_MIN_SIGNAL_URLS = 3
HOST_REASON = "host_hijacked"
HOST_LISTS_SCHEMA = "aw.spam-host-lists.v1"

MIN_DISTINCT = 3
MIN_HITS = 4
TITLE_MIN_HITS = 2
PAGE_MIN_SHARE = 0.5


def lexicon_hits(text: str) -> list[str]:
    return [m.group(1).lower() for m in _LEX_RE.finditer(text or "")]


def row_reason(title: str, text: str, url: str = "") -> str | None:
    """`'R1'` / `'R2'` / `'R4'` / `'R6'` / `'R7'` / None for one chunk."""
    hits = lexicon_hits(f"{title or ''}\n{text or ''}")
    if _BBCODE_RE.search(text or ""):
        return "R6"
    if _INJECTED_ANCHOR_RE.search(text or ""):
        return "R7"
    if _ATTACH_RE.search(url or "") and (hits or _PHARMA_RE.search(text or "")):
        return "R4"
    if len(set(hits)) >= MIN_DISTINCT and len(hits) >= MIN_HITS:
        return "R1"
    if _TITLE_RE.search(title or "") and len(hits) >= TITLE_MIN_HITS:
        return "R2"
    return None


def host_signal(title: str, text: str, url: str = "") -> bool:
    """The wider per-row signal the HOST share is measured on (never an exclusion by itself).

    The scan adds the row rule's own flags (R1-R6, incl. whole pages R3/R4) to it.
    """
    if _JOBTAG_RE.search(url or ""):
        return True
    # two DISTINCT phrases: one drug name alone is a company in a directory (bcbargaining.ca lists
    # a "Cialis ..." employer; measured 2026-10-03)
    found = {m.group(0).lower() for m in _HOST_SPAM_RE.finditer(f"{title or ''}\n{text or ''}")}
    return len(found) >= 2


def norm_domain(d: str) -> str:
    d = (d or "").strip().lower()
    return d[4:] if d.startswith("www.") else d


def load_host_lists(path) -> dict:
    """Read the sealed hold-out / exception lists; refuse a malformed file or an entry without evidence."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != HOST_LISTS_SCHEMA:
        raise SystemExit(f"REFUSED: {path} is not a {HOST_LISTS_SCHEMA} file")
    out: dict[str, dict] = {"hold_out": {}, "exceptions": {}}
    for key, field in (("hold_out", "evidence"), ("exceptions", "reason")):
        for e in doc.get(key, []):
            d = norm_domain(e.get("domain", ""))
            if not d or not e.get(field):
                raise SystemExit(f"REFUSED: {path} {key} entry {e!r} needs a domain and a {field}")
            out[key][d] = e[field]
    both = set(out["hold_out"]) & set(out["exceptions"])
    if both:
        raise SystemExit(f"REFUSED: {sorted(both)} are both held out and excepted")
    return out


def host_holdout(stats: dict[str, dict], lists: dict | None = None) -> dict[str, dict]:
    """`stats` = {domain: {"rows", "signal", "signal_urls"}} -> {domain: evidence} of held-out hosts.

    Exceptions are never held out; a sealed hold-out entry is held out whatever its share.
    """
    lists = lists or {"hold_out": {}, "exceptions": {}}
    out: dict[str, dict] = {}
    for d, s in stats.items():
        if d in lists["exceptions"]:
            continue
        share = s["signal"] / s["rows"] if s["rows"] else 0.0
        measured = (share >= HOST_SHARE_MIN and s["signal"] >= HOST_MIN_SIGNAL_ROWS
                    and s["signal_urls"] >= HOST_MIN_SIGNAL_URLS)
        sealed = d in lists["hold_out"]
        if measured or sealed:
            out[d] = {"rows": s["rows"], "signal_rows": s["signal"], "share": round(share, 3),
                      "signal_urls": s["signal_urls"],
                      "basis": "+".join(b for b, on in (("measured", measured), ("sealed_list", sealed)) if on)}
    return out


def page_flags(rows: list[tuple[str, str, str | None, int]]) -> dict[str, str]:
    """`rows` = [(point_id, url, row_reason, n_lexicon_hits)] -> {point_id: reason}, with R3."""
    by_url: dict[str, list[tuple[str, str | None, int]]] = collections.defaultdict(list)
    for pid, url, why, nh in rows:
        by_url[url].append((pid, why, nh))
    out: dict[str, str] = {}
    for items in by_url.values():
        for pid, why, _ in items:
            if why:
                out[pid] = why
        if any(w == "R4" for _, w, _ in items):  # an attachment page is spam whole
            for pid, _, _ in items:
                out.setdefault(pid, "R4")
        strong = any(w in ("R1", "R2") for _, w, _ in items)
        share = sum(1 for _, _, nh in items if nh >= 1) / len(items)
        if strong and share >= PAGE_MIN_SHARE:
            for pid, _, _ in items:
                out.setdefault(pid, "R3")
    return out


def iter_chunk_rows(chunks_dir: Path):
    for p in sorted(Path(chunks_dir).glob("*.texts.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def analyse(rows, lists: dict) -> dict:
    """The one decision, shared by `scan` (a staged chunk dir) and `screen_jsonl` (a gated passages
    file, before it is staged): flags (R1-R6 and `host_hijacked`), the host table, the held-out hosts.

    `rows` yields dicts with `point_id`, `url`, `domain`, `title`, `text`. Rows of an excepted host are
    exempt from the row rule and from the hold-out.
    """
    meta: dict[str, dict] = {}
    triples = []
    wide: set[str] = set()
    n = 0
    for r in rows:
        n += 1
        pid, dom, url = r["point_id"], norm_domain(r.get("domain", "")), r.get("url", "")
        title, text = r.get("title") or "", r.get("text") or ""
        meta[pid] = {"url": url, "domain": dom, "title": title[:120], "text": text[:300]}
        if dom in lists["exceptions"]:
            triples.append((pid, url, None, 0))
            continue
        triples.append((pid, url, row_reason(title, text, url), len(lexicon_hits(f"{title}\n{text}"))))
        if host_signal(title, text, url):
            wide.add(pid)
    flags = page_flags(triples)
    sig_rows: collections.Counter = collections.Counter()
    sig_urls: dict[str, set] = collections.defaultdict(set)
    for pid in wide | set(flags):  # the signal = the row rule's flags (incl. R3/R4 pages) or the wider phrases
        sig_rows[meta[pid]["domain"]] += 1
        sig_urls[meta[pid]["domain"]].add(meta[pid]["url"])
    tot = collections.Counter(m["domain"] for m in meta.values())
    stats = {d: {"rows": tot[d], "signal": sig_rows[d], "signal_urls": len(sig_urls[d])}
             for d in (set(sig_rows) | set(lists["hold_out"])) if tot[d]}
    held = host_holdout(stats, lists)
    for pid, m in meta.items():
        if m["domain"] in held:
            flags[pid] = HOST_REASON
    return {"n": n, "meta": meta, "flags": flags, "tot": tot, "sig_rows": sig_rows, "held": held}


def scan(chunks_dir: Path, out_dir: Path, *, sample_n: int = 120, seed: int = 1,
         host_lists: Path | None = None) -> dict:
    """Scan one stage's chunks; write `exclude.jsonl`, `hosts.tsv`, `host_holdout.json`, `samples.jsonl`.

    `exclude.jsonl` = the row rule's flags (R1-R6) plus EVERY row of a held-out host (reason
    `host_hijacked`). Rows of an excepted host are exempt from both.
    """
    lists = load_host_lists(host_lists) if host_lists else {"hold_out": {}, "exceptions": {}}
    a = analyse(iter_chunk_rows(chunks_dir), lists)
    n, meta, flags, tot, sig_rows, held = a["n"], a["meta"], a["flags"], a["tot"], a["sig_rows"], a["held"]
    if n == 0:
        raise SystemExit(f"REFUSED: {chunks_dir} holds 0 chunk rows")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "exclude.jsonl", "w", encoding="utf-8") as fh:
        for pid in sorted(flags):
            m = meta[pid]
            fh.write(json.dumps({"point_id": pid, "reason": flags[pid], "url": m["url"],
                                 "domain": m["domain"]}, ensure_ascii=False) + "\n")
    (out_dir / "host_holdout.json").write_text(json.dumps(held, indent=1, sort_keys=True), encoding="utf-8")
    hit = collections.Counter(meta[p]["domain"] for p in flags)
    with open(out_dir / "hosts.tsv", "w", encoding="utf-8") as fh:
        fh.write("domain\tspam_rows\ttotal_rows\tshare\tsignal_rows\tsignal_share\theld_out\n")
        for d, c in hit.most_common():
            fh.write(f"{d}\t{c}\t{tot[d]}\t{c / tot[d]:.3f}\t{sig_rows[d]}\t{sig_rows[d] / tot[d]:.3f}\t{int(d in held)}\n")
    rnd = random.Random(seed)
    hit_ids = sorted(flags)
    hit_hosts = set(hit)
    non_ids = sorted(p for p, m in meta.items() if p not in flags and m["domain"] in hit_hosts)
    with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as fh:
        for label, pool in (("hit", hit_ids), ("nonhit_same_host", non_ids)):
            for pid in rnd.sample(pool, min(sample_n, len(pool))):
                fh.write(json.dumps({"label": label, "point_id": pid, "reason": flags.get(pid),
                                     **meta[pid]}, ensure_ascii=False) + "\n")
    return {"rows": n, "flagged": len(flags), "share": len(flags) / n,
            "by_rule": dict(collections.Counter(flags.values())), "hosts": len(hit),
            "held_out_hosts": len(held), "held_out_rows": sum(tot[d] for d in held)}


def screen_jsonl(in_path: Path, out_path: Path, report_path: Path, *, host_lists: Path | None = None) -> dict:
    """The chain form (before a gated passages file is staged and sealed): copy the rows that are NOT
    flagged to `out_path`, write the flagged rows with their reason to `report_path`, return the counts.

    The gated passages are chunk-level rows already (`point_id`, `text`), so the rules are the same.
    A file that reads 0 rows refuses; a screen that would drop EVERYTHING refuses (a broken rule,
    never a delta).
    """
    lists = load_host_lists(host_lists) if host_lists else {"hold_out": {}, "exceptions": {}}

    def rows():
        with open(in_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    a = analyse(rows(), lists)
    if a["n"] == 0:
        raise SystemExit(f"REFUSED: {in_path} holds 0 rows")
    flags = a["flags"]
    if len(flags) == a["n"]:
        raise SystemExit(f"REFUSED: the spam screen flags all {a['n']} rows of {in_path}")
    kept = 0
    with open(in_path, encoding="utf-8") as fh, open(out_path, "w", encoding="utf-8") as out, \
            open(report_path, "w", encoding="utf-8") as rep:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            why = flags.get(r["point_id"])
            if why:
                rep.write(json.dumps({"point_id": r["point_id"], "reason": why, "url": r.get("url", ""),
                                      "domain": r.get("domain", "")}, ensure_ascii=False) + "\n")
            else:
                out.write(line if line.endswith("\n") else line + "\n")
                kept += 1
    return {"rows_in": a["n"], "rows_kept": kept, "rows_screened_out": len(flags),
            "by_rule": dict(collections.Counter(flags.values())),
            "held_out_hosts": sorted(a["held"])}


def apply(queue_dir: str, pack_id: str, exclude_file: Path, *, wait_s: int = 0,
          log=print) -> dict:
    """Pin `exclude_file` into the queued pack's manifest (reseal) once the pack is READY.

    Waits up to `wait_s` seconds for the compose timer to enqueue the pack. Never edits a pack in
    place: `cytir.index.reseal` records the reason and mints a new manifest sha. Refuses on a pack
    that is already DONE / PARKED / SUPERSEDED (a landed spam pack needs a tombstone pack).
    """
    from cytir.index import reseal as R
    pack = Path(queue_dir) / pack_id
    deadline = time.time() + wait_s
    while not (pack / "READY").is_file():
        if time.time() >= deadline:
            raise SystemExit(f"REFUSED: {pack_id} is not READY in {queue_dir}")
        time.sleep(5)
    if not Path(exclude_file).is_file() or Path(exclude_file).stat().st_size == 0:
        raise SystemExit(f"REFUSED: {exclude_file} is missing or empty")
    reason = "casino/pharma spam rows and hijacked hosts excluded (cytir.ops.spam_rows R1-R6 + host_hijacked, tasks 699/701)"
    out = R.reseal_pack(queue_dir, pack_id, reason=reason, exclude_ids_file=str(exclude_file),
                        exclude_ids_reason=reason)
    log(json.dumps(out))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scan", help="write exclude.jsonl, hosts.tsv, samples.jsonl for one stage")
    sc.add_argument("--chunks-dir", required=True, type=Path)
    sc.add_argument("--out-dir", required=True, type=Path)
    sc.add_argument("--sample-n", type=int, default=120)
    sc.add_argument("--host-lists", type=Path, default=None,
                    help="sealed hold-out / exception lists (aw.spam-host-lists.v1)")
    ap2 = sub.add_parser("apply", help="reseal a queued pack with the exclusion file")
    ap2.add_argument("--queue-dir", required=True)
    ap2.add_argument("--pack-id", required=True)
    ap2.add_argument("--exclude-file", required=True, type=Path)
    ap2.add_argument("--wait-s", type=int, default=0)
    sr = sub.add_parser("screen", help="drop flagged rows from a gated passages file (the chain form)")
    sr.add_argument("--in", dest="in_path", required=True, type=Path)
    sr.add_argument("--out", required=True, type=Path)
    sr.add_argument("--report", required=True, type=Path)
    sr.add_argument("--host-lists", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        print(json.dumps(scan(a.chunks_dir, a.out_dir, sample_n=a.sample_n, host_lists=a.host_lists)))
    elif a.cmd == "screen":
        print(json.dumps(screen_jsonl(a.in_path, a.out, a.report, host_lists=a.host_lists)))
    else:
        apply(a.queue_dir, a.pack_id, a.exclude_file, wait_s=a.wait_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
