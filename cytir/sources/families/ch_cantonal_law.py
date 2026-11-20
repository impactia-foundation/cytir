"""`scratch_ch_cantonal_law_exact` -- Swiss cantonal law, ONE adapter, portal dialects (task 1108, S7).

The family had four drivers; the unit carries a `portal` field and every verb dispatches on it:

| portal | driver (stays, as a thin CLI over the moved builder) | builder | state |
|---|---|---|---|
| `lexwork` | `ingest_lexwork.py` (sitrox platforms, article-level) | `cytir.extract.cantonal_lexwork` | SHIPPED, rows byte-identical on a sealed sample |
| `zhlex`   | `ingest_zhlex.py` (zh.ch JSON index, law-level)         | `cytir.extract.cantonal_zhlex`   | SHIPPED, rows byte-identical on a sealed sample |
| `vd`      | retired `ingest_vd` driver (deleted, task 1552)          | -                             | DECLINED: consent-declined source (task 1317); in-code refusal, never produced a row |
| `cantonal_cc` | `ingest_cantonal_cc.py` (GE/NE/JU/SZ/TI from Common Crawl) | `cytir.extract.cantonal_cc` | SHIPPED (task 1174), rows equal a recording of the driver made with the real language gate |

Asking for a declined portal (`--opt portals=vd`) raises with that reason: it is never ingested silently. Lexwork has a live path (below); the other legacy drivers still produce the served rows.

Options (`--opt KEY=VALUE`): `root` (required): a directory of tapes, one JSONL per portal
(`lexwork.jsonl`: `{canton, host, lang, sysno, idx_title, response}` = the driver's `process_law`
inputs plus the `texts_of_law/<sysno>` JSON body; `zhlex.jsonl`: one `lawcollectionsearch` index
item per line). `portals` (optional, comma list): which tapes to read (default: those present).
`include_repealed` (zhlex, default false). `cantonal_cc.jsonl`: one line per capture the driver's extractor
received, `{canton, url, digest, html}` (the CC index resolve and the WARC range read stay in the CLI; the
HTML is already cut at `AW_HTML_CAP`). A unit is one law (lexwork) or one index item (zhlex);
each ends admitted or excluded(<the builder's own reason>); a line that is not JSON is quarantined.

LIVE lexwork (task 1488): `--opt portals=lexwork` with NO `root` (no tape) acquires live. `discover` walks the
`LW.CANTONS` table (`cantons=BS,AG` narrows it, `cap=N` laws per canton/lang), lists each canton's
in-force laws (`LW.lightweight_index`) and emits one unit per law; `acquire` fetches the law through the
moved `LW._get` (same global + per-host throttle and retries as the old driver) and builds the very
tape line `extract` reads, so a live row equals a tape row. `done_file=<path>` (keys `canton|lang|sysno`)
skips laws already done and appends a key once a law is acquired. A 404 is `UnitExcluded(not_found)`.
CONSENT: the live index call contacts the host BEFORE the framework's unit gate, so `discover` resolves
each canton host first through the ONE resolver (`ctx.consent`, else `run_family.default_consent` =
`cytir.sources.runner.resolve_consent`); a host whose bucket is not `ok`/`ok_search_only` yields no unit and
is named in `ctx.notes["consent_refused"]`; `skip_consent` is refused for a live run. The framework then
gates every unit (`consent_hosts`) as for any tape.

Dispositions that differ from the driver, said: a duplicate point id is quarantined by the
framework (the driver kept both rows; the samples hold none); the lexwork driver wrote rows in
worker order, the pack orders by point id; an empty `xhtml_tol` is `excluded(no_xhtml)` (the
driver returned 0 silently). Consent: the platform host of each unit is resolved live before the
unit is read (declared basis: robots.txt/TDMRep per origin); a decline is a terminal. No consent
field is added or defaulted by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from urllib.parse import urlsplit

import httpx

from cytir.acquire import discovery_ingest as di
from cytir.extract import cantonal_cc as CC
from cytir.extract import cantonal_lexwork as LW
from cytir.extract import cantonal_zhlex as ZH
from cytir.sources.family import CHUNK_NATIVE_CHUNKED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

FAMILY_ID = "scratch_ch_cantonal_law_exact"
ZH_HOST = "www.zh.ch"
SHIPPED = ("lexwork", "zhlex", "cantonal_cc")
#: Declared consent-declined portals (task 1317, ruling 60 D10): a source we do not ingest, not a missing
#: dialect. `consent_state` is a value of `data/payload-enum-tables.v1.json`; the evidence is the
#: publisher's own `X-Robots-Tag: noindex, nofollow, noarchive` (re-probed 2026-10-25,
#: the contract row `consent.declared_declined_origins`), not only our robots.txt 403.
DECLINED = {
    "vd": {"consent_state": "declined",
           "reason": "consent-declined host (the Vaud portal: X-Robots-Tag noindex,nofollow,noarchive on "
                     "every response; robots.txt 403); refused in code (assert_crawlable) since task 433"},
}


# The page-extraction step, named (task 1726). The drivers cut the HTML at `AW_HTML_CAP` and run the
# extractor in a worker pool with a hard per-page deadline (`AW_EXTRACT_TIMEOUT`); one helper, the
# drivers' own. Tests replace this seam; production calls the helper.
_extract_in_pool = di._extract_in_pool


# Per-host circuit breaker (task 1730), the driver's own rule (`discovery_ingest._DOMAIN_TIMEOUT_BREAKER`, task 634,
# default 5, env `AW_DOMAIN_TIMEOUT_BREAKER`): after this many CONSECUTIVE `extract_timeout`s of one host, every
# later page of that host is excluded with the driver's reason string WITHOUT running extraction (each timeout
# terminates the shared pool, so a hung host would otherwise cost N x `AW_EXTRACT_TIMEOUT`). Any page of the host
# that does not time out resets its counter. The counters live on the run's context, so a new run starts clean.
_HOST_TIMEOUT_BREAKER = di._DOMAIN_TIMEOUT_BREAKER
BREAKER_REASON = "circuit_breaker_early_stop"


def _extract_page(doc: Doc, ctx: RunContext) -> list[dict]:
    """Rows of one cantonal_cc page, or `UnitExcluded("extract_timeout")` when the deadline passes, or
    `UnitExcluded("circuit_breaker_early_stop")` once the host timed out `_HOST_TIMEOUT_BREAKER` times in a row:
    the page is skipped and recorded in the ledger, the run continues."""
    host = urlsplit(doc.url).netloc.lower()
    consecutive = ctx.scratch.setdefault("cantonal_timeouts", {})
    if consecutive.get(host, 0) >= _HOST_TIMEOUT_BREAKER:
        raise UnitExcluded(BREAKER_REASON)
    meta = {"content": True, "links": False, "lang_default": doc.lang or "en",
            "passage_common": doc.extra["passage_common"], "digest": doc.extra.get("digest")}
    passages, _links, _rej, timed_out = _extract_in_pool(doc.url, doc.text[: di._HTML_CAP], meta)
    consecutive[host] = consecutive.get(host, 0) + 1 if timed_out else 0
    if timed_out:
        raise UnitExcluded("extract_timeout")
    rows = []
    for p in passages:
        row = p.payload()
        row["text"] = p.text
        row["point_id"] = p.point_id
        rows.append(row)
    return rows


def _live(ctx: RunContext) -> bool:
    """Live lexwork: the portal asked for by name, with no tape directory."""
    asked = [p.strip() for p in str(ctx.options.get("portals") or "").split(",") if p.strip()]
    return asked == ["lexwork"] and not ctx.options.get("root")


def _done_keys(path: str | None) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def _discover_live(ctx: RunContext):
    """One unit per in-force law of every consented canton host (see the module docstring)."""
    from cytir.sources import run_family
    if ctx.skip_consent:
        raise ValueError(f"{FAMILY_ID}: a live run contacts the hosts; --skip-consent is refused")
    resolver = ctx.consent or run_family.default_consent
    only = {c.strip() for c in str(ctx.options.get("cantons") or "").split(",") if c.strip()}
    unknown = only - {c[0] for c in LW.CANTONS}
    if unknown:
        raise ValueError(f"{FAMILY_ID}: unknown canton(s) {sorted(unknown)}")
    cap = int(ctx.options.get("cap") or 0)
    done = _done_keys(ctx.options.get("done_file"))
    units, refused, failed, skipped = [], {}, [], 0
    for canton, host, langs in LW.CANTONS:
        if only and canton not in only:
            continue
        try:
            verdict = resolver(host, True)
        except Exception as exc:  # noqa: BLE001 -- a recorded decline or an instrument failure
            verdict = {"bucket": "declined" if type(exc).__name__ == "ConsentBlocked" else "error"}
        if verdict.get("bucket") not in run_family.CLEAR_BUCKETS:
            refused[host] = verdict.get("bucket")
            continue
        for lang in langs:
            try:
                laws = LW.lightweight_index(host, lang)
            except Exception as exc:  # noqa: BLE001 -- named in the notes, never an empty success
                failed.append(f"{canton}/{lang}:{type(exc).__name__}")
                continue
            for it in laws[:cap] if cap else laws:
                sysno = it.get("systematic_number")
                if f"{canton}|{lang}|{sysno}" in done:
                    skipped += 1
                    continue
                units.append(Unit(
                    key=f"lexwork:{canton}:{lang}:{sysno}",
                    url=f"https://{host}/api/{lang}/texts_of_law/{sysno}",
                    meta={"portal": "lexwork", "live": True, "canton": canton, "host": host,
                          "lang": lang, "sysno": sysno, "idx_title": it.get("title", "")},
                    consent_hosts=(host,)))
    ctx.notes.update({"live": True, "portals": ["lexwork"], "units": len(units),
                      "consent_refused": refused, "index_failed": failed, "done_skipped": skipped})
    if not units:
        raise RuntimeError(f"{FAMILY_ID}: live discover found no unit (refused {refused}, "
                           f"index failed {failed}, done {skipped})")
    return units


def _acquire_live(unit: Unit, ctx: RunContext):
    m = unit.meta
    try:
        r = LW._get(m["host"], f"/api/{m['lang']}/texts_of_law/{m['sysno']}")
    except httpx.HTTPStatusError as exc:     # the package client is httpx (cytir.sources.http), never requests
        if exc.response.status_code == 404:
            raise UnitExcluded("not_found") from exc
        raise
    body = json.dumps({"canton": m["canton"], "host": m["host"], "lang": m["lang"], "sysno": m["sysno"],
                       "idx_title": m["idx_title"], "response": r.json()}, ensure_ascii=False).encode("utf-8")
    done_file = ctx.options.get("done_file")
    if done_file:
        with LW._state_lock, open(done_file, "a", encoding="utf-8") as fh:
            fh.write(f"{m['canton']}|{m['lang']}|{m['sysno']}\n")
    return [Captured(unit, body, {"member": unit.key, "portal": "lexwork", "live": True,
                                  "sha256": hashlib.sha256(body).hexdigest()})]


def _portals(ctx: RunContext, root: str) -> list[str]:
    asked = [p.strip() for p in str(ctx.options.get("portals") or "").split(",") if p.strip()]
    for p in asked:
        if p in DECLINED:
            raise ValueError(f"ch_cantonal_law: portal {p!r} is consent-declined: {DECLINED[p]['reason']}")
        if p not in SHIPPED:
            raise ValueError(f"ch_cantonal_law: unknown portal {p!r} (shipped: {', '.join(SHIPPED)})")
    return asked or [p for p in SHIPPED if os.path.isfile(os.path.join(root, f"{p}.jsonl"))]


@register_family
class ChCantonalLawFamily:
    family_id = "scratch_ch_cantonal_law_exact"
    route = "documented_source_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_NATIVE_CHUNKED
    require_records = True

    def discover(self, ctx: RunContext):
        if _live(ctx):
            return _discover_live(ctx)
        root = ctx.options.get("root")
        if not root or not os.path.isdir(root):
            raise ValueError(f"{self.family_id} needs --opt root=<directory of portal tapes>")
        portals = _portals(ctx, root)
        if not portals:
            raise ValueError(f"{self.family_id}: {root!r} holds no tape ({', '.join(SHIPPED)}.jsonl)")
        units = []
        for portal in portals:
            path = os.path.join(root, f"{portal}.jsonl")
            if not os.path.isfile(path):
                raise ValueError(f"{self.family_id}: {path!r} is not a file")
            uri = pathlib.Path(path).resolve().as_uri()
            with open(path, encoding="utf-8") as fh:
                for n, line in enumerate(fh):
                    if not line.strip():
                        continue
                    host = ZH_HOST
                    try:
                        if portal == "lexwork":
                            host = json.loads(line)["host"]
                        elif portal == "cantonal_cc":
                            host = urlsplit(json.loads(line)["url"]).hostname or ZH_HOST
                    except (ValueError, KeyError, TypeError):
                        pass        # extract quarantines it; consent falls back to nothing new
                    units.append(Unit(key=f"{portal}:{n:08d}", url=f"{uri}#{n}",
                                      meta={"portal": portal, "line": line.strip()},
                                      consent_hosts=(host,)))
        if not units:
            raise RuntimeError(f"{self.family_id}: the tapes hold no unit; an empty tape is a wrong input")
        ctx.notes.update({"root": str(root), "portals": portals, "units": len(units)})
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        if unit.meta.get("live"):
            return _acquire_live(unit, ctx)
        body = unit.meta["line"].encode("utf-8")
        return [Captured(unit, body, {"member": unit.key, "portal": unit.meta["portal"],
                                      "sha256": hashlib.sha256(body).hexdigest()})]

    def extract(self, cap: Captured, ctx: RunContext):
        portal = cap.unit.meta["portal"]
        item = json.loads(cap.body.decode("utf-8"))      # not JSON: quarantined, not an origin fact
        if portal == "lexwork":
            tol = (item.get("response") or {}).get("text_of_law", {})
            if not LW.select_html(tol):
                raise UnitExcluded("no_xhtml")
            rows = LW.law_rows(item["host"], item["lang"], item["canton"], item["sysno"],
                               item.get("idx_title", ""), tol)
        elif portal == "zhlex":
            why = ZH.skip_reason(item, str(ctx.options.get("include_repealed", "")).lower() == "true")
            if why:
                raise UnitExcluded(why)
            rows = [ZH.law_row(item.get("referenceNumber"), item)]
        elif portal == "cantonal_cc":
            canton = item.get("canton")
            if canton not in CC.TARGETS:
                raise ValueError(f"ch_cantonal_law: cantonal_cc canton {canton!r} is not a target")
            from cytir.core import common as C
            from cytir.extract import html_extract as _html_extract
            url = item["url"].split("#")[0]
            return [Doc(url=url, text=item["html"], title=_html_extract.html_title(item["html"]) or "",
                        lang=CC.TARGETS[canton]["lang"],
                        extra={"cc": {"canton": canton, "digest": item.get("digest")}})]
        else:
            raise ValueError(f"ch_cantonal_law: portal {portal!r} has no dialect")
        if not rows:
            raise UnitExcluded("no_passages")
        return [Doc(url=r["url"], text=r["text"], title=r["title"], lang=r["lang"],
                    anchor=r.get("anchor", ""), extra={"row": r}) for r in rows]

    def chunk(self, doc: Doc, ctx: RunContext):
        # chunk.native: the source's own article (lexwork) or law (zhlex) is the row
        cc = doc.extra.get("cc")
        if cc is None:
            return [doc.extra["row"]]
        # cantonal_cc: the web tail's gate and extractor (the `cc` family's door), then the canton stamp
        spec = CC.TARGETS[cc["canton"]]
        doc.extra["passage_common"] = CC.passage_common(spec, ctx.today)
        doc.extra["digest"] = cc["digest"]
        return [CC.stamp_row(r, spec, cc["canton"]) for r in _extract_page(doc, ctx)]
