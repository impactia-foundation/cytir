"""`news_daily` -- the daily news lane as a declared door on the family Protocol (task 1388).

The lane is `cytir.news` (`daily` -> `compose` -> the promotion queue), driven by the timer
`aw-news-daily-queue`. This adapter is a FACADE over it (ruling 60(c), 2026-10-28): every stage
delegates to the existing `cytir.news` function, with the lane's own arguments. No logic is copied, no
stage re-implemented, and the timer and the release path do not change.

STAGE MAP (the unit is one (UTC day, publisher host), `day-YYYY-MM-DD@HOST`; task 1446):

    discover      this module: the day, and one unit per publisher host of `cytir.news.publishers`
                  (daily_required); each unit carries its one host as its consent host
    consent       FRAMEWORK: each unit's host, resolved live per origin (`plan` stops here); a declined
                  host excludes ITS unit only, the other hosts' units proceed
    acquire       `cytir.news.daily.run_stage`  (discover + fetch + embed into the staging collection),
                  run ONCE per run for the whole day (the stage script re-resolves consent per host
                  itself); each unit's captured body is the day's staged-point counts
                  (`daily.staged_points`) cut to the unit host's arm
    compose_day   `cytir.news.daily.run_one_day` with `--no-stage`: pack, merge, consent stamp,
                  tombstones, `compose.write_pack` into the promotion queue (never applies)
    seal_arm      `cytir.news.seal_lane_stages.seal_arm`: the arm's preserved-stage evidence

`extract` and `chunk` (task 2783) take a PAGE capture (`text/html`; meta carries `host_meta` and
`fetch_meta`) and delegate to `cytir.news.page_rows.page_rows`, the function `sitemap_ingest.process_host`
runs per page. `acquire` still returns the counts capture: the lane fetches and embeds inside its
stage script, so no page bytes pass through it. The lane writes no L1 of its own, so there is no `l1()`.

`acquire` fetches from the publishers and needs the operator box's settings: it refuses unless
`--opt execute=1` is given. `cyt source plan news_daily` performs no article fetch; it makes live consent probes
(robots/AIPREF/TDMRep) per publisher host, like every adapter's plan.

CONSENT AND LICENCE are read from the family's contract row, never written here. The row freezes
neither flag (`search_ok` and `train_ok` are `null`: each host's current verdict decides, stamped
by the lane at compose time); `discover` copies the row's terms into the run notes.

Options (`--opt KEY=VALUE`): `day` (YYYY-MM-DD; default the day before `ctx.today`), `execute`
(`1` to let `acquire` run the stage).
"""
from __future__ import annotations

import datetime as _dt
import json

from cytir.sources.family import CHUNK_DECLARED, Captured, RunContext, Unit
from cytir.sources.registry import register_family


def contract_terms(ctx: RunContext) -> dict:
    """The consent and licence terms of the family's contract row, as the run notes carry them."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError("news_daily: the run has no contract row with consent and licence")
    consent, licence = row["consent"], row["licence"]
    return {"search_ok": consent.get("search_ok"), "train_ok": consent.get("train_ok"),
            "consent_basis": consent.get("basis"), "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


def _day(ctx: RunContext) -> str:
    day = ctx.options.get("day")
    if not day:
        day = (_dt.date.fromisoformat(ctx.today) - _dt.timedelta(days=1)).isoformat()
    _dt.date.fromisoformat(day)          # a malformed day refuses here, before any stage
    return day


def lane_args(argv: list[str] | None = None):
    """The lane's own parsed arguments (`cytir.news.daily.build_parser`), defaults included."""
    from cytir.news import daily
    return daily.build_parser().parse_args(argv or [])


@register_family
class NewsDailyFamily:
    family_id = "news_daily"
    route = "direct"
    direct_crawl = True
    chunk_policy_id = CHUNK_DECLARED
    #: no PARTIAL marker (task 2783): `extract`/`chunk` serve page captures; `acquire` still
    #: returns the lane's counts capture, because the lane fetches inside its stage script

    def discover(self, ctx: RunContext):
        from cytir.news import publishers
        ctx.notes["contract_terms"] = contract_terms(ctx)
        day = _day(ctx)
        hosts = publishers.hosts(daily=True)
        ctx.notes.update({"day": day, "hosts": list(hosts)})
        arms = publishers.host_arm()
        return [Unit(key=f"day-{day}@{h}", url=f"https://{h}/", consent_hosts=(h,),
                     meta={"day": day, "host": h, "arm": arms[h]}) for h in hosts]

    def acquire(self, unit: Unit, ctx: RunContext):
        """Stage the day (`daily.run_stage`).

        With `--opt execute=1` through `cyt source run`, acquire stages and embeds the day for real;
        `extract` raises ValueError on this counts capture (page-level `text/html` captures are the
        real stage input), so the unit is still recorded `extract_failed` although the stage
        succeeded and its rows stay in the staging collection. Use the lane's
        timer path for production runs.
        """
        if str(ctx.options.get("execute")) != "1":
            raise ValueError("news_daily acquire fetches from the publishers on the operator box; "
                             "pass --opt execute=1 (the timer aw-news-daily-queue runs it daily)")
        from cytir.news import daily
        day, arm = unit.meta["day"], unit.meta["arm"]
        if ctx.notes.get("staged_day") != day:       # one stage per day, whatever the unit count
            args = lane_args()
            daily.run_stage(day, out_log=daily.W / f"task640-daily-stage-{day.replace('-', '')}.out",
                            min_rows=args.min_rows, min_publisher_rows=args.min_publisher_rows,
                            host_rate=args.host_rate, global_rate=args.global_rate,
                            workers=args.workers, window_months=args.window_months,
                            hn_since_days=args.hn_since_days,
                            max_child_sitemaps=args.max_child_sitemaps,
                            egress_iface=args.egress_iface)
            ctx.notes["staged_day"] = day
        staged = daily.staged_points(day)
        n = staged["arms"].get(arm, 0)
        cut = {"arms": {arm: n}, "manifest_path": staged["manifest_path"], "total": n}
        return [Captured(unit, json.dumps(cut, sort_keys=True).encode(),
                         {"status": 200, "content_type": "application/json",
                          "source_url": unit.url, "day": day, "host": unit.meta["host"]})]

    def extract(self, cap: Captured, ctx: RunContext):
        """A page capture (`text/html` + the fetch meta) becomes one Doc; the lane's own counts
        capture (`application/json`, from `acquire`) has no text and stays a refusal."""
        from cytir.sources.family import Doc
        if not str(cap.meta.get("content_type") or "").startswith("text/html"):
            raise ValueError("news_daily extract: only a page capture (text/html with the fetch "
                             "meta) is extractable; the lane's counts capture is not")
        from cytir.sources.families.web import decode_html
        html = decode_html(cap.body, cap.meta.get("content_type"))
        return [Doc(url=cap.unit.url, text=html, title="", lang=cap.meta.get("lang") or "",
                    extra={"fetch_meta": cap.meta.get("fetch_meta") or {},
                           "host_meta": cap.meta["host_meta"]})]

    def chunk(self, doc, ctx: RunContext):
        """The page's finished rows: `cytir.news.page_rows.page_rows`, the function
        `sitemap_ingest.process_host` runs per page."""
        from cytir.news.page_rows import page_rows
        host_meta = dict(doc.extra["host_meta"])
        if doc.lang and not host_meta.get("lang_default"):
            host_meta["lang_default"] = doc.lang
        return page_rows(doc.text, doc.url, doc.extra["fetch_meta"], host_meta,
                         {"now_ref": _dt.date.fromisoformat(ctx.today)})

    def compose_day(self, ctx: RunContext) -> dict:
        """Pack, merge, stamp consent, tombstone and queue the staged day (`run_one_day`).

        Reachable from Python only; `run_family` does not call this stage.
        """
        from cytir.news import daily
        return daily.run_one_day(_day(ctx), lane_args(["--no-stage"]))

    def seal_arm(self, **kwargs) -> dict:
        """Seal one arm's preserved stages (`seal_lane_stages.seal_arm`, same keyword arguments).

        Reachable from Python only; `run_family` does not call this stage.
        """
        from cytir.news import seal_lane_stages
        return seal_lane_stages.seal_arm(**kwargs)
