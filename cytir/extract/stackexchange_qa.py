"""Stack Exchange `Posts.xml` -> question-and-answer passages (`stackexchange_nonso_qa_p0`).

Moved out of the task-226 exploration tree (task 696): the extraction logic is UNCHANGED, so a
passage made here is byte-identical to the one the sealed staging made with the old script. Only
the XML reader differs (the standard library, not `lxml`; attribute values decode the same way).

Passage unit: one QUESTION gives up to `kans` passages, each fusing

    # <Title>
    <question body as Markdown>
    ## Accepted answer (score N)      or      ## Top answer (score N)
    <answer body as Markdown>

Score cap: a question is kept when its best answer scores at least `amin` OR the question scores
at least `qmin`. Per kept question, up to `kans` answers are emitted (the accepted one first, then
by score); every emitted answer scores at least `amin`.

Licence version by the ANSWER's `CreationDate` (ADR-039): up to 2011-04-08 CC-BY-SA-2.5, up to
2018-05-02 CC-BY-SA-3.0, later CC-BY-SA-4.0. (The row also carries its own `ContentLicense`; the
two disagree for some old posts that were relicensed. The date rule is the one the sealed passages
were made with, so it is kept.)

Attribution (master-only, ADR-039): the answer's author is recorded as a display name and a profile
URL. The display name is `OwnerDisplayName` when the row has one (deleted and anonymous owners),
else the `Users.xml` name of the `OwnerUserId` when a users map is given, else the word `user`.

Sealed stage (`python -m cytir.extract.stackexchange_qa materialize ...`): `build` folds the passages of
one site into a source-native Markdown stage (one document per question-and-answer pair, outlinks,
quarantine, original evidence, metrics), each sealed by a stage receipt. `materialize` is the pure
row-to-document step. Moved unchanged from the one-shot `materialize_stackexchange_nonso_site`.

Command line (a whole dump, for a build box):

    python -m cytir.extract.stackexchange_qa --posts-7z X.7z --users-7z X.7z --host H --out F.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import IO, Iterable, Iterator, Mapping

CUT_25 = date(2011, 4, 8)
CUT_30 = date(2018, 5, 2)
#: The dump's snapshot day, the `last_seen` of passages made by the command line.
DEFAULT_LAST_SEEN = "2024-04-07"

_TAG_RE = re.compile(r"<[^>]+>")


def lic_for(iso: str) -> str:
    d = date.fromisoformat(iso[:10])
    if d <= CUT_25:
        return "CC-BY-SA-2.5"
    if d <= CUT_30:
        return "CC-BY-SA-3.0"
    return "CC-BY-SA-4.0"


def htmlmd(s: str | None) -> str:
    if not s:
        return ""
    s = s.replace("\r", "")
    s = re.sub(r"<pre[^>]*><code>(.*?)</code></pre>",
               lambda m: "\n```\n" + html.unescape(_TAG_RE.sub("", m.group(1))) + "\n```\n", s,
               flags=re.S)
    s = re.sub(r"<code>(.*?)</code>",
               lambda m: "`" + html.unescape(_TAG_RE.sub("", m.group(1))) + "`", s, flags=re.S)
    s = re.sub(r"<li[^>]*>", "\n- ", s)
    s = re.sub(r"</p>|<br\s*/?>", "\n", s)
    s = re.sub(r"<h[1-6][^>]*>", "\n### ", s)
    s = _TAG_RE.sub("", s)
    s = html.unescape(s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def snippet25(md: str) -> str:
    words = re.sub(r"\s+", " ", md.replace("`", "").replace("#", "")).strip().split(" ")
    s = " ".join(words[:25])
    if len([w for w in words if w]) > 25:
        s += " …"
    return s


def parse_tags(t: str | None) -> list[str]:
    """Tags written `<a><b>` (the older dumps). The 2024 dumps write `|a|b|`, which this does
    NOT read: it returns `[]` for them, as the sealed passages' `tags` are."""
    if not t:
        return []
    return re.findall(r"<([^>]+)>", t) or re.findall(r"&lt;([^&]+)&gt;", t)


def iter_rows(stream: IO[bytes]) -> Iterator[Mapping[str, str]]:
    """The attribute map of every `<row>` of a Stack Exchange dump file, in file order."""
    import xml.etree.ElementTree as ET

    root = None
    for event, el in ET.iterparse(stream, events=("start", "end")):
        if event == "start":
            if root is None:
                root = el
            continue
        if el.tag == "row":
            yield el.attrib
            el.clear()
            root.clear()


def iter_rows_7z(sevenz: str, member: str) -> Iterator[Mapping[str, str]]:
    """The rows of one member of a dump archive (needs the `7z` program)."""
    p = subprocess.Popen(["7z", "e", "-so", sevenz, member], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    try:
        yield from iter_rows(p.stdout)
    finally:
        p.stdout.close()
        p.wait()


def users_map(rows: Iterable[Mapping[str, str]]) -> dict[str, str]:
    """`user id -> display name` from the rows of `Users.xml`."""
    users = {}
    for el in rows:
        uid, dn = el.get("Id"), el.get("DisplayName")
        if uid and dn:
            users[uid] = dn
    return users


def score_histogram(rows: Iterable[Mapping[str, str]]) -> tuple[dict[int, int], int]:
    """`(answers by score, number of questions)`."""
    hist: dict[int, int] = {}
    n_q = 0
    for el in rows:
        pt = el.get("PostTypeId")
        if pt == "1":
            n_q += 1
        elif pt == "2":
            sc = int(el.get("Score") or 0)
            hist[sc] = hist.get(sc, 0) + 1
    return hist, n_q


def passages(rows: Iterable[Mapping[str, str]], *, host: str,
             users: Mapping[str, str] | None = None, amin: int = 3, qmin: int = 8,
             kans: int = 2, trust: float = 0.55, lang: str = "en", tier: str = "C",
             last_seen: str = DEFAULT_LAST_SEEN, stats: dict | None = None) -> Iterator[dict]:
    """One record per kept answer, in question file order. `stats`, when given, receives the counts
    `questions`, `answers`, `kept_questions`, `kept_passages` once the iterator is exhausted."""
    users = users or {}
    questions: dict[str, dict] = {}
    ans_by_q: dict[str, list] = {}
    accepted_of: dict[str, str] = {}
    n_q = n_a = 0
    for el in rows:
        pt = el.get("PostTypeId")
        if pt == "1":
            n_q += 1
            qid = el.get("Id")
            questions[qid] = dict(title=el.get("Title") or "", body=el.get("Body") or "",
                                  score=int(el.get("Score") or 0), tags=el.get("Tags") or "")
            aa = el.get("AcceptedAnswerId")
            if aa:
                accepted_of[qid] = aa
        elif pt == "2":
            n_a += 1
            sc = int(el.get("Score") or 0)
            if sc >= amin:
                ans_by_q.setdefault(el.get("ParentId"), []).append((
                    sc, el.get("Id"), el.get("CreationDate") or "",
                    el.get("OwnerUserId"), el.get("OwnerDisplayName"), el.get("Body") or ""))
    kept_q = kept_p = 0
    for qid, q in questions.items():
        cands = ans_by_q.get(qid, [])
        best = max((c[0] for c in cands), default=-999)
        acc_id = accepted_of.get(qid)
        if not ((best >= amin) or (q["score"] >= qmin)):
            continue
        cands.sort(key=lambda c: (c[1] != acc_id, -c[0]))
        emit = cands[:kans]
        if not emit:
            continue
        kept_q += 1
        qtitle = html.unescape(q["title"])
        qbody_md = htmlmd(q["body"])
        tags = parse_tags(q["tags"])
        for (sc, aid, adate, aowner, aownerdn, abody) in emit:
            is_acc = (aid == acc_id)
            abody_md = htmlmd(abody)
            head = "Accepted answer" if is_acc else "Top answer"
            text = f"# {qtitle}\n\n{qbody_md}\n\n## {head} (score {sc})\n\n{abody_md}".strip()
            lic = lic_for(adate)
            owner_dn = (aownerdn or (users.get(aowner) if aowner else None) or "user")
            kept_p += 1
            yield {
                "url": f"https://{host}/a/{aid}",
                "title": qtitle,
                "snippet": snippet25(abody_md or qbody_md),
                "license": lic,
                "lang": lang,
                "domain": host,
                "tier": tier,
                "source_type": "community_qa",
                "trust_tier": trust,
                "last_seen": last_seen,
                "content_digest": f"se-{host}-{aid}",
                "text": text,
                "provenance": "stackexchange_n1",
                "tags": tags,
                "master_author": owner_dn,
                "master_author_url": f"https://{host}/users/{aowner}" if aowner else None,
                "master_post_id": aid,
                "master_question_id": qid,
                "master_license_version": lic.split("-")[-1],
                "master_answer_score": sc,
                "master_is_accepted": is_acc,
            }
    if stats is not None:
        stats.update(questions=n_q, answers=n_a, kept_questions=kept_q, kept_passages=kept_p)


# --- sealed stage of one site (task 1260) ---

MATERIALIZE_VERSION = "stackexchange-nonso-qa-site-materialization.v1"
MATERIALIZE_PRODUCER = "aw/extract/stackexchange_qa.py"
SOURCE_PROFILE = "community_qa"
ALLOWED_LICENSES = {"CC-BY-SA-2.5", "CC-BY-SA-3.0", "CC-BY-SA-4.0"}
_USER_ID_RE = re.compile(r"/users/(\d+)\b")


def _owner_user_id(profile_url: str | None) -> str | None:
    if not profile_url:
        return None
    match = _USER_ID_RE.search(profile_url)
    return match.group(1) if match else None


def _load_passages(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
    return rows


def materialize(rows: list[dict], *, site: str) -> dict:
    markdown_rows, outlink_rows, quarantine = [], [], []
    seen_ids: set[str] = set()
    for row in rows:
        qid = row.get("master_question_id")
        pid = row.get("master_post_id")
        if qid is None or pid is None:
            quarantine.append({"row_digest": row.get("content_digest"),
                               "reason": "missing_question_or_post_identity"})
            continue
        license_ = row.get("license")
        if license_ not in ALLOWED_LICENSES:
            quarantine.append({"question_id": qid, "post_id": pid,
                               "reason": "missing_or_unrecognized_license"})
            continue
        display_name = row.get("master_author")
        profile_url = row.get("master_author_url")
        user_id = _owner_user_id(profile_url)
        if not (display_name or user_id):
            quarantine.append({"question_id": qid, "post_id": pid,
                               "reason": "irreversible_attribution_identity_missing"})
            continue
        title = row.get("title")
        markdown = row.get("text")
        url = row.get("url")
        if not title or not markdown or not url:
            quarantine.append({"question_id": qid, "post_id": pid,
                               "reason": "principal_title_or_content_or_url_missing"})
            continue
        document_id = hashlib.sha256(
            f"stackexchange-nonso-qa-document\0{site}\0{qid}\0{pid}".encode()
        ).hexdigest()
        if document_id in seen_ids:
            quarantine.append({"question_id": qid, "post_id": pid,
                               "reason": "duplicate_document_identity"})
            continue
        seen_ids.add(document_id)
        attribution = {"owner_user_id": user_id, "owner_display_name": display_name,
                       "owner_profile_url": profile_url,
                       "user_row_present": None}
        markdown_rows.append({
            "document_id": document_id, "source_profile": SOURCE_PROFILE,
            "site": site, "lang": row.get("lang") or "en", "url": url,
            "answer_url": url,
            "source_links": [{"relation": "canonical-answer", "url": url}],
            "title": title, "tags": row.get("tags") or [],
            "question": {"post_id": str(qid)},
            "accepted_answer": {
                "post_id": str(pid), "score": row.get("master_answer_score"),
                "is_accepted": bool(row.get("master_is_accepted")),
                "license": license_, "license_version": row.get("master_license_version"),
                "attribution": attribution,
            },
            "markdown": markdown,
            "extraction": {"producer": "task226:depth226/extract_se.py",
                          "method": "source-native-html-fragment-to-markdown",
                          "full_prechunk_content": True, "content_cap": None},
        })
        links = [{"target_url": url, "anchor_or_context": "canonical answer",
                  "relation": "canonical-answer"}]
        if profile_url:
            links.append({"target_url": profile_url,
                          "anchor_or_context": "answer author",
                          "relation": "attribution-profile"})
        outlink_rows.append({"document_id": document_id, "site": site,
                             "question_id": str(qid), "post_id": str(pid),
                             "links": links})
    return {"markdown": markdown_rows, "outlinks": outlink_rows,
            "quarantine": quarantine,
            "metrics": {"input_rows": len(rows), "emitted_documents": len(markdown_rows),
                       "quarantine": len(quarantine)}}


def _stage(*, artifact_root: Path, root: Path, name: str, paths: list[Path],
          records: int, fmt: str, logical: bytes) -> dict:
    from cytir.acquire import stage_receipt as SS

    return SS.seal(
        artifact_root=artifact_root, artifact_paths=paths,
        receipt_path=root / "stages" / name / "receipt.json",
        records=records, fmt=fmt,
        logical_sha256=hashlib.sha256(logical).hexdigest(),
        producer_id=MATERIALIZE_PRODUCER, producer_version=MATERIALIZE_VERSION)


def build(*, site: str, passages_jsonl: Path, dump: Path, root: Path,
          artifact_root: Path, expected_input_rows: int,
          offset: int = 0, limit: int | None = None,
          expected_total_rows: int | None = None) -> dict:
    """Materialize one non-SO site, or (offset/limit given) one deterministic
    ≤N-row window of it -- same sequential file order every call, so windows
    are disjoint and exhaustive by construction, mirroring how the
    StackOverflow tranches window their own frontier. A plain call (offset=0,
    limit=None) is unchanged: the whole file, one pack, matching every
    already-sealed non-SO site this session.
    """
    from cytir.acquire.artifact_io import atomic_publish, sha256_file
    from cytir.extract.stackoverflow_pilot import _json_bytes, _jsonl_bytes, _publish_gzip

    if (root / "RUN.json").exists():
        raise ValueError("Stack Exchange non-SO site RUN already exists")
    if offset < 0 or (limit is not None and limit <= 0):
        raise ValueError("non-SO site window offset/limit must be non-negative/positive")
    root.mkdir(parents=True, exist_ok=True)
    all_rows = _load_passages(passages_jsonl)
    if expected_total_rows is not None and len(all_rows) != expected_total_rows:
        raise ValueError(
            f"non-SO site {site} source file row conservation failed: "
            f"got {len(all_rows)}, expected {expected_total_rows}")
    rows = all_rows[offset:offset + limit] if limit is not None else all_rows[offset:]
    if len(rows) != expected_input_rows:
        raise ValueError(
            f"non-SO site {site} input row conservation failed: "
            f"got {len(rows)}, expected {expected_input_rows}")
    result = materialize(rows, site=site)

    dump_sha256 = sha256_file(dump)
    source_passages_file_sha256 = sha256_file(passages_jsonl)
    # The externally-sourced depth226 passages file lives outside the
    # artifact root (it is scratch on a different disk, not portable/sealed
    # storage) -- seal_stage_receipt refuses to bind an artifact outside the
    # portable root, correctly, so the preserved evidence is a copy published
    # INSIDE this run's own tree, not a reference to the external path. For a
    # windowed call (offset/limit) only THIS window's rows are preserved --
    # not the whole shared source file -- so the published evidence, its
    # sha256, and its row count all describe exactly what this pack contains,
    # not the multi-hundred-MB file it was cut from (which N windows would
    # otherwise each redundantly copy).
    preserved_passages_body = _jsonl_bytes(rows)
    preserved_passages_path = root / "data/source-passages.jsonl.gz"
    _publish_gzip(preserved_passages_path, preserved_passages_body)
    original_evidence = {
        "schema": "aw.stackexchange-nonso-qa-original-evidence.v1",
        "site": site, "dump_sha256": dump_sha256,
        "source_passages_file_sha256": source_passages_file_sha256,
        "window": {"offset": offset, "limit": limit},
        "extracted_passages_sha256": hashlib.sha256(preserved_passages_body).hexdigest(),
        "extracted_passages_rows": len(rows),
        "extractor": "task 226 depth226/extract_se.py (build-box extractor)",
    }
    original_path = root / "data/original-evidence.json"
    original_body = _json_bytes(original_evidence)
    atomic_publish(original_path, original_body)

    bodies = {"markdown": _jsonl_bytes(result["markdown"]),
             "outlinks": _jsonl_bytes(result["outlinks"]),
             "quarantine": _jsonl_bytes(result["quarantine"])}
    paths = {name: root / "data" / f"{name}.jsonl.gz" for name in bodies}
    for name, body in bodies.items():
        _publish_gzip(paths[name], body)

    metrics = {"schema": "aw.stackexchange-nonso-qa-site-metrics.v1",
              "version": MATERIALIZE_VERSION, "site": site,
              "classification": "exact bounded source-native site materialization "
                                 "from task 226's already-extracted passages; "
                                 "no full-network extrapolation",
              "materialization": result["metrics"]}
    metrics_path = root / "data/metrics.json"
    metrics_body = _json_bytes(metrics)
    atomic_publish(metrics_path, metrics_body)

    stages = {
        "original": _stage(artifact_root=artifact_root, root=root, name="original",
                          paths=[original_path, preserved_passages_path],
                          records=len(rows),
                          fmt="task226-extracted-passages-jsonl+dump-sha256-evidence",
                          logical=original_body),
        "markdown": _stage(artifact_root=artifact_root, root=root, name="markdown",
                          paths=[paths["markdown"]], records=len(result["markdown"]),
                          fmt="canonical-jsonl+deterministic-gzip+source-native-QA-Markdown",
                          logical=bodies["markdown"]),
        "outlinks": _stage(artifact_root=artifact_root, root=root, name="outlinks",
                          paths=[paths["outlinks"]], records=len(result["outlinks"]),
                          fmt="canonical-jsonl+deterministic-gzip+answer-bound-QA-relations",
                          logical=bodies["outlinks"]),
        "quarantine": _stage(artifact_root=artifact_root, root=root, name="quarantine",
                            paths=[paths["quarantine"]], records=len(result["quarantine"]),
                            fmt="canonical-jsonl+deterministic-gzip+explicit-source-gate-failures",
                            logical=bodies["quarantine"]),
        "metrics": _stage(artifact_root=artifact_root, root=root, name="metrics",
                         paths=[metrics_path], records=1,
                         fmt="canonical-json+exact-bounded-yield-no-extrapolation",
                         logical=metrics_body),
    }
    run = {
        "schema": "aw.stackexchange-nonso-qa-source-native-run.v1",
        "version": MATERIALIZE_VERSION,
        "status": "source_native_site_complete_pending_profile_fidelity_chunk_measurement",
        "source_profile": SOURCE_PROFILE, "site": site,
        "counts": {"selected_source_rows": len(rows),
                  "markdown_documents": len(result["markdown"]),
                  "outlink_rows": len(result["outlinks"]),
                  "quarantine": len(result["quarantine"])},
        "content_fidelity": {"exact_selected_extraction_rows": True,
                            "full_prechunk_markdown": True,
                            "source_content_cap": None},
        "stages": stages,
        "downstream": "community_qa profile/fidelity audit and measured "
                       "chunking under chunk.generic; task 226's own site "
                       "population remains extraction-only until this runs",
    }
    atomic_publish(root / "RUN.json", _json_bytes(run))
    return run


def _materialize_main(argv: list[str]) -> int:
    """`python -m cytir.extract.stackexchange_qa materialize ...`: seal one site's passages as a stage."""
    parser = argparse.ArgumentParser(prog="stackexchange_qa materialize",
                                     description=materialize.__doc__ or build.__doc__)
    parser.add_argument("--site", required=True)
    parser.add_argument("--passages-jsonl", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-input-rows", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-total-rows", type=int, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(build(
        site=args.site, passages_jsonl=args.passages_jsonl.resolve(),
        dump=args.dump.resolve(), root=args.root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        expected_input_rows=args.expected_input_rows,
        offset=args.offset, limit=args.limit,
        expected_total_rows=args.expected_total_rows), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv and argv[0] == "materialize":
        return _materialize_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--posts-7z", required=True)
    ap.add_argument("--users-7z", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--amin", type=int, default=3)
    ap.add_argument("--qmin", type=int, default=8)
    ap.add_argument("--kans", type=int, default=2)
    ap.add_argument("--trust", type=float, default=0.55)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--tier", default="C")
    ap.add_argument("--histogram", action="store_true")
    a = ap.parse_args(argv)
    if a.histogram:
        hist, n_q = score_histogram(iter_rows_7z(a.posts_7z, "Posts.xml"))
        tot = sum(hist.values())
        print(f"# {a.host} answers total={tot:,} questions={n_q:,}")
        for th in [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50]:
            print(f"  ans score>={th:3d}: {sum(v for k, v in hist.items() if k >= th):,}")
        return 0
    users = users_map(iter_rows_7z(a.users_7z, "Users.xml"))
    stats: dict = {}
    with open(a.out, "w") as out:
        for rec in passages(iter_rows_7z(a.posts_7z, "Posts.xml"), host=a.host, users=users,
                            amin=a.amin, qmin=a.qmin, kans=a.kans, trust=a.trust, lang=a.lang,
                            tier=a.tier, stats=stats):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sys.stderr.write(f"[{a.host}] q={stats['questions']:,} a={stats['answers']:,} "
                     f"kept_q={stats['kept_questions']:,} kept_p={stats['kept_passages']:,}\n")
    with open(a.out + ".manifest.json", "w") as f:
        json.dump(dict(host=a.host, questions_scanned=stats["questions"],
                       answers_scanned=stats["answers"], kept_questions=stats["kept_questions"],
                       kept_passages=stats["kept_passages"], amin=a.amin, qmin=a.qmin,
                       kans=a.kans, out=a.out), f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
