"""Stack Overflow `Posts.xml` -> question-and-accepted-answer documents (`stackoverflow_qa`).

The selection of the old production frontier builder (task 631)
and the source-native renderer (`cytir.extract.stackoverflow_pilot.materialize`, which the production
tranche materializer calls) put on one door, so that a stranger can run the family from the data
dump alone. Nothing here changes a rule; the rendering function is the one the live tranches used.

Selection (the frontier's own rule, ranked once and immutable there; here every member is taken,
in question-id order):

  * a question (`PostTypeId` 1) with an accepted answer and a score of at least `MIN_SCORE` (10);
  * the accepted answer is in the dump, and is the answer of that question;
  * both posts carry one of the three `ContentLicense` values the family admits (CC BY-SA 2.5,
    3.0 and 4.0);
  * both posts have an owner identity: a display name on the row, or an `OwnerUserId` whose row is
    in `Users.xml`.

A pair that the rule admits but the renderer refuses (a missing licence, an empty body, an owner
the renderer cannot name) is counted by reason, never dropped silently: the caller reads the counts.

`Posts.xml` is read in two streaming passes (the dump is not in id order: an answer can come before its
question). Memory: the selected rows are held as their raw bytes until they are rendered. A whole-dump
run holds a few GB (about 550,000 pairs) and the door then holds the passages of the unit; `ids=` takes
only the named questions (a tranche) and bounds both.

Persons: `Users.xml` is read only for the owners of the selected posts, and only for `DisplayName`.
Names reach the embedded text (the live rows have them) and the master-only part of a row; they do
not reach the snippet or any compact field.

Heavy libraries (`lxml`) are imported inside the functions that need them.
"""
from __future__ import annotations

from typing import BinaryIO, Iterable

SITE = "stackoverflow.com"
MIN_SCORE = 10
ALLOWED_LICENSES = ("CC BY-SA 2.5", "CC BY-SA 3.0", "CC BY-SA 4.0")
CELL = "accepted_qscore_10_plus"
#: the three reasons a pair is counted out before the renderer sees it
REASON_NO_ANSWER = "accepted_answer_not_in_dump"
REASON_LICENCE = "licence_not_admitted"
REASON_OWNER = "owner_identity_missing"


# --- the selection rule, defined once -------------------------------------------------------
# The in-memory door (`select_pairs` + `admit`) and the SQLite frontier builder
# (`cytir.sources.stackoverflow.frontier`) both read these; `eligible_sql()` is generated from the
# same constants. tests/test_aw_stackoverflow_selection_rule.py proves the two agree on edge pairs.


def score_ok(score: int | None) -> bool:
    """The question's score bar (a missing score counts as 0)."""
    return (score or 0) >= MIN_SCORE


def licence_ok(licence: str | None) -> bool:
    return licence in ALLOWED_LICENSES


def owner_ok(display_name: str | None, user_present: bool) -> bool:
    """An owner identity: a display name on the post, or an owner id whose `Users.xml` row exists."""
    return bool(display_name) or bool(user_present)


def eligible_sql() -> tuple[str, tuple]:
    """The same rule as one SQL condition over the frontier builder's tables (`q` questions,
    `a` answers, `qu`/`au` the owners' user rows); returns `(condition, licence_params)` where the
    params are for BOTH licence lists (question, then answer). The score bar is applied when
    questions are inventoried (`score_ok`)."""
    marks = ",".join("?" for _ in ALLOWED_LICENSES)
    cond = f"""a.parent_question_id=q.question_id
        AND q.license IN ({marks}) AND a.license IN ({marks})
        AND ((q.owner_user_id IS NOT NULL AND qu.user_id IS NOT NULL)
             OR coalesce(q.owner_display_name,'')!='')
        AND ((a.owner_user_id IS NOT NULL AND au.user_id IS NOT NULL)
             OR coalesce(a.owner_display_name,'')!='')"""
    return cond, (*ALLOWED_LICENSES, *ALLOWED_LICENSES)


def _int(value) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def read_ids(lines: Iterable[str]) -> set[str]:
    """Question ids from a text file, one per line (blank lines and `#` comments ignored)."""
    out = set()
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#"):
            if not s.isdigit():
                raise ValueError(f"not a question id: {s!r}")
            out.add(s)
    if not out:
        raise ValueError("the question-id file names no question")
    return out


def select_pairs(posts_path, *, ids: set[str] | None = None):
    """Read `Posts.xml` in two streaming passes; return `(pairs, counts)`.

    The dump's rows are NOT in id order (an answer can come before its question), so one pass
    cannot know which answers it needs: pass 1 takes the questions, pass 2 the accepted answers.

    `pairs` is a list of `(question_raw, answer_raw)` byte lines of the accepted pairs whose
    question scores at least `MIN_SCORE`, in question-id order. The licence and owner rules are
    applied later (`admit`), because the owner rule needs `Users.xml`."""
    import re
    from lxml import etree

    parser = etree.XMLParser(huge_tree=True, recover=False)
    wanted: dict[str, str] = {}        # accepted answer id -> question id
    questions: dict[str, bytes] = {}
    answers: dict[str, bytes] = {}
    counts = {"rows": 0, "questions": 0, "questions_with_accepted_answer": 0,
              "questions_selected": 0}
    with open(posts_path, "rb") as posts:
        for raw in posts:
            stripped = raw.strip()
            if not stripped.startswith(b"<row "):
                continue
            counts["rows"] += 1
            # one XML parse per question, not per row: a quote inside a body is escaped
            # (`&quot;`), so a literal ` PostTypeId="` can only be the attribute itself
            m = re.search(rb' PostTypeId="(\d+)"', stripped)
            if not m:
                counts["rows_without_type"] = counts.get("rows_without_type", 0) + 1
                continue
            if m.group(1) != b"1":
                continue
            el = etree.fromstring(stripped, parser=parser)
            counts["questions"] += 1
            accepted = el.get("AcceptedAnswerId")
            if not accepted:
                continue
            counts["questions_with_accepted_answer"] += 1
            if not score_ok(_int(el.get("Score"))):
                continue
            qid = el.get("Id")
            if ids is not None and qid not in ids:
                continue
            counts["questions_selected"] += 1
            wanted[accepted] = qid
            questions[qid] = raw
    with open(posts_path, "rb") as posts:
        for raw in posts:
            m = re.match(rb'\s*<row Id="(\d+)"', raw)
            if not m or m.group(1).decode() not in wanted:
                continue
            el = etree.fromstring(raw.strip(), parser=parser)
            aid = el.get("Id")
            if el.get("PostTypeId") == "2" and el.get("ParentId") == wanted[aid]:
                answers[aid] = raw
    pairs, missing = [], 0
    for aid, qid in sorted(wanted.items(), key=lambda kv: int(kv[1])):
        if aid in answers:
            pairs.append((questions[qid], answers[aid]))
        else:
            missing += 1
    counts["pairs_with_answer"] = len(pairs)
    counts[REASON_NO_ANSWER] = missing
    if ids is not None:
        counts["ids_not_selected"] = len(ids - set(questions))
    return pairs, counts


def _attrs(raw: bytes) -> dict[str, str]:
    from lxml import etree

    el = etree.fromstring(raw.strip(), etree.XMLParser(huge_tree=True, recover=False))
    return {str(k): str(v) for k, v in el.attrib.items()}


def owner_ids(pairs) -> set[str]:
    """`OwnerUserId` values the names of which `Users.xml` must give (rows without a display
    name on the post itself)."""
    out = set()
    for q, a in pairs:
        for raw in (q, a):
            at = _attrs(raw)
            if at.get("OwnerUserId") and not at.get("OwnerDisplayName"):
                out.add(at["OwnerUserId"])
    return out


def admit(pairs, users: dict) -> tuple[list, dict]:
    """The frontier's licence and owner rules. Returns `(admitted, counts_by_reason)`."""
    admitted, why = [], {REASON_LICENCE: 0, REASON_OWNER: 0}
    for q, a in pairs:
        qa, aa = _attrs(q), _attrs(a)
        if not (licence_ok(qa.get("ContentLicense")) and licence_ok(aa.get("ContentLicense"))):
            why[REASON_LICENCE] += 1
            continue
        if not all(owner_ok(x.get("OwnerDisplayName"),
                            bool(x.get("OwnerUserId") and x["OwnerUserId"] in users))
                   for x in (qa, aa)):
            why[REASON_OWNER] += 1
            continue
        admitted.append((q, a))
    return admitted, why


def read_users(stream: BinaryIO, wanted: set[str]) -> dict:
    """The records (raw row, attributes; the shape of `stackoverflow_pilot.select_users`) of the
    wanted owners, by user id. Rows are matched on their `Id` before they are parsed: the dump holds
    about 20 million users."""
    import re
    from cytir.extract import stackoverflow_pilot as P

    found = {}
    for raw in stream:
        stripped = raw.strip()
        if not stripped.startswith(b"<row "):
            continue
        m = re.search(rb' Id="(\d+)"', stripped)
        if m and m.group(1).decode() in wanted:
            found[m.group(1).decode()] = P._raw_record(raw, _attrs(raw))
    return dict(sorted(found.items(), key=lambda item: int(item[0])))


def render(pair, users: dict) -> tuple[dict | None, dict | None]:
    """One pair -> `(markdown_row, quarantine_row)` through the production renderer."""
    from cytir.extract import stackoverflow_pilot as P

    q, a = pair
    qr, ar = P._raw_record(q, _attrs(q)), P._raw_record(a, _attrs(a))
    selection = {"rows": [{
        "cell": CELL, "question": qr, "accepted_answer": ar, "top_answers_score_ge_3": [],
        "comparison_prior_amin3_qmin8_kans2": {
            "included": True, "question_min_score": 8, "answer_min_score": 3,
            "answer_cap": 2, "selected_answer_ids": []}}]}
    out = P.materialize(selection, {"users": users})
    return (out["markdown"][0] if out["markdown"] else None,
            out["quarantine"][0] if out["quarantine"] else None)
