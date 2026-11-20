"""The Stack Overflow selection rule is defined once (`cytir.extract.stackoverflow_qa`).

The in-memory door (`select_pairs` + `admit`) and the SQLite frontier builder
(`cytir.sources.stackoverflow.frontier`, whose WHERE clause comes from `eligible_sql()`) must pick the
same questions. This fixture holds the edge pairs: the score bar, a missing accepted answer, a
wrong-parent answer, each licence, owner by name / by user row / by neither."""
from io import BytesIO
import sqlite3

import pytest

pytest.importorskip("lxml")

from cytir.extract import stackoverflow_qa as E
from cytir.sources.stackoverflow import frontier as F

L25, L30, L40 = "CC BY-SA 2.5", "CC BY-SA 3.0", "CC BY-SA 4.0"


def _q(i, a, score=10, lic=L40, owner='OwnerUserId="1"'):
    return (f'<row Id="{i}" PostTypeId="1" AcceptedAnswerId="{a}" Score="{score}" '
            f'ContentLicense="{lic}" {owner} CreationDate="2020-01-01" />')


def _a(i, parent, lic=L40, owner='OwnerUserId="2"'):
    return (f'<row Id="{i}" PostTypeId="2" ParentId="{parent}" Score="1" '
            f'ContentLicense="{lic}" {owner} CreationDate="2020-01-02" />')


ROWS = [
    _q(1, 2), _a(2, 1),                                              # ok, boundary score 10
    _q(3, 4, score=9), _a(4, 3),                                     # below the bar
    _q(5, 6, score=11, lic=L25), _a(6, 5, lic=L30),                  # old licences admitted
    _q(7, 8, lic="UNKNOWN"), _a(8, 7),                               # question licence refused
    _q(9, 10), _a(10, 9, lic=""),                                    # answer licence missing
    _q(11, 12), _a(12, 99),                                          # answer of another question
    _q(13, 14), _a(14, 13, owner='OwnerUserId="77"'),                # owner id with no user row
    _q(15, 16, owner='OwnerDisplayName="deleted-q"'),
    _a(16, 15, owner='OwnerDisplayName="deleted-a"'),                # names on the posts
    _q(17, 18), _a(18, 17, owner='OwnerDisplayName=""'),             # empty name, no id
    _q(19, 99),                                                      # accepted answer not in dump
]
USERS = '<row Id="1" DisplayName="q" />\n<row Id="2" DisplayName="a" />'


def _memory(tmp_path):
    posts = tmp_path / "Posts.xml"
    posts.write_text("<posts>\n" + "\n".join(ROWS) + "\n</posts>\n")
    pairs, _ = E.select_pairs(posts)
    users = E.read_users(BytesIO(USERS.encode()), E.owner_ids(pairs))
    admitted, _ = E.admit(pairs, users)
    return {E._attrs(q)["Id"] for q, _ in admitted}


def _sqlite():
    conn = sqlite3.connect(":memory:")
    F.consume_posts(BytesIO(("<posts>\n" + "\n".join(ROWS) + "\n</posts>").encode()), conn)
    F.consume_users(BytesIO(("<users>\n" + USERS + "\n</users>").encode()), conn,
                    F.requested_user_ids(conn))
    return {str(F._row(c)["question_id"]) for c in F._eligible_rows(conn)}


def test_both_paths_pick_the_same_questions_on_edge_pairs(tmp_path):
    memory, sql = _memory(tmp_path), _sqlite()
    assert memory == sql
    assert memory == {"1", "5", "15"}


def test_thresholds_are_the_shared_constants():
    assert F.MIN_SCORE is E.MIN_SCORE and F.ALLOWED_LICENSES is E.ALLOWED_LICENSES
    assert E.score_ok(E.MIN_SCORE) and not E.score_ok(E.MIN_SCORE - 1) and not E.score_ok(None)
    _, params = E.eligible_sql()
    assert params == (*E.ALLOWED_LICENSES, *E.ALLOWED_LICENSES)
