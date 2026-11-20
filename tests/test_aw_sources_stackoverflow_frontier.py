from io import BytesIO
import gzip
import json
from pathlib import Path
import sqlite3

from cytir.sources.stackoverflow.frontier import (
    consume_posts, consume_users, finalize, requested_user_ids,
)


def _xml(rows: list[str]) -> BytesIO:
    return BytesIO(("<posts>\n" + "\n".join(rows) + "\n</posts>\n").encode())


def test_exact_score_license_answer_attribution_frontier_and_tranche(tmp_path: Path):
    connection = sqlite3.connect(tmp_path / "selection.sqlite")
    posts = _xml([
        '<row Id="1" PostTypeId="1" AcceptedAnswerId="2" Score="10" ContentLicense="CC BY-SA 4.0" OwnerUserId="1" CreationDate="2020-01-01" />',
        '<row Id="2" PostTypeId="2" ParentId="1" Score="5" ContentLicense="CC BY-SA 4.0" OwnerUserId="2" CreationDate="2020-01-02" />',
        '<row Id="3" PostTypeId="1" AcceptedAnswerId="4" Score="9" ContentLicense="CC BY-SA 4.0" OwnerUserId="1" />',
        '<row Id="4" PostTypeId="2" ParentId="3" Score="2" ContentLicense="CC BY-SA 4.0" OwnerUserId="2" />',
        '<row Id="5" PostTypeId="1" AcceptedAnswerId="6" Score="20" ContentLicense="UNKNOWN" OwnerUserId="1" />',
        '<row Id="6" PostTypeId="2" ParentId="5" Score="3" ContentLicense="CC BY-SA 4.0" OwnerUserId="2" />',
        '<row Id="7" PostTypeId="1" AcceptedAnswerId="8" Score="11" ContentLicense="CC BY-SA 3.0" OwnerDisplayName="deleted-q" />',
        '<row Id="8" PostTypeId="2" ParentId="7" Score="1" ContentLicense="CC BY-SA 3.0" OwnerDisplayName="deleted-a" />',
        '<row Id="9" PostTypeId="1" AcceptedAnswerId="10" Score="10" ContentLicense="CC BY-SA 4.0" OwnerUserId="99" />',
        '<row Id="10" PostTypeId="2" ParentId="9" Score="1" ContentLicense="CC BY-SA 4.0" OwnerUserId="2" />',
        '<row Id="11" PostTypeId="1" AcceptedAnswerId="12" Score="50" ContentLicense="CC BY-SA 4.0" OwnerUserId="1" />',
    ])
    counts = consume_posts(posts, connection)
    assert counts == {"questions_score10_with_accepted_id": 5,
                      "referenced_answers_present": 4}
    wanted = requested_user_ids(connection)
    assert wanted == {1, 2, 99}
    users = _xml([
        '<row Id="1" DisplayName="q-user" />',
        '<row Id="2" DisplayName="a-user" />',
    ])
    assert consume_users(users, connection, wanted) == {
        "requested_user_ids": 3, "found_user_rows": 2,
        "missing_user_rows": 1,
    }
    root = tmp_path / "out"; root.mkdir()
    result = finalize(connection, root=root, tranche_size=1)
    assert result["frontier_records"] == 2
    assert result["tranche_records"] == 1
    assert result["full_unique_user_rows"] == 2
    assert result["full_expected_source_rows"] == 6
    with gzip.open(result["frontier_path"], "rt") as source:
        rows = [json.loads(line) for line in source]
    assert {row["question_id"] for row in rows} == {1, 7}
    assert all(row["question_score"] >= 10 for row in rows)
