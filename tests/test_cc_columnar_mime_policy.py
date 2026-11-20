import pytest

pytest.importorskip("duckdb")

import cytir.acquire.cc_columnar as subject


class FakeResult:
    def __init__(self, owner):
        self.owner = owner

    def fetchall(self):
        return []


class FakeConnection:
    def __init__(self):
        self.queries = []

    def execute(self, query):
        self.queries.append(query)
        return FakeResult(self)

    def executemany(self, query, rows):
        self.queries.append(query)


def test_resolve_textual_mime_policy_reaches_single_batched_query(monkeypatch):
    connection = FakeConnection()
    monkeypatch.setattr(subject, "url_surtkey", lambda url: "com,example)/agents.md")
    monkeypatch.setattr(subject, "file_bounds", lambda crawl, con: [
        ("s3://commoncrawl/index.parquet", "com,example)", "com,example)z")])
    assert subject.resolve(
        "CC-MAIN-TEST", ["https://example.com/agents.md"], con=connection,
        require_complete=True,
        allowed_mime_prefixes=("text/html", "text/plain", "application/xml")) == {}
    query = connection.queries[-1]
    assert "content_mime_type LIKE 'text/plain%'" in query
    assert "content_mime_type LIKE 'application/xml%'" in query
    assert "content_truncated IS NULL" in query


def test_resolve_rejects_unsafe_mime_policy_before_query():
    try:
        subject.resolve("CC-MAIN-TEST", ["https://example.com/"], con=FakeConnection(),
                        allowed_mime_prefixes=("text/html' OR 1=1 --",))
    except ValueError as exc:
        assert "MIME-prefix" in str(exc)
    else:
        raise AssertionError("unsafe MIME prefix accepted")
