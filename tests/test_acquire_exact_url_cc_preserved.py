import base64
import gzip
import hashlib
import json
from pathlib import Path
import threading

from cytir.sources.exact_url import cc_preserved as A


def _digest(body: bytes, *, prefixed: bool = False) -> str:
    value = base64.b32encode(hashlib.sha1(body).digest()).decode().rstrip("=")
    return f"sha1:{value}" if prefixed else value


def _frontier(tmp_path: Path):
    path = tmp_path/"frontier.jsonl"
    path.write_text('{"ordinal":0,"url":"https://www.canada.ca/en/a.html"}\n'
                    '{"ordinal":1,"url":"https://www.canada.ca/en/b.html"}\n')
    receipt = tmp_path/"frontier-receipt.json"
    receipt.write_text(json.dumps({"schema":"aw.exact-url-frontier-receipt.v1",
        "result":{"frontier":{"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"urls":2}}}))
    return path, receipt


def test_plan_and_acquire_preserves_capture_and_explicit_miss(tmp_path: Path):
    frontier, receipt = _frontier(tmp_path); root=tmp_path/"cc"
    body=b"<html>full</html>"
    capture={"url":"https://www.canada.ca/en/a.html","fetch_time":"2026-01-01",
             "warc_filename":"x","offset":1,"length":2,"digest":_digest(body)}
    result=A.plan(frontier_path=frontier,frontier_receipt_path=receipt,root=root,
                  crawl="CC-MAIN-TEST",resolver=lambda urls:{capture["url"]:capture})
    assert result["plan"]["captures"]==1 and result["plan"]["misses"]==1
    raw=b"warc"; headers=b"HTTP/1.1 200 OK\r\n"
    run=A.acquire(root=root,artifact_root=tmp_path,rate=4,
                  fetcher=lambda c:(raw,{"http_status":206}),
                  parser=lambda x:(headers,body,body.decode()))
    assert run["counts"]=={"frontier":2,"cc_captures":1,"cc_misses":1}
    assert (root/"RUN.json").is_file() and (root/"stages/original/receipt.json").is_file()
    receipts=[json.loads(p.read_text()) for p in (root/"receipts").glob("*.json")]
    assert {row["status"] for row in receipts}=={"complete","cc_miss"}
    complete=next(row for row in receipts if row["status"]=="complete")
    assert all(artifact["uri"].startswith("artifact://cc/original/")
               for artifact in complete["artifacts"].values())
    resumed = A.acquire(
        root=root, artifact_root=tmp_path, rate=4,
        fetcher=lambda _capture: (_ for _ in ()).throw(
            AssertionError("terminal capture must not be refetched")),
        parser=lambda _raw: (_ for _ in ()).throw(
            AssertionError("terminal capture must not be reparsed")))
    assert resumed == run


def test_plan_rejects_unbound_frontier(tmp_path: Path):
    frontier, receipt = _frontier(tmp_path)
    frontier.write_text(frontier.read_text()+"\n")
    try:
        A.plan(frontier_path=frontier,frontier_receipt_path=receipt,root=tmp_path/"cc",
               crawl="X",resolver=lambda urls:{})
    except ValueError as exc:
        assert "not bound" in str(exc)
    else:
        raise AssertionError("unbound frontier was accepted")


def test_load_frontier_accepts_hash_bound_reproducible_gzip(tmp_path: Path):
    frontier = tmp_path / "frontier.jsonl.gz"
    with frontier.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as encoded:
            encoded.write(b'{"ordinal":0,"url":"https://example.test/a"}\n')
    receipt = tmp_path / "frontier-receipt.json"
    receipt.write_text(json.dumps({
        "schema": "aw.exact-url-frontier-receipt.v1",
        "result": {"frontier": {
            "sha256": hashlib.sha256(frontier.read_bytes()).hexdigest(),
            "urls": 1,
        }},
    }))
    rows, receipt_sha = A.load_frontier(frontier, receipt)
    assert rows == [{"ordinal": 0, "url": "https://example.test/a"}]
    assert receipt_sha == hashlib.sha256(receipt.read_bytes()).hexdigest()


def test_exact_only_plan_does_not_expand_or_normalize_frontier_urls(tmp_path: Path):
    frontier, receipt = _frontier(tmp_path); root = tmp_path / "cc"
    observed = []

    def resolver(urls):
        observed.extend(urls)
        return {}

    result = A.plan(frontier_path=frontier, frontier_receipt_path=receipt,
                    root=root, crawl="X", resolver=resolver,
                    exact_url_only=True)
    expected = ["https://www.canada.ca/en/a.html",
                "https://www.canada.ca/en/b.html"]
    assert observed == expected
    identity = json.loads((root / "plan/identity.json").read_text())
    assert identity["selection"] == "exact-url-newest-complete-html"
    assert result["frontier"]["urls"] == 2


def test_plan_resumes_hash_bound_legacy_html_identity(tmp_path: Path):
    frontier, receipt = _frontier(tmp_path); root = tmp_path / "cc"
    first = A.plan(frontier_path=frontier, frontier_receipt_path=receipt,
                   root=root, crawl="X", resolver=lambda _urls: {})
    identity_path = root / "plan/identity.json"
    identity = json.loads(identity_path.read_text())
    identity.pop("content_class")
    identity.pop("identity_sha256")
    identity.pop("plan_sha256")
    identity_sha = hashlib.sha256(A._json(identity)).hexdigest()
    identity_path.write_text(json.dumps({
        **identity, "identity_sha256": identity_sha,
        "plan_sha256": first["plan"]["sha256"]}, indent=2, sort_keys=True) + "\n")
    plan_receipt = root / "plan/RECEIPT.json"
    prior = json.loads(plan_receipt.read_text())
    prior["identity_sha256"] = identity_sha
    prior.pop("content_class")
    plan_receipt.write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n")
    resumed = A.plan(frontier_path=frontier, frontier_receipt_path=receipt,
                     root=root, crawl="X", resolver=lambda _urls: (_ for _ in ()).throw(
                         AssertionError("sealed plan must not be recomputed")))
    assert resumed["identity_sha256"] == identity_sha


def test_textual_surface_plan_preserves_non_html_decoded_source(tmp_path: Path):
    frontier, receipt = _frontier(tmp_path); root=tmp_path/"cc"
    body=b"<urlset><url><loc>https://example.test/x</loc></url></urlset>"
    capture={"url":"https://www.canada.ca/en/a.html","fetch_time":"2026-01-01",
             "warc_filename":"x","offset":1,"length":2,
             "digest":_digest(body, prefixed=True)}
    A.plan(frontier_path=frontier,frontier_receipt_path=receipt,root=root,
           crawl="CC-MAIN-TEST",content_class="textual-source",
           resolver=lambda urls:{capture["url"]:capture})
    A.acquire(root=root,artifact_root=tmp_path,rate=4,
              fetcher=lambda c:(b"warc",{"http_status":206}),
              parser=lambda x:(b"HTTP/1.1 200 OK\r\n",body,body.decode()))
    complete=next(json.loads(path.read_text()) for path in
                  (root/"receipts").glob("*.json")
                  if json.loads(path.read_text())["status"]=="complete")
    assert "decoded_source" in complete["artifacts"]
    assert "decoded_html" not in complete["artifacts"]


def test_default_fetch_transport_mode_does_not_mutate_between_captures(
        tmp_path: Path, monkeypatch):
    frontier, receipt = _frontier(tmp_path); root = tmp_path / "cc"
    body = b"<html>x</html>"
    captures = {
        url: {"url": url, "fetch_time": "2026-01-01", "warc_filename": "x",
              "offset": ordinal, "length": 2, "digest": _digest(body)}
        for ordinal, url in enumerate(("https://www.canada.ca/en/a.html",
                                       "https://www.canada.ca/en/b.html"), 1)}
    A.plan(frontier_path=frontier, frontier_receipt_path=receipt, root=root,
           crawl="CC-MAIN-TEST", resolver=lambda _urls: captures)
    observed = []

    def fake_fetch(_client, _capture, transport):
        observed.append(transport)
        return b"warc", {"schema": "aw.common-crawl-range-transport.v1"}

    monkeypatch.setattr(A.CC, "_default_fetch", fake_fetch)
    A.acquire(
        root=root, artifact_root=tmp_path, rate=4, transport="signed-s3",
        parser=lambda _raw: (b"HTTP/1.1 200 OK\r\n", body, body.decode()))
    assert observed == ["signed-s3", "signed-s3"]


def test_default_planner_checkpoints_each_physical_file_and_resumes(
        tmp_path: Path, monkeypatch):
    frontier, receipt = _frontier(tmp_path); root = tmp_path / "cc"
    work = [
        {"file": "a.parquet", "exact": [("k1", "https://www.canada.ca/en/a.html")],
         "prefixes": []},
        {"file": "b.parquet", "exact": [("k2", "https://www.canada.ca/en/b.html")],
         "prefixes": []},
    ]
    calls = []

    class Connection:
        def close(self):
            calls.append(("close",))

    def resolve(_con, filename, exact, allowed_mime_prefixes):
        calls.append((filename, allowed_mime_prefixes))
        requested = exact[0][1]
        record = {"url": requested, "warc_filename": filename, "offset": 1,
                  "length": 2, "digest": "sha1:X", "fetch_time": "2026-01-01",
                  "content_truncated": False, "truncation_reason": None}
        return {"complete": {requested: record}, "any": {requested: record}}

    from cytir.acquire import cc_columnar as cc_columnar
    monkeypatch.setattr(cc_columnar, "_con", lambda: Connection())
    monkeypatch.setattr(cc_columnar, "index_local_work", lambda *_args: work)
    monkeypatch.setattr(cc_columnar, "resolve_exact_file", resolve)
    first = A.plan(frontier_path=frontier, frontier_receipt_path=receipt,
                   root=root, crawl="CC-MAIN-TEST")
    assert first["plan"]["captures"] == 2
    assert first["checkpoint_manifest"]["checkpoints"] == 2
    resolver_calls = [call for call in calls if len(call) == 2]
    assert [call[0] for call in resolver_calls] == ["a.parquet", "b.parquet"]
    assert all(call[1] == ("text/html",) for call in resolver_calls)
    calls.clear()
    variants = [variant for source in
                ("https://www.canada.ca/en/a.html",
                 "https://www.canada.ca/en/b.html")
                for variant in A.CC.exact_url_variants(source)]
    resumed, manifest = A._incremental_resolve(
        root=root, crawl="CC-MAIN-TEST",
        variant_urls=variants,
        allowed_mime_prefixes=("text/html",))
    assert set(resumed) == {"https://www.canada.ca/en/a.html",
                            "https://www.canada.ca/en/b.html"}
    assert manifest["physical_files"] == 2
    assert calls == []


def test_default_planner_rejects_corrupt_checkpoint(tmp_path: Path, monkeypatch):
    from cytir.acquire import cc_columnar as cc_columnar
    root = tmp_path / "cc"
    work = [{"file": "a.parquet", "exact": [("k1", "https://example.test/")],
             "prefixes": []}]

    class Connection:
        def close(self):
            pass

    record = {"url": "https://example.test/", "warc_filename": "a", "offset": 1,
              "length": 2, "digest": "sha1:X", "fetch_time": "2026-01-01",
              "content_truncated": False, "truncation_reason": None}
    monkeypatch.setattr(cc_columnar, "_con", lambda: Connection())
    monkeypatch.setattr(cc_columnar, "index_local_work", lambda *_args: work)
    monkeypatch.setattr(
        cc_columnar, "resolve_exact_file",
        lambda *_args, **_kwargs: {"complete": {record["url"]: record},
                                  "any": {record["url"]: record}})
    A._incremental_resolve(root=root, crawl="X", variant_urls=[record["url"]],
                           allowed_mime_prefixes=("text/html",))
    checkpoint = next((root / "plan/checkpoints").glob("*.json"))
    body = json.loads(checkpoint.read_text())
    body["identity"]["crawl"] = "drift"
    checkpoint.write_text(json.dumps(body))
    try:
        A._incremental_resolve(root=root, crawl="X", variant_urls=[record["url"]],
                               allowed_mime_prefixes=("text/html",))
    except ValueError as exc:
        assert "checkpoint identity drift" in str(exc)
    else:
        raise AssertionError("corrupt index checkpoint was accepted")


def test_default_planner_parallel_workers_publish_distinct_checkpoints(
        tmp_path: Path, monkeypatch):
    from cytir.acquire import cc_columnar as cc_columnar
    root = tmp_path / "cc"
    work = [
        {"file": f"{ordinal}.parquet",
         "exact": [(f"k{ordinal}", f"https://example.test/{ordinal}")],
         "prefixes": []}
        for ordinal in range(4)]
    closed = []

    class Connection:
        def close(self):
            closed.append(True)

    def resolve(_con, filename, exact, allowed_mime_prefixes):
        requested = exact[0][1]
        record = {"url": requested, "warc_filename": filename, "offset": 1,
                  "length": 2, "digest": "sha1:X", "fetch_time": "2026-01-01",
                  "content_truncated": False, "truncation_reason": None}
        return {"complete": {requested: record}, "any": {requested: record}}

    monkeypatch.setattr(cc_columnar, "_con", lambda: Connection())
    monkeypatch.setattr(cc_columnar, "index_local_work", lambda *_args: work)
    monkeypatch.setattr(cc_columnar, "resolve_exact_file", resolve)
    resolved, manifest = A._incremental_resolve(
        root=root, crawl="X", variant_urls=[pair[1] for row in work
                                            for pair in row["exact"]],
        allowed_mime_prefixes=("text/html",), workers=4)
    assert len(resolved) == len(work) == manifest["physical_files"]
    assert len(list((root / "plan/checkpoints").glob("*.json"))) == len(work)
    assert (root / "plan/index-work.json").is_file()
    assert len({row["sha256"] for row in manifest["checkpoints"]}) == len(work)
    assert len(closed) == 1 + len(work)

    closed.clear()
    resumed, resumed_manifest = A._incremental_resolve(
        root=root, crawl="X", variant_urls=[pair[1] for row in work
                                            for pair in row["exact"]],
        allowed_mime_prefixes=("text/html",), workers=4)
    assert resumed == resolved
    assert resumed_manifest == manifest
    assert closed == []


def test_default_planner_rejects_unbounded_index_workers(tmp_path: Path):
    try:
        A._incremental_resolve(root=tmp_path, crawl="X", variant_urls=["https://x/"],
                               allowed_mime_prefixes=("text/html",), workers=5)
    except ValueError as exc:
        assert "between 1 and 4" in str(exc)
    else:
        raise AssertionError("unbounded index workers were accepted")


def test_partitioned_capture_conserves_and_finalizes_without_refetch(tmp_path: Path):
    frontier, receipt = _frontier(tmp_path); root = tmp_path / "cc"
    captures = {
        url: {"url": url, "fetch_time": "2026-01-01", "warc_filename": "x",
              "offset": ordinal, "length": 2,
              "digest": _digest(url.encode(), prefixed=ordinal == 2)}
        for ordinal, url in enumerate(("https://www.canada.ca/en/a.html",
                                       "https://www.canada.ca/en/b.html"), 1)}
    A.plan(frontier_path=frontier, frontier_receipt_path=receipt, root=root,
           crawl="CC-MAIN-TEST", resolver=lambda _urls: captures)

    def fetch(capture):
        return capture["url"].encode(), {"http_status": 206}

    parse = lambda raw: (b"HTTP/1.1 200 OK\r\n", raw, raw.decode())
    first = A.acquire(root=root, artifact_root=tmp_path, fetcher=fetch,
                      parser=parse, worker_index=0, worker_count=2,
                      partition_only=True)
    second = A.acquire(root=root, artifact_root=tmp_path, fetcher=fetch,
                       parser=parse, worker_index=1, worker_count=2,
                       partition_only=True)
    assert first["terminals"] == second["terminals"] == 1
    final = A.acquire(
        root=root, artifact_root=tmp_path,
        fetcher=lambda _capture: (_ for _ in ()).throw(
            AssertionError("partitioned terminal must not refetch")),
        parser=lambda _raw: (_ for _ in ()).throw(
            AssertionError("partitioned terminal must not reparse")))
    assert final["counts"] == {"frontier": 2, "cc_captures": 2, "cc_misses": 0}


def test_partition_worker_reuses_single_streamed_plan_digest(
        tmp_path: Path, monkeypatch):
    frontier, receipt = _frontier(tmp_path); root = tmp_path / "cc"
    body = b"https://www.canada.ca/en/a.html"
    capture = {"url": body.decode(), "fetch_time": "2026-01-01",
               "warc_filename": "x", "offset": 1, "length": 2,
               "digest": _digest(body)}
    A.plan(frontier_path=frontier, frontier_receipt_path=receipt, root=root,
           crawl="CC-MAIN-TEST", resolver=lambda _urls: {capture["url"]: capture})

    original_sha256_file = A.sha256_file

    def forbid_plan_reread(path):
        if Path(path) == root / "plan/captures.jsonl":
            raise AssertionError("acquisition re-hashed the full plan")
        return original_sha256_file(path)

    monkeypatch.setattr(A, "sha256_file", forbid_plan_reread)
    result = A.acquire(
        root=root, artifact_root=tmp_path,
        fetcher=lambda capture: (capture["url"].encode(), {}),
        parser=lambda raw: (b"h", raw, raw.decode()),
        worker_index=0, worker_count=2, partition_only=True)
    assert result["assigned"] == result["terminals"] == 1
    assert result["plan_sha256"] == json.loads(
        (root / "plan/RECEIPT.json").read_text())["plan"]["sha256"]


def test_signed_s3_fetches_overlap_with_bounded_worker_concurrency(tmp_path: Path):
    urls = [f"https://example.test/{index}" for index in range(4)]
    frontier = tmp_path / "frontier.jsonl"
    frontier.write_text("".join(
        json.dumps({"ordinal": index, "url": url}) + "\n"
        for index, url in enumerate(urls)))
    receipt = tmp_path / "frontier-receipt.json"
    receipt.write_text(json.dumps({
        "schema": "aw.exact-url-frontier-receipt.v1",
        "result": {"frontier": {
            "sha256": hashlib.sha256(frontier.read_bytes()).hexdigest(),
            "urls": len(urls)}}}))
    captures = {
        url: {"url": url, "fetch_time": "2026-01-01",
              "warc_filename": "x", "offset": index, "length": 2,
              "digest": _digest(url.encode())}
        for index, url in enumerate(urls)}
    root = tmp_path / "cc"
    A.plan(frontier_path=frontier, frontier_receipt_path=receipt, root=root,
           crawl="CC-MAIN-TEST", resolver=lambda _urls: captures)
    rendezvous = threading.Barrier(2)
    active = maximum = 0
    lock = threading.Lock()

    def fetch(capture):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        rendezvous.wait(timeout=2)
        with lock:
            active -= 1
        return capture["url"].encode(), {}

    result = A.acquire(
        root=root, artifact_root=tmp_path, fetcher=fetch,
        parser=lambda raw: (b"h", raw, raw.decode()),
        worker_index=0, worker_count=2, partition_only=True,
        fetch_concurrency=2)
    assert result["assigned"] == result["terminals"] == 2
    assert result["fetch_concurrency"] == maximum == 2


def test_fetch_concurrency_is_bounded_and_signed_s3_only(tmp_path: Path):
    for kwargs in ({"fetch_concurrency": 9},
                   {"worker_count": 4, "worker_index": 0,
                    "partition_only": True, "fetch_concurrency": 9},
                   {"transport": "https-mirror", "fetch_concurrency": 2}):
        try:
            A.acquire(root=tmp_path, artifact_root=tmp_path, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe concurrency accepted: {kwargs}")


def test_acquire_rejects_body_digest_mismatch_before_receipt(tmp_path: Path):
    frontier, receipt = _frontier(tmp_path); root = tmp_path / "cc"
    capture = {"url": "https://www.canada.ca/en/a.html",
               "fetch_time": "2026-01-01", "warc_filename": "x",
               "offset": 1, "length": 2, "digest": _digest(b"expected")}
    A.plan(frontier_path=frontier, frontier_receipt_path=receipt, root=root,
           crawl="CC-MAIN-TEST", resolver=lambda _urls: {capture["url"]: capture})
    try:
        A.acquire(root=root, artifact_root=tmp_path, rate=4,
                  fetcher=lambda _capture: (b"warc", {"http_status": 206}),
                  parser=lambda _raw: (b"HTTP/1.1 200 OK", b"different",
                                       "different"))
    except ValueError as error:
        assert "body digest mismatch" in str(error)
    else:
        raise AssertionError("mismatched Common Crawl body digest was accepted")
    bad_receipt = root / "receipts" / (
        hashlib.sha256(capture["url"].encode()).hexdigest() + ".json")
    assert not bad_receipt.exists()
