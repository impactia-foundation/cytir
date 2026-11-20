"""Task 510: offline replay tests for the HN discussion ingester."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import threading

import httpx

from cytir.sources.hn import discussion_ingest as subject


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hn_discussion_recorded_firebase_items.json"
API_BASE = "https://hacker-news.firebaseio.com/v0/"
ITEM_TEMPLATE = "https://news.ycombinator.com/item?id={item_id}"
FIXED_NOW = dt.datetime(2026, 9, 3, 12, 0, tzinfo=dt.timezone.utc)


class RecordedResponse:
    def __init__(self, body: bytes, status_code: int = 200):
        self.content = body
        self.status_code = status_code


class RecordedFirebaseClient:
    """Replay retained item payloads; any unexpected URL fails the test."""

    def __init__(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.items = fixture["items"]
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def get(self, url: str) -> RecordedResponse:
        with self._lock:
            self.calls.append(url)
        prefix = API_BASE + "item/"
        assert url.startswith(prefix) and url.endswith(".json"), url
        item_id = url[len(prefix):-len(".json")]
        assert item_id in self.items, f"unrecorded network request: {url}"
        body = subject.canonical_json(self.items[item_id]).encode("utf-8")
        return RecordedResponse(body)


def write_input(path: Path, *item_ids: int) -> None:
    path.write_text("".join(json.dumps({"id": value}) + "\n"
                            for value in item_ids), encoding="utf-8")


def read_records(output: Path) -> list[dict]:
    rows = []
    for path in sorted(output.glob("records-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return rows


def run_recorded(input_path: Path, output: Path, client: RecordedFirebaseClient,
                 **overrides):
    options = {
        "input_path": input_path,
        "output_dir": output,
        "api_base_url": API_BASE,
        "item_url_template": ITEM_TEMPLATE,
        "item_cap": 20,
        "shard_size": 2,
        "workers": 4,
        "client": client,
        "sleep": lambda _seconds: None,
        "clock": lambda: FIXED_NOW,
        "lang_detector": lambda _text, _url: ("en", True),
    }
    options.update(overrides)
    return subject.ingest(**options)


def test_rendering_is_bfs_block_separated_depth_limited_and_char_truncated(tmp_path):
    source = tmp_path / "ids.jsonl"
    output = tmp_path / "out"
    write_input(source, 100)
    client = RecordedFirebaseClient()

    run_recorded(source, output, client, max_depth=2, comment_chars=500)
    record = read_records(output)[0]
    blocks = record["text"].split("\n\n")

    assert blocks[0:3] == [
        "Recorded & ordered thread",
        "Story text & context.",
        "first & top-level",
    ]
    assert subject.DEFAULT_COMMENT_CHARS == 600
    assert len(blocks[3]) == 500
    assert blocks[4:] == ["nested under first", "nested under second"]
    assert "depth three" not in record["text"]
    assert "alpha_user" not in record["text"]
    assert record["_master"]["hn_comment_provenance"][0]["by"] == "alpha_user"
    # Deleted 106 is fetched to learn its status; its child is not fetched.
    assert API_BASE + "item/106.json" in client.calls
    assert API_BASE + "item/107.json" not in client.calls


def test_linked_story_is_plain_outlink_and_never_requested(tmp_path):
    source = tmp_path / "ids.jsonl"
    output = tmp_path / "out"
    write_input(source, 100)
    client = RecordedFirebaseClient()

    run_recorded(source, output, client, max_comments=1)
    record = read_records(output)[0]

    assert record["url"] == ITEM_TEMPLATE.format(item_id=100)
    assert record["outlink"] == "https://linked.example/article?do-not-fetch=1"
    assert all(call.startswith(API_BASE + "item/") for call in client.calls)
    assert record["outlink"] not in client.calls
    manifest = json.loads((output / "MANIFEST.json").read_text())
    assert manifest["source"]["linked_pages_fetched"] == 0


def test_hard_item_cap_limits_input_story_fetches(tmp_path):
    source = tmp_path / "ids.jsonl"
    output = tmp_path / "out"
    write_input(source, 200, 300, 100)
    client = RecordedFirebaseClient()

    manifest = run_recorded(source, output, client, item_cap=2, shard_size=1)

    assert [row["hn_object_id"] for row in read_records(output)] == ["200", "300"]
    assert API_BASE + "item/100.json" not in client.calls
    assert manifest["counts"]["input_item_ids"] == 3
    assert manifest["counts"]["selected_item_ids"] == 2
    assert manifest["counts"]["capped_item_ids"] == 1
    assert manifest["counts"]["shards"] == 2


def test_manifest_hashes_every_shard_and_fetch_receipt(tmp_path):
    source = tmp_path / "ids.jsonl"
    output = tmp_path / "out"
    write_input(source, 100, 200)
    client = RecordedFirebaseClient()

    manifest = run_recorded(source, output, client, shard_size=1, max_comments=2)

    assert manifest["counts"]["records"] == 2
    assert manifest["counts"]["shards"] == 2
    assert manifest["counts"]["fetch_receipts"] == 4
    assert {artifact["kind"] for artifact in manifest["artifacts"]} == {
        "record_shard", "fetch_receipt"}
    for artifact in manifest["artifacts"]:
        body = (output / artifact["path"]).read_bytes()
        assert artifact["bytes"] == len(body)
        assert artifact["sha256"] == hashlib.sha256(body).hexdigest()
    for receipt_path in (output / "receipts").glob("*.json"):
        receipt = json.loads(receipt_path.read_text())
        expected_body = subject.canonical_json(client.items[str(receipt["id"])]).encode()
        assert receipt["api_bytes"] == len(expected_body)
        assert receipt["sha256"] == hashlib.sha256(expected_body).hexdigest()


def test_default_selection_budget_depth_and_comment_ceilings(tmp_path):
    source = tmp_path / "ids.jsonl"
    output = tmp_path / "out"
    budget_story, count_story, depth_story = 1000, 2000, 3000
    write_input(source, budget_story, count_story, depth_story)
    client = RecordedFirebaseClient()

    budget_kids = list(range(1001, 1008))
    client.items[str(budget_story)] = {
        "id": budget_story, "type": "story", "title": "Budget",
        "time": 1_700_000_000, "kids": budget_kids,
    }
    budget_sizes = [300, 300, 300, 300, 299]
    for item_id, word_count in zip(budget_kids, budget_sizes):
        client.items[str(item_id)] = {
            "id": item_id, "type": "comment",
            "text": " ".join([chr(ord("a") + item_id - 1001)] * word_count),
        }
    client.items["1006"] = {
        "id": 1006, "type": "comment",
        "text": "budget-crossing-token budget-crossing-token",
    }
    client.items["1007"] = {
        "id": 1007, "type": "comment", "text": "after-budget-break",
    }

    count_kids = list(range(2001, 2042))
    client.items[str(count_story)] = {
        "id": count_story, "type": "story", "title": "Count ceiling",
        "time": 1_700_000_000, "kids": count_kids,
    }
    for item_id in count_kids:
        client.items[str(item_id)] = {
            "id": item_id, "type": "comment", "text": f"count-{item_id}",
        }

    client.items[str(depth_story)] = {
        "id": depth_story, "type": "story", "title": "Depth ceiling",
        "time": 1_700_000_000, "kids": [3001],
    }
    for item_id in range(3001, 3006):
        client.items[str(item_id)] = {
            "id": item_id, "type": "comment", "text": f"depth-{item_id}",
            "kids": [item_id + 1] if item_id < 3005 else [],
        }

    manifest = run_recorded(source, output, client)
    records = {record["hn_object_id"]: record for record in read_records(output)}
    budget_record = records[str(budget_story)]
    budget_comments = budget_record["_master"]["hn_comment_provenance"]

    assert sum(comment["word_count"] for comment in budget_comments) == 1499
    assert len(budget_record["text"].split()) <= subject.DEFAULT_WORD_BUDGET
    assert "budget-crossing-token" not in budget_record["text"]
    assert (output / "receipts/1006.json").exists()
    assert not (output / "receipts/1007.json").exists()

    count_record = records[str(count_story)]
    assert len(count_record["_master"]["hn_comment_provenance"]) == 40
    assert "count-2041" not in count_record["text"]
    assert not (output / "receipts/2041.json").exists()

    depth_record = records[str(depth_story)]
    assert [comment["depth"] for comment in
            depth_record["_master"]["hn_comment_provenance"]] == [1, 2, 3, 4]
    assert "depth-3005" not in depth_record["text"]
    assert not (output / "receipts/3005.json").exists()
    assert manifest["limits"]["word_budget_per_story"] == \
        subject.DEFAULT_WORD_BUDGET == 1500
    assert manifest["limits"]["max_comments_per_story"] == \
        subject.DEFAULT_MAX_COMMENTS == 40
    assert manifest["limits"]["max_depth"] == subject.DEFAULT_MAX_DEPTH == 4


def test_story_self_text_does_not_consume_discussion_word_budget(tmp_path):
    """Story self-text is rendered but does not consume the discussion budget."""
    source = tmp_path / "ids.jsonl"
    output = tmp_path / "out"
    write_input(source, 4000)
    client = RecordedFirebaseClient()
    client.items.update({
        "4000": {
            "id": 4000, "type": "story", "title": "Ask HN",
            "text": " ".join(["self"] * subject.DEFAULT_WORD_BUDGET),
            "time": 1_700_000_000, "kids": [4001],
        },
        "4001": {
            "id": 4001, "type": "comment", "text": "selected-comment",
        },
    })

    run_recorded(source, output, client)
    record = read_records(output)[0]

    assert record["_master"]["hn_comment_provenance"][0]["id"] == 4001
    assert "selected-comment" in record["text"]
    assert len(record["text"].split()) > subject.DEFAULT_WORD_BUDGET


def test_dry_run_validates_plan_without_network_or_writes(tmp_path, monkeypatch):
    source = tmp_path / "ids.jsonl"
    output = tmp_path / "must-not-exist"
    write_input(source, 200, 300)

    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run attempted network access")

    monkeypatch.setattr(subject.httpx, "Client", network_forbidden)
    result = subject.main([
        "--input", str(source),
        "--output-dir", str(output),
        "--api-base-url", API_BASE,
        "--item-url-template", ITEM_TEMPLATE,
        "--item-cap", "1",
        "--dry-run",
    ])

    assert result == 0
    assert not output.exists()


def test_5xx_retries_use_exponential_backoff_then_succeed():
    recorded = RecordedFirebaseClient()

    class FlakyClient:
        def __init__(self):
            self.attempts = 0

        def get(self, url):
            self.attempts += 1
            if self.attempts < 3:
                return RecordedResponse(b"temporary", 503)
            return recorded.get(url)

    sleeps = []
    fetcher = subject.FirebaseFetcher(
        client=FlakyClient(), api_base_url=API_BASE, max_retries=2,
        backoff_seconds=0.25, sleep=sleeps.append,
        clock=lambda: FIXED_NOW)

    fetched = fetcher.fetch(200)

    assert fetched.item["id"] == 200
    assert sleeps == [0.25, 0.5]


def test_transport_failure_exhaustion_is_fail_closed():
    class OfflineClient:
        def get(self, url):
            request = httpx.Request("GET", url)
            raise httpx.ConnectError("offline", request=request)

    fetcher = subject.FirebaseFetcher(
        client=OfflineClient(), api_base_url=API_BASE, max_retries=1,
        backoff_seconds=0, sleep=lambda _seconds: None,
        clock=lambda: FIXED_NOW)

    try:
        fetcher.fetch(200)
    except subject.IngestError as exc:
        assert "exhausted retries" in str(exc)
    else:
        raise AssertionError("transport exhaustion was treated as an empty item")


def test_canonical_artifacts_are_deterministic_with_a_fixed_receipt_clock(tmp_path):
    source = tmp_path / "ids.jsonl"
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_input(source, 100, 200)

    run_recorded(source, first, RecordedFirebaseClient(), max_comments=2)
    run_recorded(source, second, RecordedFirebaseClient(), max_comments=2)

    first_files = sorted(path.relative_to(first) for path in first.rglob("*")
                         if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.rglob("*")
                          if path.is_file())
    assert first_files == second_files
    for relative in first_files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


# --------------------------------------------------------------------------
# Task 601: `--preserve-root` retention seals the `hn_discussion` profile's
# original/markdown/outlinks stages (data/source-preservation-profiles.v1.json)
# from the SAME live fetch this run already performs.
# --------------------------------------------------------------------------

def test_preserve_root_seals_a_deep_valid_run_over_the_real_thread(tmp_path):
    from cytir.index import preservation_contract as PC

    source = tmp_path / "ids.jsonl"
    output = tmp_path / "out"
    preserve_root = tmp_path / "preserved"
    write_input(source, 100)
    client = RecordedFirebaseClient()

    run_recorded(source, output, client, preserve_root=preserve_root)

    run = json.loads((preserve_root / "RUN.json").read_text(encoding="utf-8"))
    assert run["schema"] == subject.RUN_SCHEMA
    assert set(run["stages"]) == {"original", "markdown", "outlinks"}
    # story 100 + 101/102/103/104/105/106 fetched (106 is a dead-end quarantine
    # check); 107 is 106's child and must NEVER be fetched/retained.
    assert run["counts"]["retained_items"] == 7
    assert not (preserve_root / "receipts" / "107.json").exists()
    for item_id in (100, 101, 102, 103, 104, 105, 106):
        assert (preserve_root / "receipts" / f"{item_id}.json").is_file()
    assert run["counts"]["discussion_documents"] == 1
    assert run["counts"]["outlinks"] == 1
    # story-level granularity for edge conservation, not the raw item count.
    assert run["stages"]["original"]["records"] == 1

    for name, stage in run["stages"].items():
        PC._validate_stage(stage, name, artifact_root=preserve_root, deep=True)
    PC._validate_edge(
        run["lineage"]["original_to_markdown"], "original_to_markdown",
        run["stages"]["original"]["records"], run["stages"]["markdown"]["records"],
        artifact_root=preserve_root, deep=True)


def test_preservation_artifact_root_seals_relative_to_a_shared_ancestor(tmp_path):
    """A tier-2 apply needs every source's stages resolvable under ONE shared
    artifact root alongside the pack's chunks/vectors stages -- not each
    sealed relative to its own working directory."""
    from cytir.index import preservation_contract as PC

    source = tmp_path / "ids.jsonl"
    output = tmp_path / "out"
    artifact_root = tmp_path / "shared-root"
    preserve_root = artifact_root / "stage-2026-09-17" / "preserved"
    write_input(source, 100)
    client = RecordedFirebaseClient()

    run_recorded(source, output, client, preserve_root=preserve_root,
                preservation_artifact_root=artifact_root)

    run = json.loads((preserve_root / "RUN.json").read_text(encoding="utf-8"))
    for name, stage in run["stages"].items():
        assert stage["receipt"]["uri"].startswith(
            "artifact://stage-2026-09-17/preserved/")
        PC._validate_stage(stage, name, artifact_root=artifact_root, deep=True)
    # Files still physically live under preserve_root, not artifact_root.
    assert (preserve_root / "RUN.json").is_file()


def test_out_of_scope_story_is_quarantined_at_the_original_to_markdown_edge(tmp_path):
    """A story the language gate rejects is still a real fetch (retained,
    sha-bound original evidence) but a quarantined, not emitted, parent --
    the edge must conserve at story granularity, never count it twice."""
    from cytir.index import preservation_contract as PC

    source = tmp_path / "ids.jsonl"
    output = tmp_path / "out"
    preserve_root = tmp_path / "preserved"
    write_input(source, 100, 200)
    client = RecordedFirebaseClient()

    def lang(_text, url):
        # story 200 (no comments) is rejected (out of EN/DE/FR/IT scope);
        # story 100's tree is admitted.
        return ("es", True) if "id=200" in url else ("en", True)

    run_recorded(source, output, client, preserve_root=preserve_root,
                lang_detector=lang)

    run = json.loads((preserve_root / "RUN.json").read_text(encoding="utf-8"))
    assert run["counts"]["discussion_documents"] == 1
    assert run["counts"]["quarantined_stories"] == 1
    edge = run["lineage"]["original_to_markdown"]
    assert edge == {
        "input_records": 2, "emitted_parent_records": 1,
        "quarantined_records": 1, "output_records": 1,
        "quarantine_receipt": edge["quarantine_receipt"],
    }
    assert run["stages"]["original"]["records"] == 2
    PC._validate_edge(edge, "original_to_markdown", 2, 1,
                      artifact_root=preserve_root, deep=True)
    assert json.loads((preserve_root / "stages" / "original" /
                       "quarantine.json").read_text()) == {
        "reason": "out_of_scope_language", "item_ids": [200]}


def test_preserved_bytes_are_the_exact_bytes_this_run_selected_from(tmp_path):
    """Retained L0 bytes and the embedded record trace to the SAME fetch.

    Rather than trusting two independent fetches to agree, cross-check the
    retained artifact's own recomputed sha256 against the per-item fetch
    receipt this same run emitted into MANIFEST.json -- both are derived from
    the one HTTP call `FirebaseFetcher.fetch` made.
    """
    from cytir.acquire import preserved_stage_io as PSIO

    source = tmp_path / "ids.jsonl"
    output = tmp_path / "out"
    preserve_root = tmp_path / "preserved"
    write_input(source, 100)
    client = RecordedFirebaseClient()

    manifest = run_recorded(source, output, client, preserve_root=preserve_root)
    fetch_receipts = {}
    for name in os.listdir(output / "receipts"):
        row = json.loads((output / "receipts" / name).read_text())
        fetch_receipts[row["id"]] = row["sha256"]
    assert fetch_receipts  # sanity: receipts were actually written

    retained = json.loads((preserve_root / "RUN.json").read_text())
    for item_id, expected_sha in fetch_receipts.items():
        receipt = json.loads(
            (preserve_root / "receipts" / f"{item_id}.json").read_text())
        body_path = PSIO.artifact_path(preserve_root, receipt["artifacts"]["body"])
        retained_bytes = PSIO.load_validated_artifact(preserve_root, receipt["artifacts"]["body"])
        assert hashlib.sha256(retained_bytes).hexdigest() == expected_sha
        assert body_path.is_file()
    del manifest, retained


def test_no_run_is_sealed_when_nothing_was_retained(tmp_path, monkeypatch):
    """`seal_preservation_run` refuses silently-empty preservation, fail-closed."""
    with_root = tmp_path / "preserved"
    with_root.mkdir()
    fetcher = subject.FirebaseFetcher(
        client=RecordedFirebaseClient(), api_base_url=API_BASE, max_retries=0,
        backoff_seconds=0, sleep=lambda _s: None, clock=lambda: FIXED_NOW,
        preserve_root=with_root)
    try:
        subject.seal_preservation_run(with_root, fetcher=fetcher, records=[])
    except subject.IngestError as exc:
        assert "no items were retained" in str(exc)
    else:
        raise AssertionError("sealed an empty preservation run")


# --------------------------------------------------------------------------
# Task 601: authored-text guard (ADR-105 companion) -- the embedded surface is
# composed ONLY from the thread's own text, never model- or code-authored
# prose, and never carries a commenter's username.
# --------------------------------------------------------------------------

def test_render_discussion_is_pure_composition_of_input_text_blocks():
    """The rendered document is exactly title/story/comments joined by blank
    lines -- no other literal text (a byline, a label, boilerplate) is ever
    injected. Each marker below is unique so any cross-contamination between
    blocks, or any extra literal content, would be caught."""
    title = "ZZQ_TITLE_MARKER_9f2"
    story_text = "ZZQ_STORY_MARKER_c81"
    comments = [
        {"text": "ZZQ_COMMENT_MARKER_1a3", "id": 1, "depth": 1, "by": "x", "time": 1, "word_count": 1},
        {"text": "ZZQ_COMMENT_MARKER_2b4", "id": 2, "depth": 1, "by": "y", "time": 2, "word_count": 1},
    ]
    rendered = subject.render_discussion(title, story_text, comments)
    blocks = rendered.split("\n\n")
    assert blocks == [title, story_text, comments[0]["text"], comments[1]["text"]]


def test_no_commenter_username_leaks_into_any_retrievable_surface(tmp_path):
    source = tmp_path / "ids.jsonl"
    output = tmp_path / "out"
    write_input(source, 100)
    client = RecordedFirebaseClient()
    usernames = {item.get("by") for item in client.items.values()
                if isinstance(item.get("by"), str)}
    assert usernames  # sanity: the fixture actually carries usernames

    run_recorded(source, output, client)
    record = read_records(output)[0]

    for field in ("text", "snippet", "anchor", "title"):
        surface = record[field]
        for username in usernames:
            assert username not in surface, (field, username)
    # Usernames survive ONLY as master-only, non-embedded provenance.
    assert record["_master"]["hn_story_provenance"]["by"] in usernames
    assert all(row["by"] in usernames | {None}
              for row in record["_master"]["hn_comment_provenance"])


class _Item:
    def __init__(self, item):
        self.item = item
        self.receipt = {"id": item["id"]}


class _DictFetcher:
    def __init__(self, items):
        self.items = items

    def fetch(self, item_id):
        return _Item(self.items[item_id])


def _story_record_for(title, comment_texts):
    kids = list(range(2, 2 + len(comment_texts)))
    items = {1: {"id": 1, "type": "story", "title": title, "time": 1_700_000_000,
                 "kids": kids}}
    for kid, text in zip(kids, comment_texts):
        items[kid] = {"id": kid, "type": "comment", "text": text, "by": "u",
                      "time": 1_700_000_001, "parent": 1}
    return subject._story_record(
        _DictFetcher(items), 1, item_url_template=ITEM_TEMPLATE, word_budget=1000,
        max_comments=10, max_depth=3, comment_chars=600,
        lang_detector=lambda _t, _u: ("en", True))


def test_label_stitched_across_comments_holds_the_row_out_not_the_pack():
    # Task 1445 case (i): comment 1 is "By", comment 2 is ": a quote".
    record, _r, _n, reason = _story_record_for("Plain title", ["By", ": a quote"])
    assert record is None
    assert reason == "q5_identity_label_spans_blocks"


def test_title_label_that_appears_only_after_the_render_clean_holds_out():
    # Task 1445 case (ii): "&amp;#32;" unescapes to "&#32;" on the first clean
    # and to a space on the render's second clean, exposing "Author: x".
    record, _r, _n, reason = _story_record_for("&amp;#32;Author: x", ["hello"])
    assert record is None
    assert reason == "q5_identity_label_in_title_after_clean"


def test_raw_title_label_uncleaned_is_held_out_not_failed():
    # Task 1539 (2): the raw title carries the label behind entities; it is cleaned
    # by the same function as the body before the source-block check -> hold out.
    for title in ("&amp;#32;by: someone", "&amp;#32;Submitted by: someone"):
        record, _r, _n, reason = _story_record_for(title, ["hello"])
        assert record is None
        assert reason == "q5_identity_label_in_title_after_clean"


def test_label_stitched_between_title_and_first_comment_holds_the_row_out():
    # Task 1692: the `\s*` also spans the "\n\n" between the title block and the
    # first comment block ("By" / ": x"); hold-out, never an IngestError.
    record, _r, _n, reason = _story_record_for("By", [": x"])
    assert record is None
    assert reason == "q5_identity_label_spans_blocks"


def test_raw_title_with_double_escaped_label_is_held_out_with_a_named_reason():
    # Task 1692: raw title "&amp;#32;user: bob" is not a label until cleaned twice.
    record, _r, _n, reason = _story_record_for("&amp;#32;user: bob", ["hello"])
    assert record is None
    assert reason == "q5_identity_label_in_title_after_clean"
