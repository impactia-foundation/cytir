"""task 659x/661, 2026-09-26: staging-collection retention.

The other half of the 09-25 outage fix -- ~130 never-removed staging
collections exhausted vm.max_map_count. These tests cover the pure
decision logic (name parsing, pack-id attribution, landed+verified
check, the drop plan) with real-shaped fixtures, no live qdrant/network.
"""
from __future__ import annotations

import datetime as _dt
import json

import cytir.ops.staging_collection_retention as R


def test_parse_collection_name_recognizes_both_dim_spellings():
    assert R.parse_collection_name("aw_dnews256_20260924_chmedia") == \
        ("2026-09-24", "chmedia")
    assert R.parse_collection_name("aw_dnews_20260921_swissinfo") == \
        ("2026-09-21", "swissinfo")


def test_parse_collection_name_covers_all_four_arms():
    for arm in ("chmedia", "swissinfo", "hn", "hn_discussion"):
        assert R.parse_collection_name(f"aw_dnews256_20260901_{arm}") == \
            ("2026-09-01", arm)


def test_parse_collection_name_ignores_unrelated_collections():
    assert R.parse_collection_name("aw_d621_regfirst") is None
    assert R.parse_collection_name("aw_e8") is None
    assert R.parse_collection_name("aw_dnews256_not_a_day_chmedia") is None


def test_owning_pack_id_for_publisher_arms_uses_the_collections_own_day():
    assert R.owning_pack_id("2026-09-24", "chmedia") == "news-daily-20260924"
    assert R.owning_pack_id("2026-09-24", "swissinfo") == "news-daily-20260924"
    assert R.owning_pack_id("2026-09-24", "hn") == "news-daily-20260924"


def test_owning_pack_id_for_hn_discussion_shifts_back_one_day():
    """hn_discussion_stage.sh anchors its own collection's stamped day to
    day+1 (a clean single-day Algolia window); the OWNING pack is the
    day before that."""
    assert R.owning_pack_id("2026-08-15", "hn_discussion") == \
        "hn-discussion-daily-20260814"


def _write_receipt(queue_dir, pack_id, *, status="LANDED", twin=2252, prod=2252):
    d = queue_dir / pack_id
    d.mkdir(parents=True)
    doc = {"status": status}
    if status == "LANDED":
        doc["twin"] = {"layout_after": {"points_count": twin}}
        doc["prod"] = {"layout_after": {"points_count": prod}}
    (d / "receipt.json").write_text(json.dumps(doc))


def test_is_pack_landed_and_verified_true_when_twin_equals_prod(tmp_path):
    _write_receipt(tmp_path, "news-daily-20260924", twin=2252, prod=2252)
    assert R.is_pack_landed_and_verified("news-daily-20260924", tmp_path) is True


def test_is_pack_landed_and_verified_false_when_twin_differs_from_prod(tmp_path):
    """A landed receipt whose own recorded counts disagree must never be
    treated as verified -- retention stays conservative."""
    _write_receipt(tmp_path, "news-daily-20260924", twin=2252, prod=2251)
    assert R.is_pack_landed_and_verified("news-daily-20260924", tmp_path) is False


def test_is_pack_landed_and_verified_false_when_refused(tmp_path):
    _write_receipt(tmp_path, "news-daily-20260924", status="REFUSED")
    assert R.is_pack_landed_and_verified("news-daily-20260924", tmp_path) is False


def test_is_pack_landed_and_verified_false_when_no_receipt(tmp_path):
    assert R.is_pack_landed_and_verified("news-daily-20260924", tmp_path) is False


def test_collections_to_drop_keeps_the_last_keep_days_regardless_of_landed(tmp_path):
    _write_receipt(tmp_path, "news-daily-20260924")
    today = _dt.date(2026, 9, 26)  # 2 days old -- inside the 7-day keep window
    plan = R.collections_to_drop(
        ["aw_dnews256_20260924_chmedia"], queue_dir=tmp_path, today=today,
        keep_days=7)
    assert plan == []


def test_collections_to_drop_drops_an_old_landed_verified_collection(tmp_path):
    _write_receipt(tmp_path, "news-daily-20260910")
    today = _dt.date(2026, 9, 26)  # 16 days old
    plan = R.collections_to_drop(
        ["aw_dnews256_20260910_chmedia", "aw_dnews256_20260910_swissinfo"],
        queue_dir=tmp_path, today=today, keep_days=7)
    names = {e["collection"] for e in plan}
    assert names == {"aw_dnews256_20260910_chmedia", "aw_dnews256_20260910_swissinfo"}
    assert plan[0]["pack_id"] == "news-daily-20260910"


def test_collections_to_drop_never_drops_an_old_unlanded_collection(tmp_path):
    """No receipt at all (e.g. the OLD hn-freshness arm's collection --
    never packed, by ADR-098 R2 design) -- must be left alone forever by
    this function; a human/operator decides that cleanup separately."""
    today = _dt.date(2026, 9, 26)
    plan = R.collections_to_drop(
        ["aw_dnews256_20260701_hn"], queue_dir=tmp_path, today=today, keep_days=7)
    assert plan == []


def test_collections_to_drop_ignores_unrelated_collection_names(tmp_path):
    _write_receipt(tmp_path, "news-daily-20260910")
    today = _dt.date(2026, 9, 26)
    plan = R.collections_to_drop(
        ["aw_d621_regfirst", "aw_dnews256_20260910_chmedia"],
        queue_dir=tmp_path, today=today, keep_days=7)
    assert len(plan) == 1
    assert plan[0]["collection"] == "aw_dnews256_20260910_chmedia"


def test_collections_to_drop_attributes_hn_discussion_to_the_shifted_pack_day(tmp_path):
    # collection stamped 2026-08-15 belongs to the hn_discussion pack for 08-14
    _write_receipt(tmp_path, "hn-discussion-daily-20260814")
    today = _dt.date(2026, 9, 26)
    plan = R.collections_to_drop(
        ["aw_dnews256_20260815_hn_discussion"], queue_dir=tmp_path, today=today,
        keep_days=7)
    assert len(plan) == 1
    assert plan[0]["day"] == "2026-08-15"
    assert plan[0]["pack_id"] == "hn-discussion-daily-20260814"


class _FakeCollections:
    def __init__(self, names):
        self.collections = [type("C", (), {"name": n})() for n in names]


class _FakeQdrantClient:
    def __init__(self, names):
        self._names = list(names)
        self.deleted = []

    def get_collections(self):
        return _FakeCollections(self._names)

    def delete_collection(self, name):
        self.deleted.append(name)
        self._names.remove(name)


def test_run_dry_run_never_calls_delete(tmp_path, monkeypatch):
    _write_receipt(tmp_path, "news-daily-20260910")
    fake_client = _FakeQdrantClient(
        ["aw_dnews256_20260910_chmedia", "aw_d621_regfirst"])
    monkeypatch.setattr(
        "cytir.core.qdrant.qdrant_client", lambda url, **kw: fake_client)
    report = R.run(staging_url="http://x", queue_dir=tmp_path,
                   today=_dt.date(2026, 9, 26), keep_days=7, dry_run=True,
                   log=lambda *a, **k: None)
    assert report["dry_run"] is True
    assert report["dropped"] == []
    assert len(report["eligible"]) == 1
    assert fake_client.deleted == []


def test_run_apply_calls_delete_only_for_eligible_collections(tmp_path, monkeypatch):
    _write_receipt(tmp_path, "news-daily-20260910")
    fake_client = _FakeQdrantClient(
        ["aw_dnews256_20260910_chmedia", "aw_d621_regfirst"])
    monkeypatch.setattr(
        "cytir.core.qdrant.qdrant_client", lambda url, **kw: fake_client)
    report = R.run(staging_url="http://x", queue_dir=tmp_path,
                   today=_dt.date(2026, 9, 26), keep_days=7, dry_run=False,
                   log=lambda *a, **k: None)
    assert report["dropped"] == ["aw_dnews256_20260910_chmedia"]
    assert fake_client.deleted == ["aw_dnews256_20260910_chmedia"]
    assert "aw_d621_regfirst" in fake_client._names  # untouched
