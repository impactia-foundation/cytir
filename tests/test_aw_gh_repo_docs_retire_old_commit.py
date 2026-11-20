"""Task 2634: the sibling-commit predicate and the old-commit retire-pack builder, on fakes."""
from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from cytir.index import github_sibling_commit as GSC
from cytir.index import promote_queue as Q
from cytir.index import url_group_supersede as US
from cytir.sources.refresh_direct import retire_old_commit as R

OLD_SHA, NEW_SHA, MID_SHA = "a" * 40, "b" * 40, "c" * 40


def u(sha, path="docs/a.md", repo="o/r"):
    return f"https://github.com/{repo}/blob/{sha}/{path}"


# ---- predicate (spec 5a) -------------------------------------------------------------------
@pytest.mark.parametrize("old,cand,want", [
    (u(OLD_SHA), u(NEW_SHA), True),
    (u(OLD_SHA, repo="Foo/Bar"), u(NEW_SHA, repo="foo/bar"), True),       # owner/repo case-folded
    (u(OLD_SHA), u(OLD_SHA), False),                                      # same sha
    (u(OLD_SHA), u(NEW_SHA, path="docs/b.md"), False),                    # other path
    (u(OLD_SHA, path="Docs/a.md"), u(NEW_SHA, path="docs/a.md"), False),  # path is exact
    (u(OLD_SHA), u(NEW_SHA, repo="o/other"), False),                      # other repo
    (u(OLD_SHA), "https://github.com/o/r/blob/main/docs/a.md", False),   # branch, not a sha
    (u(OLD_SHA), "https://example.org/o/r/blob/" + NEW_SHA + "/docs/a.md", False),
    (u(OLD_SHA), "https://github.com/o/r/tree/" + NEW_SHA + "/docs/a.md", False),
    ("https://example.org/x", "https://example.org/x", False),
])
def test_predicate(old, cand, want):
    assert GSC.is_sibling_commit_url(old, cand) is want


def test_parse():
    assert GSC.parse_blob_url(u(OLD_SHA, "a/b.md")) == ("o", "r", OLD_SHA, "a/b.md")
    assert GSC.parse_blob_url(None) is None


# ---- builder -------------------------------------------------------------------------------
PROV = "github_repo_docs"


def row(i, url, prov=PROV):
    return {"id": f"00000000-0000-0000-0000-{i:012d}", "payload": {"url": url, "provenance": prov}}


ROWS = [
    row(1, u(OLD_SHA, "a.md")), row(2, u(OLD_SHA, "a.md")),          # old, survivor at a.md
    row(3, u(OLD_SHA, "gone.md")),                                    # vanished path
    row(4, u(OLD_SHA, "p.md"), prov="other"),                         # survivor has other provenance
    row(5, u(MID_SHA, "a.md")),                                       # middle commit: old too
    row(10, u(NEW_SHA, "a.md")), row(11, u(NEW_SHA, "p.md")),
    row(20, u(OLD_SHA, "x.md", repo="z/notlanded")), row(21, u(NEW_SHA, "x.md", repo="z/notlanded")),
    row(30, u(OLD_SHA, "y.md", repo="q/nohead")),
    row(40, "https://example.org/page"),                              # other shape: ignored
]
HEADS = {"o/r": {"head_sha": NEW_SHA, "pack_id": "p-landed"},
         "z/notlanded": {"head_sha": NEW_SHA, "pack_id": "p-pending"}}


class FakeTwin:
    def __init__(self, rows, page=3):
        self.rows, self.page, self.calls = rows, page, []

    def __call__(self, url, path, body=None):
        assert path.endswith("/points/scroll") and not body["with_vector"], path
        assert body["filter"]["must"] == [{"key": "domain", "match": {"value": "github.com"}}]
        self.calls.append(body.get("offset"))
        start = body.get("offset") or 0
        chunk = self.rows[start:start + self.page]
        nxt = start + self.page if start + self.page < len(self.rows) else None
        return {"points": chunk, "next_page_offset": nxt}


def scan(rows=ROWS):
    return R.scan_slice("http://twin", "aw_e8", api=FakeTwin(rows))


@pytest.fixture
def queue(tmp_path):
    q = tmp_path / "queue"
    for name, done in (("p-landed", True), ("p-pending", False)):
        d = q / name
        d.mkdir(parents=True)
        (d / "receipt.json").write_text(json.dumps({"status": "LANDED"}))
        if done:
            (d / "DONE").write_text("x\n")
    return q


@pytest.fixture
def plan(queue):
    return R.build_plan(scan(), HEADS, queue)


def test_scan_is_one_paged_pass():
    tw = FakeTwin(ROWS)
    R.scan_slice("http://twin", "aw_e8", api=tw)
    assert tw.calls == [None, 3, 6, 9]


def test_exact_tombstone_set_with_survivor_url(plan):
    got = {t["id"]: t for t in plan["tombstones"]}
    assert sorted(got) == [ROWS[i]["id"] for i in (0, 1, 4)]
    for t in got.values():
        assert t["reason"] == "superseded_generation" and t["survivor_url"] == u(NEW_SHA, "a.md")
        assert GSC.is_sibling_commit_url(t["url"], t["survivor_url"])
    assert plan["no_survivor"] == 2                            # gone.md and the other-provenance row
    assert plan["repos"]["z/notlanded"]["status"] == "pack_not_landed"
    assert plan["repos"]["q/nohead"]["status"] == "no_head"
    assert "example.org" not in json.dumps(plan["repos"])


def test_cap_defers_whole_repos(queue):
    by_repo = scan()
    p = R.build_plan(by_repo, HEADS, queue, cap=2)
    assert p["tombstones"] == [] and p["repos"]["o/r"]["status"] == "over_cap"
    heads = dict(HEADS, **{"q/nohead": {"head_sha": NEW_SHA, "pack_id": "p-landed"}})
    by_repo["q/nohead"].append({"id": "n", "url": u(NEW_SHA, "y.md", "q/nohead"),
                                "provenance": PROV, "sha": NEW_SHA, "path": "y.md"})
    p = R.build_plan(by_repo, heads, queue, cap=4)
    assert len(p["tombstones"]) == 4 and p["deferred"] == []   # o/r: 3 + q/nohead: 1
    p = R.build_plan(by_repo, heads, queue, cap=3)
    assert len(p["tombstones"]) == 3 and p["deferred"] == ["q/nohead"]


def test_pack_is_pure_and_passes_the_drains_pack_and_export_checks(plan, tmp_path, monkeypatch):
    exp = tmp_path / "exports"
    exp.mkdir()

    def fake_export(twin_url, collection, ids, urls, path):
        with gzip.open(path, "wt") as fh:
            for i in ids:
                fh.write(json.dumps({"id": i, "payload": {"url": urls[i], "provenance": PROV},
                                     "vector": {"dense": [0.1, 0.2],
                                                "bm25": {"indices": [1], "values": [1.0]}}}) + "\n")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": len(ids)}
    monkeypatch.setattr(US, "write_export", fake_export)
    out = tmp_path / "out" / "retire-1"
    m = R.write_retire_pack(plan, pack_id="retire-1", out_dir=out, twin_url="http://twin",
                            collection="aw_e8", export_dir=exp)
    tombs = [json.loads(line) for line in (out / "tombstones.jsonl").read_text().splitlines()]
    assert m["rows"] == 0 and m["tombstones_count"] == 3 and (out / "READY").is_file()
    assert all(t["survivor_url"] and t["reason"] == "superseded_generation" for t in tombs)
    Q.check_superseded_generation_pack(m, [], tombs)           # the drain's shape check
    Q.verify_removed_rows_export(m, tombs, dense_config={"dense": {"size": 2}},
                                 sparse_names=["bm25"])        # the drain's export check
    assert m["export"]["rows"] == 3


def test_main_scan_only_dry_run_and_never_into_the_queue(queue, tmp_path, monkeypatch, capsys):
    by_repo = scan()
    monkeypatch.setattr(R, "scan_slice", lambda *a, **k: by_repo)
    heads = tmp_path / "heads.json"
    heads.write_text(json.dumps({"schema": "x", "repos": HEADS}))
    base = ["--heads", str(heads), "--queue-dir", str(queue), "--twin-url", "http://t",
            "--collection", "aw_e8"]
    assert R.main(base + ["--scan-only"]) == 0
    assert json.loads(capsys.readouterr().out)["repos"]["o/r"]["old"] == 5
    assert R.main(base + ["--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and out["tombstones"] == 3 and out["no_survivor"] == 2
    assert not (tmp_path / "out").exists()
    assert R.main(base + ["--pack-id", "p", "--out-dir", str(queue / "x")]) == 2


def test_report_counts_rows_without_provenance(queue):
    rows = ROWS + [row(50, u(OLD_SHA, "n.md"), prov=None), row(51, u(OLD_SHA, "n2.md"), prov=""),
                   row(52, u(NEW_SHA, "n.md"), prov=None)]
    rep = R.build_plan(scan(rows), HEADS, queue)["repos"]["o/r"]
    assert (rep["no_provenance_old"], rep["no_provenance_new"]) == (2, 1)
    assert rep["tombstones"] == 3 and rep["no_survivor"] == 4
