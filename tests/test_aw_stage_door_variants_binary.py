"""Stage-door guards of task 767: url variants never join a page they duplicate; a binary row
refuses the pack."""
from __future__ import annotations

import json

import pytest

from cytir.sources import family_stage as FS

P = "https://ai.google.dev/gemini-api/docs/function-calling"


def _b(url, pid, text="Function calling lets the model call your functions."):
    return (json.dumps({"url": url, "point_id": pid, "text": text, "license": "cc-by"}) + "\n").encode()


def test_variant_is_dropped_when_the_canonical_row_is_in_the_slice():
    buf = [_b(P + "?authuser=1", "v1"), _b(P, "c1"), _b(P + "?authuser=2", "v2")]
    keep, dropped = FS.drop_url_variants(buf)
    assert [json.loads(b)["point_id"] for b in keep] == ["c1"]
    assert sorted(dropped) == ["v1", "v2"]


def test_without_a_canonical_row_one_spelling_stays_and_its_chunks_all_stay():
    buf = [_b(P + "?authuser=1", "a1"), _b(P + "?authuser=1", "a2"), _b(P + "?authuser=2", "b1")]
    keep, dropped = FS.drop_url_variants(buf)
    assert [json.loads(b)["point_id"] for b in keep] == ["a1", "a2"]
    assert dropped == ["b1"]


def test_variant_is_dropped_when_a_target_serves_the_canonical_url():
    keep, dropped = FS.drop_url_variants([_b(P + "?authuser=1", "v1")], {P})
    assert keep == [] and dropped == ["v1"]


def test_canonical_and_distinct_pages_are_untouched():
    buf = [_b(P, "c1"), _b(P + "?hl=ru", "ru1"), _b("https://ai.google.dev/other", "o1")]
    keep, dropped = FS.drop_url_variants(buf)
    assert keep == buf and dropped == []


def test_split_served_drops_a_variant_of_a_served_canonical_page():
    buf = [_b(P + "?authuser=1", "v1"), _b("https://ai.google.dev/new", "n1")]
    asked = {}

    def served(url, coll, urls):
        asked["urls"] = list(urls)
        return {P}
    keep, dropped = FS._split_served(buf, {"twin": ("http://twin", "c")}, lambda *_a: set(), served)
    assert dropped == ["v1"] and [json.loads(b)["point_id"] for b in keep] == ["n1"]
    assert P in asked["urls"]                     # the canonical spelling is what gets asked


def _rows_file(tmp_path):
    rows = [{"url": "https://www.weko.admin.ch/a.pdf", "point_id": "p1", "license": "x",
             "text": "1135 0 obj <>/Font<>> endobj 1191 0 obj [1193 0 R 1194 0 R] endobj 1192 0 obj"},
            {"url": "https://www.weko.admin.ch/b.pdf", "point_id": "p2", "license": "x",
             "text": "Die WEKO hat die Zusammenarbeit geprüft."}]
    f = tmp_path / "records.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f, rows


def test_a_binary_row_is_held_out_counted_never_a_pack_refusal(tmp_path):
    f, rows = _rows_file(tmp_path)
    held = []
    assert FS._validate_rows(f, held) == 2             # task 777: the pass goes on
    assert held == [("p1", "https://www.weko.admin.ch/a.pdf", "binary_content:pdf_structure")]
    with pytest.raises(FS.StageError, match="not prose"):   # the one-shot stage_out tool: as before
        FS._validate_rows(f)
    f.write_text(json.dumps(rows[1]) + "\n")
    assert FS._validate_rows(f) == 1


def test_mechanical_content_gate_names_a_binary_row_and_refuses_past_the_share():
    from cytir.gates import mechanical_content_check as MC
    good = ("Die Wettbewerbskommission hat die Zusammenarbeit der Unternehmen geprüft und "
            "festgestellt, dass keine unzulässige Wettbewerbsabrede vorliegt, weshalb das "
            "Verfahren eingestellt wird.")
    bad = "1135 0 obj <>/Font<>> endobj 1191 0 obj [1193 0 R 1194 0 R 1195 0 R] endobj " * 4
    rows = [{"point_id": f"g{i}", "url": f"https://x.ch/{i}", "title": "t", "lang": "de",
             "text": good} for i in range(99)]
    rows.append({"point_id": "b0", "url": "https://x.ch/b", "title": "t", "lang": "de", "text": bad})
    out = MC.evaluate(rows[:99], detect_language=False)
    assert "held_out_rows" not in out and "held_out" not in out    # a clean pack's receipt is unchanged
    out = MC.evaluate(rows, detect_language=False)
    # task 777: a binary row is a counted per-row hold-out, not a failing row and never a refusal
    assert out["problems"] == {}
    assert out["held_out"] == {"b0": "binary_content:pdf_structure"}
    assert out["held_out_rows"] == 1 and out["held_out_by_reason"] == {
        "binary_content:pdf_structure": 1}
    assert out["verdict"] == "PASS" and out["other_failing_rows"] == 0
    rows.extend(dict(rows[-1], point_id=f"b{i}") for i in range(1, 30))   # 30 % binary: still PASS
    out = MC.evaluate(rows, detect_language=False)
    assert out["verdict"] == "PASS" and out["held_out_rows"] == 30 and out["failing_rows"] == 0
    only = [dict(r, text=bad) for r in rows]                       # nothing left to apply
    assert MC.evaluate(only, detect_language=False)["verdict"] == "REFUSE"
