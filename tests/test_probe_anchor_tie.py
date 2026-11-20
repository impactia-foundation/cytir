"""Task 898: a same-document anchor swap in a probe reading is an anchor_tie, not a loss."""
import json
from pathlib import Path

import pytest

from cytir.gates import compare_corpus_arms as cca

DATA = Path(__file__).parent / "data"
ROWS = json.loads((DATA / "probe_anchor_tie_fedlex_rows.json").read_text())
SPEC = json.loads((DATA / "probe_anchor_tie_art257_de_spec.json").read_text())


@pytest.fixture(autouse=True)
def _own_spec(monkeypatch):
    """The live probe fixture is not in the public cut; this test carries its own spec."""
    specs = {pid: {"id": pid} for pid in cca.PROBES}
    specs["art257-de"] = SPEC
    monkeypatch.setattr(cca, "probe_specs", lambda path=None: specs)


def _doc(urls, target):
    rank = next((i for i, u in enumerate(urls, 1) if u == target), None)
    probes = {}
    for pid in cca.PROBES:
        spec = cca.probe_specs()[pid]
        probes[pid] = {"spec": spec, "urls": ["https://x.example/a"], "rank": 1}
    probes["art257-de"] = {"spec": cca.probe_specs()["art257-de"], "urls": urls, "rank": rank}
    return {"probes": probes}


def test_target_is_the_committed_probe_target():
    assert cca.probe_specs()["art257-de"]["target_url"] == ROWS["target"]


def test_swap_is_reported_as_anchor_tie_at_gold_rank():
    ties = cca.probe_non_regression(_doc(ROWS["p0"], ROWS["target"]), _doc(ROWS["p1_swap"], ROWS["target"]), allow_anchor_tie=True)
    assert ties == [{"probe": "art257-de", "kind": "anchor_tie", "gold_rank": 5,
                     "found_rank": 6, "target_url": ROWS["target"],
                     "old_anchors": [u for u in ROWS["p0"] if u.startswith(ROWS["target"].split("#")[0])],
                     "new_anchors": [u for u in ROWS["p1_swap"] if u.startswith(ROWS["target"].split("#")[0])]}]


def test_foreign_page_entering_above_is_still_a_regression():
    with pytest.raises(cca.EvidenceError, match="exact anchor regressed 5->6"):
        cca.probe_non_regression(_doc(ROWS["p0"], ROWS["target"]), _doc(ROWS["p1_other_doc_enters"], ROWS["target"]))


def test_lost_anchor_is_still_a_regression():
    with pytest.raises(cca.EvidenceError, match="regressed 5->None"):
        cca.probe_non_regression(_doc(ROWS["p0"], ROWS["target"]), _doc(ROWS["p1_anchor_lost"], ROWS["target"]))


def test_unchanged_reading_has_no_tie():
    assert cca.probe_non_regression(_doc(ROWS["p0"], ROWS["target"]), _doc(ROWS["p0"], ROWS["target"])) == []


def test_tie_is_opt_in():
    with pytest.raises(cca.EvidenceError, match="regressed 5->6"):
        cca.probe_non_regression(_doc(ROWS["p0"], ROWS["target"]), _doc(ROWS["p1_swap"], ROWS["target"]))


def test_sibling_swap_with_target_behind_foreign_pages_is_not_a_tie():
    base = "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#"
    f = ["https://example.org/%d" % i for i in range(1, 4)]
    old = [f[0], f[1], base + "art_257", base + "art_257_a", "https://example.org/4"] + ["https://example.org/%d" % i for i in range(5, 11)]
    # 3 -> 7: three foreign pages enter above; the sibling anchor takes the target's old slot
    new = [f[0], f[1], base + "art_257_a", "https://example.org/4", "https://example.org/5",
           "https://example.org/6", base + "art_257", "https://example.org/7", "https://example.org/8", "https://example.org/9"]
    assert not cca.anchor_tie(ROWS["target"], old, new)
    with pytest.raises(cca.EvidenceError, match="regressed 3->7"):
        cca.probe_non_regression(_doc(old, ROWS["target"]), _doc(new, ROWS["target"]), allow_anchor_tie=True)
