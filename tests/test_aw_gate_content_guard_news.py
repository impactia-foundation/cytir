"""task626 (2026-09-22, Tim via coordinator): tests for AW_GUARD_MODE=news.

"changes in gold that are explainable by valuable news articles shouldn't
block." Clause (b) of the pool refinement is DROPPED entirely -- only an
actual PACK ROW among a lost gold's new dense/BM25 candidates can make it a
violation, and even then only when that pack row is not itself a relevant,
dated news article (dense similarity to the query not below the DISPLACED
GOLD's minus the measured margin, with a publisher-declared `published`
date).

THE 2026-09-23 CORRECTION. The first cut of this mode measured relevance on
the id at the dense leg's first divergent slot instead of on the pack rows,
and called every other shape `pack_row_not_at_divergence_point`. That is
clause (b) under another name -- the first divergent slot is usually a
pre-existing row that merely reordered. It stopped the real 09-21 day; see
`test_09_21_defresh015_real_arms_*` below, the sealed arms where the
divergence point sits at dense rank 5 on a pre-existing row while the
pack's own two new rows sit at dense 61 and 94.

The three REAL fixtures are actual sealed guard arms:
  * `..._20260913.json` / `..._20260921.json` -- the two days that STOPPED
    under pool mode's clause (b), both attributed to
    `authoritative/de-fresh-005`, a pre-existing row reordering with ZERO
    pack rows involved on either leg. Under news mode they PASS on
    attribution, never reaching the relevance check.
  * `..._20260921_defresh015.json` -- the 09-21 run6 arms, the first
    sealed pair in this lane where a pack row is genuinely among a lost
    gold's new candidates, so the first that exercises the relevance check
    on real records (real ids, real urls, the arms' own recorded dense
    scores, the pack artifact's own `published`). Built by
    the task 626 fixture extractor (exploration script).
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cytir.gates import content_guard as cg  # noqa: E402

TESTDATA = ROOT / "tests" / "testdata"


class FakeScorer:
    """A `DenseScorer` stand-in that never hits qdrant. The real
    zero-pack-row fixtures never call it at all; the de-fresh-015 fixture
    supplies the REAL recorded scores/dates from the sealed arms and the
    pack artifact; the synthetic rows supply their own canned values."""

    def __init__(self, table: dict):
        self._table = table

    def describe(self, point_id, query_vector):
        return self._table.get(point_id,
                               {"score": None, "published": None, "url": None})


def _load_real_fixture(name: str) -> dict:
    return json.loads((TESTDATA / name).read_text())


def _split_arms(fixture: dict):
    baseline = {"sets": {name: {"per_q": {}} for name in fixture["sets"]}}
    candidate = {"sets": {name: {"per_q": {}} for name in fixture["sets"]}}
    query_vectors = {}
    for set_name, block in fixture["sets"].items():
        for key, row in block["per_q"].items():
            qid, which = key.rsplit("__", 1)
            (baseline if which == "baseline" else candidate)["sets"][set_name]["per_q"][qid] = row
            query_vectors[f"{set_name}/{qid}"] = [0.0]
    return baseline, candidate, query_vectors


def test_09_13_real_arms_pass_under_news_mode():
    """The actual day that STOPPED under pool mode's clause (b) -- must
    PASS under news mode because clause (a)/(c) attribution is false
    (real, verified zero pack-row intersection), never reaching the
    relevance check at all."""
    baseline, candidate, query_vectors = _split_arms(
        _load_real_fixture("task626_content_guard_news_fixture_20260913.json"))
    receipt = cg.evaluate_news(baseline, candidate, pack_ids=set(),
                               scorer=FakeScorer({}), query_vectors=query_vectors,
                               names=("authoritative", "foundation"))
    assert receipt["passed"] is True, receipt
    assert receipt["lost_golds"] == 2
    assert receipt["violations"] == []
    for row in receipt["checked"]:
        assert row["reason"] == "no_pack_row_involved"
        assert row["violation"] is False


def test_09_21_real_arms_pass_under_news_mode():
    baseline, candidate, query_vectors = _split_arms(
        _load_real_fixture("task626_content_guard_news_fixture_20260921.json"))
    receipt = cg.evaluate_news(baseline, candidate, pack_ids=set(),
                               scorer=FakeScorer({}), query_vectors=query_vectors,
                               names=("authoritative", "foundation"))
    assert receipt["passed"] is True, receipt
    assert receipt["lost_golds"] == 2
    assert receipt["violations"] == []


# ---------------------------------------------------------------- real 09-21
# de-fresh-015: the regression the 2026-09-23 correction is FOR.

def _defresh015():
    fixture = _load_real_fixture(
        "task626_content_guard_news_fixture_20260921_defresh015.json")
    baseline, candidate, query_vectors = _split_arms(fixture)
    return fixture, baseline, candidate, query_vectors


def test_09_21_defresh015_real_arms_pack_rows_are_not_at_the_divergence_point():
    """The shape that made the old rule misfire, asserted on the real
    arms: the dense leg's FIRST divergence is a pre-existing row, and the
    pack's own new rows are far down the list. Amendment 2 names the pack
    rows, not the divergence point, as the things to test."""
    fixture, baseline, candidate, _ = _defresh015()
    b_row = baseline["sets"]["authoritative"]["per_q"]["de-fresh-015"]
    c_row = candidate["sets"]["authoritative"]["per_q"]["de-fresh-015"]
    pack_ids = set(fixture["pack_ids"])
    first_diff = cg._first_diff_position(b_row["trace"]["dense"],
                                         c_row["trace"]["dense"])
    assert first_diff == 5
    assert c_row["trace"]["dense"][first_diff] not in pack_ids
    positions = [r["dense_position"] for r in
                 cg._new_pack_rows(b_row, c_row, pack_ids)]
    assert positions == [61, 94]


def test_09_21_defresh015_real_arms_are_an_explained_news_displacement():
    """The real numbers: the displaced gold (https://www.ecb.europa.eu, at
    baseline fused rank 9) scores 0.37719 on the dense leg; the two pack
    rows -- two chunks of one dated swissinfo article -- score 0.38676 and
    0.36668, both at or above the 0.37719 - 0.05 floor, both carrying a
    publisher-declared date. Under Amendment 2 that is displacement by a
    relevant news article: not a violation."""
    fixture, baseline, candidate, query_vectors = _defresh015()
    scorer = FakeScorer(fixture["scorer_table"])
    receipt = cg.evaluate_news(baseline, candidate,
                               pack_ids=set(fixture["pack_ids"]), scorer=scorer,
                               query_vectors=query_vectors,
                               names=("authoritative",),
                               query_texts={fixture["qid"]: fixture["query"]})
    assert receipt["passed"] is True, receipt
    assert receipt["lost_golds"] == 1
    row = receipt["checked"][0]
    assert row["reason"] == "explained_news_displacement"
    assert row["explained"] is True
    assert row["query"] == fixture["query"]
    assert row["gold"]["url"] == "https://www.ecb.europa.eu"
    assert row["gold"]["rank"] == 9
    assert round(row["gold"]["score"], 5) == 0.37719
    assert round(row["relevance_floor"], 5) == 0.32719
    # Amendment 2: every displacement recorded with url, similarity, date.
    assert len(row["displacements"]) == 2
    for hit in row["displacements"]:
        assert hit["url"].startswith("https://www.swissinfo.ch/")
        assert hit["published"] == "2026-09-18"
        assert hit["relevant"] is True
        assert hit["score"] >= row["relevance_floor"]
    assert any("displacer=https://www.swissinfo.ch/" in line
               for line in cg.render(receipt))


def test_09_21_defresh015_holds_when_the_pack_row_loses_its_date():
    """Same real arms, one real field removed: a pack row with no
    publisher-declared date is never an explained displacement, so the day
    HOLDS -- Tim's second condition is mandatory, not a tiebreaker."""
    fixture, baseline, candidate, query_vectors = _defresh015()
    table = {k: dict(v) for k, v in fixture["scorer_table"].items()}
    table[fixture["pack_ids"][0]]["published"] = None
    receipt = cg.evaluate_news(baseline, candidate,
                               pack_ids=set(fixture["pack_ids"]),
                               scorer=FakeScorer(table),
                               query_vectors=query_vectors,
                               names=("authoritative",))
    assert receipt["passed"] is False
    assert receipt["violations"][0]["reason"] == "displacer_has_no_publication_date"


def test_09_21_defresh015_holds_when_a_pack_row_is_off_topic():
    """Same real arms, one real score lowered below the floor: an
    off-topic pack row among the new candidates still stops the day even
    though its sibling chunk is fine."""
    fixture, baseline, candidate, query_vectors = _defresh015()
    table = {k: dict(v) for k, v in fixture["scorer_table"].items()}
    table[fixture["pack_ids"][1]]["score"] = 0.20
    receipt = cg.evaluate_news(baseline, candidate,
                               pack_ids=set(fixture["pack_ids"]),
                               scorer=FakeScorer(table),
                               query_vectors=query_vectors,
                               names=("authoritative",))
    assert receipt["passed"] is False
    row = receipt["violations"][0]
    assert row["reason"] == "displacer_below_relevance_margin"
    assert [hit["relevant"] for hit in row["displacements"]] == [True, False]


# ------------------------------------------------------------------ synthetic

def test_news_violation_no_pack_row_is_never_a_violation_regardless_of_position():
    """The core behavior change from pool mode: clause (b) is GONE. A
    first_dense_diff position of 0 (as close as it gets) with zero pack
    rows must still not be a violation."""
    baseline_row = {"rank": 3, "urls": ["u1", "u2", "u3"],
                    "trace": {"dense": ["a", "b", "c"], "bm25": ["x", "y"]}}
    candidate_row = {"rank": None, "trace": {"dense": ["z", "b", "c"], "bm25": ["x", "y"]}}
    detail = cg._news_violation(baseline_row, candidate_row, pack_ids=set(),
                                scorer=FakeScorer({}), query_vector=[0.0])
    assert detail["violation"] is False
    assert detail["reason"] == "no_pack_row_involved"
    assert detail["first_dense_diff_position"] == 0


def _synthetic_rows():
    baseline_row = {"rank": 3, "urls": ["u1", "u2", "gold-url"],
                    "trace": {"dense": ["old-1", "b", "gold-id"], "bm25": [],
                              "urls": {"gold-id": "gold-url", "old-1": "u1",
                                       "b": "u2"}}}
    candidate_row = {"rank": None, "trace": {"dense": ["pack-1", "b", "gold-id"],
                                             "bm25": []}}
    return baseline_row, candidate_row


def test_news_violation_pack_row_below_the_gold_margin_is_unexplained():
    """A dated pack row that is genuinely LESS relevant than the gold it
    displaced -- by more than the measured margin -- stays a violation."""
    baseline_row, candidate_row = _synthetic_rows()
    scorer = FakeScorer({"pack-1": {"score": 0.30, "published": "2026-09-20",
                                    "url": "https://news.example/a"},
                         "gold-id": {"score": 0.49, "published": None,
                                     "url": "gold-url"}})
    detail = cg._news_violation(baseline_row, candidate_row, pack_ids={"pack-1"},
                                scorer=scorer, query_vector=[0.0])
    assert detail["violation"] is True
    assert detail["reason"] == "displacer_below_relevance_margin"
    assert detail["gold"]["score"] == 0.49
    assert detail["relevance_floor"] == 0.49 - cg.NEWS_RELEVANCE_MARGIN


def test_news_violation_pack_row_just_inside_the_margin_is_explained():
    """The margin is a band, not a strict ordering: a dated pack row a
    hair under the gold's own score is still a relevant news article."""
    baseline_row, candidate_row = _synthetic_rows()
    scorer = FakeScorer({"pack-1": {"score": 0.46, "published": "2026-09-20",
                                    "url": "https://news.example/a"},
                         "gold-id": {"score": 0.49, "published": None,
                                     "url": "gold-url"}})
    detail = cg._news_violation(baseline_row, candidate_row, pack_ids={"pack-1"},
                                scorer=scorer, query_vector=[0.0])
    assert detail["violation"] is False
    assert detail["explained"] is True
    assert detail["displacements"][0]["reason"] == "relevant_dated_news_article"


def test_news_violation_bm25_only_pack_row_is_measured_not_waved_through():
    """A pack row that entered only the BM25 leg used to be an automatic
    `pack_row_attributed_no_dense_divergence` violation. Amendment 2 says
    measure it: it is still a pack row among the new candidates, and its
    dense similarity to the query is a well-defined number."""
    baseline_row = {"rank": 3, "urls": ["u1", "u2", "gold-url"],
                    "trace": {"dense": ["a", "b", "gold-id"], "bm25": ["x", "y"],
                              "urls": {"gold-id": "gold-url"}}}
    candidate_row = {"rank": None, "trace": {"dense": ["a", "b", "gold-id"],
                                             "bm25": ["pack-1", "y"]}}
    scorer = FakeScorer({"pack-1": {"score": 0.47, "published": "2026-09-20",
                                    "url": "https://news.example/a"},
                         "gold-id": {"score": 0.49, "published": None,
                                     "url": "gold-url"}})
    detail = cg._news_violation(baseline_row, candidate_row, pack_ids={"pack-1"},
                                scorer=scorer, query_vector=[0.0])
    assert detail["violation"] is False
    assert detail["explained"] is True
    assert detail["bm25_pack_row_count"] == 1
    assert detail["dense_pack_row_count"] == 0
    assert detail["displacements"][0]["bm25_position"] == 0


def test_news_violation_unresolvable_gold_holds_the_day():
    """No route to the displaced gold's own score means no floor to
    measure against -- never a silent pass."""
    baseline_row = {"rank": 3, "urls": ["u1", "u2", "gold-url"],
                    "trace": {"dense": ["a", "b"], "bm25": [], "urls": {}}}
    candidate_row = {"rank": None, "trace": {"dense": ["pack-1", "b"], "bm25": []}}
    detail = cg._news_violation(baseline_row, candidate_row, pack_ids={"pack-1"},
                                scorer=FakeScorer({}), query_vector=[0.0])
    assert detail["violation"] is True
    assert detail["reason"] == "displaced_gold_unresolvable"


def test_load_query_vectors_and_texts_cover_the_real_eval_yaml_shapes():
    """Both real YAML shapes this lane's eval sources use: the
    `queries:` block-list (eval_authoritative.yaml) and the `eval:`
    flow-list (eval_legal_authority.yaml) -- the loaders must handle both
    without special-casing the caller."""
    qvecs_path = TESTDATA / "task626_qvecs_fixture.json"
    auth_yaml = TESTDATA / "task626_eval_authoritative_fixture.yaml"
    legal_yaml = TESTDATA / "task626_eval_legal_fixture.yaml"
    for path, content in (
        (qvecs_path, json.dumps({"query one": [1.0, 0.0, 0.0, 0.0],
                                 "query two": [0.0, 1.0, 0.0, 0.0]})),
        (auth_yaml, 'queries:\n  - id: a1\n    query: "query one"\n    lang: en\n'),
        (legal_yaml, 'eval:\n  - {id: l1, query: "query two", lang: en}\n'),
    ):
        path.write_text(content)
    try:
        sources = {"authoritative": auth_yaml, "legal": legal_yaml}
        out = cg.load_query_vectors(qvecs_path, sources,
                                    mrl_dim=0)  # 0 = no-op truncation, keep it simple
        assert set(out.keys()) == {"authoritative/a1", "legal/l1"}
        assert list(out["authoritative/a1"]) == [1.0, 0.0, 0.0, 0.0]
        assert list(out["legal/l1"]) == [0.0, 1.0, 0.0, 0.0]
        texts = cg.load_query_texts(sources)
        assert texts == {"authoritative/a1": "query one", "legal/l1": "query two"}
    finally:
        qvecs_path.unlink(missing_ok=True)
        auth_yaml.unlink(missing_ok=True)
        legal_yaml.unlink(missing_ok=True)
