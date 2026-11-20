#!/usr/bin/env python3
"""Contracts for the task-637 footprint experiment tooling.

Real records, sealed inputs, no live calls: every test below runs against
the task-637 rows-fixture-35.jsonl artifact — 35 points scrolled out of the
`aw_e8` twin on 2026-09-22 with their stored payload and both stored vectors,
selected to cover all 34 payload keys the collection actually carries. A
synthetic row would not exercise the one thing these guards are for: that a
payload NORMALIZATION is information-preserving on the shapes production really
holds (a 7-key `rule_epochs` dict, a 113-char `consent_source` sentence, a
64-hex `content_digest`, a `stamps` dict that duplicates two other keys).

The guards here are the ones whose absence would have let a wrong byte-saving
claim through:

* `slim_payload` must be REVERSIBLE through the manifest's enum tables, or
  "information-preserving" is a slogan. Tested by reconstructing every original
  value from the table the builder writes.
* `slim_payload` must not touch a key it was not told to touch — the levers are
  independent or the ledger cannot attribute bytes to them.
* the IDF/df inversion must round-trip against Qdrant's own formula, because a
  prune threshold expressed in IDF and applied in df is two numbers that have to
  be the same number (task 627 §3.2).
* a prune set must be derived from the arm's OWN corpus, and the builder must
  refuse a threshold over several row files, because `n` would be ambiguous.
* no arm may target a serving or rollback collection.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

from tests.helpers.repo_files import repo_file_or_skip

REPO = pathlib.Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"          # withheld by the public cut: each reader below skips there (task 2411)
FIXTURE = TASKS / "637-artifacts" / "rows-fixture-35.jsonl"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rows():
    repo_file_or_skip(FIXTURE)
    assert FIXTURE.exists(), "the sealed real-row fixture is the input; no live call"
    out = [json.loads(line) for line in FIXTURE.open()]
    assert len(out) == 35
    return out


def test_fixture_is_real_rows_with_both_vectors(rows):
    """The fixture is what production stores, not a hand-written shape."""
    keys = set()
    for p in rows:
        assert isinstance(p["id"], str) and len(p["id"]) == 36, "uuid point ids"
        vec = p["vector"]
        assert len(vec["dense"]) == 256, "prod serves 256-dim dense"
        assert set(vec["bm25"]) == {"indices", "values"}
        assert len(vec["bm25"]["indices"]) == len(vec["bm25"]["values"])
        keys |= set(p["payload"])
    assert len(keys) == 34, "the fixture covers every payload key aw_e8 carries"
    for required in ("url", "snippet", "title", "trust_tier", "rule_epochs",
                     "point_id", "content_digest", "stamps", "consent_source"):
        assert required in keys


NORMALIZED_KEYS_UNREAD_BY_SERVING = (
    "rule_epochs", "build_id", "chunk_rule", "consent_source", "consent_basis",
    "consent_state", "provenance", "prov_id", "stamps", "content_digest",
    "point_id",
)


def test_latency_harness_resolves_the_667_query_slice():
    """The slice's own declared counts must be reachable from the gate's sets."""
    import json as _json
    sl = _json.loads(repo_file_or_skip("data/news-daily-gate-slice.v1.json").read_text())
    assert sl["counts"]["queries"] == 667
    ga = _load("aw_gate_authority_t", REPO / "cytir" / "gates" / "authority.py")
    known = {n for n, _p in ga.SETS}
    missing = set(sl["sets"]) - known
    assert not missing, "slice names sets gate_authority does not have: %s" % missing


def test_p18_is_wired_through_the_generated_overlay_not_a_hand_edit():
    """A knob that lives only in a hand-edited compose file is dropped by the
    next `up -d` -- task 596's finding, and why serving_config owns this."""
    sc_src = (REPO / "cytir" / "core" / "serving_config.py").read_text()
    gen_src = (REPO / "cytir" / "ops" / "gen_flip_compose_env.py").read_text()
    assert "def flip_qdrant_environment" in sc_src
    assert "def flip_qdrant_temp_mount" in sc_src
    assert "flip_qdrant_environment" in gen_src, "the generator must render it"
    assert "QDRANT_TEMP_PATH_HOST_PROD" in sc_src
    import re as _re
    m = _re.search(r'QDRANT_TEMP_PATH_HOST_PROD = "([^"]+)"', sc_src)
    assert m and m.group(1).startswith("/mnt/promote-scratch"), \
        "the staging dir must be on the roomy volume, not the serving disk"


def test_the_eval_files_really_do_disagree_about_their_top_level_key():
    """The fact that motivated the fix, frozen so it cannot be forgotten."""
    import yaml as _yaml
    auth = _yaml.safe_load(repo_file_or_skip("data/eval_authoritative.yaml").read_text())
    legal = _yaml.safe_load(repo_file_or_skip("data/eval_legal_authority.yaml").read_text())
    assert isinstance(auth, dict) and "queries" in auth and "eval" not in auth
    assert isinstance(legal, dict) and "eval" in legal


def test_the_anon_attribution_is_recorded_with_its_negative_results():
    """The two negative results are the load-bearing half: the HNSW graph is not
    in anon (m=8 moved it +17 B/pt, the wrong sign) and neither prune is a RAM
    lever. Without those, someone re-proposes m=8 as a RAM fix."""
    doc = repo_file_or_skip(TASKS / "637-index-size.md").read_text()
    assert "The HNSW graph is NOT in anon" in doc
    assert "qdrant#9986" in doc
    assert "Neither prune is a RAM lever" in doc
    assert "73% of anon" in doc or "73% of all anon" in doc


def test_a_reader_landing_on_any_anon_section_is_sent_to_the_LATEST_word():
    """Sections 19, 20 and 21 say three different things about the same flag.

    The claim was made (19), retracted (20), and the retraction was itself
    withdrawn (21) once the missing state -- an index the process RELOADS from
    disk -- was measured. Two of those three sections are wrong, and they stay
    in the document because the sequence is the finding. So every one of them
    has to hand a reader forward before it hands them a number: a superseded
    section that reads as current is how a withdrawn figure gets quoted.
    """
    doc = repo_file_or_skip(TASKS / "637-index-size.md").read_text()
    i19 = doc.index("## 19. The anon split")
    i20 = doc.index("## 20. RETRACTION")
    i21 = doc.index("## 21. The retraction was itself wrong")
    assert "§21" in doc[i19:i19 + 400], \
        "section 19's heading must point at the latest word, not at section 20"
    assert "WITHDRAWN BY §21" in doc[i20:i20 + 900], \
        "section 20 must be marked withdrawn where a reader lands on it"
    # the claims that survived all three rounds must stay listed, not implicit
    for survives in ("anon is qdrant's heap", "HNSW graph is **not** in anon",
                     "587 B/pt"):
        assert survives in doc[i20:], survives
    # and 21 must state the mechanism that separates the three states
    for k in ("RAM, loaded from disk", "rebuilt in-process",
              "`on_disk`, loaded from disk"):
        assert k in doc[i21:], k


def test_the_prod_ceiling_is_not_quoted_above_what_was_measured_at_scale():
    """The arm's -505 B/pt is a 1.21M reading and prod's total anon is 587 B/pt
    over 33.7M, so the arm's saving cannot be carried across. Section 21 must
    say so rather than publish an extrapolated ceiling -- the attribution rule
    (memory: attribution-needs-counterfactual) applied to a scale jump."""
    doc = repo_file_or_skip(TASKS / "637-index-size.md").read_text()
    i21 = doc.index("## 21. The retraction was itself wrong")
    tail = doc[i21:]
    assert "No number above 43.3M may be quoted" in tail
    assert "unknown" in tail


# ---------------------------------------------------------------------------
# The flip-gate driver for aw_e9 (the task-637 gate_e9.sh + its state
# module). These are the guards for the two defects the driver's own dry run
# found on 2026-09-23 -- all three candidate 629 legs sharing one port, and a
# server whose effective config nobody compared with the arm's claim -- plus
# the binding that keeps the driver's arm list the CONTRACT's arm list.
# ---------------------------------------------------------------------------
GATE_SH = TASKS / "637-artifacts" / "gate_e9.sh"
GATE_STATE = TASKS / "637-artifacts" / "gate_e9_state.py"
LATENCY_SH = TASKS / "637-artifacts" / "latency_e9.sh"


@pytest.fixture(scope="module")
def gate_sh():
    return repo_file_or_skip(GATE_SH).read_text()


@pytest.fixture(scope="module")
def gstate():
    return _load("aw_gate_e9_state", repo_file_or_skip(GATE_STATE))


def test_gate_driver_runs_exactly_the_contract_arms(gate_sh):
    """The driver's twelve `--arm` specs ARE `required_arms`, not a copy of it.

    A battery whose arm list drifts from the contract's can pass while the
    contract refuses, or refuse while the contract would pass.
    """
    contract = json.loads(
        (REPO / "data" / "flip-gate-contract.v1.json").read_text())
    want = {(str(a["instrument"]), a["leg"], a["side"])
            for a in contract["required_arms"]}
    got = set()
    for line in gate_sh.splitlines():
        line = line.strip().strip('"')
        if line.count(":") >= 2 and "=$OUT_DIR/" in line:
            key, _, _path = line.partition("=")
            parts = key.split(":")
            if len(parts) == 3:
                got.add(tuple(parts))
    assert got == want, f"driver arms {sorted(got)} != contract {sorted(want)}"
    assert len(want) == 12


def test_gate_driver_gives_every_629_leg_its_own_port(gate_sh):
    """One port per (side, leg).

    The dry run scored `dense-only` and `sparse-only` against the still-running
    `full-8` server because all three shared port 8480 -- sealing full-stack
    numbers under a leg label. The task 533 arm driver had
    PORTBASE+0..+3 for exactly this reason.
    """
    offsets = {}
    for line in gate_sh.splitlines():
        stripped = line.strip()
        if "port=$((portbase+" not in stripped:
            continue
        leg = stripped.split(")", 1)[0].strip()
        offsets[leg] = stripped.split("port=$((portbase+", 1)[1].split(")", 1)[0]
    assert set(offsets) == {"full-8", "full-60", "dense-only", "sparse-only"}
    assert len(set(offsets.values())) == 4, f"colliding ports: {offsets}"
    bases = set()
    for line in gate_sh.splitlines():
        if "do run_629 " not in line:
            continue
        for token in line.replace(";", " ").split():
            if token.isdigit():
                bases.add(token)
    assert len(bases) == 2, f"each side needs its own port base, got {bases}"


def test_gate_driver_does_not_background_setsid_in_a_subshell(gate_sh):
    """`( setsid ... ) &` makes `$!` the SUBSHELL, while setsid moves the server
    into a new process group -- so `kill -- -$!` signals nothing and the server
    outlives its arm. `$!` has to be setsid's own pid."""
    for line in gate_sh.splitlines():
        if "setsid" in line and not line.strip().startswith("#"):
            assert not line.strip().startswith("("), line
    assert "ss -ltnH" in gate_sh, "the driver must refuse a port already in use"


def test_gate_driver_checks_health_before_it_scores(gate_sh):
    """`/health` echoes the effective config; the driver compares it with what
    the arm is about to claim BEFORE scoring, rather than sealing first and
    noticing afterwards."""
    body = gate_sh[gate_sh.index("run_629() {"):gate_sh.index("run_1418() {")]
    assert body.index("PYHEALTH") < body.index(f"{TASKS.name}/524-scripts/run_arm_524.py")
    assert "fusion weights: the server serves" in gate_sh
    assert "collection: the server serves" in gate_sh


def test_gate_driver_refuses_a_limited_run_as_evidence(gate_sh):
    """`--limit` reaches only the 629 instrument and produces a SMOKE arm.
    A limited battery is a harness check, never a measurement."""
    assert '"$LIMIT" -gt 0 ] && [ "$SELFCHECK" -ne 1 ]' in gate_sh


def test_side_state_has_no_default_and_no_third_physical_state(gstate):
    """Qdrant records nowhere how a collection came to exist. A field that
    guessed would be worse than one that asks, so there is no default."""
    assert gstate.STATES == ("built", "restored")
    with pytest.raises(gstate.StateError):
        gstate.read_side("http://127.0.0.1:1", "c", "unknown", label="x",
                         require_green=True, wait_settled_s=0)


def test_side_state_stale_ceiling_is_the_already_ruled_one():
    """0.1% is `cytir.index.idf_table.GATE_MAX_STALE_RATIO`, the ceiling already
    ruled for a gate IDF table -- reused, not invented. Two homes for one
    number is how a gate and the thing it gates come to disagree."""
    from cytir.index import idf_table as seal   # the seal half of the IDF module (task 1234)
    src = repo_file_or_skip(GATE_STATE).read_text()
    assert f"default={seal.GATE_MAX_STALE_RATIO}" in src, \
        "the driver's stale ceiling must be the ruled gate ceiling"


def test_side_state_dense_guard_ignores_the_levers_it_is_measuring(gstate):
    """aw_e9 lowers HNSW `m` (lever A9) and stores the sparse index as uint8
    (A12) on purpose. A guard that refused on those would refuse the
    experiment; one that ignored the dense width would let a second variable
    in."""
    def info(*, sparse_dt, m, size=256):
        return {"config": {"params": {
            "vectors": {"dense": {"size": size, "distance": "Cosine",
                                  "datatype": "float16", "on_disk": True}},
            "sparse_vectors": {"bm25": {"index": {"datatype": sparse_dt},
                                        "modifier": "idf"}}},
            "quantization_config": {"product": {"compression": "x8"}},
            "hnsw_config": {"m": m}}}
    e8 = gstate._dense_config(info(sparse_dt="float16", m=16))
    e9 = gstate._dense_config(info(sparse_dt="uint8", m=8))
    assert e8 == e9, "A9 and A12 are the levers; the guard must not see them"
    assert e8 != gstate._dense_config(info(sparse_dt="uint8", m=8, size=128)), \
        "a dense width change IS a second variable and must be refused"


def test_side_state_stamp_puts_the_state_inside_the_seal(gstate, tmp_path):
    """A verdict read a year from now has to say which physical state produced
    it (memory `gate-baseline-from-prod-not-restored-twin`)."""
    verdict = tmp_path / "v.json"
    verdict.write_text(json.dumps({"schema": "aw.flip-gate-verdict.v1",
                                   "verdict": "FAIL"}))
    state = tmp_path / "s.json"
    fp = {"points_count": 1, "n_per_vector": 1.0, "stale_ratio": 0.0}
    side = {"collection": "aw_e8", "url": "u", "physical_state": "restored",
            "physical_state_basis": "b", "settled_problems": [],
            "layout_converged_across_two_reads": True, "points_count": 1,
            "read_at": "t", "layout_fingerprint": fp}
    other = dict(side, collection="aw_e9", physical_state="built")
    state.write_text(json.dumps({
        "sides": {"incumbent": side, "candidate": other},
        "confound": {"kind": "physical-state"}}))

    class Args:
        pass
    args = Args()
    args.verdict, args.state = verdict, state
    assert gstate.stamp(args) == 0
    out = json.loads(verdict.read_text())
    assert out["physical_state"]["candidate"]["physical_state"] == "built"
    assert out["physical_state"]["incumbent"]["physical_state"] == "restored"
    assert out["layout_fingerprints"]["candidate"] == fp
    assert out["physical_state_confound"]["kind"] == "physical-state"


def test_latency_driver_stops_the_container_before_it_evicts():
    """`posix_fadvise(DONTNEED)` cannot evict pages a live process has mmap'd.
    A residency instrument in this same task produced a false reading that way
    (637 §18.4/§20), so the stop comes first -- and the driver must refuse to
    touch the build's scroll source at all."""
    text = repo_file_or_skip(LATENCY_SH).read_text()
    assert text.index("docker stop") < text.index("\n  evict "), \
        "stop the container before the eviction pass"
    assert "aw-qdrant-637-copy" in text, "the one container it may never restart"
    assert "build_collection_from_source.py" in text, \
        "a concurrent scroll leaves no cold state to measure"
    assert 'ENV_FILE="$OUT_DIR/prod.env.generated"' in text, \
        "the serving env is generated from prod's compose, not a hand file"


def test_latency_driver_resolves_the_box_instead_of_hardcoding_it():
    """637's M1/M2 moves `aw_e9` into its own container and storage tree.

    A script carrying either name as a constant measures the shape it was
    written against, not the shape being shipped. So the container comes from
    the port `--url` names and the segment tree from that container's own
    `/qdrant/storage` mount -- and an explicitly passed container must still be
    the one answering `--url`.
    """
    text = repo_file_or_skip(LATENCY_SH).read_text()
    assert 'PORT="${URL##*:}"' in text, "the container is resolved from --url"
    assert "/qdrant/storage" in text, \
        "the segment tree comes from the container's own storage mount"
    assert "does not publish port" in text, \
        "an explicitly named container is checked against --url too"
    # required args no longer include the two resolved ones
    required = text.split("for v in ", 1)[1].split(";", 1)[0].split()
    assert "CONTAINER" not in required and "SEGMENTS" not in required
    assert {"URL", "COLL", "OUT_DIR"} <= set(required)


def test_latency_driver_refuses_a_segment_tree_too_small_to_be_the_index():
    """`-d` is not enough: an empty or half-copied tree is a directory too.
    An eviction over it succeeds, the real pages stay resident, and pass 1
    reports a WARM number under the label `cold` -- the defect class 637
    §18.4/§20 already paid for once."""
    text = repo_file_or_skip(LATENCY_SH).read_text()
    assert "points_count" in text and "points * 200" in text, \
        "the tree's size is checked against the collection's own point count"
    assert "half-copied" in text


# The streaming builder's delta catch-up (--only-ids) moved with it: tests/test_aw_index_from_collection.py
