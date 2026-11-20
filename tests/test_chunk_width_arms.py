#!/usr/bin/env python3
"""Task 569 -- the width-sweep arms: the policies, the experiment status arm,
and the one-home `chunk_under`.

Every case runs against SEALED inputs: the four policy artifacts in `data/`,
the ratified registry, and the task-541 selector fixtures. No live call, no
collection, no GPU.

WHAT THIS GUARDS

  * The sweep's central claim -- `specs/chunk-reference-epoch.md` §3's "the A/B
    has ONE variable" -- extended to four arms. If a later edit adds a fifth
    knob to one artifact, the A/B silently stops being a width A/B and the
    numbers in the task-569 width-sweep ledger stop meaning
    what they say.
  * `assert_selectable(experiment=True)` admits `proposed` and NOTHING else it
    did not already admit. A softer arm that also let `superseded` through
    would be a waiver wearing a flag's clothes (verification-doctrine 12).
  * `Selector.chunk_under` restores `chunk.py`'s module-global policy. A width
    sweep cuts the same page under four policies in one process; a leaked
    policy would make every later page in that process be cut under the wrong
    rule, and the counts would still look plausible.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

from tests.helpers.repo_files import repo_file_or_skip

from cytir.chunk import chunker as CH
from cytir.chunk import policy as chunk_policy
import cytir.chunk.registry as CREG
from cytir.chunk.reference_selector import REFERENCE, Selector

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO / "data"

#: (arm name, artifact, soft_target, hard_max, prefix_maximum, epoch)
ARMS = [
    ("W192", "chunk-policy-struct-md-192w-o0-adjacent-sections-tail-288-tok2560.v1.json",
     192, 288, 24, 5),
    ("W384", "chunk-policy-struct-md-384w-o0-adjacent-sections-tail-576-tok2560.v1.json",
     384, 576, 48, 6),
    ("W768", "chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json",
     768, 1152, 96, 4),
    ("W1024", "chunk-policy-struct-md-1024w-o0-adjacent-sections-tail-1536-tok2560.v1.json",
     1024, 1536, 128, 7),
]

#: The keys an arm may differ in. `sizing`/`word_guard`/`prefix` carry the width
#: itself; `policy_id`/`rule_legend`/`epoch` are identity, not behaviour.
ALLOWED_TO_DIFFER = {"sizing", "word_guard", "prefix",
                     "policy_id", "rule_legend", "epoch"}


# The policy artifacts and the selector are not all in the public cut: skip, naming the file.
for _name in ("chunk-policy-struct-md-768w-o0.v1.json",
              "chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2540-ce-anchor.v1.json",
              "chunk-reference-selector.v1.json", *[a[1] for a in ARMS]):
    repo_file_or_skip(pathlib.Path("data") / _name)


@pytest.fixture(scope="module")
def docs() -> dict:
    return {name: json.loads((DATA / f).read_text(encoding="utf-8"))
            for name, f, *_ in ARMS}


# ---------------------------------------------------------------------------
# 1. Four arms, one variable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,fname,soft,hard,prefix,epoch", ARMS,
                         ids=[a[0] for a in ARMS])
def test_each_arm_carries_the_width_it_claims(name, fname, soft, hard, prefix,
                                              epoch):
    doc = json.loads((DATA / fname).read_text(encoding="utf-8"))
    assert doc["sizing"]["soft_target"] == soft
    assert doc["word_guard"]["hard_max"] == hard
    assert doc["prefix"]["maximum"] == prefix
    assert doc["epoch"] == epoch
    assert doc["rule_legend"] == f"{doc['policy_id']}@{epoch}"


def test_only_the_width_knobs_differ_between_any_two_arms(docs):
    """The A/B has one variable, checked across all four arms at once."""
    names = [a[0] for a in ARMS]
    base = docs[names[0]]
    for other in names[1:]:
        doc = docs[other]
        differing = {k for k in set(base) | set(doc)
                     if base.get(k) != doc.get(k)}
        extra = differing - ALLOWED_TO_DIFFER
        assert extra == set(), (
            f"{names[0]} and {other} differ in {sorted(extra)} -- the width "
            "sweep would no longer be measuring width alone")


def test_the_two_ratios_hold_on_every_arm(docs):
    """hard_max = 1.5 * soft_target and prefix = soft_target // 8 are the
    ratified artifacts' own derivations, not values chosen per arm."""
    for name, _f, soft, hard, prefix, _e in ARMS:
        doc = docs[name]
        assert doc["word_guard"]["hard_max"] == soft * 3 // 2
        assert doc["prefix"]["maximum"] == soft // 8


def test_every_arm_keeps_the_embedder_ceiling(docs):
    """`token_hard_max` is the EMBEDDER's limit, not a size choice; scaling it
    per arm would invent a constraint (specs/chunk-reference-epoch.md §3)."""
    ceilings = {docs[n]["token_hard_max"]["exact"] for n, *_ in ARMS}
    assert ceilings == {2560}


def test_every_arm_is_registered_with_a_family_and_an_epoch():
    doc = CREG.load()
    for name, fname, _s, _h, _p, epoch in ARMS:
        policy = chunk_policy.load_chunk_policy(DATA / fname)
        ent = CREG.entry(policy.sha256, doc)
        assert ent["epoch"] == epoch
        assert ent["policy_family"] in doc["policy_families"]


def test_the_registry_and_the_artifacts_agree():
    assert CREG.verify_against_files() == []


# ---------------------------------------------------------------------------
# 2. The experiment status arm
# ---------------------------------------------------------------------------

def test_a_proposed_arm_is_refused_for_a_production_materialization():
    policy = chunk_policy.load_chunk_policy(DATA / ARMS[1][1])   # W384
    with pytest.raises(CREG.ChunkPolicyRegistryError) as exc:
        CREG.assert_selectable(policy)
    assert "proposed" in str(exc.value)


def test_a_proposed_arm_is_admitted_for_a_measurement():
    for idx in (1, 3):                                          # W384, W1024
        policy = chunk_policy.load_chunk_policy(DATA / ARMS[idx][1])
        assert CREG.assert_selectable(policy, experiment=True) == REFERENCE


def test_the_experiment_arm_still_refuses_a_superseded_policy():
    """The flag admits `proposed` and nothing else. A measurement that could
    quietly select a superseded policy would reproduce the defect the registry
    exists to catch."""
    policy = chunk_policy.load_chunk_policy(
        DATA / "chunk-policy-struct-md-768w-o0.v1.json")
    with pytest.raises(CREG.ChunkPolicyRegistryError) as exc:
        CREG.assert_selectable(policy, experiment=True)
    assert "superseded" in str(exc.value)


def test_the_experiment_arm_still_refuses_a_special_purpose_policy():
    policy = chunk_policy.load_chunk_policy(
        DATA / ("chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-"
                "tok2540-ce-anchor.v1.json"))
    with pytest.raises(CREG.ChunkPolicyRegistryError):
        CREG.assert_selectable(policy, experiment=True)


def test_the_experiment_arm_still_refuses_an_unregistered_policy():
    class Fake:
        sha256 = "0" * 64
        policy_id = "not-registered"
    with pytest.raises(CREG.ChunkPolicyRegistryError):
        CREG.assert_selectable(Fake(), experiment=True)


def test_experiment_selectable_is_a_superset_of_selectable_by_exactly_proposed():
    assert CREG.SELECTABLE < CREG.EXPERIMENT_SELECTABLE
    assert CREG.EXPERIMENT_SELECTABLE - CREG.SELECTABLE == {"proposed"}


# ---------------------------------------------------------------------------
# 3. chunk_under: one home for "cut under this policy and put the module back"
# ---------------------------------------------------------------------------

MARKDOWN = "\n".join(
    [f"# Title\n"] +
    [f"## Section {i}\n\n" + " ".join(f"word{i}x{j}" for j in range(120)) + "\n"
     for i in range(12)])


def test_chunk_under_restores_the_module_policy():
    sel = Selector(REPO / "data" / "chunk-reference-selector.v1.json")
    before = CH.effective_policy_value()
    for _name, fname, *_ in ARMS:
        policy = chunk_policy.load_chunk_policy(DATA / fname)
        sel.chunk_under(policy, MARKDOWN)
        assert CH.effective_policy_value() == before, (
            f"cutting under {policy.policy_id} left chunk.py reconfigured; "
            "every later page in this process would be cut under the wrong "
            "rule and the counts would still look plausible")


def test_a_narrower_arm_cuts_the_same_page_into_more_chunks():
    """The mechanism, on a page whose word count spans all four guards.

    Not a threshold assertion: it asserts only the ORDERING that makes a width
    sweep meaningful. If a narrower policy did not cut a long page finer, the
    sweep would be measuring something other than width.
    """
    sel = Selector(REPO / "data" / "chunk-reference-selector.v1.json")
    counts = {}
    for name, fname, *_ in ARMS:
        policy = chunk_policy.load_chunk_policy(DATA / fname)
        counts[name] = len(sel.chunk_under(policy, MARKDOWN))
    assert counts["W192"] >= counts["W384"] >= counts["W768"] >= counts["W1024"]
    assert counts["W192"] > counts["W1024"], counts


def test_minted_records_carry_the_arm_they_were_cut_under():
    """L565-1's rule, across all four arms: a record's `chunk_rule` and
    `rule_epochs.chunk` are properties of how it was CUT, never of whatever
    policy the module happens to hold afterwards."""
    sel = Selector(REPO / "data" / "chunk-reference-selector.v1.json")
    for name, fname, _s, _h, _p, epoch in ARMS:
        policy = chunk_policy.load_chunk_policy(DATA / fname)
        chunks = sel.chunk_under(policy, MARKDOWN)
        minted = CH.assign_chunk_ids("https://example.org/page", chunks)
        assert minted, name
        assert {m.chunk_epoch for m in minted} == {epoch}, name
        assert {m.chunk_rule for m in minted} == {policy.rule_legend}, name


def test_every_chunk_respects_its_arms_hard_max():
    """A row wider than its own policy's `word_guard.hard_max` is the defect
    `payload_shape_contract.chunk_width` exists to catch."""
    sel = Selector(REPO / "data" / "chunk-reference-selector.v1.json")
    for name, fname, _s, hard, _p, _e in ARMS:
        policy = chunk_policy.load_chunk_policy(DATA / fname)
        for c in sel.chunk_under(policy, MARKDOWN):
            assert CH.word_count(c.text) <= hard, (
                f"{name}: a chunk of {CH.word_count(c.text)} words against a "
                f"declared hard_max of {hard}")
