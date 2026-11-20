from tests.helpers.repo_files import repo_glob_or_skip

# the operator tree under `deploy/` is withheld from the public cut: the test skips there
SCRIPT = repo_glob_or_skip("deploy/*/eric-materialization/chain_eric_pdf_partition_admission.sh")[0].read_text()


def test_eric_chain_waits_for_exact_partition_finalizer_and_seals_all_gates():
    assert "AW_ERIC_FINALIZER_OUTCOME" in SCRIPT
    assert "AW_EXPECTED_SELECTED" in SCRIPT
    assert "AW_ERIC_LABEL:-" in SCRIPT
    assert SCRIPT.count("-m cytir.ops.run_observed_one_shot") == 4
    assert "--source-profile web_pdf" in SCRIPT
    assert "cytir.ops.eric_web_pdf_profile" in SCRIPT
    assert "cytir.index.seal_source_profile_compliance" in SCRIPT
    assert "cytir.index.seal_content_fidelity_audit" in SCRIPT
    assert "cytir.embed.portableize_pack" in SCRIPT
    assert "cytir.embed.adapt_pack_to_stage" in SCRIPT


# -- task 669/C: ADR-105 L1-coverage proving step --------------------------

def test_eric_chain_registers_and_verifies_l1_coverage_before_staging():
    """Real incident: eric-official-pdf-tranche-020000 refused 10,000/
    10,000 pages with UnpersistedInputs because no ERIC chain script ever
    registered its L1 at all. cytir.sources.eric.l1_seal must run, and must run
    BEFORE stage_structured_markdown_for_embedding.py (composing a pack
    whose L1 was never registered would only surface the gap hours later
    at drain time)."""
    assert "cytir.sources.eric.l1_seal" in SCRIPT
    assert "--parent-root \"${AW_ERIC_PARENT_ROOT}\"" in SCRIPT
    assert "--release-name \"${SOURCE_ID}\"" in SCRIPT
    assert SCRIPT.index("cytir.sources.eric.l1_seal") < \
        SCRIPT.index("cytir.corpus.stage_structured_markdown")
    assert SCRIPT.index("AW_ERIC_FINALIZER_OUTCOME") < \
        SCRIPT.index("cytir.sources.eric.l1_seal")


def test_eric_chain_is_cpu_only_and_does_not_fetch_or_embed():
    assert "CUDA_VISIBLE_DEVICES=" in SCRIPT
    assert "curl " not in SCRIPT
    assert "wget " not in SCRIPT
    assert "run_fp16" not in SCRIPT


# -- task 663, 2026-09-25: ADR-113's 1024w arm, gated at the registry --------

def test_eric_chain_selects_the_adr113_1024w_arm_by_default():
    assert ("chunk-policy-struct-md-1024w-o0-adjacent-sections-tail-"
           "1536-tok2560-generic-arm.v1.json") in SCRIPT
    # the OLD epoch-4 768w policy this replaces for ERIC must not still be
    # the one wired into the stage invocation (a leftover default would
    # silently un-fix this the next time someone edits the script).
    assert ("--policy \"${CHUNK_POLICY}\"") in SCRIPT
    assert ("chunk-policy-struct-md-768w-o0-adjacent-sections-tail-"
           "1152-tok2560.v1.json") not in SCRIPT


def test_eric_chain_asserts_selectable_against_the_registry_before_chunking():
    """The registry, not this script's own literal, is the source of truth:
    a real `cytir.chunk.registry.assert_selectable(family_id=...)` call
    must run, and must run BEFORE `stage_structured_markdown_for_embedding.py`
    (the actual chunk cut) -- never after, and never merely asserted in a
    comment."""
    assert "cytir.chunk import registry" in SCRIPT
    assert "assert_selectable" in SCRIPT
    assert 'family_id=sys.argv[3]' in SCRIPT or "family_id=" in SCRIPT
    assert SCRIPT.index("assert_selectable") < \
        SCRIPT.index("cytir.corpus.stage_structured_markdown")
    assert 'CHUNK_FAMILY_ID="${AW_ERIC_CHUNK_FAMILY_ID:-eric_official_pdf}"' in SCRIPT


def test_eric_chain_chunk_policy_and_family_are_overridable_env_vars():
    """Not hardcoded twice: an operator (or a future ADR successor) can
    repoint both without editing the script -- matching every other
    AW_ERIC_* knob this chain already exposes via current.env."""
    assert 'CHUNK_POLICY="${AW_ERIC_CHUNK_POLICY:-' in SCRIPT
    assert 'CHUNK_FAMILY_ID="${AW_ERIC_CHUNK_FAMILY_ID:-' in SCRIPT
