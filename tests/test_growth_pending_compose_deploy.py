"""Deployment-wrapper content checks for the growth-pending compose timer (task 664).

Split out of the package test (task 1503): they read `deploy/` unit files and name box paths, so they
stay beside the other deploy tests and are not part of the shipped package tests.
"""
from __future__ import annotations

import re as re_module
from pathlib import Path

from tests.helpers.repo_files import repo_glob_or_skip

# ---------------------------------------------------- deployment (task 664)
# The unattended systemd wiring this module exists to be run BY -- content
# checks, same discipline as the chain_eric_pdf_partition_
# admission.py, so an edit that quietly drops the flock guard or the
# supervised-restart config fails a test, not just a future incident.

# the operator tree under `deploy/` is withheld from the public cut: the tests skip there
_DEPLOY_DIR = repo_glob_or_skip("deploy/systemd/*/growth_pending_compose.sh")[0].parent


def test_growth_pending_compose_service_is_flock_guarded():
    unit = (_DEPLOY_DIR / "aw-growth-pending-compose.service").read_text()
    assert "flock -n" in unit
    assert "growth_pending_compose.sh" in unit
    assert "Type=oneshot" in unit


def test_growth_pending_compose_timer_runs_every_15_minutes():
    unit = (_DEPLOY_DIR / "aw-growth-pending-compose.timer").read_text()
    assert "OnUnitActiveSec=15min" in unit
    assert "Unit=aw-growth-pending-compose.service" in unit


def test_growth_pending_compose_wrapper_refreshes_main_and_calls_the_module():
    script = (_DEPLOY_DIR / "growth_pending_compose.sh").read_text()
    # the refresh is the one shared helper (locked fetch of origin/main, then reset or release),
    # not a private fetch/reset in the wrapper
    assert "aw-checkout.sh" in script
    assert "APP=$(aw_checkout_app)" in script
    assert "reset --hard" not in script and "git fetch" not in script
    assert "cytir.ops.growth_pending_compose" in script
    assert "--cc-root" in script and "--se-root" in script


def test_growth_pending_compose_wrapper_uses_the_project_venv_not_system_python():
    """Real incident, 2026-09-26: the first live run under bare `python3`
    failed every job with `ModuleNotFoundError: sentence_transformers` --
    that module lives only in the project venv."""
    script = (_DEPLOY_DIR / "growth_pending_compose.sh").read_text()
    assert "/.venv/bin/python" in script
    assert re_module.search(r"^python3\s", script, re_module.MULTILINE) is None


def test_growth_pending_compose_wrapper_covers_nonso_and_the_corpus_vectors_tree():
    """Task 680: the wrapper names the non-SO SE root and the corpus
    `vectors/` tree (where SO tranches 3-9 and every non-SO part keep their
    vectors); without them math parts 19-27 and SO 1-7 were never discovered
    and would be re-embedded if they were."""
    script = (_DEPLOY_DIR / "growth_pending_compose.sh").read_text()
    assert "--se-nonso-root" in script
    assert "--vectors-root" in script
    # Declared sets (task 680) and the queue-depth cap that keeps a big set
    # from outrunning the one promotion door.
    assert "--sets-file data/growth-pending-sets.v1.json" in script
    assert "--max-queue-rows" in script
    # The admission ledger records that vectors exist, not that rows serve.
    assert "--se-admission-ledger" not in script


def test_eric_chain_driver_service_restarts_always():
    unit = (_DEPLOY_DIR / "aw-eric-chain-driver.service").read_text()
    assert "Restart=always" in unit
    assert "chain_all_tranches.sh" in unit
    assert "Type=simple" in unit


