
import pytest

from cytir.index import twin_first as mod


@pytest.fixture(autouse=True)
def preservation_gate(monkeypatch):
    """Existing protocol tests exercise ordering after preservation passed."""
    monkeypatch.setattr(
        mod, "verify_publish_environment",
        lambda: {"status": "PASS", "manifest_sha256": "a" * 64,
                 "collection": "aw_test"})


class Lock:
    def __init__(self): self.events = []
    def acquire(self): self.events.append("acquire")
    def release(self): self.events.append("release")


def test_dry_run_never_acquires_lock(tmp_path):
    lock = Lock(); seen = []
    mod.twin_first(tmp_path / "ledger.jsonl", lock, preflight=lambda: "state", apply_target=lambda *_: seen.append("apply"), verify_target=lambda *_: {}, armed=False)
    assert lock.events == [] and seen == []


def test_armed_apply_fails_before_lock_when_preservation_is_missing(
        tmp_path, monkeypatch):
    def fail():
        raise mod.ContractError("missing L1")

    monkeypatch.setattr(mod, "verify_publish_environment", fail)
    lock = Lock(); seen = []
    with pytest.raises(mod.ApplyAbort, match="preservation contract did not PASS"):
        mod.twin_first(
            tmp_path / "ledger.jsonl", lock, preflight=lambda: {},
            apply_target=lambda target, _: seen.append(target),
            verify_target=lambda *_: {}, armed=True)
    assert lock.events == [] and seen == []
    assert '"status": "FAILED"' in (tmp_path / "ledger.jsonl").read_text()


def test_default_path_remains_bound_to_full_release_verifier(
        tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        mod, "verify_publish_environment",
        lambda: seen.append("full") or {"status": "PASS", "scope": "full"})
    mod.twin_first(
        tmp_path / "ledger.jsonl", Lock(), preflight=lambda: {},
        apply_target=lambda *_: None, verify_target=lambda *_: {}, armed=True)
    assert seen == ["full"]


def test_explicit_narrow_verifier_is_used_without_calling_full_verifier(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        mod, "verify_publish_environment",
        lambda: pytest.fail("full verifier must not consume a narrow intent"))
    seen = []
    mod.twin_first(
        tmp_path / "ledger.jsonl", Lock(), preflight=lambda: {},
        apply_target=lambda *_: None, verify_target=lambda *_: {}, armed=True,
        preservation_verifier=lambda: seen.append("content-pack") or {
            "status": "PASS", "scope": "incremental-content-pack-only"})
    assert seen == ["content-pack"]
    ledger = (tmp_path / "ledger.jsonl").read_text()
    assert '"scope": "incremental-content-pack-only"' in ledger


def test_injected_verifier_failure_is_still_before_lock_or_write(tmp_path):
    lock = Lock(); writes = []

    def fail():
        raise mod.ContractError("intent drift")

    with pytest.raises(mod.ApplyAbort, match="intent drift"):
        mod.twin_first(
            tmp_path / "ledger.jsonl", lock, preflight=lambda: {},
            apply_target=lambda target, _: writes.append(target),
            verify_target=lambda *_: {}, armed=True,
            preservation_verifier=fail)
    assert lock.events == []
    assert writes == []


@pytest.mark.parametrize("proof", [None, {}, {"status": "FAIL"}])
def test_injected_verifier_must_return_an_actual_pass_proof(tmp_path, proof):
    lock = Lock(); writes = []
    with pytest.raises(mod.ApplyAbort, match="non-PASS proof"):
        mod.twin_first(
            tmp_path / "ledger.jsonl", lock, preflight=lambda: {},
            apply_target=lambda target, _: writes.append(target),
            verify_target=lambda *_: {}, armed=True,
            preservation_verifier=lambda: proof)
    assert lock.events == []
    assert writes == []


def test_twin_is_verified_before_prod_and_failure_releases_lock(tmp_path):
    lock = Lock(); seen = []
    def apply(target, _): seen.append("apply-" + target)
    def verify(target, _):
        seen.append("verify-" + target)
        if target == "prod": raise RuntimeError("bad readback")
        return {"target": target}
    with pytest.raises(RuntimeError):
        mod.twin_first(tmp_path / "ledger.jsonl", lock, preflight=lambda: {}, apply_target=apply, verify_target=verify, armed=True)
    assert seen == ["apply-twin", "verify-twin", "apply-prod", "verify-prod"]
    assert lock.events == ["acquire", "release"]


def test_remote_lock_release_matches_only_exact_first_token(monkeypatch):
    commands = []
    class Result:
        returncode = 0; stdout = ""; stderr = ""
    lock = mod.ProductionWriterLock("prod", "/tmp/lock", "wave")
    monkeypatch.setattr(lock, "_run", lambda command: commands.append(command) or Result())
    lock.held = True
    lock.release()
    assert 'set -- $(head -n1' in commands[-1]
    assert '"${1-}"' in commands[-1]
    assert 'token=' in commands[-1]


def run_locally(lock, *, tmp_path):
    """Execute the lock's remote program with a real shell, in tmp_path."""
    import subprocess

    def run(program):
        return subprocess.run(["sh", "-s"], input=program, text=True,
                              capture_output=True, cwd=tmp_path)
    lock._run = run
    return lock


def test_the_lock_carries_its_token_from_the_instant_it_exists(tmp_path):
    """A token-less lock can never be released; it must never be creatable."""
    path = tmp_path / "prod.lock"
    lock = run_locally(mod.ProductionWriterLock("prod", str(path), "wave"),
                       tmp_path=tmp_path)
    lock.acquire()
    assert lock.held
    body = path.read_text()
    assert body.startswith(f"token={lock.token} ") and "purpose=wave" in body
    # No staging file is left behind.
    assert [p.name for p in tmp_path.iterdir()] == ["prod.lock"]
    lock.release()
    assert not path.exists()


def test_a_second_writer_is_refused_while_the_lock_is_held(tmp_path):
    path = tmp_path / "prod.lock"
    first = run_locally(mod.ProductionWriterLock("prod", str(path), "wave"),
                        tmp_path=tmp_path)
    first.acquire()
    second = run_locally(mod.ProductionWriterLock("prod", str(path), "other"),
                         tmp_path=tmp_path)
    with pytest.raises(mod.ApplyAbort, match="production writer lock unavailable"):
        second.acquire()
    assert not second.held
    # The refused attempt leaves the holder's lock and its token untouched.
    assert path.read_text().startswith(f"token={first.token} ")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["prod.lock"]
    first.release()


def test_remote_program_is_streamed_to_sh_stdin_without_ssh_argv_reparse(monkeypatch):
    captured = {}

    class Result:
        returncode = 0; stdout = ""; stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    program = "(set -C; : > /tmp/lock) 2>/dev/null && printf %s token > /tmp/lock"
    mod.ProductionWriterLock("prod", "/tmp/lock", "wave")._run(program)

    assert captured["argv"][-2:] == ["sh", "-s"]
    assert captured["kwargs"]["input"] == program
    assert captured["kwargs"]["text"] is True
    assert captured["kwargs"]["capture_output"] is True


def test_remote_lock_changed_owner_refuses_release(monkeypatch):
    class Result:
        returncode = 1; stdout = ""; stderr = "changed"
    lock = mod.ProductionWriterLock("prod", "/tmp/lock", "wave")
    monkeypatch.setattr(lock, "_run", lambda _command: Result())
    lock.held = True
    with pytest.raises(mod.ApplyAbort, match="ownership changed"):
        lock.release()


def test_post_twin_gate_blocks_prod(tmp_path):
    lock = Lock(); seen = []
    with pytest.raises(mod.ApplyAbort, match="did not PASS"):
        mod.twin_first(tmp_path / "ledger.jsonl", lock, preflight=lambda: {},
                       apply_target=lambda target, _: seen.append(target),
                       verify_target=lambda *_: {"ok": True}, armed=True,
                       post_twin_gate=lambda *_: {"status": "FAIL"})
    assert seen == ["twin"]
    assert '"status": "FAILED"' in (tmp_path / "ledger.jsonl").read_text()


def test_post_twin_gate_runs_before_prod(tmp_path):
    lock = Lock(); seen = []
    mod.twin_first(tmp_path / "ledger.jsonl", lock, preflight=lambda: {},
                   apply_target=lambda target, _: seen.append("apply-" + target),
                   verify_target=lambda target, _: seen.append("verify-" + target) or {"target": target},
                   armed=True, post_twin_gate=lambda *_: seen.append("gate") or {"status": "PASS"})
    assert seen == ["apply-twin", "verify-twin", "gate", "apply-prod", "verify-prod"]


def test_stop_after_twin_gate_never_touches_prod_on_a_pass(tmp_path):
    """task 626: a PASS with stop_after_twin_gate=True stops -- same as a
    REFUSE already does, but deliberately, not because the gate failed."""
    lock = Lock(); seen = []
    result = mod.twin_first(
        tmp_path / "ledger.jsonl", lock, preflight=lambda: "state",
        apply_target=lambda target, _: seen.append("apply-" + target),
        verify_target=lambda target, _: seen.append("verify-" + target) or {"target": target},
        armed=True, post_twin_gate=lambda *_: seen.append("gate") or {"status": "PASS"},
        stop_after_twin_gate=True)
    assert seen == ["apply-twin", "verify-twin", "gate"]
    assert result == "state"
    ledger_text = (tmp_path / "ledger.jsonl").read_text()
    assert '"status": "STOPPED-AFTER-TWIN-GATE"' in ledger_text
    assert '"status": "PROD-VERIFIED"' not in ledger_text
    assert '"status": "TWIN-PROD-VERIFIED-OK"' not in ledger_text


def test_stop_after_twin_gate_still_acquires_and_releases_the_lock(tmp_path):
    """The production writer lock still guards the twin write itself (task
    384's cross-transaction guarantee), even though prod is never reached."""
    lock = Lock()
    mod.twin_first(
        tmp_path / "ledger.jsonl", lock, preflight=lambda: {},
        apply_target=lambda *_: None, verify_target=lambda *_: {},
        armed=True, post_twin_gate=lambda *_: {"status": "PASS"},
        stop_after_twin_gate=True)
    assert lock.events == ["acquire", "release"]


def test_stop_after_twin_gate_does_not_soften_a_refuse(tmp_path):
    """A REFUSE still raises and still never applies prod -- the flag adds a
    new stopping point on PASS, it does not change the FAIL path at all."""
    lock = Lock(); seen = []
    with pytest.raises(mod.ApplyAbort, match="did not PASS"):
        mod.twin_first(
            tmp_path / "ledger.jsonl", lock, preflight=lambda: {},
            apply_target=lambda target, _: seen.append(target),
            verify_target=lambda *_: {"ok": True}, armed=True,
            post_twin_gate=lambda *_: {"status": "FAIL"},
            stop_after_twin_gate=True)
    assert seen == ["twin"]
    assert '"status": "FAILED"' in (tmp_path / "ledger.jsonl").read_text()


def test_locked_preflight_runs_under_lock_before_twin(tmp_path):
    lock = Lock(); seen = []

    def locked(state):
        assert lock.events == ["acquire"]
        seen.append("locked-preflight")
        return {**state, "locked": True}

    mod.twin_first(
        tmp_path / "ledger.jsonl", lock, preflight=lambda: {"initial": True},
        locked_preflight=locked,
        apply_target=lambda target, state: seen.append(
            f"apply-{target}-{state.get('locked', False)}"),
        verify_target=lambda *_: {}, armed=True)
    assert seen == ["locked-preflight", "apply-twin-True", "apply-prod-True"]
    assert lock.events == ["acquire", "release"]
    assert '"status": "LOCKED-PREFLIGHT-OK"' in (
        tmp_path / "ledger.jsonl").read_text()


def test_locked_preflight_failure_never_calls_twin_or_prod(tmp_path):
    lock = Lock(); seen = []

    def fail(_state):
        raise mod.ApplyAbort("live premise changed")

    with pytest.raises(mod.ApplyAbort, match="premise changed"):
        mod.twin_first(
            tmp_path / "ledger.jsonl", lock, preflight=lambda: {},
            locked_preflight=fail,
            apply_target=lambda target, _: seen.append(target),
            verify_target=lambda *_: {}, armed=True)
    assert seen == []
    assert lock.events == ["acquire", "release"]
    assert '"status": "FAILED"' in (tmp_path / "ledger.jsonl").read_text()


def test_resume_rechecks_bound_state_then_gates_before_prod(tmp_path):
    lock = Lock(); seen = []
    state = {"failed_ledger_sha256": "a" * 64, "twin_count": 12}

    def preflight():
        seen.append("preflight")
        return state

    mod.resume_after_verified_twin_gate(
        tmp_path / "resume.jsonl", lock, preflight=preflight,
        post_twin_gate=lambda _: seen.append("gate") or {"status": "PASS"},
        apply_prod=lambda _: seen.append("apply-prod"),
        verify_mirrors=lambda _: seen.append("verify-mirrors") or {"status": "PASS"},
        armed=True)
    assert seen == ["preflight", "preflight", "gate", "apply-prod", "verify-mirrors"]
    assert lock.events == ["acquire", "release"]


def test_resume_explicitly_uses_injected_preservation_verifier(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        mod, "verify_publish_environment",
        lambda: pytest.fail("resume must use its explicit narrow verifier"))
    seen = []
    mod.resume_after_verified_twin_gate(
        tmp_path / "resume.jsonl", Lock(), preflight=lambda: {"bound": True},
        post_twin_gate=lambda _: {"status": "PASS"},
        apply_prod=lambda _: None,
        verify_mirrors=lambda _: {"status": "PASS"}, armed=True,
        preservation_verifier=lambda: seen.append("content-pack") or {
            "status": "PASS"})
    assert seen == ["content-pack"]


def test_resume_rejects_premise_tamper_before_gate_or_prod(tmp_path):
    lock = Lock(); seen = []; calls = 0

    def preflight():
        nonlocal calls
        calls += 1
        return {"twin_count": 12 if calls == 1 else 13}

    with pytest.raises(mod.ApplyAbort, match="premise changed"):
        mod.resume_after_verified_twin_gate(
            tmp_path / "resume.jsonl", lock, preflight=preflight,
            post_twin_gate=lambda _: seen.append("gate") or {"status": "PASS"},
            apply_prod=lambda _: seen.append("apply-prod"),
            verify_mirrors=lambda _: {"status": "PASS"}, armed=True)
    assert seen == []
    assert lock.events == ["acquire", "release"]


def test_resume_gate_failure_never_applies_prod(tmp_path):
    lock = Lock(); seen = []
    with pytest.raises(mod.ApplyAbort, match="did not PASS"):
        mod.resume_after_verified_twin_gate(
            tmp_path / "resume.jsonl", lock, preflight=lambda: {"bound": True},
            post_twin_gate=lambda _: {"status": "FAIL"},
            apply_prod=lambda _: seen.append("apply-prod"),
            verify_mirrors=lambda _: {"status": "PASS"}, armed=True)
    assert seen == []
    assert '"status": "FAILED"' in (tmp_path / "resume.jsonl").read_text()
