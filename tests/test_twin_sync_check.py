"""The daily twin/prod check: settings-driven twin, and unreachable vs read-failed told apart."""
import stat
import subprocess
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent
SCRIPT = APP / "deploy" / "scripts" / "twin-sync-check.sh"
pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="operator script is not in the public cut")

SSH_STUB = r"""#!/usr/bin/env bash
echo "$*" >> "$STUB_DIR/ssh.calls"
cmd="${@: -1}"
host=""
for a in "$@"; do case "$a" in -*|BatchMode=yes|ConnectTimeout=*) ;; *) host="$a"; break;; esac; done
case "$cmd" in *"/facet"*|*scroll*|*points*) exit 0;; esac
if [ "$host" = "twinbox" ]; then mode="$STUB_TWIN"; n="$STUB_TWIN_N"; else mode=count; n="$STUB_PROD_N"; fi
case "$mode" in
  count) echo "{\"result\":{\"points_count\":$n}}";;
  nocount) echo '{"status":{"error":"Not found"}}';;
  fail) echo "ssh: connect to host $host port 22: No route to host" >&2; exit 255;;
esac
"""
MAIL_STUB = '#!/usr/bin/env bash\necho called >> "$STUB_DIR/msmtp.calls"\ncat > "$STUB_DIR/mail.txt"\n'


def run(tmp_path, twin="count", twin_n=1000, prod_n=1000):
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in (("ssh", SSH_STUB), ("msmtp", MAIL_STUB)):
        p = stubs / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "check.log"
    env = {
        "PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path), "STUB_DIR": str(tmp_path),
        "TWIN_SYNC_LOG": str(log), "STUB_TWIN": twin, "STUB_TWIN_N": str(twin_n), "STUB_PROD_N": str(prod_n),
        "AW_TWIN_SSH_HOST": "twinbox", "AW_TWIN_URL": "http://127.0.0.1:6486", "AW_TWIN_COLLECTION": "aw_e10_twin",
    }
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "ssh.calls").exists()  # the stub, not a real ssh, answered
    mail = tmp_path / "mail.txt"
    return log.read_text(), (mail.read_text() if mail.exists() else None)


def test_equal_counts_is_silent(tmp_path):
    log, mail = run(tmp_path)
    assert "twin=1000 prod=1000 diff=0" in log and "alert='none'" in log
    assert mail is None and not (tmp_path / "msmtp.calls").exists()
    assert "aw_e10_twin" in (tmp_path / "ssh.calls").read_text()


def test_counts_apart_mails(tmp_path):
    log, mail = run(tmp_path, twin_n=1000, prod_n=1000 + 9000)
    assert "COUNT:" in log and "UNREACHABLE" not in log and "READ-FAILED" not in log
    assert mail and "COUNT:" in mail
    assert (tmp_path / "msmtp.calls").exists()


def test_twin_ssh_failure_is_unreachable(tmp_path):
    log, mail = run(tmp_path, twin="fail")
    assert "UNREACHABLE: twin did not answer" in log and "READ-FAILED" not in log
    assert "COUNT:" not in log  # no fake count drift from an empty read
    assert mail and "UNREACHABLE: twin did not answer" in mail


def test_reply_without_count_is_read_failed(tmp_path):
    log, mail = run(tmp_path, twin="nocount")
    assert "READ-FAILED: twin answered but the count could not be read" in log
    assert "UNREACHABLE" not in log
    assert mail and "READ-FAILED" in mail
