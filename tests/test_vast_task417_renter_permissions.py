import json
from pathlib import Path

from tests.helpers.repo_files import repo_file_or_skip


ROOT = Path(__file__).resolve().parents[1]


def test_vast_task417_key_has_only_rental_lifecycle_permissions():
    value = json.loads(
        repo_file_or_skip("data/vast-task417-renter-permissions.v1.json").read_text())
    assert set(value) == {"api"}
    assert set(value["api"]) == {
        "misc", "user_read", "instance_read", "instance_write"}
    assert all(scope == {} for scope in value["api"].values())
    assert "user_write" not in value["api"]
    assert "billing_write" not in value["api"]
