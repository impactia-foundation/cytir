import json
from pathlib import Path

from cytir.ops.public_cut import scan_text


REPO = Path(__file__).resolve().parents[1]
GLOBAL_ROOT = Path("/aw-corpus")
FORBIDDEN_FIRST_COMPONENTS = {
    "aw-corpus",
    "media",
    "mnt",
}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def test_tracked_machine_ledgers_use_global_root_relative_artifact_uris() -> None:
    failures = []
    for path in sorted((REPO / "data").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        for text in _strings(value):
            if not text.startswith("artifact://"):
                continue
            relative_text = text.removeprefix("artifact://")
            relative = Path(relative_text)
            if (
                not relative_text
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.parts[0] in FORBIDDEN_FIRST_COMPONENTS
                # a box name as the first component: the cut's own host-literal rule is the list
                or scan_text(relative.parts[0]).get("host_literal", 0)
                or (GLOBAL_ROOT / relative).is_relative_to(GLOBAL_ROOT) is False
            ):
                failures.append(f"{path.relative_to(REPO)}: {text}")
    assert not failures, "non-portable tracked artifact URIs:\n" + "\n".join(failures)
