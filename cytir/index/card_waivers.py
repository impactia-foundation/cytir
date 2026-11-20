"""Sealed per-id waivers: a SITE row may replace a WRONG entity card (ruling 26 item 19b).

The drain's card guard (`promote_queue.check_card_overwrites`) refuses a pack whose upsert would
overwrite a stored entity card, by id, with a row that is not a card of an overlapping identity
(`supersede.card_replaces`). That is right for every card but one class: a card that is WRONG
(task 621: a register card resolved to the wrong company's site). Replacing that card with the
real site row is a human decision, one id at a time. This file records the decision.

THE FILE `data/card-waivers.v1.jsonl` is the sealed SOURCE of waiver rows (changed by PR only: a
waiver is legislative-adjacent). A pack does not read the repo file. ADR-114 am.1 item 3: the pack
carries the sealed file that names the id. The pack MANIFEST names the copy it relies on:
`"card_waivers": {"path": "<absolute path>", "sha256": "<sha256 of the file bytes>"}`. The drain reads
only that file, for that pack, and refuses on a sha mismatch. A pack without the field has no
waivers. A change of the file changes the manifest, so a cached refusal is re-judged on the reseal.

File format:

    line 1      {"schema": "aw.card-waivers.v1"}
    lines 2..n  one waiver row each (fields in `ROW_FIELDS`)
    last line   {"seal": {"rows": <n>, "sha256": "<hex>"}}

`seal.sha256` is over the exact bytes of every line before the seal line, newline included. An edit
that does not re-seal, a missing seal, a row with a missing or ill-typed field, a token that is not
in `supersede.card_identity`'s canonical form, a duplicate id, or a row after the seal:
`CardWaiverError`. The drain never repairs the file and never guesses.

A row stays in force while it stands in the file. It admits an overwrite of the stored card at point
`id`, at canonical `url`, whose identity tokens are EXACTLY `wrong_card_identity`, by a non-card
row at the same canonical url. It matches again each time that same wrong card reappears at that
id. Retire a row by a PR that removes it. A later card at the id with another identity is not
covered. The receipt lists each waived id with the row's `pr` and `evidence_sha256`.
Reseal with `python3 -m cytir.index.card_waivers seal`.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import uuid

from cytir.core import common as AC
from cytir.index import dry_run as DRY
ROOT = pathlib.Path(__file__).resolve().parents[2]

WAIVER_SCHEMA = "aw.card-waivers.v1"
WAIVER_PATH = ROOT / "data" / "card-waivers.v1.jsonl"

#: field -> type. `wrong_card_identity` is a non-empty list of `supersede.card_identity` tokens.
ROW_FIELDS = {
    "id": str,                    # point id of the stored (wrong) card
    "url": str,                   # its canonical URL
    "wrong_card_identity": list,  # e.g. ["reg:CHE:CHE105974401"]
    "reason": str,
    "evidence_sha256": str,       # sha of the evidence file in the task doc
    "approved_by": str,
    "approved_at": str,           # ISO date
    "pr": str,                    # the PR that appended the row
}


_TOKEN = re.compile(r"^(reg:[A-Z0-9]+:[A-Z0-9]+|lei:[A-Z0-9]+|qid:Q[0-9]+)$")


class CardWaiverError(RuntimeError):
    """The waiver file is unsealed or malformed. The drain refuses the pack."""


def _canon_id(value) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        return str(value)


def _canon_url(value) -> str:
    return AC.canonical_url(value) if isinstance(value, str) and value.strip() else ""


def _check_row(row, n: int) -> None:
    if not isinstance(row, dict):
        raise CardWaiverError(f"waiver line {n}: not a JSON object")
    missing = [k for k in ROW_FIELDS if k not in row]
    extra = [k for k in row if k not in ROW_FIELDS]
    if missing or extra:
        raise CardWaiverError(f"waiver line {n}: missing {missing} / unknown {extra}")
    for k, typ in ROW_FIELDS.items():
        v = row[k]
        if not isinstance(v, typ) or (typ is str and not v.strip()) or (typ is list and not v):
            raise CardWaiverError(f"waiver line {n}: field {k!r} must be a non-empty {typ.__name__}")
    bad = [t for t in row["wrong_card_identity"] if not (isinstance(t, str) and _TOKEN.match(t))]
    if bad:
        raise CardWaiverError(
            f"waiver line {n}: wrong_card_identity token(s) {bad} are not in the canonical form of "
            "supersede.card_identity (reg:<SCHEME>:<ID>, lei:<LEI>, qid:Q<n>; upper case, no punctuation)")
    if len(row["evidence_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["evidence_sha256"]):
        raise CardWaiverError(f"waiver line {n}: evidence_sha256 is not a lower-case sha256 hex")
    try:
        uuid.UUID(row["id"])
    except ValueError as exc:
        raise CardWaiverError(f"waiver line {n}: id {row['id']!r} is not a UUID") from exc


def seal_line(body: bytes, rows: int) -> str:
    return json.dumps({"seal": {"rows": rows, "sha256": hashlib.sha256(body).hexdigest()}},
                      sort_keys=True) + "\n"


def load_waivers(path: pathlib.Path | None = None, *, expected_sha256: str | None = None
                 ) -> dict[str, dict]:
    """The sealed waivers, keyed by canonical point id. With `path` None the repo source file;
    absent source file = {}. With `expected_sha256` (a manifest's pin) the file must exist and its
    bytes must hash to it. Raises `CardWaiverError` on anything unsealed or malformed."""
    path = pathlib.Path(path) if path is not None else WAIVER_PATH
    if not path.exists():
        if expected_sha256 is not None:
            raise CardWaiverError(f"{path}: the manifest names this waiver file and it does not exist")
        return {}
    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise CardWaiverError(f"{path}: sha256 {hashlib.sha256(raw).hexdigest()} is not the manifest's "
                              f"{expected_sha256}")
    lines = raw.splitlines(keepends=True)
    if len(lines) < 2 or not raw.endswith(b"\n"):
        raise CardWaiverError(f"{path}: needs a header line and a seal line, each newline-ended")
    try:
        head = json.loads(lines[0])
        seal = json.loads(lines[-1]).get("seal")
    except (ValueError, AttributeError) as exc:
        raise CardWaiverError(f"{path}: header or seal line is not JSON") from exc
    if head != {"schema": WAIVER_SCHEMA}:
        raise CardWaiverError(f"{path}: header is {head!r}, want schema {WAIVER_SCHEMA!r}")
    if not isinstance(seal, dict):
        raise CardWaiverError(f"{path}: the last line is not a seal line")
    body = b"".join(lines[:-1])
    if seal.get("sha256") != hashlib.sha256(body).hexdigest():
        raise CardWaiverError(f"{path}: seal sha256 does not match the file body (edited without re-seal)")
    rows = lines[1:-1]
    if seal.get("rows") != len(rows):
        raise CardWaiverError(f"{path}: seal says {seal.get('rows')} rows, file has {len(rows)}")
    out: dict[str, dict] = {}
    for n, line in enumerate(rows, start=2):
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise CardWaiverError(f"waiver line {n}: not JSON") from exc
        _check_row(row, n)
        key = _canon_id(row["id"])
        if key in out:
            raise CardWaiverError(f"waiver line {n}: duplicate id {key}")
        out[key] = row
    return out


def admits(waivers: dict[str, dict], point_id, stored_payload: dict, stored_identity,
           new_payload: dict | None = None) -> dict | None:
    """The waiver row that admits overwriting the stored card at `point_id`, else None. Exact
    match on id, canonical URL of the stored card AND of the new row, and the whole identity-token
    set."""
    row = waivers.get(_canon_id(point_id))
    if row is None:
        return None
    if _canon_url(stored_payload.get("url")) != _canon_url(row["url"]):
        return None
    if _canon_url((new_payload or {}).get("url")) != _canon_url(row["url"]):
        return None
    if set(stored_identity) != set(row["wrong_card_identity"]):
        return None
    return row


@DRY.refuses_when_dry("re-seal the card waiver file")
def seal_file(path: pathlib.Path | None = None) -> None:
    """Rewrite `path` with a fresh header-body-seal. For the PR author; the drain never calls it."""
    path = pathlib.Path(path) if path is not None else WAIVER_PATH
    rows = []
    if path.exists():
        for line in path.read_text().splitlines()[1:]:
            if line.strip() and not line.startswith('{"seal"'):
                rows.append(line)
    for n, line in enumerate(rows, start=2):
        _check_row(json.loads(line), n)
    body = (json.dumps({"schema": WAIVER_SCHEMA}) + "\n" + "".join(r + "\n" for r in rows)).encode()
    path.write_bytes(body + seal_line(body, len(rows)).encode())


if __name__ == "__main__":
    if sys.argv[1:] == ["seal"]:
        seal_file()
    else:
        print("usage: python -m cytir.index.card_waivers seal", file=sys.stderr)
        sys.exit(2)
